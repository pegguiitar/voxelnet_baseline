"""profile_pipeline.py - breaks one training step down into its sub-components
(VFE / 3D backbone+BEV-collapse / BEV adapter conv / 2D RPNBackbone+head / target
building / loss / backward / optimizer step / data loading) for any experiment in
this folder, using forward hooks on model.vfe/model.backbone/model.head (and the
bev_project/bn/relu adapter, when present) -- this works regardless of how each
experiment's own forward() calls those submodules internally (their signatures
differ: sparse backbones take coords+index_grid+grid_size, the dense one doesn't),
since a hook only cares about entry/exit of the submodule call itself.

Answers the "why doesn't cutting 3D compute (SlotFormer/backbone stages) buy much
speedup, but replacing dense ConvMiddleLayers with sparse ConvMiddleLayers did?"
question empirically: run this on exp3_conv_middle_bev and dense_baseline_bev and
compare what fraction of total step time each bucket -- especially "backbone" (the
part that differs) vs "head" (the shared, unchanged 2D RPNBackbone+head) -- takes.

Usage:
    python profile_pipeline.py --exp exp3_conv_middle_bev
    python profile_pipeline.py --exp dense_baseline_bev
    python profile_pipeline.py --exp exp1_single_stage_bev --steps 30 --warmup 5
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent  # -> voxelnet_baseline/model
sys.path.insert(0, str(MODEL_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from sonar_diver_dataset import SonarDiverDataset, collate_fn
from sparse_bev_head import build_bev_targets
from center_loss import center_voxelnet_loss


def load_voxelnet_class(exp_dir_name: str):
    exp_dir = Path(__file__).resolve().parent / exp_dir_name
    sys.path.insert(0, str(exp_dir))
    spec = importlib.util.spec_from_file_location(f"_profile_{exp_dir_name}_voxelnet", exp_dir / "voxelnet.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and name.endswith("VoxelNet"):
            return obj
    raise RuntimeError(f"no *VoxelNet class found in {exp_dir}/voxelnet.py")


def attach_timers(model, device):
    """Forward pre/post hooks on model.vfe/model.backbone/model.head (+ the BEV
    adapter conv, if present) that accumulate wall-clock time into `timings`,
    synchronizing CUDA at each boundary so async kernel launches don't leak time
    into the wrong bucket."""
    timings = {}

    def make_hooks(name):
        def pre(module, inp):
            if device.type == "cuda":
                torch.cuda.synchronize()
            module.__t0 = time.perf_counter()

        def post(module, inp, out):
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings[name] = timings.get(name, 0.0) + (time.perf_counter() - module.__t0)

        return pre, post

    def make_bwd_hooks(name):
        def pre(module, grad_output):
            if device.type == "cuda":
                torch.cuda.synchronize()
            module.__tb0 = time.perf_counter()

        def post(module, grad_input, grad_output):
            if device.type == "cuda":
                torch.cuda.synchronize()
            key = name + "_bwd"
            timings[key] = timings.get(key, 0.0) + (time.perf_counter() - module.__tb0)

        return pre, post

    handles = []
    for name in ("vfe", "backbone", "head"):
        sub = getattr(model, name, None)
        if sub is not None:
            pre, post = make_hooks(name)
            handles.append(sub.register_forward_pre_hook(pre))
            handles.append(sub.register_forward_hook(post))
            bpre, bpost = make_bwd_hooks(name)
            handles.append(sub.register_full_backward_pre_hook(bpre))
            handles.append(sub.register_full_backward_hook(bpost))
    if getattr(model, "_project", False):
        # forward-only here (no backward hooks): bev_bn's output feeds directly into
        # bev_relu(inplace=True), and a full-backward-hook's output-wrapping trick is
        # incompatible with the next op modifying that exact tensor in place -- this
        # tiny adapter (<1% of total in practice) isn't worth working around for.
        pre, post = make_hooks("bev_adapter")
        for attr in ("bev_project", "bev_bn", "bev_relu"):
            sub = getattr(model, attr, None)
            if sub is not None:
                handles.append(sub.register_forward_pre_hook(pre))
                handles.append(sub.register_forward_hook(post))
    return timings, handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True,
                     help="experiment folder name, e.g. exp3_conv_middle_bev / dense_baseline_bev")
    ap.add_argument("--steps", type=int, default=30, help="measured steps (after warmup)")
    ap.add_argument("--warmup", type=int, default=5, help="unmeasured warmup steps (CUDA kernel caching etc.)")
    ap.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  exp: {args.exp}")

    VoxelNetCls = load_voxelnet_class(args.exp)
    model = VoxelNetCls().to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    ds = SonarDiverDataset(
        "train", point_cloud_range=config.SPARSE_BEV_POINT_CLOUD_RANGE, voxel_size=config.SPARSE_BEV_VOXEL_SIZE,
        max_points_per_voxel=config.SPARSE_BEV_MAX_POINTS_PER_VOXEL, max_voxels=config.SPARSE_BEV_MAX_VOXELS,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                         collate_fn=collate_fn, drop_last=True)
    it = iter(loader)

    timings, handles = attach_timers(model, device)
    other = {"data_load": 0.0, "target_build": 0.0, "loss": 0.0, "backward_total": 0.0, "opt_step": 0.0}
    fwd_total = 0.0
    n = 0

    for step in range(args.warmup + args.steps):
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        data_t = time.perf_counter() - t0

        voxel_features = batch["voxel_features"].to(device, non_blocking=True)
        num_points = batch["num_points"].to(device, non_blocking=True)
        coords = batch["coords"].to(device, non_blocking=True)
        gt_boxes_list = batch["gt_boxes"]

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        heatmap, offset, z, dim, rot, density = model(voxel_features, num_points, coords)
        if device.type == "cuda":
            torch.cuda.synchronize()
        fwd_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        targets = [build_bev_targets(gb.numpy(), None, model.head_grid_size, model.head_stride, model.pc_range)
                   for gb in gt_boxes_list]
        heatmap_t = torch.from_numpy(np.stack([t["heatmap"][0] for t in targets])).unsqueeze(1).to(device)
        reg_mask_t = torch.from_numpy(np.stack([t["reg_mask"] for t in targets])).to(device)
        offset_t = torch.from_numpy(np.stack([t["offset"] for t in targets])).to(device)
        z_t = torch.from_numpy(np.stack([t["z"] for t in targets])).to(device)
        dim_t = torch.from_numpy(np.stack([t["dim"] for t in targets])).to(device)
        rot_t = torch.from_numpy(np.stack([t["rot"] for t in targets])).to(device)
        target_t = time.perf_counter() - t0

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss, stats = center_voxelnet_loss(
            heatmap, offset, z, dim, rot, heatmap_t, reg_mask_t, offset_t, z_t, dim_t, rot_t,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        loss_t = time.perf_counter() - t0

        optimizer.zero_grad()
        t0 = time.perf_counter()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        backward_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        opt_t = time.perf_counter() - t0

        if step >= args.warmup:
            other["data_load"] += data_t
            other["target_build"] += target_t
            other["loss"] += loss_t
            other["backward_total"] += backward_t
            other["opt_step"] += opt_t
            fwd_total += fwd_t
            n += 1

    for h in handles:
        h.remove()

    fwd_hooked = sum(v for k, v in timings.items() if not k.endswith("_bwd"))
    bwd_hooked = sum(v for k, v in timings.items() if k.endswith("_bwd"))
    fwd_residual = fwd_total - fwd_hooked          # forward time NOT inside vfe/backbone/head
                                                    # (build_index_grid, batch_size lookup, etc.)
    bwd_residual = other["backward_total"] - bwd_hooked  # backward time NOT inside those modules
                                                           # (loss's own backward, scatter_to_bev's
                                                           # backward, autograd graph glue)

    all_buckets = dict(timings)  # per-module fwd (name) + bwd (name_bwd) buckets
    all_buckets["forward_other(index_grid,etc.)"] = fwd_residual
    all_buckets["backward_other(loss/scatter/glue)"] = bwd_residual
    all_buckets.update({k: v for k, v in other.items() if k != "backward_total"})

    total = sum(all_buckets.values())  # additive: does NOT include backward_total itself
                                        # (that's the sum of *_bwd + backward_other, kept out
                                        # to avoid double-counting the same wall-clock time twice)
    print(f"\n=== profile: {args.exp}  ({n} steps measured, {args.warmup} warmup, batch_size={args.batch_size}) ===")
    for name, t in sorted(all_buckets.items(), key=lambda kv: -kv[1]):
        avg_ms = (t / n) * 1000
        pct = 100 * t / total if total > 0 else 0.0
        print(f"  {name:28s} {avg_ms:8.2f} ms/step  ({pct:5.1f}%)")
    print(f"  {'TOTAL':28s} {(total / n) * 1000:8.2f} ms/step  ({1.0 / (total / n):.2f} it/s)")
    print(f"  (sanity check: backward_total measured as one block = {(other['backward_total'] / n) * 1000:.2f} ms/step,"
          f" vs sum of backward buckets above = {((bwd_hooked + bwd_residual) / n) * 1000:.2f} ms/step)")


if __name__ == "__main__":
    main()
