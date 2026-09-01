"""analyze_fg_gate_by_range.py - "fg_gate 신뢰도가 거리에 따라 떨어진다"는 가설을
학습 없이 확인. joint(또는 joint_ema) 체크포인트만 가능 - fg_head가 model.state_dict()
안에 있어야 재구성 가능(freeze_fit의 probe는 model 밖에 별도로 있어서 체크포인트에
안 남음).

측정: 각 BEV 셀을 GT-positive/negative로 나누고(foreground_gate.fg_target_bev),
range bucket별로 fg_gate 값의 분포를 본다. 가설이 맞다면: negative 셀의 fg_gate가
원거리에서 더 높게(오분류 방향으로) 나와야 한다 - 즉 "배경인데 foreground로
착각"하는 비율이 거리에 따라 늘어난다."""

import sys
from pathlib import Path

import numpy as np
import torch

import config
import foreground_gate as fg
from eval_voxelnet_by_range import BUCKETS
from model import VoxelNet
from analyze_recall_collapse import load_compat_ignore_shape_mismatch


def load_joint_model(ckpt_path: Path) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = VoxelNet(head=ckpt.get("head", "center"), polar=ckpt.get("polar", False),
                      use_fg_head=True)
    dropped = load_compat_ignore_shape_mismatch(model, ckpt["model"])
    if dropped:
        print(f"  [load] 누락/shape mismatch: {dropped}")
    model.eval()
    return model


def bucket_of(r: float) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= r < hi:
            return name
    return BUCKETS[-1][0]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cache-root", default="../cache/voxel")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-frames", type=int, default=300)
    args = parser.parse_args()

    model = load_joint_model(Path(args.ckpt))

    from dataset import CachedVoxelNetDataset
    ds = CachedVoxelNetDataset(args.cache_root, args.split, head="center", load_gt_boxes=True)
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(ds), size=min(args.n_frames, len(ds)), replace=False)

    # bucket -> {"pos": [gate values...], "neg": [gate values...]}
    stats = {b[0]: {"pos": [], "neg": []} for b in BUCKETS}

    with torch.no_grad():
        for i in idxs:
            sample = ds[i]
            gt_boxes = sample["gt_boxes"]
            if len(gt_boxes) == 0:
                continue
            npt = sample["num_points"]
            crd = sample["coords"]
            batch_col = torch.zeros((len(crd), 1), dtype=torch.int64)
            crd_b = torch.cat([batch_col, crd], dim=1)
            hm, off, z, dim, rot, density, fg_logit = model(
                sample["voxel_features"], npt, crd_b, return_foreground=True)
            fg_gate = torch.sigmoid(fg_logit)[0, 0]  # (H,W)
            h, w = fg_gate.shape
            target = fg.fg_target_bev([gt_boxes], h, w)[0, 0]  # (H,W) {0,1}
            xx, yy = fg.cell_centers_xy(h, w)
            rr = torch.sqrt(xx ** 2 + yy ** 2).numpy()

            gate_np = fg_gate.numpy()
            target_np = target.numpy()
            for r_row, g_row, t_row in zip(rr, gate_np, target_np):
                for r, g, t in zip(r_row, g_row, t_row):
                    b = bucket_of(float(r))
                    stats[b]["pos" if t > 0.5 else "neg"].append(float(g))

    print(f"ckpt={args.ckpt}  n_frames_used={len(idxs)}")
    print(f"{'bucket':10s} {'n_pos':>8s} {'pos_gate_mean':>14s} {'n_neg':>10s} {'neg_gate_mean':>14s} {'neg_gate>0.5_frac':>18s}")
    for name, _, _ in BUCKETS:
        pos = stats[name]["pos"]
        neg = stats[name]["neg"]
        pos_mean = np.mean(pos) if pos else float("nan")
        neg_mean = np.mean(neg) if neg else float("nan")
        neg_frac_high = np.mean([v > 0.5 for v in neg]) if neg else float("nan")
        print(f"{name:10s} {len(pos):8d} {pos_mean:14.4f} {len(neg):10d} {neg_mean:14.4f} {neg_frac_high:18.4f}")


if __name__ == "__main__":
    main()
