"""smoke_test.py (exp2_down_slot_up_bev) - synthetic-data structural test for
SparseBEVDownSlotUpVoxelNet (model construction -> forward -> sparse BEV target
building -> sparse center loss -> backward -> decode).

Usage: python smoke_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # -> voxelnet_baseline/model

import numpy as np
import torch

import config
import rotation3d
from sparse_head_bev import build_sparse_bev_targets, sparse_bev_center_loss, decode_sparse_bev_boxes

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voxelnet import SparseBEVDownSlotUpVoxelNet  # noqa: E402


def make_synthetic_batch(batch_size=2, n_voxels_per_sample=400, n_gt_per_sample=2, device="cpu"):
    Wp, Hp, Dp = config.SPARSE_BEV_GRID_SIZE
    voxel_features, coords = [], []
    for b in range(batch_size):
        zyx = torch.stack([
            torch.randint(0, Dp, (n_voxels_per_sample * 2,)),
            torch.randint(0, Hp, (n_voxels_per_sample * 2,)),
            torch.randint(0, Wp, (n_voxels_per_sample * 2,)),
        ], dim=1)
        zyx = torch.unique(zyx, dim=0)[:n_voxels_per_sample]
        k = zyx.shape[0]
        batch_col = torch.full((k, 1), b, dtype=torch.long)
        coords.append(torch.cat([batch_col, zyx], dim=1))
        voxel_features.append(torch.randn(k, config.SPARSE_BEV_MAX_POINTS_PER_VOXEL, 7))

    coords = torch.cat(coords, dim=0).to(device)
    voxel_features = torch.cat(voxel_features, dim=0).to(device)
    num_points = torch.randint(1, config.SPARSE_BEV_MAX_POINTS_PER_VOXEL, (voxel_features.shape[0],)).to(device)

    pc_range = config.SPARSE_BEV_POINT_CLOUD_RANGE
    lo = np.array(pc_range[:3])
    hi = np.array(pc_range[3:])
    gt_boxes_per_sample = []
    for _ in range(batch_size):
        rows = []
        for _ in range(n_gt_per_sample):
            center = lo + np.random.rand(3) * (hi - lo)
            dims = np.array([0.5, 1.0, 1.5]) + np.random.rand(3) * 0.3
            rx, ry, rz = (np.random.rand(3) * 360 - 180).tolist()
            R = rotation3d.euler_to_matrix(rx, ry, rz)
            six = rotation3d.matrix_to_6d(R)
            rows.append(np.concatenate([center, dims, [0.0], six]))
        gt_boxes_per_sample.append(torch.from_numpy(np.stack(rows).astype(np.float32)))

    return voxel_features, num_points, coords, gt_boxes_per_sample


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = SparseBEVDownSlotUpVoxelNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}  input_grid_size(D,H,W): {model.input_grid_size}  "
          f"head_grid_size(W'',H''): {model.head_grid_size}  head_stride(sx,sy): {model.head_stride}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch_size = 2
    voxel_features, num_points, coords, gt_boxes_per_sample = make_synthetic_batch(
        batch_size=batch_size, device=device)
    print(f"input active voxels (pre-backbone): {coords.shape[0]}")

    pred, out_coords, bs = model(voxel_features, num_points, coords)
    print(f"active cells at head: {out_coords.shape[0]}  pred[heatmap]: {tuple(pred['heatmap'].shape)}")

    target = build_sparse_bev_targets(gt_boxes_per_sample, out_coords, model.head_stride, model.pc_range)
    n_pos = int(target["reg_mask"].sum().item())
    print(f"n_pos assigned: {n_pos} / {batch_size * 2} GT boxes")

    loss, stats = sparse_bev_center_loss(pred, target)
    assert torch.isfinite(loss), "loss is not finite"
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"loss: {loss.item():.4f}  stats: {stats}")

    boxes_per_sample = decode_sparse_bev_boxes(
        pred, out_coords, model.head_stride, batch_size, model.head_grid_size_hw,
        model.pc_range, score_thresh=0.0)
    for b, boxes in enumerate(boxes_per_sample):
        print(f"  sample {b}: {len(boxes)} decoded boxes (score_thresh=0.0, pre-training so meaningless numerically)")

    print("\nOK -- exp2_down_slot_up_bev forward/target/loss/backward/decode all ran without error.")


if __name__ == "__main__":
    main()
