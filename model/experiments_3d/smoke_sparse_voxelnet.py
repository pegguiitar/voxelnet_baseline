"""smoke_sparse_voxelnet.py - synthetic-data structural test for sparse_voxelnet.py
(model construction -> forward -> target building -> loss -> backward), mirroring the
3d_point_cloud sibling project's smoke_test.py. No real sonar data needed/used --
this repo doesn't ship any (see model/README.md) -- so this only proves the sparse
pipeline's shapes/wiring are correct, not that it detects anything real.

Usage: python smoke_sparse_voxelnet.py
"""
import numpy as np
import torch

import config
import rotation3d
from sparse_voxelnet import SparseVoxelNet
from sparse_center_head import build_sparse_targets, sparse_center_loss, decode_sparse_center_boxes


def make_synthetic_batch(batch_size=2, n_voxels_per_sample=400, n_gt_per_sample=2, device="cpu"):
    Wp, Hp, Dp = config.GRID_SIZE
    voxel_features, coords = [], []
    for b in range(batch_size):
        # unique (z,y,x) coords within grid bounds, matching voxelize.py's guarantee
        zyx = torch.stack([
            torch.randint(0, Dp, (n_voxels_per_sample * 2,)),
            torch.randint(0, Hp, (n_voxels_per_sample * 2,)),
            torch.randint(0, Wp, (n_voxels_per_sample * 2,)),
        ], dim=1)
        zyx = torch.unique(zyx, dim=0)[:n_voxels_per_sample]
        k = zyx.shape[0]
        batch_col = torch.full((k, 1), b, dtype=torch.long)
        coords.append(torch.cat([batch_col, zyx], dim=1))
        voxel_features.append(torch.randn(k, config.MAX_POINTS_PER_VOXEL, 7))

    coords = torch.cat(coords, dim=0).to(device)
    voxel_features = torch.cat(voxel_features, dim=0).to(device)
    num_points = torch.randint(1, config.MAX_POINTS_PER_VOXEL, (voxel_features.shape[0],)).to(device)

    pc_range = config.POINT_CLOUD_RANGE
    lo = torch.tensor(pc_range[:3])
    hi = torch.tensor(pc_range[3:])
    gt_boxes_list = []
    for b in range(batch_size):
        rows = []
        for _ in range(n_gt_per_sample):
            center = lo + torch.rand(3) * (hi - lo)
            dims = torch.tensor([0.5, 1.0, 1.5]) + torch.rand(3) * 0.3
            rx, ry, rz = (torch.rand(3) * 360 - 180).tolist()
            R = rotation3d.euler_to_matrix(rx, ry, rz)
            six = rotation3d.matrix_to_6d(R)
            rows.append(np.concatenate([center.numpy(), dims.numpy(), [0.0], six]))
        gt_boxes_list.append(torch.from_numpy(np.stack(rows).astype(np.float32)).to(device))

    return voxel_features, num_points, coords, gt_boxes_list


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = SparseVoxelNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}  stride: {model.stride}  input_grid_size(D,H,W): {model.input_grid_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    voxel_features, num_points, coords, gt_boxes_list = make_synthetic_batch(device=device)
    print(f"input active voxels (pre-backbone): {coords.shape[0]}")

    pred, out_coords, out_grid_size = model(voxel_features, num_points, coords)
    print(f"backbone output active voxels: {out_coords.shape[0]}  grid_size(D,H,W): {out_grid_size}")
    for k, v in pred.items():
        print(f"  pred[{k}]: {tuple(v.shape)}")

    target = build_sparse_targets(gt_boxes_list, out_coords, model.stride)
    print(f"n_pos assigned: {target['reg_mask'].sum().item()} / {len(gt_boxes_list) * 2} GT boxes")

    loss, stats = sparse_center_loss(pred, target)
    assert torch.isfinite(loss), "loss is not finite"
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"loss: {loss.item():.4f}  stats: {stats}")

    boxes_per_sample = decode_sparse_center_boxes(pred, out_coords, model.stride, len(gt_boxes_list), out_grid_size,
                                                   score_thresh=0.0)  # 0.0 -- random-init model, just check it runs/shapes
    for b, boxes in enumerate(boxes_per_sample):
        print(f"  sample {b}: {len(boxes)} decoded boxes (score_thresh=0.0, pre-training so meaningless numerically)")

    print("\nOK -- sparse_voxelnet forward/target/loss/backward/decode all ran without error.")


if __name__ == "__main__":
    main()
