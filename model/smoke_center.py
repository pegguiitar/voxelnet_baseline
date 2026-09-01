"""smoke_center.py - center head(RPNCenterHead) CPU 파이프라인 검증. train.py --smoke는
VoxelNetDataset(로컬 즉석계산) 기반이라 anchor head만 지원해서(코드 주석 확인) center head는
별도로 이 스크립트로 확인한다 - 3D 회전 확장(6D) 이후 shape/gradient가 끝까지 살아있는지가
목적.

Usage:
    python smoke_center.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Triband_BEV" / "baseline"))
import common  # noqa: E402
import filter_outliers  # noqa: E402

import anchors as anchors_mod
import config
import heatmap_targets as ht
from center_loss import center_voxelnet_loss
from model import VoxelNet
from voxelize import augment_with_centroid_offset, voxelize


def load_frame(scene_id: str, frame_idx: int, objects: list):
    raw = np.fromfile(common.sonar_bin_path(scene_id, frame_idx), dtype=np.float32)
    points = raw.reshape(-1, 4 if raw.size % 4 == 0 else 3)
    if points.shape[1] == 3:
        points = np.concatenate([points, np.zeros((len(points), 1), dtype=np.float32)], axis=1)
    points = points[~np.isnan(points).any(axis=1)]
    voxel_xyzr, coords, num_points = voxelize(
        points, config.POINT_CLOUD_RANGE, config.VOXEL_SIZE,
        config.MAX_POINTS_PER_VOXEL, config.MAX_VOXELS)
    voxel_features = augment_with_centroid_offset(voxel_xyzr, num_points)
    return voxel_features, coords, num_points, points


def main():
    flagged_keys = filter_outliers.load_flagged_keys()
    frames = list(filter_outliers.iter_filtered_frames("scene_0044", flagged_keys))[:4]
    print(f"{len(frames)}개 프레임으로 스모크 테스트")

    vf_list, coords_list, npts_list = [], [], []
    heatmap_list, mask_list, offset_list, z_list, dim_list, rot_list, density_list = [], [], [], [], [], [], []
    for bi, (frame_idx, objects) in enumerate(frames):
        vf, coords, npts, points = load_frame("scene_0044", frame_idx, objects)
        batch_col = np.full((coords.shape[0], 1), bi, dtype=coords.dtype)
        coords_list.append(np.concatenate([batch_col, coords], axis=1))
        vf_list.append(vf)
        npts_list.append(npts)

        targets = ht.build_heatmap_targets(objects, points)
        heatmap_list.append(targets["heatmap"])
        mask_list.append(targets["reg_mask"])
        offset_list.append(targets["offset"])
        z_list.append(targets["z"])
        dim_list.append(targets["dim"])
        rot_list.append(targets["rot"])
        density_list.append(targets["density"])
        print(f"  frame {frame_idx}: objects={len(objects)}, rot target shape={targets['rot'].shape}, "
              f"reg_mask positives={targets['reg_mask'].sum()}")

    voxel_features = torch.from_numpy(np.concatenate(vf_list, axis=0)).float()
    coords = torch.from_numpy(np.concatenate(coords_list, axis=0)).long()
    num_points = torch.from_numpy(np.concatenate(npts_list, axis=0)).long()

    heatmap_t = torch.from_numpy(np.stack(heatmap_list)).float()      # (B,1,H,W)
    reg_mask_t = torch.from_numpy(np.stack(mask_list))                # (B,H,W) bool
    offset_t = torch.from_numpy(np.stack(offset_list)).float()        # (B,H,W,2)
    z_t = torch.from_numpy(np.stack(z_list)).float()                  # (B,H,W,1)
    dim_t = torch.from_numpy(np.stack(dim_list)).float()              # (B,H,W,3)
    rot_t = torch.from_numpy(np.stack(rot_list)).float()              # (B,H,W,6)
    density_t = torch.from_numpy(np.stack(density_list)).long()       # (B,H,W)

    model = VoxelNet(head="center")
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    print("\n=== forward ===")
    heatmap_pred, offset_pred, z_pred, dim_pred, rot_pred, density_pred = model(voxel_features, num_points, coords)
    print("heatmap_pred", heatmap_pred.shape, "offset_pred", offset_pred.shape,
          "z_pred", z_pred.shape, "dim_pred", dim_pred.shape, "rot_pred", rot_pred.shape,
          "density_pred", density_pred.shape)
    assert rot_pred.shape[1] == 6, f"rot_head 출력 채널이 6이어야 하는데 {rot_pred.shape[1]}"
    assert density_pred.shape[1] == 3, f"density_head 출력 채널이 3이어야 하는데 {density_pred.shape[1]}"

    print("\n=== loss ===")
    total, stats = center_voxelnet_loss(
        heatmap_pred, offset_pred, z_pred, dim_pred, rot_pred,
        heatmap_t, reg_mask_t, offset_t, z_t, dim_t, rot_t,
        density_pred=density_pred, density_target=density_t)
    print("total loss:", total.item())
    print("stats:", stats)
    assert "density_loss" in stats, "density_loss가 stats에 없음 - aux loss가 실제로 안 더해진 것"
    assert torch.isfinite(total), "loss가 NaN/Inf"

    print("\n=== backward ===")
    opt.zero_grad()
    total.backward()
    n_grad_none = sum(1 for p in model.parameters() if p.requires_grad and p.grad is None)
    n_grad_nan = sum(1 for p in model.parameters() if p.grad is not None and not torch.isfinite(p.grad).all())
    print(f"gradient 없는 파라미터: {n_grad_none}, NaN/Inf gradient: {n_grad_nan}")
    assert n_grad_nan == 0, "gradient에 NaN/Inf 있음"

    rot_head_grad = model.rpn.rot_head.weight.grad
    print(f"rot_head weight grad norm: {rot_head_grad.norm().item():.6f} "
          f"(0이면 안 됨 - 회전 loss가 실제로 rot_head까지 전파 안 된 것)")
    assert rot_head_grad.norm().item() > 0, "rot_head에 gradient가 전혀 안 흘렀음"

    density_head_grad = model.rpn.density_head.weight.grad
    print(f"density_head weight grad norm: {density_head_grad.norm().item():.6f}")
    assert density_head_grad.norm().item() > 0, "density_head에 gradient가 전혀 안 흘렀음"

    opt.step()
    print("\n모든 검증 통과 - center head 3D 회전 확장 파이프라인 정상")


if __name__ == "__main__":
    main()
