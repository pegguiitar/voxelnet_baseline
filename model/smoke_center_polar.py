"""smoke_center_polar.py - voxel polarization Phase1(Cylinder3D식) 파이프라인의 item4
검증(round-trip + CPU forward/backward smoke test). smoke_center.py와 같은 역할,
polar 버전(voxelize_polar/build_heatmap_targets_polar/decode_center_boxes_polar/
VoxelNet(polar=True))에 대해.

round-trip: build_heatmap_targets_polar()로 만든 타겟(offset/z/dim/rot)을 그대로
decode_center_boxes_polar()에 "완벽한 예측"인 척 넣었을 때, 원본 GT 중심/치수/회전을
거의 정확히(수치 오차만) 복원하는지 확인 - world->polar grid->world 변환 자체에
버그가 없는지의 핵심 검증.

Usage:
    python smoke_center_polar.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Triband_BEV" / "baseline"))
import common  # noqa: E402
import filter_outliers  # noqa: E402

import config
import heatmap_targets as ht
import rotation3d
from center_loss import center_voxelnet_loss
from model import VoxelNet
from voxelize import augment_with_centroid_offset, voxelize_polar


def load_points_polar(scene_id: str, frame_idx: int) -> np.ndarray:
    raw = np.fromfile(common.sonar_bin_path(scene_id, frame_idx), dtype=np.float32)
    points = raw.reshape(-1, 4 if raw.size % 4 == 0 else 3)
    if points.shape[1] == 3:
        points = np.concatenate([points, np.zeros((len(points), 1), dtype=np.float32)], axis=1)
    return points[~np.isnan(points).any(axis=1)]


def load_frame_polar(points: np.ndarray):
    voxel_xyzr, coords, num_points = voxelize_polar(
        points, config.POLAR_R_RANGE, config.POLAR_THETA_RANGE_DEG,
        (config.POINT_CLOUD_RANGE[2], config.POINT_CLOUD_RANGE[5]),
        config.POLAR_R_BINS, config.POLAR_THETA_BINS, config.GRID_SIZE[2],
        config.MAX_POINTS_PER_VOXEL, config.MAX_VOXELS)
    voxel_features = augment_with_centroid_offset(voxel_xyzr, num_points)
    return voxel_features, coords, num_points


def check_roundtrip(frames):
    print("=== round-trip 검증 (world -> polar target -> decode -> world) ===")
    max_center_err, max_dim_err, max_rot_err = 0.0, 0.0, 0.0
    n_checked = 0
    for scene_id, frame_idx, objects in frames:
        points = load_points_polar(scene_id, frame_idx)
        targets = ht.build_heatmap_targets_polar(objects, points)
        H, W = targets["heatmap"].shape[1:]
        # "완벽한 예측" - heatmap을 reg_mask 위치에서 1.0으로, 나머지 타겟은 그대로 재사용
        heatmap_pred = np.where(targets["reg_mask"][None], 1.0, 0.0).astype(np.float32)
        boxes = ht.decode_center_boxes_polar(
            heatmap_pred, targets["offset"], targets["z"], targets["dim"], targets["rot"], score_thresh=0.5)

        for o in objects:
            c, d = o["centroid"], o["dimensions"]
            # 가장 가까운 decoded box를 찾아 대조(같은 프레임에 객체가 1-2개뿐이라 range로 충분)
            if not boxes:
                continue
            best = min(boxes, key=lambda b: (b["x"] - c["x"]) ** 2 + (b["y"] - c["y"]) ** 2)
            center_err = float(np.hypot(best["x"] - c["x"], best["y"] - c["y"]))
            dim_err = float(max(abs(best["l"] - d["length"]), abs(best["w"] - d["width"]), abs(best["h"] - d["height"])))
            rx, ry, rz = o.get("rotation_x", 0.0), o.get("rotation_y", 0.0), o.get("rotation_z", 0.0)
            R_true = rotation3d.euler_to_matrix(rx, ry, rz)
            rot_err = float(np.abs(best["R"] - R_true).max())
            max_center_err = max(max_center_err, center_err)
            max_dim_err = max(max_dim_err, dim_err)
            max_rot_err = max(max_rot_err, rot_err)
            n_checked += 1

    print(f"n_checked={n_checked}")
    print(f"max center 오차: {max_center_err:.6f} m (0에 가까워야 함 - offset이 Cartesian이라 정확 복원 기대)")
    print(f"max dim 오차: {max_dim_err:.6f} m")
    print(f"max rotation(R 원소) 오차: {max_rot_err:.6f}")
    assert max_center_err < 1e-3, "round-trip center 오차가 너무 큼 - offset/decode 공식 버그 의심"
    assert max_dim_err < 1e-3, "round-trip dim 오차가 너무 큼"
    assert max_rot_err < 1e-4, "round-trip rotation 오차가 너무 큼"
    print("round-trip 통과\n")


def check_smoke(frames):
    print("=== CPU forward/backward smoke test (polar) ===")
    vf_list, coords_list, npts_list = [], [], []
    heatmap_list, mask_list, offset_list, z_list, dim_list, rot_list, density_list = [], [], [], [], [], [], []
    for bi, (scene_id, frame_idx, objects) in enumerate(frames):
        points = load_points_polar(scene_id, frame_idx)
        vf, coords, npts = load_frame_polar(points)
        batch_col = np.full((coords.shape[0], 1), bi, dtype=coords.dtype)
        coords_list.append(np.concatenate([batch_col, coords], axis=1))
        vf_list.append(vf)
        npts_list.append(npts)

        targets = ht.build_heatmap_targets_polar(objects, points)
        heatmap_list.append(targets["heatmap"])
        mask_list.append(targets["reg_mask"])
        offset_list.append(targets["offset"])
        z_list.append(targets["z"])
        dim_list.append(targets["dim"])
        rot_list.append(targets["rot"])
        density_list.append(targets["density"])
        print(f"  frame {frame_idx}: objects={len(objects)}, voxels={vf.shape[0]}, "
              f"reg_mask positives={targets['reg_mask'].sum()}")

    voxel_features = torch.from_numpy(np.concatenate(vf_list, axis=0)).float()
    coords = torch.from_numpy(np.concatenate(coords_list, axis=0)).long()
    num_points = torch.from_numpy(np.concatenate(npts_list, axis=0)).long()

    heatmap_t = torch.from_numpy(np.stack(heatmap_list)).float()
    reg_mask_t = torch.from_numpy(np.stack(mask_list))
    offset_t = torch.from_numpy(np.stack(offset_list)).float()
    z_t = torch.from_numpy(np.stack(z_list)).float()
    dim_t = torch.from_numpy(np.stack(dim_list)).float()
    rot_t = torch.from_numpy(np.stack(rot_list)).float()
    density_t = torch.from_numpy(np.stack(density_list)).long()

    model = VoxelNet(head="center", polar=True)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    heatmap_pred, offset_pred, z_pred, dim_pred, rot_pred, density_pred = model(voxel_features, num_points, coords)
    print("\nheatmap_pred", heatmap_pred.shape, "offset_pred", offset_pred.shape,
          "z_pred", z_pred.shape, "dim_pred", dim_pred.shape, "rot_pred", rot_pred.shape,
          "density_pred", density_pred.shape)
    expected_H, expected_W = config.POLAR_HEATMAP_R_BINS, config.POLAR_HEATMAP_THETA_BINS
    assert heatmap_pred.shape[2:] == (expected_H, expected_W), \
        f"heatmap 출력 해상도가 기대({expected_H},{expected_W})와 다름: {heatmap_pred.shape[2:]}"
    assert rot_pred.shape[1] == 6
    assert density_pred.shape[1] == 3

    total, stats = center_voxelnet_loss(
        heatmap_pred, offset_pred, z_pred, dim_pred, rot_pred,
        heatmap_t, reg_mask_t, offset_t, z_t, dim_t, rot_t,
        density_pred=density_pred, density_target=density_t)
    print("total loss:", total.item(), "stats:", stats)
    assert "density_loss" in stats, "density_loss가 stats에 없음"
    assert torch.isfinite(total), "loss가 NaN/Inf"

    opt.zero_grad()
    total.backward()
    n_grad_nan = sum(1 for p in model.parameters() if p.grad is not None and not torch.isfinite(p.grad).all())
    print(f"NaN/Inf gradient 파라미터 수: {n_grad_nan}")
    assert n_grad_nan == 0

    rot_head_grad = model.rpn.rot_head.weight.grad
    print(f"rot_head weight grad norm: {rot_head_grad.norm().item():.6f}")
    assert rot_head_grad.norm().item() > 0

    density_head_grad = model.rpn.density_head.weight.grad
    print(f"density_head weight grad norm: {density_head_grad.norm().item():.6f}")
    assert density_head_grad.norm().item() > 0

    opt.step()
    print("\nCPU forward/backward smoke test 통과")


def check_smoke_grr(frames):
    """PARTNER GRR(partner_grr.py) 삽입 버전 - check_smoke()와 동일하지만
    VoxelNet(use_grr=True)로 구성해 GRR 서브모듈(condense/angular1/angular2/reverse)까지
    gradient가 전부 흐르는지 확인한다."""
    print("=== CPU forward/backward smoke test (polar + GRR) ===")
    vf_list, coords_list, npts_list = [], [], []
    heatmap_list, mask_list, offset_list, z_list, dim_list, rot_list, density_list = [], [], [], [], [], [], []
    for bi, (scene_id, frame_idx, objects) in enumerate(frames):
        points = load_points_polar(scene_id, frame_idx)
        vf, coords, npts = load_frame_polar(points)
        batch_col = np.full((coords.shape[0], 1), bi, dtype=coords.dtype)
        coords_list.append(np.concatenate([batch_col, coords], axis=1))
        vf_list.append(vf)
        npts_list.append(npts)

        targets = ht.build_heatmap_targets_polar(objects, points)
        heatmap_list.append(targets["heatmap"])
        mask_list.append(targets["reg_mask"])
        offset_list.append(targets["offset"])
        z_list.append(targets["z"])
        dim_list.append(targets["dim"])
        rot_list.append(targets["rot"])
        density_list.append(targets["density"])

    voxel_features = torch.from_numpy(np.concatenate(vf_list, axis=0)).float()
    coords = torch.from_numpy(np.concatenate(coords_list, axis=0)).long()
    num_points = torch.from_numpy(np.concatenate(npts_list, axis=0)).long()

    heatmap_t = torch.from_numpy(np.stack(heatmap_list)).float()
    reg_mask_t = torch.from_numpy(np.stack(mask_list))
    offset_t = torch.from_numpy(np.stack(offset_list)).float()
    z_t = torch.from_numpy(np.stack(z_list)).float()
    dim_t = torch.from_numpy(np.stack(dim_list)).float()
    rot_t = torch.from_numpy(np.stack(rot_list)).float()
    density_t = torch.from_numpy(np.stack(density_list)).long()

    model = VoxelNet(head="center", polar=True, use_grr=True)
    assert model.grr is not None
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    heatmap_pred, offset_pred, z_pred, dim_pred, rot_pred, density_pred = model(voxel_features, num_points, coords)
    print("heatmap_pred", heatmap_pred.shape, "offset_pred", offset_pred.shape,
          "z_pred", z_pred.shape, "dim_pred", dim_pred.shape, "rot_pred", rot_pred.shape,
          "density_pred", density_pred.shape)
    expected_H, expected_W = config.POLAR_HEATMAP_R_BINS, config.POLAR_HEATMAP_THETA_BINS
    assert heatmap_pred.shape[2:] == (expected_H, expected_W), \
        f"GRR 삽입 후에도 RPNBackbone 출력 해상도가 그대로여야 함(GRR은 shape-preserving): {heatmap_pred.shape[2:]}"

    total, stats = center_voxelnet_loss(
        heatmap_pred, offset_pred, z_pred, dim_pred, rot_pred,
        heatmap_t, reg_mask_t, offset_t, z_t, dim_t, rot_t,
        density_pred=density_pred, density_target=density_t)
    print("total loss:", total.item())
    assert torch.isfinite(total), "loss가 NaN/Inf"

    opt.zero_grad()
    total.backward()
    n_grad_nan = sum(1 for p in model.parameters() if p.grad is not None and not torch.isfinite(p.grad).all())
    print(f"NaN/Inf gradient 파라미터 수: {n_grad_nan}")
    assert n_grad_nan == 0

    # residual_gate는 0으로 초기화되므로(LayerScale류 zero-init gate, GRRModule 참고)
    # 첫 backward에선 gate 자체의 gradient만 nonzero이고 그 안쪽(condense/angular/reverse)
    # 서브모듈로는 gradient가 안 흐르는 게 정상 동작이다 - 여기서 확인.
    gate_grad = model.grr.residual_gate.grad
    assert gate_grad is not None and gate_grad.abs().item() > 0, \
        "residual_gate에 gradient가 없음 - GRR 출력이 loss와 연결이 안 됐을 가능성"
    print(f"  residual_gate grad (1st pass, gate=0이라 이것만 nonzero가 정상): {gate_grad.item():.6f}")
    grr_submodules = {"condense": model.grr.condense, "angular1": model.grr.angular1,
                       "angular2": model.grr.angular2, "reverse": model.grr.reverse}
    for name, sub in grr_submodules.items():
        grad_norms = [p.grad.norm().item() for p in sub.parameters() if p.grad is not None]
        total_norm = sum(grad_norms) if grad_norms else 0.0
        assert total_norm == 0, f"GRR.{name}이 1st pass에 이미 nonzero gradient - gate=0인데 이상함"

    opt.step()  # gate가 0에서 벗어남

    # 2nd pass - 이제 gate!=0이라 condense/angular/reverse까지 gradient가 흘러야 정상.
    heatmap_pred, offset_pred, z_pred, dim_pred, rot_pred, density_pred = model(voxel_features, num_points, coords)
    total2, _ = center_voxelnet_loss(
        heatmap_pred, offset_pred, z_pred, dim_pred, rot_pred,
        heatmap_t, reg_mask_t, offset_t, z_t, dim_t, rot_t,
        density_pred=density_pred, density_target=density_t)
    opt.zero_grad()
    total2.backward()
    assert torch.isfinite(total2), "2nd pass loss가 NaN/Inf"
    for name, sub in grr_submodules.items():
        grad_norms = [p.grad.norm().item() for p in sub.parameters() if p.grad is not None]
        assert grad_norms, f"GRR.{name}에 gradient가 있는 파라미터가 하나도 없음(2nd pass)"
        total_norm = sum(grad_norms)
        print(f"  GRR.{name} grad norm sum (2nd pass, gate={model.grr.residual_gate.item():.6f}): {total_norm:.6f}")
        assert total_norm > 0, f"GRR.{name}의 gradient가 2nd pass에도 전부 0"
    opt.step()
    print("\nCPU forward/backward smoke test (GRR) 통과")


def main():
    flagged_keys = filter_outliers.load_flagged_keys()
    frames = []
    for scene_id in ("scene_0044", "scene_0000", "scene_0068"):
        for frame_idx, objects in list(filter_outliers.iter_filtered_frames(scene_id, flagged_keys))[:4]:
            if objects:
                frames.append((scene_id, frame_idx, objects))
    print(f"{len(frames)}개 프레임으로 검증\n")

    check_roundtrip(frames)
    check_smoke(frames[:4])
    check_smoke_grr(frames[:4])
    print("\n=== 모든 검증 통과 - polar Phase1 + GRR 파이프라인 정상 ===")


if __name__ == "__main__":
    main()
