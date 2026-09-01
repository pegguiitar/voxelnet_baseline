"""gather_frame.py - `Triband_BEV/dashboard/app.py`가 모델 구분 없이 재사용하는 공용
GUI에 VoxelNet을 꽂기 위한 어댑터.

`Triband_BEV/model/visualize_3d.py`의 `gather_frame()`과 반환 계약을 정확히 맞춘다:
(points, gt_objects, gt_boxes, pred_boxes, raw_pred). gt_boxes/pred_boxes가 둘 다
{"corners": (8,3), ...} 형식이라, 대시보드의 3D 인터랙티브 플롯 함수
(`visualize_3d.build_interactive_figure`)는 YOLO/VoxelNet 어느 쪽이 만든 박스든
수정 없이 그대로 그릴 수 있다 - 이게 "공용" 시각화의 핵심이다. raw_pred는 YOLO
전용(2D 오버레이 재사용 목적)이라 VoxelNet에는 없는 개념 - 항상 None.

GT 박스는 Triband_BEV/baseline/box3d.py를 그대로 재사용한다(같은 함수, 같은 z-only
가정) - TriBand-BEV 패널과 VoxelNet 패널에서 같은 GT 프레임을 봐도 다르게 그려지는
일이 없도록.
"""

import sys
from pathlib import Path

import numpy as np
import torch

import anchors as anchors_mod
import config
import heatmap_targets as ht
from decode import decode_boxes, rotated_nms
from eval_voxelnet import obb_corners
from voxelize import augment_with_centroid_offset, voxelize, voxelize_polar

TRIBAND_BASELINE = Path(__file__).resolve().parent.parent.parent / "Triband_BEV" / "baseline"
if str(TRIBAND_BASELINE) not in sys.path:
    sys.path.insert(0, str(TRIBAND_BASELINE))
import box3d  # noqa: E402
import common  # noqa: E402
import filter_outliers  # noqa: E402

_ANCHOR_GRID = anchors_mod.build_anchor_grid()  # config만의 함수 - 프레임마다 다시 지을 필요 없음


def load_model(ckpt_path: str, device: str = "cpu"):
    """head/polar를 체크포인트에 저장된 값으로 자동 감지한다(eval_voxelnet.py:main()과
    동일 관례) - 예전엔 항상 기본값(head="anchor")으로 로드해서 center-head 체크포인트를
    잘못 읽고 있었다."""
    from model import VoxelNet, load_state_dict_compat
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    head = ckpt.get("head", "anchor")
    polar = ckpt.get("polar", False)
    use_grr = ckpt.get("use_grr", False)
    model = VoxelNet(head=head, polar=polar, use_grr=use_grr).to(device)
    load_state_dict_compat(model, ckpt["model"])
    model.eval()
    return model, ckpt.get("epoch")


def gather_frame(model, scene_id: str, frame_idx: int, score_thresh: float = 0.3,
                  nms_iou: float = 0.1, device: str = "cpu"):
    """visualize_3d.gather_frame()과 동일 계약: (points, gt_objects, gt_boxes, pred_boxes, raw_pred).
    voxelize -> forward -> decode -> NMS까지 전부 이 함수 안에서 즉석 계산(로컬 CPU 전제,
    프레임 1개라 수백 ms 수준)."""
    points = np.fromfile(common.sonar_bin_path(scene_id, frame_idx), dtype=np.float32).reshape(-1, 4)
    points = points[~np.isnan(points).any(axis=1)]

    gt_by_frame = dict(filter_outliers.iter_filtered_frames(scene_id))
    gt_objects = gt_by_frame.get(frame_idx, [])
    gt_boxes = []
    for o in gt_objects:
        entry = {"corners": box3d.corners_3d(o, use_xy_rotation=False),
                 "link_id": o.get("link_id"), "obj": o}
        if box3d.has_xy_rotation(o):
            entry["corners_true"] = box3d.corners_3d(o, use_xy_rotation=True)
        gt_boxes.append(entry)

    if model.polar:
        voxel_xyzr, coords, num_points = voxelize_polar(
            points, config.POLAR_R_RANGE, config.POLAR_THETA_RANGE_DEG,
            (config.POINT_CLOUD_RANGE[2], config.POINT_CLOUD_RANGE[5]),
            config.POLAR_R_BINS, config.POLAR_THETA_BINS, config.GRID_SIZE[2],
            config.MAX_POINTS_PER_VOXEL, config.MAX_VOXELS)
    else:
        voxel_xyzr, coords, num_points = voxelize(
            points, config.POINT_CLOUD_RANGE, config.VOXEL_SIZE,
            config.MAX_POINTS_PER_VOXEL, config.MAX_VOXELS)
    voxel_features = augment_with_centroid_offset(voxel_xyzr, num_points)

    with torch.no_grad():
        voxel_features_t = torch.from_numpy(voxel_features).to(device)
        num_points_t = torch.from_numpy(num_points).to(device)
        batch_col = torch.zeros((len(coords), 1), dtype=torch.int64)
        coords_t = torch.cat([batch_col, torch.from_numpy(coords)], dim=1).to(device)

        if model.head == "anchor":
            cls_pred, reg_pred = model(voxel_features_t, num_points_t, coords_t)
            boxes = decode_boxes(cls_pred[0], reg_pred[0], _ANCHOR_GRID, score_thresh=score_thresh)
        else:
            hm, off, z, dim, rot, _density = model(voxel_features_t, num_points_t, coords_t)
            hm_np = torch.sigmoid(hm[0]).cpu().numpy()
            decode_fn = ht.decode_center_boxes_polar if model.polar else ht.decode_center_boxes
            boxes = decode_fn(
                hm_np, off[0].permute(1, 2, 0).cpu().numpy(), z[0].permute(1, 2, 0).cpu().numpy(),
                dim[0].permute(1, 2, 0).cpu().numpy(), rot[0].permute(1, 2, 0).cpu().numpy(),
                score_thresh=score_thresh)

    # decode_boxes()/decode_center_boxes() 둘 다 이번 세션 3D 회전 확장으로 "R"(3x3 회전행렬,
    # 6D->Gram-Schmidt 복원)을 반환하도록 바뀌었다 - 예전엔 여기서 footprint를 수직으로
    # 압출해(z-yaw만 반영) 실제 x,y 기울기를 무시하고 그렸는데, 이제 obb_corners()로 GT와
    # 완전히 동일한 방식(box3d.py의 local@R.T+center)으로 8개 코너를 만든다.
    boxes = rotated_nms(boxes, iou_thresh=nms_iou)

    pred_boxes = []
    for b in boxes:
        center = np.array([b["x"], b["y"], b["z"]])
        dims = np.array([b["l"], b["w"], b["h"]])
        corners = obb_corners(center, dims, b["R"])
        pred_boxes.append({"corners": corners, "world_xy": corners[:4, :2], "conf": float(b["score"])})

    return points, gt_objects, gt_boxes, pred_boxes, None


def render_topdown_overlay(points: np.ndarray, gt_boxes: list, pred_boxes: list) -> "Image.Image":
    """VoxelNet용 2D 오버레이. TriBand-BEV의 `evaluate.render_2d_overlay`는 BEV 래스터
    이미지(높이 밴드로 압축된 max-reflectance) 위에 그리지만, VoxelNet은 그런 래스터가
    없다(3D-native라 압축 안 함) - 대신 원본 포인트클라우드를 위에서 내려다본 산점도에
    같은 색 관례(GT=초록, Pred=주황)로 footprint를 그린다. gt_boxes/pred_boxes는
    {"corners": (8,3)} 형식만 있으면 되므로 어느 모델 출력이든 그릴 수 있는 범용 함수다."""
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    if len(points):
        ax.scatter(points[:, 1], points[:, 0], c=points[:, 3], cmap="viridis",
                   s=2, alpha=0.5, linewidths=0)

    def draw(boxes, color, label):
        for i, b in enumerate(boxes):
            footprint = b["corners"][:4, :2]  # bottom 4 corners, (x,y)
            xs = list(footprint[:, 1]) + [footprint[0, 1]]
            ys = list(footprint[:, 0]) + [footprint[0, 0]]
            ax.plot(xs, ys, color=color, linewidth=1.8, label=label if i == 0 else None)

    draw(gt_boxes, "#39ff14", "GT")
    draw(pred_boxes, "#ff8c1a", "Pred")

    ax.set_xlabel("y (lateral, m)")
    ax.set_ylabel("x (range, m)")
    ax.set_aspect("equal")
    ax.invert_xaxis()  # 소나가 +x(전방)를 보는 시점에서 왼쪽/오른쪽이 직관적으로 맞도록
    if gt_boxes or pred_boxes:
        ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)
