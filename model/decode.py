"""decode.py - RPN raw 출력(cls logit, reg residual) -> world-space 3D 박스.

anchor 좌표계에서 논문 Eq.1의 역변환(residual -> 절대 x,y,z,l,w,h,theta)을 적용한다.
NMS는 회전 박스라 shapely 폴리곤 기반 그리디 방식을 쓴다 - 프레임당 예측 수가
(우리 데이터 특성상) 많지 않아 O(n^2) 그리디로 충분하다."""

import math

import numpy as np
import torch
from shapely.geometry import Polygon

import config
import rotation3d
from anchors import rotated_rect_corners


def range_threshold_grid(anchors: np.ndarray, range_thresholds: list) -> np.ndarray:
    """anchors: (H,W,A,7). range_thresholds: (상한_미만, threshold) 튜플 리스트, hi 오름차순
    - 예: [(2,0.95),(2.5,0.5),(3,0.59),(3.5,0.87),(5,0.975),(inf,0.5)]. anchor world
    range(원점 기준 sqrt(x^2+y^2))가 어느 구간에 속하는지에 따라 각 anchor 위치의 threshold를
    정하는 (H,W,A) 배열을 만든다 - 매 프레임 anchor 좌표는 고정이라 학습/평가 시작 전
    한 번만 계산해 캐싱해두면 됨(Day 2 range-aware loss와 같은 재사용 패턴)."""
    anchor_range = np.hypot(anchors[..., 0], anchors[..., 1])
    grid = np.full(anchor_range.shape, range_thresholds[-1][1], dtype=np.float32)
    for hi, t in sorted(range_thresholds, reverse=True):
        grid[anchor_range < hi] = t
    return grid


def decode_boxes(cls_pred: torch.Tensor, reg_pred: torch.Tensor, anchors: np.ndarray, score_thresh: float = 0.3,
                  range_thresholds: list = None):
    """단일 샘플(배치 없음) 기준. cls_pred: (A,H,W). reg_pred: (A*12,H,W) - 마지막 6
    채널이 완전한 3D 회전의 6D representation(anchors.assign_targets/rotation3d.py 참고,
    GT 절대 회전을 그대로 회귀 - anchor heading 기준 잔차 아님, validate_rotation3d.py로
    잔차가 실제로는 이점이 없음을 확인함). anchors: (H,W,A,7).
    range_thresholds를 주면 score_thresh(스칼라) 대신 range 구간별 문턱값을 쓴다(1b - 재학습
    없는 사후 보정, range_threshold_grid() 참고) - 모델 출력은 그대로, "어느 후보를 진짜로
    인정할지" 기준만 바꾸는 순수 후처리라 어떤 체크포인트에도 바로 적용 가능.
    반환: list of dict {score, x,y,z,l,w,h,theta,R} - theta는 R의 z-yaw 성분만 뽑은
    근사값(기존 BEV NMS/footprint용), R(3,3)이 진짜 3D 회전(3D IoU 평가용)."""
    A, H, W = cls_pred.shape
    scores = torch.sigmoid(cls_pred).permute(1, 2, 0).detach().cpu().numpy()  # (H,W,A)
    reg = reg_pred.view(A, 12, H, W).permute(2, 3, 0, 1).detach().cpu().numpy()  # (H,W,A,12)

    thresh = range_threshold_grid(anchors, range_thresholds) if range_thresholds is not None else score_thresh

    boxes = []
    rows, cols, a_idx = np.where(scores >= thresh)
    for r, c, a in zip(rows, cols, a_idx):
        ax, ay, az, al, aw, ah, atheta = anchors[r, c, a]
        dx, dy, dz, dl, dw, dh = reg[r, c, a, 0:6]
        six = reg[r, c, a, 6:12]
        d_a = float(np.hypot(al, aw))
        x = dx * d_a + ax
        y = dy * d_a + ay
        z = dz * ah + az
        l = float(np.exp(dl)) * al
        w = float(np.exp(dw)) * aw
        h = float(np.exp(dh)) * ah
        R = rotation3d.sixd_to_matrix_np(six)
        theta = float(math.atan2(R[1, 0], R[0, 0]))  # BEV NMS/footprint용 z-yaw 근사
        boxes.append({"score": float(scores[r, c, a]), "x": x, "y": y, "z": z,
                       "l": l, "w": w, "h": h, "theta": theta, "R": R})
    return boxes


def rotated_nms(boxes: list, iou_thresh: float = 0.1) -> list:
    """그리디 NMS, shapely BEV 폴리곤 IoU 기준. boxes는 score 내림차순 가정 안 함(정렬함)."""
    boxes = sorted(boxes, key=lambda b: -b["score"])
    polys = [Polygon(rotated_rect_corners(b["x"], b["y"], b["l"], b["w"], b["theta"])) for b in boxes]
    keep = []
    suppressed = [False] * len(boxes)
    for i in range(len(boxes)):
        if suppressed[i]:
            continue
        keep.append(boxes[i])
        if not polys[i].is_valid or polys[i].area <= 0:
            continue
        for j in range(i + 1, len(boxes)):
            if suppressed[j] or not polys[j].is_valid or polys[j].area <= 0:
                continue
            inter = polys[i].intersection(polys[j]).area
            union = polys[i].area + polys[j].area - inter
            iou = inter / union if union > 0 else 0.0
            if iou > iou_thresh:
                suppressed[j] = True
    return keep


def box_to_footprint_and_z(box: dict):
    """eval_3d_iou.iou_3d와 바로 맞물리는 형식: (footprint(4,2), (z_bottom,z_top))."""
    footprint = rotated_rect_corners(box["x"], box["y"], box["l"], box["w"], box["theta"])
    z_bottom, z_top = box["z"] - box["h"] / 2, box["z"] + box["h"] / 2
    return footprint, (z_bottom, z_top)
