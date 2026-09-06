"""anchors.py - BEV anchor grid 생성 + IoU 기반 pos/neg/ignore 타겟 할당 (논문 §2.2).

전수 IoU(anchor 3000개 x gt) 대신 gt 박스 주변 국소 윈도우만 본다 - 프레임당 객체가
평균 0.84개뿐이고 박스 크기(~1.5m)가 anchor grid 셀(0.2m)보다 훨씬 커서, 실제로 겹칠
수 있는 anchor는 gt 중심에서 반경 ~2m 이내로 한정된다(대각선 반지름 합 기준). 이
가지치기 덕에 shapely 폴리곤 IoU를 anchor마다 개별 호출해도 프레임당 충분히 빠르다.
"""

import numpy as np
from shapely.geometry import Polygon

import config
import rotation3d


def rotated_rect_corners(x: float, y: float, l: float, w: float, theta: float) -> np.ndarray:
    """(4,2) - 중심(x,y), 길이 l(로컬 x축), 폭 w(로컬 y축), theta(라디안, z축 회전)."""
    hl, hw = l / 2, w / 2
    local = np.array([[-hl, -hw], [-hl, hw], [hl, hw], [hl, -hw]])
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return local @ R.T + np.array([x, y])


def build_anchor_grid():
    """anchors: (H'',W'',A,7) float32 [x,y,z,l,w,h,theta], world 미터/라디안.
    A=len(ANCHOR_ROTATIONS)."""
    W, H = config.ANCHOR_GRID_SIZE
    sx, sy = config.ANCHOR_STRIDE
    x0, y0 = config.POINT_CLOUD_RANGE[0], config.POINT_CLOUD_RANGE[1]
    xs = x0 + sx * (np.arange(W) + 0.5)
    ys = y0 + sy * (np.arange(H) + 0.5)
    gx, gy = np.meshgrid(xs, ys)  # (H,W) each
    l, w, h = config.ANCHOR_SIZE
    z = config.ANCHOR_Z_CENTER
    rots = config.ANCHOR_ROTATIONS
    A = len(rots)
    anchors = np.zeros((H, W, A, 7), dtype=np.float32)
    for a, theta in enumerate(rots):
        anchors[:, :, a, 0] = gx
        anchors[:, :, a, 1] = gy
        anchors[:, :, a, 2] = z
        anchors[:, :, a, 3] = l
        anchors[:, :, a, 4] = w
        anchors[:, :, a, 5] = h
        anchors[:, :, a, 6] = theta
    return anchors




def gt_boxes_from_objects(objects: list) -> np.ndarray:
    """common.sonar_objects() 결과 -> (M,13) [x,y,z,l,w,h,theta_z_rad, 6D(6)].

    theta_z_rad(idx 6): rotation_z만 반영 - anchor 할당(IoU)은 지금도 BEV(z-yaw)
    폴리곤으로만 계산한다(project_rotation_xy_full_coverage_findings 참고 - "어디에
    다이버가 있는지" 판정은 BEV로 충분하고, 3D OBB-OBB IoU로 바꾸는 건 훨씬 큰 리스크라
    의도적으로 스코프 제한함). 6D(idx 7-12, rotation3d.matrix_to_6d): rotation_x/y/z
    전부 반영한 완전한 3D 회전 - positive로 뽑힌 자리의 회귀 타겟에만 쓰인다.

    x,y 회전을 무시하면 82.5%의 박스에서 점 소속 판정이 틀리고 높이 오차 중앙값
    0.35m라는 게 실측 확인됨(rotation_xy_impact.py, 100% coverage) - 이 데이터셋을
    라벨로서 의미있게 쓰려면 회귀 타겟이 실제 3D 형상을 반영해야 한다는 게 근거."""
    out = []
    for o in objects:
        c, d = o["centroid"], o["dimensions"]
        rx = o.get("rotation_x", 0.0)
        ry = o.get("rotation_y", 0.0)
        rz = o.get("rotation_z", 0.0)
        theta = np.radians(rz)
        R = rotation3d.euler_to_matrix(rx, ry, rz)
        six = rotation3d.matrix_to_6d(R)
        out.append([c["x"], c["y"], c["z"], d["length"], d["width"], d["height"], theta, *six])
    return np.asarray(out, dtype=np.float32) if out else np.zeros((0, 13), dtype=np.float32)


def assign_targets(anchors: np.ndarray, objects: list):
    """anchors: (H,W,A,7). objects: raw label dicts (centroid/dimensions/rotation_x/y/z).

    반환:
      cls_labels: (H,W,A) int64 {1: pos, 0: neg, -1: ignore/don't-care}
      reg_targets: (H,W,A,12) float32, positive가 아닌 곳은 0 -
      [dx,dy,dz,dl,dw,dh, 6D 잔차회전(6)].
      회전은 원래 z축 하나(raw residual 또는 sin/cos)였는데, x,y 회전까지 있는 게
      확인돼(rotation_xy_impact.py, 82.5% Jaccard<0.9) 완전한 3D 회전으로 확장했다.
      Euler 3축이나 quaternion을 직접 회귀하지 않고 6D continuous representation
      (Zhou et al., CVPR 2019)을 쓴다 - rotation3d.py 참고, gimbal lock/wraparound가
      전혀 없다.

      **GT의 절대 6D 회전을 그대로 타겟으로 쓴다(anchor heading 기준 잔차 아님)**.
      원래는 sin/cos 때처럼 "anchor heading 기준 잔차가 작으니 회귀가 쉬울 것"이라고
      가정하고 R_anchor^T @ R_gt를 시도했었는데, validate_rotation3d.py로 실측해보니
      틀린 가정이었음 - 다이버 박스의 tilt 크기가 중앙값 81도(대부분 수평으로 헤엄치는
      자세라 로컬 z축이 세계좌표 수평에 가까움)라, z-yaw만 아는 anchor heading 기준
      잔차도 평균 89.7도로 사실상 무작위 수준(작은 잔차라는 이점이 없음) - 그래서
      복잡성만 더하는 잔차 방식을 버리고 절대 회전을 직접 회귀하는 것으로 단순화함.

      단, positive/negative 판정(아래 IoU 계산)은 여전히 BEV(z-yaw) 폴리곤만 본다 -
      "어디에 다이버가 있는지" 판정에는 충분하고, 3D OBB-OBB IoU로 바꾸는 건
      훨씬 큰 리스크라 의도적으로 스코프 제한함(anchor 자체는 이미 참고용이고
      center head가 최종 채택된 아키텍처)."""
    H, W, A, _ = anchors.shape
    cls_labels = np.zeros((H, W, A), dtype=np.int64)
    reg_targets = np.zeros((H, W, A, 12), dtype=np.float32)
    gt_boxes = gt_boxes_from_objects(objects)
    if len(gt_boxes) == 0:
        return cls_labels, reg_targets  # 전부 negative (라벨 없는 프레임은 호출측에서 걸러냄)

    max_iou = np.zeros((H, W, A), dtype=np.float32)
    best_gt_idx = -np.ones((H, W, A), dtype=np.int64)
    sx, sy = config.ANCHOR_STRIDE
    l_a, w_a = config.ANCHOR_SIZE[0], config.ANCHOR_SIZE[1]
    anchor_diag = float(np.hypot(l_a, w_a))
    x0, y0 = config.POINT_CLOUD_RANGE[0], config.POINT_CLOUD_RANGE[1]

    force_positive = []  # (row,col,a) - gt별 최고 IoU anchor, 0.6 미만이어도 positive로 강제
    for gi, gt in enumerate(gt_boxes):
        gx, gy, gl, gw, gtheta = gt[0], gt[1], gt[3], gt[4], gt[6]
        radius = anchor_diag / 2 + float(np.hypot(gl, gw)) / 2 + max(sx, sy)  # 여유 1셀
        c0 = max(0, int((gx - radius - x0) / sx))
        c1 = min(W, int((gx + radius - x0) / sx) + 1)
        r0 = max(0, int((gy - radius - y0) / sy))
        r1 = min(H, int((gy + radius - y0) / sy) + 1)
        gt_poly = Polygon(rotated_rect_corners(gx, gy, gl, gw, gtheta))
        if not gt_poly.is_valid or gt_poly.area <= 0:
            continue

        best_iou_this_gt, best_cell_this_gt = -1.0, None
        for r in range(r0, r1):
            for c in range(c0, c1):
                for a in range(A):
                    ax, ay, al, aw, atheta = anchors[r, c, a, 0], anchors[r, c, a, 1], \
                        anchors[r, c, a, 3], anchors[r, c, a, 4], anchors[r, c, a, 6]
                    anchor_poly = Polygon(rotated_rect_corners(ax, ay, al, aw, atheta))
                    inter = gt_poly.intersection(anchor_poly).area
                    if inter <= 0:
                        continue
                    union = gt_poly.area + anchor_poly.area - inter
                    iou = inter / union if union > 0 else 0.0
                    if iou > max_iou[r, c, a]:
                        max_iou[r, c, a] = iou
                        best_gt_idx[r, c, a] = gi
                    if iou > best_iou_this_gt:
                        best_iou_this_gt, best_cell_this_gt = iou, (r, c, a)
        if best_cell_this_gt is not None and best_iou_this_gt > 0:
            r, c, a = best_cell_this_gt
            force_positive.append((r, c, a, gi))

    cls_labels[max_iou >= config.POS_IOU_THRESH] = 1
    cls_labels[max_iou < config.NEG_IOU_THRESH] = 0
    ignore = (max_iou >= config.NEG_IOU_THRESH) & (max_iou < config.POS_IOU_THRESH)
    cls_labels[ignore] = -1
    for (r, c, a, gi) in force_positive:
        cls_labels[r, c, a] = 1
        best_gt_idx[r, c, a] = gi  # 여러 gt가 같은 anchor를 강제하면 마지막(가장 가까운) gt 우선

    pos_idx = np.argwhere(cls_labels == 1)
    for r, c, a in pos_idx:
        gi = best_gt_idx[r, c, a]
        gt = gt_boxes[gi]
        ax, ay, az, al, aw, ah, atheta = anchors[r, c, a]
        d_a = float(np.hypot(al, aw))
        gx, gy, gz, gl, gw, gh = gt[0], gt[1], gt[2], gt[3], gt[4], gt[5]
        g6d = gt[7:13]  # GT의 절대 6D 회전(anchor heading과 무관) - 위 docstring 참고
        reg_targets[r, c, a] = [
            (gx - ax) / d_a, (gy - ay) / d_a, (gz - az) / ah,
            np.log(gl / al), np.log(gw / aw), np.log(gh / ah),
            *g6d,
        ]
    return cls_labels, reg_targets
