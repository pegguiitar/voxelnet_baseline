"""heatmap_targets.py - CenterPoint(Yin et al. 2021)식 anchor-free 타겟 생성.

VoxelNet의 원래 head(anchors.py)는 6,000개 고정 anchor 각각에 IoU 기준 pos/neg
라벨을 붙인다 - 매 프레임 같은 위치가 항상 positive/negative로 고정되는 정적
할당이라, 정밀도 격차 분석(reports/precision_gap_analysis.html §B2)에서 지목한
근본 원인이다. 이 모듈은 그 대신 객체 "중심"을 Gaussian heatmap으로 직접
회귀하는 CenterPoint 스타일 타겟을 만든다 - anchor 자체가 없다.

grid는 기존과 동일한 anchor grid(50x60, stride=0.2m, config.ANCHOR_STRIDE/
ANCHOR_GRID_SIZE)를 그대로 재사용 - RPN 백본 출력 해상도가 그대로이므로
grid를 새로 설계할 필요가 없다.

회귀 타겟은 GT 중심이 반올림되는 정확히 그 grid cell 하나에서만 감독한다
(CenterPoint 표준 관례) - offset(sub-pixel 보정), z(절대값 회귀, anchor 없음),
dim(log l/w/h), rot(sin/cos - 불연속 없는 연속 회전 표현).
"""

import math

import numpy as np

import config
import rotation3d


def _point_in_obb(points_xyz: np.ndarray, center: np.ndarray, dims: np.ndarray, R: np.ndarray) -> np.ndarray:
    """eval_voxelnet.point_in_obb()와 동일 로직 - 순환 import(eval_voxelnet도 이 모듈을
    import함) 회피용 로컬 복제. points_xyz: (N,3) -> (N,) bool."""
    local = (points_xyz - center) @ R
    return np.all(np.abs(local) <= dims / 2, axis=1)


def _density_class(n_points: int) -> int:
    """RAANet(arXiv:2111.09515)식 3클래스 - config.DENSITY_THRESH_LOW/HIGH 참고(전체
    GT 박스 point 개수 tertile 실측값)."""
    if n_points <= config.DENSITY_THRESH_LOW:
        return 0  # sparse
    if n_points > config.DENSITY_THRESH_HIGH:
        return 2  # dense
    return 1  # adequate


def gaussian_radius(height: float, width: float, min_overlap: float = 0.7, tau: float = 2.0) -> float:
    """CornerNet(Law&Deng 2018) 표준 공식 - 반지름 r인 Gaussian 피크와 실제 박스의
    IoU가 min_overlap 이상이 되도록 하는 최대 반지름. 3가지 코너-정렬 케이스의
    최솟값을 취한다(원 논문/CenterNet 참조 구현 그대로).

    tau: CenterPoint(Yin et al. 2021) §4.1이 명시한 최소 반지름 바닥값(논문 기본 2.0) -
    "map-view에서 객체 분포가 image-view보다 희소해 supervisory signal이 너무
    sparse해진다"며 도입한 값. 원래 이 프로젝트는 이 바닥값 없이 공식 결과를 그대로
    썼는데, 다이버 박스가 워낙 작아(실측 길이 0.5~1.8m) 이 grid(0.2m/cell)에서
    반지름이 거의 항상 0~1로 나와 - 사실상 점 하나짜리 supervision이었다. 이게
    recall-collapse(배경 dominant gradient가 confidence를 서서히 깎는 문제)의 숨은
    원인 후보로 확인되어 추가."""
    a1, b1 = 1, height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(max(b1 ** 2 - 4 * a1 * c1, 0))
    r1 = (b1 + sq1) / 2

    a2, b2 = 4, 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = math.sqrt(max(b2 ** 2 - 4 * a2 * c2, 0))
    r2 = (b2 + sq2) / 2

    a3, b3 = 4 * min_overlap, -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = math.sqrt(max(b3 ** 2 - 4 * a3 * c3, 0))
    r3 = (b3 + sq3) / 2

    return max(min(r1, r2, r3), tau)


def draw_gaussian(heatmap: np.ndarray, center_col: float, center_row: float, radius: float):
    """heatmap(H,W)에 (center_row,center_col) 중심 Gaussian을 in-place로 찍는다
    (기존 값과 max - 여러 객체가 겹치면 더 큰 쪽이 남음, CenterNet 관례)."""
    radius = max(int(round(radius)), 0)
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    gaussian = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma + 1e-9))
    gaussian[gaussian < np.finfo(gaussian.dtype).eps * gaussian.max()] = 0

    x, y = int(round(center_col)), int(round(center_row))
    height, width = heatmap.shape
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    if left + right <= 0 or top + bottom <= 0:
        return
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)


def build_heatmap_targets(objects: list, points: np.ndarray, min_overlap: float = 0.7,
                           rotation_repr: str = "6d") -> dict:
    """objects: raw label dicts (centroid/dimensions/rotation_x/y/z). points: (N,>=3)
    raw point cloud(voxelize 전, 이 프레임 원본) - density 타겟(GT 박스 안 실제 point
    개수 3클래스, RAANet arXiv:2111.09515식 보조 head용) 계산에만 쓴다.

    반환 (모두 grid=(H,W)=ANCHOR_GRID_SIZE[::-1] 기준):
      heatmap: (1,H,W) float32, 0~1
      reg_mask: (H,W) bool - 회귀를 감독할 정확히 그 셀만 True
      offset: (H,W,2) float32 [dx,dy] sub-pixel 보정 (셀 크기 단위)
      z: (H,W,1) float32 절대 z(m)
      dim: (H,W,3) float32 [log(l),log(w),log(h)]
      rot: (H,W,C) float32 - 완전한 3D 회전, GT 절대값을 그대로 씀(rotation_repr로 표현
      선택 - 기본 "6d"=rotation3d.matrix_to_6d, C=rotation3d.ROTATION_CHANNELS[rotation_repr]
      - rotation-representation ablation용, project_paper_sections_status 참고). center
      head는 anchor가 없어 애초에 "잔차" 개념 자체가 없었으므로 anchors.py에서 겪은
      "잔차 회귀가 실제로는 이점이 없다"는 문제가 여기엔 해당 안 됨(validate_rotation3d.py로
      확인한 tilt 중앙값 81도는 anchor-heading-기준 잔차에만 영향을 준 문제였음).
      density: (H,W) int64 - {0,1,2}(sparse/adequate/dense), positive 셀에서만 유효
      (reg_mask로 걸러서 씀 - 나머지는 0이지만 무의미한 값).

    참고(의도적 스코프 제한): heatmap 반지름(gaussian_radius)은 여전히 로컬
    length/width만 보고 계산한다 - 다이버가 크게 기울어져 있으면 실제 BEV footprint가
    이보다 작을 수 있지만, 이건 positive 영역을 얼마나 넓게 퍼뜨릴지 정하는 soft
    heuristic이라(CornerNet 공식 자체가 이미 근사임) 회전까지 반영해 재계산하는 건
    지금 안 함 - 양성 셀 판정(heatmap 자체)은 anchor 할당과 마찬가지로 BEV 근사를
    유지하고, 그 셀이 회귀해야 할 값(dim/rot)만 실제 3D 형상을 반영하도록 함."""
    W, H = config.ANCHOR_GRID_SIZE
    sx, sy = config.ANCHOR_STRIDE
    x0, y0 = config.POINT_CLOUD_RANGE[0], config.POINT_CLOUD_RANGE[1]
    points_xyz = points[:, :3]

    heatmap = np.zeros((1, H, W), dtype=np.float32)
    reg_mask = np.zeros((H, W), dtype=bool)
    offset = np.zeros((H, W, 2), dtype=np.float32)
    z_t = np.zeros((H, W, 1), dtype=np.float32)
    dim_t = np.zeros((H, W, 3), dtype=np.float32)
    rot_t = np.zeros((H, W, rotation3d.ROTATION_CHANNELS[rotation_repr]), dtype=np.float32)
    density_t = np.zeros((H, W), dtype=np.int64)

    for o in objects:
        c, d = o["centroid"], o["dimensions"]
        gx_cell = (c["x"] - x0) / sx - 0.5  # grid 중심 정렬(anchors.py의 +0.5와 대칭)
        gy_cell = (c["y"] - y0) / sy - 0.5
        col, row = int(round(gx_cell)), int(round(gy_cell))
        if not (0 <= col < W and 0 <= row < H):
            continue  # 범위 밖 중심 - 드묾(POINT_CLOUD_RANGE가 GT를 여유있게 감쌈)

        l_cells, w_cells = d["length"] / sx, d["width"] / sy
        radius = gaussian_radius(w_cells, l_cells, min_overlap)
        draw_gaussian(heatmap[0], gx_cell, gy_cell, radius)

        reg_mask[row, col] = True
        offset[row, col] = [gx_cell - col, gy_cell - row]
        z_t[row, col, 0] = c["z"]
        dim_t[row, col] = [np.log(d["length"]), np.log(d["width"]), np.log(d["height"])]
        rx, ry, rz = o.get("rotation_x", 0.0), o.get("rotation_y", 0.0), o.get("rotation_z", 0.0)
        R = rotation3d.euler_to_matrix(rx, ry, rz)
        rot_t[row, col] = rotation3d.encode_rotation(R, rotation_repr)

        center = np.array([c["x"], c["y"], c["z"]])
        dims = np.array([d["length"], d["width"], d["height"]])
        n_in_box = int(_point_in_obb(points_xyz, center, dims, R).sum())
        density_t[row, col] = _density_class(n_in_box)

    return {"heatmap": heatmap, "reg_mask": reg_mask, "offset": offset, "density": density_t,
            "z": z_t, "dim": dim_t, "rot": rot_t}


def _polar_heatmap_grid_params():
    """(H,W,r0,dr,t0,dtheta_rad) - heatmap 해상도(voxel 해상도의 절반, config.POLAR_HEATMAP_*)
    기준. H=r_bins, W=theta_bins(config.POLAR_GRID_SIZE의 (W',H',D')=(theta,r,z) 관례와 동일)."""
    r0, r1 = config.POLAR_R_RANGE
    t0, t1 = config.POLAR_THETA_RANGE_DEG
    H, W = config.POLAR_HEATMAP_R_BINS, config.POLAR_HEATMAP_THETA_BINS
    dr = (r1 - r0) / H
    dtheta_rad = math.radians((t1 - t0) / W)
    t0_rad = math.radians(t0)
    return H, W, r0, dr, t0_rad, dtheta_rad


def build_heatmap_targets_polar(objects: list, points: np.ndarray, min_overlap: float = 0.7) -> dict:
    """build_heatmap_targets()의 극좌표(Cylinder3D Phase1) 버전 - grid는 (r,theta) 셀
    (config.POLAR_HEATMAP_*), z축은 그대로. 문헌 근거(project_polarization_design 참고):
      - sub-cell offset: Cartesian (dx,dy) 미터, PolarStream(arXiv:2106.07545) Sec3.3 Eq.5
        방식(dx = gt_x - cell_center_x, dy = gt_y - cell_center_y) - (dr,dtheta) 아님.
        정규화 없이 raw 미터 그대로(Cartesian 버전의 "cell-fraction" 관례와는 다름 - 두
        경로가 서로 다른 문헌을 따르는 별개 구현이라 억지로 맞추지 않음).
      - heatmap gaussian radius: PolarStream Sec3.3 원안(GT footprint를 실제 회전대로
        (r,theta) 축에 투영한 span)은 물체 회전(rotation_z)에 따라 radius가 요동치는 문제가
        있었다 - 같은 range라도 센서를 정면으로 보는 물체와 옆으로 누운 물체의 투영폭이
        크게 달라져서(2026-08-18 재검증: r~2.5m 근처에서 radius가 floor 2.0~4.58까지 흔들림,
        표준편차 0.49), 물체의 실제 난이도와 무관한 노이즈가 supervision 크기에 섞였다.
        Cartesian처럼 물체 고유 length/width(회전 무관)를 쓰되, theta축 셀 크기만 range에
        따라 국소적으로(r*dtheta_rad, 물리적 호 길이) 환산해 range-awareness는 유지하면서
        회전 의존성을 제거한다 - Cartesian의 "회전 무시, 자기 dims만 사용" 관례와
        Polar의 "range마다 셀 크기가 다르다" 특성을 절충한 것.
    반환 shape은 build_heatmap_targets()와 동일 계약(heatmap/reg_mask/offset/z/dim/rot),
    grid만 (H,W)=(POLAR_HEATMAP_R_BINS,POLAR_HEATMAP_THETA_BINS)."""
    H, W, r0, dr, t0_rad, dtheta_rad = _polar_heatmap_grid_params()
    points_xyz = points[:, :3]

    heatmap = np.zeros((1, H, W), dtype=np.float32)
    reg_mask = np.zeros((H, W), dtype=bool)
    offset = np.zeros((H, W, 2), dtype=np.float32)
    z_t = np.zeros((H, W, 1), dtype=np.float32)
    dim_t = np.zeros((H, W, 3), dtype=np.float32)
    rot_t = np.zeros((H, W, 6), dtype=np.float32)
    density_t = np.zeros((H, W), dtype=np.int64)

    for o in objects:
        c, d = o["centroid"], o["dimensions"]
        gr = math.hypot(c["x"], c["y"])
        gtheta = math.atan2(c["y"], c["x"])
        row_f = (gr - r0) / dr
        col_f = (gtheta - t0_rad) / dtheta_rad
        row, col = int(math.floor(row_f)), int(math.floor(col_f))
        if not (0 <= row < H and 0 <= col < W):
            continue  # 범위 밖(드묾, POLAR_R_RANGE/THETA_RANGE가 실측 GT 100% 커버 확인됨)

        r_c = r0 + (row + 0.5) * dr
        theta_c = t0_rad + (col + 0.5) * dtheta_rad
        cell_cx, cell_cy = r_c * math.cos(theta_c), r_c * math.sin(theta_c)

        r_cells = d["length"] / dr
        theta_cells = d["width"] / (gr * dtheta_rad)  # gr(물체 range)에서의 국소 호 길이로 환산
        radius = gaussian_radius(theta_cells, r_cells, min_overlap)
        draw_gaussian(heatmap[0], col_f, row_f, radius)  # GT의 연속 grid 위치(셀 중심 아님) - Cartesian 버전과 동일 관례

        reg_mask[row, col] = True
        offset[row, col] = [c["x"] - cell_cx, c["y"] - cell_cy]  # Cartesian 미터(PolarStream Eq.5)
        z_t[row, col, 0] = c["z"]
        dim_t[row, col] = [np.log(d["length"]), np.log(d["width"]), np.log(d["height"])]
        rx, ry, rz = o.get("rotation_x", 0.0), o.get("rotation_y", 0.0), o.get("rotation_z", 0.0)
        R = rotation3d.euler_to_matrix(rx, ry, rz)
        rot_t[row, col] = rotation3d.matrix_to_6d(R)

        center = np.array([c["x"], c["y"], c["z"]])
        dims = np.array([d["length"], d["width"], d["height"]])
        n_in_box = int(_point_in_obb(points_xyz, center, dims, R).sum())
        density_t[row, col] = _density_class(n_in_box)

    return {"heatmap": heatmap, "reg_mask": reg_mask, "offset": offset, "density": density_t,
            "z": z_t, "dim": dim_t, "rot": rot_t}


def decode_center_boxes_polar(heatmap_pred, offset_pred, z_pred, dim_pred, rot_pred, score_thresh: float = 0.3,
                               max_peaks: int = 100):
    """decode_center_boxes()의 극좌표 버전. offset_pred가 Cartesian 미터라 셀 중심(극좌표
    -> Cartesian 변환)에 그냥 더하면 된다(build_heatmap_targets_polar()의 역변환)."""
    hm = heatmap_pred[0]
    H, W = hm.shape
    padded = np.full((H + 2, W + 2), -1.0, dtype=hm.dtype)
    padded[1:-1, 1:-1] = hm
    is_peak = np.ones((H, W), dtype=bool)
    for dr_ in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr_ == 0 and dc == 0:
                continue
            is_peak &= hm >= padded[1 + dr_:1 + dr_ + H, 1 + dc:1 + dc + W]

    rows, cols = np.where(is_peak & (hm >= score_thresh))
    if len(rows) > max_peaks:
        scores_all = hm[rows, cols]
        top = np.argsort(-scores_all)[:max_peaks]
        rows, cols = rows[top], cols[top]

    _, _, r0, dr, t0_rad, dtheta_rad = _polar_heatmap_grid_params()

    boxes = []
    for row, col in zip(rows, cols):
        dx, dy = offset_pred[row, col]
        r_c = r0 + (row + 0.5) * dr
        theta_c = t0_rad + (col + 0.5) * dtheta_rad
        x = r_c * math.cos(theta_c) + float(dx)
        y = r_c * math.sin(theta_c) + float(dy)
        z = float(z_pred[row, col, 0])
        l, w, h = np.exp(dim_pred[row, col])
        R = rotation3d.sixd_to_matrix_np(rot_pred[row, col])
        theta = float(np.arctan2(R[1, 0], R[0, 0]))
        boxes.append({"score": float(hm[row, col]), "x": float(x), "y": float(y), "z": z,
                       "l": float(l), "w": float(w), "h": float(h), "theta": theta, "R": R})
    return boxes


def decode_center_boxes(heatmap_pred, offset_pred, z_pred, dim_pred, rot_pred, score_thresh: float = 0.3,
                         max_peaks: int = 100, rotation_repr: str = "6d"):
    """heatmap_pred: (1,H,W) sigmoid 확률(이미 활성화됨). offset/z/dim_pred: (H,W,C).
    rot_pred: (H,W,C) - rotation_repr(rotation3d.ROTATION_CHANNELS)로 표현 선택, 기본
    "6d"(rotation3d.py). numpy 입력 - torch 텐서는 호출측에서 .cpu().numpy()로 변환해서 넘길 것.
    3x3 max-pool NMS(peak가 자기 3x3 이웃에서 최댓값인 위치만 후보)로 로컬 극대점만 취한다.
    반환: list of dict {score,x,y,z,l,w,h,theta,R}. theta는 R의 z-yaw 성분만 뽑은 근사값
    (기존 BEV NMS/footprint 계산과의 하위호환용) - R(3,3)이 진짜 3D 회전이고, 3D IoU
    평가는 R을 써야 정확하다."""
    hm = heatmap_pred[0]
    H, W = hm.shape
    padded = np.full((H + 2, W + 2), -1.0, dtype=hm.dtype)
    padded[1:-1, 1:-1] = hm
    is_peak = np.ones((H, W), dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            is_peak &= hm >= padded[1 + dr:1 + dr + H, 1 + dc:1 + dc + W]

    rows, cols = np.where(is_peak & (hm >= score_thresh))
    if len(rows) > max_peaks:
        scores_all = hm[rows, cols]
        top = np.argsort(-scores_all)[:max_peaks]
        rows, cols = rows[top], cols[top]

    sx, sy = config.ANCHOR_STRIDE
    x0, y0 = config.POINT_CLOUD_RANGE[0], config.POINT_CLOUD_RANGE[1]

    boxes = []
    for row, col in zip(rows, cols):
        dx, dy = offset_pred[row, col]
        gx_cell, gy_cell = col + dx, row + dy
        x = x0 + (gx_cell + 0.5) * sx
        y = y0 + (gy_cell + 0.5) * sy
        z = float(z_pred[row, col, 0])
        l, w, h = np.exp(dim_pred[row, col])
        R = rotation3d.decode_rotation_np(rot_pred[row, col], rotation_repr)
        theta = float(np.arctan2(R[1, 0], R[0, 0]))  # BEV NMS/footprint용 z-yaw 근사
        boxes.append({"score": float(hm[row, col]), "x": float(x), "y": float(y), "z": z,
                       "l": float(l), "w": float(w), "h": float(h), "theta": theta, "R": R})
    return boxes
