"""augment.py - VoxelNet 논문 §3.2의 3종 data augmentation.

`cache_dataset.py`가 train split에 한해 프레임당 증강 사본을 1개 미리 구워넣을 때 쓴다
(TriBand-BEV의 `export_yolo_dataset.py`와 동일 관례 - "train만, 프레임당 사본 1개로
train set을 1배 upsample". val/test는 절대 증강하지 않음 - 평가는 항상 원본 그대로).

목적은 정밀도 격차 분석(VoxelNet/reports/precision_gap_analysis.html §B2)에서 지목한
근본 원인 중 하나 - 우리 데이터는 10개 scene의 연속 프레임이라 다이버가 프레임 간
거의 안 움직이는데, augmentation 없이는 같은 anchor cell 몇 개가 수백 프레임에 걸쳐
반복 강화돼 "장소 암기" 위험이 크다. 매 프레임 GT 위치/스케일/회전을 흔들어서 이걸 깬다.

박스별 perturbation은 그 박스 안의 points만 같이 움직여야 하므로, Triband_BEV/baseline/
box3d.points_in_box_3d(프로젝트 전체가 이미 쓰는 라벨-포인트 대응 함수)를 그대로
재사용한다 - 여기서 새로 정의하지 않는다.
"""

import copy
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Triband_BEV" / "baseline"))
import box3d  # noqa: E402
import config  # noqa: E402

PER_BOX_ROT_RANGE = np.pi / 10  # ±18도, 논문 그대로
PER_BOX_TRANSLATE_STD = 1.0     # 미터, N(0, std^2), 논문 그대로
GLOBAL_SCALE_RANGE = (0.95, 1.05)
GLOBAL_ROT_RANGE = np.pi / 4    # ±45도

# --- strong-aug 추가분(2018 이후 표준, aug 트랙 실험용) ---
GLOBAL_TRANSLATE_STD = (0.2, 0.2, 0.1)  # (x,y,z) 미터, PointPillars/CenterPoint식 전역 이동
GT_SAMPLE_TARGET = 4        # gt-sampling 후 프레임당 목표 다이버 수(현재 수가 이보다 적으면 채움)
GT_SAMPLE_MAX_TRY = 20      # 배치 시도 상한(충돌/FOV 실패 재시도)


def matrix_to_euler_zyx(R: np.ndarray):
    """(3,3) -> (rx,ry,rz) 도. euler_to_matrix/box3d.rotation_matrix의 R=Rz·Ry·Rx 규약 역변환.
    ry=±90°(gimbal lock) 근처에선 근사(드묾) - flip 증강 정도엔 허용 오차."""
    sy = -R[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    ry = np.arcsin(sy)
    cy = np.cos(ry)
    if abs(cy) > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2])
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        rx = np.arctan2(-R[1, 2], R[1, 1])
        rz = 0.0
    return tuple(np.degrees([rx, ry, rz]))


def _box_footprint_xy(obj: dict) -> np.ndarray:
    return box3d.corners_3d(obj, use_xy_rotation=False)[:4, :2]


def _boxes_collide(obj_a: dict, obj_b: dict) -> bool:
    pa, pb = Polygon(_box_footprint_xy(obj_a)), Polygon(_box_footprint_xy(obj_b))
    if not pa.is_valid or not pb.is_valid:
        return False
    return pa.intersects(pb)


def per_box_perturb(points: np.ndarray, objects: list, rng: np.random.Generator):
    """각 GT 박스를 독립적으로 회전+이동시키고, 박스 안 points도 같이 옮긴다. perturbation
    후 다른 박스와 충돌하면 그 박스만 원상복구(논문의 collision test)."""
    if not objects:
        return points, objects

    points = points.copy()
    new_objects = [copy.deepcopy(o) for o in objects]
    masks = [box3d.points_in_box_3d(points[:, :3], o, use_xy_rotation=False) for o in objects]

    for i, obj in enumerate(new_objects):
        dtheta_deg = float(np.degrees(rng.uniform(-PER_BOX_ROT_RANGE, PER_BOX_ROT_RANGE)))
        dxyz = rng.normal(0.0, PER_BOX_TRANSLATE_STD, size=3)

        c = obj["centroid"]
        center = np.array([c["x"], c["y"], c["z"]])
        mask = masks[i]
        local = points[mask, :3] - center

        rad = np.radians(dtheta_deg)
        cr, sr = np.cos(rad), np.sin(rad)
        Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])  # box3d._rot_z와 동일 관례
        new_xyz = local @ Rz.T + center + dxyz

        trial = copy.deepcopy(obj)
        trial["centroid"] = {"x": c["x"] + dxyz[0], "y": c["y"] + dxyz[1], "z": c["z"] + dxyz[2]}
        trial["rotation_z"] = obj.get("rotation_z", 0.0) + dtheta_deg

        if any(_boxes_collide(trial, new_objects[j]) for j in range(len(new_objects)) if j != i):
            continue  # 충돌 - 이 박스는 원상복구, points도 그대로 둠

        points[mask, :3] = new_xyz
        new_objects[i] = trial

    return points, new_objects


def global_scale(points: np.ndarray, objects: list, rng: np.random.Generator):
    s = rng.uniform(*GLOBAL_SCALE_RANGE)
    points = points.copy()
    points[:, :3] *= s
    new_objects = []
    for o in objects:
        o2 = copy.deepcopy(o)
        o2["centroid"] = {k: v * s for k, v in o["centroid"].items()}
        o2["dimensions"] = {k: v * s for k, v in o["dimensions"].items()}
        new_objects.append(o2)
    return points, new_objects


def global_rotate(points: np.ndarray, objects: list, rng: np.random.Generator):
    phi = rng.uniform(-GLOBAL_ROT_RANGE, GLOBAL_ROT_RANGE)
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s], [s, c]])
    points = points.copy()
    points[:, :2] = points[:, :2] @ R.T
    new_objects = []
    for o in objects:
        o2 = copy.deepcopy(o)
        cx, cy = o["centroid"]["x"], o["centroid"]["y"]
        nx, ny = R @ np.array([cx, cy])
        o2["centroid"] = {**o["centroid"], "x": float(nx), "y": float(ny)}
        o2["rotation_z"] = o.get("rotation_z", 0.0) + float(np.degrees(phi))
        new_objects.append(o2)
    return points, new_objects


def global_translate(points: np.ndarray, objects: list, rng: np.random.Generator):
    """전역 이동(PointPillars/CenterPoint). points+centroid에 N(0,std) 더함. 회전/치수 불변."""
    dxyz = rng.normal(0.0, np.array(GLOBAL_TRANSLATE_STD))
    points = points.copy()
    points[:, :3] += dxyz
    new_objects = []
    for o in objects:
        o2 = copy.deepcopy(o)
        c = o["centroid"]
        o2["centroid"] = {"x": c["x"] + dxyz[0], "y": c["y"] + dxyz[1], "z": c["z"] + dxyz[2]}
        new_objects.append(o2)
    return points, new_objects


def global_flip_y(points: np.ndarray, objects: list, rng: np.random.Generator):
    """y축 반사(y->-y). 소나 FOV(±45° azimuth)가 x축 대칭이라 물리적으로 유효.
    전체 3D 회전은 반사 S=diag(1,-1,1)로 R'=S·R·S 후 euler 역변환(matrix_to_euler_zyx)."""
    if rng.random() < 0.5:
        return points, objects  # 50%만 flip
    S = np.diag([1.0, -1.0, 1.0])
    points = points.copy()
    points[:, 1] *= -1.0
    new_objects = []
    for o in objects:
        o2 = copy.deepcopy(o)
        c = o["centroid"]
        o2["centroid"] = {**c, "y": -c["y"]}
        R = box3d.rotation_matrix(o.get("rotation_x", 0.0), o.get("rotation_y", 0.0), o.get("rotation_z", 0.0))
        rx, ry, rz = matrix_to_euler_zyx(S @ R @ S)
        o2["rotation_x"], o2["rotation_y"], o2["rotation_z"] = float(rx), float(ry), float(rz)
        new_objects.append(o2)
    return points, new_objects


def build_gt_database(frames, max_points_per_obj: int = 4000):
    """gt-sampling용 DB. frames: iterable of (points(N,>=4), objects). sonar 박스별로
    박스 안 points(centroid 상대, world축)를 추출해 저장. 반환: list of dict{obj, points_rel}."""
    db = []
    for points, objects in frames:
        for o in objects:
            if not str(o.get("class", "")).endswith("-sonar"):
                continue
            mask = box3d.points_in_box_3d(points[:, :3], o, use_xy_rotation=False)
            pts = points[mask]
            if len(pts) < 5:
                continue
            if len(pts) > max_points_per_obj:
                pts = pts[np.random.default_rng(0).choice(len(pts), max_points_per_obj, replace=False)]
            c = o["centroid"]
            rel = pts.copy()
            rel[:, 0] -= c["x"]; rel[:, 1] -= c["y"]; rel[:, 2] -= c["z"]
            # 원본 캡처 시 센서→다이버 bearing(azimuth, 도) 저장 - gt_sample에서 새 위치로
            # 옮길 때 이 bearing 변화만큼 점군을 회전해 forward-looking 소나의 시선각(aspect)
            # 정합성을 유지하기 위함(2026-08-28, single-aspect scan 이슈).
            bearing = float(np.degrees(np.arctan2(c["y"], c["x"])))
            db.append({"obj": copy.deepcopy(o), "z": c["z"], "bearing": bearing,
                       "points_rel": rel.astype(np.float32)})
    return db


GT_SAMPLE_YAW_JITTER = 7.5   # 배치 후 aspect-보존 회전 위에 얹는 소폭 yaw 흔들림(±도)


def gt_sample(points: np.ndarray, objects: list, db: list, rng: np.random.Generator,
              target: int = GT_SAMPLE_TARGET):
    """DB에서 다이버를 뽑아 빈 공간(FOV±45°·range·충돌 제약)에 붙여넣어 프레임당 물체 수를
    target까지 채운다.

    forward-looking 소나는 single-aspect(한쪽 면만 스캔)라, 점군을 임의 yaw로 강체 회전하면
    라벨 방향과 실제 점 패턴(시선각)이 어긋난 비물리적 샘플이 된다. 그래서 배치할 새 위치의
    bearing과 DB 원본 캡처 bearing의 차이(Δbearing)만큼만 회전해 센서 기준 상대 시선각을
    보존하고, 그 위에 ±GT_SAMPLE_YAW_JITTER의 소폭 흔들림만 더한다(2026-08-28)."""
    if not db or len(objects) >= target:
        return points, objects
    pcr = config.POINT_CLOUD_RANGE
    fov = config.SONAR_AZIMUTH_LIMIT_DEG
    points = points.copy()
    new_objects = [copy.deepcopy(o) for o in objects]
    add_chunks = []
    tries = 0
    while len(new_objects) < target and tries < GT_SAMPLE_MAX_TRY:
        tries += 1
        e = db[rng.integers(len(db))]
        r = rng.uniform(1.0, min(10.5, pcr[3] - 0.5))
        az_deg = rng.uniform(-fov + 3, fov - 3)
        az = np.radians(az_deg)
        tx, ty = r * np.cos(az), r * np.sin(az)
        if not (pcr[0] <= tx <= pcr[3] and pcr[1] <= ty <= pcr[4]):
            continue
        tz = float(np.clip(e["z"], pcr[2] + 0.3, pcr[5] - 0.3))
        # aspect 보존: 새 bearing - 원본 bearing 만큼 회전 + 소폭 jitter
        dyaw = (az_deg - e.get("bearing", az_deg)) + rng.uniform(-GT_SAMPLE_YAW_JITTER, GT_SAMPLE_YAW_JITTER)
        cz, sz = np.cos(np.radians(dyaw)), np.sin(np.radians(dyaw))
        trial = copy.deepcopy(e["obj"])
        trial["centroid"] = {"x": float(tx), "y": float(ty), "z": tz}
        trial["rotation_z"] = e["obj"].get("rotation_z", 0.0) + dyaw
        trial["link_id"] = -(len(add_chunks) + 1)  # 음수 id로 원본과 구분
        if any(_boxes_collide(trial, o) for o in new_objects):
            continue
        rel = e["points_rel"]
        rx = cz * rel[:, 0] - sz * rel[:, 1] + tx
        ry = sz * rel[:, 0] + cz * rel[:, 1] + ty
        chunk = rel.copy()
        chunk[:, 0], chunk[:, 1], chunk[:, 2] = rx, ry, rel[:, 2] + tz
        add_chunks.append(chunk)
        new_objects.append(trial)
    if add_chunks:
        points = np.concatenate([points] + add_chunks, axis=0)
    return points, new_objects


def augment_frame(points: np.ndarray, objects: list, seed=None, gt_db: list = None,
                  strong: bool = False):
    """기본(strong=False): 논문 §3.2 순서 per-box perturb -> global scale -> global rotate
    (기존 캐시 파이프라인과 100% 동일, 하위호환).
    strong=True: gt-sampling(gt_db 있으면) -> per-box -> flip -> global translate -> scale -> rotate."""
    rng = np.random.default_rng(seed)
    if not strong:
        points, objects = per_box_perturb(points, objects, rng)
        points, objects = global_scale(points, objects, rng)
        points, objects = global_rotate(points, objects, rng)
        return points, objects
    if gt_db:
        points, objects = gt_sample(points, objects, gt_db, rng)
    points, objects = per_box_perturb(points, objects, rng)
    points, objects = global_flip_y(points, objects, rng)
    points, objects = global_translate(points, objects, rng)
    points, objects = global_scale(points, objects, rng)
    points, objects = global_rotate(points, objects, rng)
    return points, objects
