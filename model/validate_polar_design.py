"""validate_polar_design.py - voxel polarization Phase 1(Cylinder3D식 cylindrical
partitioning) 설계를 실제 voxelize.py/heatmap_targets.py를 고치기 전에 정량
검증한다 - rotation3d 확장 때 validate_rotation3d.py가 했던 역할과 같다.

item1: r/theta 범위가 센서 물리 한계(SONAR_AZIMUTH_LIMIT_DEG=45.0,
       stamp_core.py)와 실측 GT 분포를 얼마나 커버하는지
item2: non-empty voxel 비율, Cartesian(현재) vs polar(후보), range bucket별 -
       Cylinder3D(Zhu et al. CVPR2021, arXiv:2011.10033) Sec3.2/Fig1(a)/Fig3
       방법론 재현("89% vs 61%", 원거리에서 cubic이 ~6배 더 희소)
item3: heatmap gaussian radius를 PolarStream(Sun et al. NeurIPS2021,
       arXiv:2106.07545) Sec3.3 방식대로(length/width 대신 range/azimuth span)
       계산했을 때, range bucket별로 병적인 경향(원거리 전부 tau floor에 눌어붙는지 등)이
       있는지 확인

후보 polar grid 설정(POLAR_*)은 검증용 - 최종 확정 값 아님, 여기서 나온 결과로 조정한다.

Usage:
    python validate_polar_design.py [--max-frames 200]
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Triband_BEV" / "baseline"))
import common  # noqa: E402
import filter_outliers  # noqa: E402
import stamp_core  # noqa: E402  (경로는 common.py가 이미 sys.path에 넣어둠)

import config  # noqa: E402
from anchors import rotated_rect_corners  # noqa: E402
from heatmap_targets import gaussian_radius  # noqa: E402

# --- polar grid 설정 - config.py의 실제 값을 그대로 씀(드리프트 방지) ---
POLAR_R_RANGE = config.POLAR_R_RANGE
POLAR_THETA_RANGE_DEG = config.POLAR_THETA_RANGE_DEG
POLAR_R_BINS = config.POLAR_R_BINS
POLAR_THETA_BINS = config.POLAR_THETA_BINS
Z_BINS = config.GRID_SIZE[2]  # z축은 그대로(10) - cylindrical: z는 안 바꿈

BUCKETS = [("0-2m", 0, 2), ("2-2.5m", 2, 2.5), ("2.5-3m", 2.5, 3), ("3-3.5m", 3, 3.5),
           ("3.5-5m", 3.5, 5), ("5m+", 5, float("inf"))]


def bucket_of(r: float) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= r < hi:
            return name
    return BUCKETS[-1][0]


def load_points(scene_id: str, frame_idx: int) -> np.ndarray:
    raw = np.fromfile(common.sonar_bin_path(scene_id, frame_idx), dtype=np.float32)
    points = raw.reshape(-1, 4 if raw.size % 4 == 0 else 3)
    if points.shape[1] == 3:
        points = np.concatenate([points, np.zeros((len(points), 1), dtype=np.float32)], axis=1)
    return points[~np.isnan(points).any(axis=1)]


def sample_frames(max_frames: int):
    flagged = filter_outliers.load_flagged_keys()
    scenes = common.list_scenes(include_excluded=False)
    out = []
    for scene_id in scenes:
        per_scene = 0
        for frame_idx, objects in filter_outliers.iter_filtered_frames(scene_id, flagged):
            if per_scene >= max(1, max_frames // len(scenes)):
                break
            out.append((scene_id, frame_idx, objects))
            per_scene += 1
            if len(out) >= max_frames:
                return out
    return out


# ---------------- item 1 ----------------
def check_item1():
    flagged = filter_outliers.load_flagged_keys()
    scenes = common.list_scenes(include_excluded=False)
    rs, thetas = [], []
    for scene_id in scenes:
        for _, objects in filter_outliers.iter_filtered_frames(scene_id, flagged):
            for o in objects:
                if not o.get("class", "").endswith("-sonar"):
                    continue
                c = o["centroid"]
                rs.append(math.hypot(c["x"], c["y"]))
                thetas.append(math.degrees(math.atan2(c["y"], c["x"])))
    rs, thetas = np.array(rs), np.array(thetas)
    az_limit = stamp_core.SONAR_AZIMUTH_LIMIT_DEG
    print("=== item1: r/theta 커버리지 ===")
    print(f"n={len(rs)}, r=[{rs.min():.2f},{rs.max():.2f}] p99={np.percentile(rs,99):.2f}")
    print(f"theta(deg)=[{thetas.min():.2f},{thetas.max():.2f}] "
          f"(센서 물리 한계 +-{az_limit}deg 대비 커버리지={np.mean(np.abs(thetas)<=az_limit):.1%})")
    print(f"후보 POLAR_R_RANGE={POLAR_R_RANGE} 커버리지="
          f"{np.mean((rs>=POLAR_R_RANGE[0])&(rs<POLAR_R_RANGE[1])):.1%}")
    print(f"후보 POLAR_THETA_RANGE_DEG={POLAR_THETA_RANGE_DEG} 커버리지="
          f"{np.mean((thetas>=POLAR_THETA_RANGE_DEG[0])&(thetas<POLAR_THETA_RANGE_DEG[1])):.1%}")


# ---------------- item 2 ----------------
def cartesian_total_cells_by_bucket():
    pc_range = np.array(config.POINT_CLOUD_RANGE, dtype=np.float64)
    vsize = np.array(config.VOXEL_SIZE, dtype=np.float64)
    Wx, Hy, Dz = config.GRID_SIZE
    ix, iy = np.meshgrid(np.arange(Wx), np.arange(Hy), indexing="ij")
    xc = pc_range[0] + (ix.ravel() + 0.5) * vsize[0]
    yc = pc_range[1] + (iy.ravel() + 0.5) * vsize[1]
    r = np.hypot(xc, yc)
    totals = {b[0]: 0 for b in BUCKETS}
    for rr in r:
        totals[bucket_of(rr)] += Dz
    return totals


def polar_total_cells_by_bucket():
    r0, r1 = POLAR_R_RANGE
    dr = (r1 - r0) / POLAR_R_BINS
    totals = {b[0]: 0 for b in BUCKETS}
    for ri in range(POLAR_R_BINS):
        rc = r0 + (ri + 0.5) * dr
        totals[bucket_of(rc)] += POLAR_THETA_BINS * Z_BINS
    return totals


def cartesian_occupied_by_bucket(points: np.ndarray):
    pc_range = np.array(config.POINT_CLOUD_RANGE, dtype=np.float32)
    vsize = np.array(config.VOXEL_SIZE, dtype=np.float32)
    grid = np.array(config.GRID_SIZE)
    xyz = points[:, :3]
    in_range = np.all((xyz >= pc_range[:3]) & (xyz < pc_range[3:]), axis=1)
    xyz = xyz[in_range]
    if len(xyz) == 0:
        return {b[0]: 0 for b in BUCKETS}
    idx = np.clip(np.floor((xyz - pc_range[:3]) / vsize).astype(np.int64), 0, grid - 1)
    occ = np.unique(idx, axis=0)
    xc = pc_range[0] + (occ[:, 0] + 0.5) * vsize[0]
    yc = pc_range[1] + (occ[:, 1] + 0.5) * vsize[1]
    r = np.hypot(xc, yc)
    out = {b[0]: 0 for b in BUCKETS}
    for rr in r:
        out[bucket_of(float(rr))] += 1
    return out


def polar_occupied_by_bucket(points: np.ndarray):
    x, y, z = points[:, 0].astype(np.float64), points[:, 1].astype(np.float64), points[:, 2].astype(np.float64)
    r = np.hypot(x, y)
    theta = np.degrees(np.arctan2(y, x))
    r0, r1 = POLAR_R_RANGE
    t0, t1 = POLAR_THETA_RANGE_DEG
    z0, z1 = config.POINT_CLOUD_RANGE[2], config.POINT_CLOUD_RANGE[5]
    dr, dtheta, dz = (r1 - r0) / POLAR_R_BINS, (t1 - t0) / POLAR_THETA_BINS, (z1 - z0) / Z_BINS
    keep = (r >= r0) & (r < r1) & (theta >= t0) & (theta < t1) & (z >= z0) & (z < z1)
    r, theta, z = r[keep], theta[keep], z[keep]
    if len(r) == 0:
        return {b[0]: 0 for b in BUCKETS}
    r_idx = np.clip(np.floor((r - r0) / dr).astype(np.int64), 0, POLAR_R_BINS - 1)
    t_idx = np.clip(np.floor((theta - t0) / dtheta).astype(np.int64), 0, POLAR_THETA_BINS - 1)
    z_idx = np.clip(np.floor((z - z0) / dz).astype(np.int64), 0, Z_BINS - 1)
    occ = np.unique(np.stack([r_idx, t_idx, z_idx], axis=1), axis=0)
    rc = r0 + (occ[:, 0] + 0.5) * dr
    out = {b[0]: 0 for b in BUCKETS}
    for rr in rc:
        out[bucket_of(float(rr))] += 1
    return out


def check_item2(frames, cart_totals, polar_totals):
    cart_occ_sum = {b[0]: 0 for b in BUCKETS}
    polar_occ_sum = {b[0]: 0 for b in BUCKETS}
    n = len(frames)
    for scene_id, frame_idx, _ in frames:
        points = load_points(scene_id, frame_idx)
        c = cartesian_occupied_by_bucket(points)
        p = polar_occupied_by_bucket(points)
        for k in cart_occ_sum:
            cart_occ_sum[k] += c[k]
            polar_occ_sum[k] += p[k]

    print(f"\n=== item2: non-empty voxel 비율 (Cartesian vs Polar, n_frames={n}) ===")
    print("Cylinder3D Sec3.2/Fig1(a) 재현 - cylindrical이 원거리에서 더 균형잡혀야 함")
    print(f"{'bucket':>8s} {'cart(total)':>12s} {'cart occ%':>10s} {'polar(total)':>13s} {'polar occ%':>11s}")
    for name, _, _ in BUCKETS:
        ct, pt = cart_totals[name] * n, polar_totals[name] * n
        c_pct = 100 * cart_occ_sum[name] / max(ct, 1)
        p_pct = 100 * polar_occ_sum[name] / max(pt, 1)
        print(f"{name:>8s} {cart_totals[name]:>12d} {c_pct:>9.2f}% {polar_totals[name]:>13d} {p_pct:>10.2f}%")


# ---------------- item 3 ----------------
def check_item3(frames):
    """주의: heatmap은 raw voxel grid가 아니라 그보다 2배 성긴 anchor/heatmap grid
    해상도(config.ANCHOR_STRIDE=0.2m, RPN 다운샘플 배수)에서 만들어진다 - 그래서
    polar 쪽도 item2(raw voxel 해상도)가 아니라 그 절반 해상도(POLAR_*_BINS//2)로
    맞춰야 공정한 비교가 된다(처음 버전은 이걸 안 맞춰서 polar가 인위적으로 더 커
    보였음 - item3 실행 로그 참고)."""
    sx, sy = config.ANCHOR_STRIDE  # 현재(Cartesian) 방식 - 비교 기준, 0.2m/cell
    r0, r1 = POLAR_R_RANGE
    t0, t1 = POLAR_THETA_RANGE_DEG
    dr = (r1 - r0) / config.POLAR_HEATMAP_R_BINS  # heatmap 해상도(voxel 해상도의 절반, 0.2m)
    dtheta_rad = math.radians((t1 - t0) / config.POLAR_HEATMAP_THETA_BINS)  # 2도/cell

    by_bucket_cart = {b[0]: [] for b in BUCKETS}
    by_bucket_polar = {b[0]: [] for b in BUCKETS}

    for _, _, objects in frames:
        for o in objects:
            if not o.get("class", "").endswith("-sonar"):
                continue
            c, d = o["centroid"], o["dimensions"]
            rng = math.hypot(c["x"], c["y"])
            bucket = bucket_of(rng)

            l_cells, w_cells = d["length"] / sx, d["width"] / sy
            cart_radius = gaussian_radius(w_cells, l_cells)
            by_bucket_cart[bucket].append(cart_radius)

            theta_z = o.get("rotation_z", 0.0)
            corners = rotated_rect_corners(c["x"], c["y"], d["length"], d["width"], math.radians(theta_z))
            corner_r = np.hypot(corners[:, 0], corners[:, 1])
            corner_theta = np.arctan2(corners[:, 1], corners[:, 0])
            # theta는 wrap-around 없음(+-45도 범위 내부라 안전) - 그냥 max-min
            dr_span = float(corner_r.max() - corner_r.min())
            dtheta_span = float(corner_theta.max() - corner_theta.min())
            r_cells = dr_span / dr
            theta_cells = dtheta_span / dtheta_rad
            polar_radius = gaussian_radius(theta_cells, r_cells)
            by_bucket_polar[bucket].append(polar_radius)

    print("\n=== item3: heatmap gaussian radius, range bucket별 (PolarStream Sec3.3 방식) ===")
    print(f"{'bucket':>8s} {'n':>5s} {'cart median':>12s} {'polar median':>13s} {'polar==tau(2.0) 비율':>20s}")
    for name, _, _ in BUCKETS:
        cart_r = np.array(by_bucket_cart[name])
        pol_r = np.array(by_bucket_polar[name])
        if len(cart_r) == 0:
            print(f"{name:>8s} {0:>5d}")
            continue
        floor_frac = np.mean(pol_r <= 2.0 + 1e-6)
        print(f"{name:>8s} {len(cart_r):>5d} {np.median(cart_r):>12.2f} {np.median(pol_r):>13.2f} "
              f"{floor_frac:>19.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=200)
    args = parser.parse_args()

    check_item1()

    frames = sample_frames(args.max_frames)
    print(f"\n(item2/3용 샘플 프레임 {len(frames)}개)")
    cart_totals = cartesian_total_cells_by_bucket()
    polar_totals = polar_total_cells_by_bucket()
    check_item2(frames, cart_totals, polar_totals)
    check_item3(frames)


if __name__ == "__main__":
    main()
