"""analyze_rotation_signal.py - BEV/voxel z-collapse가 pitch/roll 예측을 방해한다는
가설(reports_v2/foreground_aux_branch_proposal.md §1.4.2/1.4.3)을 검증하기 전에,
더 근본적인 질문부터 확인한다: **raw point cloud 자체에 pitch/roll 정보가 있긴 한가?**
sonar의 elevation angle ambiguity(§1.4.3-A, Westman&Kaess 등)가 지배적이면 아키텍처를
아무리 고쳐도 천장이 낮다 - 그럼 §1.4.3-B(dense tap point로 rotation head 이동)에
투자할 가치가 없다.

방법: GT box 안의 raw point(voxel_features[...,:3], 이미 world (x,y,z))에 PCA를 돌려
가장 큰 분산 축(사람 몸 길이 방향으로 기대)을 뽑고, GT rotation의 length axis(R[:,0],
eval_voxelnet.point_in_obb 관례와 동일 - local x가 length)와 비교한다. 수평(yaw, xy
평면 위 방향)과 수직(pitch류, 수평면에서 얼마나 기울었는지) 성분을 분리해서 각각
GT-PCA 정렬도를 따로 본다 - yaw는 잘 맞는데 pitch류만 안 맞으면 "point 자체에
tilt 정보가 없다"는 뜻이라 센서 한계(A)가 지배적이라는 강한 증거."""

import numpy as np

import rotation3d
from dataset import CachedVoxelNetDataset
from eval_voxelnet import point_in_obb


def pca_long_axis(points: np.ndarray) -> np.ndarray:
    """points: (N,3) -> (3,) 단위벡터, 최대 분산 축(부호는 임의 - 대칭 축이라 뒤에서
    양쪽 다 비교)."""
    centered = points - points.mean(axis=0, keepdims=True)
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)  # 오름차순
    return eigvecs[:, -1]  # 최대 고유값의 고유벡터


def axis_yaw_tilt(v: np.ndarray) -> tuple:
    """단위벡터 v(부호 무관하게 다룸 - 위쪽 반구로 정규화) -> (yaw_rad, tilt_rad).
    yaw: xy평면 투영의 방위각. tilt: 수평면에서 얼마나 기울었는지(0=완전 수평,
    90도=수직) - asin(|z 성분|)."""
    if v[2] < 0:
        v = -v
    yaw = float(np.arctan2(v[1], v[0]))
    tilt = float(np.arcsin(np.clip(abs(v[2]), 0, 1)))
    return yaw, tilt


def angle_diff(a: float, b: float) -> float:
    """두 각(rad) 사이 최소 차이, [0, pi/2] - 축(부호 없음) 비교라 180도 주기."""
    d = abs(a - b) % np.pi
    return min(d, np.pi - d)


def main():
    ds = CachedVoxelNetDataset("../cache/voxel", "test", head="center", load_gt_boxes=True)
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(ds), size=min(300, len(ds)), replace=False)

    yaw_errs, tilt_errs = [], []
    gt_tilts, pca_tilts, n_points_list = [], [], []
    skipped_few_points = 0

    for i in idxs:
        sample = ds[i]
        gt_boxes = sample["gt_boxes"]
        if len(gt_boxes) == 0:
            continue
        vf = sample["voxel_features"].numpy()
        num_points = sample["num_points"].numpy()
        T = vf.shape[1]
        valid = np.arange(T)[None, :] < num_points[:, None]
        all_pts = vf[:, :, :3][valid]  # (N,3) world xyz, 이 프레임의 전체 point

        for row in np.asarray(gt_boxes):
            center, dims = row[0:3], row[3:6]
            R_gt = rotation3d.sixd_to_matrix_np(row[7:13])
            mask = point_in_obb(all_pts, center, dims, R_gt)
            pts_in_box = all_pts[mask]
            if len(pts_in_box) < 5:
                skipped_few_points += 1
                continue

            pca_axis = pca_long_axis(pts_in_box)
            # GT의 "긴 축"은 항상 l(local x, R[:,0])이 아니다 - 실측 확인(2026-08-20,
            # 100박스): l이 최대인 경우는 17.8%뿐, h(local z)가 최대인 경우가 대부분
            # (다이버가 수직/대각으로 뻗어있는 경우가 많음). PCA(항상 진짜 최대분산 축을
            # 찾음)와 비교하려면 GT도 l/w/h 중 실제로 가장 큰 치수의 축을 써야 공정한 비교.
            longest_local_axis = int(np.argmax(dims))
            gt_axis = R_gt[:, longest_local_axis]

            pca_yaw, pca_tilt = axis_yaw_tilt(pca_axis)
            gt_yaw, gt_tilt = axis_yaw_tilt(gt_axis)

            yaw_errs.append(angle_diff(pca_yaw, gt_yaw))
            tilt_errs.append(abs(pca_tilt - gt_tilt))
            gt_tilts.append(gt_tilt)
            pca_tilts.append(pca_tilt)
            n_points_list.append(len(pts_in_box))

    yaw_errs, tilt_errs = np.array(yaw_errs), np.array(tilt_errs)
    gt_tilts, pca_tilts = np.array(gt_tilts), np.array(pca_tilts)
    n_points_arr = np.array(n_points_list)

    print(f"n_boxes_used={len(yaw_errs)}  skipped(<5 points)={skipped_few_points}")
    print(f"GT box 내 평균 point 수: {n_points_arr.mean():.1f} (median {np.median(n_points_arr):.0f})")
    print()
    print("--- yaw(수평 방위각) PCA vs GT 정렬 오차(rad, 0=완벽) ---")
    print(f"  mean={yaw_errs.mean():.3f}  median={np.median(yaw_errs):.3f}  "
          f"(참고: 무작위 축이면 기대값 ~{np.pi/4:.3f})")
    print("--- tilt(수평면에서 기운 정도) PCA vs GT 절대오차(rad, 0=완벽) ---")
    print(f"  mean={tilt_errs.mean():.3f}  median={np.median(tilt_errs):.3f}  "
          f"(참고: 무작위 축이면 기대값 ~{np.pi/4:.3f})")
    print()
    print(f"GT tilt 자체의 분포: mean={gt_tilts.mean():.3f} std={gt_tilts.std():.3f} "
          f"(0=완전 수평, {np.pi/2:.3f}=수직 - 다이버가 실제로 얼마나 기울어져 있는지)")

    corr = np.corrcoef(gt_tilts, pca_tilts)[0, 1] if gt_tilts.std() > 1e-6 else float("nan")
    print(f"GT tilt vs PCA tilt 상관계수: {corr:.3f} (핵심 지표 - 0에 가까우면 point 구조에 "
          f"tilt 정보가 없다는 뜻, 1에 가까우면 point에 신호가 있는데 network가 못 쓰고 있다는 뜻)")

    print()
    print("--- point 개수별 tilt 오차(포인트가 많을수록 PCA가 안정적이어야 함 - 그렇지 않으면 데이터 부족이 아니라 신호 자체가 없다는 뜻) ---")
    for lo, hi in [(5, 15), (15, 30), (30, 1000)]:
        m = (n_points_arr >= lo) & (n_points_arr < hi)
        if m.sum() < 5:
            continue
        print(f"  n_points in [{lo},{hi}): n={m.sum():4d}  tilt_err_mean={tilt_errs[m].mean():.3f}  "
              f"yaw_err_mean={yaw_errs[m].mean():.3f}")


if __name__ == "__main__":
    main()
