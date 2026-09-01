"""voxelize.py - point cloud(N,4)[x,y,z,intensity] -> hard voxel 표현.

논문 §2.3(Efficient Implementation)의 절차 그대로: 포인트를 셔플 -> voxel별로 최대
T개까지만 유지(random sampling) -> voxel별 (x,y,z) 평균으로 중심 오프셋 augment.
`np.unique(..., axis=0)`로 voxel 좌표를 묶는 것이 논문의 O(1) 해시테이블 lookup과
결과적으로 동일한 voxel 그룹을 만들어낸다(구현 방식만 다름, 프레임당 포인트가
~6-9k라 벡터화 unique면 충분히 빠름 - CUDA 해시테이블이 필요 없음).
"""

import numpy as np


def voxelize(points: np.ndarray, pc_range, voxel_size, max_points_per_voxel: int, max_voxels: int):
    """points: (N,4) float32 [x,y,z,intensity].

    반환:
      voxel_features: (K,T,4) float32, 빈 슬롯은 0
      voxel_coords:   (K,3) int64 [z_idx,y_idx,x_idx] (D,H,W 순서 - conv3d 텐서 축과 일치)
      num_points:     (K,) int64, voxel별 실제 포인트 수(T 이하)
    K는 실제 non-empty voxel 수 (<=max_voxels)이며 배치마다 다르다.
    """
    pc_range = np.asarray(pc_range, dtype=np.float32)
    voxel_size = np.asarray(voxel_size, dtype=np.float32)
    grid_size = np.round((pc_range[3:6] - pc_range[0:3]) / voxel_size).astype(np.int64)

    xyz = points[:, :3]
    in_range = np.all((xyz >= pc_range[0:3]) & (xyz < pc_range[3:6]), axis=1)
    points = points[in_range]
    if len(points) == 0:
        return (np.zeros((0, max_points_per_voxel, 4), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int64), np.zeros((0,), dtype=np.int64))

    perm = np.random.permutation(len(points))  # random sampling 순서를 위한 사전 셔플
    points = points[perm]

    idx_xyz = np.floor((points[:, :3] - pc_range[0:3]) / voxel_size).astype(np.int64)
    idx_xyz = np.clip(idx_xyz, 0, grid_size - 1)
    idx_zyx = idx_xyz[:, [2, 1, 0]]  # (z,y,x) - 이후 D,H,W 텐서 인덱싱과 순서를 맞춤

    voxel_coords, inverse, counts = np.unique(idx_zyx, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    if len(voxel_coords) > max_voxels:
        keep_voxel = np.zeros(len(voxel_coords), dtype=bool)
        keep_voxel[:max_voxels] = True
        keep_point = keep_voxel[inverse]
        points, inverse = points[keep_point], inverse[keep_point]
        voxel_coords, counts = voxel_coords[:max_voxels], counts[:max_voxels]

    K = len(voxel_coords)
    num_points = np.minimum(counts, max_points_per_voxel).astype(np.int64)

    # voxel 내 몇 번째 슬롯인지: 같은 voxel(inverse값)로 처음 등장한 순서를 센다.
    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    slot_in_voxel = np.arange(len(order)) - np.searchsorted(sorted_inverse, sorted_inverse, side="left")
    slot = np.empty(len(order), dtype=np.int64)
    slot[order] = slot_in_voxel

    voxel_features = np.zeros((K, max_points_per_voxel, 4), dtype=np.float32)
    valid = slot < max_points_per_voxel
    voxel_features[inverse[valid], slot[valid]] = points[valid]

    return voxel_features, voxel_coords, num_points


def voxelize_polar(points: np.ndarray, r_range, theta_range_deg, z_range, r_bins: int, theta_bins: int,
                    z_bins: int, max_points_per_voxel: int, max_voxels: int):
    """voxelize()의 극좌표(Cylinder3D식) 버전 - BEV 평면(x,y)만 r=hypot(x,y),
    theta=atan2(y,x)(도) 기준으로 binning, z축은 그대로 Cartesian(voxelize()와 동일 로직).
    반환 계약은 voxelize()와 완전히 동일: voxel_coords는 (K,3) [z_idx,r_idx,theta_idx]
    (H=r, W=theta 관례 - config.POLAR_GRID_SIZE 주석 참고) - model.py의 dense tensor
    조립 코드는 coords를 축 의미와 무관하게 인덱스로만 쓰므로 안 건드려도 된다."""
    z0, z1 = z_range
    r0, r1 = r_range
    t0, t1 = theta_range_deg
    dr, dtheta, dz = (r1 - r0) / r_bins, (t1 - t0) / theta_bins, (z1 - z0) / z_bins

    xyz = points[:, :3]
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    theta = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    z = xyz[:, 2]
    in_range = (r >= r0) & (r < r1) & (theta >= t0) & (theta < t1) & (z >= z0) & (z < z1)
    points = points[in_range]
    if len(points) == 0:
        return (np.zeros((0, max_points_per_voxel, 4), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int64), np.zeros((0,), dtype=np.int64))
    r, theta, z = r[in_range], theta[in_range], z[in_range]

    perm = np.random.permutation(len(points))
    points, r, theta, z = points[perm], r[perm], theta[perm], z[perm]

    r_idx = np.clip(np.floor((r - r0) / dr).astype(np.int64), 0, r_bins - 1)
    t_idx = np.clip(np.floor((theta - t0) / dtheta).astype(np.int64), 0, theta_bins - 1)
    z_idx = np.clip(np.floor((z - z0) / dz).astype(np.int64), 0, z_bins - 1)
    idx_zrt = np.stack([z_idx, r_idx, t_idx], axis=1)  # [z,r,theta] = [D,H,W] 순서

    voxel_coords, inverse, counts = np.unique(idx_zrt, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    if len(voxel_coords) > max_voxels:
        keep_voxel = np.zeros(len(voxel_coords), dtype=bool)
        keep_voxel[:max_voxels] = True
        keep_point = keep_voxel[inverse]
        points, inverse = points[keep_point], inverse[keep_point]
        voxel_coords, counts = voxel_coords[:max_voxels], counts[:max_voxels]

    K = len(voxel_coords)
    num_points = np.minimum(counts, max_points_per_voxel).astype(np.int64)

    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    slot_in_voxel = np.arange(len(order)) - np.searchsorted(sorted_inverse, sorted_inverse, side="left")
    slot = np.empty(len(order), dtype=np.int64)
    slot[order] = slot_in_voxel

    voxel_features = np.zeros((K, max_points_per_voxel, 4), dtype=np.float32)
    valid = slot < max_points_per_voxel
    voxel_features[inverse[valid], slot[valid]] = points[valid]

    return voxel_features, voxel_coords, num_points


def augment_with_centroid_offset(voxel_features: np.ndarray, num_points: np.ndarray) -> np.ndarray:
    """(K,T,4) -> (K,T,7): [x,y,z,intensity, x-vx,y-vy,z-vz]. vx,vy,vz는 voxel 내
    유효 포인트의 (x,y,z) 평균(논문 §2.1.1 "relative offset w.r.t. the centroid")."""
    K, T, _ = voxel_features.shape
    mask = np.arange(T)[None, :] < num_points[:, None]  # (K,T)
    xyz = voxel_features[:, :, :3]
    denom = np.maximum(num_points, 1).astype(np.float32)[:, None]
    centroid = (xyz * mask[:, :, None]).sum(axis=1) / denom  # (K,3)
    offset = (xyz - centroid[:, None, :]) * mask[:, :, None]
    out = np.concatenate([voxel_features, offset], axis=2)  # (K,T,7)
    out *= mask[:, :, None]  # 패딩 슬롯은 그대로 0 유지
    return out.astype(np.float32)
