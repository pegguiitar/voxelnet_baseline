"""rotation3d.py - 3D 회전 표현 변환. anchor/center head가 이제 z축 회전(sin/cos)
대신 완전한 3D 회전(6D continuous representation)을 회귀한다 - x,y 회전을 무시하면
82.5%의 박스에서 점 소속 판정이 틀리고 높이 오차 중앙값 0.35m라는 게 실측으로 확인됨
(project_rotation_xy_full_coverage_findings 참고).

Euler 각도나 quaternion을 직접 회귀하지 않는 이유: Euler는 gimbal lock(rotation_y=90도
근처에서 rotation_x/z가 서로 얽혀 구분 불가 - 우리 실제 라벨에서 발생 확인됨)과 wraparound
불연속이 있고, quaternion은 q=-q 이중 표현 문제가 있다. 대신 Zhou et al.,
"On the Continuity of Rotation Representations in Neural Networks" (CVPR 2019)의 6D
연속 표현을 쓴다 - 회전행렬 R의 앞 두 열(6개 숫자)만 회귀하고, Gram-Schmidt 직교화로
나머지 한 축을 복원해 완전한 회전행렬을 만든다. 불연속점이 전혀 없어 회귀 대상으로
안정적이다.

축 관례: local x=length, local y=width, local z=height (Triband_BEV/baseline/box3d.py의
labelCloud 관례와 동일 - world_col = R @ local_col).
"""

import numpy as np
import torch


def euler_to_matrix(rotation_x_deg, rotation_y_deg, rotation_z_deg) -> np.ndarray:
    """(3,3) - box3d.py의 rotation_matrix()와 동일 관례(Rz . Ry . Rx, world_col = R @ local_col).
    라벨링 툴이 만든 rotation_x/y/z(도 단위)를 캐시 생성 시 회전행렬로 바꾸는 데 쓴다."""
    rx, ry, rz = np.radians([rotation_x_deg, rotation_y_deg, rotation_z_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def matrix_to_6d(R: np.ndarray) -> np.ndarray:
    """(3,3) -> (6,). R의 앞 두 열(첫 번째=local x축, 두 번째=local y축)을 이어붙임 -
    캐시 생성 시 학습 타겟으로 저장할 값."""
    return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)


def sixd_to_matrix_np(ortho6d: np.ndarray) -> np.ndarray:
    """(6,) -> (3,3). Gram-Schmidt로 정규직교 회전행렬 복원(numpy, 디코드/검증용)."""
    a1, a2 = ortho6d[0:3], ortho6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2_proj = a2 - np.dot(b1, a2) * b1
    b2 = a2_proj / (np.linalg.norm(a2_proj) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)  # 열이 각 축


def sixd_to_matrix_torch(ortho6d: torch.Tensor) -> torch.Tensor:
    """(...,6) -> (...,3,3). Gram-Schmidt, 배치 텐서 버전(학습/디코드 경로용).
    마지막 차원이 6인 임의 shape을 지원."""
    a1 = ortho6d[..., 0:3]
    a2 = ortho6d[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=1e-8)
    a2_proj = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(a2_proj, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # (...,3,3), 열이 각 축


def matrix_to_6d_torch(R: torch.Tensor) -> torch.Tensor:
    """(...,3,3) -> (...,6). matrix_to_6d(numpy)의 배치 텐서 버전 - R의 앞 두 열을
    이어붙임(stage2_refine.py의 jitter_gt_boxes처럼 회전행렬을 합성한 뒤 다시 6D로
    되돌려야 하는 경로용)."""
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def vertices_to_box_pca(vertices: np.ndarray):
    """(8,3) -> (center(3,), dims(3,) [length,width,height 순 - 최대/중간/최소 extent],
    R(3,3)). person4(vertices 포맷) scene용 - PCA로 축 자체를 복원하므로 Euler 분해를
    거치지 않아 gimbal lock/축-라벨링 불안정 문제를 피한다. 다만 두 extent가 같으면
    (몸통 축에 수직한 단면이 정사각형에 가까우면) 그 두 축의 방향 자체는 여전히
    임의적일 수 있음 - 기하학적으로는 여전히 올바른 박스를 복원하므로 학습에는 무해함
    (project_rotation_xy_full_coverage_findings 참고, 아직 미사용 - person4 재요청 대기중)."""
    v = np.asarray(vertices, dtype=np.float64)
    center = v.mean(axis=0)
    centered = v - center
    cov = centered.T @ centered / len(v)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(-eigvals)
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    dims = 2 * np.sqrt(np.maximum(eigvals, 0))
    R = eigvecs.copy()
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    return center.astype(np.float32), dims.astype(np.float32), R.astype(np.float32)
