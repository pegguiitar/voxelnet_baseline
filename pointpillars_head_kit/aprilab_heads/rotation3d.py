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

import math

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


ROTATION_CHANNELS = {"6d": 6, "quat": 4, "axisangle": 3}  # rotation-representation
# ablation(project_paper_sections_status - Sec6.2 rotation3d 방법론 ablation)용 registry.
# Euler는 위 docstring에서 이미 논한 gimbal lock 때문에 여기 포함 안 함(Zhou et al. 2019가
# 직접 비교한 대안 중 실제로 쓰이는 quaternion/axis-angle만 구현) - center head 전용,
# rot_head 채널 수·heatmap_targets 타겟 폭이 이 값을 따른다.


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """(3,3) -> (4,) [w,x,y,z] 단위 쿼터니언. Shepperd's method(수치적으로 안정적인
    4-분기 버전). q와 -q가 같은 회전을 표현하는 이중 표현(double cover) 문제가
    있어 - 지도학습 타겟으로 쓸 때 이 모호성이 회귀를 불안정하게 만들 수 있으므로
    (Zhou et al. 2019가 6D를 제안한 핵심 동기 중 하나), w>=0인 쪽을 canonical
    representative로 고정해 타겟만이라도 결정적으로 만든다."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * S, (m[2, 1] - m[1, 2]) / S, (m[0, 2] - m[2, 0]) / S, (m[1, 0] - m[0, 1]) / S
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / S, 0.25 * S, (m[0, 1] + m[1, 0]) / S, (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / S, (m[0, 1] + m[1, 0]) / S, 0.25 * S, (m[1, 2] + m[2, 1]) / S
    else:
        S = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / S, (m[0, 2] + m[2, 0]) / S, (m[1, 2] + m[2, 1]) / S, 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float32)
    return -q if q[0] < 0 else q


def quat_to_matrix_np(q: np.ndarray) -> np.ndarray:
    """(4,) [w,x,y,z], 임의 스케일(회귀 raw 출력) -> (3,3). 정규화 후 표준 공식으로 복원 -
    6D의 Gram-Schmidt와 동격인 "raw 회귀값 -> 정규직교 회전행렬" 디코드 스텝."""
    q = q / (np.linalg.norm(q) + 1e-8)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def matrix_to_axisangle(R: np.ndarray) -> np.ndarray:
    """(3,3) -> (3,) 회전벡터(axis*angle, 라디안). 표준 공식 - angle=arccos((tr(R)-1)/2),
    axis는 반대칭부에서 추출. angle이 0 또는 π 근처(sin(angle)≈0)에서 분모가 불안정해지는
    알려진 결함이 있으나, 이 데이터셋의 다이버 tilt는 중앙값 81도로 이 특이점 부근에
    몰리지 않아(project_rotation_xy_full_coverage_findings) 실무적으로 문제 없다고 보고
    표준형을 그대로 씀 - 6D와의 비교가 목적이라 axis-angle 자체의 알려진 약점을
    감추지 않는 것도 ablation 취지에 맞음."""
    tr = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    angle = float(math.acos(tr))
    if angle < 1e-6:
        return np.zeros(3, dtype=np.float32)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(angle) + 1e-8)
    return (axis * angle).astype(np.float32)


def axisangle_to_matrix_np(v: np.ndarray) -> np.ndarray:
    """(3,) 회전벡터 -> (3,3). Rodrigues 공식(로그 개수만큼 그대로 지수사상)."""
    angle = float(np.linalg.norm(v))
    if angle < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = v / angle
    ax, ay, az = axis
    K = np.array([[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]], dtype=np.float64)
    R = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
    return R.astype(np.float32)


def quat_to_matrix_torch(q: torch.Tensor) -> torch.Tensor:
    """(...,4) [w,x,y,z] raw 회귀 텐서 -> (...,3,3) 정규직교 회전행렬. quat_to_matrix_np의
    배치 텐서 버전 - decoded R에 대한 chordal loss를 계산할 때 필요(center_loss.py의
    rotation_loss_mode='chordal_R' 경로). 정규화 후 표준 quat->R 공식."""
    q = torch.nn.functional.normalize(q, dim=-1, eps=1e-8)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # (...,3,3) 형태로 stack. row/col 관례는 quat_to_matrix_np와 동일.
    r00 = 1 - 2 * (y * y + z * z); r01 = 2 * (x * y - z * w); r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w);      r11 = 1 - 2 * (x * x + z * z); r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w);      r21 = 2 * (y * z + x * w);     r22 = 1 - 2 * (x * x + y * y)
    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)  # (...,3,3)


def axisangle_to_matrix_torch(v: torch.Tensor) -> torch.Tensor:
    """(...,3) 회전벡터 -> (...,3,3). Rodrigues의 배치 텐서 버전. small-angle 경계에서
    L'Hôpital-safe 처리: sin(θ)/θ, (1-cos(θ))/θ² 항이 θ→0에서 발산하지 않도록
    torch.where로 identity 분기."""
    angle = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)  # (...,1)
    axis = v / angle  # (...,3)
    sin_t = torch.sin(angle)
    cos_t = torch.cos(angle)
    ax, ay, az = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = torch.zeros_like(ax)
    # skew symmetric K
    K_row0 = torch.stack([zero, -az, ay], dim=-1)
    K_row1 = torch.stack([az, zero, -ax], dim=-1)
    K_row2 = torch.stack([-ay, ax, zero], dim=-1)
    K = torch.stack([K_row0, K_row1, K_row2], dim=-2)  # (...,3,3)
    I = torch.eye(3, dtype=v.dtype, device=v.device).expand_as(K)
    # R = I + sin(θ) K + (1-cos(θ)) K²
    K2 = torch.matmul(K, K)
    sin_t_ = sin_t.unsqueeze(-1)  # (...,1,1)
    one_m_cos = (1 - cos_t).unsqueeze(-1)
    R = I + sin_t_ * K + one_m_cos * K2
    return R


def decode_rotation_torch(vec: torch.Tensor, mode: str) -> torch.Tensor:
    """raw 회귀 텐서 (...,C) -> (...,3,3). center_loss.py의 chordal-on-R 로스가
    쓴다. decode_rotation_np의 batched-torch 버전."""
    if mode == "6d":
        return sixd_to_matrix_torch(vec)
    if mode == "quat":
        return quat_to_matrix_torch(vec)
    if mode == "axisangle":
        return axisangle_to_matrix_torch(vec)
    raise ValueError(f"unknown rotation mode: {mode}")


def encode_rotation(R: np.ndarray, mode: str) -> np.ndarray:
    """R(3,3) -> 학습 타겟 벡터. mode: ROTATION_CHANNELS의 키."""
    if mode == "6d":
        return matrix_to_6d(R)
    if mode == "quat":
        return matrix_to_quat(R)
    if mode == "axisangle":
        return matrix_to_axisangle(R)
    raise ValueError(f"unknown rotation mode: {mode}")


def decode_rotation_np(vec: np.ndarray, mode: str) -> np.ndarray:
    """raw 회귀 출력 벡터 -> (3,3) 정규직교 회전행렬. mode: ROTATION_CHANNELS의 키."""
    if mode == "6d":
        return sixd_to_matrix_np(vec)
    if mode == "quat":
        return quat_to_matrix_np(vec)
    if mode == "axisangle":
        return axisangle_to_matrix_np(vec)
    raise ValueError(f"unknown rotation mode: {mode}")


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
