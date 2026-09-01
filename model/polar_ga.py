"""polar_ga.py - PARTNER(Nie et al., arXiv:2308.03982v2) §3.4 Geometry-aware Adaptive
module의 minimal 변형. 폴라 좌표계 feature map(B,C,R,A)에 aux fg signal + geometry-
aware aggregation을 적용해 feature 품질을 개선한다. GRR과 동일한 삽입 지점(RPNBackbone
직전)의 전처리 블록으로 동작 — 입출력 shape (B,C,R,A) 동일.

원 논문 vs 우리 변형:
  - 원안: fg segmentation head + center offset head 두 개의 명시적 supervision을
    두고, 그 예측을 attention weight로 사용해 feature를 재집계. reg loss가 3개(det + fg + offset).
  - 우리 변형(v1, 2026-08-22): explicit supervision을 뺀 self-contained 형태.
    내부에서 폴라 위치 encoding(r_idx, theta_idx normalized)을 계산하고, feat에서
    자체적으로 fg-like gate를 뽑아 위치 encoding-modulated feature aggregation을 함.
    zero-init residual gate로 삽입 안전성 확보(project_grr_collapse_diagnosis 참고 —
    R=102 스케일에서 bespoke 무정규화 모듈은 BN 불안정 이력 있음).

  이 minimal 변형을 먼저 검증하는 이유: (a) 폴라 target grid에 fg mask/offset을
  raster해야 하는 aux supervision은 fg_target_bev(cartesian 전용) 재구현이 필요,
  (b) v1이 net-positive면 v2에서 aux supervision을 추가하는 순서가 리스크 관리에
  낫다(GRR 첫 시도 실패 → normfix 각색으로 회복한 패턴과 동일).

인접 모듈과의 관계:
  - GRR과 orthogonal: GRR은 radius축 재분배(condense-attend-decondense), GA는 셀별
    geometry-aware feature refinement. 파이프라인 조합 시 GRR → GA 순서 권장(GRR이
    radius 축 정보를 재분배한 뒤 GA가 셀별 refinement).
  - PD-Conv과 orthogonal: PD-Conv는 ConvMiddleLayers 단(z 차원 포함 3D conv), GA는
    ConvMiddleLayers 이후 (B,C,R,A) 2D 단계. 조합 순서는 자연스럽게 PD-Conv → GRR → GA.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def polar_position_encoding(r_bins: int, theta_bins: int, r_range, theta_range_deg,
                             channels: int) -> torch.Tensor:
    """폴라 셀별 (r_norm, theta_norm)을 sinusoidal encoding해서 channel 차원에
    맞게 확장. -> (channels, R, A). GRR의 pos_diff와 달리 절대 좌표를 그대로 인코딩
    (aggregation에서 셀 위치 자체가 필요).

    channels % 4 == 0이면 (r_sin, r_cos, theta_sin, theta_cos) 4-way 균등 분할.
    아니면 균등 분할 후 나머지는 마지막 축(theta_cos)에 몰아넣음."""
    r_min, r_max = r_range
    theta_min_deg, theta_max_deg = theta_range_deg

    r_centers = r_min + (torch.arange(r_bins, dtype=torch.float32) + 0.5) / r_bins * (r_max - r_min)
    theta_centers_deg = theta_min_deg + (torch.arange(theta_bins, dtype=torch.float32) + 0.5) / theta_bins \
        * (theta_max_deg - theta_min_deg)
    theta_centers_rad = torch.deg2rad(theta_centers_deg)

    r_norm = (r_centers - r_min) / max(r_max - r_min, 1e-6)  # (R,) in [0,1]
    theta_norm = (theta_centers_rad - torch.deg2rad(torch.tensor(theta_min_deg))) / \
        max(math.radians(theta_max_deg - theta_min_deg), 1e-6)  # (A,) in [0,1]

    n_per_axis = channels // 4
    freqs = 2.0 ** torch.arange(n_per_axis, dtype=torch.float32) * math.pi

    r_grid = r_norm[:, None].expand(r_bins, theta_bins)  # (R,A)
    theta_grid = theta_norm[None, :].expand(r_bins, theta_bins)  # (R,A)

    r_angles = r_grid.unsqueeze(-1) * freqs  # (R,A,n_per_axis)
    t_angles = theta_grid.unsqueeze(-1) * freqs

    enc = torch.cat([torch.sin(r_angles), torch.cos(r_angles),
                      torch.sin(t_angles), torch.cos(t_angles)], dim=-1)  # (R,A, 4*n_per_axis)

    used = enc.shape[-1]
    if used < channels:
        pad = torch.zeros(r_bins, theta_bins, channels - used, dtype=torch.float32)
        enc = torch.cat([enc, pad], dim=-1)
    return enc.permute(2, 0, 1).contiguous()  # (C,R,A)


class GAModule(nn.Module):
    """Geometry-aware Adaptive minimal (v1). GRR 뒤(또는 없으면 ConvMiddleLayers 직후)에
    삽입되는 전처리 블록. 입출력 (B,C,R,A) 동일.

    파이프라인:
      1. Position encoding: 폴라 셀 절대좌표(r_norm, theta_norm)를 sinusoidal encoding
         → position map (C,R,A). buffer로 등록해 forward마다 재계산 없음.
      2. Internal gate: score_conv(feat)로 fg-like weight 산출(supervision 없음, self-
         attention에 가까운 soft gating). fg_head와 달리 loss 안 걸림.
      3. Aggregation: feat + position map → gate로 modulate → 3x3 conv로 이웃 참조.
         3x3은 폴라 (r,θ)로 해석되므로 근접 r/θ 이웃만 봄.
      4. Residual with zero-init gate: 학습 초반 no-op에서 시작, 점진적 기여.
         GRR과 동일 안정성 장치(project_grr_collapse_diagnosis 근거)."""

    def __init__(self, channels: int, r_bins: int, theta_bins: int, r_range,
                 theta_range_deg, hidden: int = None):
        super().__init__()
        self.channels = channels
        pos_enc = polar_position_encoding(r_bins, theta_bins, r_range, theta_range_deg, channels)
        self.register_buffer("pos_enc", pos_enc, persistent=False)  # (C,R,A)

        # Internal gate — feat → per-cell scalar in [0,1], softly weight aggregation
        self.gate_conv = nn.Conv2d(channels, 1, 1)

        # Aggregation: 3x3 conv sees local r/θ neighborhood
        h = hidden or channels
        self.agg_conv = nn.Conv2d(channels, h, 3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(h)
        self.proj = nn.Conv2d(h, channels, 1)

        # Zero-init residual gate (LayerScale/CaiT류) — 학습 극초반 GA=no-op
        self.residual_gate = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B,C,R,A) — GRR 출력 또는 그 이전 stage feat. -> (B,C,R,A)."""
        B, C, R, A = feat.shape
        assert C == self.channels, f"expected C={self.channels}, got {C}"

        # 1) Position-modulated feat: 위치 encoding을 feat에 additive로 주입
        feat_pos = feat + self.pos_enc.unsqueeze(0)  # broadcast over B

        # 2) Internal soft gate (no supervision, self-attention-like)
        gate = torch.sigmoid(self.gate_conv(feat))  # (B,1,R,A)

        # 3) Gated aggregation with local (r,θ) 이웃 참조
        aggregated = self.agg_conv(feat_pos * gate)
        aggregated = F.relu(self.norm(aggregated))
        refined = self.proj(aggregated)

        # 4) Zero-init residual
        return feat + self.residual_gate * refined
