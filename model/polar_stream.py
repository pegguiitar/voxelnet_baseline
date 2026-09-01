"""polar_stream.py - PolarStream(Chen et al. NeurIPS 2021, arXiv:2106.07545) §3.3의
두 모듈을 우리 rot3d + center head 파이프라인에 이식.

1) RangeStratifiedConv1x1 — 하나의 1×1 conv+BN 대신 radial band별로 독립된 1×1 conv+BN
   여러 개를 두고 셀별 range에 따라 다른 파라미터를 적용. PolarStream Tab.4에서 regression
   head(offset/z/dim/rot)에 붙였을 때 +0.9 mAP 단독, Feature Undistortion과 결합 +2.1 mAP.
   근/원거리에서 회귀 target 통계(offset std가 polar에선 near vs far가 크게 다름)가 완전히
   다르다는 관찰에서 출발 — BN running stats까지 band별로 분리해야 효과가 나옴(단일 BN이면
   두 통계의 평균에 수렴해 어느 쪽에도 정확하지 않음).

   우리 세팅 매핑: eval_voxelnet_by_range.BUCKETS(0-2/2-2.5/2.5-3/3-3.5/3.5-5/5+, 6개)와
   동일하게 band를 나눔. Cartesian은 셀당 sqrt(x²+y²)로 band 인덱스 계산, polar는 첫 축(H)
   = r_idx이므로 인덱스만으로 결정. band 경계에서 hard split이라 gradient 불연속이 있지만,
   PolarStream 원 논문도 동일.

2) FeatureUndistortion — heatmap head 앞에 위치-의존 weight/bias를 학습해 feat에 곱/합.
   PolarStream Tab.4에서 단독 +0.4 mAP. Bilinear resampling polar→Cartesian을 명시적으로
   구현하는 대신, 셀별 (r, θ)에 조건화된 scalar w, b를 tiny conv로 학습해 유사 효과.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# 우리 eval bucket과 동일 (eval_voxelnet_by_range.BUCKETS 참고)
_RADIAL_BAND_EDGES = (0.0, 2.0, 2.5, 3.0, 3.5, 5.0, 999.0)  # 6개 band


def _build_band_map_polar(H: int, W: int, r_min: float, r_max: float, device) -> torch.Tensor:
    """polar 격자: 첫 축 H = r_idx. 각 셀의 band 인덱스 (H,W) int64."""
    rs = r_min + (r_max - r_min) * (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H
    # bucketize edges (제외 upper): index 0 = 0-2m, index 5 = 5m+
    band = torch.bucketize(rs, torch.tensor(_RADIAL_BAND_EDGES[1:-1], device=device)) \
        .clamp(0, len(_RADIAL_BAND_EDGES) - 2)
    band_map = band.unsqueeze(1).expand(H, W).contiguous()
    return band_map


def _build_band_map_cartesian(H: int, W: int, device) -> torch.Tensor:
    """Cartesian 격자: 셀 중심 sqrt(x²+y²)로 band."""
    sx, sy = config.ANCHOR_STRIDE
    x0, y0 = config.POINT_CLOUD_RANGE[:2]
    xs = x0 + sx * (torch.arange(W, device=device, dtype=torch.float32) + 0.5)
    ys = y0 + sy * (torch.arange(H, device=device, dtype=torch.float32) + 0.5)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    r = torch.sqrt(xx * xx + yy * yy)  # (H,W)
    band = torch.bucketize(r, torch.tensor(_RADIAL_BAND_EDGES[1:-1], device=device)) \
        .clamp(0, len(_RADIAL_BAND_EDGES) - 2)
    return band  # (H,W)


class RangeStratifiedConv1x1(nn.Module):
    """1×1 conv + BN을 radial band 개수만큼 병렬로 두고, 셀별 band 인덱스에 따라
    선택적으로 적용. drop-in 대체: nn.Sequential(Conv2d(c_in,c_out,1), BN, ReLU) 대신
    이 모듈을 쓰면 됨. 파라미터는 band 수 × 원본 파라미터 (우리 6-band면 6배).

    구현: band별 forward를 병렬 계산(모든 band conv 다 돌린 뒤 mask로 선택 합산).
    band당 conv 자체가 1×1이라 총 계산량은 band 수 × 원본이지만 아직 backbone 대비
    미미(offset/z/dim/rot head는 아주 얕음)."""

    def __init__(self, in_channels: int, out_channels: int, polar: bool, r_min: float = None,
                 r_max: float = None):
        super().__init__()
        self.n_bands = len(_RADIAL_BAND_EDGES) - 1
        self.polar = polar
        self.r_min = r_min if r_min is not None else config.POLAR_R_RANGE[0]
        self.r_max = r_max if r_max is not None else config.POLAR_R_RANGE[1]
        self.convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, 1) for _ in range(self.n_bands)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(self.n_bands)])
        self._band_map_cache = None  # (1,1,H,W) or (H,W)

    def _band_map(self, H: int, W: int, device) -> torch.Tensor:
        cache = self._band_map_cache
        if cache is not None and cache.shape[-2:] == (H, W) and cache.device == device:
            return cache
        if self.polar:
            bm = _build_band_map_polar(H, W, self.r_min, self.r_max, device)
        else:
            bm = _build_band_map_cartesian(H, W, device)
        self._band_map_cache = bm  # (H,W)
        return bm

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B,C_in,H,W) -> (B,C_out,H,W)."""
        B, _, H, W = feat.shape
        band_map = self._band_map(H, W, feat.device)  # (H,W)
        out = None
        for b_idx in range(self.n_bands):
            mask = (band_map == b_idx).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            if mask.sum() == 0:
                continue  # 이 band는 셀 없음 (Cartesian 코너 등) - skip
            band_out = self.bns[b_idx](self.convs[b_idx](feat))  # (B,C_out,H,W)
            contrib = band_out * mask
            out = contrib if out is None else out + contrib
        return out


class FeatureUndistortion(nn.Module):
    """PolarStream §3.3 Feature Undistortion. feat과 셀별 position(r,θ)에서
    tiny 3×3 conv → 1×1 conv → tanh로 (w, b) 두 채널 예측 → feat = w*feat + b.
    (w,b)는 원래 bilinear resampling polar→Cartesian의 approximation. tanh로 [-1,1] 안정화.

    heatmap head **앞에만** 붙임 (classification 전용, PolarStream Tab.4 원안).
    regression head는 RangeStratifiedConv1x1이 담당."""

    def __init__(self, in_channels: int, hidden: int = 32):
        super().__init__()
        # 위치 encoding은 첫 conv가 자동 학습 (전체 (H,W) feature map에서 위치 정보를 3×3
        # receptive로 볼 수 있으므로 explicit pos map 불필요)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2 * in_channels, 1),  # w (C), b (C) - channel-wise
            nn.Tanh(),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B,C,H,W) -> (B,C,H,W). w,b는 [-1,1], w는 (1+w)로 scale해 대략 [0,2] 배."""
        wb = self.net(feat)  # (B, 2*C, H, W)
        C = feat.shape[1]
        w, b = wb[:, :C], wb[:, C:]
        return (1.0 + w) * feat + b
