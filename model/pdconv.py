"""pdconv.py - PVP (Xue et al., arXiv:2412.07616) Section 3.3의 PD-Conv
(Plane Decomposed Convolution). polar 좌표계 3D feature volume의 각 평면별
왜곡 특성에 맞춰 표준 3×3×3 Conv3D를 세 개의 비대칭 2D conv로 분해한다.

polar grid 텐서 배치 (D,H,W) = (z, r, θ) 기준:
  - slice view  (θ,z): 왜곡 없음 → identity   → kernel (3, 1, 3)
  - range view  (r,z): scale 왜곡 → scale      → kernel (3, 3, 1)
  - BEV   view  (r,θ): 투영 왜곡  → projection → kernel (1, 3, 3)

세 conv를 serial로 쌓는다(PVP Table 5 option (a) — 성능 차이 미미, 가장 단순).
ConvMiddleLayers의 drop-in 대체용으로, 입출력 shape은 완전히 동일하다.
"""

import torch.nn as nn
import torch.nn.functional as F


class PDConvBlock(nn.Module):
    """3×3×3 Conv3d 하나를 세 개의 평면별 2D conv(serial)로 분해한 블록.

    첫 conv(slice)가 채널 변환(in→out) + z-axis stride/padding을 담당하고,
    이후 두 conv는 채널·공간 유지. conv_range의 z-padding은 항상 1로 고정해서
    첫 conv가 줄인 z 차원을 이후 conv가 다시 줄이지 않도록 한다."""

    def __init__(self, in_channels: int, out_channels: int,
                 stride: tuple = (1, 1, 1), padding: tuple = (1, 1, 1)):
        super().__init__()
        sz, sr, st = stride
        pz, pr, pt = padding

        self.conv_slice = nn.Conv3d(in_channels, out_channels, (3, 1, 3),
                                    stride=(sz, 1, 1), padding=(pz, 0, pt))
        self.bn_slice = nn.BatchNorm3d(out_channels)

        self.conv_range = nn.Conv3d(out_channels, out_channels, (3, 3, 1),
                                    stride=(1, sr, 1), padding=(1, pr, 0))
        self.bn_range = nn.BatchNorm3d(out_channels)

        self.conv_bev = nn.Conv3d(out_channels, out_channels, (1, 3, 3),
                                  stride=(1, 1, st), padding=(0, pr, pt))
        self.bn_bev = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn_slice(self.conv_slice(x)))
        x = F.relu(self.bn_range(self.conv_range(x)))
        x = F.relu(self.bn_bev(self.conv_bev(x)))
        return x


class PDConvMiddleLayers(nn.Module):
    """ConvMiddleLayers의 PD-Conv 대체 버전. polar 전용.
    3× Conv3d(3,3,3) → 3× PDConvBlock(각 3개 decomposed conv = 총 9 conv).
    입출력 shape 동일: (B,128,D',H',W') → (B,64,D'',H',W'), D'=10 → D''=2."""

    def __init__(self):
        super().__init__()
        self.pd1 = PDConvBlock(128, 64, stride=(2, 1, 1), padding=(1, 1, 1))
        self.pd2 = PDConvBlock(64, 64, stride=(1, 1, 1), padding=(0, 1, 1))
        self.pd3 = PDConvBlock(64, 64, stride=(2, 1, 1), padding=(1, 1, 1))

    def forward(self, x):
        """x: (B,128,D',H',W') -> (B,64,D'',H',W') with D''=2 for D'=10."""
        x = self.pd1(x)
        x = self.pd2(x)
        x = self.pd3(x)
        return x
