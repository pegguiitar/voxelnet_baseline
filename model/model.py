"""model.py - VoxelNet 본체: VFE 스택 -> dense grid scatter -> Conv3D middle layers
-> RPN. 논문 §2.1 그대로, sparse convolution 라이브러리(spconv) 없이 순수 PyTorch
dense 텐서로 구현한다(§2.3 "efficient implementation"이 원래도 dense grid scatter
설계라 spconv 없이도 논문 그대로임 - spconv는 VoxelNet 이후 SECOND가 이 dense
Conv3D를 대체하려고 만든 후속 라이브러리).
"""

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

import anchors as anchors_mod
import config
import stage2_refine
from partner_grr import GRRModule
from pdconv import PDConvMiddleLayers
from polar_ga import GAModule
import polar_stream

_LEGACY_RPN_KEY = re.compile(r"^rpn\.(block\d|deconv\d)\.")


def remap_legacy_state_dict(state_dict: dict) -> dict:
    """RPNBackbone 분리 리팩터(anchor-free head 도입 시점) 이전 체크포인트 호환용.
    그때는 rpn.block*/rpn.deconv*가 RPN의 직속 자식이었는데, 지금은 RPNBackbone
    서브모듈로 한 단계 더 감싸여 rpn.backbone.block*/rpn.backbone.deconv*가 됐다 -
    텐서 값은 완전히 동일하고 키 경로만 바뀐 것이므로 이름만 다시 매핑하면 그대로
    로드된다(voxelnet_run1.pt/run2.pt처럼 리팩터 이전에 학습된 체크포인트에 필요)."""
    return {_LEGACY_RPN_KEY.sub(r"rpn.backbone.\1.", k): v for k, v in state_dict.items()}


def load_state_dict_compat(model: nn.Module, state_dict: dict) -> None:
    """현재 구조로 먼저 시도, 실패하면 구버전 키 리매핑으로 재시도, 그래도 실패하면
    strict=False로 마지막 시도한다 - 체크포인트가 지금 구조보다 오래돼서(예: density_head
    도입 이전) 일부 레이어가 아예 없는 경우(remap으로도 못 고침, 이름이 아니라 키 자체가
    없는 문제) 그 레이어만 랜덤 초기화로 남기고 나머지는 정상 로드한다. 추론/decode에서
    안 쓰는 보조 head(예: density_head)라면 랜덤 초기화로 남아도 결과에 영향 없음 -
    다만 missing/unexpected 키를 항상 출력해서 "정말 안전한 차이인지" 눈으로 확인 가능하게 함."""
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass
    try:
        model.load_state_dict(remap_legacy_state_dict(state_dict))
        return
    except RuntimeError:
        pass
    result = model.load_state_dict(state_dict, strict=False)
    print(f"[load_state_dict_compat] strict=False로 로드 - missing={result.missing_keys}, "
          f"unexpected={result.unexpected_keys}")


def _pool_points(pointwise: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """voxel 내 point-wise feature -> voxel-wise 대표값(hard max-pool).
    pointwise: (K,T,C), mask: (K,T) bool. -> (K,1,C)."""
    neg_inf = torch.finfo(pointwise.dtype).min
    masked_for_max = pointwise.masked_fill(~mask.unsqueeze(-1), neg_inf)
    aggregated, _ = masked_for_max.max(dim=1, keepdim=True)  # (K,1,C)
    return aggregated.clamp_min(0.0)  # 전부 무효인 voxel은 없지만(K는 non-empty voxel만) 방어적으로


class VFELayer(nn.Module):
    """FCN(Linear+BN+ReLU) -> voxel별 hard max-pool -> point-wise concat. 패딩된(무효)
    포인트가 BatchNorm 통계를 오염시키지 않도록, Linear+BN은 유효 포인트만 모아서
    (flatten+mask) 계산한다."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        assert out_channels % 2 == 0
        self.units = out_channels // 2
        self.linear = nn.Linear(in_channels, self.units)
        self.bn = nn.BatchNorm1d(self.units)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (K,T,Cin), mask: (K,T) bool (True=valid). -> (K,T,Cout)."""
        K, T, _ = x.shape
        flat = x.reshape(K * T, -1)
        flat_mask = mask.reshape(K * T)
        pointwise = flat.new_zeros(K * T, self.units)
        if flat_mask.any():
            pointwise[flat_mask] = F.relu(self.bn(self.linear(flat[flat_mask])))
        pointwise = pointwise.reshape(K, T, self.units)

        aggregated = _pool_points(pointwise, mask)  # (K,1,units)
        aggregated = aggregated.expand(-1, T, -1)

        out = torch.cat([pointwise, aggregated], dim=2)  # (K,T,2*units=Cout)
        return out * mask.unsqueeze(-1)


class StackedVFE(nn.Module):
    """VFE-1(7,32) -> VFE-2(32,128) -> FCN(128,128)+BN+ReLU -> hard max-pool over points.
    (K,T,7) -> (K,128) voxel-wise feature."""

    def __init__(self):
        super().__init__()
        self.vfe1 = VFELayer(config.INPUT_FEATURE_DIM, 32)
        self.vfe2 = VFELayer(32, 128)
        self.final_linear = nn.Linear(128, 128)
        self.final_bn = nn.BatchNorm1d(128)

    def forward(self, voxel_features: torch.Tensor, num_points: torch.Tensor) -> torch.Tensor:
        K, T, _ = voxel_features.shape
        mask = torch.arange(T, device=voxel_features.device)[None, :] < num_points[:, None]

        x = self.vfe1(voxel_features, mask)
        x = self.vfe2(x, mask)

        flat = x.reshape(K * T, -1)
        flat_mask = mask.reshape(K * T)
        pointwise = flat.new_zeros(K * T, 128)
        if flat_mask.any():
            pointwise[flat_mask] = F.relu(self.final_bn(self.final_linear(flat[flat_mask])))
        pointwise = pointwise.reshape(K, T, 128)

        voxelwise = _pool_points(pointwise, mask).squeeze(1)  # (K,128)
        return voxelwise


class ConvMiddleLayers(nn.Module):
    """3x Conv3D, D'(=10)를 2로 줄이면서 채널을 64로 - 논문 car config 그대로
    (config.py에서 vz=0.5로 D'=10을 맞춰놨기 때문에 그대로 재사용 가능)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(128, 64, 3, stride=(2, 1, 1), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 64, 3, stride=(1, 1, 1), padding=(0, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 64, 3, stride=(2, 1, 1), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,128,D',H',W') -> (B,64,D'',H',W') with D''=2 for D'=10."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x


class RPNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                   nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 1):
            layers += [nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
                       nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


class RPNBackbone(nn.Module):
    """논문 Fig.4의 3-block downsample + deconv-upsample-concat neck - head(cls/reg
    anchor 방식이든 heatmap anchor-free 방식이든) 이전까지 공통인 부분만 뗐다.
    `RPN`(anchor 기반)과 `RPNCenterHead`(anchor-free, heatmap_targets.py 참고)가
    이 클래스를 그대로 공유한다 - "head만 바꾼 버전"을 만들 때 백본이 한 글자도
    안 달라야 비교가 공정하다는 게 이 리팩터의 목적.

    Fig.4에 박힌 deconv1 파라미터(Deconv2D(128,256,3,1,0))는 그대로 구현하면 크기가
    안 맞는다 - block1 출력이 이미 목표 해상도(H/2,W/2)인데 kernel3/stride1/pad0인
    ConvTranspose2d는 출력을 +2px 키워버려서(공식: out=(in-1)*stride-2*pad+kernel)
    block2/3의 deconv 출력과 concat이 안 된다. 이 불일치를 검증하려고 VoxelNet의
    직계 후속작 SECOND의 공식 참조 구현(traveller59/second.pytorch, rpn.py)을
    확인한 결과, 거기서도 이 지점을 kernel_size=stride=upsample_strides(1,2,4),
    padding=0 규칙으로 우회하고 있었다(block1은 kernel1/stride1로 사실상 순수
    채널 투영, spatial 변화 없음) - 이 프로젝트도 그 관례를 그대로 따른다(block2:
    k2/s2, block3: k4/s4 - 논문 Fig.4 수치와 정확히 일치, block1만 논문 수치
    대신 SECOND 관례로 대체).
    """

    def __init__(self):
        super().__init__()
        c_in = config.RPN_IN_CHANNELS
        c1, c2, c3 = config.RPN_BLOCK_CHANNELS
        n1, n2, n3 = config.RPN_BLOCK_LAYERS
        c_up = config.RPN_UPSAMPLE_CHANNELS

        self.block1 = RPNBlock(c_in, c1, n1)
        self.block2 = RPNBlock(c1, c2, n2)
        self.block3 = RPNBlock(c2, c3, n3)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(c1, c_up, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(c_up), nn.ReLU(inplace=True))
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(c2, c_up, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(c_up), nn.ReLU(inplace=True))
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(c3, c_up, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(c_up), nn.ReLU(inplace=True))

        self.out_channels = c_up * 3

    @staticmethod
    def _match_hw(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """H'가 8로 나누어떨어지지 않으면(예: 100/8=12.5) block3의 stride-2 conv
        3번이 floor 반올림되어 최종 deconv 출력이 목표보다 몇 px 커진다 - 중앙 crop으로 맞춘다."""
        h, w = x.shape[-2], x.shape[-1]
        top, left = (h - target_h) // 2, (w - target_w) // 2
        return x[..., top:top + target_h, left:left + target_w]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, RPN_IN_CHANNELS, H', W') -> (B, 768, H'/2, W'/2) concat feature."""
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        u1, u2, u3 = self.deconv1(f1), self.deconv2(f2), self.deconv3(f3)
        target_h, target_w = u1.shape[-2], u1.shape[-1]
        u2 = self._match_hw(u2, target_h, target_w)
        u3 = self._match_hw(u3, target_h, target_w)
        feat = torch.cat([u1, u2, u3], dim=1)
        return feat


class RPN(nn.Module):
    """RPNBackbone + 논문 원안 anchor 기반 cls/reg head."""

    def __init__(self, num_anchors_per_loc: int):
        super().__init__()
        self.backbone = RPNBackbone()
        A = num_anchors_per_loc
        self.cls_head = nn.Conv2d(self.backbone.out_channels, A, 1)
        self.reg_head = nn.Conv2d(self.backbone.out_channels, A * 12, 1)

    def forward(self, x: torch.Tensor):
        """x: (B, RPN_IN_CHANNELS, H', W') -> cls (B,A,H'/2,W'/2), reg (B,A*12,H'/2,W'/2) -
        마지막 6채널(A당)이 완전한 3D 회전의 6D continuous representation(rotation3d.py,
        anchors.assign_targets 참고) - z-yaw만 담던 sin/cos 2채널에서 확장됨."""
        feat = self.backbone(x)
        return self.cls_head(feat), self.reg_head(feat)


class ForegroundHead(nn.Module):
    """Direction 3(reports_v2/foreground_aux_branch_proposal.md 4.7) - RPNCenterHead가
    공유하는 backbone 출력 feat(768ch, heatmap_head 등 다른 head와 동일 지점)에서
    foreground/background gating 신호(P_fg)를 뽑는다. density_head와 똑같은 패턴(작은
    1x1 conv 스택, 학습 때만 의미 있고 추론 시 안 써도 됨)이라 구조를 그대로 재사용했다 -
    작은 hidden layer 하나만 추가(Linear(768,64)->ReLU->Linear(64,1)를 1x1 conv로 표현).

    2026-08-22: use_range_cond=True면 각 셀의 r 값을 sinusoidal encoding해 feat과 concat.
    3-3.5m bucket에서 fg_gate가 유독 헷갈리는(train 10.8% vs test 23.8% mismatch, §3
    분석) 원인이 인코더의 암묵적 range encoding 부실에 있다는 가설을 완화한다.
    n_freqs=4로 파장 0.8/1.6/3.2/6.4m 커버(우리 range 0-6m 대역).
    polar=True면 첫 축(r_idx)에서 config.POLAR_R_RANGE 매핑, 아니면 셀 중심 sqrt(x²+y²)."""

    def __init__(self, in_channels: int, hidden: int = 64,
                 use_range_cond: bool = False, polar: bool = False, n_freqs: int = 4):
        super().__init__()
        self.use_range_cond = use_range_cond
        self.polar = polar
        self.n_freqs = n_freqs
        extra = 2 * n_freqs if use_range_cond else 0  # sin + cos
        self.net = nn.Sequential(
            nn.Conv2d(in_channels + extra, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1))
        self._range_enc_cache = None  # (1, 2*n_freqs, H, W) - 형상 바뀌면 재계산

    def _range_encoding(self, H: int, W: int, device, dtype) -> torch.Tensor:
        cache = self._range_enc_cache
        if cache is not None and cache.shape[-2:] == (H, W) \
                and cache.device == device and cache.dtype == dtype:
            return cache

        if self.polar:
            r_lo, r_hi = config.POLAR_R_RANGE
            # polar 격자: 첫 축(H) = r_idx, 두 번째 축(W) = theta_idx
            rs = r_lo + (r_hi - r_lo) * (torch.arange(H, device=device, dtype=dtype) + 0.5) / H
            r_map = rs.unsqueeze(1).expand(H, W)
        else:
            sx, sy = config.ANCHOR_STRIDE
            x0, y0 = config.POINT_CLOUD_RANGE[:2]
            xs = x0 + sx * (torch.arange(W, device=device, dtype=dtype) + 0.5)
            ys = y0 + sy * (torch.arange(H, device=device, dtype=dtype) + 0.5)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            r_map = torch.sqrt(xx * xx + yy * yy)  # (H,W)

        # 파장 0.8m → 6.4m 로그 스케일 (우리 range 0-6m에 맞춤)
        base = 2.0 * math.pi / 6.4
        freqs = base * (2.0 ** torch.arange(self.n_freqs, device=device, dtype=dtype))
        angles = r_map.unsqueeze(-1) * freqs  # (H,W,n_freqs)
        enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (H,W,2n)
        enc = enc.permute(2, 0, 1).unsqueeze(0).contiguous()  # (1,2n,H,W)
        self._range_enc_cache = enc
        return enc

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if self.use_range_cond:
            B, _, H, W = feat.shape
            enc = self._range_encoding(H, W, feat.device, feat.dtype).expand(B, -1, -1, -1)
            feat = torch.cat([feat, enc], dim=1)
        return self.net(feat)  # (B,1,H,W) raw logit - sigmoid는 호출측에서


class RotVolBranch(nn.Module):
    """`rot_head` 전용 보조 입력 - `dense`(VFE 직후, z가 아직 spatial axis로 살아있는
    지점, ConvMiddleLayers 이전)에서 갈라져 나온 별도 Conv3D 스택(2026-08-20, reports_v2/
    foreground_aux_branch_proposal.md §1.4.4).

    **정정(2026-08-20, 사용자 지적)**: 처음엔 "2D conv는 z구조를 못 본다"고 설명했는데
    부정확했다 - `ConvMiddleLayers`도 이미 진짜 Conv3D 3개로 z를 10->5->3->2까지
    줄인다(z-비교 자체는 이미 하고 있음). 그리고 conv2가 RPNBackbone.block1과 같은
    conv 공식(kernel3/stride2/padding1)을 쓰는 것도 z 보존과 무관하다 - 그건 순전히
    (H,W) 해상도를 `feat`과 맞춰 concat 가능하게 하려는 것뿐.

    그럼 별도 branch를 왜 두나(ConvMiddleLayers 중간 출력에서 그냥 갈라지면 안 되나)?
    **진짜 이유는 z 해상도 차이가 아니라 파라미터를 공유하지 않는다는 것.**
    `ConvMiddleLayers`의 3개 conv는 heatmap/offset/dim/density 넷과 완전히 공유되고,
    이 넷의 loss(hm_loss+reg_loss, 학습 로그에서 rotation을 포함한 reg_loss 전체가
    수 단위인 것과 비교해 heatmap도 비슷한 크기 - 즉 rotation은 여러 항 중 하나일
    뿐)가 detection 정확도 쪽으로 훨씬 크게 지배적이다. `ConvMiddleLayers`의 conv
    가중치는 그 지배적인 gradient에 맞춰 최적화되므로, 설령 z-비교 능력 자체는
    있어도 "detection에 필요한 z 요약"으로 수렴하지 학습된다 - rotation에 최적인
    z 요약과 같으리라는 보장이 없다(multi-task 학습에서 공유 trunk가 지배적 task
    쪽으로 표현을 끌고 가는 건 잘 알려진 현상, Mask R-CNN류가 box/mask에 별도
    head를 두는 것과 같은 이유). 별도 branch는 rotation loss만 보고 z를 요약하는
    법을 독립적으로 배울 수 있다 - 이게 진짜 차별점이다.

    z 해상도 자체(로컬 진단, GT tilt vs PCA tilt 상관 0.898, shuffle 대조군 0.035로
    raw point엔 신호가 충분함을 확인)는 부가적 이점이라 conv3의 z-stride를 1로 둬서
    z=5를 끝까지 유지한다(기존 ConvMiddleLayers의 z=2보다 낫게) - 하지만 이게
    주된 정당화 근거는 아니다.

    설계: ConvMiddleLayers와 완전히 별개 파라미터로 z를 10->5->5->5로 유지하고
    (conv2에서만 (H,W)를 RPNBackbone.block1과 같은 conv 공식으로 2x 다운샘플해
    `feat`과 크기를 맞춤), 마지막 1x1 conv(`proj`)만 0-init - 도입 시점에 기존
    모델과 완전히 동일한 출력에서 시작하고, 학습되며 유용하면 서서히 이 채널을
    쓰게 된다(PARTNER GRR/RAANet 게이팅과 동일한 안전장치).

    검증된 특이 동작(버그 아님, 2026-08-20 실측) - **첫 optimizer step에서는
    conv1/conv2/conv3의 gradient가 정확히 0이다.** `proj.weight`가 0이라 rot_head->proj
    까지는 정상적으로 gradient가 흐르지만(dL/dy는 rot_head.weight에만 의존, proj의
    현재 값과 무관), proj를 넘어 conv3/conv2/conv1로 더 거슬러 가려면 dL/d(conv3_out)
    = dL/dy @ proj.weight를 계산해야 하는데 proj.weight=0이라 여기서 막힌다. 첫
    optimizer.step()이 proj.weight를 0에서 살짝 밀어내면(자기 자신의 gradient로),
    그 다음 step부터는 conv1~3에도 정상적으로 gradient가 흐른다(실측 확인: step0
    conv1.grad=0.0 -> step1 conv1.grad=0.031). Zero-init 게이팅 설계의 잘 알려진
    "부트스트랩" 특성 - 여기서 처음 발견한 문제가 아니라 이 패턴 자체의 일반적 성질."""

    def __init__(self, in_channels: int = 128, c1: int = 64, c2: int = 32, c3: int = 16,
                 out_channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, c1, 3, stride=(2, 1, 1), padding=1)
        self.bn1 = nn.BatchNorm3d(c1)
        self.conv2 = nn.Conv3d(c1, c2, 3, stride=(1, 2, 2), padding=1)  # (H,W) 2x다운 - block1과 동일 arithmetic, z는 유지
        self.bn2 = nn.BatchNorm3d(c2)
        self.conv3 = nn.Conv3d(c2, c3, 3, stride=(1, 1, 1), padding=1)  # z-stride=1 - z=5 그대로 유지(기존 z=2보다 많이 남김)
        self.bn3 = nn.BatchNorm3d(c3)
        self.z_out = 5
        self.proj = nn.Conv2d(c3 * self.z_out, out_channels, 1)  # conv3 출력 z=5 -> reshape해 채널로
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.out_channels = out_channels

    def forward(self, dense: torch.Tensor) -> torch.Tensor:
        """dense: (B,128,Dp=10,Hp,Wp) -> (B,out_channels,Hp/2,Wp/2), rot_head 입력에
        concat할 것(호출측 책임)."""
        x = F.relu(self.bn1(self.conv1(dense)))   # (B,c1,5,Hp,Wp)
        x = F.relu(self.bn2(self.conv2(x)))        # (B,c2,5,Hp/2,Wp/2)
        x = F.relu(self.bn3(self.conv3(x)))        # (B,c3,5,Hp/2,Wp/2) - z-stride=1이라 z 유지
        B, C, D, H, W = x.shape
        return self.proj(x.reshape(B, C * D, H, W))


class RPNCenterHead(nn.Module):
    """RPNBackbone + CenterPoint(Yin et al. 2021) 스타일 anchor-free head.
    heatmap_targets.py와 짝을 이룬다 - 정적 IoU anchor 매칭(RPN의 cls_head) 대신
    객체 중심을 Gaussian heatmap으로 직접 회귀하고, 크기/회전/z도 anchor 없이
    절대값(회전은 sin/cos 연속 표현)으로 회귀한다. 백본은 RPN과 완전히 동일
    (RPNBackbone 공유) - "head만 바꾼" 비교가 성립하는 이유."""

    def __init__(self, use_fg_head: bool = False, use_stage2: bool = False, use_rot_branch: bool = False,
                 fg_range_cond: bool = False, polar: bool = False,
                 use_rs_conv: bool = False, use_feat_undistort: bool = False):
        super().__init__()
        self.polar = polar
        self.use_rs_conv = use_rs_conv
        self.use_feat_undistort = use_feat_undistort
        self.backbone = RPNBackbone()
        c = self.backbone.out_channels
        # PolarStream §3.3 Feature Undistortion - heatmap head 앞에 위치-adaptive scale/shift
        # (polar_stream.py). polar 전용, +0.4 mAP (PolarStream Tab.4). 없으면 no-op.
        self.feat_undistort = polar_stream.FeatureUndistortion(c) if use_feat_undistort else None
        self.heatmap_head = nn.Conv2d(c, 1, 1)   # raw logit - sigmoid는 loss/decode에서
        # PolarStream §3.3 Range-Stratified Conv+BN - regression head 대체.
        # radial band(6개)별 독립 1×1 conv+BN. +0.9 alone, +2.1 combined w/ FU. polar 전용
        # (Cartesian은 원래 정규 grid라 이득이 논문에서도 미미).
        if use_rs_conv:
            self.offset_head = polar_stream.RangeStratifiedConv1x1(c, 2, polar=polar)
            self.z_head = polar_stream.RangeStratifiedConv1x1(c, 1, polar=polar)
            self.dim_head = polar_stream.RangeStratifiedConv1x1(c, 3, polar=polar)
        else:
            self.offset_head = nn.Conv2d(c, 2, 1)    # [dx,dy] sub-pixel, 셀 크기 단위
            self.z_head = nn.Conv2d(c, 1, 1)         # 절대 z(m), anchor 없이 직접 회귀
            self.dim_head = nn.Conv2d(c, 3, 1)       # [log l, log w, log h]
        # RotVolBranch(§1.4.4) 전용 - rot_head만 dense(z 보존) 기반 보조 입력을 추가로
        # 받는다. 0-init이라 use_rot_branch=True로 켜도 학습 시작 시점엔 rot_branch=False와
        # 완전히 동일한 출력.
        self.rot_branch = RotVolBranch(in_channels=config.RPN_IN_CHANNELS) if use_rot_branch else None
        rot_in = c + (self.rot_branch.out_channels if self.rot_branch is not None else 0)
        self.rot_head = nn.Conv2d(rot_in, 6, 1)  # 6D continuous rotation(rotation3d.py) - 완전한 3D 회전
        # RAANet(arXiv:2111.09515)식 보조 density-level classification(sparse/adequate/
        # dense 3클래스, config.DENSITY_THRESH_LOW/HIGH) - heatmap_targets.py의 "density"
        # 타겟과 짝. 학습 때만 의미 있고(positive cell에서 CE loss) 추론 시엔 그냥 무시하면
        # 됨(decode 경로는 안 씀) - 1x1 conv라 항상 계산해도 비용이 사실상 0이라 forward()
        # 에서 조건부로 끄는 분기 자체를 안 둠(단순함이 우선).
        self.density_head = nn.Conv2d(c, 3, 1)
        # Direction 3 "jointly" variant 전용(기본 off) - freeze_fit variant는 fg_head를
        # 이 모델 밖(train.py)에 별도로 두고 메인 optimizer에 안 넣으므로 여기 없음.
        self.fg_head = ForegroundHead(c, use_range_cond=fg_range_cond, polar=polar) if use_fg_head else None
        # CenterPoint stage-2(RoI refinement, stage2_refine.py) 전용 - forward()에서는
        # 안 부른다(freeze_fit과 같은 hook-capture 패턴으로 train.py가 외부에서 feat.detach()
        # 를 받아 직접 호출함 - candidate가 1단계 decode 출력이 아니라 GT jitter이므로
        # forward() 안에 넣을 이유가 없음). 그래도 model.parameters()에는 포함되도록
        # submodule로 등록(메인 optimizer가 같이 학습 - feat이 detach되므로 backbone
        # 으로는 역전파 안 됨, stage2 자체 파라미터만 갱신).
        self.stage2 = stage2_refine.Stage2Head(c) if use_stage2 else None

        # focal-loss 표준 bias 초기화(RetinaNet Lin et al. 2017 §3.3, CenterNet Zhou
        # et al. 2019 동일 관례) - 기본 PyTorch 초기화(bias≈0)로 두면 학습 시작 시
        # sigmoid(0)=0.5가 grid 전체 3,000셀(60x50)에 균일하게 찍히는데, 프레임당
        # positive는 보통 1~2개뿐이라 negative 쪽 gradient가 압도적으로 커서 heatmap
        # head가 "어디든 배경"으로 붕괴한다(실측: voxelnet_center_run1 학습된 가중치의
        # heatmap_head.weight 평균이 -8.8까지 밀려나 있었고, sigmoid 출력 최댓값이
        # 전체 grid에서 0.126을 못 넘어 score_thresh=0.3 decode가 매번 0개를 반환).
        # bias를 미리 -log((1-pi)/pi)로 낮춰두면 학습 시작 시 sigmoid 출력이 pi(작은
        # 값)에서 시작해 이 붕괴를 막는다.
        prior_prob = 0.1
        nn.init.constant_(self.heatmap_head.bias, -math.log((1 - prior_prob) / prior_prob))

    def forward(self, x: torch.Tensor, return_foreground: bool = False, dense: torch.Tensor = None):
        """x: (B, RPN_IN_CHANNELS, H', W') -> heatmap(B,1,H'/2,W'/2),
        offset/z/dim/rot/density 전부 (B,C,H'/2,W'/2).

        return_foreground=True(기본 False)면 7번째 원소로 fg_logit(B,1,H'/2,W'/2)을 덧붙여
        반환한다 - self.fg_head가 없으면(use_fg_head=False로 생성된 모델) 에러. 기본값
        False를 유지해야 model()을 6-tuple로 unpack하는 기존 8개 호출부(train.py,
        eval_voxelnet.py 등)가 안 깨진다.

        dense: (B,128,Dp,Hp,Wp) - self.rot_branch가 있을 때만 필요(VoxelNet.forward가
        전달). rot_head 입력에만 concat되고 다른 head는 안 건드림(§1.4.4)."""
        feat = self.backbone(x)
        rot_in = feat
        if self.rot_branch is not None:
            assert dense is not None, "use_rot_branch=True인 모델은 forward()에 dense를 넘겨야 함"
            rot_in = torch.cat([feat, self.rot_branch(dense)], dim=1)
        # PolarStream §3.3 Feature Undistortion - heatmap head 앞에만 (polar_stream.py).
        feat_hm = self.feat_undistort(feat) if self.feat_undistort is not None else feat
        out = (self.heatmap_head(feat_hm), self.offset_head(feat), self.z_head(feat),
               self.dim_head(feat), self.rot_head(rot_in), self.density_head(feat))
        if return_foreground:
            assert self.fg_head is not None, "return_foreground=True인데 use_fg_head=False로 생성된 모델"
            out = out + (self.fg_head(feat),)
        return out


class VoxelNet(nn.Module):
    def __init__(self, head: str = "anchor", polar: bool = False, use_grr: bool = False,
                 use_fg_head: bool = False, use_stage2: bool = False, use_rot_branch: bool = False,
                 use_pdconv: bool = False, grr_n: int = None, use_ga: bool = False,
                 fg_range_cond: bool = False,
                 use_rs_conv: bool = False, use_feat_undistort: bool = False):
        super().__init__()
        assert head in ("anchor", "center")
        assert not (use_grr and not polar), "GRR은 polar 전용 모듈이다"
        assert not (use_pdconv and not polar), "PD-Conv는 polar 전용 모듈이다"
        assert not (use_ga and not polar), "GA는 polar 전용 모듈이다"
        assert not (use_rs_conv and not polar), "Range-Stratified Conv는 polar 전용 (PolarStream §3.3)"
        assert not (use_feat_undistort and not polar), "Feature Undistortion은 polar 전용"
        assert not (use_fg_head and head != "center"), "fg_head는 center head 전용"
        assert not (use_stage2 and head != "center"), "stage2는 center head 전용"
        assert not (use_rot_branch and head != "center"), "rot_branch는 center head 전용"
        self.head = head
        self.polar = polar
        self.use_grr = use_grr
        self.use_ga = use_ga
        self.use_rot_branch = use_rot_branch
        self.vfe = StackedVFE()
        self.middle = PDConvMiddleLayers() if use_pdconv else ConvMiddleLayers()
        self.rpn = RPN(num_anchors_per_loc=len(config.ANCHOR_ROTATIONS)) if head == "anchor" else \
            RPNCenterHead(use_fg_head=use_fg_head, use_stage2=use_stage2, use_rot_branch=use_rot_branch,
                          fg_range_cond=fg_range_cond, polar=polar,
                          use_rs_conv=use_rs_conv, use_feat_undistort=use_feat_undistort)
        self.grid_size = config.POLAR_GRID_SIZE if polar else config.GRID_SIZE  # (W',H',D')
        # PARTNER(arXiv:2308.03982) GRR - polar feat2d(B,128,R,A)를 RPNBackbone 이전에
        # 재정렬하는 전처리 블록, 입출력 shape 동일(partner_grr.py 참고). Cartesian 경로는
        # 건드리지 않는다(use_grr=False가 기본, 순수 polar baseline과 나란히 비교하려고
        # 기존 가중치/체크포인트 구조를 안 바꿈).
        # n_rep(대표 feature 수)은 GRR의 range축 압축 해상도를 결정한다 - R=102를 N개로
        # 압축하므로 유효 range 해상도 = R/N (N=4면 셀당 ~2.55m, 근거리 버킷 0-2/2-2.5/2.5-3m를
        # 구분 못 함). grr_n으로 실험적으로 키운다(project_grr_collapse_diagnosis의 radius축
        # 희석 진단 참고). 파라미터 shape엔 영향 없음(순수 런타임 값) - 단 계산이 달라지므로
        # 체크포인트 payload에 grr_n을 저장해 eval 때 같은 값으로 복원해야 한다.
        self.grr_n = grr_n if grr_n is not None else config.PARTNER_GRR_N
        self.grr = GRRModule(
            channels=config.RPN_IN_CHANNELS, r_bins=config.POLAR_R_BINS,
            theta_bins=config.POLAR_THETA_BINS, r_range=config.POLAR_R_RANGE,
            theta_range_deg=config.POLAR_THETA_RANGE_DEG, n_rep=self.grr_n,
            filter_window=config.PARTNER_GRR_FILTER_WINDOW, window_a=config.PARTNER_GRR_WINDOW_A,
        ) if use_grr else None
        # GA(PARTNER §3.4) - GRR과 orthogonal, 파이프라인에서 GRR 뒤 위치. 지금은 minimal
        # v1 (aux fg/offset supervision 없이 self-contained geometry-aware refinement),
        # zero-init residual gate로 삽입 안전(polar_ga.py 참고).
        self.ga = GAModule(
            channels=config.RPN_IN_CHANNELS, r_bins=config.POLAR_R_BINS,
            theta_bins=config.POLAR_THETA_BINS, r_range=config.POLAR_R_RANGE,
            theta_range_deg=config.POLAR_THETA_RANGE_DEG,
        ) if use_ga else None

    def forward(self, voxel_features, num_points, coords, return_foreground: bool = False):
        """voxel_features: (K_total,T,7), num_points: (K_total,), coords: (K_total,4)
        [batch_idx,z,y,x] - 여러 샘플을 K축으로 이어붙인(concat) 배치. polar=True면 coords의
        y,x 자리는 각각 r_idx,theta_idx(voxelize_polar() 참고) - 축 의미만 다를 뿐 인덱싱
        코드 자체는 동일하게 동작한다.
        head='anchor': cls (B,A,H'',W''), reg (B,A*12,H'',W'')
        head='center': (heatmap, offset, z, dim, rot, density[, fg_logit]) 튜플, 전부 (B,C,H'',W'')
        return_foreground=True는 head='center'+use_fg_head=True 조합에서만 유효(RPNCenterHead.forward 참고)."""
        voxelwise = self.vfe(voxel_features, num_points)  # (K_total,128)

        B = int(coords[:, 0].max().item()) + 1 if len(coords) else 1
        Wp, Hp, Dp = self.grid_size
        dense = voxelwise.new_zeros(B, 128, Dp, Hp, Wp)
        if len(coords):
            b, z, y, x = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
            dense[b, :, z, y, x] = voxelwise

        mid = self.middle(dense)  # (B,64,D'',H',W')
        B_, C, D_, H_, W_ = mid.shape
        feat2d = mid.reshape(B_, C * D_, H_, W_)  # 논문 §3.1 "reshaping"
        if self.grr is not None:
            feat2d = self.grr(feat2d)
        if self.ga is not None:
            feat2d = self.ga(feat2d)
        if self.head == "center":
            return self.rpn(feat2d, return_foreground=return_foreground,
                             dense=dense if self.use_rot_branch else None)
        return self.rpn(feat2d)
