"""partner_grr.py - PARTNER(Nie et al., arXiv:2308.03982v2)의 GRR(Global Representation
Re-alignment) 모듈. Polar 좌표계 feature map(B,C,R,A)의 range(R)축이 표준 컨볼루션엔
불리한 비균일 해상도를 갖는다는 문제를, "열(azimuth column)마다 대표 feature 몇 개로
압축 -> 이웃 각도끼리 attention -> 원해상도로 복원" 3단계로 완화한다. 입출력 shape이
완전히 동일해(B,C,R,A) RPNBackbone 직전에 끼워 넣는 전처리 블록으로 동작한다.

논문 Eq.1-4(Condense Attention)를 그대로 따르되, 두 지점은 우리 데이터에 맞게 명시적으로
각색했다:
  1. Angular Attention의 shifted window - 원안은 360도 전방위 lidar를 가정해 azimuth
     축을 원형(circular)으로 다뤄 torch.roll로 shift한다. 우리 소나는
     config.SONAR_AZIMUTH_LIMIT_DEG(=45도)의 유한 FOV라 azimuth 양 끝이 물리적으로
     이어져 있지 않으므로, zero-padding 기반 shift로 바꿨다.
  2. 위치 인코딩 E(p_i,p'_i) - 원 논문은 attention 항 안에서의 정확한 결합 방식을
     명시하지 않는다. 여기서는 p_i=극좌표(r,theta), p'_i=같은 셀의 데카르트좌표(x,y)로
     해석하고, attention-weighted V에 위치 인코딩을 더하는 표준적 형태(Shaw et al. 2018
     relative position 방식)로 구현했다 - study_v2/grr_and_model_architecture_study.html
     Level 4에 상세 근거.

N(대표 feature 수)=4, S(1D max filter window)=3, W_a(angular window)=8은 논문 원안
기본값 - Waymo(R≈1155)에서 검증됐고 N에 대한 ablation이 없어, R=102인 우리 스케일에서
최적인지는 미검증(config.py PARTNER_GRR_* 주석 참고). 1차 실험은 원안 기본값 그대로 쓴다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def polar_cell_pos_diff(r_bins: int, theta_bins: int, r_range, theta_range_deg) -> torch.Tensor:
    """각 (r_idx,theta_idx) 셀의 극좌표 p=(r,theta_rad)와 그 물리적 위치의 데카르트좌표
    p'=(x,y)의 차이 p-p'를 미리 계산해둔다(둘 다 미터 단위 정도 스케일이라 뺄셈이
    의미를 가짐) - PARTNER Eq.4의 E(p_i,p'_i) 입력. 격자 기하가 학습 중 안 바뀌므로
    forward마다 재계산할 필요 없이 buffer로 등록해 재사용한다. -> (R,A,2)."""
    r_min, r_max = r_range
    theta_min_deg, theta_max_deg = theta_range_deg
    r_centers = r_min + (torch.arange(r_bins, dtype=torch.float32) + 0.5) / r_bins * (r_max - r_min)
    theta_centers_deg = theta_min_deg + (torch.arange(theta_bins, dtype=torch.float32) + 0.5) / theta_bins \
        * (theta_max_deg - theta_min_deg)
    theta_centers_rad = torch.deg2rad(theta_centers_deg)

    r_grid = r_centers[:, None].expand(r_bins, theta_bins)          # (R,A)
    theta_grid = theta_centers_rad[None, :].expand(r_bins, theta_bins)  # (R,A) radians

    x = r_grid * torch.cos(theta_grid)
    y = r_grid * torch.sin(theta_grid)
    polar_p = torch.stack([r_grid, theta_grid], dim=-1)  # (R,A,2)
    cart_p = torch.stack([x, y], dim=-1)                 # (R,A,2)
    return polar_p - cart_p  # (R,A,2)


class CondenseAttention(nn.Module):
    """PARTNER Eq.1-4. 열(azimuth column)마다 R개 위치 중 top-N을 골라 query로 삼고,
    열 전체(R개)를 key/value로 cross-attention -> N개 대표 feature로 압축."""

    def __init__(self, channels: int, n_rep: int, filter_window: int, pos_diff: torch.Tensor):
        super().__init__()
        self.channels = channels
        self.n_rep = n_rep
        self.score_conv = nn.Conv2d(channels, 1, 1)
        self.max_filter = nn.MaxPool2d(kernel_size=(filter_window, 1), stride=1,
                                        padding=(filter_window // 2, 0))
        self.w_q = nn.Linear(channels, channels)
        self.w_k = nn.Linear(channels, channels)
        self.w_v = nn.Linear(channels, channels)
        self.w_pos = nn.Linear(2, channels)
        self.scale = channels ** -0.5
        self.norm = nn.LayerNorm(channels)
        self.register_buffer("pos_diff", pos_diff, persistent=False)  # (R,A,2)

    def forward(self, feat: torch.Tensor):
        """feat: (B,C,R,A) -> f_rep: (B,N,A,C), topk_idx: (B,N,A) (디버그/역참조용)."""
        B, C, R, A = feat.shape

        score = self.score_conv(feat)                 # (B,1,R,A)
        score = self.max_filter(score)[:, :, :R, :]    # 패딩으로 R+1 나오면 앞쪽만 사용
        score = score.squeeze(1)                       # (B,R,A)
        _, topk_idx = torch.topk(score, self.n_rep, dim=1)  # (B,N,A) - R축 인덱스

        feat_col = feat.permute(0, 3, 2, 1).reshape(B * A, R, C)      # (B*A,R,C) - 열별 시퀀스
        idx_flat = topk_idx.permute(0, 2, 1).reshape(B * A, self.n_rep)  # (B*A,N)

        candidates = torch.gather(feat_col, 1, idx_flat.unsqueeze(-1).expand(-1, -1, C))  # (B*A,N,C)

        Q = self.w_q(candidates)   # (B*A,N,C)
        K = self.w_k(feat_col)     # (B*A,R,C)
        V = self.w_v(feat_col)     # (B*A,R,C)
        attn = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)  # (B*A,N,R)
        context = attn @ V  # (B*A,N,C)

        pos_enc_full = F.relu(self.w_pos(self.pos_diff.permute(1, 0, 2)))  # (A,R,C)
        pos_enc_full = pos_enc_full.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * A, R, C)
        pos_enc = torch.gather(pos_enc_full, 1, idx_flat.unsqueeze(-1).expand(-1, -1, C))  # (B*A,N,C)

        f_rep = self.norm(context + pos_enc).reshape(B, A, self.n_rep, C).permute(0, 2, 1, 3)  # (B,N,A,C)
        return f_rep, topk_idx


class AngularAttentionLayer(nn.Module):
    """압축 표현 F^rep(B,N,A,C)에 대해 azimuth(A)축을 폭 window_a로 나눠 윈도우 내부
    self-attention(윈도우 하나에 N*window_a개 토큰). shift>0이면 shifted-window(Swin
    스타일) - 원 논문은 360도 lidar라 circular roll을 쓰지만, 우리 소나는 유한 FOV
    (양 끝이 안 이어짐)라 zero-padding으로 shift한다."""

    def __init__(self, channels: int, window_a: int, shift: int = 0):
        super().__init__()
        self.window_a = window_a
        self.shift = shift
        self.w_q = nn.Linear(channels, channels)
        self.w_k = nn.Linear(channels, channels)
        self.w_v = nn.Linear(channels, channels)
        self.scale = channels ** -0.5
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,N,A,C) -> (B,N,A,C)."""
        B, N, A, C = x.shape
        Wa = self.window_a

        x_in = x
        if self.shift:
            x_in = F.pad(x_in, (0, 0, self.shift, 0))  # A축 앞단에 zero-pad(비순환 shift)
        A_ext = x_in.shape[2]
        pad_a = (Wa - A_ext % Wa) % Wa
        x_in = F.pad(x_in, (0, 0, 0, pad_a))
        A_pad = x_in.shape[2]
        num_windows = A_pad // Wa

        xw = x_in.reshape(B, N, num_windows, Wa, C).permute(0, 2, 1, 3, 4)  # (B,num_windows,N,Wa,C)
        xw = xw.reshape(B * num_windows, N * Wa, C)

        Q, K, V = self.w_q(xw), self.w_k(xw), self.w_v(xw)
        attn = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        out = self.norm(attn @ V + xw)  # residual + LN (표준 transformer block 관례로 보강)

        out = out.reshape(B, num_windows, N, Wa, C).permute(0, 2, 1, 3, 4).reshape(B, N, A_pad, C)
        if self.shift:
            out = out[:, :, self.shift:self.shift + A, :]
        else:
            out = out[:, :, :A, :]
        return out


class ReverseCondenseAttention(nn.Module):
    """Condense Attention과 대칭적인 cross-attention: 원본 R개 위치 전부가 query,
    압축된 N개 대표 feature가 key/value. 출력은 (B,C,R,A) 원해상도로 복원된다.

    LayerNorm은 PARTNER 원 논문 Eq.3-4에 없다(Condense Attention과 마찬가지로 논문
    수식 그대로 구현하면 정규화가 없다 - study_v2 학습자료의 "알려진 불확실성"
    참고). 그런데도 여기 추가한 이유: voxelnet_center_polar_grr_s0(2026-08-15, peak
    LR=0.01) 학습에서 이 모듈 출력이 RPNBackbone 첫 BatchNorm으로 바로 들어가며
    running_var가 5.2e4->1.6e9로 폭주하는 걸 체크포인트 비교로 확인했다(project_grr_
    collapse_diagnosis 메모리) - 논문이 검증한 R≈1155 스케일과 달리 R=102인 우리
    스케일+짧은 학습(16-20epoch)에서 이 bespoke·무정규화 설계가 불안정했다는 뜻이라,
    논문 재현이 아니라 우리 스케일에 맞춘 명시적 각색으로 정규화를 추가한다."""

    def __init__(self, channels: int, pos_diff: torch.Tensor):
        super().__init__()
        self.channels = channels
        self.w_q = nn.Linear(channels, channels)
        self.w_k = nn.Linear(channels, channels)
        self.w_v = nn.Linear(channels, channels)
        self.w_pos = nn.Linear(2, channels)
        self.scale = channels ** -0.5
        self.norm = nn.LayerNorm(channels)
        self.register_buffer("pos_diff", pos_diff, persistent=False)  # (R,A,2)

    def forward(self, feat_full: torch.Tensor, f_rep: torch.Tensor) -> torch.Tensor:
        """feat_full: (B,C,R,A) - query 소스(원해상도). f_rep: (B,N,A,C) - key/value(압축
        표현). -> (B,C,R,A)."""
        B, C, R, A = feat_full.shape
        N = f_rep.shape[1]

        feat_col = feat_full.permute(0, 3, 2, 1).reshape(B * A, R, C)  # (B*A,R,C)
        rep_col = f_rep.permute(0, 2, 1, 3).reshape(B * A, N, C)       # (B*A,N,C)

        Q = self.w_q(feat_col)
        K = self.w_k(rep_col)
        V = self.w_v(rep_col)
        attn = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)  # (B*A,R,N)
        context = attn @ V  # (B*A,R,C)

        pos_enc = F.relu(self.w_pos(self.pos_diff.permute(1, 0, 2)))  # (A,R,C)
        pos_enc = pos_enc.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * A, R, C)

        out = self.norm(context + pos_enc).reshape(B, A, R, C).permute(0, 3, 2, 1)  # (B,C,R,A)
        return out


class GRRModule(nn.Module):
    """Condense Attention -> Angular Attention(x2, 두번째는 shifted window) -> Reverse
    Condense Attention. 입출력 shape이 동일(B,C,R,A)한 전처리 블록 - RPNBackbone 이전에
    삽입된다.

    PARTNER 원 논문은 이 출력이 원본 feat를 완전히 대체한다(residual 없음, study_v2
    학습자료 참고). 여기서는 학습 초반 안정성을 위해 **residual + zero-init learnable
    gate**(LayerScale/CaiT류 관례)로 바꿨다 - `residual_gate`가 0으로 시작해 학습
    극초반엔 GRR이 사실상 no-op(출력=feat 그대로)이었다가, gradient가 흐르면서
    점진적으로 기여도를 키운다. 이것도 논문 복원이 아니라 project_grr_collapse_
    diagnosis에서 확인된 불안정성(BN running_var 폭주)에 대한 우리 쪽 명시적 각색."""

    def __init__(self, channels: int, r_bins: int, theta_bins: int, r_range, theta_range_deg,
                 n_rep: int = 4, filter_window: int = 3, window_a: int = 8):
        super().__init__()
        pos_diff = polar_cell_pos_diff(r_bins, theta_bins, r_range, theta_range_deg)
        self.condense = CondenseAttention(channels, n_rep, filter_window, pos_diff)
        self.angular1 = AngularAttentionLayer(channels, window_a, shift=0)
        self.angular2 = AngularAttentionLayer(channels, window_a, shift=window_a // 2)
        self.reverse = ReverseCondenseAttention(channels, pos_diff)
        self.residual_gate = nn.Parameter(torch.zeros(1))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        f_rep, _ = self.condense(feat)
        f_rep = self.angular1(f_rep)
        f_rep = self.angular2(f_rep)
        delta = self.reverse(feat, f_rep)
        return feat + self.residual_gate * delta
