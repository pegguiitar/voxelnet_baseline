"""foreground_gate.py - Direction 3(reports_v2/foreground_aux_branch_proposal.md 4.7) 지원 코드.

GT box로부터 BEV foreground target을 만들고, "joint"/"freeze_fit"/"joint_ema" 세 variant를
위한 유틸을 모은다. 의도적으로 "확률"이라는 표현을 안 쓴다 - fg_gate는 heatmap loss의 negative
항을 조절하는 학습된 gating 신호일 뿐이다(추후 베이지안 시간축 결합을 시도할 때 그 표현을
다시 꺼낸다). 지리적(GT 근처) 제약은 의도적으로 없다 - fg_gate가 높은 곳이면 어디든
negative penalty를 완화한다(불필요한 이중 제약이라 판단해 뺌, 자세한 논의는 proposal 4.7).

joint_ema: joint 20epoch 결과가 decay 구간 recall 보존에서 freeze_fit보다 뚜렷이 나았지만
(late-epoch recall(IoU0.25) 평균 0.936 vs 0.857 vs baseline 0.814), box-fit 정밀도(flat
AP3D(0.5))는 freeze_fit이 더 나았다 - joint의 raw per-step gate가 매 스텝 노이즈를 그대로
box regression head에 흘려보내는 게 원인일 수 있다는 가설로, "joint처럼 매 스텝 학습은
계속하되 gating에 실제로 쓰는 신호는 EMA로 부드럽게" 만드는 절충안.
"""

import copy

import torch
import torch.nn.functional as F

import config


def local_density_reliability(coords: torch.Tensor, num_points: torch.Tensor,
                               batch_size: int, h: int, w: int) -> torch.Tensor:
    """2026-08-19 range-bucket 분석(proposal 4.7.1/7.4) - fg_gate의 오탐이 "관측 안 됨"이
    아니라 "관측은 됐는데 신호가 애매함"(2.5-3.5m 집중)에서 온다는 걸 확인 - fg_gate 자체를
    재학습시키는 대신, 이미 있는 point density만으로 만드는 저비용 reliability factor.

    coords: (K,4) [batch_idx, z_idx, y_idx, x_idx], voxelize.py 관례로 VOXEL_SIZE(0.1m)
    해상도. num_points: (K,) 그 voxel 안 실제 point 개수. h,w: heatmap/BEV 해상도
    (ANCHOR_STRIDE=0.2m=VOXEL_SIZE*2, ANCHOR_GRID_SIZE=GRID_SIZE//2로 정확히 2배 관계라
    y_idx//2, x_idx//2로 단순 집계 가능 - config.py 확인됨).

    반환: (B,1,H,W) in [0,1] - 배치별 최댓값으로 정규화(saturating). Direction 3의 fg_gate
    와는 완전히 별개 텐서라 fg_head/probe 학습에 전혀 영향 없음(곱셈 factor로만 씀,
    center_loss.gaussian_focal_loss의 fg_reliability 참고)."""
    device = coords.device
    flat = torch.zeros(batch_size * h * w, device=device, dtype=torch.float32)
    if len(coords):
        b_idx = coords[:, 0].long()
        y_idx = (coords[:, 2] // 2).long().clamp(0, h - 1)
        x_idx = (coords[:, 3] // 2).long().clamp(0, w - 1)
        flat_idx = b_idx * (h * w) + y_idx * w + x_idx
        flat.scatter_add_(0, flat_idx, num_points.float())
    density = flat.view(batch_size, 1, h, w)
    norm = density.amax(dim=(2, 3), keepdim=True).clamp_min(1.0)
    return (density / norm).clamp(0.0, 1.0)


_INTENSITY_RANGE_EDGES = (2.0, 2.5, 3.0, 3.5, 5.0)  # eval_voxelnet_by_range.BUCKETS와 동일 경계, 6구간


def local_intensity_reliability(voxel_features: torch.Tensor, coords: torch.Tensor,
                                 num_points: torch.Tensor, fg_target: torch.Tensor,
                                 batch_size: int, h: int, w: int, scale: float = None,
                                 variant: str = "mean") -> torch.Tensor:
    """density가 fg_gate와 같은 "뭉쳐 보임" 단서를 재사용해 순환 논리였던 것(compactness도
    마찬가지, proposal 7.5)과 달리, intensity는 이 프로젝트가 이미 확인한 TVG
    over-compensation 때문에 range의 단순 대리 신호가 아니다(raw intensity가 range에
    따라 오히려 증가). 로컬 검증(analyze_intensity_by_range.py, 300프레임): GT-positive
    셀이 배경 셀보다 같은 range bucket 안에서 유의미하게 강하게 반사됨, **2.5-3.5m
    (compactness가 무너진 바로 그 구간)에서도 이 격차가 살아있음**.

    variant="mean"(기본, 원래 설계, proposal §7.6) - 셀 집계·baseline 둘 다 mean.
    2026-08-20 3-seed 실측(+15.6%/1-seed였던 s0 포함 - s1/s2는 §7.7 이후 재검증)
    기준 지금까지 실전에서 가장 좋은 버전.

    variant="max_median" - 셀 집계 max, baseline median(proposal §7.6 로컬 검증에서
    gap이 3~6배 커지고 5m+ 역전도 사라진 버전). **주의: 로컬 검증(300프레임 집계,
    수천 개 배경 셀)과 달리 실제 학습 중엔 매 스텝 배치(4프레임) 안에서만 median을
    계산해서 표본이 적다 - 3-seed 실측(§7.7)에서 3-seed 전부 baseline보다 나쁘게
    나왔다(-25.6%/-4.7%/-10.5%), mean/mean보다도 나쁨. 표본 부족 시 mean 폴백이나
    배치 간 EMA 같은 보정 없이는 추천하지 않음 - 기록/재실험용으로만 남겨둠.**

    voxel_features: (K,T,7) [x,y,z,intensity,...]. coords: (K,4) [batch,z,y,x].
    num_points: (K,). fg_target: (B,1,H,W) {0,1} - 이 배치의 GT-positive 셀(foreground_gate.
    fg_target_bev 출력) - baseline은 배경(non-GT) 셀만으로 계산해야 진짜 diver 신호가
    baseline을 오염시키지 않는다. scale: None이면 variant별 관측된 gap 크기에 맞는
    기본값 사용(mean=0.02, max_median=0.05) - 명시하면 그 값을 그대로 씀.

    반환: (B,1,H,W) in [0,1] - 같은 배치·같은 range bucket 배경 셀 대비 이 셀이
    얼마나 더 강한지(고정 테이블이 아니라 매 스텝 그 배치에서 즉석 계산돼 씬마다
    다른 clutter 조건에 자동 적응)."""
    assert variant in ("mean", "max_median")
    if scale is None:
        scale = 0.02 if variant == "mean" else 0.05
    device = voxel_features.device
    K, T, _ = voxel_features.shape
    n_cells = batch_size * h * w

    valid = (torch.arange(T, device=device).unsqueeze(0) < num_points.unsqueeze(1))  # (K,T)
    b_idx = coords[:, 0].long()
    y_idx = (coords[:, 2] // 2).long().clamp(0, h - 1)
    x_idx = (coords[:, 3] // 2).long().clamp(0, w - 1)
    flat_idx_per_voxel = b_idx * (h * w) + y_idx * w + x_idx  # (K,)
    flat_idx = flat_idx_per_voxel.unsqueeze(1).expand(-1, T)[valid]  # (N_valid,)
    intensity = voxel_features[..., 3][valid]

    count = torch.zeros(n_cells, device=device)
    count.scatter_add_(0, flat_idx, torch.ones_like(intensity))
    has_point = count > 0
    if variant == "mean":
        sum_i = torch.zeros(n_cells, device=device)
        sum_i.scatter_add_(0, flat_idx, intensity)
        cell_intensity = (sum_i / count.clamp_min(1.0)).view(batch_size, 1, h, w)
    else:
        cell_max = torch.zeros(n_cells, device=device)
        cell_max.scatter_reduce_(0, flat_idx, intensity, reduce="amax", include_self=False)
        cell_intensity = cell_max.view(batch_size, 1, h, w)

    xx, yy = cell_centers_xy(h, w, device=device)  # (H,W)
    rr = torch.sqrt(xx ** 2 + yy ** 2)
    edges = torch.tensor(_INTENSITY_RANGE_EDGES, device=device)
    bucket_idx = torch.bucketize(rr, edges)  # (H,W), 값 0~5(6구간)
    n_buckets = len(_INTENSITY_RANGE_EDGES) + 1
    bucket_idx_flat = bucket_idx.view(1, -1).expand(batch_size, -1).reshape(-1)  # (B*H*W,)
    batch_of_cell = torch.arange(batch_size, device=device).view(-1, 1).expand(-1, h * w).reshape(-1)
    combined_idx = batch_of_cell * n_buckets + bucket_idx_flat  # (B*H*W,) in [0, B*n_buckets)

    is_bg = (fg_target.view(-1) < 0.5) & has_point  # (B*H*W,)
    n_combined = batch_size * n_buckets
    vals = cell_intensity.view(-1)
    if variant == "mean":
        baseline_sum = torch.zeros(n_combined, device=device)
        baseline_count = torch.zeros(n_combined, device=device)
        baseline_sum.scatter_add_(0, combined_idx[is_bg], vals[is_bg])
        baseline_count.scatter_add_(0, combined_idx[is_bg], torch.ones_like(vals[is_bg]))
        baseline = (baseline_sum / baseline_count.clamp_min(1.0))  # (B*n_buckets,)
    else:
        # median은 scatter_reduce에 없어 그룹(batch x bucket, <=수십개) 단위 파이썬
        # 루프로 계산한다 - 그룹 수가 H*W보다 훨씬 적어(예: batch_size=4, n_buckets=6
        # -> 24개) 매 스텝 오버헤드가 무시할 만하다.
        baseline = torch.zeros(n_combined, device=device)
        for g in torch.unique(combined_idx[is_bg]):
            g = int(g.item())
            group_vals = vals[is_bg & (combined_idx == g)]
            baseline[g] = group_vals.median()

    baseline_per_cell = baseline[combined_idx].view(batch_size, 1, h, w)
    reliability = ((cell_intensity - baseline_per_cell) / scale).clamp(0.0, 1.0)
    return reliability


def cell_centers_xy(h: int, w: int, device=None) -> tuple:
    """(H,W) heatmap grid 각 셀의 world (x,y) 중심 - torch 버전. analyze_recall_collapse.py의
    numpy 버전(cell_centers_xy)과 동일 계산이지만, 학습 루프 안에서 GPU 텐서로 바로 쓰기
    위해 별도 구현(그쪽은 오프라인 분석 스크립트라 numpy로 남겨둠)."""
    pc = config.POINT_CLOUD_RANGE
    sx, sy = (pc[3] - pc[0]) / w, (pc[4] - pc[1]) / h
    xs = pc[0] + (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * sx
    ys = pc[1] + (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * sy
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # 각 (H,W)
    return xx, yy


def fg_target_bev(gt_boxes_list: list, h: int, w: int, device=None) -> torch.Tensor:
    """gt_boxes_list: collate_fn(dataset.py)이 만든 리스트(길이 B), 각 원소 (M_b,13)
    [x,y,z,l,w,h,theta_z,6D]. -> (B,1,H,W) {0,1} binary foreground target - GT box의 2D BEV
    footprint(z-yaw만 사용, heatmap_targets.build_heatmap_targets_polar의 2D 라벨과 동일
    로직) 안이면 1. GT box 개수(프레임당 1~2개)가 워낙 적어 Python 루프로도 비용이
    무시할 만하다(벡터화가 필요한 건 voxel/point 개수가 큰 연산뿐)."""
    xx, yy = cell_centers_xy(h, w, device=device)  # (H,W) each
    b_size = len(gt_boxes_list)
    target = torch.zeros(b_size, 1, h, w, device=device)
    for b, gt_boxes in enumerate(gt_boxes_list):
        if gt_boxes is None or len(gt_boxes) == 0:
            continue
        gt_boxes = gt_boxes.to(device=device, dtype=torch.float32)
        for row in gt_boxes:
            cx, cy, l, wd, theta = row[0], row[1], row[3], row[4], row[6]
            c, s = torch.cos(theta), torch.sin(theta)
            dx, dy = xx - cx, yy - cy
            lx = c * dx + s * dy
            ly = -s * dx + c * dy
            inside = (lx.abs() <= l / 2) & (ly.abs() <= wd / 2)
            target[b, 0] = torch.logical_or(target[b, 0].bool(), inside).float()
    return target


class GatingProbe(torch.nn.Module):
    """freeze_fit variant 전용 - model.py의 ForegroundHead와 구조는 동일(작은 1x1 conv
    스택)하지만 완전히 별개 인스턴스다 - 메인 model.parameters()에 절대 포함되지 않고,
    train.py가 주기적으로 별도 optimizer로 짧게(refit_probe) 재학습시킨다."""

    def __init__(self, in_channels: int, hidden: int = 64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden, 1), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden, 1, 1))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


def refit_probe(probe: torch.nn.Module, feats: torch.Tensor, targets: torch.Tensor,
                 n_steps: int = 300, lr: float = 0.01) -> None:
    """probe(이미 새로 초기화된 상태여야 함 - reinit은 호출측 책임)를 feats/targets로
    처음부터 다시 짧게 fit한다 - Stage 0 probe(analyze_recall_collapse.train_probe_and_auc)와
    동일하게 "매번 독립적으로 짧게" 학습시키는 게 핵심(heatmap_head가 겪은 것 같은 누적된
    긴 학습 경로 문제를 구조적으로 피하려는 의도). in-place로 probe 파라미터를 덮어쓴다."""
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    n_pos = targets.sum().clamp_min(1)
    n_neg = (targets.numel() - targets.sum()).clamp_min(1)
    pos_weight = (n_neg / n_pos).clamp(max=50.0)
    for _ in range(n_steps):
        opt.zero_grad()
        logit = probe(feats)
        loss = F.binary_cross_entropy_with_logits(logit, targets, pos_weight=pos_weight)
        loss.backward()
        opt.step()


def blend_state_dicts(old_sd: dict, new_sd: dict, alpha: float) -> dict:
    """alpha=0 -> old_sd 그대로, alpha=1 -> new_sd 그대로. refit 직후 alpha를 0에서 1로
    서서히(정확히 refit_every epoch에 걸쳐, 다음 refit 시점에 딱 1.0에 도달하도록) 올려서
    갑자기 gating 신호가 바뀌는 충격을 피한다 - blend 기간을 refit 주기와 다르게 잡으면
    다음 refit이 이전 blend가 안 끝난 상태에서 겹쳐버리므로(사용자 지적), 항상
    blend_epochs == refit_every로 맞춘다(train.py에서 강제)."""
    return {k: (1 - alpha) * old_sd[k] + alpha * new_sd[k] for k in old_sd}


class EMAShadow:
    """joint_ema variant 전용 - `model.rpn.fg_head`(live, 매 스텝 자기 BCE loss로 backprop
    되는 본체, joint와 동일)의 파라미터를 느리게 따라가는 shadow 복사본.
    `gaussian_focal_loss`에 실제로 들어가는 gate는 이 shadow의 출력만 쓴다 - freeze_fit의
    "천천히 blend"를 continuous 학습 위에서 재현하려는 절충안(전체 재초기화+짧은 재학습
    없이, 매 스텝 조금씩 EMA로만 따라감). live fg_head 자체의 학습(및 자기 BCE loss)은
    joint와 완전히 동일 - shadow는 그 결과를 부드럽게 소비만 할 뿐 별도 최적화 대상이
    아니다(shadow 파라미터는 requires_grad=False, update()도 no_grad)."""

    def __init__(self, module: torch.nn.Module, decay: float = 0.999):
        self.shadow = copy.deepcopy(module).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, module: torch.nn.Module) -> None:
        for s_p, p in zip(self.shadow.parameters(), module.parameters()):
            s_p.mul_(self.decay).add_(p, alpha=1 - self.decay)
