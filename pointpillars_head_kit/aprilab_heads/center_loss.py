"""center_loss.py - CenterPoint/CenterNet 스타일 loss. heatmap_targets.py가 만든
타겟과 짝을 이룬다.

분류: Gaussian-penalty-reduced focal loss(Law&Deng CornerNet 2018, Zhou et al.
CenterNet 2019에서 정착된 형태) - focal loss의 직접 후예(정밀도 격차 분석
reports/precision_gap_analysis.html §B3의 focal loss 계보 참고)를 heatmap
회귀에 맞게 확장한 버전. Positive(피크)뿐 아니라 negative 쪽도 Gaussian 값 자체로
페널티를 깎아준다 - 피크 바로 옆 셀은 "완전히 틀린 negative"가 아니라 "거의 맞는
근처"로 취급.

회귀(offset/z/dim/rot): 피크 셀 하나에서만 L1 loss, anchor 기반 SmoothL1과 달리
anchor 없이 절대값을 직접 맞춘다.
"""

import torch
import torch.nn.functional as F

import config
import rotation3d


def gaussian_focal_loss(pred_logit: torch.Tensor, target: torch.Tensor,
                         alpha: float = 2.0, beta: float = 4.0, pos_weight: float = 1.0,
                         fg_gate: torch.Tensor = None, fg_alpha_relief: float = 1.0,
                         fg_r_max: float = 0.5, fg_reliability: torch.Tensor = None,
                         range_weight_map: torch.Tensor = None) -> torch.Tensor:
    """pred_logit/target: (B,1,H,W). target은 0~1 Gaussian(피크=1). CenterNet 표준 정규화:
    양성(피크) 개수로 나눈다 - 최소 1로 clamp.

    log(p)/log(1-p)는 sigmoid(logit)을 clamp한 뒤 log를 씌우지 않고 F.logsigmoid로
    직접 계산한다 - 처음엔 clamp(1e-4,1-1e-4) 방식으로 짰는데, 3프레임 overfit
    테스트에서 hm_loss가 정확히 같은 값(9.208498...)으로 90 step 동안 안 움직이는
    걸 발견했다 - 예측이 자신 있어질수록(overfit 후반부 당연한 현상) sigmoid 출력이
    clamp 경계에 붙어버리고, clamp는 그 경계에서 gradient가 정확히 0이라 거기서부터
    학습이 완전히 멈춘다. logsigmoid는 내부적으로 log-sum-exp 안정화를 쓰기 때문에
    같은 수치적 문제 없이 gradient가 끝까지 살아있다.

    pos_weight(기본 1.0=no-op): positive 항에만 곱하는 스칼라 - alpha/beta와 달리
    이 loss엔 원래 RetinaNet식 pos/neg class-balance 가중치가 없었다(alpha는 pos/neg
    양쪽에 대칭 적용되는 focusing 지수일 뿐). recall-collapse 완화용으로 학습 후반
    구간에만 스케줄링해서 곱하려는 용도(train.py의 pos_weight_at_epoch 참고) - config.py의
    GRAD_CLIP_NORM=35.0가 추가된 계기였던 range-weight loss 폭주 사고(loss 항을 스칼라로
    키웠다가 극초반 큰 gradient로 불안정해진 사례)와 같은 위험군이라, 호출 쪽에서 완만한
    ramp + 보수적 상한을 지키는 게 전제.

    fg_gate(기본 None=no-op): Direction 3 - foreground auxiliary head가 낸 P_fg(B,1,H,W),
    이미 sigmoid 통과했고 호출 전에 반드시 .detach()된 상태여야 한다(gate가 메인 loss로
    직접 학습되는 걸 막음 - fg_head는 자기 자신의 별도 loss로만 학습). 지리적(GT 근처)
    제약 없이 P_fg가 높은 곳이면 어디든 negative penalty를 완화한다 - "GT 근처만" 이라는
    이중 제약을 넣었던 이전 설계는 불필요한 것으로 판단해 뺐다(자세한 논의는
    reports_v2/foreground_aux_branch_proposal.md 4.7). relief는 [0, fg_r_max]로 상한을
    둬서 penalty를 완전히 0으로 만들지 않는다.

    fg_reliability(기본 None=no-op): 2026-08-19 range-bucket 분석(4.7.1)에서 fg_gate의
    오탐이 "관측 안 됨"이 아니라 "관측은 됐는데 신호가 애매함"(2.5-3.5m 특정 구간에서
    집중)인 걸 확인 - fg_gate 자체를 다시 학습시키는 대신, point density 기반
    reliability(B,1,H,W, [0,1], foreground_gate.local_density_reliability())를
    relief에 곱해서 "신호가 약한 곳에서는 relief를 억제"한다. fg_gate와 완전히 분리된
    별도 factor라 fg_head/probe를 전혀 안 건드림(재학습 불필요) - 자세한 논의는
    proposal 7.4."""
    pred = torch.sigmoid(pred_logit)
    pos_mask = (target == 1).float()
    neg_mask = (target < 1).float()
    neg_weight = torch.pow(1 - target, beta)
    if fg_gate is not None:
        relief = (fg_alpha_relief * fg_gate).clamp(0.0, fg_r_max)
        if fg_reliability is not None:
            relief = relief * fg_reliability
        neg_weight = neg_weight * (1.0 - relief)

    log_p = F.logsigmoid(pred_logit)
    log_1mp = F.logsigmoid(-pred_logit)

    pos_loss = -log_p * torch.pow(1 - pred, alpha) * pos_mask * pos_weight
    neg_loss = -log_1mp * torch.pow(pred, alpha) * neg_weight * neg_mask

    # range-conditioned loss weighting (2026-08-22) - 각 셀의 loss에 range-bucket 기반
    # weight 곱. train 3-3.5m 10.8% vs test 23.8% mismatch 완화 목적. sampling 대비
    # memorization 리스크 낮음 (gradient scale만 조절, 같은 프레임 여러 번 안 봄).
    # range_weight_map: (1,1,H,W) 또는 (B,1,H,W), positive 값.
    if range_weight_map is not None:
        pos_loss = pos_loss * range_weight_map
        neg_loss = neg_loss * range_weight_map

    n_pos = pos_mask.sum().clamp_min(1)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def center_voxelnet_loss(heatmap_pred, offset_pred, z_pred, dim_pred, rot_pred,
                          heatmap_target, reg_mask, offset_target, z_target, dim_target, rot_target,
                          density_pred=None, density_target=None,
                          reg_weight: float = 1.0, focal_alpha: float = 2.0, focal_beta: float = 4.0,
                          pos_weight: float = 1.0,
                          fg_gate: torch.Tensor = None, fg_alpha_relief: float = 1.0, fg_r_max: float = 0.5,
                          fg_pred: torch.Tensor = None, fg_target: torch.Tensor = None,
                          fg_weight: float = 1.0, fg_pos_weight: float = 1.0,
                          fg_reliability: torch.Tensor = None, fg_reg_alpha: float = 0.0,
                          range_weight_map: torch.Tensor = None,
                          rotation_repr: str = "6d",
                          rotation_loss_mode: str = "l1_target",
                          rotation_loss_weight: float = 1.0):
    """*_pred: 모델 출력 (B,C,H,W). *_target/reg_mask: 캐시된 타겟 (B,H,W,C)/(B,H,W) -
    permute해서 pred와 축을 맞춘다. focal_alpha/focal_beta: gaussian_focal_loss의 focusing
    지수(기본값 2.0/4.0=CornerNet/CenterNet 표준). pos_weight: gaussian_focal_loss 참고 -
    recall-collapse 완화용 positive 가중치(기본 1.0=no-op).

    density_pred/density_target: RAANet(arXiv:2111.09515)식 보조 head(project_polarization_design
    과 별개 트랙, config.DENSITY_AUX_WEIGHT=0.2). density_pred (B,3,H,W) raw logit,
    density_target (B,H,W) int64 {0,1,2} - positive cell(reg_mask)에서만 CE loss로 감독,
    나머지는 무시. 둘 다 None이면(기본) aux loss 자체를 안 더함 - 이 head 없이 학습하던
    기존 체크포인트/스크립트와 호환 유지.

    fg_gate/fg_alpha_relief/fg_r_max/fg_reliability: Direction 3 - gaussian_focal_loss로 그대로 전달(위 참고).
    fg_pred/fg_target: Direction 3의 "jointly" variant 전용 - foreground head 자체를 학습시키는
    보조 BCE loss. fg_pred (B,1,H,W) raw logit, fg_target (B,1,H,W) {0,1}. fg_weight로 총 loss에
    더해지는 비중 조절. "freeze_fit" variant는 fg_head를 메인 optimizer로 학습 안 시키므로
    (train.py에서 별도 optimizer로 주기적 재학습) fg_pred/fg_target을 안 넘기면 이 항 자체가 스킵됨.

    fg_reg_alpha(기본 0.0=no-op): Phase 1 "A2" - fg-gated regression weighting(2026-08-28). 지금까지
    fg_gate는 heatmap의 negative 항만 완화(relief)해서 box regression엔 간접적으로만 영향을 줬는데,
    Phase 0 진단에서 joint의 이득이 diffuse한 box-fit(ATE/AOE/3D IoU 소폭 개선)으로 나타났다 -
    이를 직접 증폭하려고, 각 GT peak 셀의 회귀 L1(offset/z/dim/rot)을 (1 + fg_reg_alpha * P_fg)로
    가중한다. P_fg는 detach라 fg_head로 backprop되지 않고(gradient 크기만 조절), fg_head는 여전히
    자기 BCE로만 학습된다. 정규화도 peak 셀 수(n_pos) 대신 가중치 합(w_sum)으로 바꿔 전체 reg loss
    스케일을 보존한다 - 즉 전역 증폭이 아니라 "fg 확신이 높은 셀로 emphasis 재분배". fg_reg_alpha=0
    이거나 fg_gate=None이면 정확히 기존과 동일(가중치=reg_mask, w_sum=n_pos). 상세: study_v2/
    fg_a2_regression_weighting.html."""
    hm_loss = gaussian_focal_loss(heatmap_pred, heatmap_target, alpha=focal_alpha, beta=focal_beta,
                                   pos_weight=pos_weight, fg_gate=fg_gate,
                                   fg_alpha_relief=fg_alpha_relief, fg_r_max=fg_r_max,
                                   fg_reliability=fg_reliability,
                                   range_weight_map=range_weight_map)

    def to_bhwc(x):
        return x.permute(0, 2, 3, 1)  # (B,C,H,W) -> (B,H,W,C)

    n_pos = reg_mask.float().sum().clamp_min(1)  # density aux 정규화용(변경 없음)

    # A2(2026-08-28): fg-gated regression weighting. fg_reg_alpha>0면 peak 셀 가중치를
    # (1 + fg_reg_alpha * P_fg)로 두고 정규화도 그 합으로 바꾼다(스케일 보존, 재분배). off면 기존과 동일.
    if fg_reg_alpha > 0.0 and fg_gate is not None:
        w_map = reg_mask.float() * (1.0 + fg_reg_alpha * fg_gate.squeeze(1).detach())  # (B,H,W)
        reg_cell_w = w_map.unsqueeze(-1)     # (B,H,W,1)
        reg_denom = w_map.sum().clamp_min(1.0)
    else:
        reg_cell_w = reg_mask.unsqueeze(-1).float()  # (B,H,W,1)
        reg_denom = n_pos

    offset_loss = (F.l1_loss(to_bhwc(offset_pred), offset_target, reduction="none") * reg_cell_w).sum() / reg_denom
    z_loss = (F.l1_loss(to_bhwc(z_pred), z_target, reduction="none") * reg_cell_w).sum() / reg_denom
    dim_loss = (F.l1_loss(to_bhwc(dim_pred), dim_target, reduction="none") * reg_cell_w).sum() / reg_denom

    # rotation loss - rotation_loss_mode로 계산 방식 선택.
    # 'l1_target'(기본): raw 회귀 벡터에 L1(SECOND/CenterPoint 관례, 우리가 지금까지 쓴 것)
    # 'chordal_R': raw 회귀 벡터를 R로 디코드한 뒤 rotation-matrix-space L2:
    #     ‖R_pred − R_gt‖_F² (요소별 차이 제곱합). 이 loss form은:
    #   - Zhou 2019 Sec 5.1(sanity test)의 학습 loss("L2 loss on rotation matrix")와 사실상 동일.
    #     Zhou 논문에서 6D representation이 quat/axisangle을 empirical로 이긴 실험 세팅과 정합.
    #   - 별개 문헌(Huynh 2009, "Metrics for 3D Rotations")의 명명법으로는 chordal distance²
    #     (Φ₁(R₁,R₂)=‖R₁-R₂‖_F). Zhou 논문 자체엔 "chordal" 단어는 안 나옴 - metric 이름 통일용.
    #   가설: L1-on-target에서는 6D의 continuity 이점이 활성화 안 됨(6채널로 3-DOF 회전을
    #     redundant하게 표현해서 gauge 방향에도 L1 페널티 소모). rotation-matrix-space에선
    #     같은 R로 디코드되는 여러 raw 벡터가 loss=0이라 gauge 여유가 무해.
    # 'combined': l1_target + 0.5 * chordal_R (안정성 우선 워밍업 후 rotation-space 감독 추가)
    rot_pred_bhwc = to_bhwc(rot_pred)  # (B,H,W,C)
    if rotation_loss_mode == "l1_target":
        rot_loss = (F.l1_loss(rot_pred_bhwc, rot_target, reduction="none") * reg_cell_w).sum() / reg_denom
    elif rotation_loss_mode in ("chordal_R", "combined"):
        # positive cell에서만 R decode해서 chordal loss - 나머지는 gradient 흘리지 않음.
        # reg_mask (B,H,W) bool에서 positive positions만 잘라내 (n_pos, C)로 처리하면 forward
        # 비용이 급감(전체 grid 30000+cell에서 pos ~1-2 cell / frame이라 decode도 그만큼만).
        pos_idx = reg_mask.nonzero(as_tuple=False)  # (n_pos, 3) [b,h,w]
        if pos_idx.numel() > 0:
            pp = rot_pred_bhwc[pos_idx[:, 0], pos_idx[:, 1], pos_idx[:, 2]]  # (n_pos, C)
            tt = rot_target[pos_idx[:, 0], pos_idx[:, 1], pos_idx[:, 2]]     # (n_pos, C)
            R_pred = rotation3d.decode_rotation_torch(pp, rotation_repr)     # (n_pos, 3, 3)
            R_target = rotation3d.decode_rotation_torch(tt, rotation_repr)   # (n_pos, 3, 3)
            # chordal (squared Frobenius) - geodesic(arccos)보다 gradient 안정적(no singularity).
            # 3x3 R끼리 differencing이라 스케일 ~O(1), positive cell 수로 정규화.
            # A2(fg_reg_alpha) 가중치 반영 위해 pos_idx별 weight 필요 시 향후 확장(현재는 균등).
            chordal = (R_pred - R_target).pow(2).sum(dim=(-1, -2))  # (n_pos,)
            rot_loss_chordal = chordal.sum() / reg_denom
        else:
            rot_loss_chordal = rot_pred_bhwc.sum() * 0.0  # dtype/device 유지용 no-op
        if rotation_loss_mode == "chordal_R":
            rot_loss = rot_loss_chordal * rotation_loss_weight
        else:  # combined
            rot_loss_l1 = (F.l1_loss(rot_pred_bhwc, rot_target, reduction="none") * reg_cell_w).sum() / reg_denom
            rot_loss = rot_loss_l1 + 0.5 * rot_loss_chordal * rotation_loss_weight
    else:
        raise ValueError(f"unknown rotation_loss_mode: {rotation_loss_mode}")

    reg_loss = offset_loss + z_loss + dim_loss + rot_loss
    stats = {"hm_loss": hm_loss.item(), "reg_loss": reg_loss.item(),
             "offset_loss": offset_loss.item(), "z_loss": z_loss.item(),
             "dim_loss": dim_loss.item(), "rot_loss": rot_loss.item(),
             "n_pos": int(reg_mask.sum().item())}
    if fg_reg_alpha > 0.0 and fg_gate is not None:
        stats["fg_reg_alpha"] = fg_reg_alpha

    total = hm_loss + reg_weight * reg_loss
    if density_pred is not None:
        ce = F.cross_entropy(density_pred, density_target, reduction="none")  # (B,H,W)
        density_loss = (ce * reg_mask.float()).sum() / n_pos
        total = total + config.DENSITY_AUX_WEIGHT * density_loss
        stats["density_loss"] = density_loss.item()

    if fg_pred is not None:
        fg_loss = F.binary_cross_entropy_with_logits(
            fg_pred, fg_target, pos_weight=torch.tensor(fg_pos_weight, device=fg_pred.device))
        total = total + fg_weight * fg_loss
        stats["fg_loss"] = fg_loss.item()

    return total, stats
