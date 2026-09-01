"""loss.py - 논문 Eq.2의 정규화 골격(Lcls/Npos,Nneg + Lreg/Npos)은 유지하되, Lcls을
plain weighted BCE 대신 focal loss(Lin et al. 2017, RetinaNet)로 바꿨다.

원 논문의 BCE(α/β로 pos/neg 각각 평균 정규화)는 negative 6,000개(프레임당, 우리 실측
positive 0.041%)를 전부 동일 가중치로 취급한다 - "빈 voxel"과 "다이버처럼 생긴 클러터"를
구분 못 한다. Focal loss의 (1-p_t)^gamma 항이 이미 잘 맞히는 쉬운 negative의 loss를
자동으로 짓눌러서, 학습 신호가 헷갈리는 소수 예시에 집중되게 한다 - 이 프로젝트의 실측
실패 유형(score_thresh를 0.3->0.7로 올려도 AP3D가 거의 안 변함 = FP가 confidence 꼬리의
노이즈가 아니라 구조적으로 자신 있게 틀리는 것)과 정확히 맞아떨어지는 처방이라 채택했다.
자세한 배경은 VoxelNet/reports/precision_gap_analysis.html 참고.
"""

import torch
import torch.nn.functional as F

import config


def sigmoid_focal_loss(logits: torch.Tensor, targets: torch.Tensor,
                        alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """RetinaNet(Lin et al. 2017) 표준형. logits/targets: 임의 shape, elementwise."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return alpha_t * ce * (1 - p_t).pow(gamma)


def voxelnet_loss(cls_pred: torch.Tensor, reg_pred: torch.Tensor,
                   cls_labels: torch.Tensor, reg_targets: torch.Tensor):
    """cls_pred: (B,A,H,W) raw logit. reg_pred: (B,A*12,H,W) - 마지막 6채널이 완전한
    3D 회전의 6D continuous representation(anchors.assign_targets/rotation3d.py 참고,
    이전 sin/cos 2채널에서 확장 - x,y 회전 무시 시 82.5% 박스에서 점 소속 판정이
    틀리는 게 확인돼 확장함, project_rotation_xy_full_coverage_findings 참고).
    cls_labels: (B,A,H,W) {1,0,-1}. reg_targets: (B,A,H,W,12)."""
    B, A12, H, W = reg_pred.shape
    A = A12 // 12
    reg_pred = reg_pred.view(B, A, 12, H, W).permute(0, 1, 3, 4, 2)  # (B,A,H,W,12)

    pos_mask = cls_labels == 1
    neg_mask = cls_labels == 0
    valid_mask = pos_mask | neg_mask  # ignore(-1)는 제외
    n_pos = pos_mask.sum().clamp_min(1)

    cls_target = pos_mask.float()
    focal = sigmoid_focal_loss(cls_pred, cls_target, alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)
    # RetinaNet 표준 정규화: pos/neg 별도로 안 나누고 전체를 n_pos로 나눈다 - easy negative는
    # (1-p_t)^gamma가 이미 거의 0으로 짓눌러놔서 n_neg로 또 나눠 "1개당 평균"을 맞출 필요가 없다.
    cls_loss = (focal * valid_mask).sum() / n_pos

    reg_diff = reg_pred - reg_targets
    smooth_l1 = F.smooth_l1_loss(reg_diff, torch.zeros_like(reg_diff),
                                  beta=config.SMOOTH_L1_BETA, reduction="none")
    reg_loss = (smooth_l1.sum(dim=-1) * pos_mask.float()).sum() / n_pos

    # 회전(6D) 채널이 새로 추가된 컴포넌트라 다른 성분(위치/크기)을 압도하거나
    # 무시되지 않는지 학습 중 모니터링할 수 있도록 위치/크기/회전을 분리해 로깅한다
    # (loss 자체는 여전히 reg_loss 하나로 합쳐서 최적화 - 가중치는 그대로 균등).
    pos_f = pos_mask.float()
    xyz_loss = (smooth_l1[..., 0:3].sum(dim=-1) * pos_f).sum() / n_pos
    dim_loss = (smooth_l1[..., 3:6].sum(dim=-1) * pos_f).sum() / n_pos
    rot_loss = (smooth_l1[..., 6:12].sum(dim=-1) * pos_f).sum() / n_pos

    total = cls_loss + reg_loss
    return total, {"cls_loss": cls_loss.item(), "reg_loss": reg_loss.item(),
                   "xyz_loss": xyz_loss.item(), "dim_loss": dim_loss.item(),
                   "rot_loss": rot_loss.item(),
                   "n_pos": int(pos_mask.sum().item()), "n_neg": int(neg_mask.sum().item())}
