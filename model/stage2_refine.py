"""stage2_refine.py - CenterPoint stage-2(RoI refinement) 구현.

reports_v2/foreground_aux_branch_proposal.md 6.3 참고 - SonarPoint의 "Markov refinement"는
별개 메커니즘이 아니라 CenterPoint stage-2(원 논문 [Yin et al., CVPR 2021])를 그대로 쓰고
후보점 feature 추출 방식만 바꾼 것(원문 확인됨). 여기서는 vanilla 버전(linear/bilinear
interpolation)부터 구현한다 - Markov point-extraction 업그레이드는 이걸로 부족함이
확인된 뒤에만 검토.

핵심 아이디어: 1단계(RPNCenterHead)가 뱉은 후보 박스마다, 그 박스의 BEV footprint
5개 지점(중심+4변 중점, theta_z로 회전)에서 backbone feature `feat`(768ch)를
bilinear sampling으로 뽑아 concat한 뒤, 작은 MLP로 (1) IoU 기반 refined score와
(2) box parameter residual(Δx,Δy,Δz,Δlog_l,Δlog_w,Δlog_h + 직접 예측 6D rotation)을
낸다. 원 논문처럼 `feat`는 반드시 detach 후 샘플링한다 - stage-2 loss가 backbone까지
역전파해서 기존 학습(Direction 3 등)을 흔드는 걸 막기 위함(이 프로젝트가 새 loss
항으로 두 번 데인 전례 - GRR BatchNorm 폭주, AdamW BN collapse - 를 반영한 안전장치).

z/3D tilt는 이 방식의 근본적 한계를 공유한다 - `feat`가 z를 채널로 collapse해버려서
RoI 샘플링도 BEV (x,y)까지만 가능하고, z/rotation의 실제 정보는 결국 같은 BEV feature
에서 나온다(1단계와 동일한 한계).

학습 시 후보(candidate)는 실제 1단계 예측이 아니라 GT를 노이즈로 흔든 pseudo-candidate를
쓴다(CenterPoint 원 논문 방식) - 우리는 프레임당 GT가 1~2개뿐이라 실제 1단계 출력만으론
다양성이 부족해 stage-2가 배울 게 없다."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
import rotation3d
# eval_voxelnet은 함수 내부에서 지연 import한다(model.py -> stage2_refine.py ->
# eval_voxelnet.py -> model.py 순환 참조 방지 - eval_voxelnet.py가 `from model import
# VoxelNet`을 이미 하고 있어서, 모듈 최상단에서 import하면 순환된다).

N_ROI_POINTS = 5  # 중심 + 전/후/좌/우 변 중점(원 CenterPoint와 동일)


def bev_roi_points(cx: torch.Tensor, cy: torch.Tensor, l: torch.Tensor, w: torch.Tensor,
                    theta_z: torch.Tensor) -> torch.Tensor:
    """박스(N,) 파라미터들 -> (N,5,2) BEV world (x,y) 샘플 지점. 중심 1개 + 변 중점 4개
    (전/후는 length 방향, 좌/우는 width 방향), theta_z로 회전만 반영 - `feat`가 BEV라
    3D tilt는 애초에 여기서 표현 불가(모델 구조 자체의 한계, docstring 참고)."""
    c, s = torch.cos(theta_z), torch.sin(theta_z)
    # local frame에서의 5개 오프셋: center, +front, -back, +left, -right
    local = torch.stack([
        torch.zeros_like(cx), torch.zeros_like(cx),
        l / 2, torch.zeros_like(cx),
        -l / 2, torch.zeros_like(cx),
        torch.zeros_like(cx), w / 2,
        torch.zeros_like(cx), -w / 2,
    ], dim=-1).reshape(-1, N_ROI_POINTS, 2)  # (N,5,2) [local_x, local_y]
    lx, ly = local[..., 0], local[..., 1]
    world_x = cx.unsqueeze(-1) + c.unsqueeze(-1) * lx - s.unsqueeze(-1) * ly
    world_y = cy.unsqueeze(-1) + s.unsqueeze(-1) * lx + c.unsqueeze(-1) * ly
    return torch.stack([world_x, world_y], dim=-1)  # (N,5,2)


def sample_roi_feat(feat: torch.Tensor, points_xy: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
    """feat: (B,C,H,W) - 반드시 호출측에서 detach된 걸 넘길 것(모듈 docstring 참고).
    points_xy: (N,5,2) world 좌표. batch_idx: (N,) 각 박스가 배치 내 몇 번째 프레임인지.
    반환: (N, 5*C) - 5개 지점 feature concat.

    실측(로컬 train split 400프레임): GT box RoI 5점 중 2.67%가 FOV 경계 근처 박스에서
    grid 밖으로 나감(전체 후보가 아니라 개별 지점 단위) - `padding_mode="border"`로
    경계값을 clamp해서, zero-padding(정보 완전 손실)보다 그나마 가까운 실제 feature를
    쓰도록 함."""
    pc = config.POINT_CLOUD_RANGE
    sx, sy = config.ANCHOR_STRIDE
    # world -> [-1,1] normalized grid_sample 좌표. grid_sample은 (x,y) 순서(즉 W축이 x).
    gx = (points_xy[..., 0] - pc[0]) / sx  # cell 단위, [0, W)
    gy = (points_xy[..., 1] - pc[1]) / sy  # cell 단위, [0, H)
    H, W = feat.shape[-2:]
    norm_x = (gx / W) * 2 - 1
    norm_y = (gy / H) * 2 - 1
    out = torch.zeros(points_xy.shape[0], N_ROI_POINTS, feat.shape[1], device=feat.device)
    for b in batch_idx.unique():
        m = batch_idx == b
        grid = torch.stack([norm_x[m], norm_y[m]], dim=-1).unsqueeze(0)  # (1,n_b,5,2)
        sampled = F.grid_sample(feat[b:b + 1], grid, mode="bilinear", padding_mode="border", align_corners=False)
        # sampled: (1,C,n_b,5) -> (n_b,5,C)
        out[m] = sampled[0].permute(1, 2, 0)
    return out.reshape(points_xy.shape[0], -1)  # (N, 5*C)


class Stage2Head(nn.Module):
    """RoI feature(5*768=3840)에서 refined score + box residual을 예측하는 작은 MLP.
    density_head/ForegroundHead와 같은 패턴(작은 conv/linear 스택) - 학습 때만 의미
    있고 추론 시 안 써도 기존 1단계 출력은 그대로 유효(순수 add-on)."""

    def __init__(self, feat_channels: int, hidden: int = 256):
        super().__init__()
        in_dim = N_ROI_POINTS * feat_channels
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.score_head = nn.Linear(hidden, 1)
        # residual: Δx,Δy,Δz,Δlog_l,Δlog_w,Δlog_h (6) + 직접 예측 6D rotation(6) = 12
        self.reg_head = nn.Linear(hidden, 12)

    def forward(self, roi_feat: torch.Tensor):
        h = self.trunk(roi_feat)
        return self.score_head(h).squeeze(-1), self.reg_head(h)  # (N,), (N,12)


def jitter_gt_boxes(gt_boxes: torch.Tensor, n_per_gt: int = 8,
                     pos_noise_frac: float = 0.3, rot_noise_deg: float = 15.0,
                     dim_noise_frac: float = 0.15) -> torch.Tensor:
    """gt_boxes: (M,13) [x,y,z,l,w,h,theta_z,6D rotation]. 각 GT마다 n_per_gt개의
    noisy pseudo-candidate를 만든다(CenterPoint 원 논문 방식 - 실제 1단계 예측은
    프레임당 1~2개뿐이라 다양성이 부족해 이 방식이 필요). pos_noise는 박스
    크기(l,w) 비례, rot_noise는 도 단위, dim_noise는 비율."""
    if len(gt_boxes) == 0:
        return gt_boxes.new_zeros(0, 13)
    device = gt_boxes.device
    reps = gt_boxes.repeat_interleave(n_per_gt, dim=0)  # (M*n,13)
    n = reps.shape[0]
    l, w = reps[:, 3], reps[:, 4]
    dx = (torch.rand(n, device=device) * 2 - 1) * pos_noise_frac * l
    dy = (torch.rand(n, device=device) * 2 - 1) * pos_noise_frac * w
    dz = (torch.rand(n, device=device) * 2 - 1) * 0.1  # z는 박스 크기와 무관하게 고정폭(10cm)
    dl = 1.0 + (torch.rand(n, device=device) * 2 - 1) * dim_noise_frac
    dw = 1.0 + (torch.rand(n, device=device) * 2 - 1) * dim_noise_frac
    dh = 1.0 + (torch.rand(n, device=device) * 2 - 1) * dim_noise_frac
    dtheta = (torch.rand(n, device=device) * 2 - 1) * np.radians(rot_noise_deg)

    cand = reps.clone()
    cand[:, 0] += dx
    cand[:, 1] += dy
    cand[:, 2] += dz
    cand[:, 3] *= dl
    cand[:, 4] *= dw
    cand[:, 5] *= dh
    cand[:, 6] += dtheta
    # 6D rotation도 같은 dtheta만큼 z축 회전시켜 theta_z와 일관되게 유지
    R = rotation3d.sixd_to_matrix_torch(reps[:, 7:13])
    c, s = torch.cos(dtheta), torch.sin(dtheta)
    Rz = torch.zeros(n, 3, 3, device=device)
    Rz[:, 0, 0], Rz[:, 0, 1] = c, -s
    Rz[:, 1, 0], Rz[:, 1, 1] = s, c
    Rz[:, 2, 2] = 1.0
    R_new = torch.bmm(Rz, R)
    cand[:, 7:13] = rotation3d.matrix_to_6d_torch(R_new)
    return cand


def stage2_targets(candidates: torch.Tensor, matched_gt: torch.Tensor, n_iou_samples: int = 300):
    """candidates/matched_gt: (N,13), 이미 1:1 매칭된 상태(jitter_gt_boxes로 만들었으면
    reps와 순서가 그대로 대응). Score target Υ=min(1,max(0,2·IoU-0.5))는
    eval_voxelnet.iou_3d_obb()를 그대로 재사용(SonarPoint/CenterPoint 원 공식) - 학습
    타겟 용도라 n_samples를 eval 기본값(8000)보다 낮춰 속도를 확보(정밀도보다
    throughput이 중요한 soft target)."""
    from eval_voxelnet import iou_3d_obb  # 지연 import, 모듈 docstring 참고(순환 참조 방지)

    cand_np = candidates.detach().cpu().numpy()
    gt_np = matched_gt.detach().cpu().numpy()
    rng = np.random.default_rng(0)
    scores = np.zeros(len(cand_np), dtype=np.float32)
    for i in range(len(cand_np)):
        R_c = rotation3d.sixd_to_matrix_np(cand_np[i, 7:13])
        R_g = rotation3d.sixd_to_matrix_np(gt_np[i, 7:13])
        iou = iou_3d_obb(cand_np[i, :3], cand_np[i, 3:6], R_c,
                          gt_np[i, :3], gt_np[i, 3:6], R_g, n_samples=n_iou_samples, rng=rng)
        scores[i] = min(1.0, max(0.0, 2 * iou - 0.5))
    score_target = torch.from_numpy(scores).to(candidates.device)

    reg_target = torch.zeros(len(cand_np), 12, device=candidates.device)
    reg_target[:, 0] = matched_gt[:, 0] - candidates[:, 0]
    reg_target[:, 1] = matched_gt[:, 1] - candidates[:, 1]
    reg_target[:, 2] = matched_gt[:, 2] - candidates[:, 2]
    reg_target[:, 3] = torch.log(matched_gt[:, 3]) - torch.log(candidates[:, 3])
    reg_target[:, 4] = torch.log(matched_gt[:, 4]) - torch.log(candidates[:, 4])
    reg_target[:, 5] = torch.log(matched_gt[:, 5]) - torch.log(candidates[:, 5])
    reg_target[:, 6:12] = matched_gt[:, 7:13]  # rotation은 residual 아님(직접 예측 6D)
    return score_target, reg_target


def decoded_candidates_and_targets(hm: torch.Tensor, off: torch.Tensor, z: torch.Tensor,
                                    dim: torch.Tensor, rot: torch.Tensor, gt_boxes_list: list,
                                    device, score_thresh: float = 0.05, max_peaks: int = 32,
                                    n_iou_samples: int = 300, fg_iou_thresh: float = 0.55,
                                    fg_ratio: float = 0.5):
    """jitter_gt_boxes() 대신 실제 1st-stage 디코딩 결과를 stage2 학습 후보로 쓴다
    (CenterPoint 원 논문 방식 - tianweiy/CenterPoint의 two_stage.py가 실제
    `one_stage_pred`를 RoI로 쓰는 것과 동일 구조, 2026-08-20 원 저장소 코드로 확인).
    GT+jitter는 stage2가 "진짜 1단계가 내는 실패 패턴"이 아니라 합성 노이즈만 보고
    학습한다는 분포 불일치 문제가 있었다(project_stage2_rng_confound_confirmed 메모리).

    hm/off/z/dim/rot: (B,C,H,W) 메인 center head raw 출력 - 이 함수 안에서 전부
    .detach()해서 디코드하므로 메인 head로 역전파 안 됨(기존 GT+jitter 경로와 동일하게
    stage2는 여전히 메인 학습을 못 건드림). gt_boxes_list: batch["gt_boxes"](길이 B).

    프레임마다 heatmap_targets.decode_center_boxes를 eval과 동일하게 재사용해 후보를
    뽑고(디코드 로직이 두 곳에서 갈리면 val/test 결과와 안 맞음), 각 후보를 그 프레임의
    GT들과 3D IoU로 매칭한다. score_target은 stage2_targets()와 동일한
    Υ=min(1,max(0,2·IoU-0.5)) 공식 재사용.

    FG/BG 비율 서브샘플링(2026-08-20 추가, PV-RCNN/공식 CenterPoint stage-2의
    ProposalTargetLayer 관례) - 이전 버전은 GT와 전혀 안 겹치는 후보(best_iou=0)를
    통째로 스킵해서 stage2가 "이건 확실히 배경"이라는 진짜 hard negative를 하나도 못
    배웠다(공식 레시피와 다른 지점이라 비교가 안 됨 - 사용자 지적). 이제 fg_iou_thresh
    이상은 FG, 미만(0 포함)은 BG로 분류하고, 프레임당 fg_ratio 비율을 목표로 BG를
    서브샘플링한다(FG가 남는 걸 버리진 않음 - 원래도 희귀함). reg_loss는 FG에만
    적용해야 하므로(BG를 엉뚱한 GT로 regression시키면 노이즈) is_fg를 같이 반환."""
    from eval_voxelnet import _decode_center_sample, iou_3d_obb

    B = hm.shape[0]
    all_cand, all_batch_idx, all_score_t, all_reg_t, all_is_fg = [], [], [], [], []
    rng = np.random.default_rng(0)
    py_rng = np.random.default_rng(1)  # 서브샘플링용 - IoU Monte-Carlo(rng)와 스트림 분리
    with torch.no_grad():
        for b in range(B):
            gt_boxes = gt_boxes_list[b]
            if gt_boxes is None or len(gt_boxes) == 0:
                continue
            # raw decode 상한은 max_peaks보다 넉넉하게 잡는다(FG/BG 분류 전에 미리
            # score로만 잘라버리면 낮은 score의 진짜 FG 후보를 놓칠 수 있음) - 최종
            # 개수는 아래 FG/BG 비율 서브샘플링에서 max_peaks로 맞춘다.
            boxes = _decode_center_sample(hm[b].detach(), off[b].detach(), z[b].detach(),
                                           dim[b].detach(), rot[b].detach(), score_thresh=score_thresh)
            # 학습 초반(특히 epoch0-4) dim_head 출력이 아직 안 안정된 상태에서 튀는 값이
            # 나오면 np.exp(dim_pred)가 overflow해서 l/w/h가 inf가 되고, 그대로 iou_3d_obb의
            # rng.uniform(lo,hi,...)에 들어가면 OverflowError로 학습 자체가 죽는다(실측,
            # 2026-08-20 voxelnet_center_stage2_decoded_s0 epoch4에서 크래시). GT+jitter는
            # 항상 GT(항상 유한/합리적 범위)에서 시작하니 이 문제가 없었지만, 실제 디코딩
            # 결과는 아직 안 익은 모델의 튀는 예측을 그대로 포함할 수 있어 방어적으로
            # 걸러야 한다 - 센서 range(POINT_CLOUD_RANGE, 12x10x5m)보다 훨씬 넉넉한
            # 20m 상한으로 non-finite/터무니없는 후보만 스킵(진짜 유효한 큰 오탐까지
            # 걸러내려는 목적이 아님).
            boxes = [b_ for b_ in boxes if np.isfinite([b_["x"], b_["y"], b_["z"], b_["l"], b_["w"], b_["h"]]).all()
                     and max(b_["l"], b_["w"], b_["h"]) < 20.0]
            if len(boxes) > max_peaks * 4:
                boxes = sorted(boxes, key=lambda x: -x["score"])[:max_peaks * 4]
            if not boxes:
                continue
            gt_np = gt_boxes.detach().cpu().numpy() if torch.is_tensor(gt_boxes) else np.asarray(gt_boxes)

            frame_cand, frame_score_t, frame_reg_t, frame_is_fg = [], [], [], []
            for box in boxes:
                best_iou, best_gi = 0.0, 0  # GT가 있는 프레임이라 매칭 GT 없으면 0번(BG는 reg_loss 안 씀)
                for gi, grow in enumerate(gt_np):
                    R_g = rotation3d.sixd_to_matrix_np(grow[7:13])
                    iou = iou_3d_obb(np.array([box["x"], box["y"], box["z"]]),
                                      np.array([box["l"], box["w"], box["h"]]), box["R"],
                                      grow[0:3], grow[3:6], R_g, n_samples=n_iou_samples, rng=rng)
                    if iou > best_iou:
                        best_iou, best_gi = iou, gi
                grow = gt_np[best_gi]
                cand13 = torch.tensor(
                    [box["x"], box["y"], box["z"], box["l"], box["w"], box["h"], box["theta"]]
                    + list(rotation3d.matrix_to_6d(box["R"])), dtype=torch.float32)
                reg = torch.zeros(12)
                reg[0] = float(grow[0] - box["x"])
                reg[1] = float(grow[1] - box["y"])
                reg[2] = float(grow[2] - box["z"])
                reg[3] = float(np.log(grow[3]) - np.log(box["l"]))
                reg[4] = float(np.log(grow[4]) - np.log(box["w"]))
                reg[5] = float(np.log(grow[5]) - np.log(box["h"]))
                reg[6:12] = torch.from_numpy(grow[7:13].astype(np.float32))

                frame_cand.append(cand13)
                frame_score_t.append(min(1.0, max(0.0, 2 * best_iou - 0.5)))
                frame_reg_t.append(reg)
                frame_is_fg.append(best_iou >= fg_iou_thresh)

            # FG/BG 비율 서브샘플링(프레임 단위, PV-RCNN ProposalTargetLayer 관례) - FG는
            # 원래도 희귀해서 버리지 않고 전부 쓰고, BG만 목표 비율에 맞춰 무작위 축소.
            frame_is_fg = np.array(frame_is_fg)
            fg_idx = np.where(frame_is_fg)[0]
            bg_idx = np.where(~frame_is_fg)[0]
            n_fg = len(fg_idx)
            n_bg_target = max(int(round(n_fg * (1 - fg_ratio) / max(fg_ratio, 1e-6))), min(len(bg_idx), 8)) \
                if n_fg > 0 else min(len(bg_idx), 8)
            n_bg_target = min(n_bg_target, len(bg_idx))
            bg_keep = py_rng.choice(bg_idx, size=n_bg_target, replace=False) if n_bg_target < len(bg_idx) else bg_idx
            keep_idx = np.concatenate([fg_idx, bg_keep]) if n_fg > 0 else bg_keep
            if len(keep_idx) > max_peaks:
                keep_idx = py_rng.choice(keep_idx, size=max_peaks, replace=False)

            for i in keep_idx:
                all_cand.append(frame_cand[i])
                all_batch_idx.append(b)
                all_score_t.append(frame_score_t[i])
                all_reg_t.append(frame_reg_t[i])
                all_is_fg.append(bool(frame_is_fg[i]))

    if not all_cand:
        empty = torch.zeros(0, device=device)
        return (torch.zeros(0, 13, device=device), torch.zeros(0, dtype=torch.long, device=device),
                empty, torch.zeros(0, 12, device=device), torch.zeros(0, dtype=torch.bool, device=device))
    candidates = torch.stack(all_cand).to(device)
    batch_idx = torch.tensor(all_batch_idx, dtype=torch.long, device=device)
    score_target = torch.tensor(all_score_t, dtype=torch.float32, device=device)
    is_fg = torch.tensor(all_is_fg, dtype=torch.bool, device=device)
    reg_target = torch.stack(all_reg_t).to(device)
    return candidates, batch_idx, score_target, reg_target, is_fg


def refine_candidates(stage2_head, feat: torch.Tensor, candidates: list) -> list:
    """eval 전용(추론 시 stage2를 실제로 반영) - tianweiy/CenterPoint의 two_stage.py가
    `return_loss=False`일 때 roi_head 출력을 최종 detection으로 쓰는 것과 동일 구조
    (원 저장소 코드로 확인, 2026-08-20). 이전엔 eval_voxelnet.py/eval_voxelnet_by_range.py
    어디서도 stage2를 호출하지 않아 학습만 되고 추론엔 전혀 반영되지 않았다
    (project_stage2_rng_confound_confirmed 메모리).

    stage2_head: model.rpn.stage2(None이면 안 됨 - 호출측에서 확인할 것).
    feat: (1,C,H,W) 이 프레임 하나의 backbone feature(@torch.no_grad() 컨텍스트에서
    호출할 것 - grad 불필요). candidates: heatmap_targets.decode_center_boxes()가 만든
    box dict list({score,x,y,z,l,w,h,theta,R}). 반환: 같은 형식의 새 list, score/box
    파라미터가 stage2로 refine된 값으로 갱신됨(원본 candidates는 안 건드림)."""
    if not candidates:
        return candidates
    device = feat.device
    cx = torch.tensor([b["x"] for b in candidates], dtype=torch.float32, device=device)
    cy = torch.tensor([b["y"] for b in candidates], dtype=torch.float32, device=device)
    cz = torch.tensor([b["z"] for b in candidates], dtype=torch.float32, device=device)
    l = torch.tensor([b["l"] for b in candidates], dtype=torch.float32, device=device)
    w = torch.tensor([b["w"] for b in candidates], dtype=torch.float32, device=device)
    h = torch.tensor([b["h"] for b in candidates], dtype=torch.float32, device=device)
    theta = torch.tensor([b["theta"] for b in candidates], dtype=torch.float32, device=device)
    batch_idx = torch.zeros(len(candidates), dtype=torch.long, device=device)

    pts = bev_roi_points(cx, cy, l, w, theta)
    roi_feat = sample_roi_feat(feat, pts, batch_idx)
    score_pred, reg_pred = stage2_head(roi_feat)
    refined_score = torch.sigmoid(score_pred)

    new_cx = cx + reg_pred[:, 0]
    new_cy = cy + reg_pred[:, 1]
    new_cz = cz + reg_pred[:, 2]
    new_l = l * torch.exp(reg_pred[:, 3])
    new_w = w * torch.exp(reg_pred[:, 4])
    new_h = h * torch.exp(reg_pred[:, 5])
    R_new = rotation3d.sixd_to_matrix_torch(reg_pred[:, 6:12])
    new_theta = torch.atan2(R_new[:, 1, 0], R_new[:, 0, 0])

    refined = []
    for i in range(len(candidates)):
        refined.append({
            "score": float(refined_score[i]), "x": float(new_cx[i]), "y": float(new_cy[i]),
            "z": float(new_cz[i]), "l": float(new_l[i]), "w": float(new_w[i]), "h": float(new_h[i]),
            "theta": float(new_theta[i]), "R": R_new[i].detach().cpu().numpy(),
        })
    return refined


def stage2_loss(score_pred, reg_pred, score_target, reg_target, score_weight: float = 1.0,
                 is_fg: torch.Tensor = None):
    """is_fg(기본 None=전부 FG 취급, gt_jitter 경로와 하위호환 - jitter는 항상 GT에서
    출발해 사실상 다 FG) - decoded 경로에서 넘어온 BG 후보(GT와 안 겹치거나 조금만
    겹치는 후보)를 엉뚱한 GT로 regression시키면 노이즈만 추가되므로(PV-RCNN 관례,
    2026-08-20) reg_loss는 FG에만 적용. score_loss(BCE)는 FG/BG 전부에 적용 -
    "이건 배경이다"를 배우는 것 자체가 목적이라 여기서 배제하면 안 됨."""
    if is_fg is None:
        is_fg = torch.ones(len(reg_target), dtype=torch.bool, device=reg_target.device)
    if is_fg.any():
        reg_loss = F.smooth_l1_loss(reg_pred[is_fg], reg_target[is_fg])
    else:
        reg_loss = reg_pred.sum() * 0.0  # FG가 하나도 없는 배치(드묾) - 0이지만 그래프는 유지
    score_loss = F.binary_cross_entropy_with_logits(score_pred, score_target)
    return reg_loss + score_weight * score_loss, {
        "stage2_reg_loss": reg_loss.item(), "stage2_score_loss": score_loss.item(),
        "stage2_n_fg": int(is_fg.sum().item()), "stage2_n_bg": int((~is_fg).sum().item())}
