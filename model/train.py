"""train.py - VoxelNet 학습 진입점.

로컬(CPU)에서는 --smoke로 scene_0044(87프레임, 가장 작은 scene - Triband_BEV
train.py --smoke와 동일 관례)만 몇 iteration 돌려 shape/loss 파이프라인을 검증한다.
실 학습(Colab GPU)은 --cache-root로 `cache_dataset.py`가 미리 계산해둔 캐시를 읽는다 -
voxelize/anchor 매칭을 매 스텝 다시 하지 않으므로(cache_dataset.py 참고) epoch당
CPU 전처리 오버헤드가 사실상 0에 가깝다.
"""

import argparse
import json
import math
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:  # 로컬 CPU 스모크 테스트 환경엔 없을 수 있음 - 있으면 쓰고 없으면 조용히 폴백
    HAS_TQDM = False

import anchors as anchors_mod
import config
import eval_voxelnet as ev
import foreground_gate as fg
import stage2_refine as s2
from center_loss import center_voxelnet_loss
from dataset import CachedVoxelNetDataset, VoxelNetDataset, collate_fn, load_scene_split
from loss import voxelnet_loss
from model import VoxelNet

VAL_SCORE_THRESH = 0.3
VAL_NMS_IOU = 0.1
# best.pt 선택/로깅 IoU 기준. 교수님 피드백(2026-08-20: IoU0.5는 너무 엄격) + 2026-08-21 threshold
# sweep 실측(0.35가 sweet spot — joint seed 편차 최소 0.005, 절벽 0.4→0.5 직전 마지막 안정점)으로
# 타겟을 0.35로 확정. 이 값들은 모두 ev.IOU_THRESHOLDS(현재 0.25/0.3/0.35/0.4/0.5)의 부분집합이어야 함.
VAL_TARGET_IOU = 0.35                          # primary best (보고/배포용) 선택 기준
VAL_TRACK_IOUS = (0.3, 0.35, 0.4)              # 각각 {run}_best_iou{NN}.pt로 독립 추적(0.35=primary)
VAL_LOG_IOUS = (0.25, 0.3, 0.35, 0.4, 0.5)     # val_history JSON에 남길 전체 기준(evaluate가 다 계산, 공짜)
VAL_PRINT_IOUS = (0.3, 0.35, 0.4)              # 매 epoch stdout에 찍을 타겟존(0.25/0.5는 JSON에만)
# best.pt 선정 강건화(2026-08-28). 계기: joint s0에서 epoch2 raw val AP 스파이크(iou35 0.523,
# 이웃 epoch 0.33/0.36)가 best.pt로 뽑혔는데 test에선 오히려 최악 체크포인트(-9%)였다 -
# 안정 스냅샷(epoch14)은 baseline 대비 +4~6%. 단일 epoch val 노이즈가 best 선정을 오염시키는
# 걸 막으려고 (1) 최근 W epoch 이동평균으로 판정, (2) 초반 warmup 구간은 후보에서 제외한다.
# 상세: reports_v2 / project_fg_refinement_session_2026_08_28.
VAL_SMOOTH_WINDOW = 3                           # best 선정용 val AP 이동평균 창(단일 epoch 스파이크 희석)
VAL_BEST_WARMUP_FRAC = 0.2                      # 이 비율 이전 epoch은 best 후보에서 제외(warmup 스파이크 방어)


def build_optimizer(model, lr, optimizer_name: str = "sgd", weight_decay: float = None):
    """optimizer_name='sgd'(기본, 원 VoxelNet 논문 설정 그대로) - momentum=0.9.
    'adamw' - CenterPoint 공식 코드가 실제로 쓰는 optimizer. Adam 계열은 파라미터별
    최근 gradient 제곱평균(second moment)으로 step을 정규화하므로, SGD+momentum과
    달리 "빈도 높게 계속 비슷한 방향으로 오는 gradient"(우리 데이터의 negative 다수)의
    영향을 자동으로 감쇠시킨다 - momentum이 오히려 그런 신호를 누적/증폭시키는 SGD와
    반대 방향. recall-collapse 메커니즘(negative-dominant aggregate gradient가 저LR에서
    confidence를 서서히 깎음) 대응 후보로 검토 중. SGD와 적정 LR 스케일이 10~100배
    다르므로(Adam 계열은 훨씬 낮은 LR 필요) optimizer를 바꾸면 LR range test를 새로
    해야 한다 - 기존 SGD용 base_lr을 그대로 재사용하면 거의 확실히 발산한다."""
    wd = weight_decay if weight_decay is not None else config.WEIGHT_DECAY
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    assert optimizer_name == "sgd", f"unknown optimizer_name: {optimizer_name}"
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)


def lr_at_epoch(epoch, total_epochs, base_lr, decay_frac=None, decay_factor=None,
                 schedule: str = "step", pct_start: float = 0.4, div_factor: float = 10.0,
                 min_lr_frac: float = 0.0):
    """schedule='step'(기본, 기존 방식) - decay_frac 지점에서 decay_factor로 한 번에
    뚝 떨어짐(원 논문의 160epoch 기준 스케줄을 그대로 이식한 값 - 16~40epoch 규모에서
    재검증된 적 없음, 이 서브필드에서도 이미 벗어난 옛날(2D COCO 탐지 1x/2x 관례) 방식).
    40epoch center 실험에서 decay 이후 recall이 계속 깎이는 문제를 발견
    (module_novelty_ablation_plan.html) - "낮은 LR 구간이 길게 이어지며 negative-
    dominant gradient가 방해 없이 계속 confidence를 깎는다"는 가설.

    schedule='onecycle' - One-Cycle policy(Smith 2018). CenterPoint 원저자 공식 코드
    (tianweiy/CenterPoint, `lr_config=dict(type="one_cycle", lr_max=..., div_factor=10.0,
    pct_start=0.4)`)와 mmdetection3d 재현(`configs/_base_/schedules/cyclic-20e.py`,
    올릴 때/내릴 때 둘 다 CosineAnnealingLR로 구현) 둘 다 이 방식 - "cosine annealing"은
    정책 이름이 아니라 이 정책을 만드는 재료(상승/하강 두 구간의 곡선 모양)였음.
    처음 pct_start(기본 40%) 동안 base_lr/div_factor -> base_lr까지 코사인 곡선으로
    올리고, 나머지 기간 동안 base_lr -> base_lr*min_lr_frac까지 코사인 곡선으로 내림 -
    step처럼 갑자기 뚝 떨어져 오래 머무는 지점이 없어 위 가설의 "긴 저LR 정체 구간"이
    원천적으로 안 생긴다."""
    if schedule == "onecycle":
        warmup_epochs = total_epochs * pct_start
        init_lr = base_lr / div_factor
        min_lr = base_lr * min_lr_frac
        if epoch < warmup_epochs:
            progress = epoch / max(warmup_epochs, 1e-9)
            return init_lr + 0.5 * (base_lr - init_lr) * (1 - math.cos(math.pi * progress))
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1e-9)
        progress = min(max(progress, 0.0), 1.0)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))

    decay_frac = config.LR_DECAY_EPOCH_FRAC if decay_frac is None else decay_frac
    decay_factor = config.LR_DECAY_FACTOR if decay_factor is None else decay_factor
    if epoch >= total_epochs * decay_frac:
        return base_lr * decay_factor
    return base_lr


def momentum_at_epoch(epoch, total_epochs, mom_max: float = 0.95, mom_min: float = 0.85,
                       pct_start: float = 0.4):
    """One-Cycle의 momentum cycling (Smith 2018) - LR과 정반대 방향으로 움직인다:
    LR이 낮을 때(warmup 시작) momentum은 높게(mom_max), LR이 peak에 도달하는 순간
    momentum은 낮게(mom_min), 이후 LR이 다시 낮아지며 momentum도 다시 높아진다.
    CBGS(Zhu et al. 2019, arXiv:1908.09492)·CenterPoint(Yin et al. 2021) 둘 다 이
    레시피를 그대로 씀(momentum 0.95<->0.85) - 이전까지 우리 구현은 LR만 cycling하고
    이 부분이 통째로 빠져있었다. lr_at_epoch()와 완전히 대칭되는 warmup/decay 구간
    나누기를 그대로 재사용한다(진행률 progress는 같고 곡선만 위아래 반전)."""
    warmup_epochs = total_epochs * pct_start
    if epoch < warmup_epochs:
        progress = epoch / max(warmup_epochs, 1e-9)
        return mom_max + 0.5 * (mom_min - mom_max) * (1 - math.cos(math.pi * progress))
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1e-9)
    progress = min(max(progress, 0.0), 1.0)
    return mom_min + 0.5 * (mom_max - mom_min) * (1 - math.cos(math.pi * progress))


def pos_weight_at_epoch(epoch, total_epochs, w_max: float = 1.0, start_frac: float = 0.6):
    """recall-collapse 완화 실험용 - center_loss.gaussian_focal_loss의 positive 항
    가중치를 학습 후반에만 완만하게(cosine ease-in) 올린다. start_frac 이전(이미
    잘 되고 있는 초반 구간)은 그대로 1.0(no-op)로 둬서 건드리지 않고, 그 이후부터
    total_epochs까지 1.0->w_max로 부드럽게 올라간다 - 계단식으로 훅 뛰면 config.py의
    GRAD_CLIP_NORM=35.0가 추가된 계기였던 range-weight loss 폭주 사고와 같은 위험군이라
    momentum_at_epoch()와 동일한 half-cosine 곡선을 재사용해 급격한 변화를 피한다.
    w_max=1.0(기본)이면 항상 1.0을 반환하는 완전 no-op - 하위 호환."""
    start_epoch = total_epochs * start_frac
    if epoch < start_epoch:
        return 1.0
    progress = (epoch - start_epoch) / max(total_epochs - start_epoch, 1e-9)
    progress = min(max(progress, 0.0), 1.0)
    return 1.0 + 0.5 * (w_max - 1.0) * (1 - math.cos(math.pi * progress))


def set_momentum(optimizer, momentum: float):
    """SGD는 param_group['momentum']이 스칼라, AdamW/Adam은 param_group['betas']의
    beta1이 momentum 역할(beta2는 CBGS/CenterPoint 레시피에도 명시 없음 - PyTorch
    기본값 0.999 유지)."""
    for g in optimizer.param_groups:
        if "momentum" in g:
            g["momentum"] = momentum
        elif "betas" in g:
            _, beta2 = g["betas"]
            g["betas"] = (momentum, beta2)


def _step_anchor(model, batch, device):
    cls_pred, reg_pred = model(batch["voxel_features"], batch["num_points"], batch["coords"])
    loss, stats = voxelnet_loss(cls_pred, reg_pred, batch["cls_labels"], batch["reg_targets"])
    return loss, stats


def _register_feat_capture(module: torch.nn.Module) -> dict:
    """module에 forward hook을 걸어 매 forward의 출력을 담아두는 dict를 새로 만들어
    반환한다. 반드시 별도 함수로 감싸야 한다 - train() 안에서 `capture = {}`를 여러
    군데(fg_ctx/stage2_ctx)에서 같은 변수명으로 쓰면, 두 hook 람다 모두 train()의
    같은 지역변수 `capture`를 close-over하게 되어 Python의 late-binding 때문에 나중에
    재할당된 dict로 둘 다 쓰게 되는 버그가 생긴다(실측: fg_ctx["capture"]가 끝까지
    비어있고 stage2_ctx["capture"]에 두 hook 결과가 다 쓰이는 사고 발생). 함수 호출마다
    새 지역 스코프가 생기므로 이 문제를 원천 차단한다."""
    capture = {}
    module.register_forward_hook(lambda m, i, o: capture.__setitem__("feat", o))
    return capture


def _step_center(model, batch, device, focal_alpha=2.0, focal_beta=4.0, use_density_loss=False,
                  pos_weight=1.0, fg_ctx=None, stage2_ctx=None, range_weight_map=None,
                  reg_weight=1.0):
    """fg_ctx(기본 None=Direction 3 없이 기존 동작 그대로): dict, mode in
    {"joint","freeze_fit","joint_ema"}. "joint"는 model에 내장된 fg_head를 매 step 같이
    학습시키고(자기 자신의 BCE loss), "freeze_fit"은 model 밖의 별도 probe(train()이
    주기적으로 갱신)를 detach된 gating 신호로만 쓴다. "joint_ema"는 fg_head 학습 자체는
    joint와 동일하되, gating엔 그 EMA shadow(foreground_gate.EMAShadow)의 출력만 써서
    joint의 recall 보존력 + freeze_fit급 gate 안정성을 동시에 노린다(20epoch 결과 비교 -
    joint가 decay 구간 recall(0.25) 평균 0.936으로 freeze_fit 0.857/baseline 0.814보다
    뚜렷이 나았지만 flat AP3D(0.5)는 freeze_fit이 더 높았음 - joint의 raw per-step gate
    노이즈가 box regression을 방해했을 가능성에 대한 절충 실험). 어느 모드든 메인 heatmap
    loss의 negative 항을 fg_gate로 조절하는 건 동일(center_loss.gaussian_focal_loss 참고)."""
    mode = fg_ctx["mode"] if fg_ctx else "none"
    fg_pred, fg_target, fg_gate = None, None, None

    if mode in ("joint", "joint_ema"):
        hm, off, z, dim, rot, density, fg_logit = model(
            batch["voxel_features"], batch["num_points"], batch["coords"], return_foreground=True)
        h, w = hm.shape[-2:]
        fg_target = fg.fg_target_bev(batch["gt_boxes"], h, w, device=hm.device)
        fg_pred = fg_logit  # live fg_head는 두 모드 다 자기 BCE loss로 매 step 학습
        if mode == "joint":
            fg_gate = torch.sigmoid(fg_logit).detach()
        else:  # joint_ema - gating엔 raw fg_logit 대신 EMA shadow의 출력만 씀
            feat = fg_ctx["capture"]["feat"]
            with torch.no_grad():
                fg_gate = torch.sigmoid(fg_ctx["ema"].shadow(feat))
    elif mode == "freeze_fit":
        hm, off, z, dim, rot, density = model(batch["voxel_features"], batch["num_points"], batch["coords"])
        if fg_ctx["active"]:
            feat = fg_ctx["capture"]["feat"]
            with torch.no_grad():
                fg_gate = torch.sigmoid(fg_ctx["probe"](feat))
    else:
        hm, off, z, dim, rot, density = model(batch["voxel_features"], batch["num_points"], batch["coords"])

    fg_reliability = None
    if fg_ctx and fg_ctx.get("reliability") and fg_gate is not None:
        h, w = hm.shape[-2:]
        rel_mode = fg_ctx.get("reliability_mode", "density")
        if rel_mode == "intensity":
            assert fg_target is not None, \
                "intensity reliability는 fg_target이 필요 - joint/joint_ema 모드 전용(freeze_fit은 fg_target을 안 만듦)"
            fg_reliability = fg.local_intensity_reliability(
                batch["voxel_features"], batch["coords"], batch["num_points"],
                fg_target, hm.shape[0], h, w)
        else:
            fg_reliability = fg.local_density_reliability(
                batch["coords"], batch["num_points"], hm.shape[0], h, w)

    loss, stats = center_voxelnet_loss(hm, off, z, dim, rot, batch["heatmap"], batch["reg_mask"],
                                        batch["offset"], batch["z_center"], batch["dim_center"],
                                        batch["rot_center"],
                                        density_pred=density if use_density_loss else None,
                                        density_target=batch["density_center"] if use_density_loss else None,
                                        reg_weight=reg_weight,
                                        focal_alpha=focal_alpha, focal_beta=focal_beta,
                                        pos_weight=pos_weight,
                                        fg_gate=fg_gate,
                                        fg_alpha_relief=fg_ctx["alpha_relief"] if fg_ctx else 1.0,
                                        fg_r_max=fg_ctx["r_max"] if fg_ctx else 0.5,
                                        fg_reliability=fg_reliability,
                                        fg_pred=fg_pred, fg_target=fg_target,
                                        fg_weight=fg_ctx["weight"] if fg_ctx else 1.0,
                                        fg_reg_alpha=fg_ctx["reg_alpha"] if fg_ctx else 0.0,
                                        range_weight_map=range_weight_map)

    if stage2_ctx is not None:
        gt_list = batch["gt_boxes"]
        if stage2_ctx["candidate_source"] == "decoded":
            # 실제 1st-stage 디코딩 결과를 후보로 씀(CenterPoint 원 논문 방식) - GT+jitter는
            # train/inference 분포가 달라 stage2가 실제 1단계 실패 패턴을 못 배운다는
            # 문제가 있었다(project_stage2_rng_confound_confirmed 메모리, 2026-08-20).
            cand, cand_batch_idx, score_target, reg_target, is_fg = s2.decoded_candidates_and_targets(
                hm, off, z, dim, rot, gt_list, device,
                score_thresh=stage2_ctx["score_thresh"], max_peaks=stage2_ctx["max_peaks"],
                n_iou_samples=stage2_ctx["n_iou_samples"], fg_iou_thresh=stage2_ctx["fg_iou_thresh"],
                fg_ratio=stage2_ctx["fg_ratio"])
        else:  # "gt_jitter" - 이전 방식, 비교/폴백용으로 유지
            gt_flat = torch.cat([g for g in gt_list if len(g)], dim=0) if any(len(g) for g in gt_list) else None
            cand = cand_batch_idx = score_target = reg_target = is_fg = None
            if gt_flat is not None:
                batch_idx = torch.cat([torch.full((len(g),), b, dtype=torch.long)
                                        for b, g in enumerate(gt_list) if len(g)]).to(device)
                gt_flat = gt_flat.to(device)
                cand = s2.jitter_gt_boxes(gt_flat, n_per_gt=stage2_ctx["n_per_gt"],
                                           pos_noise_frac=stage2_ctx["pos_noise_frac"],
                                           rot_noise_deg=stage2_ctx["rot_noise_deg"],
                                           dim_noise_frac=stage2_ctx["dim_noise_frac"])
                cand_batch_idx = batch_idx.repeat_interleave(stage2_ctx["n_per_gt"])
                matched_gt = gt_flat.repeat_interleave(stage2_ctx["n_per_gt"], dim=0)
                score_target, reg_target = s2.stage2_targets(
                    cand, matched_gt, n_iou_samples=stage2_ctx["n_iou_samples"])

        if cand is not None and len(cand):
            pts = s2.bev_roi_points(cand[:, 0], cand[:, 1], cand[:, 3], cand[:, 4], cand[:, 6])
            feat_detached = stage2_ctx["capture"]["feat"].detach()
            roi_feat = s2.sample_roi_feat(feat_detached, pts, cand_batch_idx)
            score_pred, reg_pred = model.rpn.stage2(roi_feat)
            s2_loss, s2_stats = s2.stage2_loss(score_pred, reg_pred, score_target, reg_target,
                                                score_weight=stage2_ctx["score_weight"], is_fg=is_fg)
            loss = loss + stage2_ctx["weight"] * s2_loss
            stats.update(s2_stats)
    return loss, stats


def _run_validation(model, device, head, anchor_grid, val_cache_root, val_split, polar: bool = False):
    """eval_voxelnet.py의 evaluate()/compute_ap()를 그대로 재사용 - AP3D 정의가
    두 곳에서 갈리면 학습 중 추적한 best와 학습 후 재평가 결과가 어긋나므로 절대
    따로 재구현하지 않는다."""
    model.eval()
    samples = ev.iter_cached(val_cache_root, val_split)
    with open(f"{val_cache_root}/manifest.json") as f:
        total = len(json.load(f)[val_split])
    detections, n_gt = ev.evaluate(model, device, samples, anchor_grid, [VAL_SCORE_THRESH],
                                    VAL_NMS_IOU, head=head, total=total, polar=polar)
    # evaluate()가 ev.IOU_THRESHOLDS 전체(현재 5개)를 한 번의 forward로 다 계산해두므로,
    # 모든 threshold의 (ap, prec, rec)를 dict로 반환한다 - 어느 걸 best-tracking/logging에
    # 쓸지는 호출부(VAL_TRACK_IOUS/VAL_LOG_IOUS)가 결정. 추가 연산 비용 없음.
    out = {thr: ev.compute_ap(detections[VAL_SCORE_THRESH][thr], n_gt)
           for thr in ev.IOU_THRESHOLDS}
    model.train()
    return out


def train(run_name: str, dataset, num_epochs: int, batch_size: int, device: str,
          head: str = "anchor", num_workers: int = 0, log_every: int = 10, save_every: int = 0,
          empty_weight: float = 1.0,
          val_cache_root: str = None, val_split: str = "val", val_every: int = 1,
          lr_schedule: str = "step", lr_pct_start: float = 0.4, lr_div_factor: float = 10.0,
          lr_min_frac: float = 0.0,
          focal_alpha: float = 2.0, focal_beta: float = 4.0, base_lr: float = None,
          optimizer_name: str = "sgd", momentum_cycling: bool = False,
          mom_max: float = 0.95, mom_min: float = 0.85, weight_decay: float = None,
          polar: bool = False, use_grr: bool = False, use_density_loss: bool = False,
          pos_weight_max: float = 1.0, pos_weight_start_frac: float = 0.6,
          fg_mode: str = "none", fg_alpha_relief: float = 1.0, fg_r_max: float = 0.5,
          fg_weight: float = 1.0, fg_refit_every: int = 4, fg_warmup_epochs: int = 4,
          fg_probe_lr: float = 0.01, fg_probe_steps: int = 300, fg_probe_batches: int = 20,
          fg_ema_decay: float = 0.999, fg_reliability: bool = False, fg_reliability_mode: str = "density",
          fg_reg_alpha: float = 0.0,
          use_stage2: bool = False, stage2_weight: float = 1.0, stage2_score_weight: float = 1.0,
          stage2_n_per_gt: int = 8, stage2_pos_noise_frac: float = 0.3,
          stage2_rot_noise_deg: float = 15.0, stage2_dim_noise_frac: float = 0.15,
          stage2_n_iou_samples: int = 300, stage2_candidate_source: str = "decoded",
          stage2_score_thresh: float = 0.05, stage2_max_peaks: int = 32,
          stage2_fg_iou_thresh: float = 0.55, stage2_fg_ratio: float = 0.5, seed: int = 0,
          use_rot_branch: bool = False, use_pdconv: bool = False, grr_n: int = None,
          use_ga: bool = False, fg_range_cond: bool = False,
          range_weight_mode: str = "none",
          use_rs_conv: bool = False, use_feat_undistort: bool = False,
          reg_weight: float = 1.0):
    """save_every>0: 매 epoch 덮어쓰는 {run_name}.pt(최종본)와 별개로
    {run_name}_epoch{N:03d}.pt 스냅샷도 남긴다 - 장기학습 비교 실험에서 수렴 지점을
    epoch 단위로 촘촘히 보고 싶을 때 필요(val 성능만으로는 "그 근처 어느 epoch이
    진짜 최선인지"까지는 못 잡음, save_every로 스냅샷 자체를 남겨야 사후 비교 가능).

    val_cache_root가 주어지면(cache_dataset.py 출력, train과 같은 캐시에 val split도
    들어있는 게 보통) val_every epoch마다 eval_voxelnet.py와 동일한 AP3D(score_thresh
    0.3)를 검증한다. evaluate()가 ev.IOU_THRESHOLDS 전체(현재 5개)를 한 번의 forward로
    다 계산하므로, VAL_LOG_IOUS(0.25/0.3/0.35/0.4/0.5) 전부를 val_history JSON에 남기고,
    VAL_TRACK_IOUS(0.3/0.35/0.4) 각각 개선될 때마다 {run_name}_best_iou{NN}.pt를 독립적으로
    갱신 저장한다 - **0.35가 primary 타겟**(교수님 피드백 후 2026-08-21 확정, VAL_TARGET_IOU).
    stdout에는 타겟존(VAL_PRINT_IOUS=0.3/0.35/0.4)만 찍고 0.25/0.5는 JSON에만 남긴다.
    옛날엔 0.25/0.5 두 기준으로만 뽑았는데(IoU0.25는 seed 간 부호가 뒤집힐 만큼 불안정,
    IoU0.5는 너무 엄격), 타겟존 세 개로 바꿔 best epoch이 threshold별로 어긋나는지도
    함께 분석 가능하게 했다(0.35 best를 반드시 확보하면서 0.3/0.4 best도 덤으로 남김).
    Ultralytics의 자동 best.pt 추적과 동등한 기능. 매 epoch val 프레임 수백~수천 개를
    도는 비용은 GPU에서 train epoch 자체보다 훨씬 작다(val이 보통 train보다 작고,
    backward pass가 없음)."""
    assert len(dataset) > 0, "데이터셋에 유효 라벨 프레임이 없음"
    # cudnn.benchmark: 입력 크기가 고정(voxel grid 일정)이라 최적 conv 커널을 캐싱해 dense
    # Conv3D를 가속. 결정성은 약간 포기하지만 shuffle은 별도 generator로 고정하므로 재현성
    # 영향 미미(2026-08-28, 속도 최적화).
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    collate = (lambda b: collate_fn(b, head=head))
    # DataLoader(shuffle=True)에 generator를 명시하지 않으면 RandomSampler가 매 epoch
    # `torch.empty(()).random_()`으로 전역 RNG에서 셔플 시드를 뽑는다(torch.utils.data.
    # sampler.RandomSampler.__iter__ 소스 확인, 2026-08-20) - stage2의 jitter_gt_boxes()/
    # decoded_candidates_and_targets()가 학습 스텝마다 torch.rand()를 추가로 소비하면
    # 그 순간부터 "같은 --seed"라도 shuffle 순서가 stage2 없는 런과 완전히 갈라진다(실측:
    # stage2 켠/끈 "같은 seed" 비교가 seed 노이즈와 구분 안 될 정도로 흔들림, project 메모리
    # project_stage2_rng_confound_confirmed 참고). shuffle 순서를 학습 스텝 내부의 다른
    # RNG 소비와 완전히 분리하기 위해 seed로 고정된 독립 generator를 명시적으로 넘긴다.
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    # persistent_workers: on-the-fly Dataset(gt_db를 __init__에서 1회 구축)에서 worker를
    # epoch마다 재생성하면 gt_db가 매번 re-fork/재구축되는 낭비가 생긴다 - worker를 유지해
    # 방지. prefetch_factor로 미리 배치를 준비해 CPU voxelize 지연을 GPU 연산 뒤에 숨긴다.
    # (num_workers=0이면 두 옵션 모두 무효라 조건부로 넘김.)
    _loader_kw = {}
    if num_workers > 0:
        _loader_kw = {"persistent_workers": True, "prefetch_factor": 4}
    # CBGS식 균형: --empty-weight<1 이면 empty(배경) 프레임을 상대적으로 subsample(=pos
    # oversample)해 노출을 통제한다. empty = 현재 split(train_full)에 있고 train(pos)엔 없는
    # 프레임. empty ablation이 recall을 떨어뜨린 것(전량 무균형 포함)에 대한 균형 arm.
    _sampler = None
    if empty_weight != 1.0 and hasattr(dataset, "entries") and val_cache_root:
        from torch.utils.data import WeightedRandomSampler
        with open(f"{val_cache_root}/manifest.json") as _f:
            _pos = set(json.load(_f).get("train", []))
        _w = [1.0 if e in _pos else empty_weight for e in dataset.entries]
        n_emp = sum(1 for e in dataset.entries if e not in _pos)
        _sampler = WeightedRandomSampler(_w, num_samples=len(dataset), replacement=True,
                                          generator=loader_generator)
        print(f"[balance] empty-weight={empty_weight}: empty={n_emp}/{len(dataset)} "
              f"WeightedRandomSampler(pos oversample)", flush=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(_sampler is None),
                         sampler=_sampler, collate_fn=collate, num_workers=num_workers,
                         generator=loader_generator, **_loader_kw)

    assert not (polar and head == "anchor"), "polar는 center head만 지원(오늘 방침 - anchor head는 스코프 밖)"
    assert not (use_grr and not polar), "GRR은 polar 전용(model.py VoxelNet 참고)"
    assert not (use_pdconv and not polar), "PD-Conv는 polar 전용(pdconv.py 참고)"
    assert not (use_ga and not polar), "GA는 polar 전용(polar_ga.py 참고)"
    assert not (grr_n is not None and not use_grr), "--grr-n은 --use-grr와 함께만 유효"
    assert not (fg_range_cond and fg_mode not in ("joint", "joint_ema")), \
        "--fg-range-cond는 --fg-mode joint/joint_ema와 함께만 유효(freeze_fit은 별도 probe)"
    assert range_weight_mode in ("none", "linear", "sqrt"), \
        "--range-weight-mode는 none/linear/sqrt 중 하나"
    assert fg_mode in ("none", "joint", "freeze_fit", "joint_ema")
    assert not (fg_reg_alpha > 0.0 and fg_mode == "none"), \
        "--fg-reg-alpha(A2)는 fg_gate가 있는 모드 필요(joint/joint_ema/freeze_fit), fg_mode=none과 무의미"
    assert not (fg_mode != "none" and head != "center"), "Direction 3는 center head 전용"
    assert not (use_stage2 and head != "center"), "stage2는 center head 전용"
    assert stage2_candidate_source in ("decoded", "gt_jitter")
    assert fg_reliability_mode in ("density", "intensity")
    assert not (fg_reliability and fg_reliability_mode == "intensity" and fg_mode == "freeze_fit"), \
        "intensity reliability는 fg_target이 필요 - freeze_fit은 fg_target을 안 만들어서 지원 안 함(joint/joint_ema만)"
    model = VoxelNet(head=head, polar=polar, use_grr=use_grr,
                      use_fg_head=(fg_mode in ("joint", "joint_ema")),
                      use_stage2=use_stage2, use_rot_branch=use_rot_branch,
                      use_pdconv=use_pdconv, grr_n=grr_n, use_ga=use_ga,
                      fg_range_cond=fg_range_cond,
                      use_rs_conv=use_rs_conv,
                      use_feat_undistort=use_feat_undistort).to(device)
    base_lr = base_lr if base_lr is not None else config.LR
    optimizer = build_optimizer(model, base_lr, optimizer_name, weight_decay)
    cls_key, reg_key = ("cls_loss", "reg_loss") if head == "anchor" else ("hm_loss", "reg_loss")

    # Direction 3(reports_v2/foreground_aux_branch_proposal.md 4.7) - "확률"이라 안 부르고
    # fg_gate라고 부름(gating 신호일 뿐). freeze_fit/joint_ema는 별도 probe+hook이 필요해서 여기서 준비.
    fg_ctx = None
    if fg_mode != "none":
        fg_ctx = {"mode": fg_mode, "alpha_relief": fg_alpha_relief, "r_max": fg_r_max, "weight": fg_weight,
                   "reliability": fg_reliability, "reliability_mode": fg_reliability_mode,
                   "reg_alpha": fg_reg_alpha,  # A2(2026-08-28): fg-gated regression weighting, 0=off
                   "active": fg_mode in ("joint", "joint_ema")}  # joint/joint_ema는 처음부터 활성,
        # freeze_fit은 첫 refit 전까지 fg_gate=None(no-op)로 둠(warmup)
        if fg_mode in ("freeze_fit", "joint_ema"):
            fg_ctx["capture"] = _register_feat_capture(model.rpn.backbone)
        if fg_mode == "freeze_fit":
            fg_ctx["probe"] = fg.GatingProbe(model.rpn.backbone.out_channels).to(device)
        if fg_mode == "joint_ema":
            fg_ctx["ema"] = fg.EMAShadow(model.rpn.fg_head, decay=fg_ema_decay)
        print(f"Direction 3 fg_mode={fg_mode} alpha_relief={fg_alpha_relief} r_max={fg_r_max} "
              f"weight={fg_weight}" + (f" refit_every={fg_refit_every} warmup_epochs={fg_warmup_epochs}"
                                        if fg_mode == "freeze_fit" else "")
              + (f" ema_decay={fg_ema_decay}" if fg_mode == "joint_ema" else "")
              + (f" fg_reliability={fg_reliability_mode}-based(7.4)" if fg_reliability else "")
              + (f" fg_reg_alpha={fg_reg_alpha}(A2)" if fg_reg_alpha > 0.0 else ""))

    # CenterPoint stage-2(RoI refinement, reports_v2/foreground_aux_branch_proposal.md 6.3) -
    # candidate는 1단계 decode 출력이 아니라 GT jitter라서 fg_ctx와 별개로 자체 hook을
    # 둔다(model.py 수정 없이 feat.detach()를 얻는 freeze_fit과 동일 패턴).
    stage2_ctx = None
    if use_stage2:
        stage2_ctx = {"capture": _register_feat_capture(model.rpn.backbone),
                      "weight": stage2_weight, "score_weight": stage2_score_weight,
                      "n_per_gt": stage2_n_per_gt, "pos_noise_frac": stage2_pos_noise_frac,
                      "rot_noise_deg": stage2_rot_noise_deg, "dim_noise_frac": stage2_dim_noise_frac,
                      "n_iou_samples": stage2_n_iou_samples, "candidate_source": stage2_candidate_source,
                      "score_thresh": stage2_score_thresh, "max_peaks": stage2_max_peaks,
                      "fg_iou_thresh": stage2_fg_iou_thresh, "fg_ratio": stage2_fg_ratio}
        print(f"CenterPoint stage2 candidate_source={stage2_candidate_source} weight={stage2_weight} "
              f"score_weight={stage2_score_weight} n_per_gt={stage2_n_per_gt} "
              f"pos_noise_frac={stage2_pos_noise_frac} rot_noise_deg={stage2_rot_noise_deg} "
              f"dim_noise_frac={stage2_dim_noise_frac}" +
              (f" decode_score_thresh={stage2_score_thresh} max_peaks={stage2_max_peaks}"
               if stage2_candidate_source == "decoded" else ""))

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"run={run_name} head={head} polar={polar} use_grr={use_grr} grr_n={model.grr_n if use_grr else '-'} "
          f"use_pdconv={use_pdconv} use_ga={use_ga} fg_range_cond={fg_range_cond} "
          f"range_weight_mode={range_weight_mode} use_rs_conv={use_rs_conv} "
          f"use_feat_undistort={use_feat_undistort} reg_weight={reg_weight} "
          f"use_density_loss={use_density_loss} use_rot_branch={use_rot_branch} "
          f"pos_weight_max={pos_weight_max} pos_weight_start_frac={pos_weight_start_frac} "
          f"samples={len(dataset)} batches/epoch={len(loader)} epochs={num_epochs} "
          f"total_steps={len(loader) * num_epochs} device={device}")

    # anchor head면 val 여부와 무관하게 anchor grid가 필요(eval_voxelnet.py 재사용) - polar는
    # anchor head를 안 쓰므로 항상 None이라도 위 assert가 이미 막아줌.
    anchor_grid = anchors_mod.build_anchor_grid() if (val_cache_root or head == "anchor") else None
    if (focal_alpha, focal_beta) != (2.0, 4.0):
        print(f"focal_alpha={focal_alpha} focal_beta={focal_beta} (CornerNet 기본값 2.0/4.0에서 재조정, center head 전용)")
    # Range-conditioned loss weight map (2026-08-22) - 학습 중 정적이라 한 번만 계산.
    # 각 셀의 range r에 따라 loss에 곱해질 weight. train 대비 test range 분포 비율 기반.
    # 실측 test/train 비율: 0-2m 0.05, 2-2.5m 0.81, 2.5-3m 1.00, 3-3.5m 2.20, 3.5-5m 1.57, 5m+ 0.68
    range_weight_map = None
    if range_weight_mode != "none" and head == "center":
        import math as _math
        _bucket_ratio_linear = [(0.0, 2.0, 0.05), (2.0, 2.5, 0.81), (2.5, 3.0, 1.00),
                                 (3.0, 3.5, 2.20), (3.5, 5.0, 1.57), (5.0, 999.0, 0.68)]
        # sqrt로 완화하면 극단값(0.05, 2.20)이 부드러워짐
        _f = (lambda x: _math.sqrt(x)) if range_weight_mode == "sqrt" else (lambda x: x)
        # 반드시 [1e-3, ~2.5] 안으로 clamp - 학습 안정성
        if polar:
            r_lo, r_hi = config.POLAR_R_RANGE
            H_grid = config.POLAR_HEATMAP_R_BINS
            W_grid = config.POLAR_THETA_BINS
            rs = r_lo + (r_hi - r_lo) * (torch.arange(H_grid) + 0.5) / H_grid
            r_map = rs.unsqueeze(1).expand(H_grid, W_grid).float()
        else:
            W_g, H_g = config.ANCHOR_GRID_SIZE
            sx, sy = config.ANCHOR_STRIDE
            x0, y0 = config.POINT_CLOUD_RANGE[:2]
            xs = x0 + sx * (torch.arange(W_g) + 0.5)
            ys = y0 + sy * (torch.arange(H_g) + 0.5)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            r_map = torch.sqrt(xx * xx + yy * yy)
        w_map = torch.ones_like(r_map)
        for lo, hi, ratio in _bucket_ratio_linear:
            mask = (r_map >= lo) & (r_map < hi)
            w_map[mask] = _f(ratio)
        w_map = w_map.clamp(0.1, 3.0)  # 안전 clamp
        range_weight_map = w_map.unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
        print(f"[range-conditioned loss weighting] mode={range_weight_mode} "
              f"grid={tuple(r_map.shape)} range=[{w_map.min():.3f}, {w_map.max():.3f}]")

    step_fn = (lambda m, b, d, pw: _step_anchor(m, b, d)) if head == "anchor" \
        else (lambda m, b, d, pw: _step_center(m, b, d, focal_alpha, focal_beta, use_density_loss, pw,
                                                fg_ctx, stage2_ctx, range_weight_map=range_weight_map,
                                                reg_weight=reg_weight))
    if pos_weight_max != 1.0:
        print(f"pos_weight schedule: 1.0 -> {pos_weight_max} starting at epoch "
              f"{pos_weight_start_frac * num_epochs:.1f}/{num_epochs} (center head 전용, recall-collapse 완화 실험)")
    val_history = []
    best_ap = {thr: -1.0 for thr in VAL_TRACK_IOUS}      # smoothed(이동평균) 기준 best - .pt 선정에 사용
    best_epoch = {thr: None for thr in VAL_TRACK_IOUS}
    best_raw_at_best = {thr: None for thr in VAL_TRACK_IOUS}  # 참고용: 그 epoch의 raw AP(선정엔 안 씀)

    step = 0
    for epoch in range(num_epochs):
        lr = lr_at_epoch(epoch, num_epochs, base_lr,
                          schedule=lr_schedule, pct_start=lr_pct_start, div_factor=lr_div_factor,
                          min_lr_frac=lr_min_frac)
        for g in optimizer.param_groups:
            g["lr"] = lr
        if momentum_cycling:
            mom = momentum_at_epoch(epoch, num_epochs, mom_max=mom_max, mom_min=mom_min,
                                     pct_start=lr_pct_start)
            set_momentum(optimizer, mom)
        pw = pos_weight_at_epoch(epoch, num_epochs, w_max=pos_weight_max, start_frac=pos_weight_start_frac)

        if fg_ctx and fg_ctx["mode"] == "freeze_fit" and epoch >= fg_warmup_epochs \
                and (epoch - fg_warmup_epochs) % fg_refit_every == 0:
            # Stage 0 probe(analyze_recall_collapse.train_probe_and_auc)와 동일하게 "짧고
            # 독립적으로" fg_head를 다시 fit - 매 refit마다 완전히 새로 초기화해서, heatmap_head가
            # 겪은 것 같은 누적된 긴 학습 경로 문제를 구조적으로 피한다(자세한 논의는 이 세션의
            # Direction 3 설계 검토 참고).
            model.eval()
            new_probe = fg.GatingProbe(model.rpn.backbone.out_channels).to(device)
            feats_buf, targets_buf = [], []
            refit_gen = torch.Generator()
            refit_gen.manual_seed(seed * 1000 + epoch)  # 메인 loader_generator와 다른 스트림 - epoch마다 바뀌어야 매 refit이 다른 배치를 봄
            refit_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                       collate_fn=collate, num_workers=0, generator=refit_gen)
            with torch.no_grad():
                for i, rbatch in enumerate(refit_loader):
                    if i >= fg_probe_batches:
                        break
                    rbatch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in rbatch.items()}
                    hm_tmp, *_ = model(rbatch["voxel_features"], rbatch["num_points"], rbatch["coords"])
                    h, w = hm_tmp.shape[-2:]
                    feats_buf.append(fg_ctx["capture"]["feat"].detach())
                    targets_buf.append(fg.fg_target_bev(rbatch["gt_boxes"], h, w, device=device))
            fg.refit_probe(new_probe, torch.cat(feats_buf, 0), torch.cat(targets_buf, 0),
                            n_steps=fg_probe_steps, lr=fg_probe_lr)
            fg_ctx["old_sd"] = ({k: v.clone() for k, v in fg_ctx["probe"].state_dict().items()}
                                 if fg_ctx["active"] else new_probe.state_dict())
            fg_ctx["new_sd"] = new_probe.state_dict()
            fg_ctx["blend_start_epoch"] = epoch
            fg_ctx["active"] = True
            print(f"epoch {epoch}: fg probe refit ({fg_probe_batches}배치, n_pos={int(torch.cat(targets_buf,0).sum().item())})")
            model.train()

        if fg_ctx and fg_ctx["mode"] == "freeze_fit" and fg_ctx["active"]:
            alpha = min((epoch - fg_ctx["blend_start_epoch"]) / max(fg_refit_every, 1), 1.0)
            fg_ctx["probe"].load_state_dict(fg.blend_state_dicts(fg_ctx["old_sd"], fg_ctx["new_sd"], alpha))

        model.train()
        t0 = time.time()
        running = {"loss": 0.0, "cls": 0.0, "reg": 0.0, "n": 0}

        iterable = tqdm(loader, desc=f"epoch {epoch}/{num_epochs - 1}", unit="batch",
                         leave=True) if HAS_TQDM else loader
        for batch in iterable:
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            loss, stats = step_fn(model, batch, device, pw)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()
            if fg_ctx and fg_ctx["mode"] == "joint_ema":
                fg_ctx["ema"].update(model.rpn.fg_head)  # live fg_head가 막 갱신된 직후 shadow도 한 스텝 따라감

            running["loss"] += loss.item()
            running["cls"] += stats[cls_key]
            running["reg"] += stats[reg_key]
            running["n"] += 1

            if HAS_TQDM:
                gpu_mem = f"{torch.cuda.memory_reserved(device) / 1e9:.2f}G" if device.startswith("cuda") else "-"
                iterable.set_postfix(gpu_mem=gpu_mem, loss=f"{loss.item():.3f}",
                                      **{cls_key: f"{stats[cls_key]:.3f}", reg_key: f"{stats[reg_key]:.3f}"},
                                      lr=f"{lr:.4f}", instances=stats["n_pos"])
            elif step % log_every == 0:
                print(f"epoch {epoch} step {step} lr {lr:.4f} loss {loss.item():.4f} "
                      f"{cls_key} {stats[cls_key]:.4f} {reg_key} {stats[reg_key]:.4f} "
                      f"npos {stats['n_pos']}")
            step += 1

        n = max(running["n"], 1)
        weight_std_log = ""
        if head == "center":
            hm_w_std = model.rpn.heatmap_head.weight.std().item()
            weight_std_log = f" heatmap_head_w_std={hm_w_std:.4f}"
            if fg_ctx and fg_ctx["mode"] in ("joint", "joint_ema"):
                weight_std_log += f" fg_head_w_std={model.rpn.fg_head.net[0].weight.std().item():.4f}"
            if fg_ctx and fg_ctx["mode"] == "joint_ema":
                weight_std_log += f" fg_ema_w_std={fg_ctx['ema'].shadow.net[0].weight.std().item():.4f}"
            elif fg_ctx and fg_ctx["mode"] == "freeze_fit" and fg_ctx["active"]:
                weight_std_log += f" fg_probe_w_std={fg_ctx['probe'].net[0].weight.std().item():.4f}"
        print(f"epoch {epoch} done in {time.time() - t0:.1f}s | "
              f"avg_loss={running['loss'] / n:.4f} avg_{cls_key}={running['cls'] / n:.4f} "
              f"avg_{reg_key}={running['reg'] / n:.4f} lr={lr:.4f}{weight_std_log}")

        ckpt_path = config.CHECKPOINT_DIR / f"{run_name}.pt"
        config_snapshot = {k: v for k, v in vars(config).items()
                            if k.isupper() and isinstance(v, (int, float, str, tuple, list))}
        ckpt_payload = {"model": model.state_dict(), "epoch": epoch, "config": config_snapshot,
                         "head": head, "polar": polar, "use_grr": use_grr, "use_pdconv": use_pdconv,
                         "use_ga": use_ga, "fg_range_cond": fg_range_cond,
                         "use_rs_conv": use_rs_conv, "use_feat_undistort": use_feat_undistort,
                         "grr_n": (model.grr_n if use_grr else None),
                         "use_stage2": use_stage2, "use_rot_branch": use_rot_branch}
        torch.save(ckpt_payload, ckpt_path)
        if save_every > 0 and (epoch + 1) % save_every == 0:
            snapshot_path = config.CHECKPOINT_DIR / f"{run_name}_epoch{epoch:03d}.pt"
            torch.save(ckpt_payload, snapshot_path)
            print(f"snapshot saved {snapshot_path}")

        if val_cache_root and (epoch + 1) % val_every == 0:
            vt0 = time.time()
            val = _run_validation(
                model, device, head, anchor_grid, val_cache_root, val_split, polar=polar)
            # val_history JSON: 전체 기준(VAL_LOG_IOUS) 기록 - 나중에 어느 threshold든 재분석 가능
            rec_entry = {"epoch": epoch}
            for thr in VAL_LOG_IOUS:
                ap_t, p_t, r_t = val[thr]
                tag = f"{int(round(thr * 100)):02d}"
                rec_entry[f"ap3d_iou{tag}"] = ap_t
                rec_entry[f"precision_iou{tag}"] = p_t
                rec_entry[f"recall_iou{tag}"] = r_t
            val_history.append(rec_entry)
            with open(config.CHECKPOINT_DIR / f"{run_name}_val_history.json", "w") as f:
                json.dump(val_history, f, indent=2)
            # best 추적: VAL_TRACK_IOUS 각각 독립적으로 {run}_best_iou{NN}.pt 갱신(0.35=primary).
            # 단일 epoch 스파이크 방어(2026-08-28): raw AP가 아니라 최근 VAL_SMOOTH_WINDOW epoch
            # 이동평균으로 판정하고, 초반 warmup(VAL_BEST_WARMUP_FRAC * num_epochs) 구간은 후보에서
            # 제외한다. val_history엔 이미 이번 epoch rec_entry가 append돼 있어 [-W:]가 현재 포함.
            improved = []
            warmup_cut = int(VAL_BEST_WARMUP_FRAC * num_epochs)
            for thr in VAL_TRACK_IOUS:
                tag = f"{int(round(thr * 100)):02d}"
                recent = [e[f"ap3d_iou{tag}"] for e in val_history[-VAL_SMOOTH_WINDOW:]]
                ap_smooth = sum(recent) / len(recent)
                if epoch >= warmup_cut and ap_smooth > best_ap[thr]:
                    best_ap[thr], best_epoch[thr] = ap_smooth, epoch
                    best_raw_at_best[thr] = val[thr][0]
                    torch.save(ckpt_payload, config.CHECKPOINT_DIR / f"{run_name}_best_iou{tag}.pt")
                    improved.append(int(round(thr * 100)))
            # stdout: 타겟존(VAL_PRINT_IOUS)만 찍음(0.25/0.5는 JSON에만) - 0.35 타겟
            log_zone = " ".join(f"iou{int(round(thr * 100))}={val[thr][0]:.4f}" for thr in VAL_PRINT_IOUS)
            best_tags = ('  <- best@' + ','.join(str(t) for t in improved)) if improved else ''
            print(f"epoch {epoch} val AP3D[{log_zone}] target=iou{int(round(VAL_TARGET_IOU * 100))} "
                  f"({time.time() - vt0:.1f}s){best_tags}")
    print(f"saved {ckpt_path}")
    if val_cache_root:
        for thr in VAL_TRACK_IOUS:
            tag = f"{int(round(thr * 100)):02d}"
            star = " (target)" if abs(thr - VAL_TARGET_IOU) < 1e-9 else ""
            raw = best_raw_at_best[thr]
            raw_str = f" raw={raw:.4f}" if raw is not None else ""
            print(f"best(iou{tag}): epoch {best_epoch[thr]} AP3D_smooth{VAL_SMOOTH_WINDOW}={best_ap[thr]:.4f}"
                  f"{raw_str} -> {run_name}_best_iou{tag}.pt{star}")


def run_lr_range_test(run_name: str, dataset, batch_size: int, device: str, head: str = "anchor",
                       num_workers: int = 0, lr_min: float = 1e-5, lr_max: float = 1.0,
                       num_steps: int = 500, optimizer_name: str = "sgd",
                       polar: bool = False, use_grr: bool = False):
    """LR range test(Smith, "Cyclical Learning Rates for Training Neural Networks", 2017) -
    정식 학습 없이 lr_min~lr_max를 num_steps에 걸쳐 지수적으로(기하급수) 올려가며 step별
    loss를 기록한다. 결과 JSON을 나중에 (lr, loss) 그래프로 보고, loss가 발산하기
    시작하는 지점 바로 아래를 적정 base_lr로 채택 - 여러 번 풀 학습을 돌려보는 것보다
    훨씬 싼 진단 방법. val/체크포인트 저장 없음(진단 전용, 정식 학습 아님)."""
    assert len(dataset) > 0, "데이터셋에 유효 라벨 프레임이 없음"
    assert not (polar and head == "anchor"), "polar는 center head만 지원"
    assert not (use_grr and not polar), "GRR은 polar 전용"
    collate = (lambda b: collate_fn(b, head=head))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         collate_fn=collate, num_workers=num_workers)

    model = VoxelNet(head=head, polar=polar, use_grr=use_grr).to(device)
    optimizer = build_optimizer(model, lr_min, optimizer_name)
    step_fn = _step_anchor if head == "anchor" else _step_center

    gamma = (lr_max / lr_min) ** (1.0 / max(num_steps - 1, 1))  # 기하급수 - 매 step마다 이 배율만큼 LR 증가
    print(f"lr_range_test run={run_name} head={head} polar={polar} use_grr={use_grr} "
          f"lr_min={lr_min} lr_max={lr_max} num_steps={num_steps} device={device}")

    model.train()
    history = []
    step = 0
    loader_iter = iter(loader)
    while step < num_steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}

        lr = lr_min * (gamma ** step)
        for g in optimizer.param_groups:
            g["lr"] = lr

        loss, stats = step_fn(model, batch, device)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
        optimizer.step()

        loss_val = loss.item()
        history.append({"step": step, "lr": lr, "loss": loss_val})
        if step % 20 == 0 or step == num_steps - 1:
            print(f"step {step}/{num_steps} lr={lr:.2e} loss={loss_val:.4f}")
        step += 1

    out_path = config.CHECKPOINT_DIR / f"{run_name}_lr_range_test.json"
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="voxelnet_run1")
    parser.add_argument("--scenes", nargs="*", default=None,
                         help="로컬 모드: 지정 없으면 splits.json의 train split 사용")
    parser.add_argument("--cache-root", default=None,
                         help="지정하면 cache_dataset.py 출력(CachedVoxelNetDataset)을 사용 - Colab용, 권장. "
                              "--on-the-fly와 함께면 이건 VAL 전용 캐시로 쓰인다(val은 증강 안 함).")
    parser.add_argument("--split", default="train")
    # --- on-the-fly augmentation(train 전용, center head) ---
    parser.add_argument("--on-the-fly", action="store_true",
                         help="train을 pre-baked 캐시 대신 매 __getitem__ 즉석 증강+voxelize(dataset_onthefly). "
                              "매 epoch 새 랜덤 증강. val은 --cache-root(pre-baked) 사용.")
    parser.add_argument("--annotations-dir", default=None, help="on-the-fly: 변환 annotation json 디렉토리")
    parser.add_argument("--points-root", default=None, help="on-the-fly: raw sonar bin 루트(uScenes식)")
    parser.add_argument("--splits-json", default=None, help="on-the-fly: splits.json 경로")
    parser.add_argument("--otf-strong", action="store_true", help="on-the-fly train에 strong 증강(GT-sampling+flip+translation)")
    parser.add_argument("--otf-include-raw", action="store_true",
                        help="on-the-fly: 각 프레임을 raw 1개+매 epoch 새 증강 1개로(50/50, pre-baked 캐시 구조와 동일). "
                             "미지정 시 100% 증강(교과서 online). online-vs-offline 통제비교(D1)엔 지정 권장")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LR,
                         help="base_lr(=One-Cycle peak, step 스케줄의 시작 LR). 기본 config.LR=0.01")
    parser.add_argument("--optimizer", default="adamw", choices=["sgd", "adamw"],
                         help="adamw(기본, 확정 레시피 - CenterPoint/CBGS 공식 설정) 또는 sgd(원 논문 "
                              "설정, momentum=0.9, k0/longtrain류 옛 baseline 재현 목적) - optimizer를 "
                              "바꾸면 적정 LR 스케일이 10~100배 달라지므로 --lr도 반드시 그에 맞게 "
                              "재보정할 것(LR range test로)")
    parser.add_argument("--weight-decay", type=float, default=None,
                         help="기본 None=config.WEIGHT_DECAY(1e-4, SGD 기준). CBGS/CenterPoint "
                              "레시피의 AdamW는 0.01을 씀(100배 차이) - optimizer=adamw일 때는 "
                              "명시적으로 재보정할 것")
    parser.add_argument("--momentum-cycling", action="store_true",
                         help="One-Cycle의 momentum cycling(Smith 2018, CBGS/CenterPoint 표준) - "
                              "LR과 반대로 움직임(--mom-max에서 시작 -> peak LR 지점에서 --mom-min "
                              "-> 다시 --mom-max). 기본 꺼짐(LR만 cycling하던 기존 동작 유지)")
    parser.add_argument("--mom-max", type=float, default=0.95)
    parser.add_argument("--mom-min", type=float, default=0.85)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--head", default="anchor", choices=["anchor", "center"],
                         help="anchor=원 논문 RPN, center=CenterPoint식 anchor-free (model.RPNCenterHead)")
    parser.add_argument("--smoke", action="store_true", help="scene_0044만 2 epoch, CPU 파이프라인 검증용 "
                                                               "(anchor head 전용 - VoxelNetDataset이 center 타겟을 안 만듦)")
    parser.add_argument("--seed", type=int, default=0, help="재현성/시드 반복 실험용 - 이전엔 시드 고정이 "
                                                              "전혀 없었음(매 실행마다 다른 초기화)")
    parser.add_argument("--save-every", type=int, default=0,
                         help="N>0이면 매 N epoch마다 {run_name}_epoch{N}.pt 스냅샷도 저장 - "
                              "장기학습에서 수렴 지점을 epoch 단위로 촘촘히 보고 싶을 때")
    parser.add_argument("--val-every", type=int, default=1,
                         help="--cache-root 모드에서 N>0이면 N epoch마다 val AP3D를 재고 "
                              "개선 시 {run_name}_best_iou{30,35,40}.pt 갱신(0.35=타겟). 0이면 학습 중 val 끔")
    parser.add_argument("--empty-weight", type=float, default=1.0,
                         help="CBGS식 균형: <1이면 empty(배경, train_full−train) 프레임을 "
                              "WeightedRandomSampler로 상대 subsample(pos oversample). 1.0=균형 없음")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--lr-schedule", default="onecycle", choices=["step", "onecycle"],
                         help="'onecycle'(기본, 확정 레시피) - One-Cycle policy(Smith 2018), "
                              "CenterPoint/CBGS 원저자 공식 코드(tianweiy/CenterPoint)가 실제로 쓰는 "
                              "방식. 'step'은 이 레시피 이전의 옛 방식(decay_frac 지점에서 한 번에 "
                              "뚝 떨어짐, config.LR_DECAY_EPOCH_FRAC/LR_DECAY_FACTOR 고정값 사용) - "
                              "k0/longtrain류 옛 baseline 재현 목적이 아니면 쓸 일 없음")
    parser.add_argument("--lr-pct-start", type=float, default=0.4,
                         help="--lr-schedule onecycle 전용 - 총 epoch 중 앞쪽 몇 %%를 LR 상승 구간으로 "
                              "쓸지(코사인 곡선). 기본 0.4 = CenterPoint 공식 기본값 그대로")
    parser.add_argument("--lr-div-factor", type=float, default=10.0,
                         help="--lr-schedule onecycle 전용 - 초기 LR = base_lr / 이 값. 기본 10.0 = "
                              "CenterPoint 공식 기본값 그대로")
    parser.add_argument("--lr-min-frac", type=float, default=0.0,
                         help="--lr-schedule onecycle 전용 - 최종 LR = base_lr * 이 값(기본 0)")
    parser.add_argument("--focal-alpha", type=float, default=2.0,
                         help="center head 전용, gaussian_focal_loss의 focusing 지수(RetinaNet gamma "
                              "역할, pos/neg 둘 다에 적용). 기본 2.0=CornerNet/CenterNet 표준")
    parser.add_argument("--focal-beta", type=float, default=4.0,
                         help="center head 전용, gaussian_focal_loss의 negative Gaussian-peak 근접 "
                              "penalty 감쇠 지수. 기본 4.0=CornerNet/CenterNet 표준")
    parser.add_argument("--pos-weight-max", type=float, default=1.0,
                         help="center head 전용, recall-collapse 완화 실험 - gaussian_focal_loss의 "
                              "positive 항에 곱하는 가중치, --pos-weight-start-frac 지점부터 완만하게 "
                              "1.0에서 이 값까지 올라감. 기본 1.0=no-op.")
    parser.add_argument("--pos-weight-start-frac", type=float, default=0.6,
                         help="--pos-weight-max 스케줄이 시작되는 지점(전체 epoch 대비 비율). 기본 0.6")
    parser.add_argument("--lr-range-test", action="store_true",
                         help="정식 학습 대신 LR range test(Smith 2017) 실행 - --lr-range-min에서 "
                              "--lr-range-max까지 --lr-range-steps 스텝에 걸쳐 지수적으로 LR을 올려가며 "
                              "step별 loss를 {run_name}_lr_range_test.json에 기록. val/체크포인트 저장 없음 "
                              "- loss가 발산하기 시작하는 지점을 보고 적정 base_lr을 고르는 진단 전용 모드")
    parser.add_argument("--polar", action="store_true",
                         help="voxel polarization Phase1(Cylinder3D식) - cache_dataset.py --polar로 만든 "
                              "캐시를 --cache-root로 넘겨야 함. center head 전용(anchor는 미지원).")
    parser.add_argument("--use-grr", action="store_true",
                         help="PARTNER(arXiv:2308.03982) Phase2 GRR 모듈(partner_grr.py) - polar 전용, "
                              "--polar와 함께 써야 함.")
    parser.add_argument("--grr-n", type=int, default=None,
                         help="GRR 대표 feature 수 N(n_rep). R=102를 N개로 압축하므로 유효 range "
                              "해상도=R/N(N=4→셀당 ~2.55m로 근거리 버킷 구분 불가). 기본 None=config."
                              "PARTNER_GRR_N(=4, 논문 원안). radius축 희석 완화용으로 8~12 실험 "
                              "권장(project_grr_collapse_diagnosis). --use-grr와 함께만 유효.")
    parser.add_argument("--use-pdconv", action="store_true",
                         help="PVP(arXiv:2412.07616) PD-Conv(pdconv.py) - 3×3×3 Conv3D를 polar 평면별 "
                              "비대칭 2D conv 3개로 분해. polar 전용, --polar와 함께 써야 함.")
    parser.add_argument("--use-ga", action="store_true",
                         help="PARTNER(arXiv:2308.03982) §3.4 GA(Geometry-aware Adaptive) v1 minimal "
                              "(polar_ga.py) - GRR과 orthogonal, 파이프라인 GRR→GA. polar 전용.")
    parser.add_argument("--fg-range-cond", action="store_true",
                         help="ForegroundHead에 셀별 r 값을 sinusoidal encoding해 concat "
                              "(model.py:ForegroundHead) - 3-3.5m fg_gate 오탐 완화 목적. "
                              "--fg-mode joint/joint_ema와 함께만 유효.")
    parser.add_argument("--use-rs-conv", action="store_true",
                         help="PolarStream(arXiv:2106.07545) §3.3 Range-Stratified Conv+BN "
                              "(polar_stream.py) - regression head(offset/z/dim)를 radial band"
                              " 6개별 독립 conv+BN으로 대체. polar 전용.")
    parser.add_argument("--use-feat-undistort", action="store_true",
                         help="PolarStream §3.3 Feature Undistortion (polar_stream.py) - "
                              "heatmap head 앞에 위치-adaptive scale/shift. polar 전용.")
    parser.add_argument("--reg-weight", type=float, default=1.0,
                         help="regression loss 전체 스케일 factor (offset/z/dim/rot 합). "
                              "PolarStream §4.1: polar offset이 Cartesian 대비 ~2× 크기라 "
                              "reg loss weight를 0.5로 하향 권장.")
    parser.add_argument("--range-weight-mode", default="none", choices=["none", "linear", "sqrt"],
                         help="range-conditioned loss weighting - 각 셀의 loss에 range-bucket 기반 "
                              "weight 곱. 'linear'=test/train 비율 그대로, 'sqrt'=제곱근으로 완화. "
                              "sampling 대비 memorization 리스크 낮음(gradient scale만 조절).")
    parser.add_argument("--density-loss", action="store_true",
                         help="RAANet(arXiv:2111.09515)식 보조 density-level CE loss(density_head) 활성화 - "
                              "기본 off(opt-in). GRR 등 다른 축을 단독 검증할 때 density loss와 섞이지 "
                              "않도록 기본값을 꺼둔다 - 켜려면 캐시에 density_center가 있어야 함(patch_density_target.py).")
    parser.add_argument("--lr-range-min", type=float, default=1e-5)
    parser.add_argument("--lr-range-max", type=float, default=1.0)
    parser.add_argument("--lr-range-steps", type=int, default=500)
    parser.add_argument("--fg-mode", default="none", choices=["none", "joint", "freeze_fit", "joint_ema"],
                         help="Direction 3(reports_v2/foreground_aux_branch_proposal.md 4.7), center head 전용. "
                              "'joint'=fg_head를 메인 optimizer로 같이 학습(model.py ForegroundHead). "
                              "'freeze_fit'=별도 probe를 주기적으로 짧게 재학습(foreground_gate.py) - "
                              "heatmap_head처럼 긴 누적 학습 경로를 안 거치게 하려는 대안. "
                              "'joint_ema'=fg_head 학습은 joint와 동일하되 gating엔 그 EMA shadow만 사용 - "
                              "joint의 recall 보존력 + freeze_fit급 gate 안정성을 절충. 기본 'none'=기존 동작 그대로.")
    parser.add_argument("--fg-alpha-relief", type=float, default=1.0)
    parser.add_argument("--fg-r-max", type=float, default=0.5,
                         help="negative penalty를 최대 얼마나(비율) 완화할지 상한 - 완전히 0으로는 안 만듦")
    parser.add_argument("--fg-weight", type=float, default=1.0, help="'joint'/'joint_ema' 전용, fg_head 자체 BCE loss 비중")
    parser.add_argument("--fg-refit-every", type=int, default=4, help="'freeze_fit' 전용, N epoch마다 재학습")
    parser.add_argument("--fg-warmup-epochs", type=int, default=4, help="'freeze_fit' 전용, 이 전엔 gate 비활성")
    parser.add_argument("--fg-probe-lr", type=float, default=0.01)
    parser.add_argument("--fg-probe-steps", type=int, default=300)
    parser.add_argument("--fg-probe-batches", type=int, default=20)
    parser.add_argument("--fg-ema-decay", type=float, default=0.999,
                         help="'joint_ema' 전용, shadow EMA decay(클수록 느리게 따라감, ~1/(1-decay) step 평균창)")
    parser.add_argument("--fg-reliability", action="store_true",
                         help="fg_gate!=None인 모든 모드에 적용 - reliability factor(density 또는 intensity, "
                              "--fg-reliability-mode로 선택)를 relief에 곱함(proposal 7.4/7.5) - fg_gate/fg_head는 "
                              "안 건드리고 곱셈 factor만 추가.")
    parser.add_argument("--fg-reliability-mode", default="density", choices=["density", "intensity"],
                         help="density: point count 기반(모든 fg_mode 지원). intensity: 같은 range bucket 내 "
                              "배경 대비 반사 강도 기반(joint/joint_ema 전용, fg_target 필요 - proposal 7.5).")
    parser.add_argument("--fg-reg-alpha", type=float, default=0.0,
                         help="Phase1 'A2'(2026-08-28): fg-gated regression weighting. >0면 각 GT peak 셀의 "
                              "회귀 L1(offset/z/dim/rot)을 (1 + fg_reg_alpha * P_fg)로 가중(정규화도 가중치 합으로 "
                              "바꿔 스케일 보존, emphasis 재분배). fg_gate가 있는 모드(joint/joint_ema/freeze_fit)에서만 "
                              "의미. 0=off(기존과 동일). 상세: study_v2/fg_a2_regression_weighting.html")
    parser.add_argument("--use-stage2", action="store_true",
                         help="CenterPoint stage-2(RoI refinement, reports_v2/foreground_aux_branch_proposal.md "
                              "6.3) - GT jitter로 만든 pseudo-candidate에서 BEV RoI feature를 뽑아 refined "
                              "score/box를 학습. feat는 detach 후 샘플링(backbone 역전파 없음, center head 전용).")
    parser.add_argument("--stage2-weight", type=float, default=1.0, help="stage2 loss 전체 가중치")
    parser.add_argument("--stage2-score-weight", type=float, default=1.0, help="stage2 score(BCE) vs box residual 상대 가중치")
    parser.add_argument("--stage2-n-per-gt", type=int, default=8, help="GT 1개당 만드는 jittered candidate 개수")
    parser.add_argument("--stage2-pos-noise-frac", type=float, default=0.3, help="위치 노이즈, 박스 l/w 비례")
    parser.add_argument("--stage2-rot-noise-deg", type=float, default=15.0, help="회전 노이즈(도)")
    parser.add_argument("--stage2-dim-noise-frac", type=float, default=0.15, help="크기 노이즈 비율")
    parser.add_argument("--stage2-n-iou-samples", type=int, default=300,
                         help="score target용 3D IoU Monte-Carlo 샘플 수(eval 기본 8000보다 낮춰 throughput 확보)")
    parser.add_argument("--stage2-candidate-source", default="decoded", choices=["decoded", "gt_jitter"],
                         help="decoded(기본, 2026-08-20부터): 실제 1st-stage 디코딩 결과를 후보로 씀"
                              "(CenterPoint 원 논문 방식). gt_jitter: 이전 방식(GT+노이즈), 비교/폴백용.")
    parser.add_argument("--stage2-score-thresh", type=float, default=0.05,
                         help="'decoded' 후보 소스 전용 - 이 threshold 이상 heatmap peak만 stage2 후보로 씀")
    parser.add_argument("--stage2-max-peaks", type=int, default=32,
                         help="'decoded' 후보 소스 전용 - 프레임당 stage2 후보 상한(compute 제한용)")
    parser.add_argument("--stage2-fg-iou-thresh", type=float, default=0.55,
                         help="'decoded' 후보 소스 전용 - 이 IoU 이상이면 FG(PV-RCNN 관례), "
                              "미만은 BG로 분류해 FG/BG 비율 서브샘플링에 씀")
    parser.add_argument("--stage2-fg-ratio", type=float, default=0.5,
                         help="'decoded' 후보 소스 전용 - 프레임당 목표 FG 비율(FG는 항상 다 씀, "
                              "BG를 이 비율에 맞춰 서브샘플링) - 공식 CenterPoint stage-2와 정합성 맞추려는 목적")
    parser.add_argument("--use-rot-branch", action="store_true",
                         help="dense(z 보존)에서 갈라진 전용 Conv3D branch를 rot_head에 0-init concat "
                              "(reports_v2/foreground_aux_branch_proposal.md §1.4.4) - center head 전용")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda가 요청됐지만 torch.cuda.is_available()==False. "
            "torch 자체에 CUDA 빌드가 없으면(AssertionError: Torch not compiled with CUDA "
            "enabled) Colab의 Runtime > Change runtime type이 GPU가 아닌 상태(CPU/TPU)로 "
            "torch가 설치됐다는 뜻 - GPU(T4)로 바꾸고 런타임을 재시작한 뒤(로컬 디스크가 "
            "초기화되므로 데이터 복사 셀부터 다시 실행) 재시도할 것.")

    if args.smoke:
        assert args.head == "anchor", "--smoke는 VoxelNetDataset(로컬 즉석계산) 기반이라 anchor head만 지원"
        dataset, epochs, batch_size = VoxelNetDataset(["scene_0044"]), 2, 2
    elif args.on_the_fly:
        assert args.head == "center", "--on-the-fly는 center head 전용"
        assert args.annotations_dir and args.points_root and args.splits_json, \
            "--on-the-fly는 --annotations-dir/--points-root/--splits-json 필요"
        from dataset_onthefly import OnTheFlyVoxelDataset
        dataset = OnTheFlyVoxelDataset(args.annotations_dir, args.points_root, args.splits_json,
                                        split="train", strong=args.otf_strong,
                                        exclude={("scene_0042", 83)},  # 원본 빈-voxel 프레임 제외
                                        include_raw=args.otf_include_raw)
        epochs, batch_size = args.epochs, args.batch_size
        print(f"[on-the-fly] train samples={len(dataset)} strong={args.otf_strong} "
              f"include_raw={args.otf_include_raw} "
              f"gt_db={len(dataset.gt_db) if dataset.gt_db else 0}", flush=True)
    elif args.cache_root:
        dataset = CachedVoxelNetDataset(args.cache_root, args.split, head=args.head,
                                         load_gt_boxes=(args.fg_mode != "none" or args.use_stage2))
        epochs, batch_size = args.epochs, args.batch_size
    else:
        assert args.head == "anchor", "--cache-root 없이 로컬 즉석계산은 anchor head만 지원"
        scenes = args.scenes if args.scenes else load_scene_split("train")
        dataset = VoxelNetDataset(scenes)
        epochs, batch_size = args.epochs, args.batch_size

    if args.lr_range_test:
        run_lr_range_test(args.run_name, dataset, batch_size, args.device, head=args.head,
                           num_workers=args.num_workers, lr_min=args.lr_range_min,
                           lr_max=args.lr_range_max, num_steps=args.lr_range_steps,
                           optimizer_name=args.optimizer, polar=args.polar, use_grr=args.use_grr)
        return

    val_cache_root = args.cache_root if (args.cache_root and args.val_every > 0) else None
    train(args.run_name, dataset, epochs, batch_size, args.device, head=args.head,
          num_workers=args.num_workers, save_every=args.save_every, empty_weight=args.empty_weight,
          val_cache_root=val_cache_root, val_split=args.val_split, val_every=args.val_every,
          lr_schedule=args.lr_schedule,
          lr_pct_start=args.lr_pct_start, lr_div_factor=args.lr_div_factor, lr_min_frac=args.lr_min_frac,
          focal_alpha=args.focal_alpha, focal_beta=args.focal_beta,
          base_lr=args.lr, optimizer_name=args.optimizer, weight_decay=args.weight_decay,
          momentum_cycling=args.momentum_cycling, mom_max=args.mom_max, mom_min=args.mom_min,
          polar=args.polar, use_grr=args.use_grr, use_density_loss=args.density_loss,
          pos_weight_max=args.pos_weight_max, pos_weight_start_frac=args.pos_weight_start_frac,
          fg_mode=args.fg_mode, fg_alpha_relief=args.fg_alpha_relief, fg_r_max=args.fg_r_max,
          fg_weight=args.fg_weight, fg_refit_every=args.fg_refit_every,
          fg_warmup_epochs=args.fg_warmup_epochs, fg_probe_lr=args.fg_probe_lr,
          fg_probe_steps=args.fg_probe_steps, fg_probe_batches=args.fg_probe_batches,
          fg_ema_decay=args.fg_ema_decay, fg_reliability=args.fg_reliability,
          fg_reliability_mode=args.fg_reliability_mode, fg_reg_alpha=args.fg_reg_alpha,
          use_stage2=args.use_stage2, stage2_weight=args.stage2_weight,
          stage2_score_weight=args.stage2_score_weight, stage2_n_per_gt=args.stage2_n_per_gt,
          stage2_pos_noise_frac=args.stage2_pos_noise_frac, stage2_rot_noise_deg=args.stage2_rot_noise_deg,
          stage2_dim_noise_frac=args.stage2_dim_noise_frac, stage2_n_iou_samples=args.stage2_n_iou_samples,
          stage2_candidate_source=args.stage2_candidate_source, stage2_score_thresh=args.stage2_score_thresh,
          stage2_max_peaks=args.stage2_max_peaks, stage2_fg_iou_thresh=args.stage2_fg_iou_thresh,
          stage2_fg_ratio=args.stage2_fg_ratio, seed=args.seed, use_rot_branch=args.use_rot_branch,
          use_pdconv=args.use_pdconv, grr_n=args.grr_n, use_ga=args.use_ga,
          fg_range_cond=args.fg_range_cond,
          range_weight_mode=args.range_weight_mode,
          use_rs_conv=args.use_rs_conv, use_feat_undistort=args.use_feat_undistort,
          reg_weight=args.reg_weight)


if __name__ == "__main__":
    main()
