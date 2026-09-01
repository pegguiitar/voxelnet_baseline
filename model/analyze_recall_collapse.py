"""analyze_recall_collapse.py - Stage 0 진단 스크립트 (reports_v2/foreground_aux_branch_proposal.md 참고).

기존 파일(model.py/center_loss.py/decode.py/heatmap_targets.py)은 전혀 수정하지 않는다 -
전부 여기서 재사용/조립만 한다. 4가지를 epoch(checkpoint)별로 계측한다:

  A. score threshold별 recall/precision (중심거리 매칭 - 이유는 아래 NOTE 참고)
  B. GT 배정 셀의 heatmap confidence 분포
  C. raw heatmap peak 개수 / threshold 통과 개수
  D. positive/negative heatmap focal loss 분리
  + VFE / feat2d(post-scatter, pre-backbone) / feat(post-backbone) 3-point linear probe
    (foreground/background separability가 어느 layer에서 무너지는지 localize)

NOTE(회전 미사용 이유): `voxelnet_center_onecycle40_k0_s0`는 rotation3d 확장 이전
체크포인트라 rot_head 채널 수(2 vs 현재 6)가 안 맞는다. load_state_dict_compat()의
strict=False도 "이름은 같은데 shape이 다른 키"는 못 건너뛰므로(PyTorch 자체 제약),
이 스크립트는 shape이 일치하는 키만 골라 로드하고(rot_head는 랜덤 초기화로 남김) IoU
기반 매칭 대신 중심거리 매칭을 쓴다 - 이러면 회전 예측 품질과 무관하게 "GT 근처에
confident한 박스가 있는가"만 본다. 오히려 지금 진단 목적(위치/confidence 문제인지
box-fit 문제인지 분리)에는 IoU보다 이 방식이 더 적합하다.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import config
import rotation3d
from model import VoxelNet

DEFAULT_CKPT_DIR = config.CHECKPOINT_DIR
DEFAULT_RUN = "voxelnet_center_onecycle40_k0_s0"
DEFAULT_EPOCHS = (4, 9, 14, 19, 24, 29, 34)
SCORE_THRESHES = (0.05, 0.1, 0.2, 0.3, 0.5)
MATCH_RADIUS_M = 1.0  # 중심거리 매칭 반경 - 실측 dims 평균(l=1.57,w=1.02) 기준 대략 한 물체 크기
PROBE_MAX_PER_CLASS = 30000  # checkpoint당 tap point별 probe 학습에 쓸 최대 negative 샘플 수
PROBE_MAX_NEG_PER_FRAME = 150  # 프레임당 즉시 subsample 상한(메모리 보호, 이 머신 RAM 7.5GB 기준)


# ---------------------------------------------------------------- checkpoint 로딩

def load_compat_ignore_shape_mismatch(model: torch.nn.Module, state_dict: dict) -> list:
    """load_state_dict_compat()의 strict=False가 "shape이 다른 동명 키"는 못 건너뛰는
    PyTorch 자체 제약을 우회 - shape이 맞는 키만 필터링해서 로드. 나머지는 랜덤 초기화로
    남는다(rot_head처럼 채널 수 자체가 바뀐 head). 반환: 드롭된 키 목록."""
    own = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in own and own[k].shape == v.shape}
    dropped = [k for k in state_dict if k not in filtered]
    model.load_state_dict(filtered, strict=False)
    return dropped


def load_model_for_analysis(ckpt_path: Path) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = VoxelNet(head=ckpt.get("head", "center"), polar=ckpt.get("polar", False),
                      use_grr=ckpt.get("use_grr", False))
    dropped = load_compat_ignore_shape_mismatch(model, ckpt["model"])
    if dropped:
        print(f"  [load] shape mismatch로 랜덤 초기화 유지: {dropped}")
    model.eval()
    return model


# ---------------------------------------------------------------- 좌표/기하 유틸

def voxel_world_centers_xyz(coords_zyx: np.ndarray) -> np.ndarray:
    """coords: (K,3) [z_idx,y_idx,x_idx] (voxelize.py 관례) -> (K,3) world [x,y,z]."""
    pc = config.POINT_CLOUD_RANGE
    vx, vy, vz = config.VOXEL_SIZE
    x = pc[0] + (coords_zyx[:, 2].astype(np.float32) + 0.5) * vx
    y = pc[1] + (coords_zyx[:, 1].astype(np.float32) + 0.5) * vy
    z = pc[2] + (coords_zyx[:, 0].astype(np.float32) + 0.5) * vz
    return np.stack([x, y, z], axis=1)


def cell_centers_xy(h: int, w: int) -> tuple:
    """(H,W) 그리드(어느 해상도든, 텐서 shape에서 그대로 유도) 각 셀의 world (x,y) 중심.
    -> (xx,yy) 각 (H,W)."""
    pc = config.POINT_CLOUD_RANGE
    sx, sy = (pc[3] - pc[0]) / w, (pc[4] - pc[1]) / h
    xs = pc[0] + (np.arange(w, dtype=np.float32) + 0.5) * sx
    ys = pc[1] + (np.arange(h, dtype=np.float32) + 0.5) * sy
    return np.meshgrid(xs, ys)  # 각 (H,W)


def point_in_any_obb_3d(points_xyz: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """VFE(voxel-level, z 보존) foreground label - 진짜 3D OBB 내부 판정."""
    inside = np.zeros(len(points_xyz), dtype=bool)
    for row in gt_boxes:
        center, dims = row[0:3], row[3:6]
        R = rotation3d.sixd_to_matrix_np(row[7:13])
        local = (points_xyz - center) @ R
        inside |= np.all(np.abs(local) <= dims / 2, axis=1)
    return inside


def point_in_any_obb_2d(xy: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """feat2d/feat(BEV, z-collapse 이후) foreground label - 2D footprint(z-yaw) 판정만,
    z는 이미 이 지점에서 정보가 섞여있으므로 BEV 배치 관례(heatmap target 생성과 동일)를 따름."""
    inside = np.zeros(len(xy), dtype=bool)
    for row in gt_boxes:
        cx, cy, l, w, theta = row[0], row[1], row[3], row[4], row[6]
        c, s = np.cos(theta), np.sin(theta)
        dx, dy = xy[:, 0] - cx, xy[:, 1] - cy
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        inside |= (np.abs(lx) <= l / 2) & (np.abs(ly) <= w / 2)
    return inside


# ---------------------------------------------------------------- A: 중심거리 기반 threshold sweep

def decode_all_peaks(hm_sigmoid: np.ndarray, off: np.ndarray, z: np.ndarray) -> list:
    """score_thresh 없이 3x3 local-max peak 전부 decode (x,y,z,score)만 - 회전/크기 불필요."""
    hmap = hm_sigmoid[0]
    h, w = hmap.shape
    padded = np.full((h + 2, w + 2), -1.0, dtype=hmap.dtype)
    padded[1:-1, 1:-1] = hmap
    is_peak = np.ones((h, w), dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            is_peak &= hmap >= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    rows, cols = np.where(is_peak)
    sx, sy = config.ANCHOR_STRIDE
    x0, y0 = config.POINT_CLOUD_RANGE[0], config.POINT_CLOUD_RANGE[1]
    peaks = []
    for r, c in zip(rows, cols):
        dx, dy = off[r, c]
        x = x0 + (c + dx + 0.5) * sx
        y = y0 + (r + dy + 0.5) * sy
        peaks.append((float(hmap[r, c]), x, y, float(z[r, c, 0])))
    return peaks, int(is_peak.sum())


def match_by_distance(peaks: list, gt_xyz: np.ndarray, score_thresh: float, radius: float) -> list:
    """score_thresh 이상 peak만 confidence 내림차순으로 greedy 중심거리 매칭.
    반환: [(score,is_tp), ...]."""
    cand = [p for p in peaks if p[0] >= score_thresh]
    cand.sort(key=lambda p: -p[0])
    matched = set()
    out = []
    for score, x, y, z in cand:
        is_tp = False
        if len(gt_xyz):
            d = np.linalg.norm(gt_xyz - np.array([x, y, z]), axis=1)
            d[list(matched)] = np.inf
            j = int(np.argmin(d))
            if d[j] <= radius:
                matched.add(j)
                is_tp = True
        out.append((score, is_tp))
    return out


def compute_pr(dets: list, n_gt: int) -> tuple:
    if n_gt == 0:
        return 0.0, 0.0
    tp = sum(1 for _, t in dets if t)
    fp = len(dets) - tp
    recall = tp / n_gt
    precision = tp / max(tp + fp, 1)
    return precision, recall


# ---------------------------------------------------------------- D: pos/neg loss 분리 (center_loss.gaussian_focal_loss와 동일 수식, 분리만)

def split_focal_loss(pred_logit: torch.Tensor, target: torch.Tensor, alpha=2.0, beta=4.0) -> tuple:
    pred = torch.sigmoid(pred_logit)
    pos_mask = (target == 1).float()
    neg_mask = (target < 1).float()
    neg_weight = torch.pow(1 - target, beta)
    log_p = F.logsigmoid(pred_logit)
    log_1mp = F.logsigmoid(-pred_logit)
    pos_loss = -log_p * torch.pow(1 - pred, alpha) * pos_mask
    neg_loss = -log_1mp * torch.pow(pred, alpha) * neg_weight * neg_mask
    n_pos = pos_mask.sum().clamp_min(1)
    return (pos_loss.sum() / n_pos).item(), (neg_loss.sum() / n_pos).item()


# ---------------------------------------------------------------- linear probe (torch, sklearn 없이)

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    n_pos, n_neg = int(labels.sum()), int((~labels.astype(bool)).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    sum_ranks_pos = ranks[labels.astype(bool)].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def train_probe_and_auc(feats: np.ndarray, labels: np.ndarray, seed: int = 0,
                         n_steps: int = 300, lr: float = 0.05) -> dict:
    """단일 Linear layer + BCE, 80/20 train/test split. feats: (N,C), labels: (N,) {0,1}."""
    n = len(labels)
    n_pos, n_neg = int(labels.sum()), n - int(labels.sum())
    if n_pos < 20 or n_neg < 20:
        return {"auroc_test": float("nan"), "n": n, "n_pos": n_pos, "n_neg": n_neg}

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(int(n * 0.2), 1)
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    x = torch.from_numpy(feats).float()
    y = torch.from_numpy(labels).float()
    mu, sigma = x[train_idx].mean(0, keepdim=True), x[train_idx].std(0, keepdim=True).clamp_min(1e-6)
    x = (x - mu) / sigma

    pos_weight = torch.tensor([max(n_neg / max(n_pos, 1), 1.0)])
    linear = torch.nn.Linear(x.shape[1], 1)
    opt = torch.optim.Adam(linear.parameters(), lr=lr)
    x_tr, y_tr = x[train_idx], y[train_idx]
    for _ in range(n_steps):
        opt.zero_grad()
        logit = linear(x_tr).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logit, y_tr, pos_weight=pos_weight)
        loss.backward()
        opt.step()

    with torch.no_grad():
        test_logit = linear(x[test_idx]).squeeze(-1).numpy()
    return {"auroc_test": auroc(test_logit, labels[test_idx]), "n": n, "n_pos": n_pos, "n_neg": n_neg}


def subsample_balanced(feats: list, labels: list, max_per_class: int, seed: int) -> tuple:
    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)
    rng = np.random.default_rng(seed)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    if len(pos_idx) > max_per_class:
        pos_idx = rng.choice(pos_idx, max_per_class, replace=False)
    if len(neg_idx) > max_per_class:
        neg_idx = rng.choice(neg_idx, max_per_class, replace=False)
    idx = np.concatenate([pos_idx, neg_idx])
    return feats[idx], labels[idx]


def subsample_frame(feat: np.ndarray, label: np.ndarray, max_neg: int, rng: np.random.Generator) -> tuple:
    """프레임 하나 분량의 (feat,label)을 즉시 축소 - 전체 프레임을 다 쌓아두면
    feat2d(12000셀)/feat(3000셀) x 300프레임이 수십 GB로 불어나 이 머신(RAM 7.5GB)에서
    OOM이 난다(첫 시도에서 실측). positive는 전부 보존, negative만 프레임당 상한을 둔다."""
    pos_mask = label == 1
    neg_idx = np.where(~pos_mask)[0]
    if len(neg_idx) > max_neg:
        neg_idx = rng.choice(neg_idx, max_neg, replace=False)
    idx = np.concatenate([np.where(pos_mask)[0], neg_idx])
    return feat[idx], label[idx]


# ---------------------------------------------------------------- 메인 per-checkpoint 처리

def analyze_checkpoint(ckpt_path: Path, cache_root: Path, split: str, n_frames: int, seed: int) -> dict:
    print(f"[{ckpt_path.name}] loading...")
    model = load_model_for_analysis(ckpt_path)

    captured = {}
    h_pre = model.rpn.backbone.register_forward_pre_hook(lambda m, inp: captured.__setitem__("feat2d", inp[0]))
    h_post = model.rpn.backbone.register_forward_hook(lambda m, inp, out: captured.__setitem__("feat", out))

    with open(cache_root / "manifest.json") as f:
        entries = json.load(f)[split]
    if n_frames < len(entries):
        rng = np.random.default_rng(seed)
        entries = [entries[i] for i in sorted(rng.choice(len(entries), n_frames, replace=False))]

    gt_conf_all, det_by_thresh = [], {t: [] for t in SCORE_THRESHES}
    n_gt_total, raw_peak_counts = 0, []
    pos_losses, neg_losses = [], []
    probe_feats = {"vfe": [], "feat2d": [], "feat": []}
    probe_labels = {"vfe": [], "feat2d": [], "feat": []}
    frame_rng = np.random.default_rng(seed)

    t0 = time.time()
    with torch.no_grad():
        for i, rel in enumerate(entries):
            with np.load(cache_root / rel) as npz:
                vf = torch.from_numpy(npz["voxel_features"])
                npnt = torch.from_numpy(npz["num_points"])
                coords = torch.from_numpy(npz["coords"])
                gt_boxes = npz["gt_boxes"]
                heatmap_t = torch.from_numpy(npz["heatmap"]).unsqueeze(0)
                reg_mask = npz["reg_mask"]

            batch_col = torch.zeros((len(coords), 1), dtype=torch.int64)
            coords_b = torch.cat([batch_col, coords], dim=1)

            vfe_out = model.vfe(vf, npnt)  # (K,128) - hook 없이 직접 호출, forward()와 별개 그래프지만 읽기 전용이라 안전
            hm, off, z, dim_, rot, density = model(vf, npnt, coords_b)

            hm_sig = torch.sigmoid(hm[0]).numpy()  # (1,H,W)
            reg_mask_t = reg_mask.astype(bool)
            if reg_mask_t.any():
                gt_conf_all.extend(hm_sig[0][reg_mask_t].tolist())

            peaks, n_raw_peaks = decode_all_peaks(hm_sig, off[0].permute(1, 2, 0).numpy(),
                                                   z[0].permute(1, 2, 0).numpy())
            raw_peak_counts.append(n_raw_peaks)
            gt_xyz = gt_boxes[:, 0:3] if len(gt_boxes) else np.zeros((0, 3))
            n_gt_total += len(gt_boxes)
            for thr in SCORE_THRESHES:
                det_by_thresh[thr].extend(match_by_distance(peaks, gt_xyz, thr, MATCH_RADIUS_M))

            pos_l, neg_l = split_focal_loss(hm[0], heatmap_t[0])
            pos_losses.append(pos_l)
            neg_losses.append(neg_l)

            # --- probe features/labels (프레임마다 즉시 subsample - 안 그러면 300프레임 누적 시
            # feat2d/feat이 수십 GB로 불어나 이 머신(RAM 7.5GB)에서 OOM 남, 첫 시도에서 실측 확인) ---
            coords_np = coords.numpy()
            vfe_xyz = voxel_world_centers_xyz(coords_np)
            vfe_label = point_in_any_obb_3d(vfe_xyz, gt_boxes).astype(np.float32)
            f, l = subsample_frame(vfe_out.numpy(), vfe_label, PROBE_MAX_NEG_PER_FRAME, frame_rng)
            probe_feats["vfe"].append(f)
            probe_labels["vfe"].append(l)

            for tap in ("feat2d", "feat"):
                t = captured[tap][0]  # (C,H,W)
                c, h, w = t.shape
                xx, yy = cell_centers_xy(h, w)
                flat_feat = t.permute(1, 2, 0).reshape(h * w, c).numpy()
                flat_xy = np.stack([xx.ravel(), yy.ravel()], axis=1)
                flat_label = point_in_any_obb_2d(flat_xy, gt_boxes).astype(np.float32)
                f, l = subsample_frame(flat_feat, flat_label, PROBE_MAX_NEG_PER_FRAME, frame_rng)
                probe_feats[tap].append(f)
                probe_labels[tap].append(l)

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(entries)} frames ({time.time() - t0:.0f}s)")

    h_pre.remove()
    h_post.remove()

    result = {
        "checkpoint": ckpt_path.name,
        "n_frames": len(entries),
        "n_gt": n_gt_total,
        "gt_center_confidence": {
            "mean": float(np.mean(gt_conf_all)) if gt_conf_all else None,
            "median": float(np.median(gt_conf_all)) if gt_conf_all else None,
            "p10": float(np.percentile(gt_conf_all, 10)) if gt_conf_all else None,
        },
        "raw_peak_count_per_frame_mean": float(np.mean(raw_peak_counts)),
        "pos_neg_loss": {"pos_loss_mean": float(np.mean(pos_losses)), "neg_loss_mean": float(np.mean(neg_losses))},
        "threshold_sweep": {},
        "linear_probe": {},
    }
    for thr in SCORE_THRESHES:
        p, r = compute_pr(det_by_thresh[thr], n_gt_total)
        result["threshold_sweep"][str(thr)] = {"precision": p, "recall": r, "n_dets": len(det_by_thresh[thr])}

    for tap in ("vfe", "feat2d", "feat"):
        feats, labels = subsample_balanced(probe_feats[tap], probe_labels[tap], PROBE_MAX_PER_CLASS, seed)
        result["linear_probe"][tap] = train_probe_and_auc(feats, labels, seed=seed)
        print(f"  probe[{tap}] AUROC={result['linear_probe'][tap]['auroc_test']:.4f} "
              f"(n_pos={result['linear_probe'][tap]['n_pos']}, n_neg={result['linear_probe'][tap]['n_neg']})")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=DEFAULT_RUN)
    parser.add_argument("--epochs", type=int, nargs="*", default=list(DEFAULT_EPOCHS))
    parser.add_argument("--include-best", action="store_true", default=True)
    parser.add_argument("--cache-root", default=str(config.VOXELNET_ROOT / "cache" / "voxel"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--n-frames", type=int, default=300, help="val subsample 크기(고정 seed)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ckpt_names = [f"{args.run_name}_epoch{e:03d}.pt" for e in args.epochs]
    if args.include_best:
        ckpt_names.append(f"{args.run_name}_best.pt")

    results = []
    for name in ckpt_names:
        path = DEFAULT_CKPT_DIR / name
        if not path.exists():
            print(f"skip (없음): {path}")
            continue
        results.append(analyze_checkpoint(path, Path(args.cache_root), args.split, args.n_frames, args.seed))

    out_path = Path(args.out) if args.out else (config.VOXELNET_ROOT / "reports" / f"{args.run_name}_stage0_analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {out_path}")

    print(f"\n{'checkpoint':<45} {'GTconf_mean':>12} {'peaks/fr':>9} {'pos_L':>7} {'neg_L':>7} "
          f"{'R@.05':>6} {'R@.3':>6} {'R@.5':>6} {'AUROC_vfe':>10} {'AUROC_feat2d':>13} {'AUROC_feat':>11}")
    for r in results:
        ts = r["threshold_sweep"]
        lp = r["linear_probe"]
        print(f"{r['checkpoint']:<45} {r['gt_center_confidence']['mean'] or float('nan'):>12.4f} "
              f"{r['raw_peak_count_per_frame_mean']:>9.1f} {r['pos_neg_loss']['pos_loss_mean']:>7.3f} "
              f"{r['pos_neg_loss']['neg_loss_mean']:>7.3f} {ts['0.05']['recall']:>6.3f} {ts['0.3']['recall']:>6.3f} "
              f"{ts['0.5']['recall']:>6.3f} {lp['vfe']['auroc_test']:>10.4f} {lp['feat2d']['auroc_test']:>13.4f} "
              f"{lp['feat']['auroc_test']:>11.4f}")


if __name__ == "__main__":
    main()
