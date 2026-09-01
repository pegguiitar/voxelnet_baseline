"""eval_range_threshold_calibration.py - Day1 1b 검증: range_threshold_calibration.json에서
역산한 range-조건부 threshold를 decode_boxes()에 적용했을 때, flat threshold(0.5) baseline
대비 실제로 precision/recall/FP:TP가 개선되는지 확인한다. 모델은 그대로(재학습 없음) -
순수 후처리 비교라 CPU로도 충분(Modal 불필요).

한 번의 forward pass 결과(cls_pred/reg_pred)에 대해 baseline과 range-조건부, 두 threshold
정책을 각각 decode해서 같은 프레임에서 바로 비교한다.

Usage:
    python eval_range_threshold_calibration.py --ckpt ../checkpoints/voxelnet_run1.pt \
        --cache-root ../cache/voxel --split val
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import anchors as anchors_mod
import config
from decode import decode_boxes, rotated_nms
from eval_voxelnet import gt_obb_from_row, iou_3d_obb, iter_cached, pred_obb_from_box
from model import VoxelNet, load_state_dict_compat

BUCKETS = [("0-2m", 0, 2), ("2-2.5m", 2, 2.5), ("2.5-3m", 2.5, 3), ("3-3.5m", 3, 3.5),
           ("3.5-5m", 3.5, 5), ("5m+", 5, float("inf"))]
IOU_THRESH = 0.25
BASELINE_THRESH = 0.5  # fp_pattern_analysis.json이 이 기준으로 만들어졌으므로 그대로 맞춤


def bucket_of(rng: float) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= rng < hi:
            return name
    return BUCKETS[-1][0]


def match_and_tally(boxes, gt_boxes, nms_iou=0.1):
    """boxes(decode_boxes 출력)를 NMS+GT매칭 후, range bucket별 TP/FP 카운트 dict 반환."""
    filtered = rotated_nms(boxes, iou_thresh=nms_iou)
    gt_list = [gt_obb_from_row(row) for row in gt_boxes]
    gt_ranges = [float(np.hypot(row[0], row[1])) for row in gt_boxes]

    pred_list = [pred_obb_from_box(b) for b in filtered]
    pred_ranges = [float(np.hypot(b["x"], b["y"])) for b in filtered]

    n_p, n_g = len(pred_list), len(gt_list)
    iou_mat = np.zeros((n_p, n_g))
    rng = np.random.default_rng(0)  # 프레임마다 고정 시드 - 재현성
    for pi, (pc, pd, pR) in enumerate(pred_list):
        for gi, (gc, gd, gR) in enumerate(gt_list):
            iou_mat[pi, gi] = iou_3d_obb(pc, pd, pR, gc, gd, gR, rng=rng)

    order = np.argsort(-np.array([b["score"] for b in filtered])) if n_p else np.array([], dtype=int)
    matched_gt = set()
    stats = {b[0]: {"tp": 0, "fp": 0} for b in BUCKETS}
    for pi in order:
        candidate_gt = [gi for gi in range(n_g) if gi not in matched_gt]
        is_tp = False
        if candidate_gt:
            ious = iou_mat[pi, candidate_gt]
            best_local = int(np.argmax(ious))
            best_gt, best_iou = candidate_gt[best_local], ious[best_local]
            if best_iou >= IOU_THRESH:
                matched_gt.add(best_gt)
                is_tp = True
        if is_tp:
            stats[bucket_of(gt_ranges[best_gt])]["tp"] += 1
        else:
            stats[bucket_of(pred_ranges[pi])]["fp"] += 1
    gt_by_bucket = {b[0]: 0 for b in BUCKETS}
    for r in gt_ranges:
        gt_by_bucket[bucket_of(r)] += 1
    return stats, gt_by_bucket


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--calibration", default=str(config.VOXELNET_ROOT / "reports" / "range_threshold_calibration.json"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    calib = json.load(open(args.calibration))
    # 5m+는 목표(FP:TP~1.5) 도달 못 함 + recall 손실만 큼 -> baseline 그대로 유지(안 올림)
    range_thresholds = []
    for name, lo, hi in BUCKETS:
        t = BASELINE_THRESH if name == "5m+" else calib[name]["threshold"]
        range_thresholds.append((hi, t))
    print("적용할 range_thresholds:", range_thresholds)

    ckpt = torch.load(args.ckpt, map_location=args.device)
    head = ckpt.get("head", "anchor")
    assert head == "anchor", "1b는 anchor head 대상 - decode_boxes()는 anchor 전용"
    model = VoxelNet(head=head).to(args.device)
    load_state_dict_compat(model, ckpt["model"])
    model.eval()

    anchor_grid = anchors_mod.build_anchor_grid()
    samples = iter_cached(args.cache_root, args.split)
    with open(Path(args.cache_root) / "manifest.json") as f:
        total = len(json.load(f)[args.split])

    from tqdm import tqdm
    baseline_stats = {b[0]: {"tp": 0, "fp": 0} for b in BUCKETS}
    range_stats = {b[0]: {"tp": 0, "fp": 0} for b in BUCKETS}
    gt_totals = {b[0]: 0 for b in BUCKETS}

    with torch.no_grad():
        for sample_id, voxel_features, coords, num_points, gt_boxes in tqdm(samples, total=total, desc="calib-eval"):
            voxel_features_t = torch.from_numpy(voxel_features).to(args.device)
            num_points_t = torch.from_numpy(num_points).to(args.device)
            batch_col = torch.zeros((len(coords), 1), dtype=torch.int64)
            coords_t = torch.cat([batch_col, torch.from_numpy(coords)], dim=1).to(args.device)

            cls_pred, reg_pred = model(voxel_features_t, num_points_t, coords_t)

            base_boxes = decode_boxes(cls_pred[0], reg_pred[0], anchor_grid, score_thresh=BASELINE_THRESH)
            range_boxes = decode_boxes(cls_pred[0], reg_pred[0], anchor_grid, range_thresholds=range_thresholds)

            b_stats, gt_b = match_and_tally(base_boxes, gt_boxes)
            r_stats, _ = match_and_tally(range_boxes, gt_boxes)
            for name, _, _ in BUCKETS:
                for k in ("tp", "fp"):
                    baseline_stats[name][k] += b_stats[name][k]
                    range_stats[name][k] += r_stats[name][k]
                gt_totals[name] += gt_b[name]

    print(f"\n{'bucket':8} {'gt':>5} | {'base TP':>7} {'base FP':>7} {'base FP:TP':>10} {'base P':>7} {'base R':>7} | "
          f"{'new TP':>7} {'new FP':>7} {'new FP:TP':>9} {'new P':>7} {'new R':>7}")
    out = {"baseline_thresh": BASELINE_THRESH, "range_thresholds": range_thresholds, "buckets": {}}
    tot_b_tp = tot_b_fp = tot_r_tp = tot_r_fp = tot_gt = 0
    for name, _, _ in BUCKETS:
        gt = gt_totals[name]
        b = baseline_stats[name]; r = range_stats[name]
        b_fptp = b["fp"] / b["tp"] if b["tp"] else float("inf")
        r_fptp = r["fp"] / r["tp"] if r["tp"] else float("inf")
        b_p = b["tp"] / (b["tp"] + b["fp"]) if (b["tp"] + b["fp"]) else float("nan")
        r_p = r["tp"] / (r["tp"] + r["fp"]) if (r["tp"] + r["fp"]) else float("nan")
        b_r = b["tp"] / gt if gt else float("nan")
        r_r = r["tp"] / gt if gt else float("nan")
        print(f"{name:8} {gt:5} | {b['tp']:7} {b['fp']:7} {b_fptp:10.2f} {b_p:7.3f} {b_r:7.3f} | "
              f"{r['tp']:7} {r['fp']:7} {r_fptp:9.2f} {r_p:7.3f} {r_r:7.3f}")
        out["buckets"][name] = {"gt": gt, "baseline": {**b, "fptp": b_fptp, "precision": b_p, "recall": b_r},
                                 "range_conditioned": {**r, "fptp": r_fptp, "precision": r_p, "recall": r_r}}
        tot_b_tp += b["tp"]; tot_b_fp += b["fp"]; tot_r_tp += r["tp"]; tot_r_fp += r["fp"]; tot_gt += gt

    print(f"\n전체: baseline TP={tot_b_tp} FP={tot_b_fp} P={tot_b_tp/(tot_b_tp+tot_b_fp):.3f} R={tot_b_tp/tot_gt:.3f} | "
          f"range TP={tot_r_tp} FP={tot_r_fp} P={tot_r_tp/(tot_r_tp+tot_r_fp):.3f} R={tot_r_tp/tot_gt:.3f}")

    out_path = config.VOXELNET_ROOT / "reports" / f"range_threshold_validation_{Path(args.ckpt).stem}_{args.split}.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
