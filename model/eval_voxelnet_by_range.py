"""eval_voxelnet_by_range.py - eval_voxelnet.py와 동일한 3D IoU 매칭을 range
bucket별로 쪼개서 본다. Triband_BEV/model/eval_3d_iou_by_range.py와 버킷 경계·
지표(GT range 기준 recall, 예측 range 기준 TP/FP)를 맞춰서 두 모델을 직접
비교할 수 있게 했다.

Usage:
    python eval_voxelnet_by_range.py --ckpt ../checkpoints/voxelnet_longtrain_s0_epoch034.pt \
        --cache-root ../cache/voxel --split val --out-name range_breakdown_voxelnet_s0.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import anchors as anchors_mod
import config
import stage2_refine as s2
from decode import decode_boxes, rotated_nms
from eval_voxelnet import compute_ap, gt_obb_from_row, iou_3d_obb, iter_cached, pred_obb_from_box
from model import VoxelNet, load_state_dict_compat

BUCKETS = [("0-2m", 0, 2), ("2-2.5m", 2, 2.5), ("2.5-3m", 2.5, 3), ("3-3.5m", 3, 3.5),
           ("3.5-5m", 3.5, 5), ("5m+", 5, float("inf"))]
IOU_THRESHOLDS = (0.25, 0.35, 0.5)  # eval_voxelnet.py의 flat eval과 동일 - 프레임당 iou_mat은
# 한 번만 계산하고 두 threshold의 greedy matching만 독립적으로 돌려서 재사용(Cartesian vs
# Polar가 flat IoU>=0.5에서 크게 갈리는 게 확인돼, 그 격차가 어느 range bucket에 몰려있는지
# 보려고 추가함 - 2026-08-15).


def bucket_of(rng: float) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= rng < hi:
            return name
    return BUCKETS[-1][0]


@torch.no_grad()
def evaluate_by_range(model, device, samples, anchor_grid, score_thresh: float, nms_iou: float,
                       head: str, total: int = None, ap_decode_thresh: float = 0.05, polar: bool = False):
    """score_thresh(예: 0.3) 기준 recall/precision(gt_stats/pred_stats)뿐 아니라,
    ap_decode_thresh(훨씬 낮음)로 디코드한 전체 후보 풀에서 confidence 랭킹 전체를
    bucket별로 모아 AP도 같이 낸다 - 디코드+NMS는 프레임당 한 번만(가장 낮은
    threshold 기준) 하고, score_thresh 쪽은 그 결과를 다시 필터링만 해서 재사용
    (eval_voxelnet.py의 evaluate()/_score_frame()과 동일한 절약 패턴).

    IOU_THRESHOLDS(0.25, 0.5) 둘 다 한 번에 낸다 - iou_mat은 프레임당 한 번만 계산하고,
    두 threshold의 greedy matching(같은 prediction이 threshold에 따라 TP/FP가 갈릴 수
    있음)만 독립적으로 돌린다."""
    from tqdm import tqdm
    gt_stats = {t: {b[0]: {"gt_total": 0, "gt_matched": 0} for b in BUCKETS} for t in IOU_THRESHOLDS}
    pred_stats = {t: {b[0]: {"pred_total": 0, "pred_tp": 0} for b in BUCKETS} for t in IOU_THRESHOLDS}
    ap_detections = {t: {b[0]: [] for b in BUCKETS} for t in IOU_THRESHOLDS}
    ap_n_gt = {b[0]: 0 for b in BUCKETS}  # threshold와 무관(GT 자체는 안 바뀜)

    # eval_voxelnet.evaluate()와 동일한 이유로 stage2를 추론에도 반영(project_stage2_rng_confound_confirmed 메모리)
    use_stage2_eval = head == "center" and getattr(model.rpn, "stage2", None) is not None
    stage2_capture = {}
    if use_stage2_eval:
        model.rpn.backbone.register_forward_hook(
            lambda m, i, o: stage2_capture.__setitem__("feat", o))

    for sample_id, voxel_features, coords, num_points, gt_boxes in tqdm(samples, total=total, desc="eval"):
        voxel_features_t = torch.from_numpy(voxel_features).to(device)
        num_points_t = torch.from_numpy(num_points).to(device)
        batch_col = torch.zeros((len(coords), 1), dtype=torch.int64)
        coords_t = torch.cat([batch_col, torch.from_numpy(coords)], dim=1).to(device)

        if head == "anchor":
            cls_pred, reg_pred = model(voxel_features_t, num_points_t, coords_t)
            candidates = decode_boxes(cls_pred[0], reg_pred[0], anchor_grid, score_thresh=ap_decode_thresh)
        else:
            from eval_voxelnet import _decode_center_sample, _decode_center_sample_polar
            hm, off, z, dim, rot, _density = model(voxel_features_t, num_points_t, coords_t)
            decode_fn = _decode_center_sample_polar if polar else _decode_center_sample
            candidates = decode_fn(hm[0], off[0], z[0], dim[0], rot[0], score_thresh=ap_decode_thresh)
            if use_stage2_eval:
                candidates = s2.refine_candidates(model.rpn.stage2, stage2_capture["feat"], candidates)

        # ap_decode_thresh 기준 전체 후보 풀에 NMS 한 번 - AP용 랭킹은 이 풀 전체를 쓰고,
        # score_thresh(0.3) recall/precision은 이 풀을 다시 필터링해서 얻는다.
        low_pool = rotated_nms(candidates, iou_thresh=nms_iou)

        gt_list = [gt_obb_from_row(row) for row in gt_boxes]
        gt_ranges = [float(np.hypot(row[0], row[1])) for row in gt_boxes]
        for r in gt_ranges:
            ap_n_gt[bucket_of(r)] += 1
            for t in IOU_THRESHOLDS:
                gt_stats[t][bucket_of(r)]["gt_total"] += 1

        pred_list = [pred_obb_from_box(b) for b in low_pool]
        pred_ranges = [float(np.hypot(b["x"], b["y"])) for b in low_pool]
        pred_conf = [b["score"] for b in low_pool]

        n_p, n_g = len(pred_list), len(gt_list)
        iou_mat = np.zeros((n_p, n_g))
        rng = np.random.default_rng(0)  # 프레임마다 고정 시드 - 평가 재현성
        for pi, (pc, pd, pR) in enumerate(pred_list):
            for gi, (gc, gd, gR) in enumerate(gt_list):
                iou_mat[pi, gi] = iou_3d_obb(pc, pd, pR, gc, gd, gR, rng=rng)

        order = np.argsort(-np.array(pred_conf)) if n_p else np.array([], dtype=int)
        for t in IOU_THRESHOLDS:
            matched_gt = set()
            for pi in order:
                bucket = bucket_of(pred_ranges[pi])
                candidate_gt = [gi for gi in range(n_g) if gi not in matched_gt]
                is_tp = False
                if candidate_gt:
                    ious = iou_mat[pi, candidate_gt]
                    best_local = int(np.argmax(ious))
                    best_gt, best_iou = candidate_gt[best_local], ious[best_local]
                    if best_iou >= t:
                        matched_gt.add(best_gt)
                        is_tp = True
                # AP 누적 버킷은 TP/FP를 다르게 정한다 - TP는 "매칭된 GT가 속한 버킷"으로
                # 넣어야 그 버킷의 n_gt(GT range 기준)와 분모/분자가 맞는다. 예측 자신의
                # range로 TP까지 넣으면(이전 버그) 한 버킷이 다른 버킷 GT를 "빌려와" 매칭한
                # TP까지 카운트하게 되어 tp_cum이 n_gt를 넘어 recall/AP가 1을 초과했었다.
                # FP는 매칭된 GT가 없으니 예측 자신의 range로 넣는 것만 유효한 선택.
                ap_bucket = bucket_of(gt_ranges[best_gt]) if is_tp else bucket
                ap_detections[t][ap_bucket].append((pred_conf[pi], is_tp))
                if pred_conf[pi] >= score_thresh:
                    pred_stats[t][bucket]["pred_total"] += 1
                    if is_tp:
                        pred_stats[t][bucket]["pred_tp"] += 1
                        gt_stats[t][bucket_of(gt_ranges[best_gt])]["gt_matched"] += 1

    ap_by_bucket = {t: {} for t in IOU_THRESHOLDS}
    for t in IOU_THRESHOLDS:
        for name, _, _ in BUCKETS:
            ap, prec, rec = compute_ap(ap_detections[t][name], ap_n_gt[name])
            ap_by_bucket[t][name] = {"ap3d": ap, "precision_at_conf": prec, "recall_at_conf": rec,
                                      "n_preds": len(ap_detections[t][name]), "n_gt": ap_n_gt[name]}

    return gt_stats, pred_stats, ap_by_bucket


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--nms-iou", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--head", default=None, choices=["anchor", "center"])
    parser.add_argument("--out-name", required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    head = args.head or ckpt.get("head", "anchor")
    polar = ckpt.get("polar", False)
    use_grr = ckpt.get("use_grr", False)
    use_stage2 = ckpt.get("use_stage2", False)
    use_rot_branch = ckpt.get("use_rot_branch", False)
    model = VoxelNet(head=head, polar=polar, use_grr=use_grr, use_stage2=use_stage2,
                      use_rot_branch=use_rot_branch).to(args.device)
    load_state_dict_compat(model, ckpt["model"])
    model.eval()
    print(f"head={head} polar={polar} use_grr={use_grr} use_stage2={use_stage2} "
          f"use_rot_branch={use_rot_branch} (ckpt epoch={ckpt.get('epoch')})")

    anchor_grid = anchors_mod.build_anchor_grid()
    samples = iter_cached(args.cache_root, args.split)
    with open(Path(args.cache_root) / "manifest.json") as f:
        total = len(json.load(f)[args.split])

    gt_stats, pred_stats, ap_by_bucket = evaluate_by_range(
        model, args.device, samples, anchor_grid, args.score_thresh, args.nms_iou, head, total=total, polar=polar)

    out_dir = config.VOXELNET_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / args.out_name, "w") as f:
        json.dump({"checkpoint": str(args.ckpt), "split": args.split, "iou_thresholds": list(IOU_THRESHOLDS),
                   "gt_stats": {str(t): v for t, v in gt_stats.items()},
                   "pred_stats": {str(t): v for t, v in pred_stats.items()},
                   "ap_by_bucket": {str(t): v for t, v in ap_by_bucket.items()}}, f, indent=2)
    print(f"Wrote {out_dir / args.out_name}")

    for t in IOU_THRESHOLDS:
        print(f"=== {args.split} recall by GT range bucket (IoU>={t}, score_thresh={args.score_thresh}) ===")
        for name, _, _ in BUCKETS:
            g = gt_stats[t][name]
            recall = g["gt_matched"] / g["gt_total"] if g["gt_total"] else float("nan")
            print(f"  {name:8} gt={g['gt_total']:5d} matched={g['gt_matched']:5d} recall={recall:.3f}")

        print(f"=== {args.split} AP3D by bucket (IoU>={t}, 전체 confidence 랭킹 적분) ===")
        for name, _, _ in BUCKETS:
            a = ap_by_bucket[t][name]
            print(f"  {name:8} AP3D={a['ap3d']:.3f} n_gt={a['n_gt']:5d} n_preds={a['n_preds']:5d}")


if __name__ == "__main__":
    main()
