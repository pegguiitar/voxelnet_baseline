"""eval_extra_metrics.py - eval_voxelnet.py를 재사용해 추가 지표를 낸다(공유 파일 미수정).
산출: BEV AP(@0.25/0.35/0.5), TP 평균 3D IoU / BEV IoU, ATE(중심오차 m),
ASE(1-aligned IoU), AOE(yaw 오차 deg, 직육면체 π-대칭 fold).

TP 정의: 3D OBB IoU >= --tp-iou(기본 0.3)로 greedy(conf순) 매칭된 pred-gt 쌍.
center head 전용. eval_voxelnet의 decode/model 로딩/iou 함수 그대로 사용.
"""
import argparse, json, math
from pathlib import Path
import numpy as np
import torch
from shapely.geometry import Polygon

import eval_voxelnet as ev
import anchors as anchors_mod
from decode import rotated_nms
from anchors import rotated_rect_corners
from model import VoxelNet
from eval_voxelnet import (iou_3d_obb, gt_obb_from_row, pred_obb_from_box,
                           iter_cached, _decode_center_sample, _decode_center_sample_polar,
                           load_state_dict_compat, compute_ap)
from eval_voxelnet_by_range import BUCKETS, bucket_of


def yaw_of(R):
    return math.atan2(R[1, 0], R[0, 0])


def bev_iou(ca, da, Ra, cb, db, Rb):
    pa = Polygon(rotated_rect_corners(ca[0], ca[1], da[0], da[1], yaw_of(Ra)))
    pb = Polygon(rotated_rect_corners(cb[0], cb[1], db[0], db[1], yaw_of(Rb)))
    if not pa.is_valid or not pb.is_valid:
        return 0.0
    inter = pa.intersection(pb).area
    union = pa.area + pb.area - inter
    return inter / union if union > 0 else 0.0


def aligned_iou_3d(da, db):
    """같은 중심·자세로 정렬했을 때의 3D IoU(치수 불일치만 반영) - ASE용."""
    inter = np.prod(np.minimum(da, db))
    va, vb = np.prod(da), np.prod(db)
    union = va + vb - inter
    return inter / union if union > 0 else 0.0


def yaw_err_folded(ya, yb):
    """직육면체는 π 회전 대칭 → yaw 오차를 mod π 후 [0, π/2]로 fold."""
    d = abs((ya - yb + math.pi) % (2 * math.pi) - math.pi)  # [0,π]
    d = d % math.pi                                          # [0,π)
    return min(d, math.pi - d)                               # [0,π/2]


def build_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    head = ckpt.get("head", "center")
    polar = ckpt.get("polar", False)
    has_fg = any(k.startswith("rpn.fg_head.") for k in ckpt["model"].keys())
    m = VoxelNet(head=head, polar=polar, use_grr=ckpt.get("use_grr", False),
                 use_stage2=ckpt.get("use_stage2", False),
                 use_rot_branch=ckpt.get("use_rot_branch", False),
                 use_pdconv=ckpt.get("use_pdconv", False), grr_n=ckpt.get("grr_n", None),
                 use_ga=ckpt.get("use_ga", False), use_fg_head=has_fg,
                 fg_range_cond=ckpt.get("fg_range_cond", False),
                 use_rs_conv=ckpt.get("use_rs_conv", False),
                 use_feat_undistort=ckpt.get("use_feat_undistort", False)).to(device)
    load_state_dict_compat(m, ckpt["model"])
    m.eval()
    return m, head, polar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--score-thresh", type=float, default=0.3)
    ap.add_argument("--nms-iou", type=float, default=0.1)
    ap.add_argument("--tp-iou", type=float, default=0.3, help="TP 판정 3D IoU")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model, head, polar = build_model(args.ckpt, args.device)
    anchor_grid = anchors_mod.build_anchor_grid()
    decode_fn = _decode_center_sample_polar if polar else _decode_center_sample

    BEV_THR = [0.25, 0.35, 0.5]
    bev_dets = {t: [] for t in BEV_THR}
    tp_iou3d, tp_ioubev, tp_ate, tp_ase, tp_aoe = [], [], [], [], []
    n_gt_total = 0
    n_done = 0

    # --- range-bucket breakdown (2026-08-28, fg_gate 정교화 Phase 0/1) - fg_gate win이
    # box-fit/localization 계열이라 ATE/ASE/AOE·BEV-IoU를 range별로 쪼개야 어디서 이득이
    # 나는지 보인다. eval_voxelnet_by_range의 BUCKETS/관례를 그대로 재사용(TP→GT range
    # 버킷, FP→pred range 버킷). 같은 단일 패스에서 추가 산출, GPU 비용 0. ---
    _bnames = [b[0] for b in BUCKETS]
    bucket_n_gt = {b: 0 for b in _bnames}
    bev_dets_bucket = {t: {b: [] for b in _bnames} for t in BEV_THR}
    tp_bucket = {b: {"iou3d": [], "ioubev": [], "ate": [], "ase": [], "aoe": []} for b in _bnames}

    for sample_id, vf, coords, num_points, gt_boxes in iter_cached(args.cache_root, args.split):
        n_done += 1
        if n_done % 300 == 0:
            print(f"[extra] {n_done} frames", flush=True)
        n_gt_total += len(gt_boxes)

        vf_t = torch.from_numpy(vf).to(args.device)
        np_t = torch.from_numpy(num_points).to(args.device)
        bc = torch.zeros((len(coords), 1), dtype=torch.int64)
        coords_t = torch.cat([bc, torch.from_numpy(coords)], dim=1).to(args.device)
        with torch.no_grad():
            hm, off, z, dim, rot, _ = model(vf_t, np_t, coords_t)
        cand = decode_fn(hm[0], off[0], z[0], dim[0], rot[0], score_thresh=args.score_thresh)
        cand = [b for b in cand if b["score"] >= args.score_thresh]
        cand = rotated_nms(cand, iou_thresh=args.nms_iou)

        gts = [gt_obb_from_row(r) for r in gt_boxes]
        preds = [pred_obb_from_box(b) for b in cand]
        conf = [b["score"] for b in cand]
        order = np.argsort(-np.array(conf)) if preds else []

        gt_ranges = [float(np.hypot(r[0], r[1])) for r in gt_boxes]
        pred_ranges = [float(np.hypot(b["x"], b["y"])) for b in cand]
        for r in gt_ranges:
            bucket_n_gt[bucket_of(r)] += 1

        n_p, n_g = len(preds), len(gts)
        iou3d = np.zeros((n_p, n_g)); ioubev = np.zeros((n_p, n_g))
        rng = np.random.default_rng(0)
        for pi, (pc, pd, pR) in enumerate(preds):
            for gi, (gc, gd, gR) in enumerate(gts):
                iou3d[pi, gi] = iou_3d_obb(pc, pd, pR, gc, gd, gR, rng=rng)
                ioubev[pi, gi] = bev_iou(pc, pd, pR, gc, gd, gR)

        # BEV AP matching (per threshold, greedy by conf)
        for t in BEV_THR:
            matched = set()
            for pi in order:
                cands = [gi for gi in range(n_g) if gi not in matched]
                is_tp = False; bg = None
                if cands:
                    j = int(np.argmax(ioubev[pi, cands])); bg = cands[j]
                    if ioubev[pi, bg] >= t:
                        matched.add(bg); is_tp = True
                bev_dets[t].append((conf[pi], is_tp))
                bkt = bucket_of(gt_ranges[bg]) if is_tp else bucket_of(pred_ranges[pi])
                bev_dets_bucket[t][bkt].append((conf[pi], is_tp))

        # TP error metrics (match by 3D IoU >= tp_iou, greedy by conf)
        matched = set()
        for pi in order:
            cands = [gi for gi in range(n_g) if gi not in matched]
            if not cands:
                continue
            j = int(np.argmax(iou3d[pi, cands])); bg = cands[j]
            if iou3d[pi, bg] >= args.tp_iou:
                matched.add(bg)
                pc, pd, pR = preds[pi]; gc, gd, gR = gts[bg]
                _i3, _ib = iou3d[pi, bg], ioubev[pi, bg]
                _ate = float(np.linalg.norm(np.array(pc) - np.array(gc)))
                _ase = 1.0 - aligned_iou_3d(np.array(pd), np.array(gd))
                _aoe = math.degrees(yaw_err_folded(yaw_of(pR), yaw_of(gR)))
                tp_iou3d.append(_i3); tp_ioubev.append(_ib)
                tp_ate.append(_ate); tp_ase.append(_ase); tp_aoe.append(_aoe)
                _tb = tp_bucket[bucket_of(gt_ranges[bg])]
                _tb["iou3d"].append(_i3); _tb["ioubev"].append(_ib)
                _tb["ate"].append(_ate); _tb["ase"].append(_ase); _tb["aoe"].append(_aoe)

    def _mean(x):
        return float(np.mean(x)) if len(x) else 0.0

    out = {"checkpoint": Path(args.ckpt).name, "split": args.split, "n_gt": n_gt_total,
           "n_tp": len(tp_iou3d),
           "bev_ap": {str(t): compute_ap(bev_dets[t], n_gt_total)[0] for t in BEV_THR},
           "tp_mean_iou3d": _mean(tp_iou3d),
           "tp_mean_ioubev": _mean(tp_ioubev),
           "ATE_m": _mean(tp_ate),
           "ASE": _mean(tp_ase),
           "AOE_deg": _mean(tp_aoe),
           "by_range": {b: {
               "n_gt": bucket_n_gt[b],
               "n_tp": len(tp_bucket[b]["iou3d"]),
               "bev_ap": {str(t): compute_ap(bev_dets_bucket[t][b], bucket_n_gt[b])[0]
                          for t in BEV_THR},
               "tp_mean_iou3d": _mean(tp_bucket[b]["iou3d"]),
               "tp_mean_ioubev": _mean(tp_bucket[b]["ioubev"]),
               "ATE_m": _mean(tp_bucket[b]["ate"]),
               "ASE": _mean(tp_bucket[b]["ase"]),
               "AOE_deg": _mean(tp_bucket[b]["aoe"]),
           } for b in _bnames}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
