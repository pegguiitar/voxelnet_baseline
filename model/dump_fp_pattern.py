"""dump_fp_pattern.py - fp_pattern_analysis.json과 동일한 형식(scene_frame, score, x,
y, z, range, l, w, h)으로 TP/FP 후보를 낮은 threshold로 전부 덤프한다. 1b의 원래
분석(voxelnet_run1 전용, 스크립트 없이 만들어졌던 것)을 다른 체크포인트/split에도
재사용할 수 있게 일반화한 버전 - val/test를 따로 낼 수 있어 "val에서 뽑고 test에서
검증"하는 held-out 절차가 가능하다.

Usage:
    python dump_fp_pattern.py --ckpt ../checkpoints/voxelnet_run3.pt --cache-root ../cache/voxel \
        --split val --out-name fp_pattern_run3_val.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import anchors as anchors_mod
import config
from decode import decode_boxes, rotated_nms
from eval_voxelnet import _decode_center_sample, gt_obb_from_row, iou_3d_obb, iter_cached, pred_obb_from_box
from model import VoxelNet, load_state_dict_compat

IOU_THRESH = 0.25


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-name", required=True)
    parser.add_argument("--decode-thresh", type=float, default=0.05,
                         help="체크포인트마다 confidence 스케일이 다르므로(run1은 최대~0.999,"
                              " run3는 관측상 최대~0.34) 고정값(예: 0.5)을 재사용하지 말 것 -"
                              " 낮은 값으로 전체 후보 풀을 넓게 잡고 threshold 탐색은 그 안에서 함")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    head = ckpt.get("head", "anchor")
    model = VoxelNet(head=head).to(args.device)
    load_state_dict_compat(model, ckpt["model"])
    model.eval()

    anchor_grid = anchors_mod.build_anchor_grid()
    samples = iter_cached(args.cache_root, args.split)
    with open(Path(args.cache_root) / "manifest.json") as f:
        total = len(json.load(f)[args.split])

    from tqdm import tqdm
    fp_rows, tp_rows = [], []
    n_gt_total = 0

    with torch.no_grad():
        for sample_id, voxel_features, coords, num_points, gt_boxes in tqdm(samples, total=total, desc="dump"):
            voxel_features_t = torch.from_numpy(voxel_features).to(args.device)
            num_points_t = torch.from_numpy(num_points).to(args.device)
            batch_col = torch.zeros((len(coords), 1), dtype=torch.int64)
            coords_t = torch.cat([batch_col, torch.from_numpy(coords)], dim=1).to(args.device)

            if head == "anchor":
                cls_pred, reg_pred = model(voxel_features_t, num_points_t, coords_t)
                candidates = decode_boxes(cls_pred[0], reg_pred[0], anchor_grid, score_thresh=args.decode_thresh)
            else:
                hm, off, z, dim, rot, _density = model(voxel_features_t, num_points_t, coords_t)
                candidates = _decode_center_sample(hm[0], off[0], z[0], dim[0], rot[0],
                                                    score_thresh=args.decode_thresh)
            filtered = rotated_nms(candidates, iou_thresh=0.1)
            n_gt_total += len(gt_boxes)

            gt_list = [gt_obb_from_row(row) for row in gt_boxes]
            pred_list = [pred_obb_from_box(b) for b in filtered]
            n_p, n_g = len(filtered), len(gt_list)
            iou_mat = np.zeros((n_p, n_g))
            iou_rng = np.random.default_rng(0)  # 프레임마다 고정 시드 - 재현성
            for pi, (pc, pd, pR) in enumerate(pred_list):
                for gi, (gc, gd, gR) in enumerate(gt_list):
                    iou_mat[pi, gi] = iou_3d_obb(pc, pd, pR, gc, gd, gR, rng=iou_rng)

            order = np.argsort(-np.array([b["score"] for b in filtered])) if n_p else np.array([], dtype=int)
            matched_gt = set()
            for pi in order:
                b = filtered[pi]
                rng = float(np.hypot(b["x"], b["y"]))
                row = {"scene_frame": sample_id, "score": float(b["score"]), "x": float(b["x"]), "y": float(b["y"]),
                       "z": float(b["z"]), "range": rng, "l": float(b["l"]), "w": float(b["w"]), "h": float(b["h"])}
                candidate_gt = [gi for gi in range(n_g) if gi not in matched_gt]
                is_tp = False
                if candidate_gt:
                    ious = iou_mat[pi, candidate_gt]
                    best_local = int(np.argmax(ious))
                    best_gt, best_iou = candidate_gt[best_local], ious[best_local]
                    if best_iou >= IOU_THRESH:
                        matched_gt.add(best_gt)
                        is_tp = True
                (tp_rows if is_tp else fp_rows).append(row)

    out = {"n_frames": total, "n_gt": n_gt_total, "n_tp": len(tp_rows), "n_fp": len(fp_rows),
           "fp_rows": fp_rows, "tp_rows": tp_rows}
    out_path = config.VOXELNET_ROOT / "reports" / args.out_name
    json.dump(out, open(out_path, "w"))
    print(f"n_tp={len(tp_rows)} n_fp={len(fp_rows)} n_gt={n_gt_total}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
