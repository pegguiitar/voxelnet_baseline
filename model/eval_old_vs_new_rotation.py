"""eval_old_vs_new_rotation.py - x,y 회전 확장의 "순수 효과"를 깨끗하게 재기 위한
before-측정 스크립트.

문제: decode.py/model.py를 이미 12/6채널(3D 회전 확장)로 바꿔버려서, 옛 체크포인트
(rot_head 2채널, z-yaw sin/cos)를 지금 코드로 바로 못 읽는다. 그렇다고 옛 eval(z-yaw
근사 IoU)로 새 모델을 채점하면 "회전을 추가한 효과"와 "eval 방식이 달라진 효과"가
뒤섞여 의미가 없어진다.

해법: RPNBackbone(안 바뀜)은 그대로 재사용하고, rot_head만 옛 구조(2채널)로 별도
정의해서 옛 체크포인트를 정확히 로드 -> 옛 방식(sin/cos, z-yaw만)으로 디코드 ->
채점은 **새 GT(진짜 x,y 회전 반영)와 새 iou_3d_obb**로 한다. 이러면:
  - "before" = 옛 모델(z-yaw만) 예측을 진짜 3D 형상 기준으로 정직하게 채점한 점수
  - "after"(새 모델 학습 완료 후) = 새 모델 예측을 똑같은 채점 기준으로 잰 점수
  - 두 점수의 차이 = eval 방식 변화와 무관하게, 순수하게 "x,y 회전을 모델에 가르친 효과"

Usage:
    python eval_old_vs_new_rotation.py --ckpt ../checkpoints/voxelnet_center_centerpoint_s0_best.pt \
        --cache-root ../cache/voxel --split test
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import anchors as anchors_mod
import config
from eval_voxelnet import compute_ap, gt_obb_from_row, iou_3d_obb
from eval_voxelnet_by_range import BUCKETS, bucket_of
from decode import rotated_nms
from model import RPNBackbone, StackedVFE, ConvMiddleLayers, load_state_dict_compat
import rotation3d

IOU_THRESHOLDS = (0.25, 0.5)


class OldRPNCenterHead(nn.Module):
    """3D 회전 확장 이전의 RPNCenterHead 구조 그대로(rot_head=2채널, sin/cos) - 옛
    체크포인트를 정확히 로드하기 위한 것. RPNBackbone 자체는 이번 확장으로 안 바뀌었으므로
    그대로 재사용한다."""

    def __init__(self):
        super().__init__()
        self.backbone = RPNBackbone()
        c = self.backbone.out_channels
        self.heatmap_head = nn.Conv2d(c, 1, 1)
        self.offset_head = nn.Conv2d(c, 2, 1)
        self.z_head = nn.Conv2d(c, 1, 1)
        self.dim_head = nn.Conv2d(c, 3, 1)
        self.rot_head = nn.Conv2d(c, 2, 1)  # 옛 구조: sin/cos 2채널

    def forward(self, x):
        feat = self.backbone(x)
        return (self.heatmap_head(feat), self.offset_head(feat), self.z_head(feat),
                self.dim_head(feat), self.rot_head(feat))


class OldVoxelNetCenter(nn.Module):
    """VoxelNet(head="center")의 옛 구조 버전 - vfe/middle은 안 바뀌었으므로 그대로,
    rpn만 OldRPNCenterHead."""

    def __init__(self):
        super().__init__()
        self.vfe = StackedVFE()
        self.middle = ConvMiddleLayers()
        self.rpn = OldRPNCenterHead()
        self.grid_size = config.GRID_SIZE

    def forward(self, voxel_features, num_points, coords):
        voxelwise = self.vfe(voxel_features, num_points)
        B = int(coords[:, 0].max().item()) + 1 if len(coords) else 1
        Wp, Hp, Dp = self.grid_size
        dense = voxelwise.new_zeros(B, 128, Dp, Hp, Wp)
        if len(coords):
            b, z, y, x = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
            dense[b, :, z, y, x] = voxelwise
        mid = self.middle(dense)
        B_, C, D_, H_, W_ = mid.shape
        feat2d = mid.reshape(B_, C * D_, H_, W_)
        return self.rpn(feat2d)


def decode_old_center_boxes(heatmap_pred, offset_pred, z_pred, dim_pred, rot_pred,
                             score_thresh: float = 0.3, max_peaks: int = 100):
    """heatmap_targets.decode_center_boxes의 확장 이전 버전(rot_pred 2채널, sin/cos) -
    옛 체크포인트 예측을 옛 방식 그대로 디코드하기 위해 그대로 복제."""
    hm = heatmap_pred[0]
    H, W = hm.shape
    padded = np.full((H + 2, W + 2), -1.0, dtype=hm.dtype)
    padded[1:-1, 1:-1] = hm
    is_peak = np.ones((H, W), dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            is_peak &= hm >= padded[1 + dr:1 + dr + H, 1 + dc:1 + dc + W]

    rows, cols = np.where(is_peak & (hm >= score_thresh))
    if len(rows) > max_peaks:
        scores_all = hm[rows, cols]
        top = np.argsort(-scores_all)[:max_peaks]
        rows, cols = rows[top], cols[top]

    sx, sy = config.ANCHOR_STRIDE
    x0, y0 = config.POINT_CLOUD_RANGE[0], config.POINT_CLOUD_RANGE[1]

    boxes = []
    for row, col in zip(rows, cols):
        dx, dy = offset_pred[row, col]
        gx_cell, gy_cell = col + dx, row + dy
        x = x0 + (gx_cell + 0.5) * sx
        y = y0 + (gy_cell + 0.5) * sy
        z = float(z_pred[row, col, 0])
        l, w, h = np.exp(dim_pred[row, col])
        sin_t, cos_t = rot_pred[row, col]
        theta = float(np.arctan2(sin_t, cos_t))
        R = rotation3d.euler_to_matrix(0.0, 0.0, math.degrees(theta))  # 옛 모델은 z-yaw만 예측
        boxes.append({"score": float(hm[row, col]), "x": float(x), "y": float(y), "z": z,
                       "l": float(l), "w": float(w), "h": float(h), "theta": theta, "R": R})
    return boxes


def pred_obb_from_box(box: dict):
    center = np.array([box["x"], box["y"], box["z"]])
    dims = np.array([box["l"], box["w"], box["h"]])
    return center, dims, box["R"]


def iter_cached(cache_root: str, split: str):
    cache_root = Path(cache_root)
    with open(cache_root / "manifest.json") as f:
        entries = json.load(f)[split]
    for rel_path in entries:
        with np.load(cache_root / rel_path) as npz:
            yield (rel_path, npz["voxel_features"], npz["coords"], npz["num_points"], npz["gt_boxes"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--nms-iou", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    model = OldVoxelNetCenter().to(args.device)
    load_state_dict_compat(model, ckpt["model"])
    model.eval()
    print(f"옛 구조(rot_head=2채널)로 로드 완료: {args.ckpt} (epoch={ckpt.get('epoch')})")

    samples = iter_cached(args.cache_root, args.split)
    detections = {t: [] for t in IOU_THRESHOLDS}
    n_gt_total = 0
    # range bucket 분해 - IOU_THRESHOLDS(0.25, 0.5) 둘 다 집계한다(eval_voxelnet_by_range.py와
    # 동일 관례: TP는 매칭된 GT의 버킷, FP는 예측 자신의 버킷). 2026-08-21 이전엔 0.25만
    # 집계했었는데, 이 프로젝트의 주 지표가 IoU0.5로 바뀌면서(project 메모리) before/after
    # range-bucket 비교도 0.5 기준이 필요해져 확장.
    ap_detections_by_bucket = {t: {b[0]: [] for b in BUCKETS} for t in IOU_THRESHOLDS}
    ap_n_gt_by_bucket = {b[0]: 0 for b in BUCKETS}

    from tqdm import tqdm
    with torch.no_grad():
        for sample_id, voxel_features, coords, num_points, gt_boxes in tqdm(samples, desc="eval(old model, new gt)"):
            if gt_boxes.shape[1] != 13:
                raise RuntimeError(
                    f"gt_boxes가 {gt_boxes.shape[1]}열 - 캐시가 아직 새 13열(x,y,z,l,w,h,theta_z,6D) "
                    "포맷으로 재생성 안 됐을 수 있음. cache_dataset.py --force부터 실행할 것.")
            vf_t = torch.from_numpy(voxel_features).to(args.device)
            np_t = torch.from_numpy(num_points).to(args.device)
            batch_col = torch.zeros((len(coords), 1), dtype=torch.int64)
            coords_t = torch.cat([batch_col, torch.from_numpy(coords)], dim=1).to(args.device)

            hm, off, z, dim, rot = model(vf_t, np_t, coords_t)
            candidates = decode_old_center_boxes(
                hm[0].cpu().numpy(), off[0].permute(1, 2, 0).cpu().numpy(), z[0].permute(1, 2, 0).cpu().numpy(),
                dim[0].permute(1, 2, 0).cpu().numpy(), rot[0].permute(1, 2, 0).cpu().numpy(),
                score_thresh=args.score_thresh)
            filtered = rotated_nms(candidates, iou_thresh=args.nms_iou)

            gt_list = [gt_obb_from_row(row) for row in gt_boxes]
            gt_ranges = [float(np.hypot(row[0], row[1])) for row in gt_boxes]
            for r in gt_ranges:
                ap_n_gt_by_bucket[bucket_of(r)] += 1
            pred_list = [pred_obb_from_box(b) for b in filtered]
            pred_ranges = [float(np.hypot(b["x"], b["y"])) for b in filtered]
            pred_conf = [b["score"] for b in filtered]
            n_gt_total += len(gt_list)

            n_p, n_g = len(pred_list), len(gt_list)
            iou_mat = np.zeros((n_p, n_g))
            rng = np.random.default_rng(0)
            for pi, (pc, pd, pR) in enumerate(pred_list):
                for gi, (gc, gd, gR) in enumerate(gt_list):
                    iou_mat[pi, gi] = iou_3d_obb(pc, pd, pR, gc, gd, gR, rng=rng)

            order = np.argsort(-np.array(pred_conf)) if n_p else np.array([], dtype=int)
            for thr in IOU_THRESHOLDS:
                matched_gt = set()
                for pi in order:
                    candidate_gt = [gi for gi in range(n_g) if gi not in matched_gt]
                    is_tp = False
                    best_gt = None
                    if candidate_gt:
                        ious = iou_mat[pi, candidate_gt]
                        best_local = int(np.argmax(ious))
                        best_gt, best_iou = candidate_gt[best_local], ious[best_local]
                        if best_iou >= thr:
                            matched_gt.add(best_gt)
                            is_tp = True
                    detections[thr].append((float(pred_conf[pi]), is_tp))
                    ap_bucket = bucket_of(gt_ranges[best_gt]) if is_tp else bucket_of(pred_ranges[pi])
                    ap_detections_by_bucket[thr][ap_bucket].append((float(pred_conf[pi]), is_tp))

    print(f"\n=== {args.split}: 옛 모델(z-yaw만) 예측을 새 GT(진짜 3D 회전)로 채점 ===")
    print("(이게 'before' 기준점 - 나중에 새로 학습한 회전-인식 모델을 같은 방식으로 채점해서 비교)")
    overall = {}
    for thr in IOU_THRESHOLDS:
        dets = detections[thr]
        ap, prec, rec = compute_ap(dets, n_gt_total)
        overall[str(thr)] = {"ap3d": ap, "precision": prec, "recall": rec,
                              "n_preds": len(dets), "n_tp": sum(d[1] for d in dets), "n_gt": n_gt_total}
        print(f"IoU>={thr}: AP3D={ap:.4f}  P={prec:.4f}  R={rec:.4f} "
              f"(preds={len(dets)}, tp={sum(d[1] for d in dets)}, gt={n_gt_total})")

    ap_by_bucket = {}
    for thr in IOU_THRESHOLDS:
        print(f"\n=== range bucket별 (IoU>={thr}) ===")
        ap_by_bucket[str(thr)] = {}
        for name, _, _ in BUCKETS:
            dets = ap_detections_by_bucket[thr][name]
            ap, prec, rec = compute_ap(dets, ap_n_gt_by_bucket[name])
            ap_by_bucket[str(thr)][name] = {"ap3d": ap, "precision_at_conf": prec, "recall_at_conf": rec,
                                             "n_preds": len(dets), "n_gt": ap_n_gt_by_bucket[name]}
            print(f"  {name:>8s}: AP3D={ap:.4f}  n_preds={len(dets):4d}  n_gt={ap_n_gt_by_bucket[name]:4d}")

    out_dir = config.VOXELNET_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    out_name = f"{Path(args.ckpt).stem}_old_arch_{args.split}_range_eval.json"
    with open(out_dir / out_name, "w") as f:
        json.dump({"checkpoint": str(args.ckpt), "split": args.split, "overall": overall,
                    "iou_thresholds": list(IOU_THRESHOLDS), "ap_by_bucket": ap_by_bucket}, f, indent=2)
    print(f"\nWrote {out_dir / out_name}")


if __name__ == "__main__":
    main()
