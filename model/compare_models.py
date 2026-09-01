"""compare_models.py - VoxelNet과 TriBand-BEV(YOLO-OBB)가 "같은 GT 박스"를 실제로
얼마나 겹쳐서/따로 맞추는지 프레임·박스 단위로 교차비교한다 (일회성 분석 스크립트,
Colab 학습/평가 파이프라인과 무관 - VoxelNet/reports에 결과만 남김).

집계된 AP/precision/recall만 보면 "두 모델이 다른 종류의 실수를 하는지"는 안 보인다 -
이 스크립트는 GT 박스 하나하나에 대해 [TriBand만 맞음 / VoxelNet만 맞음 / 둘 다 맞음 /
둘 다 놓침] 4가지로 분류해서 실제 겹침을 잰다.

Usage:
    python compare_models.py [--vn-score-thresh 0.7]
"""

import argparse
import json
import sys
from pathlib import Path

TRIBAND = Path("/home/eugene/Data/APRILab_baseline/Triband_BEV")
VOXELNET = Path("/home/eugene/Data/APRILab_baseline/VoxelNet")

sys.path.insert(0, str(TRIBAND / "baseline"))
# mpl3d_fix.py 자체의 docstring 그대로: matplotlib/torch/ultralytics 그 무엇보다도
# 먼저 import돼야 한다 - numpy/pandas/torch를 위에 먼저 적어놨다가 이 순서를 어기면
# YOLO.predict() 호출 시점에 torchvision NMS 등록이 mpl_toolkits 네임스페이스 오염과
# 충돌해 TypeError가 난다(실제로 겪음).
import mpl3d_fix  # noqa: E402,F401

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import box3d  # noqa: E402
import common  # noqa: E402
import filter_outliers  # noqa: E402

sys.path.insert(0, str(TRIBAND / "model"))
import evaluate as ev  # noqa: E402
import reconstruct_3d as r3d  # noqa: E402
from ultralytics import YOLO  # noqa: E402

sys.path.insert(0, str(VOXELNET / "model"))
import anchors as anchors_mod  # noqa: E402
import gather_frame as vn_gf  # noqa: E402
from anchors import rotated_rect_corners  # noqa: E402
from decode import decode_boxes, rotated_nms  # noqa: E402

IOU_THR = 0.25
CACHE_ROOT = VOXELNET / "cache" / "voxel"


def iou_3d(footprint_a, za, footprint_b, zb) -> float:
    from shapely.geometry import Polygon
    height_overlap = max(0.0, min(za[1], zb[1]) - max(za[0], zb[0]))
    if height_overlap <= 0:
        return 0.0
    poly_a, poly_b = Polygon(footprint_a), Polygon(footprint_b)
    if not poly_a.is_valid or not poly_b.is_valid:
        return 0.0
    inter_area = poly_a.intersection(poly_b).area
    if inter_area <= 0:
        return 0.0
    inter_vol = inter_area * height_overlap
    union_vol = poly_a.area * (za[1] - za[0]) + poly_b.area * (zb[1] - zb[0]) - inter_vol
    return float(inter_vol / union_vol) if union_vol > 0 else 0.0


def gt_box_world(obj: dict):
    corners = box3d.corners_3d(obj, use_xy_rotation=False)
    return corners[:4, :2], (float(corners[:, 2].min()), float(corners[:, 2].max()))


def triband_preds_for_frame(model, default_height, scene_id, split, frame_idx, points, conf=0.25,
                             img_dir=None):
    stem = f"{scene_id}_frame_{frame_idx:06d}"
    img_root = Path(img_dir) if img_dir else ev.IMG_DIR
    img_path = img_root / split / f"{stem}.png"
    if not img_path.exists():
        return []
    r = next(model.predict(source=str(img_path), imgsz=384, conf=conf, iou=0.7,
                            device="cpu", verbose=False, stream=True))
    out = []
    if r.obb is not None and len(r.obb):
        pred_corners_px = r.obb.xyxyxyxy.cpu().numpy()
        pred_conf = r.obb.conf.cpu().numpy()
        for px, cf in zip(pred_corners_px, pred_conf):
            world_xy = np.stack(ev.pixel_to_world(px[:, 1], px[:, 0]), axis=1)
            range_m = float(np.hypot(*world_xy.mean(axis=0)))
            rec = r3d.reconstruct_height(points, world_xy, range_m, default_height, apply_iqr=True)
            if rec is None:
                continue
            out.append((world_xy, (rec["z_bottom"], rec["z_top"]), float(cf)))
    return out


def voxelnet_preds_for_cached_frame(model, anchor_grid, npz, score_thresh, nms_iou, device="cpu"):
    with torch.no_grad():
        vf = torch.from_numpy(npz["voxel_features"]).to(device)
        npnt = torch.from_numpy(npz["num_points"]).to(device)
        batch_col = torch.zeros((len(npz["coords"]), 1), dtype=torch.int64)
        coords_t = torch.cat([batch_col, torch.from_numpy(npz["coords"])], dim=1).to(device)
        cls_pred, reg_pred = model(vf, npnt, coords_t)
    boxes = decode_boxes(cls_pred[0], reg_pred[0], anchor_grid, score_thresh=score_thresh)
    boxes = rotated_nms(boxes, iou_thresh=nms_iou)
    out = []
    for b in boxes:
        footprint = rotated_rect_corners(b["x"], b["y"], b["l"], b["w"], b["theta"])
        z_bottom, z_top = b["z"] - b["h"] / 2, b["z"] + b["h"] / 2
        out.append((footprint, (z_bottom, z_top), float(b["score"])))
    return out


def match_gt(gt_list: list, pred_list: list, iou_thr: float = IOU_THR):
    """gt_list: [(footprint,z)]. pred_list: [(footprint,z,conf)]. greedy, confidence 내림차순.
    반환: (matched: [bool]*len(gt_list), n_fp: int)."""
    order = sorted(range(len(pred_list)), key=lambda i: -pred_list[i][2])
    matched_gt = set()
    n_tp = 0
    for pi in order:
        pf, pz, _ = pred_list[pi]
        best_iou, best_gi = 0.0, -1
        for gi, (gf, gz) in enumerate(gt_list):
            if gi in matched_gt:
                continue
            iou = iou_3d(pf, pz, gf, gz)
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_gi >= 0 and best_iou >= iou_thr:
            matched_gt.add(best_gi)
            n_tp += 1
    n_fp = len(pred_list) - n_tp
    return [i in matched_gt for i in range(len(gt_list))], n_fp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triband-ckpt", default=str(TRIBAND / "checkpoints" / "baseline_run2.pt"))
    parser.add_argument("--voxelnet-ckpt", default=str(VOXELNET / "checkpoints" / "voxelnet_run1.pt"))
    parser.add_argument("--vn-score-thresh", type=float, default=0.7)
    parser.add_argument("--vn-nms-iou", type=float, default=0.1)
    parser.add_argument("--triband-conf", type=float, default=0.25)
    parser.add_argument("--triband-img-dir", default=None,
                         help="기본은 production yolo_dataset/images - literal 등 다른 "
                              "scheme 체크포인트 비교 시 지정")
    parser.add_argument("--split", default="val")
    parser.add_argument("--out-suffix", default="",
                         help="출력 파일명에 붙일 접미사 - 기본(baseline_run2 vs voxelnet_run1) "
                              "결과를 안 덮어쓰려면 지정 (예: _literal)")
    args = parser.parse_args()

    with open(common.REPORTS_DIR / "splits.json") as f:
        scenes = json.load(f)["split"][args.split]
    flagged = filter_outliers.load_flagged_keys()

    default_height = r3d.compute_default_height()
    triband_model = YOLO(args.triband_ckpt, task="obb")
    vn_model, _epoch = vn_gf.load_model(args.voxelnet_ckpt, device="cpu")
    anchor_grid = anchors_mod.build_anchor_grid()

    rows = []
    n_done = 0
    for scene_id in scenes:
        for frame_idx, objects in filter_outliers.iter_filtered_frames(scene_id, flagged):
            n_done += 1
            if n_done % 200 == 0:
                print(f"[compare_models] {n_done}개 프레임 처리됨")

            points = np.fromfile(common.sonar_bin_path(scene_id, frame_idx), dtype=np.float32).reshape(-1, 4)
            points = points[~np.isnan(points).any(axis=1)]
            gt_list = [gt_box_world(o) for o in objects]

            tb_preds = triband_preds_for_frame(triband_model, default_height, scene_id,
                                                args.split, frame_idx, points, conf=args.triband_conf,
                                                img_dir=args.triband_img_dir)
            tb_hits, tb_fp = match_gt(gt_list, tb_preds)

            cache_path = CACHE_ROOT / scene_id / f"frame_{frame_idx:06d}.npz"
            with np.load(cache_path) as npz:
                vn_preds = voxelnet_preds_for_cached_frame(
                    vn_model, anchor_grid, npz, args.vn_score_thresh, args.vn_nms_iou)
            vn_hits, vn_fp = match_gt(gt_list, vn_preds)

            for gi, o in enumerate(objects):
                rows.append({
                    "scene_id": scene_id, "frame_idx": frame_idx, "link_id": o.get("link_id"),
                    "triband_hit": tb_hits[gi], "voxelnet_hit": vn_hits[gi],
                    "n_triband_preds": len(tb_preds), "n_voxelnet_preds": len(vn_preds),
                    "n_triband_fp": tb_fp, "n_voxelnet_fp": vn_fp,
                })

    df = pd.DataFrame(rows)
    both = int((df.triband_hit & df.voxelnet_hit).sum())
    only_tb = int((df.triband_hit & ~df.voxelnet_hit).sum())
    only_vn = int((~df.triband_hit & df.voxelnet_hit).sum())
    neither = int((~df.triband_hit & ~df.voxelnet_hit).sum())
    total = len(df)

    summary = {
        "split": args.split, "iou_thr": IOU_THR,
        "voxelnet_score_thresh": args.vn_score_thresh, "voxelnet_nms_iou": args.vn_nms_iou,
        "triband_conf": args.triband_conf,
        "n_gt_total": total,
        "both_hit": both, "only_triband_hit": only_tb, "only_voxelnet_hit": only_vn, "neither_hit": neither,
        "triband_recall": (both + only_tb) / total if total else None,
        "voxelnet_recall": (both + only_vn) / total if total else None,
        "total_triband_preds": int(df.n_triband_preds.sum() / max(len(df.groupby(['scene_id', 'frame_idx'])), 1) * len(df.groupby(['scene_id', 'frame_idx']))),
        "total_triband_fp": int(df.groupby(["scene_id", "frame_idx"]).n_triband_fp.first().sum()),
        "total_voxelnet_fp": int(df.groupby(["scene_id", "frame_idx"]).n_voxelnet_fp.first().sum()),
    }
    print(json.dumps(summary, indent=2))

    out_dir = VOXELNET / "reports"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f"compare_models_per_gt{args.out_suffix}.csv"
    json_path = out_dir / f"compare_models_summary{args.out_suffix}.json"
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {csv_path}, {json_path}")


if __name__ == "__main__":
    main()
