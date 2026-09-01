"""eval_voxelnet.py - AP3D 평가. `Triband_BEV/model/eval_3d_iou.py`와 정의를 맞춘
`iou_3d`/`compute_ap`(VOC2010-style all-point AP)를 그대로 재구현해서 TriBand-BEV/
VoxelKP와 직접 비교 가능한 숫자를 낸다(무거운 ultralytics/reconstruct_3d 의존성을
끌어오지 않으려고 import 대신 재구현 - 두 파일 다 짧은 순수함수라 동기화 비용 작음).

`cache_dataset.py`가 만든 캐시(voxel_features/coords/num_points까지 이미 계산됨,
gt_boxes도 world-space로 같이 저장됨)를 그대로 읽는다 - voxelize를 다시 하지 않는다.

score_thresh는 여러 값을 한꺼번에 줄 수 있다(`--score-thresh 0.3 0.5 0.7`) - 후보 박스는
가장 낮은 threshold 기준으로 딱 한 번만 decode하고(모델 forward도 프레임당 한 번뿐),
그보다 높은 threshold들은 이미 decode된 후보 리스트를 다시 필터링만 하므로 정확히
"그 threshold로 처음부터 decode한 것"과 동일한 결과이면서 forward pass는 중복 없음.

Usage:
    python eval_voxelnet.py --ckpt checkpoints/voxelnet_run1.pt --cache-root ../cache/voxel --split val --score-thresh 0.3 0.5 0.7
    python eval_voxelnet.py --ckpt checkpoints/voxelnet_run1.pt --scenes scene_0044   # 로컬 raw 데이터(캐시 없이 즉석 계산)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from shapely.geometry import Polygon

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:  # train.py와 동일 관례 - 있으면 쓰고 없으면 조용히 폴백
    HAS_TQDM = False

import anchors as anchors_mod
import config
import heatmap_targets as ht
import rotation3d
import stage2_refine as s2
from decode import decode_boxes, rotated_nms
from model import VoxelNet, load_state_dict_compat
from voxelize import augment_with_centroid_offset, voxelize

IOU_THRESHOLDS = (0.25, 0.3, 0.35, 0.4, 0.5)  # main()이 --iou-thresholds로 덮어쓸 수 있음(모듈 전역
                                              # 이라 evaluate/_score_frame이 호출 시점에 이 값을 읽음).
                                              # 교수님 피드백(2026-08-20: IoU0.5는 너무 엄격)에 따라
                                              # 0.3/0.35/0.4를 기본에 추가 — 0.35는 0.3(안정)~0.4(전환)
                                              # 사이 기울기를 채우는 후보 타겟.

_LOCAL_CORNERS_UNIT = np.array([
    [-1, -1, -1], [-1, 1, -1], [1, 1, -1], [1, -1, -1],
    [-1, -1, 1], [-1, 1, 1], [1, 1, 1], [1, -1, 1],
], dtype=np.float64)


def iou_3d(footprint_a, za, footprint_b, zb) -> float:
    """z-only(z-yaw) 근사 IoU - footprint 2D 다각형 교집합 x 높이구간 겹침. GT/예측
    둘 다 진짜로 기울어져 있으면(x,y 회전) 근사가 된다 - iou_3d_obb()가 정확한 버전."""
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


def obb_corners(center: np.ndarray, dims: np.ndarray, R: np.ndarray) -> np.ndarray:
    """(8,3) world corners - center(3,), dims(3,)=[l,w,h], R(3,3) local->world."""
    return _LOCAL_CORNERS_UNIT * (dims / 2) @ R.T + center


def point_in_obb(points: np.ndarray, center: np.ndarray, dims: np.ndarray, R: np.ndarray) -> np.ndarray:
    """points: (N,3) -> (N,) bool. local = R^T @ (p-center), R이 직교행렬이라
    (p-center) @ R == R^T @ (p-center)를 행벡터로 쓴 것과 같다."""
    local = (points - center) @ R
    half = dims / 2
    return np.all(np.abs(local) <= half, axis=1)


def iou_3d_obb(center_a, dims_a, R_a, center_b, dims_b, R_b,
                n_samples: int = 8000, rng: np.random.Generator = None) -> float:
    """진짜 3D OBB-OBB IoU, Monte-Carlo 근사. box A/B의 부피는 정확한 해석값(l*w*h)을
    쓰고, 교집합 부피만 union AABB 안에 점을 뿌려 몬테카를로로 추정한다(전체를
    몬테카를로로 하는 것보다 노이즈가 훨씬 적음 - 두 항은 정확, 한 항만 근사).
    validate_iou_3d_obb.py로 z-yaw-only 특수 케이스에서 기존 정확한 iou_3d()와
    일치하는지 검증됨."""
    # 학습 초반(모델이 아직 안 익었을 때) dim_head 출력이 튀면 np.exp(dim_pred)가
    # overflow해서 l/w/h가 inf/nan이 될 수 있다(heatmap_targets.decode_center_boxes가
    # 만드는 후보라면 어디서든 발생 가능 - eval_voxelnet.evaluate()의 in-training
    # validation, stage2_refine.decoded_candidates_and_targets 둘 다 이 경로를 씀).
    # 그대로면 rng.uniform(lo,hi,...)이 OverflowError로 학습 자체를 죽인다(실측,
    # 2026-08-20 voxelnet_center_stage2_decoded_fgbg_s2 epoch3 val 중 크래시 - 최초
    # 발견은 stage2_refine.py 쪽에서 방어했지만 근본 원인은 이 함수라 여기서 한 번에
    # 막는다 - 호출부마다 따로 막으면 또 다른 경로에서 재발할 수 있음). 비정상 입력은
    # "겹침 없음"(IoU=0)으로 안전하게 처리 - 진짜 오탐/이상치를 걸러내는 목적이 아니라
    # 계산 자체가 불가능한 입력만 배제.
    if not (np.all(np.isfinite(center_a)) and np.all(np.isfinite(dims_a))
            and np.all(np.isfinite(center_b)) and np.all(np.isfinite(dims_b))):
        return 0.0

    rng = rng or np.random.default_rng()
    vol_a = float(np.prod(dims_a))
    vol_b = float(np.prod(dims_b))
    if vol_a <= 0 or vol_b <= 0:
        return 0.0

    corners_a = obb_corners(center_a, dims_a, R_a)
    corners_b = obb_corners(center_b, dims_b, R_b)
    lo = np.minimum(corners_a.min(axis=0), corners_b.min(axis=0))
    hi = np.maximum(corners_a.max(axis=0), corners_b.max(axis=0))
    if np.any(hi <= lo) or not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        return 0.0
    sample_vol = float(np.prod(hi - lo))

    pts = rng.uniform(lo, hi, size=(n_samples, 3))
    in_a = point_in_obb(pts, center_a, dims_a, R_a)
    in_b = point_in_obb(pts, center_b, dims_b, R_b)
    inter_frac = float(np.mean(in_a & in_b))
    inter_vol = inter_frac * sample_vol
    union_vol = vol_a + vol_b - inter_vol
    return float(inter_vol / union_vol) if union_vol > 0 else 0.0


def gt_obb_from_row(row: np.ndarray):
    """row: (13,) [x,y,z,l,w,h,theta_z_rad,6D(6)] - cache_dataset.py의 gt_boxes 한 줄
    (anchors.gt_boxes_from_objects 참고). -> (center(3,), dims(3,), R(3,3))."""
    center = row[0:3]
    dims = row[3:6]
    R = rotation3d.sixd_to_matrix_np(row[7:13])
    return center, dims, R


def pred_obb_from_box(box: dict):
    """decode_boxes()/decode_center_boxes()가 만든 box dict(둘 다 이제 "R" 키를 가짐)
    -> (center(3,), dims(3,), R(3,3))."""
    center = np.array([box["x"], box["y"], box["z"]])
    dims = np.array([box["l"], box["w"], box["h"]])
    return center, dims, box["R"]


def compute_ap(detections: list, n_gt: int):
    if n_gt == 0 or not detections:
        return 0.0, 0.0, 0.0
    confs = np.array([d[0] for d in detections])
    is_tp = np.array([d[1] for d in detections])
    order = np.argsort(-confs)
    is_tp = is_tp[order]

    tp_cum = np.cumsum(is_tp)
    fp_cum = np.cumsum(~is_tp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap, float(precision[-1]) if len(precision) else 0.0, float(recall[-1]) if len(recall) else 0.0


def iter_cached(cache_root: str, split: str):
    """cache_dataset.py 출력 - voxel_features/coords/num_points/gt_boxes 전부 이미 계산됨."""
    cache_root = Path(cache_root)
    with open(cache_root / "manifest.json") as f:
        entries = json.load(f)[split]
    for rel_path in entries:
        with np.load(cache_root / rel_path) as npz:
            yield (rel_path, npz["voxel_features"], npz["coords"], npz["num_points"], npz["gt_boxes"])


def iter_local(scenes: list):
    """캐시 없이 로컬 raw 데이터에서 즉석 계산 (--scenes 전용, 느림 - 소규모 확인용)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Triband_BEV" / "baseline"))
    import common  # noqa: E402
    import filter_outliers  # noqa: E402
    flagged_keys = filter_outliers.load_flagged_keys()
    for scene_id in scenes:
        for frame_idx, objects in filter_outliers.iter_filtered_frames(scene_id, flagged_keys):
            raw = np.fromfile(common.sonar_bin_path(scene_id, frame_idx), dtype=np.float32)
            points = raw.reshape(-1, 4 if raw.size % 4 == 0 else 3)
            if points.shape[1] == 3:
                points = np.concatenate([points, np.zeros((len(points), 1), dtype=np.float32)], axis=1)
            points = points[~np.isnan(points).any(axis=1)]
            voxel_xyzr, coords, num_points = voxelize(
                points, config.POINT_CLOUD_RANGE, config.VOXEL_SIZE,
                config.MAX_POINTS_PER_VOXEL, config.MAX_VOXELS)
            voxel_features = augment_with_centroid_offset(voxel_xyzr, num_points)
            gt_boxes = anchors_mod.gt_boxes_from_objects(objects)
            yield (f"{scene_id}_frame_{frame_idx:06d}", voxel_features, coords, num_points, gt_boxes)


def _score_frame(boxes: list, gt_boxes: np.ndarray, detections: dict, score_thresh: float, nms_iou: float):
    """decode_boxes()가 이미 만든 후보 `boxes`(가장 낮은 threshold 기준 decode됨)를
    `score_thresh`로 다시 필터링 -> NMS -> IoU 매칭까지. `detections[score_thresh][iou_thr]`에
    (conf, is_tp) 튜플을 append한다."""
    filtered = [b for b in boxes if b["score"] >= score_thresh]
    filtered = rotated_nms(filtered, iou_thresh=nms_iou)  # NMS는 여전히 z-yaw BEV footprint 근사(decode.py 참고)

    gt_list = [gt_obb_from_row(row) for row in gt_boxes]
    pred_list = [pred_obb_from_box(b) for b in filtered]
    pred_conf = [b["score"] for b in filtered]

    n_p, n_g = len(pred_list), len(gt_list)
    iou_mat = np.zeros((n_p, n_g))
    rng = np.random.default_rng(0)  # 프레임마다 고정 시드 - 평가 재현성(같은 체크포인트는 항상 같은 AP3D)
    for pi, (pc, pd, pR) in enumerate(pred_list):
        for gi, (gc, gd, gR) in enumerate(gt_list):
            iou_mat[pi, gi] = iou_3d_obb(pc, pd, pR, gc, gd, gR, rng=rng)

    order = np.argsort(-np.array(pred_conf)) if n_p else np.array([], dtype=int)
    for iou_thr in IOU_THRESHOLDS:
        matched_gt = set()
        for pi in order:
            candidate_gt = [gi for gi in range(n_g) if gi not in matched_gt]
            is_tp = False
            if candidate_gt:
                ious = iou_mat[pi, candidate_gt]
                best_local = int(np.argmax(ious))
                best_gt, best_iou = candidate_gt[best_local], ious[best_local]
                if best_iou >= iou_thr:
                    matched_gt.add(best_gt)
                    is_tp = True
            detections[score_thresh][iou_thr].append((pred_conf[pi], is_tp))


def _decode_center_sample(hm, off, z, dim, rot, score_thresh: float):
    """model(head='center') 출력(배치 1개분, (C,H,W) 각각)을 heatmap_targets.decode_center_boxes에
    맞는 (H,W,C) numpy로 변환해서 디코드."""
    hm_np = torch.sigmoid(hm).cpu().numpy()  # (1,H,W)
    return ht.decode_center_boxes(hm_np, off.permute(1, 2, 0).cpu().numpy(),
                                   z.permute(1, 2, 0).cpu().numpy(), dim.permute(1, 2, 0).cpu().numpy(),
                                   rot.permute(1, 2, 0).cpu().numpy(), score_thresh=score_thresh)


def _decode_center_sample_polar(hm, off, z, dim, rot, score_thresh: float):
    """_decode_center_sample()의 polar(Phase1) 버전 - heatmap_targets.decode_center_boxes_polar 사용."""
    hm_np = torch.sigmoid(hm).cpu().numpy()
    return ht.decode_center_boxes_polar(hm_np, off.permute(1, 2, 0).cpu().numpy(),
                                         z.permute(1, 2, 0).cpu().numpy(), dim.permute(1, 2, 0).cpu().numpy(),
                                         rot.permute(1, 2, 0).cpu().numpy(), score_thresh=score_thresh)


@torch.no_grad()
def evaluate(model, device, samples, anchor_grid, score_threshes: list, nms_iou: float, head: str = "anchor",
             total: int | None = None, polar: bool = False):
    """score_threshes 각각에 대해 독립적으로 AP를 낼 수 있는 detections를 모은다.
    반환: detections[score_thresh][iou_thr] -> list[(conf, is_tp)], n_gt_total.
    total: tqdm 진행바용 총 프레임 수 힌트(iter_cached는 manifest 길이로 알 수 있음 -
    iter_local은 제너레이터라 모르면 None, tqdm이 카운트/속도만 보여줌)."""
    detections = {st: {it: [] for it in IOU_THRESHOLDS} for st in score_threshes}
    decode_thresh = min(score_threshes)
    n_gt_total = 0
    n_done = 0

    # stage2가 있으면 추론에도 실제로 반영한다 - 이전엔 학습만 되고 eval에 전혀 안
    # 쓰였다(tianweiy/CenterPoint의 two_stage.py는 return_loss=False일 때 roi_head
    # 출력을 최종 detection으로 쓰는데 우리 구현엔 그 경로가 아예 없었음, 원 저장소
    # 코드로 확인, 2026-08-20 - project_stage2_rng_confound_confirmed 메모리 참고).
    use_stage2_eval = head == "center" and getattr(model.rpn, "stage2", None) is not None
    stage2_capture = {}
    if use_stage2_eval:
        model.rpn.backbone.register_forward_hook(
            lambda m, i, o: stage2_capture.__setitem__("feat", o))

    iterable = tqdm(samples, total=total, desc="eval", unit="frame") if HAS_TQDM else samples
    for sample_id, voxel_features, coords, num_points, gt_boxes in iterable:
        n_done += 1
        if not HAS_TQDM and n_done % 500 == 0:
            print(f"[eval_voxelnet] {n_done}개 프레임 처리됨")
        n_gt_total += len(gt_boxes)

        voxel_features_t = torch.from_numpy(voxel_features).to(device)
        num_points_t = torch.from_numpy(num_points).to(device)
        batch_col = torch.zeros((len(coords), 1), dtype=torch.int64)
        coords_t = torch.cat([batch_col, torch.from_numpy(coords)], dim=1).to(device)

        # 가장 낮은 threshold로 딱 한 번만 decode - forward pass도 프레임당 한 번뿐.
        if head == "anchor":
            cls_pred, reg_pred = model(voxel_features_t, num_points_t, coords_t)
            candidates = decode_boxes(cls_pred[0], reg_pred[0], anchor_grid, score_thresh=decode_thresh)
        else:
            hm, off, z, dim, rot, _density = model(voxel_features_t, num_points_t, coords_t)
            decode_fn = _decode_center_sample_polar if polar else _decode_center_sample
            candidates = decode_fn(hm[0], off[0], z[0], dim[0], rot[0], score_thresh=decode_thresh)
            if use_stage2_eval:
                candidates = s2.refine_candidates(model.rpn.stage2, stage2_capture["feat"], candidates)

        for score_thresh in score_threshes:
            _score_frame(candidates, gt_boxes, detections, score_thresh, nms_iou)

    return detections, n_gt_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cache-root", default=None, help="cache_dataset.py 출력 - 권장")
    parser.add_argument("--split", default="val")
    parser.add_argument("--scenes", nargs="*", default=None, help="--cache-root 대신 로컬 raw 데이터 즉석 계산")
    parser.add_argument("--score-thresh", type=float, nargs="+", default=[0.3])
    parser.add_argument("--nms-iou", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--head", default=None, choices=["anchor", "center"],
                         help="지정 안 하면 체크포인트에 저장된 head를 그대로 씀(train.py가 항상 저장)")
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=None,
                         help="AP3D 매칭 IoU threshold들(공백 구분). 미지정 시 모듈 기본값 "
                              "(0.25 0.3 0.4 0.5) 사용.")
    args = parser.parse_args()

    if args.iou_thresholds:
        global IOU_THRESHOLDS
        IOU_THRESHOLDS = tuple(args.iou_thresholds)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda가 요청됐지만 torch.cuda.is_available()==False - Colab Runtime > "
            "Change runtime type이 GPU(T4)로 설정됐는지 확인할 것.")

    ckpt = torch.load(args.ckpt, map_location=args.device)
    head = args.head or ckpt.get("head", "anchor")
    polar = ckpt.get("polar", False)
    use_grr = ckpt.get("use_grr", False)
    use_pdconv = ckpt.get("use_pdconv", False)
    use_ga = ckpt.get("use_ga", False)
    fg_range_cond = ckpt.get("fg_range_cond", False)
    use_rs_conv = ckpt.get("use_rs_conv", False)
    use_feat_undistort = ckpt.get("use_feat_undistort", False)
    grr_n = ckpt.get("grr_n", None)  # None이면 model이 config.PARTNER_GRR_N로 폴백 - 옛 ckpt 호환
    use_stage2 = ckpt.get("use_stage2", False)
    use_rot_branch = ckpt.get("use_rot_branch", False)
    # fg_head는 state_dict에 있어야 fg_range_cond 채널 수 매칭 가능 - 상태 사전에서 감지.
    has_fg_head = any(k.startswith("rpn.fg_head.") for k in ckpt["model"].keys())
    model = VoxelNet(head=head, polar=polar, use_grr=use_grr, use_stage2=use_stage2,
                      use_rot_branch=use_rot_branch, use_pdconv=use_pdconv, grr_n=grr_n,
                      use_ga=use_ga, use_fg_head=has_fg_head,
                      fg_range_cond=fg_range_cond,
                      use_rs_conv=use_rs_conv,
                      use_feat_undistort=use_feat_undistort).to(args.device)
    load_state_dict_compat(model, ckpt["model"])
    model.eval()
    print(f"head={head} polar={polar} use_grr={use_grr} grr_n={model.grr_n if use_grr else '-'} "
          f"use_pdconv={use_pdconv} use_ga={use_ga} fg_range_cond={fg_range_cond} "
          f"use_stage2={use_stage2} use_rot_branch={use_rot_branch} "
          f"(ckpt epoch={ckpt.get('epoch')})")

    anchor_grid = anchors_mod.build_anchor_grid()
    total = None
    if args.cache_root:
        samples = iter_cached(args.cache_root, args.split)
        with open(Path(args.cache_root) / "manifest.json") as f:
            total = len(json.load(f)[args.split])
    else:
        from dataset import load_scene_split
        scenes = args.scenes if args.scenes else load_scene_split(args.split)
        samples = iter_local(scenes)

    detections, n_gt_total = evaluate(model, args.device, samples, anchor_grid, args.score_thresh,
                                       args.nms_iou, head=head, total=total, polar=polar)

    results = {}
    for score_thresh in args.score_thresh:
        results[score_thresh] = {}
        print(f"--- score_thresh={score_thresh} ---")
        for thr in IOU_THRESHOLDS:
            dets = detections[score_thresh][thr]
            ap, prec, rec = compute_ap(dets, n_gt_total)
            n_preds, n_tp = len(dets), int(sum(d[1] for d in dets))
            results[score_thresh][f"iou_{thr}"] = {"ap3d": ap, "precision_at_conf": prec,
                                                      "recall_at_conf": rec, "n_preds": n_preds,
                                                      "n_tp": n_tp, "n_gt": n_gt_total}
            print(f"IoU>={thr}: AP3D={ap:.4f}  P={prec:.4f}  R={rec:.4f}  "
                  f"(preds={n_preds}, tp={n_tp}, gt={n_gt_total})")

    out_dir = config.VOXELNET_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{Path(args.ckpt).stem}_{args.split}_eval.json"
    with open(out_path, "w") as f:
        json.dump({"checkpoint": str(args.ckpt), "split": args.split,
                    "results_by_score_thresh": {str(k): v for k, v in results.items()}}, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
