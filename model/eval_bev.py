"""eval_bev.py - lightweight per-epoch AP3D check for the BEV-collapsing sparse
experiments (experiments/*_bev/train.py), reusing eval_voxelnet.py's exact IoU/AP
definitions (iou_3d_obb Monte-Carlo OBB IoU, compute_ap VOC2010-style all-point AP)
so numbers are directly comparable to this repo's own dense-pipeline eval and to
eval_sparse.py's (sparse_voxelnet.py's fully-sparse-head experiment) equivalent.

Deliberately cheaper than a full eval_voxelnet.py run (this runs every epoch during
training, not as a one-off final report): a single score_thresh (not a sweep), a
smaller Monte-Carlo sample count per IoU pair, and greedy per-frame matching -- same
algorithm as eval_voxelnet._score_frame / eval_sparse.evaluate_test_ap, just against
sparse_bev_head.decode_bev_center_boxes's box dict shape instead.

Works for any experiments/*_bev/voxelnet.py model (exp1/exp2/exp3/dense_baseline):
they all share the same forward() signature (voxel_features, num_points, coords) ->
(heatmap, offset, z, dim, rot, density) and the same head_grid_size/head_stride/
pc_range attributes, so this one function serves all of them.

2026-09-03: widened from AP-only at 3 thresholds to AP+precision+recall at 5
thresholds, matching the confirmed dense-pipeline baseline's own per-epoch val
metric set exactly (model/train.py's VAL_LOG_IOUS=(0.25,0.3,0.35,0.4,0.5), each
logged with AP3D/precision/recall to {run}_val_history.json there) -- precision/
recall were already being computed for free by eval_voxelnet.compute_ap, just
discarded here before. VAL_TARGET_IOU=0.35 is the baseline's primary/reported
threshold (model/train.py's own comment: "0.35가 primary 타겟, 교수님 피드백 후
확정"), kept here for any caller that wants a single headline number."""
import time

import numpy as np
import torch

import rotation3d
from eval_voxelnet import iou_3d_obb, compute_ap
from sparse_bev_head import decode_bev_center_boxes

IOU_THRESHOLDS = (0.25, 0.30, 0.35, 0.40, 0.50)  # = model/train.py's VAL_LOG_IOUS
PRINT_IOUS = (0.30, 0.35, 0.40)                  # = model/train.py's VAL_PRINT_IOUS (stdout subset)
VAL_TARGET_IOU = 0.35                            # = model/train.py's VAL_TARGET_IOU (primary/reported)


@torch.no_grad()
def evaluate_bev_ap(model, data_loader, device, score_thresh: float = 0.1,
                     mc_samples: int = 2000, max_frames: int = None):
    """Returns ({iou_thr: (ap, precision, recall)}, elapsed_seconds) -- precision/
    recall are compute_ap's cumulative values at the full detection list (i.e. every
    detection at or above `score_thresh`), same definition model/train.py's
    _run_validation/evaluate use. max_frames caps how many frames (not batches) get
    scored, for keeping per-epoch overhead bounded -- None = whole split (val split
    here is ~8000 frames; the Monte-Carlo IoU is per pred-GT pair, so an uncapped run
    every epoch would dominate total training time)."""
    t0 = time.time()
    model.eval()
    rng = np.random.default_rng(0)  # fixed seed -- same checkpoint always scores identically

    detections = {t: [] for t in IOU_THRESHOLDS}
    n_gt_total = 0
    n_frames_done = 0

    for batch in data_loader:
        if max_frames is not None and n_frames_done >= max_frames:
            break
        voxel_features = batch["voxel_features"].to(device, non_blocking=True)
        num_points = batch["num_points"].to(device, non_blocking=True)
        coords = batch["coords"].to(device, non_blocking=True)
        gt_boxes_list = batch["gt_boxes"]  # keep on CPU -- only used for numpy IoU below

        heatmap, offset, z, dim, rot, density = model(voxel_features, num_points, coords)
        batch_size = heatmap.shape[0]

        for b in range(batch_size):
            if max_frames is not None and n_frames_done >= max_frames:
                break
            n_frames_done += 1
            gt_boxes = gt_boxes_list[b].numpy()
            n_gt_total += len(gt_boxes)

            pred_boxes = decode_bev_center_boxes(
                torch.sigmoid(heatmap[b]).cpu().numpy(),
                offset[b].permute(1, 2, 0).cpu().numpy(),
                z[b].permute(1, 2, 0).cpu().numpy(),
                dim[b].permute(1, 2, 0).cpu().numpy(),
                rot[b].permute(1, 2, 0).cpu().numpy(),
                model.head_stride, model.pc_range, score_thresh=score_thresh,
            )

            gt_list = [(row[0:3], row[3:6], rotation3d.sixd_to_matrix_np(row[7:13])) for row in gt_boxes]
            n_p, n_g = len(pred_boxes), len(gt_list)
            iou_mat = np.zeros((n_p, n_g))
            for pi, pb in enumerate(pred_boxes):
                pc, pd, pR = np.array([pb["x"], pb["y"], pb["z"]]), np.array([pb["l"], pb["w"], pb["h"]]), pb["R"]
                for gi, (gc, gd, gR) in enumerate(gt_list):
                    iou_mat[pi, gi] = iou_3d_obb(pc, pd, pR, gc, gd, gR, n_samples=mc_samples, rng=rng)

            pred_conf = [pb["score"] for pb in pred_boxes]
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
                    detections[iou_thr].append((pred_conf[pi], is_tp))

    model.train()
    metrics_by_thresh = {}
    for iou_thr in IOU_THRESHOLDS:
        ap, precision, recall = compute_ap(detections[iou_thr], n_gt_total)
        metrics_by_thresh[iou_thr] = (ap, precision, recall)
    return metrics_by_thresh, time.time() - t0
