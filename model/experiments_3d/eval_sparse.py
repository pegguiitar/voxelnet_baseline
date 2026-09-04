"""eval_sparse.py - lightweight per-epoch AP3D check for the sparse backbone
experiment (train_sparse.py), reusing eval_voxelnet.py's exact IoU/AP definitions
(iou_3d_obb Monte-Carlo OBB IoU, compute_ap VOC2010-style all-point AP) so numbers
are directly comparable to this repo's own dense-pipeline eval.

Deliberately cheaper than a full eval_voxelnet.py run (this runs every epoch during
training, not as a one-off final report): a single score_thresh (not a sweep), a
smaller Monte-Carlo sample count per IoU pair, and greedy per-frame matching --
same algorithm as eval_voxelnet._score_frame, just inlined against
decode_sparse_center_boxes's box dict shape (already has "R" as a full rotation
matrix, no anchors.py residual decoding needed)."""

import time

import numpy as np
import torch

import rotation3d
from eval_voxelnet import iou_3d_obb, compute_ap
from sparse_center_head import decode_sparse_center_boxes

IOU_THRESHOLDS = (0.30, 0.35, 0.40)


@torch.no_grad()
def evaluate_test_ap(model, test_loader, device, score_thresh: float = 0.1,
                      mc_samples: int = 2000, max_frames: int = None):
    """Returns ({iou_thr: ap}, elapsed_seconds). max_frames caps how many frames
    (not batches) get scored, for keeping per-epoch overhead bounded -- None = whole
    test set."""
    t0 = time.time()
    model.eval()
    rng = np.random.default_rng(0)  # fixed seed -- same checkpoint always scores identically

    detections = {t: [] for t in IOU_THRESHOLDS}
    n_gt_total = 0
    n_frames_done = 0

    for batch in test_loader:
        if max_frames is not None and n_frames_done >= max_frames:
            break
        voxel_features = batch["voxel_features"].to(device, non_blocking=True)
        num_points = batch["num_points"].to(device, non_blocking=True)
        coords = batch["coords"].to(device, non_blocking=True)
        gt_boxes_list = batch["gt_boxes"]  # keep on CPU -- only used for numpy IoU below
        batch_size = batch["batch_size"]

        pred, out_coords, out_grid_size = model(voxel_features, num_points, coords)
        boxes_per_sample = decode_sparse_center_boxes(
            pred, out_coords, model.stride, batch_size, out_grid_size, score_thresh=score_thresh)

        for b in range(batch_size):
            if max_frames is not None and n_frames_done >= max_frames:
                break
            n_frames_done += 1
            gt_boxes = gt_boxes_list[b].numpy()
            n_gt_total += len(gt_boxes)
            pred_boxes = boxes_per_sample[b]

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
    ap_by_thresh = {}
    for iou_thr in IOU_THRESHOLDS:
        ap, _, _ = compute_ap(detections[iou_thr], n_gt_total)
        ap_by_thresh[iou_thr] = ap
    return ap_by_thresh, time.time() - t0
