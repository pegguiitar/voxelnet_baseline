"""eval_sparse_bev.py - per-epoch AP3D/precision/recall check for
experiments/exp4_fully_sparse_bev, same metric set as eval_bev.py (matching the
confirmed dense-pipeline baseline: AP3D/precision/recall @ IoU 0.25/0.3/0.35/0.4/0.5)
but against SparseBEVFullyVoxelNet's (pred_dict, coords, batch_size) output and
sparse_head_bev.decode_sparse_bev_boxes, instead of the dense-tuple output +
sparse_bev_head.decode_bev_center_boxes eval_bev.py assumes."""
import time

import numpy as np
import torch

import rotation3d
from eval_voxelnet import iou_3d_obb, compute_ap
from sparse_head_bev import decode_sparse_bev_boxes

IOU_THRESHOLDS = (0.25, 0.30, 0.35, 0.40, 0.50)  # = model/train.py's VAL_LOG_IOUS
PRINT_IOUS = (0.30, 0.35, 0.40)                  # = model/train.py's VAL_PRINT_IOUS
VAL_TARGET_IOU = 0.35                            # = model/train.py's VAL_TARGET_IOU


@torch.no_grad()
def evaluate_sparse_bev_ap(model, data_loader, device, score_thresh: float = 0.1,
                            mc_samples: int = 2000, max_frames: int = None):
    """Returns ({iou_thr: (ap, precision, recall)}, elapsed_seconds). max_frames caps
    how many frames get scored (see eval_bev.py's identical rationale)."""
    t0 = time.time()
    model.eval()
    rng = np.random.default_rng(0)

    detections = {t: [] for t in IOU_THRESHOLDS}
    n_gt_total = 0
    n_frames_done = 0

    for batch in data_loader:
        if max_frames is not None and n_frames_done >= max_frames:
            break
        voxel_features = batch["voxel_features"].to(device, non_blocking=True)
        num_points = batch["num_points"].to(device, non_blocking=True)
        coords = batch["coords"].to(device, non_blocking=True)
        gt_boxes_list = batch["gt_boxes"]

        pred, out_coords, batch_size = model(voxel_features, num_points, coords)
        boxes_per_sample = decode_sparse_bev_boxes(
            pred, out_coords, model.head_stride, batch_size, model.head_grid_size_hw,
            model.pc_range, score_thresh=score_thresh)

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
    metrics_by_thresh = {}
    for iou_thr in IOU_THRESHOLDS:
        ap, precision, recall = compute_ap(detections[iou_thr], n_gt_total)
        metrics_by_thresh[iou_thr] = (ap, precision, recall)
    return metrics_by_thresh, time.time() - t0
