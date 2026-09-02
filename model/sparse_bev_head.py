"""sparse_bev_head.py - Parameterized versions of heatmap_targets.build_heatmap_targets
/decode_center_boxes for the experiments/*_bev/voxelnet.py models' output grid, which
has a DIFFERENT resolution/range than the dense pipeline's hardcoded
config.ANCHOR_GRID_SIZE/config.ANCHOR_STRIDE/config.POINT_CLOUD_RANGE (those describe
the ORIGINAL VoxelNet car-config grid, not these experiments' SPARSE_BEV_* one).
Shared by all 3 experiments (exp1_single_stage_bev/exp2_down_slot_up_bev/exp3_zdown_bev).

gt_boxes input format matches sparse_center_head.build_sparse_targets exactly --
(M,13) [x,y,z,l,w,h,theta_z(unused),6D-rot(6)], cache_dataset.py's layout, the same
tensor sonar_diver_dataset.py's collate_fn already produces as batch["gt_boxes"] --
no euler round-trip needed (rotation3d.py has no matrix->euler inverse; going through
6D directly avoids needing one).

Reuses gaussian_radius/draw_gaussian/_point_in_obb/_density_class from
heatmap_targets.py unchanged, same pattern build_sparse_targets already uses.
center_loss.center_voxelnet_loss itself needs NO wrapper: it already takes plain
(B,C,H,W) tensors with no grid-size assumptions baked in.
"""
import numpy as np

import rotation3d
from heatmap_targets import gaussian_radius, draw_gaussian, _point_in_obb, _density_class


def build_bev_targets(gt_boxes: np.ndarray, points: np.ndarray, grid_size, stride_xy, pc_range,
                       min_overlap: float = 0.7) -> dict:
    """gt_boxes: (M,13) [x,y,z,l,w,h,theta_z(unused),6D-rot(6)] for ONE sample (numpy).
    points: (P,3+) raw points for this sample, for the density aux target -- pass
    None to skip density supervision (density_target comes back all zeros, unused
    unless center_loss.center_voxelnet_loss is given density_pred/density_target).
    grid_size: (W,H) output grid the model actually produces (RPNCenterHead's
    (B,*,H,W) resolution). stride_xy: (sx,sy) meters/cell AT THAT resolution (the
    full effective stride from raw points to this grid, not just the voxel size).
    pc_range: (xmin,ymin,zmin,xmax,ymax,zmax) -- only x0,y0 used, same as
    heatmap_targets.build_heatmap_targets.

    Same return shape/keys/semantics as heatmap_targets.build_heatmap_targets."""
    W, H = grid_size
    sx, sy = stride_xy
    x0, y0 = pc_range[0], pc_range[1]
    points_xyz = points[:, :3] if points is not None else None

    heatmap = np.zeros((1, H, W), dtype=np.float32)
    reg_mask = np.zeros((H, W), dtype=bool)
    offset = np.zeros((H, W, 2), dtype=np.float32)
    z_t = np.zeros((H, W, 1), dtype=np.float32)
    dim_t = np.zeros((H, W, 3), dtype=np.float32)
    rot_t = np.zeros((H, W, 6), dtype=np.float32)
    density_t = np.zeros((H, W), dtype=np.int64)

    for row_box in gt_boxes:
        x, y, z, l, w, h = row_box[:6]
        sixd = row_box[7:13]
        gx_cell = (x - x0) / sx - 0.5
        gy_cell = (y - y0) / sy - 0.5
        col, row = int(round(gx_cell)), int(round(gy_cell))
        if not (0 <= col < W and 0 <= row < H):
            continue

        l_cells, w_cells = l / sx, w / sy
        radius = gaussian_radius(w_cells, l_cells, min_overlap)
        draw_gaussian(heatmap[0], gx_cell, gy_cell, radius)

        reg_mask[row, col] = True
        offset[row, col] = [gx_cell - col, gy_cell - row]
        z_t[row, col, 0] = z
        dim_t[row, col] = [np.log(l), np.log(w), np.log(h)]
        rot_t[row, col] = sixd

        if points_xyz is not None:
            R = rotation3d.sixd_to_matrix_np(sixd)
            center = np.array([x, y, z])
            dims = np.array([l, w, h])
            n_in_box = int(_point_in_obb(points_xyz, center, dims, R).sum())
            density_t[row, col] = _density_class(n_in_box)

    return {"heatmap": heatmap, "reg_mask": reg_mask, "offset": offset, "density": density_t,
            "z": z_t, "dim": dim_t, "rot": rot_t}


def decode_bev_center_boxes(heatmap_pred, offset_pred, z_pred, dim_pred, rot_pred,
                             stride_xy, pc_range, score_thresh: float = 0.3, max_peaks: int = 100):
    """Parameterized twin of heatmap_targets.decode_center_boxes -- same inputs/return
    shape, stride_xy/pc_range passed explicitly instead of read from config.py."""
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

    sx, sy = stride_xy
    x0, y0 = pc_range[0], pc_range[1]

    boxes = []
    for row, col in zip(rows, cols):
        dx, dy = offset_pred[row, col]
        gx_cell, gy_cell = col + dx, row + dy
        x = x0 + (gx_cell + 0.5) * sx
        y = y0 + (gy_cell + 0.5) * sy
        z = float(z_pred[row, col, 0])
        l, w, h = np.exp(dim_pred[row, col])
        R = rotation3d.sixd_to_matrix_np(rot_pred[row, col])
        theta = float(np.arctan2(R[1, 0], R[0, 0]))
        boxes.append({"score": float(hm[row, col]), "x": float(x), "y": float(y), "z": z,
                       "l": float(l), "w": float(w), "h": float(h), "theta": theta, "R": R})
    return boxes
