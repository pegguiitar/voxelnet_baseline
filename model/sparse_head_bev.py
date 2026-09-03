"""sparse_head_bev.py - RPNCenterHead's head set (heatmap/offset/z/dim/rot/density),
dimension-transformed from dense (B,C,H,W) 2D-conv-head operation to sparse
per-active-cell operation, for experiments/exp4_fully_sparse_bev -- the "never go
dense" sibling of sparse_center_head.py (which does the same transform for a
genuinely-3D backbone where z is still a spatial axis). Here z has already been
compressed away as a spatial axis (by the z-down stages, see that experiment's
voxelnet.py) before this head ever runs, so it comes back as a REGRESSED value
(offset_head/z_head split, exactly like model.RPNCenterHead), not folded into a 3D
offset the way sparse_center_head.SparseCenterHead does.

Target assignment reuses sparse_center_head.build_sparse_targets' core idea
(CenterPoint single-nearest-active-cell positive assignment, graded-Gaussian heatmap
credit to every active cell by distance) but in 2D (x,y cell-distance only, coords
are (N,3) [batch,y,x] not (N,4)) and with a z regression target added (z is no
longer a coordinate here, same as sparse_bev_head.build_bev_targets' dense analog
handles it). Unlike a dense grid, there's no guarantee any active cell sits at/near
a GT's exact target cell -- a GT can still contribute heatmap credit to nearby
active cells without ever getting a positive regression assignment (see
build_sparse_bev_targets' docstring).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import rotation3d
from heatmap_targets import gaussian_radius


class SparseBEVCenterHead(nn.Module):
    """Same head set as model.RPNCenterHead's baseline -- every nn.Conv2d(...,
    kernel_size=1) becomes nn.Linear, exactly like sparse_center_head.SparseCenterHead
    does for the 3D case (a 1x1 conv never mixes spatial neighbors, so this is a pure
    dimension transform, not a redesign)."""

    def __init__(self, in_channels: int):
        super().__init__()
        c = in_channels
        self.heatmap_head = nn.Linear(c, 1)
        self.offset_head = nn.Linear(c, 2)   # [dx,dy] sub-cell, cell-size units
        self.z_head = nn.Linear(c, 1)        # absolute z (m) -- no spatial z left to offset from
        self.dim_head = nn.Linear(c, 3)      # [log l, log w, log h]
        self.rot_head = nn.Linear(c, 6)      # 6D continuous rotation (rotation3d.py)
        self.density_head = nn.Linear(c, 3)  # RAANet-style aux (sparse/adequate/dense) -- kept for
        # structural parity with RPNCenterHead/SparseCenterHead; not supervised here (same
        # reason exp1/2/3's build_bev_targets skips it -- collate_fn doesn't expose raw points).

        prior_prob = 0.1
        nn.init.constant_(self.heatmap_head.bias, -math.log((1 - prior_prob) / prior_prob))

    def forward(self, features: torch.Tensor):
        """features: (N,C) sparse 2D backbone output. Returns dict of (N,*) raw
        predictions (heatmap is a raw logit -- sigmoid happens in the loss/decode)."""
        return {
            "heatmap": self.heatmap_head(features),
            "offset": self.offset_head(features),
            "z": self.z_head(features),
            "dim": self.dim_head(features),
            "rot": self.rot_head(features),
            "density": self.density_head(features),
        }


@torch.no_grad()
def build_sparse_bev_targets(gt_boxes_list: list, coords: torch.Tensor, stride_xy, pc_range,
                              min_overlap: float = 0.7, search_radius: float = 1.5):
    """Per-active-cell targets, the 2D-after-height-compression analog of
    sparse_center_head.build_sparse_targets.
    gt_boxes_list: list[B] of (M_b,13) [x,y,z,l,w,h,theta_z(unused),6D-rot(6)] tensors
    (sonar_diver_dataset.py's collate_fn layout).
    coords: (N,3) [batch,y_idx,x_idx], this batch's sparse-2D-backbone-output active cells.
    stride_xy/pc_range: (sx,sy) meters/cell and (x0,y0,...) world origin at this
    output resolution -- same role as sparse_bev_head.build_bev_targets' dense analog.

    Positive assignment (CenterPoint single-nearest-cell): for each GT, find the
    active cell in that sample closest (in cell units) to the GT's x,y center; if
    none exists within `search_radius` cells (can happen -- unlike a dense grid,
    there's no guarantee *some* cell sits exactly at/near the rounded target cell),
    that GT contributes to the heatmap (nearby cells still get graded credit) but not
    to reg_mask/offset/z/dim/rot (nothing to regress from).

    Returns dict of (N,*) tensors matching SparseBEVCenterHead's output shapes
    (except density, not built here), plus "reg_mask": (N,) bool.
    """
    device = coords.device
    N = coords.shape[0]
    sx, sy = stride_xy
    x0, y0 = pc_range[0], pc_range[1]

    heatmap = torch.zeros(N, 1, device=device)
    reg_mask = torch.zeros(N, dtype=torch.bool, device=device)
    offset_t = torch.zeros(N, 2, device=device)
    z_t = torch.zeros(N, 1, device=device)
    dim_t = torch.zeros(N, 3, device=device)
    rot_t = torch.zeros(N, 6, device=device)

    cell_xy = coords[:, [2, 1]].float()  # (N,2) [x_idx,y_idx] per active cell
    batch_col = coords[:, 0]

    for b, gt_boxes in enumerate(gt_boxes_list):
        if gt_boxes.shape[0] == 0:
            continue
        sample_mask = batch_col == b
        sample_idx = sample_mask.nonzero(as_tuple=True)[0]
        if sample_idx.numel() == 0:
            continue  # no active cells at all for this sample -- nothing to supervise
        sample_xy = cell_xy[sample_idx]

        for row in gt_boxes:
            x, y, z, l, w, h = (row[i].item() for i in range(6))
            six = row[7:13]

            gx_cell = (x - x0) / sx - 0.5
            gy_cell = (y - y0) / sy - 0.5
            target_cell = torch.tensor([gx_cell, gy_cell], device=device)

            l_cells, w_cells = l / sx, w / sy
            radius = gaussian_radius(w_cells, l_cells, min_overlap)
            sigma = (2 * radius + 1) / 6.0
            dist2 = ((sample_xy - target_cell) ** 2).sum(dim=1)
            gauss = torch.exp(-dist2 / (2 * sigma * sigma + 1e-9))
            gauss[gauss < 1e-4] = 0
            heatmap[sample_idx, 0] = torch.maximum(heatmap[sample_idx, 0], gauss)

            rounded = torch.tensor([round(gx_cell), round(gy_cell)], device=device, dtype=torch.float32)
            d2_to_rounded = ((sample_xy - rounded) ** 2).sum(dim=1)
            nearest_local = torch.argmin(d2_to_rounded)
            if d2_to_rounded[nearest_local].item() > search_radius ** 2:
                continue  # no cell close enough to regress from -- heatmap credit above still applies
            pos_row = sample_idx[nearest_local]

            reg_mask[pos_row] = True
            offset_t[pos_row] = target_cell - sample_xy[nearest_local]
            z_t[pos_row, 0] = z
            dim_t[pos_row] = torch.tensor([math.log(l), math.log(w), math.log(h)], device=device)
            rot_t[pos_row] = six

    return {"heatmap": heatmap, "reg_mask": reg_mask, "offset": offset_t,
            "z": z_t, "dim": dim_t, "rot": rot_t}


def sparse_bev_center_loss(pred: dict, target: dict, reg_weight: float = 1.0,
                            focal_alpha: float = 2.0, focal_beta: float = 4.0):
    """Same math as center_loss.center_voxelnet_loss / sparse_center_head.sparse_center_loss,
    with a z term (regressed value here, not folded into offset)."""
    from center_loss import gaussian_focal_loss  # reuse verbatim -- shape-agnostic (elementwise + .sum())

    hm_loss = gaussian_focal_loss(pred["heatmap"], target["heatmap"], alpha=focal_alpha, beta=focal_beta)

    reg_mask = target["reg_mask"]
    n_pos = reg_mask.float().sum().clamp_min(1)
    w = reg_mask.unsqueeze(-1).float()  # (N,1)

    offset_loss = (F.l1_loss(pred["offset"], target["offset"], reduction="none") * w).sum() / n_pos
    z_loss = (F.l1_loss(pred["z"], target["z"], reduction="none") * w).sum() / n_pos
    dim_loss = (F.l1_loss(pred["dim"], target["dim"], reduction="none") * w).sum() / n_pos
    rot_loss = (F.l1_loss(pred["rot"], target["rot"], reduction="none") * w).sum() / n_pos
    reg_loss = offset_loss + z_loss + dim_loss + rot_loss

    stats = {"hm_loss": hm_loss.item(), "reg_loss": reg_loss.item(),
              "offset_loss": offset_loss.item(), "z_loss": z_loss.item(),
              "dim_loss": dim_loss.item(), "rot_loss": rot_loss.item(),
              "n_pos": int(reg_mask.sum().item())}

    total = hm_loss + reg_weight * reg_loss
    return total, stats


def _build_index_grid_2d(coords: torch.Tensor, batch_size: int, grid_size, device=None) -> torch.Tensor:
    """2D analog of sparse_ops.build_index_grid (that one is hardcoded to (N,4)
    coords / 3D grid_size -- kept separate rather than generalizing it, since its
    only other caller, sparse_center_head.py, is 3D-only)."""
    H, W = grid_size
    device = device or coords.device
    grid = torch.full((batch_size, H, W), -1, dtype=torch.long, device=device)
    if coords.shape[0] > 0:
        grid[coords[:, 0], coords[:, 1], coords[:, 2]] = torch.arange(coords.shape[0], device=device)
    return grid


def find_local_peaks_sparse_2d(scores: torch.Tensor, coords: torch.Tensor, index_grid: torch.Tensor,
                                grid_size, score_thresh: float) -> torch.Tensor:
    """2D (8-neighbor) analog of sparse_center_head.find_local_peaks_sparse.
    grid_size must be (H,W), matching coords' [batch,y_idx,x_idx] row/col convention
    (NOT the (W,H) order some of this codebase's head_grid_size attributes use for
    naming consistency with the dense pipeline -- swap before calling if needed)."""
    H, W = grid_size
    n = coords.shape[0]
    is_peak = torch.ones(n, dtype=torch.bool, device=scores.device)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nb = coords.clone()
            nb[:, 1] += dy
            nb[:, 2] += dx
            valid = (nb[:, 1] >= 0) & (nb[:, 1] < H) & (nb[:, 2] >= 0) & (nb[:, 2] < W)
            nb_idx = torch.full((n,), -1, dtype=torch.long, device=scores.device)
            if valid.any():
                v = nb[valid]
                nb_idx[valid] = index_grid[v[:, 0], v[:, 1], v[:, 2]]
            has_nb = nb_idx >= 0
            nb_scores = torch.zeros_like(scores)
            nb_scores[has_nb] = scores[nb_idx[has_nb]]
            is_peak &= scores >= nb_scores
    return is_peak & (scores > score_thresh)


def decode_sparse_bev_boxes(pred: dict, coords: torch.Tensor, stride_xy, batch_size: int,
                             grid_size_hw, pc_range, score_thresh: float = 0.3):
    """2D analog of sparse_center_head.decode_sparse_center_boxes. pred: dict from
    SparseBEVCenterHead.forward. coords: this batch's sparse-2D-backbone-output
    active cells. grid_size_hw: (H,W) at that resolution (see find_local_peaks_sparse_2d's
    note on ordering). Returns list[B] of list-of-dict {score,x,y,z,l,w,h,R} (world
    space; local-max peak-picking on the active set acts as NMS, same as
    decode_sparse_center_boxes)."""
    device = coords.device
    scores = torch.sigmoid(pred["heatmap"].squeeze(-1))  # (N,)
    index_grid = _build_index_grid_2d(coords, batch_size, grid_size_hw, device=device)
    peak_mask = find_local_peaks_sparse_2d(scores, coords, index_grid, grid_size_hw, score_thresh)

    sx, sy = stride_xy
    x0, y0 = pc_range[0], pc_range[1]
    cell_xy = coords[:, [2, 1]].float()  # [x_idx,y_idx]
    offset = pred["offset"]
    gx = cell_xy[:, 0] + offset[:, 0]
    gy = cell_xy[:, 1] + offset[:, 1]
    xs = x0 + (gx + 0.5) * sx
    ys = y0 + (gy + 0.5) * sy
    zs = pred["z"].squeeze(-1)
    dims = pred["dim"].exp()
    R_all = rotation3d.sixd_to_matrix_torch(pred["rot"])

    results = []
    for b in range(batch_size):
        m = peak_mask & (coords[:, 0] == b)
        idx = m.nonzero(as_tuple=True)[0]
        boxes = []
        for i in idx.tolist():
            R = R_all[i].detach().cpu().numpy()
            boxes.append({
                "score": scores[i].item(),
                "x": xs[i].item(), "y": ys[i].item(), "z": zs[i].item(),
                "l": dims[i, 0].item(), "w": dims[i, 1].item(), "h": dims[i, 2].item(),
                "R": R,
            })
        results.append(boxes)
    return results
