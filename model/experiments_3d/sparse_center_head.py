"""sparse_center_head.py - RPNCenterHead's baseline head set (heatmap, offset, dim,
6D rotation, density), dimension-transformed from dense (B,C,H,W) 2D-conv-head
operation to sparse per-voxel operation on a 3D backbone's (N,C) active-voxel output.

Kept identical to model.py's RPNCenterHead: heatmap/dim/rot(6D)/density heads, the
CornerNet-style gaussian-penalty-reduced focal loss (center_loss.py, alpha=2/beta=4 --
same formula this project's dense center head already uses), and the CenterPoint
single-nearest-cell positive-assignment philosophy (heatmap_targets.py). The one real
structural change (not just a dimension reshape): the dense head's separate `z_head`
(needed only because ConvMiddleLayers collapsed z into channels, so z had no spatial
sub-cell position left to offset from) is folded into a 3D `offset_head` (dx,dy,dz),
since z is now a real spatial axis with its own active-voxel granularity -- exactly
how offset regression already works in a true 3D backbone.

Every nn.Conv2d(..., kernel_size=1) in the dense head is mathematically a per-position
Linear layer (a 1x1 conv never mixes spatial neighbors) -- that's what makes this a
pure dimension transform rather than a redesign: nn.Linear(C, out) applied to the
sparse backbone's (N,C) active-voxel features is the *same computation*, just without
the (B,H,W) grid these voxels no longer live on.

Coordinate convention: matches voxelize.py's native voxel_coords, NOT sparse_ops.py's
own [batch,x,y,z] docstring -- coords here are (N,4) [batch_idx, z_idx, y_idx, x_idx]
(sparse_ops/backbone3d treat all 3 spatial columns generically, so this is safe as
long as grid_size is passed in the same [D,H,W] order everywhere, which sparse_voxelnet.py
does)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
import rotation3d
from heatmap_targets import gaussian_radius, _density_class, _point_in_obb
from sparse_ops import build_index_grid


class SparseCenterHead(nn.Module):
    """Same head set as model.RPNCenterHead's baseline (no fg_head/stage2/rot_branch/
    polar extras -- those are experimental tracks the README marks as not part of the
    confirmed recipe). Operates on (N,C) sparse per-voxel features instead of
    (B,C,H,W): every head is nn.Linear instead of nn.Conv2d(...,kernel_size=1)."""

    def __init__(self, in_channels: int):
        super().__init__()
        c = in_channels
        self.heatmap_head = nn.Linear(c, 1)
        self.offset_head = nn.Linear(c, 3)   # [dx,dy,dz] sub-voxel, voxel-size units -- absorbs the dense head's separate z_head
        self.dim_head = nn.Linear(c, 3)      # [log l, log w, log h]
        self.rot_head = nn.Linear(c, 6)      # 6D continuous rotation (rotation3d.py)
        self.density_head = nn.Linear(c, 3)  # RAANet-style aux (sparse/adequate/dense)

        # Same focal-loss bias init as model.RPNCenterHead, same reasoning (see its
        # comment): without this, a uniform sigmoid(0)=0.5 prediction across thousands
        # of active voxels with ~1-2 positives per frame collapses the heatmap head to
        # "always background".
        prior_prob = 0.1
        nn.init.constant_(self.heatmap_head.bias, -math.log((1 - prior_prob) / prior_prob))

    def forward(self, features: torch.Tensor):
        """features: (N,C) sparse backbone output, N = active voxels pooled across the
        whole batch. Returns dict of (N,*) raw predictions (heatmap/density are raw
        logits -- sigmoid/softmax happen in the loss/decode, matching model.py)."""
        return {
            "heatmap": self.heatmap_head(features),   # (N,1)
            "offset": self.offset_head(features),     # (N,3)
            "dim": self.dim_head(features),           # (N,3)
            "rot": self.rot_head(features),           # (N,6)
            "density": self.density_head(features),   # (N,3)
        }


def _voxel_centers_world(coords: torch.Tensor, pc_range, voxel_size, stride: int) -> torch.Tensor:
    """coords: (N,4) [batch,z_idx,y_idx,x_idx] at the BACKBONE'S OUTPUT resolution
    (already downsampled by `stride`). -> (N,3) world [x,y,z] of each voxel's center."""
    vx, vy, vz = voxel_size
    eff = torch.tensor([vx * stride, vy * stride, vz * stride], device=coords.device, dtype=torch.float32)
    x0, y0, z0 = pc_range[0], pc_range[1], pc_range[2]
    origin = torch.tensor([x0, y0, z0], device=coords.device, dtype=torch.float32)
    idx_xyz = coords[:, [3, 2, 1]].float()  # reorder [z,y,x] -> [x,y,z]
    return origin + (idx_xyz + 0.5) * eff


@torch.no_grad()
def build_sparse_targets(gt_boxes_list: list, coords: torch.Tensor, stride: int,
                          points_list: list = None, min_overlap: float = 0.7):
    """Per-active-voxel targets, the sparse analog of heatmap_targets.build_heatmap_targets.
    gt_boxes_list: list[B] of (M_b,13) tensors [x,y,z,l,w,h,theta_z(unused),6D-rot(6)]
    (cache_dataset.py's gt_boxes layout -- these must be loaded via
    CachedVoxelNetDataset(..., load_gt_boxes=True); the dense heatmap/reg_mask/offset/
    z/dim/rot/density cache entries are NOT used here since they're baked for the fixed
    dense (H,W) grid, not this batch's actual active-voxel set).
    coords: (N,4) [batch,z_idx,y_idx,x_idx], this batch's backbone-output active voxels.
    points_list: optional list[B] of (P,3+) raw points, for the density aux target
    (point-in-box count) -- omit to skip density supervision (density_target=None).

    Positive assignment (CenterPoint single-nearest-cell, same philosophy as the dense
    version): for each GT, find the active voxel in that sample closest (in cell units)
    to the GT's center; if none exists within `search_radius` cells (can happen -- unlike
    a dense grid, there's no guarantee *some* voxel sits exactly at/near the rounded
    target cell), that GT contributes to the heatmap (nearby voxels still get graded
    credit) but not to reg_mask/offset/dim/rot/density (nothing to regress from).

    Returns dict of (N,*) tensors matching SparseCenterHead's output shapes, plus
    "reg_mask": (N,) bool and "density": (N,) int64 (only meaningful where reg_mask).
    """
    device = coords.device
    N = coords.shape[0]
    pc_range = config.SPARSE_POINT_CLOUD_RANGE
    voxel_size = config.SPARSE_VOXEL_SIZE
    vx, vy, vz = voxel_size
    eff_vx, eff_vy, eff_vz = vx * stride, vy * stride, vz * stride
    search_radius = 1.5  # cells, in the coarser post-stride grid -- generous enough to
    # tolerate the GT's rounded target cell being unoccupied by exactly one cell, without
    # accidentally grabbing a voxel that belongs to a different nearby object.

    heatmap = torch.zeros(N, 1, device=device)
    reg_mask = torch.zeros(N, dtype=torch.bool, device=device)
    offset_t = torch.zeros(N, 3, device=device)
    dim_t = torch.zeros(N, 3, device=device)
    rot_t = torch.zeros(N, 6, device=device)
    density_t = torch.zeros(N, dtype=torch.long, device=device)

    voxel_xyz_idx = coords[:, [3, 2, 1]].float()  # (N,3) [x_idx,y_idx,z_idx] per active voxel
    batch_col = coords[:, 0]

    for b, gt_boxes in enumerate(gt_boxes_list):
        if gt_boxes.shape[0] == 0:
            continue
        sample_mask = batch_col == b
        sample_idx = sample_mask.nonzero(as_tuple=True)[0]
        if sample_idx.numel() == 0:
            continue  # no active voxels at all for this sample (empty frame) -- nothing to supervise
        sample_xyz_idx = voxel_xyz_idx[sample_idx]  # (n_b,3)

        points = points_list[b] if points_list is not None else None
        for row in gt_boxes:
            x, y, z, l, w, h = (row[i].item() for i in range(6))
            six = row[7:13]

            gx_cell = (x - pc_range[0]) / eff_vx - 0.5
            gy_cell = (y - pc_range[1]) / eff_vy - 0.5
            gz_cell = (z - pc_range[2]) / eff_vz - 0.5
            target_cell = torch.tensor([gx_cell, gy_cell, gz_cell], device=device)

            # heatmap: graded credit to every active voxel in this sample by 3D cell-distance
            # (exact generalization of heatmap_targets.draw_gaussian's dense-array write to
            # an arbitrary point set -- same gaussian_radius/sigma formula, CornerNet/
            # CenterPoint standard).
            l_cells, w_cells = l / eff_vx, w / eff_vy
            radius = gaussian_radius(w_cells, l_cells, min_overlap)
            sigma = (2 * radius + 1) / 6.0
            dist2 = ((sample_xyz_idx - target_cell) ** 2).sum(dim=1)
            gauss = torch.exp(-dist2 / (2 * sigma * sigma + 1e-9))
            gauss[gauss < 1e-4] = 0
            heatmap[sample_idx, 0] = torch.maximum(heatmap[sample_idx, 0], gauss)

            # positive cell: nearest active voxel to the rounded target cell, within search_radius
            rounded = torch.tensor([round(gx_cell), round(gy_cell), round(gz_cell)], device=device, dtype=torch.float32)
            d2_to_rounded = ((sample_xyz_idx - rounded) ** 2).sum(dim=1)
            nearest_local = torch.argmin(d2_to_rounded)
            if d2_to_rounded[nearest_local].item() > search_radius ** 2:
                continue  # no voxel close enough to regress from -- heatmap credit above still applies
            pos_row = sample_idx[nearest_local]

            reg_mask[pos_row] = True
            heatmap[pos_row, 0] = 1.0  # force exact 1.0 at the assigned voxel -- gaussian_focal_loss's
            # pos_mask = (target == 1) needs this exactly; the distance-based gauss above almost never
            # lands on exactly 1.0 (the true GT center rarely coincides with an active voxel), so without
            # this the assigned positive was never recognized as positive at all (see incident notes).
            offset_t[pos_row] = target_cell - sample_xyz_idx[nearest_local]
            dim_t[pos_row] = torch.tensor([math.log(l), math.log(w), math.log(h)], device=device)
            rot_t[pos_row] = six

            if points is not None:
                R = rotation3d.sixd_to_matrix_torch(six).cpu().numpy()
                n_in_box = int(_point_in_obb(points[:, :3].cpu().numpy(), row[:3].cpu().numpy(),
                                              row[3:6].cpu().numpy(), R).sum())
                density_t[pos_row] = _density_class(n_in_box)

    return {"heatmap": heatmap, "reg_mask": reg_mask, "offset": offset_t,
            "dim": dim_t, "rot": rot_t, "density": density_t}


def sparse_center_loss(pred: dict, target: dict, reg_weight: float = 1.0,
                        focal_alpha: float = 2.0, focal_beta: float = 4.0,
                        use_density: bool = True):
    """Same math as center_loss.center_voxelnet_loss, simplified: everything here is
    already a flat (N,*) tensor (no (B,C,H,W)<->(B,H,W,C) permute dance needed --
    that permute only existed to reconcile conv-channel-first layout with the dense
    grid, which doesn't apply to a sparse per-voxel tensor)."""
    from center_loss import gaussian_focal_loss  # reuse verbatim -- shape-agnostic (elementwise + .sum())

    hm_loss = gaussian_focal_loss(pred["heatmap"], target["heatmap"], alpha=focal_alpha, beta=focal_beta)

    reg_mask = target["reg_mask"]
    n_pos = reg_mask.float().sum().clamp_min(1)
    w = reg_mask.unsqueeze(-1).float()  # (N,1)

    offset_loss = (F.l1_loss(pred["offset"], target["offset"], reduction="none") * w).sum() / n_pos
    dim_loss = (F.l1_loss(pred["dim"], target["dim"], reduction="none") * w).sum() / n_pos
    rot_loss = (F.l1_loss(pred["rot"], target["rot"], reduction="none") * w).sum() / n_pos
    reg_loss = offset_loss + dim_loss + rot_loss

    stats = {"hm_loss": hm_loss.item(), "reg_loss": reg_loss.item(),
              "offset_loss": offset_loss.item(), "dim_loss": dim_loss.item(),
              "rot_loss": rot_loss.item(), "n_pos": int(reg_mask.sum().item())}

    total = hm_loss + reg_weight * reg_loss
    if use_density:
        ce = F.cross_entropy(pred["density"], target["density"], reduction="none")  # (N,)
        density_loss = (ce * reg_mask.float()).sum() / n_pos
        total = total + config.DENSITY_AUX_WEIGHT * density_loss
        stats["density_loss"] = density_loss.item()

    return total, stats


def find_local_peaks_sparse(scores: torch.Tensor, coords: torch.Tensor, index_grid: torch.Tensor,
                             grid_size, score_thresh: float) -> torch.Tensor:
    """Sparse analog of decode.py's dense 3x3 max-pool peak test (and identical in
    spirit to this project's own 3d-point-cloud sibling, models/decode.py): a voxel is
    a peak if its score is >= all 26 face/edge/corner neighbors in the active set, using
    index_grid for O(1) neighbor lookup instead of the dense array's implicit adjacency."""
    D, H, W = grid_size
    n = coords.shape[0]
    is_peak = torch.ones(n, dtype=torch.bool, device=scores.device)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                nb = coords.clone()
                nb[:, 1] += dz
                nb[:, 2] += dy
                nb[:, 3] += dx
                valid = (nb[:, 1] >= 0) & (nb[:, 1] < D) & (nb[:, 2] >= 0) & (nb[:, 2] < H) & (nb[:, 3] >= 0) & (nb[:, 3] < W)
                nb_idx = torch.full((n,), -1, dtype=torch.long, device=scores.device)
                if valid.any():
                    v = nb[valid]
                    nb_idx[valid] = index_grid[v[:, 0], v[:, 1], v[:, 2], v[:, 3]]
                has_nb = nb_idx >= 0
                nb_scores = torch.zeros_like(scores)
                nb_scores[has_nb] = scores[nb_idx[has_nb]]
                is_peak &= scores >= nb_scores
    return is_peak & (scores > score_thresh)


def decode_sparse_center_boxes(pred: dict, coords: torch.Tensor, stride: int, batch_size: int,
                                grid_size, score_thresh: float = 0.3):
    """Sparse analog of heatmap_targets.decode_center_boxes. pred: dict from
    SparseCenterHead.forward (raw logits). coords/grid_size: this batch's backbone-output
    active set (grid_size in the same [D,H,W] convention used to build index_grid).
    Returns list[B] of list-of-dict {score,x,y,z,l,w,h,R} (world space; no BEV NMS here --
    unlike the dense anchor/heatmap head, there's no dense grid to run shapely NMS
    candidates from cheaply, and local-max peak-picking on the active set already acts as
    NMS the same way this project's 3d-point-cloud sibling relies on -- no separate NMS
    step there either)."""
    device = coords.device
    scores = torch.sigmoid(pred["heatmap"].squeeze(-1))  # (N,)
    index_grid = build_index_grid(coords, batch_size, grid_size, device=device)
    peak_mask = find_local_peaks_sparse(scores, coords, index_grid, grid_size, score_thresh)

    centers_world = _voxel_centers_world(coords, config.SPARSE_POINT_CLOUD_RANGE, config.SPARSE_VOXEL_SIZE, stride)
    eff = torch.tensor([config.SPARSE_VOXEL_SIZE[0] * stride, config.SPARSE_VOXEL_SIZE[1] * stride, config.SPARSE_VOXEL_SIZE[2] * stride], device=device)
    offset_world = pred["offset"][:, [0, 1, 2]] * eff  # (N,3) cell-fraction -> meters, [dx,dy,dz]
    xyz = centers_world + offset_world
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
                "x": xyz[i, 0].item(), "y": xyz[i, 1].item(), "z": xyz[i, 2].item(),
                "l": dims[i, 0].item(), "w": dims[i, 1].item(), "h": dims[i, 2].item(),
                "R": R,
            })
        results.append(boxes)
    return results
