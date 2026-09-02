"""sparse_ops.py - small sparse-tensor utilities shared by the sparse experiments.

2026-09-03: this used to also hold a from-scratch sparse conv implementation
(SubMConv3d/SparseConv3dDown/SparseInverseConv3d, built entirely on core PyTorch to
avoid a spconv/torch_scatter compiled-extension dependency). Profiling
(experiments/profile_pipeline.py) found its backward pass ate ~80% of total step
time in exp3_conv_middle_bev, ~23x its own forward cost -- switching
backbone3d.py/backbone3d_down_slot_up.py to spconv (already installed in this venv)
fixed that directly (measured ~4.4x real-training speedup) and made those hand-rolled
primitives dead code, so they were removed. See backbone3d.py's module docstring for
the full story and git history for the removed code if it's ever needed again
(e.g. porting back to a spconv-free environment).

What's left here is genuinely still used: build_index_grid (sparse_center_head.py's
NMS/decoding needs a dense coordinate->row-index lookup) and scatter_to_bev (every
experiments/*_bev/voxelnet.py backbone's final step -- scatter a sparse (N,C) tensor
to dense (B,C,D,H,W), then merge z into channels). Neither was ever the bottleneck:
scatter_to_bev's backward is a plain gather over unique indices, not the
duplicate-index scatter-add pattern that was actually slow.

A "sparse tensor" here is just:
  features: (N, C) float
  coords:   (N, 4) long, columns = [batch_idx, x_idx, y_idx, z_idx]
  index_grid: (B, X, Y, Z) long, dense lookup -> row index into features, or -1
"""
import torch


def build_index_grid(coords: torch.Tensor, batch_size: int, grid_size, device=None) -> torch.Tensor:
    X, Y, Z = grid_size
    device = device or coords.device
    grid = torch.full((batch_size, X, Y, Z), -1, dtype=torch.long, device=device)
    if coords.shape[0] > 0:
        grid[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]] = torch.arange(
            coords.shape[0], device=device
        )
    return grid


def scatter_to_bev(features: torch.Tensor, coords: torch.Tensor, grid_size, batch_size: int,
                    channels: int) -> torch.Tensor:
    """Scatter a sparse (N,C) tensor to dense (B,C,D,H,W) via coords, then merge z
    into the channel dim -> (B,C*D,H,W). The one shared implementation of the
    "collapse a sparse backbone's output into a BEV feature map" step every
    experiments/*_bev/voxelnet.py backbone does at the end of its forward -- lets
    each backbone return a ready-to-use dense BEV tensor directly instead of the
    caller having to scatter/reshape sparse tensors itself."""
    D, H, W = grid_size
    dense = features.new_zeros(batch_size, channels, D, H, W)
    if coords.shape[0] > 0:
        b, z, y, x = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
        dense[b, :, z, y, x] = features
    return dense.reshape(batch_size, channels * D, H, W)
