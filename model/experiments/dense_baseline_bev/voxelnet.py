"""dense_baseline_bev - the literal DENSE counterpart to exp3_conv_middle_bev, for a
direct dense-vs-sparse comparison. VFE -> dense scatter -> model.ConvMiddleLayers
(UNCHANGED, the real nn.Conv3d dense implementation, not a sparse mirror) -> reshape
z into channels -> dense_head_bev.DenseBEVCenterHeadDirect (NO RPNBackbone). Same
SPARSE_BEV_* voxelization, same SonarDiverDataset/collate_fn, same
sparse_bev_head.build_bev_targets + center_loss.center_voxelnet_loss as exp3 -- this
differs from exp3_conv_middle_bev in EXACTLY one thing: ConvMiddleLayers computed
with real nn.Conv3d over the full dense grid instead of SparseConv3d over only the
active voxels. Everything else (VFE, head, data, loss, targets, output
resolution) is byte-for-byte the same, so any speed/memory difference measured
between this and exp3 is attributable to sparse vs. dense compute alone.

2026-09-03: exp3's fully-sparse redesign (README's "Fully-sparse BEV experiments")
dropped RPNBackbone entirely (predicts straight off the z-compressed sparse
features, at the FULL input x,y resolution, no 2D backbone at all) and swapped
RPNCenterHead for sparse_head_bev.SparseBEVCenterHead. This file is updated to
match: model.RPNCenterHead can't be reused (it always runs RPNBackbone internally,
not optional), so dense_head_bev.DenseBEVCenterHeadDirect (the same head set as
plain nn.Conv2d, no backbone) replaces it here, and the old bev_project 1x1-conv
adapter (needed only to feed RPNBackbone's fixed RPN_IN_CHANNELS input) is gone --
DenseBEVCenterHeadDirect just takes ConvMiddleLayers' own output width directly.

Why not the original model/train.py (the repo's actual dense pipeline entry point)?
That script needs either a pre-built voxel cache (data_final/cache_strong/voxel) or
on-the-fly mode's converted-annotation-JSON + splits.json inputs -- neither exists in
this checkout (only the raw labeling-tool-main/dataset this branch's
sonar_diver_dataset.py already reads directly). Reusing the already-verified
exp1/exp2/exp3 data harness gets a real, working dense-vs-sparse number without
first having to build that missing data-prep infrastructure.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # -> voxelnet_baseline/model

import torch
import torch.nn as nn

import config
from model import StackedVFE, ConvMiddleLayers
from dense_head_bev import DenseBEVCenterHeadDirect


class DenseBEVBackbone(nn.Module):
    """model.ConvMiddleLayers, UNCHANGED -- scatters the VFE's per-voxel output to a
    dense (B,128,D,H,W) grid first (a real dense conv needs a real dense tensor,
    unlike the sparse experiments which only ever materialize a dense tensor once,
    at the very end), then runs the same 3 nn.Conv3d layers
    exp3_conv_middle_bev/voxelnet.py's ZDownTo2D-based z-compression mirrors, then
    reshapes z into channels. Never touches H,W (all 3 layers use stride=1/"same"
    padding there), matching exp3's own "x,y untouched" property exactly."""

    def __init__(self, in_channels=128):
        super().__init__()
        self.conv_middle = ConvMiddleLayers()
        self.in_channels = in_channels
        self.out_channels = 64  # ConvMiddleLayers' fixed output width

    @staticmethod
    def output_grid_size(grid_size):
        """Same arithmetic as exp3_conv_middle_bev's z-down stages (identical
        kernel/stride/padding triples, just 3 fixed layers here instead of 5) --
        D_out comes out identical to whatever ConvMiddleLayers' own D'->D'' is;
        H,W are unchanged (this class's docstring)."""
        D, H, W = grid_size
        D = (D + 2 * 1 - 3) // 2 + 1  # conv1: stride=2, padding=1
        D = (D + 2 * 0 - 3) // 1 + 1  # conv2: stride=1, padding=0
        D = (D + 2 * 1 - 3) // 2 + 1  # conv3: stride=2, padding=1
        return (D, H, W)

    def forward(self, voxelwise, coords, grid_size, batch_size):
        D, H, W = grid_size
        dense = voxelwise.new_zeros(batch_size, self.in_channels, D, H, W)
        if coords.shape[0] > 0:
            b, z, y, x = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
            dense[b, :, z, y, x] = voxelwise
        mid = self.conv_middle(dense)  # (B,64,D_out,H,W)
        B_, C_, D_, H_, W_ = mid.shape
        return mid.reshape(B_, C_ * D_, H_, W_)


class DenseBEVVoxelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.vfe = StackedVFE()

        # grid_size in [D,H,W] to match coords' [batch,z_idx,y_idx,x_idx] column order
        # (config.SPARSE_BEV_GRID_SIZE is (W',H',D')=(x,y,z) counts -- reversed here).
        Wp, Hp, Dp = config.SPARSE_BEV_GRID_SIZE
        self.input_grid_size = (Dp, Hp, Wp)

        self.backbone = DenseBEVBackbone(in_channels=128)

        self.out_grid_size = self.backbone.output_grid_size(self.input_grid_size)  # (D_out,H_out,W_out)
        D_out, H_out, W_out = self.out_grid_size
        assert (H_out, W_out) == (Hp, Wp)  # ConvMiddleLayers never touches H,W -- sanity check

        bev_channels = self.backbone.out_channels * D_out
        self.head = DenseBEVCenterHeadDirect(bev_channels)  # no RPNBackbone -- matches exp3's sparse head exactly

        # x,y are untouched by ConvMiddleLayers -- head runs at the FULL input x,y
        # resolution, same as exp3_conv_middle_bev's sparse head.
        sx, sy, _ = config.SPARSE_BEV_VOXEL_SIZE
        self.head_grid_size = (Wp, Hp)
        self.head_stride = (sx, sy)
        self.pc_range = config.SPARSE_BEV_POINT_CLOUD_RANGE

    def forward(self, voxel_features: torch.Tensor, num_points: torch.Tensor, coords: torch.Tensor):
        voxelwise = self.vfe(voxel_features, num_points)
        batch_size = int(coords[:, 0].max().item()) + 1 if len(coords) else 1

        feat2d = self.backbone(voxelwise, coords, self.input_grid_size, batch_size)
        return self.head(feat2d)
