"""dense_baseline_bev - the literal DENSE counterpart to exp3_conv_middle_bev, for a
direct dense-vs-sparse comparison. VFE -> dense scatter -> model.ConvMiddleLayers
(UNCHANGED, the real nn.Conv3d dense implementation, not a sparse mirror) -> reshape
z into channels -> RPNCenterHead. Same SPARSE_BEV_* voxelization, same
SonarDiverDataset/collate_fn, same sparse_bev_head.build_bev_targets +
center_loss.center_voxelnet_loss, same RPNCenterHead as exp1/exp2/exp3 -- this
differs from exp3_conv_middle_bev in EXACTLY one thing: ConvMiddleLayers computed
with real nn.Conv3d over the full dense grid instead of SparseConv3dDown over only
the active voxels. Everything else (VFE, head, data, loss, targets, output shape --
same kernel/stride/padding arithmetic means D_out/H_out/W_out come out identical to
exp3's) is byte-for-byte the same, so any speed/memory difference measured between
this and exp3 is attributable to sparse vs. dense compute alone.

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
from model import StackedVFE, RPNCenterHead, ConvMiddleLayers


class DenseBEVBackbone(nn.Module):
    """model.ConvMiddleLayers, UNCHANGED -- scatters the VFE's per-voxel output to a
    dense (B,128,D,H,W) grid first (a real dense conv needs a real dense tensor,
    unlike the sparse experiments which only ever materialize a dense tensor once,
    at the very end), then runs the same 3 nn.Conv3d layers exp3_conv_middle_bev's
    ConvMiddleBEVBackbone mirrors, then reshapes z into channels."""

    def __init__(self, in_channels=128):
        super().__init__()
        self.conv_middle = ConvMiddleLayers()
        self.in_channels = in_channels
        self.out_channels = 64  # ConvMiddleLayers' fixed output width

    @staticmethod
    def output_grid_size(grid_size):
        """Same arithmetic as ConvMiddleBEVBackbone.output_grid_size (identical
        kernel/stride/padding triples) -- D_out/H_out/W_out come out identical to
        exp3_conv_middle_bev's, by construction."""
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

        bev_channels = self.backbone.out_channels * D_out
        self._project = bev_channels != config.RPN_IN_CHANNELS
        if self._project:
            self.bev_project = nn.Conv2d(bev_channels, config.RPN_IN_CHANNELS, kernel_size=1)
            self.bev_bn = nn.BatchNorm2d(config.RPN_IN_CHANNELS)
            self.bev_relu = nn.ReLU(inplace=True)

        self.head = RPNCenterHead()  # unchanged dense pipeline head (model.py)

        # RPNBackbone's block1 (RPNBlock's first conv: kernel=3,stride=2,padding=1) sets the
        # final output size -- out=(in+2*1-3)//2+1=(in-1)//2+1 (== ceil(in/2), NOT in//2 --
        # those two only agree for EVEN in).
        sx, sy, _ = config.SPARSE_BEV_VOXEL_SIZE
        self.head_grid_size = ((W_out - 1) // 2 + 1, (H_out - 1) // 2 + 1)  # (W'', H'')
        self.head_stride = (sx * (W_out / self.head_grid_size[0]),
                             sy * (H_out / self.head_grid_size[1]))  # meters/cell at head resolution
        self.pc_range = config.SPARSE_BEV_POINT_CLOUD_RANGE

    def forward(self, voxel_features: torch.Tensor, num_points: torch.Tensor, coords: torch.Tensor):
        voxelwise = self.vfe(voxel_features, num_points)
        batch_size = int(coords[:, 0].max().item()) + 1 if len(coords) else 1

        feat2d = self.backbone(voxelwise, coords, self.input_grid_size, batch_size)
        if self._project:
            feat2d = self.bev_relu(self.bev_bn(self.bev_project(feat2d)))

        return self.head(feat2d)
