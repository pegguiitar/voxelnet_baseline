"""exp3_conv_middle_bev - Direct sparse-conv mirror of model.ConvMiddleLayers, now
built on spconv (traveller59/spconv2, installed as spconv-cu126==2.3.8 in this venv)
instead of this repo's own from-scratch sparse_ops.py primitives.

2026-09-03: profiling (experiments/profile_pipeline.py) found the from-scratch
SparseConv3dDown's BACKWARD pass -- not forward, not the shared 2D head -- ate 80%
of total step time here, ~23x its own forward cost (a typical dense conv's backward
is roughly 2x its forward). That's specific to how sparse_ops.py computes gradients
through its hand-rolled gather+einsum+advanced-indexing implementation (plain
autograd deriving through those ops, no purpose-built backward kernel) -- not
something fixable by changing backbone depth or adding/removing SlotFormer, since
even this experiment's minimal 3-layer backbone showed the same bottleneck. spconv
ships real CUDA kernels with a proper backward for both submanifold (SubMConv3d) and
regular/strided (SparseConv3d) sparse convolution -- exactly the two conv types
sparse_ops.py was hand-rolling as SubMConv3d/SparseConv3dDown.

This also simplifies the code: spconv's SparseConvTensor manages its own internal
index/rulebook bookkeeping, so build_index_grid and the manual candidate-coordinate
search sparse_ops.py needed are gone entirely -- SparseConvTensor.dense() directly
replaces sparse_ops.scatter_to_bev's scatter step too (still followed by the same
z-into-channels reshape).

Still the exact same 3-layer shape as model.ConvMiddleLayers (channels
128->64->64->64, kernel=3, stride/padding (2,1,1)/(1,1,1) -> (1,1,1)/(0,1,1) ->
(2,1,1)/(1,1,1)), verified to produce the identical output spatial shape as before
(D: 22->11->9->5 for this experiment's grid) -- only the compute backend changed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # -> voxelnet_baseline/model

import torch
import torch.nn as nn
import spconv.pytorch as spconv

import config
from model import StackedVFE, RPNCenterHead


class ConvMiddleBEVBackbone(nn.Module):
    """spconv mirror of model.ConvMiddleLayers -- same 3-layer shape, same channel
    widths, same per-layer (kernel, stride, padding), computed with spconv.SparseConv3d
    instead of nn.Conv3d (dense) or the hand-rolled SparseConv3dDown (slow backward,
    see module docstring). Outputs a dense BEV feature map directly."""

    def __init__(self, in_channels=128, mid_channels=64):
        super().__init__()
        self.conv1 = spconv.SparseConv3d(in_channels, mid_channels, kernel_size=3,
                                          stride=(2, 1, 1), padding=(1, 1, 1), bias=True)
        self.bn1 = nn.BatchNorm1d(mid_channels)
        self.conv2 = spconv.SparseConv3d(mid_channels, mid_channels, kernel_size=3,
                                          stride=(1, 1, 1), padding=(0, 1, 1), bias=True)
        self.bn2 = nn.BatchNorm1d(mid_channels)
        self.conv3 = spconv.SparseConv3d(mid_channels, mid_channels, kernel_size=3,
                                          stride=(2, 1, 1), padding=(1, 1, 1), bias=True)
        self.bn3 = nn.BatchNorm1d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.out_channels = mid_channels

    @staticmethod
    def output_grid_size(grid_size):
        """Pure conv-arithmetic (kernel=3 throughout; same stride/padding triples as
        __init__ above) -- identical formula to the sparse_ops.py version, still
        correct regardless of which backend actually computes the convolution."""
        def out(g, pad, stride):
            return (g + 2 * pad - 3) // stride + 1
        D, H, W = grid_size
        D, H, W = out(D, 1, 2), out(H, 1, 1), out(W, 1, 1)  # conv1
        D, H, W = out(D, 0, 1), out(H, 1, 1), out(W, 1, 1)  # conv2
        D, H, W = out(D, 1, 2), out(H, 1, 1), out(W, 1, 1)  # conv3
        return (D, H, W)

    def forward(self, voxelwise: torch.Tensor, coords: torch.Tensor, grid_size, batch_size: int):
        x = spconv.SparseConvTensor(voxelwise, coords.int(), spatial_shape=list(grid_size), batch_size=batch_size)
        x = self.conv1(x)
        x = x.replace_feature(self.relu(self.bn1(x.features)))
        x = self.conv2(x)
        x = x.replace_feature(self.relu(self.bn2(x.features)))
        x = self.conv3(x)
        x = x.replace_feature(self.relu(self.bn3(x.features)))
        dense = x.dense()  # (B, C, D_out, H_out, W_out) -- zeros where nothing was active
        B_, C_, D_, H_, W_ = dense.shape
        return dense.reshape(B_, C_ * D_, H_, W_)


class SparseBEVConvMiddleVoxelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.vfe = StackedVFE()

        # grid_size in [D,H,W] to match coords' [batch,z_idx,y_idx,x_idx] column order
        # (config.SPARSE_BEV_GRID_SIZE is (W',H',D')=(x,y,z) counts -- reversed here).
        Wp, Hp, Dp = config.SPARSE_BEV_GRID_SIZE
        self.input_grid_size = (Dp, Hp, Wp)

        self.backbone = ConvMiddleBEVBackbone(in_channels=128, mid_channels=config.SPARSE_BEV_CONVMID_CHANNELS)

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
