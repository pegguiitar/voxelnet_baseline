"""exp3_conv_middle_bev - Direct sparse-conv mirror of model.ConvMiddleLayers, the
ONLY part of the dense pipeline changed here. VFE and RPNBackbone/RPNCenterHead are
the exact same code the dense baseline uses; there's no SlotFormer and no extra
backbone stages, unlike exp1/exp2 in this experiments/ folder (which change the
whole 3D backbone's structure). This isolates the effect of doing JUST the z-collapse
step sparsely instead of also changing backbone depth/adding attention.

model.ConvMiddleLayers is 3 dense Conv3d layers (channels 128->64->64->64, kernel=3,
stride/padding (2,1,1)/(1,1,1) -> (1,1,1)/(0,1,1) -> (2,1,1)/(1,1,1)) that shrink D
while leaving H,W's SIZE unchanged (stride=1, "same" padding=1 there in layers 1/3;
layer 2 is stride=1 in every axis but padding=0 on D specifically, a "valid" conv
that shrinks D by 2 without striding -- x,y still "same" there too). ConvMiddleBEVBackbone
below reuses SparseConv3dDown with the identical (kernel, stride, padding) triples for
all 3 layers -- the only change is which primitive computes each layer's output.

2026-09-02: verifying this exposed a real correctness gap in SparseConv3dDown's
stride==1 handling (fixed in sparse_ops.py, shared by all 3 experiments): the
exp3_zdown_bev design this replaces needed stride==1 axes to behave as true
submanifold (output support IDENTICAL to input, since x,y were meant to never
shrink or grow) -- but this experiment's own middle layer is ALSO stride==1 on
every axis, with padding=0 on D specifically, which is a genuinely SHRINKING
("valid") conv, not a resolution-preserving one. Naively treating every stride==1
axis as submanifold (as the first fix did) would have wrongly forced out=in there
instead of discovering the true, smaller valid-conv output domain -- silently wrong
results, not an error. Fixed by only applying the submanifold restriction when
padding == kernel_size // 2 ("same" mode) -- true for x,y in every layer here and
for D in layers 1/3, false for D in layer 2, which correctly falls back to the full
kernel-offset search there.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # -> voxelnet_baseline/model

import torch
import torch.nn as nn

import config
from model import StackedVFE, RPNCenterHead
from sparse_ops import build_index_grid, SparseConv3dDown, scatter_to_bev


class ConvMiddleBEVBackbone(nn.Module):
    """Sparse mirror of model.ConvMiddleLayers -- same 3-layer shape, same channel
    widths, same per-layer (kernel, stride, padding), computed with SparseConv3dDown
    instead of nn.Conv3d. Outputs a dense BEV feature map directly (scatter-to-dense
    + z-into-channels merge happens here, same as exp1/exp2's backbones)."""

    def __init__(self, in_channels=128, mid_channels=64):
        super().__init__()
        self.conv1 = SparseConv3dDown(in_channels, mid_channels, kernel_size=3, stride=(2, 1, 1), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm1d(mid_channels)
        self.conv2 = SparseConv3dDown(mid_channels, mid_channels, kernel_size=3, stride=(1, 1, 1), padding=(0, 1, 1))
        self.bn2 = nn.BatchNorm1d(mid_channels)
        self.conv3 = SparseConv3dDown(mid_channels, mid_channels, kernel_size=3, stride=(2, 1, 1), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm1d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.out_channels = mid_channels

    @staticmethod
    def output_grid_size(grid_size):
        """Same arithmetic as model.ConvMiddleLayers.forward's docstring
        (D'->D''), generalized to whatever H,W this experiment's SPARSE_BEV_GRID_SIZE
        uses -- deterministic from config alone."""
        gs = SparseConv3dDown.output_grid_size(grid_size, 3, (2, 1, 1), (1, 1, 1))
        gs = SparseConv3dDown.output_grid_size(gs, 3, (1, 1, 1), (0, 1, 1))
        gs = SparseConv3dDown.output_grid_size(gs, 3, (2, 1, 1), (1, 1, 1))
        return gs

    def forward(self, features, coords, index_grid, grid_size, batch_size):
        x, c, ig, gs = self.conv1(features, coords, index_grid, grid_size, batch_size)
        x = self.relu(self.bn1(x))
        x, c, ig, gs = self.conv2(x, c, ig, gs, batch_size)
        x = self.relu(self.bn2(x))
        x, c, ig, gs = self.conv3(x, c, ig, gs, batch_size)
        x = self.relu(self.bn3(x))
        return scatter_to_bev(x, c, gs, batch_size, self.out_channels)


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
        index_grid = build_index_grid(coords, batch_size, self.input_grid_size, device=voxelwise.device)

        feat2d = self.backbone(voxelwise, coords, index_grid, self.input_grid_size, batch_size)
        if self._project:
            feat2d = self.bev_relu(self.bev_bn(self.bev_project(feat2d)))

        return self.head(feat2d)
