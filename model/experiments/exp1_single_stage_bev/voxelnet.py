"""exp1_single_stage_bev - single-stage sparse backbone (backbone3d.Sparse3DBackbone,
config.SPARSE_BACKBONE_STAGE_CHANNELS=(128,) -- one SparseConv3dDown + residual
blocks) + external SlotFormer (2 cycles = 6L), replacing the old
sparse_bev_simple_voxelnet.py.

Same "backbone outputs BEV directly" shape as exp2/exp3 in this experiments/
folder: SingleStageBEVBackbone.forward returns a dense (B,C*D,H,W) feature map
directly (scatter-to-dense + z-into-channels merge happens inside the backbone,
via sparse_ops.scatter_to_bev) instead of the outer VoxelNet class doing that.

2026-09-02: dropped the old sparse_bev_simple_voxelnet.py's use of
model.ConvMiddleLayers for the z-collapse (a real learned z-strided Conv3D
instead of a plain reshape) -- measured to NOT reduce memory/speed versus a
plain scatter+reshape (see exp3_zdown_bev's history), and standardizing all 3
experiments on the SAME plain scatter_to_bev collapse keeps this a fair,
apples-to-apples comparison of backbone STRUCTURE alone (single-stage vs.
down-slot-up vs. z-only-stride), rather than each experiment also varying in
how it collapses to BEV.

2026-09-03: backbone3d.Sparse3DBackbone (imported below) migrated to spconv --
see that module's docstring and exp3_conv_middle_bev/voxelnet.py's for why
(the from-scratch sparse_ops.py backward pass was the real bottleneck, ~80% of
total step time, not backbone depth/SlotFormer). Its (features, coords,
grid_size) tuple interface is unchanged, just no more index_grid argument
(spconv manages its own internal indices, so build_index_grid is gone too).
scatter_to_bev is still used unchanged here -- profiling found it was never
the bottleneck (its backward is a plain gather over unique indices, not the
duplicate-index scatter-add pattern that was actually slow).

Reuses config.SPARSE_BACKBONE_*/SPARSE_SLOTFORMER_* (same backbone shape/
SlotFormer depth as the sparse_voxelnet.py fully-sparse-head experiment) and
config.SPARSE_BEV_* (voxel size / grid / RPN_IN_CHANNELS) for the BEV side.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # -> voxelnet_baseline/model

import torch
import torch.nn as nn

import config
from model import StackedVFE, RPNCenterHead
from backbone3d import Sparse3DBackbone
from slotformer import SlotFormerBackbone
from sparse_ops import scatter_to_bev


def _stage_grid_sizes(grid_size, num_stages, kernel, stride):
    """[grid_size, size after stage 1, ..., size after stage num_stages] --
    deterministic from config alone (pure conv-arithmetic, independent of which
    backend actually computes the convolution)."""
    padding = kernel // 2

    def out(g):
        return (g + 2 * padding - kernel) // stride + 1

    sizes = [tuple(grid_size)]
    for _ in range(num_stages):
        sizes.append(tuple(out(g) for g in sizes[-1]))
    return sizes


class SingleStageBEVBackbone(nn.Module):
    """Sparse3DBackbone (isotropic stride, N stages -- N=1 at this experiment's
    config) + external SlotFormer at the backbone's output, then scatter-to-dense +
    z-into-channels merge. Outputs a dense BEV feature map directly."""

    def __init__(self, in_channels, stage_channels, num_blocks_per_stage, down_kernel, down_stride,
                 slotformer_enabled, slotformer_win_size, slotformer_num_cycles, slotformer_num_heads):
        super().__init__()
        self.backbone = Sparse3DBackbone(
            in_channels=in_channels, stage_channels=stage_channels,
            num_blocks_per_stage=num_blocks_per_stage, down_kernel=down_kernel, down_stride=down_stride,
        )
        self.out_channels = self.backbone.out_channels

        self.use_slotformer = slotformer_enabled
        if self.use_slotformer:
            self.slotformer = SlotFormerBackbone(self.out_channels, slotformer_win_size,
                                                  slotformer_num_cycles, slotformer_num_heads)

        self._num_stages = len(stage_channels)
        self._kernel = down_kernel
        self._stride = down_stride

    def output_grid_size(self, grid_size):
        return _stage_grid_sizes(grid_size, self._num_stages, self._kernel, self._stride)[-1]

    def forward(self, voxelwise, coords, grid_size, batch_size):
        x, c, gs = self.backbone(voxelwise, coords, grid_size, batch_size)
        if self.use_slotformer:
            x = self.slotformer(x, c)
        return scatter_to_bev(x, c, gs, batch_size, self.out_channels)


class SparseBEVSingleStageVoxelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.vfe = StackedVFE()

        # grid_size in [D,H,W] to match coords' [batch,z_idx,y_idx,x_idx] column order
        # (config.SPARSE_BEV_GRID_SIZE is (W',H',D')=(x,y,z) counts -- reversed here).
        Wp, Hp, Dp = config.SPARSE_BEV_GRID_SIZE
        self.input_grid_size = (Dp, Hp, Wp)

        self.backbone = SingleStageBEVBackbone(
            in_channels=128,  # StackedVFE's fixed output width
            stage_channels=config.SPARSE_BACKBONE_STAGE_CHANNELS,
            num_blocks_per_stage=config.SPARSE_BACKBONE_NUM_BLOCKS_PER_STAGE,
            down_kernel=config.SPARSE_BACKBONE_DOWNSAMPLE_KERNEL,
            down_stride=config.SPARSE_BACKBONE_DOWNSAMPLE_STRIDE,
            slotformer_enabled=config.SPARSE_SLOTFORMER_ENABLED,
            slotformer_win_size=config.SPARSE_SLOTFORMER_WIN_SIZE,
            slotformer_num_cycles=config.SPARSE_SLOTFORMER_NUM_CYCLES,
            slotformer_num_heads=config.SPARSE_SLOTFORMER_NUM_HEADS,
        )

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
        # those two only agree for EVEN in; this backbone's W_out is often odd).
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
