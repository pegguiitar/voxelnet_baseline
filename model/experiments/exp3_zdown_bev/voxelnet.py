"""exp3_zdown_bev - z-ONLY-stride sparse conv, mirroring the ORIGINAL VoxelNet's
ConvMiddleLayers design philosophy (compress z via learned conv, leave x,y
completely alone for the 2D head to handle) -- but implemented as SPARSE conv
instead of dense Conv3d, so the compression only costs compute where voxels are
actually active. Replaces the old sparse_bev_zdown_voxelnet.py.

Differs from exp1/exp2 in this experiments/ folder: those use an ISOTROPIC
backbone (x,y,z all downsample together) and only collapse z into channels AFTER
the backbone; this one never downsamples x,y at the sparse-conv stage at all --
SparseConv3dDown's stride is (2,1,1) every stage (coords are [batch,z,y,x], so the
first stride value is z), so x,y stay at the full input voxel resolution the whole
way through the sparse backbone, and only z shrinks.

2026-09-02: SparseConv3dDown's candidate-coordinate search used to loop over the
full kernel window on EVERY axis regardless of stride, which for a stride==1 axis
(x,y here) meant every active voxel spawned up to k candidates per axis every
stage -- i.e. the active set "dilated" outward in x,y each stage instead of
staying put, compounding across stages (measured: 800 synthetic input voxels ->
CUDA OOM by the 4th stage). Fixed in sparse_ops.py itself (shared by all 3
experiments): for a stride==1 axis, only the center tap (koff=padding, giving
out=in exactly) is used to generate candidates, keeping output support on that
axis IDENTICAL to input support (true submanifold behavior on the non-strided
axes, matching SubMConv3d) while the actual per-position gather still uses the
full k^3 kernel window, so the conv's receptive field in x,y is unaffected.

Pipeline: VFE -> N z-only-stride SparseConv3dDown+residual stages (D: 22->11->6->3->2
at STAGE_CHANNELS' default depth) -> SlotFormer (on the z-compressed sparse output,
x,y still at full resolution here) -> scatter to dense (B,C,D_out,H,W) -> reshape z
into channels (sparse_ops.scatter_to_bev, same helper exp1/exp2 use) -> 1x1 adapter
conv (if needed) -> RPNCenterHead (model.py, unchanged).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # -> voxelnet_baseline/model

import torch
import torch.nn as nn

import config
from model import StackedVFE, RPNCenterHead
from sparse_ops import build_index_grid, SparseConv3dDown, SparseBasicBlock, scatter_to_bev
from slotformer import SlotFormerBackbone


class _ZDownStage(nn.Module):
    """z-only-stride down-conv + residual refinement. x,y (coords columns 2,3 in this
    repo's [batch,z,y,x] convention) are left untouched (stride=1)."""

    def __init__(self, in_channels, out_channels, num_blocks, kernel):
        super().__init__()
        self.down = SparseConv3dDown(in_channels, out_channels, kernel_size=kernel,
                                      stride=(2, 1, 1), padding=(kernel // 2,) * 3)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList([SparseBasicBlock(out_channels) for _ in range(num_blocks)])

    def forward(self, features, coords, index_grid, grid_size, batch_size):
        x, coords, index_grid, grid_size = self.down(features, coords, index_grid, grid_size, batch_size)
        x = self.relu(self.bn(x))
        for block in self.blocks:
            x, coords, index_grid = block(x, coords, index_grid, grid_size)
        return x, coords, index_grid, grid_size


class SparseZDownBEVBackbone(nn.Module):
    """z-only-stride sparse backbone that outputs a dense BEV feature map DIRECTLY --
    the scatter-to-dense + channel-merge (z into channels) happens inside the
    backbone itself, so callers get a ready-to-use (B, out_channels*D_out, H, W)
    tensor instead of having to scatter/reshape sparse tensors themselves."""

    def __init__(self, in_channels, stage_channels, num_blocks_per_stage, kernel,
                 slotformer_enabled, slotformer_win_size, slotformer_num_cycles, slotformer_num_heads):
        super().__init__()
        stage_channels = list(stage_channels)
        in_ch = in_channels
        stages = []
        for c_out in stage_channels:
            stages.append(_ZDownStage(in_ch, c_out, num_blocks_per_stage, kernel))
            in_ch = c_out
        self.stages = nn.ModuleList(stages)
        self.out_channels = stage_channels[-1]
        self._kernel = kernel

        self.use_slotformer = slotformer_enabled
        if self.use_slotformer:
            self.slotformer = SlotFormerBackbone(
                self.out_channels, slotformer_win_size, slotformer_num_cycles, slotformer_num_heads,
            )

    def output_grid_size(self, grid_size):
        """D after the z-only-stride stages -- x,y (H,W) never change (stride=1
        there every stage), only D shrinks. Deterministic from config alone
        (kernel/stride=2/padding=kernel//2 on the z axis each stage)."""
        D, H, W = grid_size
        pad = self._kernel // 2
        for _ in self.stages:
            D = (D + 2 * pad - self._kernel) // 2 + 1
        return (D, H, W)

    def forward(self, voxelwise, coords, index_grid, grid_size, batch_size):
        x, c, ig, gs = voxelwise, coords, index_grid, grid_size
        for stage in self.stages:
            x, c, ig, gs = stage(x, c, ig, gs, batch_size)

        if self.use_slotformer:
            x = self.slotformer(x, c)

        D, H, W = gs
        return scatter_to_bev(x, c, gs, batch_size, self.out_channels)


class SparseBEVZDownVoxelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.vfe = StackedVFE()

        Wp, Hp, Dp = config.SPARSE_BEV_GRID_SIZE
        self.input_grid_size = (Dp, Hp, Wp)

        self.backbone = SparseZDownBEVBackbone(
            in_channels=128,  # StackedVFE's fixed output width
            stage_channels=config.SPARSE_BEV_ZDOWN_STAGE_CHANNELS,
            num_blocks_per_stage=config.SPARSE_BEV_ZDOWN_NUM_BLOCKS_PER_STAGE,
            kernel=config.SPARSE_BEV_ZDOWN_DOWNSAMPLE_KERNEL,
            slotformer_enabled=config.SPARSE_BEV_ZDOWN_SLOTFORMER_ENABLED,
            slotformer_win_size=config.SPARSE_BEV_ZDOWN_SLOTFORMER_WIN_SIZE,
            slotformer_num_cycles=config.SPARSE_BEV_ZDOWN_SLOTFORMER_NUM_CYCLES,
            slotformer_num_heads=config.SPARSE_BEV_ZDOWN_SLOTFORMER_NUM_HEADS,
        )

        self.out_grid_size = self.backbone.output_grid_size(self.input_grid_size)  # (D,H,W) -- x,y unchanged
        self.D_out = self.out_grid_size[0]

        bev_channels = self.backbone.out_channels * self.D_out
        self._project = bev_channels != config.RPN_IN_CHANNELS
        if self._project:
            self.bev_project = nn.Conv2d(bev_channels, config.RPN_IN_CHANNELS, kernel_size=1)
            self.bev_bn = nn.BatchNorm2d(config.RPN_IN_CHANNELS)
            self.bev_relu = nn.ReLU(inplace=True)

        self.head = RPNCenterHead()  # unchanged dense pipeline head (model.py)

        sx, sy, _ = config.SPARSE_BEV_VOXEL_SIZE
        # x,y (Hp,Wp) are untouched by this backbone -- head_grid_size comes straight from
        # RPNBackbone halving (Hp,Wp) itself (ceil, not floor -- see exp1's comment for why
        # in//2 is wrong for odd in).
        self.head_grid_size = ((Wp - 1) // 2 + 1, (Hp - 1) // 2 + 1)  # (W'', H'')
        self.head_stride = (sx * (Wp / self.head_grid_size[0]), sy * (Hp / self.head_grid_size[1]))
        self.pc_range = config.SPARSE_BEV_POINT_CLOUD_RANGE

    def forward(self, voxel_features: torch.Tensor, num_points: torch.Tensor, coords: torch.Tensor):
        voxelwise = self.vfe(voxel_features, num_points)
        batch_size = int(coords[:, 0].max().item()) + 1 if len(coords) else 1
        index_grid = build_index_grid(coords, batch_size, self.input_grid_size, device=voxelwise.device)

        feat2d = self.backbone(voxelwise, coords, index_grid, self.input_grid_size, batch_size)
        if self._project:
            feat2d = self.bev_relu(self.bev_bn(self.bev_project(feat2d)))

        return self.head(feat2d)
