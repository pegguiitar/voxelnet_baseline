"""exp2_down_slot_up_bev - N-stage sparse down-sample encoder -> SlotFormer at the
bottleneck -> M-stage up-sample decoder (backbone3d_down_slot_up.SparseDownSlotUpBackbone,
unchanged), replacing the old sparse_bev_voxelnet.py.

Same "backbone outputs BEV directly" shape as exp1/exp3 in this experiments/
folder: DownSlotUpBEVBackbone.forward returns a dense (B,C*D,H,W) feature map
directly (scatter-to-dense + z-into-channels merge happens inside the backbone,
via sparse_ops.scatter_to_bev) instead of the outer VoxelNet class doing that.

Why this needs its own VOXEL_SIZE (config.SPARSE_BEV_VOXEL_SIZE, z=0.5 instead of
sparse_voxelnet.py's z=0.1): collapsing z into channels means whatever D (z-bin
count) survives to the point of collapse multiplies directly into the channel count
fed to the projection conv -- at z=0.1 the sonar range's ~11.2m z-extent gives
D~112, so collapsing that would need a 128*112=14336-channel projection input.
z=0.5 keeps D~22; the decoder restores back to the INPUT D exactly (a structural
property of SparseDownSlotUpBackbone's decoder, not re-derived from data), so the
channel count fed to the projection conv is backbone.out_channels * 22 -- large but
tractable for a single 1x1 conv. x,y stay at 0.1m (unchanged from sparse_voxelnet.py)
-- only z is coarsened, mirroring the original VoxelNet's own asymmetry (aggressive
z reduction via ConvMiddleLayers, x/y handled entirely by the 2D RPNBackbone
afterward), just achieved here by choosing VOXEL_SIZE instead of an anisotropic
conv stride (this backbone's SparseConv3dDown always uses the same stride for
x/y/z every stage).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # -> voxelnet_baseline/model

import torch
import torch.nn as nn

import config
from model import StackedVFE, RPNCenterHead
from backbone3d_down_slot_up import SparseDownSlotUpBackbone
from sparse_ops import build_index_grid, SparseConv3dDown, scatter_to_bev


def _stage_grid_sizes(grid_size, num_stages, kernel, stride):
    """[grid_size, size after stage 1, ..., size after stage num_stages] --
    deterministic from config alone (same arithmetic SparseConv3dDown.forward uses)."""
    padding = kernel // 2
    sizes = [tuple(grid_size)]
    for _ in range(num_stages):
        sizes.append(SparseConv3dDown.output_grid_size(sizes[-1], kernel, stride, padding))
    return sizes


class DownSlotUpBEVBackbone(nn.Module):
    """SparseDownSlotUpBackbone (N-stage down, SlotFormer at the bottleneck,
    M-stage up -- SlotFormer already lives inside this backbone, unlike exp1's
    external one), then scatter-to-dense + z-into-channels merge. Outputs a dense
    BEV feature map directly."""

    def __init__(self, in_channels, stage_channels, num_blocks_per_stage, down_kernel, down_stride,
                 upsample_stages, decoder_blocks_per_stage, slot_win_size, slot_num_cycles, slot_num_heads):
        super().__init__()
        self.backbone = SparseDownSlotUpBackbone(
            in_channels=in_channels, stage_channels=stage_channels, num_blocks_per_stage=num_blocks_per_stage,
            down_kernel=down_kernel, down_stride=down_stride, upsample_stages=upsample_stages,
            decoder_blocks_per_stage=decoder_blocks_per_stage,
            slot_win_size=slot_win_size, slot_num_cycles=slot_num_cycles, slot_num_heads=slot_num_heads,
        )
        self.out_channels = self.backbone.out_channels
        self._num_stages = len(stage_channels)
        self._kernel = down_kernel
        self._stride = down_stride
        self._upsample_stages = upsample_stages

    def output_grid_size(self, grid_size):
        # The decoder restores back to stage_sizes[n-m] exactly (SparseInverseConv3d
        # writes to the cached parent coords of the stage it's inverting) -- a
        # structural property of the decoder, not re-derived from data.
        sizes = _stage_grid_sizes(grid_size, self._num_stages, self._kernel, self._stride)
        return sizes[self._num_stages - self._upsample_stages]

    def forward(self, voxelwise, coords, index_grid, grid_size, batch_size):
        x, c, _, gs = self.backbone(voxelwise, coords, index_grid, grid_size, batch_size)
        return scatter_to_bev(x, c, gs, batch_size, self.out_channels)


class SparseBEVDownSlotUpVoxelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.vfe = StackedVFE()

        # grid_size in [D,H,W] to match coords' [batch,z_idx,y_idx,x_idx] column order
        # (config.SPARSE_BEV_GRID_SIZE is (W',H',D')=(x,y,z) counts -- reversed here).
        Wp, Hp, Dp = config.SPARSE_BEV_GRID_SIZE
        self.input_grid_size = (Dp, Hp, Wp)

        self.backbone = DownSlotUpBEVBackbone(
            in_channels=128,  # StackedVFE's fixed output width
            stage_channels=config.SPARSE_BEV_STAGE_CHANNELS,
            num_blocks_per_stage=config.SPARSE_BEV_NUM_BLOCKS_PER_STAGE,
            down_kernel=config.SPARSE_BEV_DOWNSAMPLE_KERNEL,
            down_stride=config.SPARSE_BEV_DOWNSAMPLE_STRIDE,
            upsample_stages=config.SPARSE_BEV_UPSAMPLE_STAGES,
            decoder_blocks_per_stage=config.SPARSE_BEV_DECODER_BLOCKS_PER_STAGE,
            slot_win_size=config.SPARSE_BEV_SLOTFORMER_WIN_SIZE,
            slot_num_cycles=config.SPARSE_BEV_SLOTFORMER_NUM_CYCLES,
            slot_num_heads=config.SPARSE_BEV_SLOTFORMER_NUM_HEADS,
        )

        self.out_grid_size = self.backbone.output_grid_size(self.input_grid_size)  # (D_out,H_out,W_out)
        D_out, H_out, W_out = self.out_grid_size

        bev_channels = self.backbone.out_channels * D_out
        self._project = bev_channels != config.RPN_IN_CHANNELS
        if self._project:
            self.bev_project = nn.Conv2d(bev_channels, config.RPN_IN_CHANNELS, kernel_size=1)
            self.bev_bn = nn.BatchNorm2d(config.RPN_IN_CHANNELS)
            self.bev_relu = nn.ReLU(inplace=True)

        self.head = RPNCenterHead()  # unchanged dense pipeline head (model.py) -- 6-tuple output

        # RPNBackbone's block1 (RPNBlock's first conv: kernel=3,stride=2,padding=1) sets the
        # final output size -- out=(in+2*1-3)//2+1=(in-1)//2+1 (== ceil(in/2), NOT in//2 --
        # those two only agree for EVEN in).
        sx, sy, _ = config.SPARSE_BEV_VOXEL_SIZE
        self.head_grid_size = ((W_out - 1) // 2 + 1, (H_out - 1) // 2 + 1)  # (W'', H'')
        self.head_stride = (sx * (W_out / self.head_grid_size[0]),
                             sy * (H_out / self.head_grid_size[1]))  # meters/cell at head resolution
        self.pc_range = config.SPARSE_BEV_POINT_CLOUD_RANGE

    def forward(self, voxel_features: torch.Tensor, num_points: torch.Tensor, coords: torch.Tensor):
        """Same collate_fn inputs as sparse_voxelnet.py/model.py's dense path.
        Returns RPNCenterHead's raw 6-tuple (heatmap, offset, z, dim, rot, density),
        each (B,C,H'',W'') -- pass straight to center_loss.center_voxelnet_loss with
        targets built by sparse_bev_head.build_bev_targets(..., self.head_grid_size,
        self.head_stride, self.pc_range)."""
        voxelwise = self.vfe(voxel_features, num_points)  # (K_total,128)
        batch_size = int(coords[:, 0].max().item()) + 1 if len(coords) else 1
        index_grid = build_index_grid(coords, batch_size, self.input_grid_size, device=voxelwise.device)

        feat2d = self.backbone(voxelwise, coords, index_grid, self.input_grid_size, batch_size)
        if self._project:
            feat2d = self.bev_relu(self.bev_bn(self.bev_project(feat2d)))

        return self.head(feat2d)
