"""exp1_single_stage_bev - same z-only-stride compression as exp3 (x,y never
touched anywhere), but with SlotFormer windowed attention (2-axis: x,y only, since
z has already been compressed away by the time this runs) applied to the resulting
sparse 2D features before the head. The "+ attention" step up from exp3's minimal
baseline -- exp2 adds a real spatial (2D) backbone on top of this same idea instead.

2026-09-03: this experiment's SECOND design. First was `backbone3d.Sparse3DBackbone`
(isotropic single-stage downsample) + external SlotFormer + dense scatter for
RPNCenterHead -- superseded for the same "still pays dense-2D-backbone cost"
reason exp2/exp3 were. This one never scatters to dense: zdown_to_sparse2d.ZDownTo2D
handles the z-compression, slotformer.SlotFormerBackbone(num_axes=2) runs directly
on the sparse 2D result (see that module's num_axes docstring -- the default
3-axis (x,y,z) cycling assumes 4-column coords and would index out of bounds on
the 3-column [batch,y,x] coords this experiment has after z-compression), and
sparse_head_bev.SparseBEVCenterHead predicts straight from that -- no
backbone2d_sparse.Sparse2DBackbone here (that's exp2's addition).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # -> voxelnet_baseline/model

import torch
import torch.nn as nn
import spconv.pytorch as spconv

import config
from model import StackedVFE
from zdown_to_sparse2d import ZDownTo2D
from slotformer import SlotFormerBackbone
from sparse_head_bev import SparseBEVCenterHead


class SparseBEVSingleStageVoxelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.vfe = StackedVFE()

        # grid_size in [D,H,W] to match coords' [batch,z_idx,y_idx,x_idx] column order
        # (config.SPARSE_BEV_GRID_SIZE is (W',H',D')=(x,y,z) counts -- reversed here).
        Wp, Hp, Dp = config.SPARSE_BEV_GRID_SIZE
        self.input_grid_size = (Dp, Hp, Wp)

        stage_channels = list(config.SPARSE_FULLY_ZDOWN_STAGE_CHANNELS)
        kernel = config.SPARSE_FULLY_ZDOWN_DOWNSAMPLE_KERNEL
        D_out = ZDownTo2D.output_d(Dp, len(stage_channels), kernel)
        assert D_out == 1, (
            f"SPARSE_FULLY_ZDOWN_STAGE_CHANNELS has {len(stage_channels)} stages, which "
            f"takes D={Dp} to {D_out}, not 1 -- adjust the stage count in config.py."
        )
        self.zdown = ZDownTo2D(128, stage_channels, kernel_size=kernel, indice_key_prefix="exp1_zdown",
                                num_refine_blocks=config.SPARSE_FULLY_ZDOWN_NUM_REFINE_BLOCKS)

        self.slotformer = SlotFormerBackbone(
            self.zdown.out_channels, config.SPARSE_FULLY_SLOTFORMER_WIN_SIZE,
            config.SPARSE_FULLY_SLOTFORMER_NUM_CYCLES, config.SPARSE_FULLY_SLOTFORMER_NUM_HEADS,
            num_axes=2,
        )

        self.head = SparseBEVCenterHead(self.zdown.out_channels)

        # x,y are untouched anywhere in this model (attention doesn't change resolution)
        # -- the head runs at the FULL input x,y resolution, same as exp3.
        sx, sy, _ = config.SPARSE_BEV_VOXEL_SIZE
        self.head_grid_size = (Wp, Hp)        # (W'',H'') naming convention shared with siblings
        self.head_grid_size_hw = (Hp, Wp)     # (H'',W'') -- matches coords' [batch,y,x] order
        self.head_stride = (sx, sy)
        self.pc_range = config.SPARSE_BEV_POINT_CLOUD_RANGE

    def forward(self, voxel_features: torch.Tensor, num_points: torch.Tensor, coords: torch.Tensor):
        voxelwise = self.vfe(voxel_features, num_points)
        batch_size = int(coords[:, 0].max().item()) + 1 if len(coords) else 1

        x = spconv.SparseConvTensor(voxelwise, coords.int(), spatial_shape=list(self.input_grid_size),
                                     batch_size=batch_size)
        x2d = self.zdown(x)
        feat = self.slotformer(x2d.features, x2d.indices)
        pred = self.head(feat)
        return pred, x2d.indices, batch_size
