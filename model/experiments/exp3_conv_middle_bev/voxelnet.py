"""exp3_conv_middle_bev - VoxelNeXt-style: z-only-stride sparse conv straight down
to D=1 (x,y never touched anywhere in this model), then predict directly from the
resulting sparse 2D features. No attention (exp1's addition), no spatial U-Net
(exp2's addition) -- the minimal member of this experiments/ family, and the one
closest in spirit to VoxelNeXt (Chen et al., CVPR 2023): stay sparse through the
head, never materialize a dense (B,C,H,W) tensor anywhere.

2026-09-03: this is this experiment's THIRD design. First was a hand-rolled
sparse-conv single-stage+SlotFormer variant (superseded when the whole
experiments/ series was restructured). Second was a literal sparse mirror of
model.ConvMiddleLayers, scattered to dense for RPNCenterHead (superseded because
profiling found the from-scratch backward was the real bottleneck, then again
because even the spconv-based fix still paid full dense-2D-backbone cost). This
one drops the dense scatter entirely: zdown_to_sparse2d.ZDownTo2D handles the
z-compression (see that module's docstring for why spconv.SparseConv3d needs the
extra x,y-support-restriction step it applies), then sparse_head_bev.SparseBEVCenterHead
predicts straight from the sparse 2D result -- no backbone2d_sparse.Sparse2DBackbone
here at all (that's exp2's job).
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
from sparse_head_bev import SparseBEVCenterHead


class SparseBEVConvMiddleVoxelNet(nn.Module):
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
        self.zdown = ZDownTo2D(128, stage_channels, kernel_size=kernel, indice_key_prefix="exp3_zdown")

        self.head = SparseBEVCenterHead(self.zdown.out_channels)

        # x,y are untouched anywhere in this model -- the head runs at the FULL input
        # x,y resolution (unlike exp1/exp2, which add attention/a 2D backbone after this
        # same z-compression step -- see their own voxelnet.py for why their head
        # resolution differs).
        sx, sy, _ = config.SPARSE_BEV_VOXEL_SIZE
        self.head_grid_size = (Wp, Hp)        # (W'',H'') naming convention shared with siblings
        self.head_grid_size_hw = (Hp, Wp)     # (H'',W'') -- matches coords' [batch,y,x] order
        self.head_stride = (sx, sy)
        self.pc_range = config.SPARSE_BEV_POINT_CLOUD_RANGE

    def forward(self, voxel_features: torch.Tensor, num_points: torch.Tensor, coords: torch.Tensor):
        """Returns (pred, out_coords, batch_size) -- see sparse_head_bev.py's
        build_sparse_bev_targets/decode_sparse_bev_boxes for what to do with them."""
        voxelwise = self.vfe(voxel_features, num_points)
        batch_size = int(coords[:, 0].max().item()) + 1 if len(coords) else 1

        x = spconv.SparseConvTensor(voxelwise, coords.int(), spatial_shape=list(self.input_grid_size),
                                     batch_size=batch_size)
        x2d = self.zdown(x)
        pred = self.head(x2d.features)
        return pred, x2d.indices, batch_size
