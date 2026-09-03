"""exp3_conv_middle_bev - VoxelNeXt-style: z-only-stride sparse conv straight down
to D=1 (x,y never touched by the z-down stage), then backbone2d_sparse.Sparse2DBackbone
(a spconv mirror of model.RPNBackbone -- VoxelNet paper Fig.4's 3-block downsample +
deconv-concat neck) processes the resulting sparse 2D features before the head. Still
never materializes a dense (B,C,H,W) tensor anywhere, closest in spirit to VoxelNeXt
(Chen et al., CVPR 2023) among this experiments/ family, but no longer "no 2D backbone
at all" -- see 2026-09-03 note below.

2026-09-03: this is this experiment's FOURTH design. First was a hand-rolled
sparse-conv single-stage+SlotFormer variant (superseded when the whole
experiments/ series was restructured). Second was a literal sparse mirror of
model.ConvMiddleLayers, scattered to dense for RPNCenterHead (superseded because
profiling found the from-scratch backward was the real bottleneck, then again
because even the spconv-based fix still paid full dense-2D-backbone cost). Third
dropped the dense scatter entirely and predicted straight from zdown_to_sparse2d.
ZDownTo2D's output with no 2D backbone at all (exp1/exp3's sole difference was
exp1's added attention). This (fourth) design adds backbone2d_sparse.Sparse2DBackbone
after the same z-down stage -- explicitly requested as its own comparison point,
briefly built as a separate exp4_sparse_rpn_bev and then folded back into exp3
directly instead (that folder no longer exists). Sparse2DBackbone's own output lands
at ITS OWN input's block1 resolution = input/2 (same "net downsample, not a restore"
behavior as dense RPNBackbone itself), so the head here now runs at HALF the input
x,y resolution -- unlike exp1 (still full resolution) and exp2 (fully restored via
its own decoder). Matches dense_baseline_bev's head_grid_size convention exactly
(that file computes the same input/2, from RPNBackbone itself).
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
from backbone2d_sparse import Sparse2DBackbone
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
        self.zdown = ZDownTo2D(128, stage_channels, kernel_size=kernel, indice_key_prefix="exp3_zdown",
                                num_refine_blocks=config.SPARSE_FULLY_ZDOWN_NUM_REFINE_BLOCKS)

        self.backbone2d = Sparse2DBackbone(
            in_channels=self.zdown.out_channels,
            block_channels=config.SPARSE_FULLY_BLOCK_CHANNELS,
            block_layers=config.SPARSE_FULLY_BLOCK_LAYERS,
            upsample_channels=config.SPARSE_FULLY_UPSAMPLE_CHANNELS,
        )

        self.head = SparseBEVCenterHead(self.backbone2d.out_channels)

        # Sparse2DBackbone nets /2 (its own block1 resolution, same as dense RPNBackbone) --
        # the head runs at HALF the input x,y resolution here, unlike exp1 (full resolution)
        # or exp2 (fully restored via its own decoder). See module docstring.
        sx, sy, _ = config.SPARSE_BEV_VOXEL_SIZE
        Wpp, Hpp = Wp // 2, Hp // 2
        self.head_grid_size = (Wpp, Hpp)      # (W'',H'') naming convention shared with siblings
        self.head_grid_size_hw = (Hpp, Wpp)   # (H'',W'') -- matches coords' [batch,y,x] order
        self.head_stride = (sx * 2, sy * 2)
        self.pc_range = config.SPARSE_BEV_POINT_CLOUD_RANGE

    def forward(self, voxel_features: torch.Tensor, num_points: torch.Tensor, coords: torch.Tensor):
        """Returns (pred, out_coords, batch_size) -- see sparse_head_bev.py's
        build_sparse_bev_targets/decode_sparse_bev_boxes for what to do with them."""
        voxelwise = self.vfe(voxel_features, num_points)
        batch_size = int(coords[:, 0].max().item()) + 1 if len(coords) else 1

        x = spconv.SparseConvTensor(voxelwise, coords.int(), spatial_shape=list(self.input_grid_size),
                                     batch_size=batch_size)
        x2d = self.zdown(x)
        feat, out_coords = self.backbone2d(x2d)
        pred = self.head(feat)
        return pred, out_coords, batch_size
