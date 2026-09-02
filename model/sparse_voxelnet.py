"""sparse_voxelnet.py - VoxelNet's VFE + a genuinely-3D sparse backbone (never
collapses z into channels / never reshapes to a dense BEV feature map) +
SparseCenterHead. The "backbone-swap experiment" alternative to model.VoxelNet's
dense path (VFE -> dense scatter -> ConvMiddleLayers[z-collapse] -> RPNBackbone[2D]).

Reuses model.StackedVFE verbatim (VFE is not what's being experimented on here).
Everything downstream of the backbone (sparse_center_head.py) keeps RPNCenterHead's
baseline head design, dimension-transformed from dense-2D-conv to sparse-per-voxel
(see its module docstring).

`sparse` branch: backbone is now backbone3d_down_slot_up.SparseDownSlotUpBackbone
(4-stage downsample -> SlotFormer at the bottleneck -> 4-stage upsample, fully
restoring resolution -- see that file's docstring and config.py's
SPARSE_DOWN_SLOT_UP_* comments) instead of the flat encoder-only
backbone3d.Sparse3DBackbone + external SlotFormer the earlier 6L experiment used.
SlotFormer now lives INSIDE the backbone, so there's no separate slot_backbone here.

Coordinate convention: coords stay in voxelize.py's native (N,4) [batch,z_idx,y_idx,x_idx]
order end to end -- sparse_ops.py's ops are axis-order-agnostic (they just need
grid_size passed in the same column order as coords), so no permutation is needed as
long as every grid_size here is (D,H,W) to match.
"""

import torch
import torch.nn as nn

import config
from model import StackedVFE
from backbone3d_down_slot_up import SparseDownSlotUpBackbone
from sparse_ops import build_index_grid
from sparse_center_head import SparseCenterHead


class SparseVoxelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.vfe = StackedVFE()
        self.backbone = SparseDownSlotUpBackbone(
            in_channels=128,  # StackedVFE's fixed output width
            stage_channels=config.SPARSE_DOWN_SLOT_UP_STAGE_CHANNELS,
            num_blocks_per_stage=config.SPARSE_DOWN_SLOT_UP_NUM_BLOCKS_PER_STAGE,
            down_kernel=config.SPARSE_DOWN_SLOT_UP_DOWNSAMPLE_KERNEL,
            down_stride=config.SPARSE_DOWN_SLOT_UP_DOWNSAMPLE_STRIDE,
            upsample_stages=config.SPARSE_DOWN_SLOT_UP_UPSAMPLE_STAGES,
            decoder_blocks_per_stage=config.SPARSE_DOWN_SLOT_UP_DECODER_BLOCKS_PER_STAGE,
            slot_win_size=config.SPARSE_DOWN_SLOT_UP_SLOTFORMER_WIN_SIZE,
            slot_num_cycles=config.SPARSE_DOWN_SLOT_UP_SLOTFORMER_NUM_CYCLES,
            slot_num_heads=config.SPARSE_DOWN_SLOT_UP_SLOTFORMER_NUM_HEADS,
        )
        self.head = SparseCenterHead(self.backbone.out_channels)
        # grid_size in [D,H,W] to match coords' [batch,z_idx,y_idx,x_idx] column order
        # (config.SPARSE_GRID_SIZE is (W',H',D')=(x,y,z) counts -- reversed here). Uses
        # SPARSE_GRID_SIZE (derived from SPARSE_POINT_CLOUD_RANGE/SPARSE_VOXEL_SIZE), NOT
        # the dense-pipeline's GRID_SIZE -- see config.py's comment for why they differ.
        Wp, Hp, Dp = config.SPARSE_GRID_SIZE
        self.input_grid_size = (Dp, Hp, Wp)
        self.stride = self.backbone.total_stride

    def forward(self, voxel_features: torch.Tensor, num_points: torch.Tensor, coords: torch.Tensor):
        """voxel_features: (K_total,T,7), num_points: (K_total,), coords: (K_total,4)
        [batch_idx,z_idx,y_idx,x_idx] -- same inputs collate_fn already produces for the
        dense path (dataset.py/voxelize.py), no dataset changes needed.

        Returns (pred, out_coords, out_grid_size): pred is SparseCenterHead's dict of
        (N,*) raw predictions; out_coords/out_grid_size describe the backbone's
        downsampled active set (needed by build_sparse_targets/decode_sparse_center_boxes,
        which must operate on the SAME active set the predictions came from)."""
        voxelwise = self.vfe(voxel_features, num_points)  # (K_total,128)
        batch_size = int(coords[:, 0].max().item()) + 1 if len(coords) else 1
        index_grid = build_index_grid(coords, batch_size, self.input_grid_size, device=voxelwise.device)

        bb_feat, bb_coords, _, bb_grid_size = self.backbone(
            voxelwise, coords, index_grid, self.input_grid_size, batch_size
        )
        pred = self.head(bb_feat)
        return pred, bb_coords, bb_grid_size
