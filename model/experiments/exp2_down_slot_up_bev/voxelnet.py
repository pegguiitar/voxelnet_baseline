"""exp2_down_slot_up_bev - U-Net-style: backbone3d_down_slot_up.SparseDownSlotUpBackbone
(UNCHANGED -- N-stage isotropic down [x,y,z together], SlotFormer at the bottleneck
[3-axis, built into that backbone], M-stage up restoring x,y AND z together via
paired inverse convs + skip concat) fully restores resolution, THEN z is compressed
back down to D=1 (zdown_to_sparse2d.ZDownTo2D, x,y untouched) before the head. The
only one of this experiments/ family whose backbone actually changes x,y resolution
along the way (down then back up) instead of leaving it untouched throughout.

2026-09-03: this experiment's FOURTH design. The original intent was "encoder
downsamples x,y,z together, decoder upsamples x,y only" -- but spconv.
SparseInverseConv3d can only invert a conv call that used the EXACT SAME
(kernel,stride,padding) it's paired with via indice_key, so a decoder that
restores only 2 of an isotropic encoder's 3 downsampled axes isn't directly
expressible (a THIRD design tried working around this by stacking an independent
2D down+up backbone after an early z-compression, but that backbone's own output
lands at ITS OWN input/2 -- not upsampled back toward the original resolution at
all, just yet another downsample on top of the encoder's -- wrong). This design
gets the intended NET EFFECT (x,y ends up genuinely restored to ~input resolution,
z ends up compressed) by restoring ALL 3 axes via the existing, already-correct
paired-inverse-plus-skip-concat U-Net (SparseDownSlotUpBackbone, unchanged), then
independently re-compressing z right afterward -- z's own round trip (down then
back up then back down) is a no-op on z's OWN resolution, but x,y genuinely
benefits from the U-Net's coarse-context features the whole way through.

No extra SlotFormer after the z-compression here (unlike exp1) -- the bottleneck
SlotFormer already built into SparseDownSlotUpBackbone is this experiment's only
attention step, kept as-is rather than doubled up.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # -> voxelnet_baseline/model

import torch
import torch.nn as nn
import spconv.pytorch as spconv

import config
from model import StackedVFE
from backbone3d_down_slot_up import SparseDownSlotUpBackbone
from zdown_to_sparse2d import ZDownTo2D
from sparse_head_bev import SparseBEVCenterHead


class SparseBEVDownSlotUpVoxelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.vfe = StackedVFE()

        # grid_size in [D,H,W] to match coords' [batch,z_idx,y_idx,x_idx] column order
        # (config.SPARSE_BEV_GRID_SIZE is (W',H',D')=(x,y,z) counts -- reversed here).
        Wp, Hp, Dp = config.SPARSE_BEV_GRID_SIZE
        self.input_grid_size = (Dp, Hp, Wp)

        self.encoder = SparseDownSlotUpBackbone(
            in_channels=128,  # StackedVFE's fixed output width
            stage_channels=config.SPARSE_BEV_STAGE_CHANNELS,
            num_blocks_per_stage=config.SPARSE_BEV_NUM_BLOCKS_PER_STAGE,
            down_kernel=config.SPARSE_BEV_DOWNSAMPLE_KERNEL,
            down_stride=config.SPARSE_BEV_DOWNSAMPLE_STRIDE,
            upsample_stages=config.SPARSE_BEV_UPSAMPLE_STAGES,  # = len(stage_channels) -> full restore
            decoder_blocks_per_stage=config.SPARSE_BEV_DECODER_BLOCKS_PER_STAGE,
            slot_win_size=config.SPARSE_BEV_SLOTFORMER_WIN_SIZE,
            slot_num_cycles=config.SPARSE_BEV_SLOTFORMER_NUM_CYCLES,
            slot_num_heads=config.SPARSE_BEV_SLOTFORMER_NUM_HEADS,
        )
        # UPSAMPLE_STAGES == len(STAGE_CHANNELS) -> the decoder fully restores resolution,
        # so the encoder's output grid_size equals the INPUT grid_size (Dp,Hp,Wp) exactly
        # -- a structural property of SparseDownSlotUpBackbone's decoder (see that class),
        # not re-derived from data.
        assert config.SPARSE_BEV_UPSAMPLE_STAGES == len(config.SPARSE_BEV_STAGE_CHANNELS)

        zdown_kernel = config.SPARSE_FULLY_ZDOWN_DOWNSAMPLE_KERNEL
        zdown_channels = list(config.SPARSE_FULLY_ZDOWN_STAGE_CHANNELS)  # sized for Dp=22 -> 1 (5 stages)
        D_out = ZDownTo2D.output_d(Dp, len(zdown_channels), zdown_kernel)
        assert D_out == 1, (
            f"SPARSE_FULLY_ZDOWN_STAGE_CHANNELS has {len(zdown_channels)} stages, which "
            f"takes D={Dp} to {D_out}, not 1 -- adjust the stage count in config.py."
        )
        self.zdown = ZDownTo2D(self.encoder.out_channels, zdown_channels, kernel_size=zdown_kernel,
                                indice_key_prefix="exp2_zdown")

        self.head = SparseBEVCenterHead(self.zdown.out_channels)

        # x,y are fully restored to (Wp,Hp) by the encoder's decoder, then untouched by
        # zdown -- head runs at the FULL input x,y resolution, same as exp1/exp3.
        sx, sy, _ = config.SPARSE_BEV_VOXEL_SIZE
        self.head_grid_size = (Wp, Hp)        # (W'',H'') naming convention shared with siblings
        self.head_grid_size_hw = (Hp, Wp)     # (H'',W'') -- matches coords' [batch,y,x] order
        self.head_stride = (sx, sy)
        self.pc_range = config.SPARSE_BEV_POINT_CLOUD_RANGE

    def forward(self, voxel_features: torch.Tensor, num_points: torch.Tensor, coords: torch.Tensor):
        voxelwise = self.vfe(voxel_features, num_points)
        batch_size = int(coords[:, 0].max().item()) + 1 if len(coords) else 1

        enc_feat, enc_coords, enc_grid_size = self.encoder(voxelwise, coords, self.input_grid_size, batch_size)
        x = spconv.SparseConvTensor(enc_feat, enc_coords.int(), spatial_shape=list(enc_grid_size),
                                     batch_size=batch_size)
        x2d = self.zdown(x)
        pred = self.head(x2d.features)
        return pred, x2d.indices, batch_size
