"""exp2_down_slot_up_bev - U-Net-style: an isotropic 3D encoder (x,y,z downsampled
together, backbone3d.Sparse3DBackbone) reaches a bottleneck, z is compressed the
rest of the way to D=1 there (zdown_to_sparse2d.ZDownTo2D -- cheap since x,y is
already small at the bottleneck), SlotFormer windowed attention (2-axis, same as
exp1) runs on that bottleneck's sparse 2D features, and then a genuinely-2D U-Net
(backbone2d_sparse.Sparse2DBackbone, its own self-contained down+up structure)
brings x,y back up to a more usable resolution before the head. The only one of
this experiments/ family whose backbone actually changes x,y resolution.

2026-09-03: this experiment's THIRD design. Originally backbone3d_down_slot_up.
SparseDownSlotUpBackbone (N-stage down, SlotFormer at the bottleneck, M-stage up
restoring x,y AND z together via paired inverse convs) + dense scatter for
RPNCenterHead. The intent going into this redesign was "encoder downsamples x,y,z
together, decoder upsamples x,y only" -- but spconv.SparseInverseConv3d can only
invert a conv call that used the EXACT SAME (kernel,stride,padding) it's paired
with via indice_key, so a decoder that restores only 2 of the 3 axes an isotropic
encoder touched isn't directly expressible. This design gets the same practical
effect a different way: finish compressing z to 1 at the bottleneck (where it's
cheap), then hand off to a backbone that is *only* 2D from there on -- "the decoder
only ever touches x,y" becomes true by construction, since there's no z axis left
for it to touch.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # -> voxelnet_baseline/model

import torch
import torch.nn as nn
import spconv.pytorch as spconv

import config
from model import StackedVFE
from backbone3d import Sparse3DBackbone
from zdown_to_sparse2d import ZDownTo2D
from backbone2d_sparse import Sparse2DBackbone
from slotformer import SlotFormerBackbone
from sparse_head_bev import SparseBEVCenterHead


def _stages_needed_for_d1(d_in: int, kernel_size: int = 3) -> int:
    """How many ZDownTo2D stages (kernel/stride=2/padding=kernel//2 on z) it takes
    to bring d_in down to exactly 1 -- computed at construction time so this model
    can slice config.SPARSE_FULLY_ZDOWN_STAGE_CHANNELS (sized for exp1/exp3's
    Dp=22 starting point) down to however many stages THIS model's smaller
    bottleneck D actually needs."""
    n, d = 0, d_in
    while d != 1:
        d = ZDownTo2D.output_d(d, 1, kernel_size)
        n += 1
        if n > 20:
            raise ValueError(f"d_in={d_in} doesn't reach 1 in a reasonable number of stages")
    return n


class SparseBEVDownSlotUpVoxelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.vfe = StackedVFE()

        # grid_size in [D,H,W] to match coords' [batch,z_idx,y_idx,x_idx] column order
        # (config.SPARSE_BEV_GRID_SIZE is (W',H',D')=(x,y,z) counts -- reversed here).
        Wp, Hp, Dp = config.SPARSE_BEV_GRID_SIZE
        self.input_grid_size = (Dp, Hp, Wp)

        self.encoder = Sparse3DBackbone(
            in_channels=128,  # StackedVFE's fixed output width
            stage_channels=config.SPARSE_FULLY_ENCODER_STAGE_CHANNELS,
            num_blocks_per_stage=config.SPARSE_FULLY_ENCODER_NUM_BLOCKS_PER_STAGE,
            down_kernel=config.SPARSE_FULLY_ENCODER_DOWNSAMPLE_KERNEL,
            down_stride=config.SPARSE_FULLY_ENCODER_DOWNSAMPLE_STRIDE,
        )
        n_enc = len(config.SPARSE_FULLY_ENCODER_STAGE_CHANNELS)
        k_enc, s_enc = config.SPARSE_FULLY_ENCODER_DOWNSAMPLE_KERNEL, config.SPARSE_FULLY_ENCODER_DOWNSAMPLE_STRIDE
        pad_enc = k_enc // 2

        def _iso_out(g):
            for _ in range(n_enc):
                g = (g + 2 * pad_enc - k_enc) // s_enc + 1
            return g

        D_bn, H_bn, W_bn = _iso_out(Dp), _iso_out(Hp), _iso_out(Wp)  # bottleneck grid size

        zdown_kernel = config.SPARSE_FULLY_ZDOWN_DOWNSAMPLE_KERNEL
        n_zdown = _stages_needed_for_d1(D_bn, zdown_kernel)
        all_channels = list(config.SPARSE_FULLY_ZDOWN_STAGE_CHANNELS)
        assert len(all_channels) >= n_zdown, (
            f"bottleneck D={D_bn} needs {n_zdown} z-down stages, but "
            f"SPARSE_FULLY_ZDOWN_STAGE_CHANNELS only has {len(all_channels)} entries."
        )
        zdown_channels = all_channels[-n_zdown:]  # reuse the tail (same final width, 128, as exp1/exp3)
        self.zdown = ZDownTo2D(self.encoder.out_channels, zdown_channels, kernel_size=zdown_kernel,
                                indice_key_prefix="exp2_zdown")
        assert ZDownTo2D.output_d(D_bn, n_zdown, zdown_kernel) == 1  # sanity check on the slicing above

        self.slotformer = SlotFormerBackbone(
            self.zdown.out_channels, config.SPARSE_FULLY_SLOTFORMER_WIN_SIZE,
            config.SPARSE_FULLY_SLOTFORMER_NUM_CYCLES, config.SPARSE_FULLY_SLOTFORMER_NUM_HEADS,
            num_axes=2,
        )

        self.backbone2d = Sparse2DBackbone(
            in_channels=self.zdown.out_channels,
            block_channels=config.SPARSE_FULLY_BLOCK_CHANNELS,
            block_layers=config.SPARSE_FULLY_BLOCK_LAYERS,
            upsample_channels=config.SPARSE_FULLY_UPSAMPLE_CHANNELS,
        )

        self.head = SparseBEVCenterHead(self.backbone2d.out_channels)

        # Sparse2DBackbone shrinks its own input by 2x (its output lands on its
        # block1's resolution -- see that module's docstring), on top of the isotropic
        # encoder's own s_enc^n_enc shrink of x,y.
        H_out, W_out = (H_bn - 1) // 2 + 1, (W_bn - 1) // 2 + 1
        sx, sy, _ = config.SPARSE_BEV_VOXEL_SIZE
        self.head_grid_size = (W_out, H_out)      # (W'',H'') naming convention shared with siblings
        self.head_grid_size_hw = (H_out, W_out)   # (H'',W'') -- matches coords' [batch,y,x] order
        self.head_stride = (sx * (Wp / W_out), sy * (Hp / H_out))  # meters/cell at head resolution
        self.pc_range = config.SPARSE_BEV_POINT_CLOUD_RANGE

    def forward(self, voxel_features: torch.Tensor, num_points: torch.Tensor, coords: torch.Tensor):
        voxelwise = self.vfe(voxel_features, num_points)
        batch_size = int(coords[:, 0].max().item()) + 1 if len(coords) else 1

        enc_feat, enc_coords, enc_grid_size = self.encoder(voxelwise, coords, self.input_grid_size, batch_size)
        x = spconv.SparseConvTensor(enc_feat, enc_coords.int(), spatial_shape=list(enc_grid_size),
                                     batch_size=batch_size)
        x2d = self.zdown(x)
        feat = self.slotformer(x2d.features, x2d.indices)

        x2d = x2d.replace_feature(feat)
        out_feat, out_coords = self.backbone2d(x2d)
        pred = self.head(out_feat)
        return pred, out_coords, batch_size
