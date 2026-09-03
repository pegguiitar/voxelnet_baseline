"""dense_baseline_exp2_bev - the literal DENSE counterpart to exp2_down_slot_up_bev,
for a direct dense-vs-sparse comparison of that experiment's specific backbone
shape (isotropic 3D encoder -> z-compress bottleneck -> attention -> 2D U-Net).

VFE -> dense scatter -> dense_backbone3d.DenseIsotropicEncoder (real nn.Conv3d,
SPARSE_FULLY_ENCODER_* shape) -> dense_zdown.DenseZDown (real nn.Conv3d, the
SAME per-stage channels exp2's sparse ZDownTo2D slice uses) -> squeeze z (now 1) ->
dense_slotformer2d.DenseSlotFormerBackbone2D (dense windowed attention, same
win_size/cycles/heads) -> model.RPNBackbone (UNCHANGED -- this literally *is* the
dense 2D U-Net exp2's Sparse2DBackbone mirrors the shape of, so it's reused as-is
rather than reimplemented) -> dense_head_bev.DenseBEVCenterHeadDirect (no
RPNCenterHead -- that would re-run RPNBackbone a second time). Same SPARSE_BEV_*
voxelization, same data/loss/targets as exp2 -- differs in EXACTLY one thing:
every conv from the encoder through the 2D backbone is computed over the full dense
grid instead of only at active voxels/cells.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # -> voxelnet_baseline/model

import torch
import torch.nn as nn

import config
from model import StackedVFE, RPNBackbone
from dense_backbone3d import DenseIsotropicEncoder
from dense_zdown import DenseZDown
from dense_slotformer2d import DenseSlotFormerBackbone2D
from dense_head_bev import DenseBEVCenterHeadDirect


def _stages_needed_for_d1(d_in: int, kernel_size: int = 3) -> int:
    """Same helper as experiments/exp2_down_slot_up_bev/voxelnet.py's -- how many
    kernel=3/stride=2/padding=1 stages it takes to bring d_in down to exactly 1."""
    n, d = 0, d_in
    while d != 1:
        d = DenseZDown.output_d(d, 1, kernel_size)
        n += 1
        if n > 20:
            raise ValueError(f"d_in={d_in} doesn't reach 1 in a reasonable number of stages")
    return n


class DenseBEVDownSlotUpVoxelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.vfe = StackedVFE()

        # grid_size in [D,H,W] to match coords' [batch,z_idx,y_idx,x_idx] column order
        # (config.SPARSE_BEV_GRID_SIZE is (W',H',D')=(x,y,z) counts -- reversed here).
        Wp, Hp, Dp = config.SPARSE_BEV_GRID_SIZE
        self.input_grid_size = (Dp, Hp, Wp)

        self.encoder = DenseIsotropicEncoder(
            in_channels=128,
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

        D_bn, H_bn, W_bn = _iso_out(Dp), _iso_out(Hp), _iso_out(Wp)  # bottleneck grid size -- same
        # arithmetic as exp2_down_slot_up_bev/voxelnet.py, since it's the SAME kernel/stride/padding

        zdown_kernel = config.SPARSE_FULLY_ZDOWN_DOWNSAMPLE_KERNEL
        n_zdown = _stages_needed_for_d1(D_bn, zdown_kernel)
        all_channels = list(config.SPARSE_FULLY_ZDOWN_STAGE_CHANNELS)
        assert len(all_channels) >= n_zdown
        zdown_channels = all_channels[-n_zdown:]  # same slice exp2's sparse ZDownTo2D uses
        self.zdown = DenseZDown(self.encoder.out_channels, zdown_channels, kernel_size=zdown_kernel)
        assert DenseZDown.output_d(D_bn, n_zdown, zdown_kernel) == 1

        self.slotformer = DenseSlotFormerBackbone2D(
            self.zdown.out_channels, config.SPARSE_FULLY_SLOTFORMER_WIN_SIZE,
            config.SPARSE_FULLY_SLOTFORMER_NUM_CYCLES, config.SPARSE_FULLY_SLOTFORMER_NUM_HEADS,
        )

        assert self.zdown.out_channels == config.RPN_IN_CHANNELS, (
            f"RPNBackbone (reused unchanged) expects {config.RPN_IN_CHANNELS} input channels, "
            f"got {self.zdown.out_channels} -- adjust SPARSE_FULLY_ZDOWN_STAGE_CHANNELS' last entry."
        )
        self.backbone2d = RPNBackbone()  # UNCHANGED dense pipeline 2D neck -- see module docstring

        self.head = DenseBEVCenterHeadDirect(self.backbone2d.out_channels)
        self.out_grid_size = (1, H_bn, W_bn)  # post-zdown, pre-RPNBackbone -- for parity with siblings' printouts

        # RPNBackbone's block1 (RPNBlock's first conv: kernel=3,stride=2,padding=1) sets the
        # final output size -- out=(in+2*1-3)//2+1=(in-1)//2+1 (ceil, not floor).
        H_out, W_out = (H_bn - 1) // 2 + 1, (W_bn - 1) // 2 + 1
        sx, sy, _ = config.SPARSE_BEV_VOXEL_SIZE
        self.head_grid_size = (W_out, H_out)      # (W'',H'') naming convention shared with siblings
        self.head_stride = (sx * (Wp / W_out), sy * (Hp / H_out))  # meters/cell at head resolution
        self.pc_range = config.SPARSE_BEV_POINT_CLOUD_RANGE

    def forward(self, voxel_features: torch.Tensor, num_points: torch.Tensor, coords: torch.Tensor):
        voxelwise = self.vfe(voxel_features, num_points)
        batch_size = int(coords[:, 0].max().item()) + 1 if len(coords) else 1

        D, H, W = self.input_grid_size
        dense = voxelwise.new_zeros(batch_size, 128, D, H, W)
        if coords.shape[0] > 0:
            b, z, y, x = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
            dense[b, :, z, y, x] = voxelwise

        enc = self.encoder(dense)          # (B,128,D_bn,H_bn,W_bn)
        z1 = self.zdown(enc)               # (B,128,1,H_bn,W_bn)
        feat2d = z1.squeeze(2)             # (B,128,H_bn,W_bn) -- D==1 by construction, plain squeeze
        feat2d = self.slotformer(feat2d)
        feat2d = self.backbone2d(feat2d)   # (B,768,H_out,W_out)
        return self.head(feat2d)
