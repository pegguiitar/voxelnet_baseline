"""zdown_to_sparse2d.py - shared "z-only-stride sparse conv down to D=1" tail, used
by all 3 experiments/*_bev/voxelnet.py redesigns (2026-09-03: replacing the old
"scatter to dense -> dense 2D RPNBackbone" ending all 3 used to share) so that x,y
never gets swept into a dense grid. Once D==1, every surviving (batch,y,x) has at
most one row, so dropping z needs no merge -- just a column drop -- handing off a
genuinely 2D sparse tensor to whatever comes next (Sparse2DBackbone, a 2D
SlotFormer, or straight to the head).

Uses spconv.SparseConv3d (real CUDA kernels) for the actual convolution, but that
alone isn't enough: SparseConv3d, unlike SubMConv3d, doesn't restrict a stride==1
axis to its input support -- for stride=(2,1,1) (z-only downsample, x,y meant to
stay untouched) it still discovers every neighbor-reachable (y,x) the k^3 kernel
touches, "dilating" the active set outward in x,y every stage even though nothing
is being downsampled there. Measured building this: a single stage grew 800
synthetic voxels to 10000+ before applying sparse_ops.restrict_xy_support after
each stage; with it, growth stays bounded (some growth from z-kernel overlap, then
shrinks back down as D keeps shrinking) and lands on exactly the original
unique-(y,x) count once D==1 -- see that function's docstring for the full story.

2026-09-03: added `num_refine_blocks` SparseBasicBlock(s) (backbone3d.py's -- two
SubMConv3d + BN + ReLU + residual, active-set-preserving) after each stage's
restrict_xy_support, same down-then-refine pattern backbone3d.Sparse3DStage already
uses -- exp1/exp3 had no refinement at all before this (just the raw strided
conv), unlike exp2's encoder (Sparse3DStage + residual blocks every stage).
"""
import torch
import torch.nn as nn
import spconv.pytorch as spconv

from sparse_ops import yx_key, restrict_xy_support
from backbone3d import SparseBasicBlock


class ZDownTo2D(nn.Module):
    def __init__(self, in_channels, stage_channels, kernel_size=3, indice_key_prefix="zdown2d",
                 num_refine_blocks=1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.refine_blocks = nn.ModuleList()
        c_in = in_channels
        for i, c_out in enumerate(stage_channels):
            self.convs.append(spconv.SparseConv3d(c_in, c_out, kernel_size=kernel_size,
                                                    stride=(2, 1, 1), padding=(1, 1, 1),
                                                    bias=False, indice_key=f"{indice_key_prefix}_{i}"))
            self.bns.append(nn.BatchNorm1d(c_out))
            self.refine_blocks.append(nn.ModuleList([
                SparseBasicBlock(c_out, kernel_size=kernel_size) for _ in range(num_refine_blocks)
            ]))
            c_in = c_out
        self.relu = nn.ReLU(inplace=True)
        self.out_channels = c_in
        self.kernel_size = kernel_size

    @staticmethod
    def output_d(d_in: int, num_stages: int, kernel_size: int = 3) -> int:
        """Deterministic from config alone (kernel/stride=2/padding=kernel//2 on z
        each stage) -- lets a caller assert D reaches exactly 1 at construction time,
        before any data flows."""
        d = d_in
        pad = kernel_size // 2
        for _ in range(num_stages):
            d = (d + 2 * pad - kernel_size) // 2 + 1
        return d

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        """x: sparse 3D tensor (any starting D). Returns a sparse 2D SparseConvTensor
        (spatial_shape=[H,W], coords=[batch,y,x]) -- caller must ensure `stage_channels`
        was sized so D reaches exactly 1 (asserted by the model constructing this)."""
        H, W = x.spatial_shape[1], x.spatial_shape[2]
        allowed_keys = torch.unique(yx_key(x.indices, H, W))

        for conv, bn, blocks in zip(self.convs, self.bns, self.refine_blocks):
            x = conv(x)
            x = x.replace_feature(self.relu(bn(x.features)))
            x = restrict_xy_support(x, allowed_keys, H, W)
            for block in blocks:
                x = block(x)

        D_final = x.spatial_shape[0]
        assert D_final == 1, (
            f"ZDownTo2D must reach D=1 (got D={D_final}) so dropping z needs no merge -- "
            f"adjust stage_channels' length for this input D."
        )
        coords_2d = x.indices[:, [0, 2, 3]].contiguous()
        return spconv.SparseConvTensor(x.features, coords_2d, spatial_shape=[H, W], batch_size=x.batch_size)
