"""backbone3d_down_slot_up.py - spconv-based sparse 3D backbone: N-stage downsample
encoder (reuses backbone3d.Sparse3DStage) -> SlotFormer (global context on the
deepest, most-downsampled voxels -- fewest active voxels there, so cheapest place to
run attention) -> M-stage upsample decoder (spconv.SparseInverseConv3d + skip
fusion) -> optional output projection.

2026-09-03: migrated to spconv for the same reason backbone3d.py was -- see that
file's module docstring and experiments/exp3_conv_middle_bev/voxelnet.py's for the
full profiling story (the from-scratch sparse_ops.py backward pass was the real
bottleneck, ~80% of total step time, not backbone depth/SlotFormer -- switching to
spconv's real CUDA kernels measured a ~4.4x real-training speedup there). Each
encoder downsample stage gets a unique indice_key so the matching decoder stage's
spconv.SparseInverseConv3d can invert it via the cached rulebook -- the mechanism
spconv itself provides for exactly this "paired inverse, not a generative
transposed conv" design: it never invents a coordinate that wasn't in the
corresponding downsample's INPUT set, which is what lets a skip-connection merge be
a plain index-aligned concat instead of a coordinate-hash join.

Public interface intentionally stays close to the old tuple-based one (features,
coords, grid_size) -- see backbone3d.py's module docstring for why (callers don't
need to learn the spconv.SparseConvTensor API).
"""
import torch
import torch.nn as nn
import spconv.pytorch as spconv

from backbone3d import Sparse3DStage, SparseBasicBlock
from slotformer import SlotFormerBackbone


class _DecoderStage(nn.Module):
    """Inverts one encoder downsample stage: spconv.SparseInverseConv3d (paired via
    indice_key with that stage's Sparse3DStage.down) restores the pre-downsample
    coordinate set/order exactly, so concatenating with the cached skip features is
    a plain index-aligned concat, then a SubMConv3d fuses the concatenated channels
    back down to skip_channels, then `num_blocks` residual blocks refine at that
    width."""

    def __init__(self, in_channels, skip_channels, num_blocks, kernel_size, indice_key):
        super().__init__()
        self.up = spconv.SparseInverseConv3d(in_channels, skip_channels, kernel_size,
                                              indice_key=indice_key, bias=False)
        self.up_bn = nn.BatchNorm1d(skip_channels)
        self.fuse = spconv.SubMConv3d(skip_channels * 2, skip_channels, kernel_size=3, bias=False)
        self.fuse_bn = nn.BatchNorm1d(skip_channels)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList([SparseBasicBlock(skip_channels) for _ in range(num_blocks)])

    def forward(self, x: spconv.SparseConvTensor, skip: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        up = self.up(x)
        up = up.replace_feature(self.relu(self.up_bn(up.features)))
        fused = up.replace_feature(torch.cat([up.features, skip.features], dim=1))
        fused = self.fuse(fused)
        fused = fused.replace_feature(self.relu(self.fuse_bn(fused.features)))
        for block in self.blocks:
            fused = block(fused)
        return fused


class SparseDownSlotUpBackbone(nn.Module):
    def __init__(self, in_channels, stage_channels, num_blocks_per_stage, down_kernel, down_stride,
                 upsample_stages, decoder_blocks_per_stage=None, decoder_out_channels=None,
                 slot_win_size=12, slot_num_cycles=1, slot_num_heads=4):
        super().__init__()
        stage_channels = list(stage_channels)
        n = len(stage_channels)
        assert 0 <= upsample_stages <= n, \
            f"upsample_stages must be between 0 and {n} (= len(stage_channels)), got {upsample_stages}"
        if decoder_blocks_per_stage is None:
            decoder_blocks_per_stage = num_blocks_per_stage
        encoder_channels = [in_channels] + stage_channels

        self.encoder_stages = nn.ModuleList([
            Sparse3DStage(encoder_channels[i], encoder_channels[i + 1], num_blocks_per_stage,
                          down_kernel, down_stride, indice_key=f"down{i}")
            for i in range(n)
        ])
        self.slotformer = SlotFormerBackbone(stage_channels[-1], slot_win_size, slot_num_cycles, slot_num_heads)
        self.num_upsample = upsample_stages
        # Invert only the M *deepest* encoder stages, deepest first -- M=0 -> empty
        # ModuleList (no decoder at all); M=n -> fully restores to input resolution.
        self.decoder_stages = nn.ModuleList([
            _DecoderStage(encoder_channels[i + 1], encoder_channels[i], decoder_blocks_per_stage,
                          down_kernel, indice_key=f"down{i}")
            for i in reversed(range(n - upsample_stages, n))
        ])

        final_channels = encoder_channels[n - upsample_stages]
        out_channels = decoder_out_channels or final_channels
        self._project = out_channels != final_channels
        if self._project:
            self.out_conv = spconv.SubMConv3d(final_channels, out_channels, kernel_size=3, bias=False)
            self.out_bn = nn.BatchNorm1d(out_channels)
            self.out_relu = nn.ReLU(inplace=True)

        self.out_channels = out_channels
        self.total_stride = down_stride ** (n - upsample_stages)

    def forward(self, features: torch.Tensor, coords: torch.Tensor, grid_size, batch_size: int):
        x = spconv.SparseConvTensor(features, coords.int(), spatial_shape=list(grid_size), batch_size=batch_size)

        skips = []  # SparseConvTensor cached BEFORE each encoder stage runs
        for stage in self.encoder_stages:
            skips.append(x)
            x = stage(x)

        x = x.replace_feature(self.slotformer(x.features, x.indices.long()))
        # ↑ global context at the coarsest scale -- fewest active voxels here

        for stage in self.decoder_stages:
            skip = skips.pop()
            x = stage(x, skip)

        if self._project:
            x = self.out_conv(x)
            x = x.replace_feature(self.out_relu(self.out_bn(x.features)))

        return x.features, x.indices.long(), tuple(x.spatial_shape)
