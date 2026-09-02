"""backbone3d.py - spconv-based sparse 3D backbone, shared by sparse_voxelnet.py's
predecessor design and experiments/exp1_single_stage_bev.

2026-09-03: migrated from this repo's own from-scratch sparse_ops.py primitives to
spconv (traveller59/spconv2, installed as spconv-cu126 in this venv) after profiling
(experiments/profile_pipeline.py) found sparse_ops.py's hand-rolled SparseConv3dDown
backward pass ate 80% of total step time in exp3_conv_middle_bev -- ~23x its own
forward cost, versus spconv's real CUDA kernels which brought that down to a normal
forward:backward ratio and a measured ~4.4x real-training speedup. See
experiments/exp3_conv_middle_bev/voxelnet.py's docstring for the full profiling
story; this file applies the same fix to the OTHER sparse backbone shape (isotropic
multi-stage downsample + residual blocks, no z-only-stride/decoder).

Public interface intentionally stays close to the old tuple-based one (features,
coords, grid_size) -- NOT a raw spconv.SparseConvTensor -- so callers (this repo's
VoxelNet wrapper classes) don't need to learn the spconv API: build the
SparseConvTensor once at the top of forward(), unwrap it once at the bottom.
coords are returned as torch.long (spconv needs int32 internally; every OTHER
module in this codebase assumes long, e.g. slotformer.py's coords indexing)."""
import torch
import torch.nn as nn
import spconv.pytorch as spconv


class SparseBasicBlock(nn.Module):
    """Two SubMConv3d + BN + ReLU with a residual connection (ResNet-style).
    Submanifold (spconv.SubMConv3d): output support == input support, so this never
    changes the active set -- only refines features at the current resolution."""

    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv1 = spconv.SubMConv3d(channels, channels, kernel_size, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = spconv.SubMConv3d(channels, channels, kernel_size, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        identity = x.features
        out = self.conv1(x)
        out = out.replace_feature(self.relu(self.bn1(out.features)))
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        out = out.replace_feature(self.relu(out.features + identity))
        return out


class Sparse3DStage(nn.Module):
    """One backbone stage: a strided spconv.SparseConv3d (changes channels AND
    downsamples x,y,z together -- isotropic stride, same reasoning as the old
    stem's downsample: z-only striding would reintroduce ground-plane bias)
    followed by `num_blocks` SparseBasicBlock residual blocks that refine features
    at the new channel count without moving the active set again.

    indice_key: only needed when a caller (backbone3d_down_slot_up.py's encoder)
    wants a matching spconv.SparseInverseConv3d elsewhere to invert this exact
    downsample later -- None (default) here, since a plain Sparse3DBackbone (no
    decoder) never needs to pair anything.

    block_dilations (optional, one entry per block, default all 1s = old behavior):
    SubMConv3d dilation grows receptive field WITHOUT changing the active set or
    resolution (unlike down_stride), so e.g. [1,2,4] is a cheap way to widen context
    inside a single stage."""

    def __init__(self, in_channels, out_channels, num_blocks, down_kernel=3, down_stride=2,
                 block_dilations=None, indice_key=None):
        super().__init__()
        self.down = spconv.SparseConv3d(in_channels, out_channels, kernel_size=down_kernel,
                                         stride=down_stride, padding=down_kernel // 2, bias=False,
                                         indice_key=indice_key)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        block_dilations = block_dilations or [1] * num_blocks
        assert len(block_dilations) == num_blocks
        self.blocks = nn.ModuleList([SparseBasicBlock(out_channels, dilation=d) for d in block_dilations])

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        x = self.down(x)
        x = x.replace_feature(self.relu(self.bn(x.features)))
        for block in self.blocks:
            x = block(x)
        return x


class Sparse3DBackbone(nn.Module):
    """Multi-stage 3D sparse conv backbone, every stage downsampling x, y AND z by
    `down_stride` -- no 2D BEV backbone follows in sparse_voxelnet.py's use of this
    (per-voxel 3D coords, including z, stay intact all the way to the detection
    head); experiments/exp1_single_stage_bev instead scatters this backbone's
    output to a dense BEV feature map afterward (see that file)."""

    def __init__(self, in_channels, stage_channels=(16, 32, 48, 64), num_blocks_per_stage=3,
                 down_kernel=3, down_stride=2, block_dilations=None):
        super().__init__()
        stages = []
        c_in = in_channels
        for c_out in stage_channels:
            stages.append(Sparse3DStage(c_in, c_out, num_blocks_per_stage, down_kernel, down_stride,
                                         block_dilations=block_dilations))
            c_in = c_out
        self.stages = nn.ModuleList(stages)
        self.out_channels = stage_channels[-1]
        self.total_stride = down_stride ** len(stage_channels)

    def forward(self, features: torch.Tensor, coords: torch.Tensor, grid_size, batch_size: int):
        x = spconv.SparseConvTensor(features, coords.int(), spatial_shape=list(grid_size), batch_size=batch_size)
        for stage in self.stages:
            x = stage(x)
        return x.features, x.indices.long(), tuple(x.spatial_shape)
