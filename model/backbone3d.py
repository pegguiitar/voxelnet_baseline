import torch.nn as nn

from sparse_ops import SparseBasicBlock, SparseConv3dDown, make_norm1d


class Sparse3DStage(nn.Module):
    """One backbone stage: a strided SparseConv3dDown (changes channels AND
    downsamples x,y,z together -- isotropic stride, same reasoning as the old
    stem's downsample: z-only striding would reintroduce ground-plane bias)
    followed by `num_blocks` SubMConv3d residual blocks that refine features
    at the new channel count without moving the active set again.

    block_dilations (optional, one entry per block, default all 1s = old behavior):
    SubMConv3d dilation grows receptive field WITHOUT changing the active set or
    resolution (unlike down_stride), so e.g. [1,2,4] is a cheap way to widen context
    inside a single stage -- see SubMConv3d's docstring in sparse_ops.py."""

    def __init__(self, in_channels, out_channels, num_blocks, down_kernel=3, down_stride=2,
                 block_dilations=None, norm_type="batch"):
        super().__init__()
        self.down = SparseConv3dDown(
            in_channels, out_channels, kernel_size=down_kernel, stride=down_stride, padding=down_kernel // 2
        )
        self.bn = make_norm1d(norm_type, out_channels)
        self.relu = nn.ReLU(inplace=True)
        block_dilations = block_dilations or [1] * num_blocks
        assert len(block_dilations) == num_blocks
        self.blocks = nn.ModuleList([
            SparseBasicBlock(out_channels, dilation=d, norm_type=norm_type) for d in block_dilations
        ])

    def forward(self, features, coords, index_grid, grid_size, batch_size):
        x, coords, index_grid, grid_size = self.down(features, coords, index_grid, grid_size, batch_size)
        x = self.relu(self.bn(x))
        for block in self.blocks:
            x, coords, index_grid = block(x, coords, index_grid, grid_size)
        return x, coords, index_grid, grid_size


class Sparse3DBackbone(nn.Module):
    """Four-stage 3D sparse conv backbone (filter widths 16/32/48/64 by
    default), every stage downsampling x, y AND z by `down_stride` -- no 2D
    BEV backbone follows, so the per-voxel 3D coordinates (including z) stay
    intact all the way to the detection head, which is what keeps height
    information in the predictions instead of collapsing it into a BEV grid."""

    def __init__(self, in_channels, stage_channels=(16, 32, 48, 64), num_blocks_per_stage=3,
                 down_kernel=3, down_stride=2, block_dilations=None, norm_type="batch"):
        super().__init__()
        stages = []
        c_in = in_channels
        for c_out in stage_channels:
            stages.append(Sparse3DStage(c_in, c_out, num_blocks_per_stage, down_kernel, down_stride,
                                         block_dilations=block_dilations, norm_type=norm_type))
            c_in = c_out
        self.stages = nn.ModuleList(stages)
        self.out_channels = stage_channels[-1]
        self.total_stride = down_stride ** len(stage_channels)

    def forward(self, features, coords, index_grid, grid_size, batch_size):
        for stage in self.stages:
            features, coords, index_grid, grid_size = stage(features, coords, index_grid, grid_size, batch_size)
        return features, coords, index_grid, grid_size
