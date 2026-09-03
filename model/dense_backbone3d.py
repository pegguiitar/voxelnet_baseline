"""dense_backbone3d.py - dense (nn.Conv3d) mirror of backbone3d.Sparse3DBackbone,
for dense_baseline_exp2_bev (the dense counterpart of exp2_down_slot_up_bev's
isotropic 3D encoder). Same channel widths/kernel/stride/block count as
config.SPARSE_FULLY_ENCODER_* (backbone3d.Sparse3DStage + SparseBasicBlock), just
computed with a real dense (B,C,D,H,W) tensor throughout instead of only at active
voxels -- so any speed/memory difference measured against exp2 is attributable to
sparse vs. dense compute alone, same rationale as dense_baseline_bev/exp3."""
import torch.nn as nn


class _DenseBasicBlock(nn.Module):
    """Dense mirror of sparse_ops.py's (now spconv's) SparseBasicBlock: two 3x3x3
    conv + BN + ReLU with a residual connection."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv3d(channels, channels, kernel_size, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class _DenseStage(nn.Module):
    """Dense mirror of backbone3d.Sparse3DStage: one strided downsample conv +
    `num_blocks` residual blocks."""

    def __init__(self, in_channels, out_channels, num_blocks, down_kernel=3, down_stride=2):
        super().__init__()
        pad = down_kernel // 2
        self.down = nn.Conv3d(in_channels, out_channels, down_kernel, stride=down_stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList([_DenseBasicBlock(out_channels) for _ in range(num_blocks)])

    def forward(self, x):
        x = self.relu(self.bn(self.down(x)))
        for block in self.blocks:
            x = block(x)
        return x


class DenseIsotropicEncoder(nn.Module):
    def __init__(self, in_channels, stage_channels, num_blocks_per_stage, down_kernel=3, down_stride=2):
        super().__init__()
        stages = []
        c_in = in_channels
        for c_out in stage_channels:
            stages.append(_DenseStage(c_in, c_out, num_blocks_per_stage, down_kernel, down_stride))
            c_in = c_out
        self.stages = nn.ModuleList(stages)
        self.out_channels = c_in
        self.total_stride = down_stride ** len(stage_channels)

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x
