"""backbone2d_sparse.py - spconv 2D sparse mirror of model.RPNBackbone (VoxelNet
paper Fig.4's 3-block downsample + deconv-upsample-concat FPN neck), for
experiments/exp4_fully_sparse_bev -- the whole point of that experiment is to never
materialize a dense (B,C,H,W) tensor, so this backbone processes the already-sparse,
already-height-compressed-to-2D output of the z-down stages directly with spconv's
2D sparse ops instead of nn.Conv2d.

Same channel/layer counts as RPNBackbone (config.SPARSE_FULLY_* mirrors
RPN_BLOCK_CHANNELS/RPN_BLOCK_LAYERS/RPN_UPSAMPLE_CHANNELS) so the only variable in
that comparison is sparse-vs-dense compute, not backbone capacity.

One real design difference from RPNBackbone: dense's deconv1/2/3 use DIFFERENT,
independently-chosen (kernel,stride) = (1,1)/(2,2)/(4,4) to jump straight from each
block's resolution to block1's resolution in one step (SECOND's convention -- see
model.py's RPNBackbone docstring for why). spconv.SparseInverseConv2d can only
invert a conv call that used the EXACT SAME (kernel,stride,padding) it's paired
with via indice_key, so a one-shot k2s2/k4s4 "inverse" of a k3s2 encoder stage isn't
directly expressible. Instead, each deeper block's features are walked back to
block1's resolution via a CHAIN of exact per-stage inverses (reusing each stage's
own indice_key) -- this lands on block1's exact coordinate set as a byproduct (no
_match_hw-style rounding/cropping needed, unlike the dense version), so a plain
channel concat is correct. Functionally the same "combine multi-scale features at
one resolution" goal as the original Fig.4 neck, just built from exact paired
inverses instead of independently-shaped one-shot deconvs.
"""
import torch
import torch.nn as nn
import spconv.pytorch as spconv


class _DownBlock(nn.Module):
    """Mirrors model.RPNBlock: one strided SparseConv2d (downsample) + (num_layers-1)
    SubMConv2d refinement layers, all stride=1 after the first -- a straight
    sequential chain (no residual), matching RPNBlock's plain nn.Sequential exactly."""

    def __init__(self, in_channels, out_channels, num_layers, indice_key):
        super().__init__()
        self.down = spconv.SparseConv2d(in_channels, out_channels, kernel_size=3, stride=2,
                                         padding=1, bias=False, indice_key=indice_key)
        self.down_bn = nn.BatchNorm1d(out_channels)
        self.refine_convs = nn.ModuleList([
            spconv.SubMConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            for _ in range(num_layers - 1)
        ])
        self.refine_bns = nn.ModuleList([nn.BatchNorm1d(out_channels) for _ in range(num_layers - 1)])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        x = self.down(x)
        x = x.replace_feature(self.relu(self.down_bn(x.features)))
        for conv, bn in zip(self.refine_convs, self.refine_bns):
            x = conv(x)
            x = x.replace_feature(self.relu(bn(x.features)))
        return x


class Sparse2DBackbone(nn.Module):
    def __init__(self, in_channels, block_channels, block_layers, upsample_channels):
        super().__init__()
        c1, c2, c3 = block_channels
        n1, n2, n3 = block_layers
        self.block1 = _DownBlock(in_channels, c1, n1, indice_key="s2d_blk1")
        self.block2 = _DownBlock(c1, c2, n2, indice_key="s2d_blk2")
        self.block3 = _DownBlock(c2, c3, n3, indice_key="s2d_blk3")

        c_up = upsample_channels
        # deconv1 (dense: k1/s1, pure channel projection, no spatial op) -- a kernel=1
        # conv never mixes neighbors, so this is just nn.Linear on the sparse features.
        self.proj1 = nn.Linear(c1, c_up)
        self.proj1_bn = nn.BatchNorm1d(c_up)

        # deconv2 (dense: k2/s2) -- here, the exact paired inverse of block2's own
        # down-conv, landing on block1's coordinate set directly.
        self.up2 = spconv.SparseInverseConv2d(c2, c_up, kernel_size=3, indice_key="s2d_blk2", bias=False)
        self.up2_bn = nn.BatchNorm1d(c_up)

        # deconv3 (dense: k4/s4) -- two chained exact inverses (block3->block2 res,
        # then block2->block1 res, reusing s2d_blk2's rulebook a second time with its
        # own separate learned weights).
        self.up3a = spconv.SparseInverseConv2d(c3, c2, kernel_size=3, indice_key="s2d_blk3", bias=False)
        self.up3a_bn = nn.BatchNorm1d(c2)
        self.up3b = spconv.SparseInverseConv2d(c2, c_up, kernel_size=3, indice_key="s2d_blk2", bias=False)
        self.up3b_bn = nn.BatchNorm1d(c_up)

        self.relu = nn.ReLU(inplace=True)
        self.out_channels = c_up * 3

    def forward(self, x: spconv.SparseConvTensor):
        """x: sparse 2D input (already height-compressed, see exp4's voxelnet.py).
        Returns (features, coords) at block1's resolution, channels = out_channels
        (3x upsample_channels, concatenated) -- coords as plain long tensor (not a
        SparseConvTensor), since callers (the head / target building) only need the
        flat (N,C) features + coords."""
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)

        u1_feat = self.relu(self.proj1_bn(self.proj1(f1.features)))

        u2 = self.up2(f2)
        u2_feat = self.relu(self.up2_bn(u2.features))

        u3a = self.up3a(f3)
        u3a = u3a.replace_feature(self.relu(self.up3a_bn(u3a.features)))
        u3b = self.up3b(u3a)
        u3_feat = self.relu(self.up3b_bn(u3b.features))

        # u1/u2/u3 all land on f1's exact coordinate set (index-aligned by
        # construction -- each inverse writes to the cached parent indices of the
        # forward conv it's paired with, ultimately f1's own), so a plain channel
        # concat is correct here, same as dense RPNBackbone's torch.cat([u1,u2,u3]).
        feat = torch.cat([u1_feat, u2_feat, u3_feat], dim=1)
        return feat, f1.indices.long()
