"""dense_head_bev.py - RPNCenterHead's head set (heatmap/offset/z/dim/rot/density)
WITHOUT RPNBackbone -- applied directly to whatever dense (B,C,H,W) feature map is
handed to it, at whatever resolution that already is (no forced downsampling).

Needed because model.RPNCenterHead always runs RPNBackbone internally (it's not
optional), but the dense counterparts of the new fully-sparse exp1/exp2/exp3
(2026-09-03 redesign, see README's "Fully-sparse BEV experiments" section) must
match their sparse counterpart's structure exactly except for the backbone -- and
exp3's sparse counterpart has NO 2D backbone at all (predicts directly off the
z-compressed sparse features), so its dense twin can't route through RPNBackbone
either without breaking the "identical except backbone" comparison. Returns the
same (heatmap, offset, z, dim, rot, density) 6-tuple RPNCenterHead does, so
sparse_bev_head.build_bev_targets/decode_bev_center_boxes and
center_loss.center_voxelnet_loss all work unchanged."""
import math

import torch
import torch.nn as nn


class DenseBEVCenterHeadDirect(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        c = in_channels
        self.heatmap_head = nn.Conv2d(c, 1, 1)   # raw logit -- sigmoid happens in loss/decode
        self.offset_head = nn.Conv2d(c, 2, 1)    # [dx,dy] sub-pixel, cell-size units
        self.z_head = nn.Conv2d(c, 1, 1)         # absolute z (m)
        self.dim_head = nn.Conv2d(c, 3, 1)       # [log l, log w, log h]
        self.rot_head = nn.Conv2d(c, 6, 1)       # 6D continuous rotation (rotation3d.py)
        self.density_head = nn.Conv2d(c, 3, 1)   # RAANet-style aux (sparse/adequate/dense)

        # Same focal-loss bias init as model.RPNCenterHead/SparseCenterHead/
        # SparseBEVCenterHead, same reasoning (see their comments).
        prior_prob = 0.1
        nn.init.constant_(self.heatmap_head.bias, -math.log((1 - prior_prob) / prior_prob))

    def forward(self, feat2d: torch.Tensor):
        """feat2d: (B,C,H,W). Returns RPNCenterHead's raw 6-tuple, each (B,*,H,W)."""
        return (self.heatmap_head(feat2d), self.offset_head(feat2d), self.z_head(feat2d),
                self.dim_head(feat2d), self.rot_head(feat2d), self.density_head(feat2d))
