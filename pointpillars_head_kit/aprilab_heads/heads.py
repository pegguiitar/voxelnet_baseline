"""heads.py — APRILab detection heads, decoupled from our encoder/backbone.

우리(VoxelNet/model/model.py)의 RPN(anchor) / RPNCenterHead(center) 예측 head를,
내부 2D backbone(RPNBackbone)을 떼고 **임의의 BEV feature map**에 바로 붙일 수 있게
분리한 버전. PointPillars 인코더+2D backbone 출력(B, C, H, W)을 그대로 먹인다.

세 비교 config (우리 Sec 6.2 ablation과 동일):
  1. "anchor"     : AnchorHead        + anchor target(anchors.assign_targets) + loss.voxelnet_loss
  2. "center"     : CenterHead        + center target(build_heatmap_targets)  + center_loss.center_voxelnet_loss
  3. "ours"       : CenterHead        + center target                          + center_loss.center_voxelnet_loss

  ⚠️ 2번(center=CenterPoint baseline)과 3번(ours)의 **head는 완전히 동일**하다.
     유일한 차이는 rotation TARGET을 만드는 방식:
       - center(zyaw) : GT의 rotation_x/y를 0으로 밀어(z-yaw만) 6D 인코딩  → zyaw_flatten(objects) 사용
       - ours (full)  : GT의 완전한 3D 회전을 그대로 6D 인코딩              → objects 그대로
     anchor(1번)도 rotation은 항상 6D(reg 채널 12 = xyz3 + dim3 + rot6). anchor 역시
     zyaw 비교를 하려면 target을 zyaw_flatten한 objects로 만들면 된다.

  즉 우리 기여의 본질은 "새 head 아키텍처"가 아니라 **BEV-yaw 관례를 full-3D 회전으로 확장**한
  것(같은 center head, target만 full-3D). 이 kit도 그 사실을 그대로 반영한다.

BEV grid 계약(반드시 PointPillars 쪽을 여기에 맞출 것):
  - 예측 feature map 해상도 = (H, W) = config.ANCHOR_GRID_SIZE[::-1] = (50, 60)
  - cell 크기 = 0.2 m  (config.ANCHOR_STRIDE)
  - 물리 범위 = x∈[0,12] m (W축, 60셀), y∈[-5,5] m (H축, 50셀)  (config.POINT_CLOUD_RANGE)
  - 즉 PointPillars pillar grid + 2D backbone stride를 조합해 최종 BEV가 50×60 @0.2m가
    되도록 맞추면 target/decode 코드가 그대로 동작한다. 채널 수 C는 자유(head가 in_channels로 받음).
"""
import math

import torch
import torch.nn as nn

import config
import rotation3d


class AnchorHead(nn.Module):
    """VoxelNet 원안 anchor 기반 cls/reg head (model.RPN에서 backbone만 제거).

    feat: (B, in_channels, H, W) → cls (B, A, H, W), reg (B, A*12, H, W)
      A   = anchor/cell 개수 (= len(config.ANCHOR_ROTATIONS) = 2, 0·π/2 회전)
      12  = [dx,dy,dz (3), dl,dw,dh (3), 6D rotation (6)]   (anchor-relative 잔차)
    target: anchors.build_anchor_grid() + anchors.assign_targets(anchors, objects)
    loss  : loss.voxelnet_loss(cls, reg, cls_labels, reg_targets)
    decode: decode.decode_boxes(cls, reg, anchors, ...) + decode.rotated_nms(...)
    """

    def __init__(self, in_channels: int, num_anchors_per_loc: int = len(config.ANCHOR_ROTATIONS)):
        super().__init__()
        A = num_anchors_per_loc
        self.cls_head = nn.Conv2d(in_channels, A, 1)
        self.reg_head = nn.Conv2d(in_channels, A * 12, 1)

    def forward(self, feat: torch.Tensor):
        return self.cls_head(feat), self.reg_head(feat)


class CenterHead(nn.Module):
    """CenterPoint(Yin et al. 2021) 스타일 anchor-free head (model.RPNCenterHead에서
    backbone·density·fg·stage2 등 부가 branch를 제거한 최소본 - 세 config 공정 비교에
    필요한 5개 예측 head만).

    feat: (B, in_channels, H, W) →
      heatmap (B,1,H,W)  raw logit (sigmoid는 loss/decode에서)
      offset  (B,2,H,W)  [dx,dy] 셀 내부 sub-pixel (셀 크기 단위)
      z       (B,1,H,W)  절대 z(m)
      dim     (B,3,H,W)  [log l, log w, log h]
      rot     (B,C,H,W)  회전, C=rotation3d.ROTATION_CHANNELS[rotation_repr] (기본 6D=6)
    target: build_heatmap_targets(objects, points, rotation_repr=...) → dict
            (heatmap/reg_mask/offset/z/dim/rot). zyaw 비교는 objects를 zyaw_flatten 후 넘김.
    loss  : center_loss.center_voxelnet_loss(hm,off,z,dim,rot, heatmap,reg_mask,
                                             offset,z_center,dim_center,rot_center, ...)
    decode: heatmap_targets.decode_center_boxes(hm_sig, off, z, dim, rot,
                                                score_thresh, rotation_repr)

    rotation_repr: "6d"(기본, ours) | "quat" | "axisangle". center head 전용 대안 표현.
      "center(zyaw)" vs "ours(full)"는 rotation_repr가 아니라 **target의 rotation_mode**로
      구분한다(같은 head). rotation_repr는 rotation-representation ablation(별개 축)용.
    """

    def __init__(self, in_channels: int, rotation_repr: str = "6d"):
        super().__init__()
        self.rotation_repr = rotation_repr
        self.heatmap_head = nn.Conv2d(in_channels, 1, 1)
        self.offset_head = nn.Conv2d(in_channels, 2, 1)
        self.z_head = nn.Conv2d(in_channels, 1, 1)
        self.dim_head = nn.Conv2d(in_channels, 3, 1)
        self.rot_head = nn.Conv2d(in_channels, rotation3d.ROTATION_CHANNELS[rotation_repr], 1)
        # focal-loss 표준 bias 초기화(RetinaNet/CenterNet 관례) - 없으면 heatmap head가
        # "어디든 배경"으로 붕괴한다(model.py RPNCenterHead 주석 참고). 반드시 유지.
        prior_prob = 0.1
        nn.init.constant_(self.heatmap_head.bias, -math.log((1 - prior_prob) / prior_prob))

    def forward(self, feat: torch.Tensor):
        return (self.heatmap_head(feat), self.offset_head(feat), self.z_head(feat),
                self.dim_head(feat), self.rot_head(feat))


def zyaw_flatten(objects: list) -> list:
    """rotation_x/rotation_y를 0으로 밀어 z-yaw만 남긴 라벨 사본 반환
    (cache_final._flatten_to_zyaw와 동일). center/anchor 어느 head든 "zyaw" 비교 target을
    만들 때 build_heatmap_targets / assign_targets에 넘기기 전에 objects를 이걸로 감싼다.
    rotation_z(실제 yaw 라벨)와 centroid/dimensions는 그대로 둔다."""
    out = []
    for o in objects:
        o2 = dict(o)
        o2["rotation_x"] = 0.0
        o2["rotation_y"] = 0.0
        out.append(o2)
    return out
