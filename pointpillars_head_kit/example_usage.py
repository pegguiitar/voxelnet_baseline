"""example_usage.py — 세 config(anchor / center / ours)를 PointPillars BEV feature에
붙여 target→loss→decode까지 도는 최소 예제. `python example_usage.py`로 그대로 실행되며
kit이 self-contained임을 검증한다(더미 feature/라벨 사용).

PointPillars 통합 시 바꿔야 할 것은 단 하나: `feat = torch.randn(B, C, H, W)` 자리에
PointPillars 인코더+2D backbone의 실제 BEV 출력을 넣는 것. 단, 그 출력 해상도가
(H, W)=(50, 60), cell 0.2m, x∈[0,12]·y∈[-5,5]가 되도록 PointPillars grid를 맞출 것.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "aprilab_heads"))

import numpy as np
import torch

import config
import heads
import heatmap_targets as ht
import anchors as A
import center_loss
import loss as anchor_loss
import decode

H, W = config.ANCHOR_GRID_SIZE[::-1]      # (50, 60)  예측 grid (row=y, col=x)
C = 64                                     # PointPillars BEV 채널 수 (예시; 자유)
B = 1

# --- 더미 GT 라벨 1개(기울어진 다이버) + 그 안의 점들 ---
# 라벨 스키마: centroid={x,y,z}(m), dimensions={length,width,height}(m), rotation_x/y/z(rad)
obj = {"centroid": {"x": 6.0, "y": 0.0, "z": 0.1},
       "dimensions": {"length": 1.5, "width": 1.0, "height": 1.1},
       "rotation_x": 0.2, "rotation_y": 0.1, "rotation_z": 0.8}
points = np.zeros((200, 4), dtype=np.float32)
points[:, 0], points[:, 1], points[:, 2] = 6.0, 0.0, 0.1   # (x,y,z,intensity)


def center_config(objects, tag):
    """center head. objects를 그대로 넘기면 'ours'(full 3D rot), heads.zyaw_flatten로
    감싸 넘기면 'center'(CenterPoint, z-yaw only). head 코드는 완전히 동일."""
    t = ht.build_heatmap_targets(objects, points, rotation_repr="6d")
    # per-sample numpy -> (B,...) tensor. 실제로는 collate로 여러 프레임 stack.
    hm = torch.tensor(t["heatmap"])[None]     # (B,1,H,W)
    reg_mask = torch.tensor(t["reg_mask"])[None]  # (B,H,W)
    off = torch.tensor(t["offset"])[None]     # (B,H,W,2)
    z = torch.tensor(t["z"])[None]            # (B,H,W,1)
    dim = torch.tensor(t["dim"])[None]        # (B,H,W,3)
    rot = torch.tensor(t["rot"])[None]        # (B,H,W,6)

    head = heads.CenterHead(in_channels=C, rotation_repr="6d")
    feat = torch.randn(B, C, H, W)            # <-- PointPillars BEV 출력으로 교체
    p_hm, p_off, p_z, p_dim, p_rot = head(feat)
    l, stats = center_loss.center_voxelnet_loss(
        p_hm, p_off, p_z, p_dim, p_rot,
        hm, reg_mask, off, z, dim, rot, rotation_repr="6d")

    boxes = ht.decode_center_boxes(
        torch.sigmoid(p_hm)[0].detach().numpy(),
        p_off[0].permute(1, 2, 0).detach().numpy(),
        p_z[0].permute(1, 2, 0).detach().numpy(),
        p_dim[0].permute(1, 2, 0).detach().numpy(),
        p_rot[0].permute(1, 2, 0).detach().numpy(),
        score_thresh=0.3, rotation_repr="6d")
    print(f"[{tag:13s}] n_pos={int(reg_mask.sum())} loss={l.item():.3f} decode(@0.3)={len(boxes)} boxes")


def anchor_config(objects, tag):
    grid = A.build_anchor_grid()                      # (H,W,A,7)
    cls_lab, reg_tgt = A.assign_targets(grid, objects)  # (H,W,A), (H,W,A,12)
    # loss는 (B,A,H,W) / (B,A,H,W,12) 레이아웃을 기대 -> permute 필수(gotcha).
    cls_lab = np.transpose(cls_lab, (2, 0, 1))          # (A,H,W)
    reg_tgt = np.transpose(reg_tgt, (2, 0, 1, 3))       # (A,H,W,12)

    head = heads.AnchorHead(in_channels=C)
    feat = torch.randn(B, C, H, W)                     # <-- PointPillars BEV 출력으로 교체
    cls, reg = head(feat)                              # (B,A,H,W), (B,A*12,H,W)
    l, stats = anchor_loss.voxelnet_loss(
        cls, reg, torch.tensor(cls_lab)[None], torch.tensor(reg_tgt)[None])

    boxes = decode.decode_boxes(cls[0], reg[0], grid, score_thresh=0.3)  # per-sample
    kept = decode.rotated_nms(boxes, iou_thresh=0.1)
    print(f"[{tag:13s}] n_pos={int((cls_lab==1).sum())} loss={l.item():.3f} "
          f"decode(@0.3)={len(boxes)} -> NMS={len(kept)} boxes")


if __name__ == "__main__":
    print("grid (H,W)=%s  cell=%.2fm  x=[%.0f,%.0f] y=[%.0f,%.0f]" % (
        (H, W), config.ANCHOR_STRIDE[0],
        config.POINT_CLOUD_RANGE[0], config.POINT_CLOUD_RANGE[3],
        config.POINT_CLOUD_RANGE[1], config.POINT_CLOUD_RANGE[4]))
    print("--- config 1: anchor (zyaw target) ---")
    anchor_config(heads.zyaw_flatten([obj]), "anchor")
    print("--- config 2: center = CenterPoint baseline (zyaw target) ---")
    center_config(heads.zyaw_flatten([obj]), "center")
    print("--- config 3: ours = center + full 3D rotation ---")
    center_config([obj], "ours")
    print("OK - kit self-contained")
