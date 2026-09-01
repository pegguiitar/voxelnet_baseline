"""validate_rotation3d.py - 3D 회전 확장(rotation3d.py, anchors.py의 gt_boxes_from_objects/
assign_targets 변경)이 실제 데이터에서 문제를 일으킬 수 있는 지점을 정량 검증한다.

합성 각도(30,20,45 등)로 한 round-trip 검증은 이미 통과했지만, 실데이터에는 다음
리스크가 있을 수 있어 별도로 확인함:
  1. Round-trip 정확도가 실제 rotation_x/y/z 조합 전부에서 유지되는가(gimbal lock
     근처인 rotation_y=+-90도 근방이 실제로 얼마나 자주 나오는가 포함).
  2. tilt 크기(순수 z회전에서 얼마나 벗어났는가) 분포 - z-only 근사가 "얼마나 자주,
     얼마나 심하게" 틀리는지를 각도 관점에서 재확인(rotation_xy_impact.py의
     Jaccard/높이오차는 이미 있지만 각도 자체는 안 봤음).
  3. anchor 헤딩 기준 잔차 회전(assign_targets의 R_residual)이 실제로 "작은 값" 근처에
     머무는지 - 이게 8-dim sin/cos 때와 같은 이유(작은 잔차가 절대값보다 회귀하기 쉬움)로
     설계했는데, 실데이터에서 이 가정이 깨지면(잔차가 항상 180도 근처 등) 재설계 필요.

Usage:
    python validate_rotation3d.py
"""

import json
import sys
from pathlib import Path

import numpy as np

import anchors as anchors_mod
import config
import rotation3d

TRIBAND_ANNOTATIONS_DIR = Path(__file__).resolve().parent.parent.parent / "Triband_BEV" / "data" / "annotations"


def load_all_objects():
    """모든 scene의 merged annotation에서 rotation_x/y가 있는 Diver-sonar 객체를 전부 모음."""
    objs = []
    for ann_path in sorted(TRIBAND_ANNOTATIONS_DIR.glob("*.json")):
        with open(ann_path) as f:
            ann = json.load(f)
        for fid, fdata in ann.get("frames", {}).items():
            for o in fdata.get("objects", []):
                if o.get("class", "").endswith("-sonar") and "rotation_x" in o:
                    objs.append(o)
    return objs


def tilt_angle_deg(R: np.ndarray) -> float:
    """순수 z-회전에서 얼마나 벗어났는지(도) - local z축(R의 3번째 열)이 world z축과
    이루는 각도. 0이면 순수 z-yaw(우리가 지금까지 가정해온 것), 클수록 진짜 3D 기울기."""
    local_z_world = R[:, 2]
    cos_angle = np.clip(local_z_world[2], -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def main():
    objects = load_all_objects()
    print(f"총 {len(objects)}개 객체(rotation_x/y 보유) 로드")
    if not objects:
        print("데이터 없음 - backfill_rotation_xy.py를 먼저 돌렸는지 확인")
        return

    tilt_angles = []
    roundtrip_errs = []
    gimbal_near_count = 0
    residual_angles = []

    # anchor heading 후보(잔차 계산용) - config.ANCHOR_ROTATIONS 전부와 비교해 최소 잔차 선택
    anchor_headings = list(config.ANCHOR_ROTATIONS)

    for o in objects:
        rx, ry, rz = o.get("rotation_x", 0.0), o.get("rotation_y", 0.0), o.get("rotation_z", 0.0)
        R = rotation3d.euler_to_matrix(rx, ry, rz)

        # 1) round-trip 정확도
        six = rotation3d.matrix_to_6d(R)
        R2 = rotation3d.sixd_to_matrix_np(six)
        roundtrip_errs.append(np.abs(R - R2).max())

        # 2) tilt 크기
        tilt = tilt_angle_deg(R)
        tilt_angles.append(tilt)
        if abs(abs(ry) - 90) < 5 or abs(abs(rx) - 90) < 5:
            gimbal_near_count += 1

        # 3) 가장 가까운 anchor heading 기준 잔차 회전 크기(각도)
        best_residual_angle = 180.0
        for atheta in anchor_headings:
            R_anchor = rotation3d.euler_to_matrix(0.0, 0.0, np.degrees(atheta))
            R_res = R_anchor.T @ R
            # 잔차 회전각 = arccos((trace(R_res)-1)/2)
            tr = np.clip((np.trace(R_res) - 1) / 2, -1.0, 1.0)
            angle = np.degrees(np.arccos(tr))
            best_residual_angle = min(best_residual_angle, angle)
        residual_angles.append(best_residual_angle)

    tilt_angles = np.array(tilt_angles)
    roundtrip_errs = np.array(roundtrip_errs)
    residual_angles = np.array(residual_angles)

    def stats(a, name):
        print(f"{name}: mean={a.mean():.4f} median={np.median(a):.4f} "
              f"p90={np.percentile(a, 90):.4f} max={a.max():.4f}")

    print()
    print("=== 1) round-trip 정확도 (전체 실데이터) ===")
    stats(roundtrip_errs, "round-trip max element error")
    print(f"허용오차(1e-4) 초과 개수: {(roundtrip_errs > 1e-4).sum()} / {len(roundtrip_errs)}")

    print()
    print("=== 2) tilt 크기(순수 z-yaw에서 벗어난 정도, 도) ===")
    stats(tilt_angles, "tilt angle (deg)")
    for thresh in [15, 30, 45, 60]:
        pct = 100 * (tilt_angles > thresh).mean()
        print(f"  tilt > {thresh}도: {pct:.1f}%")
    print(f"gimbal lock 근접(|rx| or |ry| within 5deg of 90): {gimbal_near_count}개 "
          f"({100*gimbal_near_count/len(objects):.1f}%)")

    print()
    print("=== 3) anchor heading 기준 잔차 회전 크기(도) - 설계 가정(작은 잔차) 검증 ===")
    stats(residual_angles, "residual angle vs nearest anchor heading (deg)")
    for thresh in [45, 90, 135]:
        pct = 100 * (residual_angles > thresh).mean()
        print(f"  잔차 > {thresh}도: {pct:.1f}%")


if __name__ == "__main__":
    main()
