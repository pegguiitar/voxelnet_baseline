"""validate_iou_3d_obb.py - eval_voxelnet.py의 새 3D OBB IoU(iou_3d_obb, Monte-Carlo)가
정확한지 검증. 일반 OBB-OBB IoU는 닫힌 해가 없어(그래서 Monte-Carlo를 택함) 직접
정답과 비교할 수는 없지만, z-yaw만 있는 특수 케이스에서는 기존 iou_3d()(정확한 폴리곤x
높이구간 방식, 이 프로젝트 전체가 지금까지 써온 방법)와 반드시 일치해야 한다 - 이걸
기준점으로 삼는다.

Usage:
    python validate_iou_3d_obb.py
"""

import numpy as np

import rotation3d
from anchors import rotated_rect_corners
from eval_voxelnet import iou_3d, iou_3d_obb


def zyaw_case(cx, cy, cz, l, w, h, theta_deg, n_samples, rng):
    R = rotation3d.euler_to_matrix(0.0, 0.0, theta_deg)
    center = np.array([cx, cy, cz])
    dims = np.array([l, w, h])
    return center, dims, R


def main():
    rng = np.random.default_rng(42)

    print("=== 1) z-yaw 전용 특수 케이스: 기존 정확한 iou_3d()와 일치하는지 ===")
    test_cases = [
        # (A: cx,cy,cz,l,w,h,theta_deg), (B: 같은 형식) - 다양한 겹침/각도/크기
        ((0, 0, 0, 1.0, 1.0, 1.0, 0), (0, 0, 0, 1.0, 1.0, 1.0, 0)),        # 완전 동일
        ((0, 0, 0, 1.0, 1.0, 1.0, 0), (0.5, 0, 0, 1.0, 1.0, 1.0, 0)),      # 부분 겹침
        ((0, 0, 0, 1.0, 1.0, 1.0, 0), (5, 5, 5, 1.0, 1.0, 1.0, 0)),        # 안 겹침
        ((0, 0, 0, 1.5, 0.9, 0.6, 30), (0.3, 0.2, 0, 1.2, 0.8, 0.6, 45)),  # 회전+다른 크기
        ((2, -1, 0.5, 0.8, 1.3, 0.7, 90), (2.1, -0.9, 0.5, 0.8, 1.3, 0.7, 80)),
        ((1, 1, -0.5, 0.5, 0.5, 1.8, 15), (1.05, 1.05, -0.4, 0.6, 0.6, 1.7, 20)),
    ]
    errs = []
    for (ca, cfg_a), (cb, cfg_b) in [((c[:3], c[3:]), (d[:3], d[3:])) for c, d in test_cases]:
        pass
    for a, b in test_cases:
        acx, acy, acz, al, aw, ah, atheta = a
        bcx, bcy, bcz, bl, bw, bh, btheta = b
        ca, da, Ra = zyaw_case(acx, acy, acz, al, aw, ah, atheta, 3000, rng)
        cb, db, Rb = zyaw_case(bcx, bcy, bcz, bl, bw, bh, btheta, 3000, rng)

        # 기존 방식(정확)
        fa = rotated_rect_corners(acx, acy, al, aw, np.radians(atheta))
        fb = rotated_rect_corners(bcx, bcy, bl, bw, np.radians(btheta))
        za = (acz - ah / 2, acz + ah / 2)
        zb = (bcz - bh / 2, bcz + bh / 2)
        iou_exact = iou_3d(fa, za, fb, zb)

        # 새 방식(Monte-Carlo, n=20000으로 노이즈 최소화해서 비교)
        iou_mc = iou_3d_obb(ca, da, Ra, cb, db, Rb, n_samples=20000, rng=rng)

        err = abs(iou_exact - iou_mc)
        errs.append(err)
        print(f"  exact={iou_exact:.4f} mc={iou_mc:.4f} 차이={err:.4f}")

    errs = np.array(errs)
    print(f"\n최대 오차: {errs.max():.4f}, 평균 오차: {errs.mean():.4f}")

    print()
    print("=== 2) n_samples에 따른 수렴 확인(같은 케이스 반복) ===")
    a, b = test_cases[3]
    acx, acy, acz, al, aw, ah, atheta = a
    bcx, bcy, bcz, bl, bw, bh, btheta = b
    ca, da, Ra = zyaw_case(acx, acy, acz, al, aw, ah, atheta, 0, rng)
    cb, db, Rb = zyaw_case(bcx, bcy, bcz, bl, bw, bh, btheta, 0, rng)
    fa = rotated_rect_corners(acx, acy, al, aw, np.radians(atheta))
    fb = rotated_rect_corners(bcx, bcy, bl, bw, np.radians(btheta))
    za = (acz - ah / 2, acz + ah / 2)
    zb = (bcz - bh / 2, bcz + bh / 2)
    iou_exact = iou_3d(fa, za, fb, zb)
    print(f"정확값: {iou_exact:.4f}")
    for n in [200, 500, 1000, 3000, 10000, 30000]:
        vals = [iou_3d_obb(ca, da, Ra, cb, db, Rb, n_samples=n, rng=np.random.default_rng(s))
                for s in range(10)]
        vals = np.array(vals)
        print(f"  n_samples={n:6d}: mean={vals.mean():.4f} std={vals.std():.4f} "
              f"|mean-exact|={abs(vals.mean() - iou_exact):.4f}")

    print()
    print("=== 3) 기본 성질(대칭성, 자기자신=1, 진짜 3D 기울기 케이스도 범위 내인지) ===")
    R_tilt_a = rotation3d.euler_to_matrix(20, 60, 10)
    R_tilt_b = rotation3d.euler_to_matrix(25, 55, 15)
    ca2 = np.array([0.0, 0.0, 0.0]); da2 = np.array([1.5, 0.9, 0.6])
    cb2 = np.array([0.2, 0.1, 0.05]); db2 = np.array([1.4, 0.8, 0.7])
    iou_ab = iou_3d_obb(ca2, da2, R_tilt_a, cb2, db2, R_tilt_b, n_samples=20000, rng=rng)
    iou_ba = iou_3d_obb(cb2, db2, R_tilt_b, ca2, da2, R_tilt_a, n_samples=20000, rng=rng)
    iou_self = iou_3d_obb(ca2, da2, R_tilt_a, ca2, da2, R_tilt_a, n_samples=20000, rng=rng)
    print(f"  대칭성: iou(A,B)={iou_ab:.4f} vs iou(B,A)={iou_ba:.4f} (같아야 함)")
    print(f"  자기자신 IoU: {iou_self:.4f} (1.0이어야 함)")
    print(f"  범위: 0<=iou<=1 확인: {0 <= iou_ab <= 1 and 0 <= iou_ba <= 1}")


if __name__ == "__main__":
    main()
