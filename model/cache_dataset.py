"""cache_dataset.py - `Triband_BEV/model/cache_bev.py`와 완전히 같은 전략을 VoxelNet에
적용한다: 학습이 매 스텝(사실상 매 epoch) 다시 계산하던 CPU 전처리(voxelize + anchor
IoU 매칭)를 프레임당 **한 번만** 계산해 압축 npz로 캐싱하고, 학습은 그 캐시만 읽는다.

이전 버전(export_dataset.py)은 raw point cloud + 라벨 json만 내보내고 voxelize/anchor
매칭은 Dataset.__getitem__에서 매번 다시 했다 - Colab에서 epoch 0 중반(step 800/1851)
까지도 한참 걸리는 원인이었다(voxelize ~27ms + anchor 매칭 shapely IoU ~40ms, 프레임당
합산 ~67ms x 7401 샘플 x 16 epoch = 약 2.2시간이 CPU 전처리에만 소모).

voxelize의 point 서브샘플링 randomness(논문 §2.1.1 "random sampling")는 캐싱하면서
잃는다 - 다만 실측(120프레임 샘플)상 voxel당 평균 포인트 수가 ~2개로 T=35에 거의
못 미쳐 애초에 서브샘플링이 거의 발동하지 않으므로(포인트가 T개보다 적으면 그냥 전부
씀), 잃는 게 실질적으로 거의 없다. anchor 매칭은 GT 박스+anchor grid만으로 결정되는
100% 결정론적 값이라 애초에 캐싱 안 하는 게 손해였다.

캐시 레이아웃: cache/voxel/<scene_id>/frame_XXXXXX.npz (+ train만 frame_XXXXXX_aug1.npz)
  - voxel_features: (K,T,7) float32
  - coords: (K,3) int64 [z,y,x]
  - num_points: (K,) int64
  - cls_labels: (A,H,W) int64  (모델 (B,A,H,W) 컨벤션과 동일)
  - reg_targets: (A,H,W,12) float32 - [dx,dy,dz,dl,dw,dh,6D회전(6)] (완전한 3D 회전,
    rotation3d.py 참고 - x,y 회전 무시 시 82.5% 박스 점 소속 판정이 틀리는 게 확인돼
    z-yaw sin/cos 2채널에서 확장됨)
  - gt_boxes: (M,13) float32 [x,y,z,l,w,h,theta_z_rad,6D회전(6)] - eval_voxelnet.py의
    AP 채점용(world-space 원본 GT, anchor 매칭과 무관 - 학습에는 안 쓰이고 평가 전용).
    theta_z_rad는 BEV NMS/footprint 근사용, 6D가 3D IoU(iou_3d_obb) 채점용 진짜 회전
  - heatmap: (1,H,W) float32 - anchor-free(CenterPoint식) head용, model.RPNCenterHead 짝
  - reg_mask: (H,W) bool, offset/z_center/dim_center: (H,W,2/1/3) float32,
    rot_center: (H,W,6) float32(마찬가지로 6D 완전 회전)
    (heatmap_targets.py 참고 - 기존 anchor 기반 타겟과 같은 grid를 공유하므로 voxelize를
    두 번 할 필요 없이 이 npz 하나로 두 head 실험을 다 지원한다)
  - density_center: (H,W) int64 {0,1,2} - RAANet(arXiv:2111.09515)식 보조 head 타겟
    (sparse/adequate/dense, config.DENSITY_THRESH_LOW/HIGH), reg_mask 위치에서만 유효

`_aug1` 사본은 `augment.py`(VoxelNet 논문 §3.2, 3종)를 한 번 적용한 뒤 원본과 완전히
같은 절차로 voxelize+anchor 매칭까지 다시 계산한 것 - TriBand-BEV의
`export_yolo_dataset.py`와 동일 관례(train만, 프레임당 사본 1개, val/test는 증강 안 함).
정밀도 격차 분석(reports/precision_gap_analysis.html §B4) 참고.

Usage:
    python cache_dataset.py [--force]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# common/filter_outliers는 Triband_BEV/baseline에 있고 common이 labeling-tool(stamp_core)을
# 최상단 import한다 - 이 레포가 없는 환경(Modal 컨테이너 등)에서 모듈 import 자체가 실패하지
# 않도록 지연 로드로 미룬다. build_cache_arrays 등 순수 타겟 생성 함수는 common을 안 쓰므로
# on-the-fly Dataset이 이 모듈의 build_cache_arrays를 Modal에서 재사용할 수 있다.
def _load_common():
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Triband_BEV" / "baseline"))
    import common  # noqa: E402
    import filter_outliers  # noqa: E402
    return common, filter_outliers

import anchors as anchors_mod
import augment
import config
import heatmap_targets as ht
from voxelize import augment_with_centroid_offset, voxelize, voxelize_polar

CACHE_DIR = config.VOXELNET_ROOT / "cache" / "voxel"
# Phase1 polarization(project_polarization_design 참고) 전용 - 기존 Cartesian 캐시와
# 완전히 분리된 디렉토리에 써서 rotation3d 기준선을 안 건드리고 나란히 비교 가능하게 함.
POLAR_CACHE_DIR = config.VOXELNET_ROOT / "cache" / "voxel_polar"


def load_splits() -> dict:
    common, _ = _load_common()
    with open(common.REPORTS_DIR / "splits.json") as f:
        splits = json.load(f)["split"]
    return {s: split for split, scenes in splits.items() for s in scenes}


def load_points(scene_id: str, frame_idx: int) -> np.ndarray:
    common, _ = _load_common()
    raw = np.fromfile(common.sonar_bin_path(scene_id, frame_idx), dtype=np.float32)
    points = raw.reshape(-1, 4 if raw.size % 4 == 0 else 3)
    if points.shape[1] == 3:
        points = np.concatenate([points, np.zeros((len(points), 1), dtype=np.float32)], axis=1)
    return points[~np.isnan(points).any(axis=1)]


def build_cache_arrays(points: np.ndarray, objects: list, anchor_grid: np.ndarray,
                        center_only: bool = False) -> dict:
    """center_only=True면 anchor head 타겟(assign_targets: anchor grid IoU 매칭, 무거움)을
    스킵하고 center head 타겟만 만든다 - on-the-fly center 학습 전용 최적화(2026-08-28).
    캐시 생성 경로는 기본값(False)이라 영향 없음."""
    voxel_xyzr, coords, num_points = voxelize(
        points, config.POINT_CLOUD_RANGE, config.VOXEL_SIZE,
        config.MAX_POINTS_PER_VOXEL, config.MAX_VOXELS)
    voxel_features = augment_with_centroid_offset(voxel_xyzr, num_points)

    # anchor-free(CenterPoint식) 타겟 - 같은 grid를 공유하므로 voxelize를 한 번만 하고
    # 두 head용 타겟을 같이 만든다(model.py의 RPNCenterHead/anchors.py의 RPN과 각각 짝).
    center = ht.build_heatmap_targets(objects, points)
    out = {"voxel_features": voxel_features, "coords": coords, "num_points": num_points,
           "heatmap": center["heatmap"], "reg_mask": center["reg_mask"],
           "offset": center["offset"], "z_center": center["z"],
           "dim_center": center["dim"], "rot_center": center["rot"],
           "density_center": center["density"]}
    if not center_only:
        cls_labels, reg_targets = anchors_mod.assign_targets(anchor_grid, objects)
        out["cls_labels"] = np.transpose(cls_labels, (2, 0, 1)).copy()       # (H,W,A)->(A,H,W)
        out["reg_targets"] = np.transpose(reg_targets, (2, 0, 1, 3)).copy()  # (H,W,A,12)->(A,H,W,12)
        out["gt_boxes"] = anchors_mod.gt_boxes_from_objects(objects)
    return out


def build_cache_arrays_polar(points: np.ndarray, objects: list) -> dict:
    """build_cache_arrays()의 polar Phase1 버전. anchor head는 오늘 방침대로(center head가
    최종 확정) 스코프에서 뺐다 - cls_labels/reg_targets를 아예 안 만들어 빌드 시간/용량을
    절약한다(gt_boxes_from_objects는 eval용이라 grid와 무관하게 그대로 유지)."""
    voxel_xyzr, coords, num_points = voxelize_polar(
        points, config.POLAR_R_RANGE, config.POLAR_THETA_RANGE_DEG,
        (config.POINT_CLOUD_RANGE[2], config.POINT_CLOUD_RANGE[5]),
        config.POLAR_R_BINS, config.POLAR_THETA_BINS, config.GRID_SIZE[2],
        config.MAX_POINTS_PER_VOXEL, config.MAX_VOXELS)
    voxel_features = augment_with_centroid_offset(voxel_xyzr, num_points)

    gt_boxes = anchors_mod.gt_boxes_from_objects(objects)
    center = ht.build_heatmap_targets_polar(objects, points)

    return {"voxel_features": voxel_features, "coords": coords, "num_points": num_points,
            "gt_boxes": gt_boxes,
            "heatmap": center["heatmap"], "reg_mask": center["reg_mask"],
            "offset": center["offset"], "z_center": center["z"],
            "dim_center": center["dim"], "rot_center": center["rot"],
            "density_center": center["density"]}


def cache_frame(scene_id: str, frame_idx: int, objects: list, anchor_grid: np.ndarray, force: bool,
                 cache_dir: Path = CACHE_DIR, polar: bool = False) -> Path:
    out_path = cache_dir / scene_id / f"frame_{frame_idx:06d}.npz"
    if out_path.exists() and not force:
        return out_path
    points = load_points(scene_id, frame_idx)
    arrays = build_cache_arrays_polar(points, objects) if polar else build_cache_arrays(points, objects, anchor_grid)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    return out_path


def cache_augmented_frame(scene_id: str, frame_idx: int, objects: list, anchor_grid: np.ndarray,
                           force: bool, seed: int, cache_dir: Path = CACHE_DIR, polar: bool = False) -> Path:
    out_path = cache_dir / scene_id / f"frame_{frame_idx:06d}_aug1.npz"
    if out_path.exists() and not force:
        return out_path
    points = load_points(scene_id, frame_idx)
    aug_points, aug_objects = augment.augment_frame(points, objects, seed=seed)
    arrays = build_cache_arrays_polar(aug_points, aug_objects) if polar else \
        build_cache_arrays(aug_points, aug_objects, anchor_grid)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="이미 있는 캐시도 재계산")
    parser.add_argument("--no-augment", action="store_true", help="train _aug1 사본 생성 스킵")
    parser.add_argument("--splits", default=None,
                         help="쉼표로 구분된 split 목록만 재생성(예: val,test) - 기본은 전체. "
                              "지정한 split만 처리하고 manifest.json의 나머지 split은 "
                              "기존 값을 그대로 보존한다(덮어쓰지 않음).")
    parser.add_argument("--polar", action="store_true",
                         help="voxel polarization Phase1(Cylinder3D식) 캐시를 별도 디렉토리"
                              "(cache/voxel_polar)에 생성 - 기존 Cartesian 캐시는 안 건드림. "
                              "anchor head 타겟은 스킵(center head만 지원, 오늘 방침).")
    args = parser.parse_args()
    target_splits = set(args.splits.split(",")) if args.splits else None
    cache_dir = POLAR_CACHE_DIR if args.polar else CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    _, filter_outliers = _load_common()
    scene_to_split = load_splits()
    flagged_keys = filter_outliers.load_flagged_keys()
    anchor_grid = None if args.polar else anchors_mod.build_anchor_grid()

    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {}
    for split in ("train", "val", "test"):
        if target_splits is None or split in target_splits:
            manifest[split] = []  # 이번에 처리하는 split만 새로 채움
        else:
            manifest.setdefault(split, [])  # 처리 안 하는 split은 기존 값 유지

    t0 = time.time()
    n_written = n_skipped = 0
    aug_seed = 0
    for scene_id, split in scene_to_split.items():
        if target_splits is not None and split not in target_splits:
            continue
        for frame_idx, objects in filter_outliers.iter_filtered_frames(scene_id, flagged_keys):
            before_exists = (cache_dir / scene_id / f"frame_{frame_idx:06d}.npz").exists()
            out_path = cache_frame(scene_id, frame_idx, objects, anchor_grid, args.force,
                                    cache_dir=cache_dir, polar=args.polar)
            manifest[split].append(str(out_path.relative_to(cache_dir)))
            n_skipped += before_exists and not args.force
            n_written += not (before_exists and not args.force)

            if split == "train" and not args.no_augment:
                before_aug_exists = (cache_dir / scene_id / f"frame_{frame_idx:06d}_aug1.npz").exists()
                aug_path = cache_augmented_frame(scene_id, frame_idx, objects, anchor_grid,
                                                  args.force, seed=aug_seed,
                                                  cache_dir=cache_dir, polar=args.polar)
                aug_seed += 1
                manifest["train"].append(str(aug_path.relative_to(cache_dir)))
                n_skipped += before_aug_exists and not args.force
                n_written += not (before_aug_exists and not args.force)
        print(f"{scene_id} ({split}): 누적 {sum(len(v) for v in manifest.values())}개 항목 캐시 완료")

    with open(cache_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n총 {sum(len(v) for v in manifest.values())}개 항목 "
          f"(신규/재계산 {n_written}, 이미 있어 스킵 {n_skipped}), {elapsed:.1f}초 소요")
    print(", ".join(f"{k}={len(v)}" for k, v in manifest.items()))
    print(f"Wrote {cache_dir / 'manifest.json'}")
    tar_name = "voxelnet_cache_polar.tar.gz" if args.polar else "voxelnet_cache.tar.gz"
    print(f"다음: tar czf {tar_name} -C {cache_dir.parent.parent} cache/{cache_dir.name}")


if __name__ == "__main__":
    main()
