"""cache_revised.py - 수정 라벨(revised_labels)용 voxel 캐시 생성.

기존 cache_dataset.py 로직을 그대로 재사용하되, 데이터 경로만 신규 소스로 리다이렉트한다:
  - annotations : data_revised/annotations (rotations{x,y,z}->rotation_x/y/z 변환본)
  - sonar bins  : /home/eugene/Data/uScenes/{scene}/sonar
  - splits/flag : data_revised/reports (proposed splits.json + 빈 flagged_frames.json)
  - cache 출력  : data_revised/cache/voxel  (기존 cache/voxel 은 안 건드림)

파이프라인 코드(voxelize/heatmap_targets/rotation3d/anchors/augment)는 손대지 않으므로
baseline(cartesian rotation3d) recipe와 완전히 동일한 타겟이 생성된다.
"""
import argparse, json, time, sys
from pathlib import Path

ROOT = Path("/home/eugene/Data/APRILab_baseline")
NEW_ANN = ROOT / "data_revised" / "annotations"
NEW_REPORTS = ROOT / "data_revised" / "reports"
USCENES = Path("/home/eugene/Data/uScenes")
NEW_CACHE = ROOT / "data_revised" / "cache" / "voxel"

sys.path.insert(0, str(ROOT / "Triband_BEV" / "baseline"))
import common
# --- path redirects (attribute lookup happens at call-time, so patching works) ---
common.ANNOTATIONS_DIR = NEW_ANN
common.DATA_DIR = USCENES            # sonar_bin_path = DATA_DIR/{scene}/sonar/frame_*.bin
common.REPORTS_DIR = NEW_REPORTS

import filter_outliers
import cache_dataset as cd
import anchors as anchors_mod
import augment
import numpy as np


def _cache_strong_aug_frame(scene_id, frame_idx, objects, anchor_grid, gt_db, seed, cache_dir):
    """strong-aug 증강 사본 1개 생성(gt-sampling+flip+translation 포함). cache_augmented_frame의
    strong 버전 - 파일명은 _augS.npz로 구분."""
    out = cache_dir / scene_id / f"frame_{frame_idx:06d}_augS.npz"
    if out.exists():
        return out  # resume: 이미 만든 aug 사본 재사용
    points = cd.load_points(scene_id, frame_idx)
    aug_points, aug_objects = augment.augment_frame(points, objects, seed=seed, gt_db=gt_db, strong=True)
    arrays = cd.build_cache_arrays(aug_points, aug_objects, anchor_grid)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=None, help="쉼표구분 (예: val,test). 기본 전체")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="scene당 프레임 수 제한(스모크용)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--strong-aug", action="store_true",
                    help="train 증강 사본을 strong stack(gt-sampling+flip+translation)으로 생성(Arm B). "
                         "별도 --out-cache 권장(Arm A 캐시 보존).")
    ap.add_argument("--out-cache", default=None, help="캐시 출력 dir override(기본 data_revised/cache/voxel)")
    args = ap.parse_args()

    out_cache = Path(args.out_cache) if args.out_cache else NEW_CACHE
    target = set(args.splits.split(",")) if args.splits else None
    out_cache.mkdir(parents=True, exist_ok=True)

    scene_to_split = cd.load_splits()
    flagged = filter_outliers.load_flagged_keys()   # 빈 세트여야 정상
    assert len(flagged) == 0, f"flagged keys should be empty for revised labels, got {len(flagged)}"
    anchor_grid = anchors_mod.build_anchor_grid()

    gt_db = None
    if args.strong_aug:
        print("[strong-aug] gt-sampling DB 구축(train frames)...", flush=True)
        def _train_frames():
            for sid, sp in scene_to_split.items():
                if sp != "train":
                    continue
                for fi, objs in filter_outliers.iter_filtered_frames(sid, flagged):
                    yield cd.load_points(sid, fi), objs
        gt_db = augment.build_gt_database(_train_frames())
        print(f"[strong-aug] DB entries: {len(gt_db)}", flush=True)

    manifest_path = out_cache / "manifest.json"
    manifest = json.load(open(manifest_path)) if manifest_path.exists() else {}
    for sp in ("train", "val", "test"):
        if target is None or sp in target:
            manifest[sp] = []
        else:
            manifest.setdefault(sp, [])

    t0 = time.time(); nw = 0
    for scene_id, split in scene_to_split.items():
        if target is not None and split not in target:
            continue
        n_scene = 0
        for frame_idx, objects in filter_outliers.iter_filtered_frames(scene_id, flagged):
            if args.limit and n_scene >= args.limit:
                break
            out = cd.cache_frame(scene_id, frame_idx, objects, anchor_grid, args.force,
                                 cache_dir=out_cache, polar=False)
            # 빈-voxel 프레임(in-range 포인트 0개, FOV/range 클리핑) 제외 - 학습 불가 샘플이고
            # model.py의 B=coords.max()+1 가정과 충돌해 배치 크래시를 유발한다.
            with np.load(out) as _z:
                if _z["voxel_features"].shape[0] == 0:
                    print(f"SKIP empty-voxel {scene_id}/frame_{frame_idx:06d}", flush=True)
                    continue
            manifest[split].append(str(out.relative_to(out_cache))); nw += 1
            if split == "train" and not args.no_augment:
                if args.strong_aug:
                    ap_ = _cache_strong_aug_frame(scene_id, frame_idx, objects, anchor_grid,
                                                  gt_db, seed=nw, cache_dir=out_cache)
                else:
                    ap_ = cd.cache_augmented_frame(scene_id, frame_idx, objects, anchor_grid,
                                                   args.force, seed=nw, cache_dir=out_cache, polar=False)
                manifest["train"].append(str(ap_.relative_to(out_cache))); nw += 1
            n_scene += 1
            if nw % 500 == 0:
                el = time.time() - t0
                print(f"[{el:6.1f}s] {nw} items ({nw/el:.1f}/s) ... {scene_id}", flush=True)
        print(f"DONE {scene_id} ({split}): scene frames={n_scene}, total items={sum(len(v) for v in manifest.values())}", flush=True)

    json.dump(manifest, open(out_cache / "manifest.json", "w"), indent=2)
    el = time.time() - t0
    print(f"\nALL DONE: {sum(len(v) for v in manifest.values())} items, {el:.1f}s", flush=True)


if __name__ == "__main__":
    main()
