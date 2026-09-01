"""cache_final.py - labels_final(최종 라벨) data_final용 voxel 캐시 빌드 (D단계).

cache_revised.py와 달리:
 (1) 경로를 data_final로 redirect
 (2) empty(0-sonar-객체) 프레임도 포함한다 - common.iter_labeled_frames(전 프레임) 사용.
     empty 프레임 = 순수 negative(all-zero heatmap). augS는 안 만든다(gt-sampling이 다이버를
     붙여 positive로 바꿔버리므로 - negative supervision 목적 보존). 0-포인트 프레임만 skip(크래시).
 (3) manifest에 split별 pos/full/empty 3-리스트 생성 → 학습/평가에서 --split으로 override:
       train      : pos non-empty (raw + augS)         [기존 호환, 학습 기본]
       train_full : train + empty(raw)                 [negative supervision 포함 학습]
       val        : pos non-empty (raw)                [참고용 pos-only]
       val_full   : val + empty(raw)                   [업계 표준 - best 선정에 사용, FP 반영]
       val_sub    : val_full(pos+empty)의 scene당 서브샘플 [학습 중 best 선정 속도, empty 포함]
       test       : pos non-empty (raw)                [positive-only AP, 기존 호환]
       test_full  : test + empty(raw)                  [실전 AP - 빈 프레임 FP가 정밀도에 반영]
       test_empty : empty(raw)만                        [순수 FP율 지표]

사용:
  python cache_final.py --strong-aug --out-cache data_final/cache_strong/voxel
  (--splits val,test 로 부분 빌드/스모크, --limit N 로 scene당 제한)
"""
import argparse, json, time, sys, collections
from pathlib import Path

ROOT = Path("/home/eugene/Data/APRILab_baseline")
NEW_ANN = ROOT / "data_final" / "annotations"
NEW_REPORTS = ROOT / "data_final" / "reports"
USCENES = Path("/home/eugene/Data/uScenes")
DEFAULT_CACHE = ROOT / "data_final" / "cache" / "voxel"

sys.path.insert(0, str(ROOT / "Triband_BEV" / "baseline"))
import common
common.ANNOTATIONS_DIR = NEW_ANN
common.DATA_DIR = USCENES
common.REPORTS_DIR = NEW_REPORTS

import cache_dataset as cd
import anchors as anchors_mod
import augment
import numpy as np

VAL_SUB_EVERY = 3  # val_sub: scene당 매 3프레임 1개(전 scene 유지)


def _all_frames(scene_id):
    """annotation의 모든 프레임을 yield - 0-객체(배경) 프레임 포함.
    common.iter_labeled_frames는 objects>=1인 프레임만 주므로(그래서 진짜 배경 누락),
    여기선 frames dict를 직접 순회한다. sonar bin이 없는 프레임은 skip(배경도 포인트 필요)."""
    ann = common.load_annotation(scene_id)
    for fid, fdata in ann.get("frames", {}).items():
        fi = int(fid)
        binp = USCENES / scene_id / "sonar" / f"frame_{fi:06d}.bin"
        if not binp.exists():
            continue
        yield fi, fdata.get("objects", [])


def _build_raw(scene_id, frame_idx, objects, anchor_grid, force, cache_dir):
    """raw 프레임 npz 생성. 0-포인트(빈-voxel)면 None 반환(skip)."""
    out = cd.cache_frame(scene_id, frame_idx, objects, anchor_grid, force,
                         cache_dir=cache_dir, polar=False)
    with np.load(out) as z:
        if z["voxel_features"].shape[0] == 0:
            return None
    return out


def _build_augS(scene_id, frame_idx, objects, anchor_grid, gt_db, seed, cache_dir):
    out = cache_dir / scene_id / f"frame_{frame_idx:06d}_augS.npz"
    if out.exists():
        return out
    points = cd.load_points(scene_id, frame_idx)
    aug_points, aug_objects = augment.augment_frame(points, objects, seed=seed, gt_db=gt_db, strong=True)
    arrays = cd.build_cache_arrays(aug_points, aug_objects, anchor_grid)
    if arrays["voxel_features"].shape[0] == 0:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=None, help="쉼표구분(예: val,test). 기본 전체")
    ap.add_argument("--strong-aug", action="store_true", help="train pos 프레임에 strong augS 사본 생성")
    ap.add_argument("--out-cache", default=None)
    ap.add_argument("--limit", type=int, default=None, help="scene당 프레임 제한(스모크)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_cache = Path(args.out_cache) if args.out_cache else DEFAULT_CACHE
    out_cache.mkdir(parents=True, exist_ok=True)
    target = set(args.splits.split(",")) if args.splits else None

    scene_to_split = cd.load_splits()
    anchor_grid = anchors_mod.build_anchor_grid()

    gt_db = None
    if args.strong_aug:
        print("[strong-aug] gt-sampling DB 구축(train pos frames)...", flush=True)
        def _train_pos_frames():
            for sid, sp in scene_to_split.items():
                if sp != "train":
                    continue
                for fi, objs in common.iter_labeled_frames(sid):
                    s = common.sonar_objects(objs)
                    if s:
                        yield cd.load_points(sid, fi), s
        gt_db = augment.build_gt_database(_train_pos_frames())
        print(f"[strong-aug] DB entries: {len(gt_db)}", flush=True)

    mpath = out_cache / "manifest.json"
    manifest = json.load(open(mpath)) if mpath.exists() else {}
    keys = ["train", "train_full", "val", "val_sub", "val_full", "test", "test_full", "test_empty"]
    for k in keys:
        if target is None:
            manifest[k] = []
        else:
            manifest.setdefault(k, [])

    val_scene_frames = collections.defaultdict(list)  # val_sub 구성용
    t0 = time.time(); nw = 0; n_empty = 0; n_skip = 0
    for scene_id, split in scene_to_split.items():
        if target is not None and split not in target:
            continue
        n_scene = 0
        for frame_idx, objects in _all_frames(scene_id):  # empty(배경) 프레임 포함
            if args.limit and n_scene >= args.limit:
                break
            n_scene += 1
            s_objs = common.sonar_objects(objects)
            out = _build_raw(scene_id, frame_idx, s_objs, anchor_grid, args.force, out_cache)
            if out is None:
                n_skip += 1
                continue
            rel = str(out.relative_to(out_cache)); nw += 1
            if s_objs:  # ---- positive frame ----
                manifest[split].append(rel)
                manifest[f"{split}_full"].append(rel)
                if split == "train" and args.strong_aug:
                    a = _build_augS(scene_id, frame_idx, s_objs, anchor_grid, gt_db, seed=nw, cache_dir=out_cache)
                    if a is not None:
                        r = str(a.relative_to(out_cache))
                        manifest["train"].append(r); manifest["train_full"].append(r); nw += 1
            else:       # ---- empty frame (pure negative) ----
                n_empty += 1
                manifest[f"{split}_full"].append(rel)
                if split == "test":
                    manifest["test_empty"].append(rel)
            # val_sub는 업계표준(empty 포함)대로 val_full 기준으로 뽑는다(pos+empty 모두 수집)
            if split == "val":
                val_scene_frames[scene_id].append(rel)
            if nw % 500 == 0:
                el = time.time() - t0
                print(f"[{el:6.1f}s] {nw} items ({nw/el:.1f}/s) empty={n_empty} skip0pt={n_skip} ... {scene_id}", flush=True)
        print(f"DONE {scene_id} ({split}): frames={n_scene}", flush=True)

    # val_sub: scene당 매 VAL_SUB_EVERY 프레임 1개(전 scene 유지)
    if target is None or "val" in target:
        sub = []
        for sc, rels in val_scene_frames.items():
            kept = sorted(rels)[::VAL_SUB_EVERY] or sorted(rels)[:1]
            sub.extend(kept)
        manifest["val_sub"] = sub

    json.dump(manifest, open(mpath, "w"), indent=2)
    el = time.time() - t0
    print("\n=== manifest 요약 ===", flush=True)
    for k in keys:
        print(f"  {k:<12}{len(manifest[k])}", flush=True)
    print(f"ALL DONE: {nw} items 작성, empty={n_empty}, 0pt-skip={n_skip}, {el:.1f}s", flush=True)


if __name__ == "__main__":
    main()
