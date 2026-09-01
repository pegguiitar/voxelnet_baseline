"""cache_static_segments.py - 정지 구간(analyze_static_scenes.py가 탐지) 프레임을
누적해 "밀도만 높은 가짜 프레임"을 만들어 train 캐시에 추가한다(project_postmeeting_
5day_sprint_plan §4 Phase A - static_segment_accumulation_check.html/_3d.html로 스미어링
없이 밀도만 느는 것 확인 완료).

WINDOW_FRAMES=5로 캡을 씌우는 이유: 연속 프레임 배경 상관관계는 ~0.9대로 매우 안정적이지만
96프레임처럼 긴 구간에서는 0.58까지 떨어짐(센서 미세 드리프트 또는 장면 자체 변화 추정) -
검증된 안전 구간(수 프레임)만 누적한다. 긴 구간은 겹치지 않는 WINDOW_FRAMES개씩 여러 윈도우로
쪼개 오히려 더 많은 샘플을 뽑는다(97프레임 구간 -> 19개 밀집 샘플).

GT는 각 윈도우의 첫 프레임 objects를 그대로 재사용 - 다이버가 거의 안 움직이는 구간이므로
무방(analyze_static_scenes.py의 threshold=0.1m 기준으로 이미 필터된 구간).

train split에만 추가(val/test는 절대 증강 안 함 - cache_dataset.py의 _aug1 관례와 동일).

Usage:
    python cache_static_segments.py [--window 5] [--force]
"""

import argparse
import json
from pathlib import Path

import numpy as np

import anchors as anchors_mod
import config
from cache_dataset import CACHE_DIR, build_cache_arrays, load_points, load_splits

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Triband_BEV" / "baseline"))
import filter_outliers  # noqa: E402

SEGMENTS_PATH = config.VOXELNET_ROOT / "reports" / "static_scene_segments.json"
WINDOW_FRAMES = 5


def accumulate_points(scene_id: str, frame_indices: list) -> np.ndarray:
    return np.concatenate([load_points(scene_id, f) for f in frame_indices], axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=WINDOW_FRAMES,
                         help="누적할 연속 프레임 수(캡) - 검증된 안전 구간 기본 5")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(SEGMENTS_PATH) as f:
        segments_data = json.load(f)
    scene_to_split = load_splits()
    flagged_keys = filter_outliers.load_flagged_keys()
    anchor_grid = anchors_mod.build_anchor_grid()

    manifest_path = CACHE_DIR / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    existing_train = set(manifest.get("train", []))

    n_new_entries = 0
    n_written = n_skipped = 0
    new_train_paths = []

    for scene_id, segs in segments_data["segments_by_scene"].items():
        if scene_to_split.get(scene_id) != "train":
            continue  # val/test는 증강 안 함
        frames_by_idx = dict(filter_outliers.iter_filtered_frames(scene_id, flagged_keys))

        for seg in segs:
            if seg["n_frames"] < args.window:
                continue
            start, end = seg["start_frame"], seg["end_frame"]
            all_frames = list(range(start, end + 1))
            # 연속 프레임 구간 전체를 window 크기로 겹치지 않게 슬라이스
            for w0 in range(0, len(all_frames) - args.window + 1, args.window):
                window = all_frames[w0:w0 + args.window]
                rep_frame = window[0]
                objects = frames_by_idx.get(rep_frame, [])
                if not objects:
                    continue

                out_path = CACHE_DIR / scene_id / f"frame_{rep_frame:06d}_static{args.window}.npz"
                rel_path = str(out_path.relative_to(CACHE_DIR))
                n_new_entries += 1
                if rel_path not in existing_train:
                    new_train_paths.append(rel_path)

                if out_path.exists() and not args.force:
                    n_skipped += 1
                    continue

                points = accumulate_points(scene_id, window)
                arrays = build_cache_arrays(points, objects, anchor_grid)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out_path, **arrays)
                n_written += 1

    manifest["train"] = list(existing_train) + [p for p in new_train_paths]
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"정지구간 누적 샘플 {n_new_entries}개 (신규 기록 {len(new_train_paths)}개, "
          f"신규계산 {n_written}, 이미있어 스킵 {n_skipped})")
    print(f"train split 전체: {len(manifest['train'])}개 (기존 {len(existing_train)} + "
          f"신규 {len(new_train_paths)})")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
