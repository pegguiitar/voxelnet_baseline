"""patch_heatmap_radius.py - heatmap_targets.build_heatmap_targets_polar()의 회전-불변
radius 수정(2026-08-18)을 기존 cache/voxel_polar에 반영한다. patch_density_target.py와
동일 패턴 - voxelize/augment는 다시 안 하고(비싼 부분), heatmap 필드만 다시 계산해서
기존 npz에 덮어쓴다. reg_mask/offset/z/dim/rot/density는 radius 계산과 무관해서 안 바뀌므로
그대로 둔다.

Usage:
    python patch_heatmap_radius.py --cache-root ../cache/voxel_polar
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Triband_BEV" / "baseline"))
import common  # noqa: E402
import filter_outliers  # noqa: E402

import heatmap_targets as ht


def load_points(scene_id: str, frame_idx: int) -> np.ndarray:
    raw = np.fromfile(common.sonar_bin_path(scene_id, frame_idx), dtype=np.float32)
    points = raw.reshape(-1, 4 if raw.size % 4 == 0 else 3)
    if points.shape[1] == 3:
        points = np.concatenate([points, np.zeros((len(points), 1), dtype=np.float32)], axis=1)
    return points[~np.isnan(points).any(axis=1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True)
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    with open(cache_root / "manifest.json") as f:
        manifest = json.load(f)
    all_entries = sorted({e for split in manifest.values() for e in split})
    print(f"패치 대상: {len(all_entries)}개 항목 (cache_root={cache_root})")

    flagged_keys = filter_outliers.load_flagged_keys()
    scene_frames_cache = {}

    def objects_for(scene_id: str, frame_idx: int) -> list:
        if scene_id not in scene_frames_cache:
            scene_frames_cache[scene_id] = dict(filter_outliers.iter_filtered_frames(scene_id, flagged_keys))
        return scene_frames_cache[scene_id].get(frame_idx, [])

    t0 = time.time()
    n_patched = 0
    bad_files = []
    for i, rel_path in enumerate(all_entries):
        npz_path = cache_root / rel_path
        try:
            with np.load(npz_path) as npz:
                existing = {k: npz[k] for k in npz.files}
        except Exception as e:
            # 손상된 npz(예: 이전 작업이 쓰던 도중 죽어서 생긴 파일)는 스킵하고 계속 -
            # 파일 하나 때문에 몇 분짜리 patch 전체가 죽는 걸 막는다. 스킵된 목록은
            # 끝에 모아서 출력하니 별도로 재생성(cache_dataset.cache_frame)하면 됨.
            bad_files.append((rel_path, str(e)))
            continue

        scene_id = Path(rel_path).parent.name
        stem = Path(rel_path).stem
        frame_idx = int(stem.replace("_aug1", "").split("_")[1])

        objects = objects_for(scene_id, frame_idx)
        points = load_points(scene_id, frame_idx)  # _aug1도 patch_density_target.py와 동일하게 원본 objects/points로 근사
        targets = ht.build_heatmap_targets_polar(objects, points)
        existing["heatmap"] = targets["heatmap"]
        np.savez_compressed(npz_path, **existing)
        n_patched += 1

        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(all_entries)} 처리, {time.time() - t0:.1f}s 경과")

    elapsed = time.time() - t0
    print(f"\n완료: {n_patched}개 패치, 손상돼서 스킵 {len(bad_files)}개, {elapsed:.1f}초")
    for rel_path, err in bad_files:
        print(f"  [스킵] {rel_path}: {err}")


if __name__ == "__main__":
    main()
