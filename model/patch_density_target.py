"""patch_density_target.py - 이미 만들어진 캐시(cache/voxel, cache/voxel_polar)에
"density_center" 필드만 추가한다(RAANet arXiv:2111.09515식 보조 density-level
classification head, project 메모리 참고). voxelize/anchor 매칭은 이미 캐시에 맞게
계산돼있으므로 다시 안 하고, heatmap_targets.build_heatmap_targets()(또는 _polar 버전)를
다시 불러 "density" 필드만 뽑아서 기존 npz에 병합 저장 - 전체 재생성보다 훨씬 가볍다
(voxelize 없이 point-in-box 계산만).

Usage:
    python patch_density_target.py --cache-root ../cache/voxel
    python patch_density_target.py --cache-root ../cache/voxel_polar --polar
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

FRAME_RE_HELP = "frame_XXXXXX[_aug1].npz 경로에서 scene_id/frame_idx/aug 여부를 역으로 읽는다"


def load_points(scene_id: str, frame_idx: int) -> np.ndarray:
    raw = np.fromfile(common.sonar_bin_path(scene_id, frame_idx), dtype=np.float32)
    points = raw.reshape(-1, 4 if raw.size % 4 == 0 else 3)
    if points.shape[1] == 3:
        points = np.concatenate([points, np.zeros((len(points), 1), dtype=np.float32)], axis=1)
    return points[~np.isnan(points).any(axis=1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--polar", action="store_true")
    parser.add_argument("--force", action="store_true", help="이미 density_center 있어도 재계산")
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    with open(cache_root / "manifest.json") as f:
        manifest = json.load(f)
    all_entries = sorted({e for split in manifest.values() for e in split})
    print(f"패치 대상: {len(all_entries)}개 항목 (cache_root={cache_root}, polar={args.polar})")

    flagged_keys = filter_outliers.load_flagged_keys()
    # scene별 objects를 매 프레임 다시 읽지 않도록 캐싱(iter_filtered_frames가 이미 파일
    # 전체를 한 번에 읽으므로, scene당 한 번만 돌리고 frame_idx로 바로 찾게 dict화).
    scene_frames_cache = {}

    def objects_for(scene_id: str, frame_idx: int) -> list:
        if scene_id not in scene_frames_cache:
            scene_frames_cache[scene_id] = dict(filter_outliers.iter_filtered_frames(scene_id, flagged_keys))
        return scene_frames_cache[scene_id].get(frame_idx, [])

    build_fn = ht.build_heatmap_targets_polar if args.polar else ht.build_heatmap_targets

    t0 = time.time()
    n_patched = n_skipped = n_missing_objects = 0
    for i, rel_path in enumerate(all_entries):
        npz_path = cache_root / rel_path
        with np.load(npz_path) as npz:
            existing = {k: npz[k] for k in npz.files}
        if "density_center" in existing and not args.force:
            n_skipped += 1
            continue

        # 파일명에서 scene_id/frame_idx/aug 여부 파싱: "<scene_id>/frame_NNNNNN[_aug1].npz"
        scene_id = Path(rel_path).parent.name
        stem = Path(rel_path).stem  # frame_NNNNNN 또는 frame_NNNNNN_aug1
        is_aug = stem.endswith("_aug1")
        frame_idx = int(stem.replace("_aug1", "").split("_")[1])

        objects = objects_for(scene_id, frame_idx)
        if not objects:
            n_missing_objects += 1  # 그냥 카운트만(라벨 없는 프레임, 드묾) - density_center는
            # 여전히 써줘야 함(build_fn이 objects=[]면 전부 0인 배열을 만들어줌 - reg_mask도
            # 전부 False라 학습 loss에선 아무 영향 없지만, npz에 키 자체가 없으면
            # CachedVoxelNetDataset이 KeyError로 죽으므로 반드시 채워야 함).

        # _aug1 사본도 원본(미증강) objects/points로 density를 근사한다 - augment.py의
        # 박스별 perturbation(회전 ±18도, 평행이동)이 박스 안 points를 통째로 강체 이동시키는
        # 방식이라(box3d.points_in_box_3d로 잘라서 같이 움직임) 박스 안 point "개수" 자체는
        # 거의 안 바뀜(global scale 0.95~1.05가 경계값 근처 point 소속을 아주 조금 바꿀 수
        # 있는 정도) - 정확한 재현(augment_frame 시드 재사용)보다 훨씬 싸면서 오차도 작다.
        points = load_points(scene_id, frame_idx)
        targets = build_fn(objects, points)
        existing["density_center"] = targets["density"]
        np.savez_compressed(npz_path, **existing)
        n_patched += 1

        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(all_entries)} 처리, {time.time() - t0:.1f}s 경과")

    elapsed = time.time() - t0
    print(f"\n완료: 패치 {n_patched}개, 이미 있어 스킵 {n_skipped}개, "
          f"objects 없어 스킵 {n_missing_objects}개, {elapsed:.1f}초")


if __name__ == "__main__":
    main()
