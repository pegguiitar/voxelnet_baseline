"""dataset.py - 두 가지 Dataset 구현.

`VoxelNetDataset`(로컬 전용, 즉석 계산): Triband_BEV의 raw 데이터 + baseline/common.py,
filter_outliers.py를 그대로 재사용해 voxel/anchor 타겟을 __getitem__마다 계산한다. 다만
common.py가 모듈 최상단에서 `labeling-tool` 레포(stamp_core.py)를 무조건 import하므로
이 레포가 없는 환경(Colab)에서는 아예 import가 실패한다 - 그래서 이 클래스의 의존성
import는 지연 로드로 미룬다(모듈 로드 자체는 항상 성공하게). 스모크 테스트처럼 프레임
수가 적을 때만 쓸 것 - epoch마다 다시 계산하므로 전체 train split 규모에는 느리다.

`CachedVoxelNetDataset`(Colab 학습용 - 기본): `cache_dataset.py`(로컬에서 한 번 실행,
Triband_BEV/model/cache_bev.py와 동일한 전략)가 미리 voxelize+anchor 매칭까지 끝내
압축 npz로 저장해둔 결과를 그대로 읽기만 한다 - __getitem__에서 계산이 전혀 없다.
common.py/labeling-tool 의존성도 없어 tar로 압축해 Colab에 올리기만 하면 된다."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import anchors as anchors_mod
import config
from voxelize import augment_with_centroid_offset, voxelize


def _make_voxel_sample(points: np.ndarray, objects: list, anchor_grid: np.ndarray) -> dict:
    if points.shape[1] == 3:
        points = np.concatenate([points, np.zeros((len(points), 1), dtype=np.float32)], axis=1)
    points = points[~np.isnan(points).any(axis=1)]

    voxel_xyzr, coords, num_points = voxelize(
        points, config.POINT_CLOUD_RANGE, config.VOXEL_SIZE,
        config.MAX_POINTS_PER_VOXEL, config.MAX_VOXELS)
    voxel_features = augment_with_centroid_offset(voxel_xyzr, num_points)

    cls_labels, reg_targets = anchors_mod.assign_targets(anchor_grid, objects)
    # (H,W,A[,7]) -> (A,H,W[,7]): 모델/loss가 쓰는 (B,A,H,W) 컨벤션에 맞춤
    cls_labels = np.transpose(cls_labels, (2, 0, 1)).copy()
    reg_targets = np.transpose(reg_targets, (2, 0, 1, 3)).copy()

    return {
        "voxel_features": torch.from_numpy(voxel_features),
        "num_points": torch.from_numpy(num_points),
        "coords": torch.from_numpy(coords),  # (K,3) [z,y,x], batch idx는 collate에서 붙임
        "cls_labels": torch.from_numpy(cls_labels),
        "reg_targets": torch.from_numpy(reg_targets),
    }


class VoxelNetDataset(Dataset):
    """로컬 전용 - Triband_BEV/data를 baseline/common.py 경유로 직접 읽는다."""

    def __init__(self, scene_ids: list):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Triband_BEV" / "baseline"))
        import common  # noqa: E402
        import filter_outliers  # noqa: E402
        self._common = common

        self.samples = []  # (scene_id, frame_idx, objects)
        flagged_keys = filter_outliers.load_flagged_keys()
        for scene_id in scene_ids:
            for frame_idx, objects in filter_outliers.iter_filtered_frames(scene_id, flagged_keys):
                self.samples.append((scene_id, frame_idx, objects))
        self.anchor_grid = anchors_mod.build_anchor_grid()  # (H,W,A,7), 전 샘플 공통

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scene_id, frame_idx, objects = self.samples[idx]
        bin_path = self._common.sonar_bin_path(scene_id, frame_idx)
        raw = np.fromfile(bin_path, dtype=np.float32)
        points = raw.reshape(-1, 4 if raw.size % 4 == 0 else 3)
        sample = _make_voxel_sample(points, objects, self.anchor_grid)
        sample["scene_id"], sample["frame_idx"] = scene_id, frame_idx
        return sample


ANCHOR_KEYS = ("cls_labels", "reg_targets")
CENTER_KEYS = ("heatmap", "reg_mask", "offset", "z_center", "dim_center", "rot_center", "density_center")


class CachedVoxelNetDataset(Dataset):
    """Colab 전용 - `cache_dataset.py`가 만든 `cache/voxel/manifest.json` + npz만 읽는다.
    __getitem__에 voxelize/anchor 매칭 계산이 전혀 없음 - 디스크 로드 + 텐서 변환뿐.

    `head`: "anchor"(기존 RPN, cls_labels/reg_targets) 또는 "center"(RPNCenterHead,
    heatmap/reg_mask/offset/z_center/dim_center/rot_center) - cache_dataset.py가
    한 npz에 둘 다 저장해두므로 같은 캐시로 양쪽 다 학습 가능."""

    def __init__(self, cache_root: str, split: str, head: str = "anchor", load_gt_boxes: bool = False):
        assert head in ("anchor", "center")
        self.head = head
        self.load_gt_boxes = load_gt_boxes  # Direction 3(foreground_gate.py) target 생성용 -
        # gt_boxes는 npz에 이미 있지만(eval_voxelnet.iter_cached()가 씀) 학습 경로는 지금까지 안 읽었음.
        self.cache_root = Path(cache_root)
        with open(self.cache_root / "manifest.json") as f:
            self.entries = json.load(f)[split]  # 상대경로 리스트, 예: "scene_0000/frame_000001.npz"

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        rel_path = self.entries[idx]
        keys = ("voxel_features", "coords", "num_points") + \
            (ANCHOR_KEYS if self.head == "anchor" else CENTER_KEYS)
        with np.load(self.cache_root / rel_path) as npz:
            sample = {k: torch.from_numpy(npz[k]) for k in keys}
            if self.load_gt_boxes:
                sample["gt_boxes"] = torch.from_numpy(npz["gt_boxes"])  # (M,13) - M 프레임마다 다름
        sample["scene_id"], sample["frame_idx"] = rel_path, -1
        return sample


def collate_fn(batch: list, head: str = "anchor") -> dict:
    voxel_features, num_points, coords = [], [], []
    for b_idx, item in enumerate(batch):
        voxel_features.append(item["voxel_features"])
        num_points.append(item["num_points"])
        K = item["coords"].shape[0]
        batch_col = torch.full((K, 1), b_idx, dtype=torch.int64)
        coords.append(torch.cat([batch_col, item["coords"]], dim=1))
    out = {
        "voxel_features": torch.cat(voxel_features, dim=0) if voxel_features else torch.zeros(0, config.MAX_POINTS_PER_VOXEL, 7),
        "num_points": torch.cat(num_points, dim=0) if num_points else torch.zeros(0, dtype=torch.int64),
        "coords": torch.cat(coords, dim=0) if coords else torch.zeros(0, 4, dtype=torch.int64),
        "meta": [(item["scene_id"], item["frame_idx"]) for item in batch],
    }
    keys = ANCHOR_KEYS if head == "anchor" else CENTER_KEYS
    for k in keys:
        out[k] = torch.stack([item[k] for item in batch])
    if "gt_boxes" in batch[0]:
        out["gt_boxes"] = [item["gt_boxes"] for item in batch]  # 프레임마다 M이 달라 리스트로(meta와 동일 패턴)
    return out


def load_scene_split(split: str) -> list:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Triband_BEV" / "baseline"))
    import common  # noqa: E402
    with open(common.REPORTS_DIR / "splits.json") as f:
        return json.load(f)["split"][split]
