"""sonar_diver_dataset.py - Dataset adapter reading the 3d_point_cloud sibling
project's real sonar data directly (labeling-tool-main/dataset/PersonX/scene_XXXX/
{sonar,labels}), bypassing this repo's own Triband_BEV-specific cache_final.py/
CachedVoxelNetDataset pipeline entirely -- that pipeline expects a different raw
data layout this project doesn't have. No caching: like 3d_point_cloud's own
data/dataset.py, this reads+voxelizes on the fly in __getitem__ (voxelize.voxelize()
is fast enough per-frame that a Colab-scale training doesn't need a precomputed cache
for this dataset size -- see 3d_point_cloud/data/dataset.py's own docstring, same
reasoning applies here since it's the same data).

Label conversion: label JSON's `rotations` field (Euler x/y/z degrees) uses the exact
same convention rotation3d.euler_to_matrix expects (Rz.Ry.Rx, world_col = R @ local_col,
local x=length/y=width/z=height) -- both this repo and the labeling tool that produced
these labels trace back to the same labelCloud convention (see rotation3d.py's
docstring), so no re-derivation from quaternion is needed.

TRAINVAL_SCENES/TEST_SCENES mirror 3d-point-cloud/configs/default.yaml as of
2026-09-01 (28 trainval + 8 test scenes) -- keep in sync by hand if that config
changes; there's no shared source of truth between the two repos.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import config
import rotation3d
from voxelize import augment_with_centroid_offset, voxelize

DATASET_ROOT = Path(__file__).resolve().parent.parent.parent / "labeling-tool-main" / "dataset"

TEST_SCENES = [
    "Person2/scene_0035", "Person3/scene_0043", "Person4/scene_0055", "Person3/scene_0067",
    "Person4/scene_0077", "Person4/scene_0088", "Person2/scene_0089", "Person4/scene_0093",
]

TRAINVAL_SCENES = [
    "Person2/scene_0050", "Person2/scene_0091", "Person2/scene_0079", "Person2/scene_0078",
    "Person3/scene_0019", "Person1/scene_0068", "Person1/scene_0063", "Person3/scene_0022",
    "Person4/scene_0090", "Person3/scene_0080", "Person4/scene_0074", "Person1/scene_0036",
    "Person3/scene_0064", "Person1/scene_0042", "Person3/scene_0048", "Person3/scene_0072",
    "Person4/scene_0075", "Person1/scene_0049", "Person4/scene_0021", "Person2/scene_0009",
    "Person3/scene_0082", "Person1/scene_0000", "Person1/scene_0044", "Person3/scene_0070",
    "Person2/scene_0076", "Person3/scene_0092", "Person4/scene_0065", "Person4/scene_0081",
]

VAL_FRAME_RATIO = 0.1
SPLIT_SEED = 42


def _list_scene_frames(root: Path, scene_ids: list) -> list:
    frames = []
    for scene_id in scene_ids:
        sonar_dir = root / scene_id / "sonar"
        if not sonar_dir.is_dir():
            raise FileNotFoundError(f"expected sonar dir at {sonar_dir}")
        frames.extend(sorted(sonar_dir.glob("frame_*.bin")))
    return frames


def _label_path_for(sonar_path: Path) -> Path:
    return sonar_path.parent.parent / "labels" / (sonar_path.stem + ".json")


def _load_gt_boxes(sonar_path: Path) -> np.ndarray:
    """-> (M,13) float32 [x,y,z,l,w,h,theta_z(unused,0),6D-rot(6)] -- cache_dataset.py's
    gt_boxes layout, so decode/eval code written against that shape works unchanged."""
    label_path = _label_path_for(sonar_path)
    if not label_path.is_file():
        return np.zeros((0, 13), dtype=np.float32)
    with open(label_path) as f:
        label = json.load(f)
    rows = []
    for obj in label.get("objects", []):
        if "centroid" not in obj or "dimensions" not in obj or "rotations" not in obj:
            continue
        c, d, r = obj["centroid"], obj["dimensions"], obj["rotations"]
        R = rotation3d.euler_to_matrix(r["x"], r["y"], r["z"])
        six = rotation3d.matrix_to_6d(R)
        rows.append(np.concatenate([
            [c["x"], c["y"], c["z"], d["length"], d["width"], d["height"], 0.0],
            six,
        ]))
    if not rows:
        return np.zeros((0, 13), dtype=np.float32)
    return np.stack(rows).astype(np.float32)


class SonarDiverDataset(Dataset):
    """split: "train"/"val" (frame-level random split of TRAINVAL_SCENES, seeded) or
    "test" (whole held-out scenes, TEST_SCENES) -- exact same split semantics as
    3d_point_cloud/data/dataset.py's SonarDiverDataset."""

    def __init__(self, split: str):
        assert split in ("train", "val", "test")
        self.split = split
        self.root = DATASET_ROOT

        if split == "test":
            self.samples = _list_scene_frames(self.root, TEST_SCENES)
        else:
            import random
            all_frames = _list_scene_frames(self.root, TRAINVAL_SCENES)
            all_frames.sort(key=lambda p: str(p))
            rng = random.Random(SPLIT_SEED)
            rng.shuffle(all_frames)
            n_val = round(len(all_frames) * VAL_FRAME_RATIO)
            self.samples = all_frames[:n_val] if split == "val" else all_frames[n_val:]

        if len(self.samples) == 0:
            raise RuntimeError(f"no frames found for split={split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sonar_path = self.samples[idx]
        data = np.fromfile(sonar_path, dtype=np.float32)
        points = np.zeros((0, 4), dtype=np.float32) if data.size == 0 else data.reshape(-1, 4)
        points = points[~np.isnan(points).any(axis=1)]

        voxel_xyzr, coords, num_points = voxelize(
            points, config.SPARSE_POINT_CLOUD_RANGE, config.SPARSE_VOXEL_SIZE,
            config.SPARSE_MAX_POINTS_PER_VOXEL, config.SPARSE_MAX_VOXELS)
        voxel_features = augment_with_centroid_offset(voxel_xyzr, num_points)
        gt_boxes = _load_gt_boxes(sonar_path)

        return {
            "voxel_features": torch.from_numpy(voxel_features),
            "num_points": torch.from_numpy(num_points),
            "coords": torch.from_numpy(coords),  # (K,3) [z_idx,y_idx,x_idx], batch idx added in collate
            "gt_boxes": torch.from_numpy(gt_boxes),
            "frame_id": str(sonar_path.relative_to(self.root)),
        }


def collate_fn(batch: list) -> dict:
    voxel_features, num_points, coords, gt_boxes, frame_ids = [], [], [], [], []
    for b_idx, item in enumerate(batch):
        voxel_features.append(item["voxel_features"])
        num_points.append(item["num_points"])
        k = item["coords"].shape[0]
        batch_col = torch.full((k, 1), b_idx, dtype=torch.int64)
        coords.append(torch.cat([batch_col, item["coords"]], dim=1))
        gt_boxes.append(item["gt_boxes"])
        frame_ids.append(item["frame_id"])

    return {
        "voxel_features": torch.cat(voxel_features, dim=0) if voxel_features
            else torch.zeros(0, config.SPARSE_MAX_POINTS_PER_VOXEL, 7),
        "num_points": torch.cat(num_points, dim=0) if num_points else torch.zeros(0, dtype=torch.int64),
        "coords": torch.cat(coords, dim=0) if coords else torch.zeros(0, 4, dtype=torch.int64),
        "gt_boxes": gt_boxes,  # list[B] of (M_b,13), M varies per frame
        "frame_ids": frame_ids,
        "batch_size": len(batch),
    }
