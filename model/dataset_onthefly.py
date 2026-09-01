"""dataset_onthefly.py - on-the-fly augmentation Dataset.

pre-baked 캐시(CachedVoxelNetDataset)와 달리, 매 __getitem__마다
  raw points 로드 -> (train이면) augment_frame(strong) -> voxelize + 타겟 생성
을 즉석에서 수행한다. 매 epoch 새 랜덤 증강이 걸려 pre-bake 1× 대비 실효 다양성이
훨씬 크다(특히 GT-sampling: 매번 다른 다이버를 붙임).

Modal/Colab 호환: common.py(labeling-tool 의존)를 전혀 안 쓴다. 대신 변환된 annotation
JSON(rotation_x/y/z 포함)과 raw bin 디렉토리 경로를 직접 받는다. box3d는 augment가
지연 import(model dir 또는 Triband_BEV/baseline 경로).
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import anchors as anchors_mod
import config
import augment
import cache_dataset as cd  # build_cache_arrays (common은 lazy라 import 안전)

CENTER_KEYS = ("heatmap", "reg_mask", "offset", "z_center", "dim_center", "rot_center", "density_center")


def _sonar_bin_path(points_root: Path, scene_id: str, frame_idx: int) -> Path:
    return points_root / scene_id / "sonar" / f"frame_{frame_idx:06d}.bin"


def _load_points(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    pts = raw.reshape(-1, 4 if raw.size % 4 == 0 else 3)
    if pts.shape[1] == 3:
        pts = np.concatenate([pts, np.zeros((len(pts), 1), dtype=np.float32)], axis=1)
    return pts[~np.isnan(pts).any(axis=1)]


def _sonar_objects(objs):
    return [o for o in objs if str(o.get("class", "")).endswith("-sonar")]


class OnTheFlyVoxelDataset(Dataset):
    """center head 전용. split=train이고 strong=True면 매 __getitem__에 fresh 증강."""

    def __init__(self, annotations_dir: str, points_root: str, splits_json: str, split: str,
                 strong: bool = False, gt_db=None, exclude=None, include_raw: bool = False):
        """include_raw: strong일 때 각 프레임을 (raw 1개 + 매 epoch 새 증강 1개)로 넣는다
        (epoch=2N). pre-baked strong 캐시(raw 50% + augS 50%)와 구조를 맞춰, online vs
        offline을 '증강 절반이 고정이냐 매번 새것이냐'로만 다르게 하는 통제 비교(D1)용.
        False면 모든 train 프레임을 매번 증강(100%, 교과서 online, D2)."""
        self.points_root = Path(points_root)
        self.split = split
        self.strong = strong and split == "train"
        self.include_raw = include_raw and self.strong
        self.anchor_grid = anchors_mod.build_anchor_grid()
        ann_dir = Path(annotations_dir)
        scenes = json.load(open(splits_json))["split"][split]
        exclude = set(exclude or [])  # {(scene_id, frame_idx)} 빈-voxel 등 제외

        self.samples = []  # (scene_id, frame_idx, objects[sonar], do_aug)
        for sid in scenes:
            ann = json.load(open(ann_dir / f"{sid}.json"))
            for fid, fd in ann.get("frames", {}).items():
                objs = _sonar_objects(fd.get("objects", []))
                if not objs:
                    continue
                fi = int(fid)
                if (sid, fi) in exclude:
                    continue
                if self.include_raw:
                    # 50/50: raw 1개 + 매 epoch 새 증강 1개 (pre-baked 캐시와 동일 구조)
                    self.samples.append((sid, fi, objs, False))
                    self.samples.append((sid, fi, objs, True))
                else:
                    self.samples.append((sid, fi, objs, self.strong))

        # gt-sampling DB: train points에서 1회 구축(strong일 때만). worker fork로 공유됨.
        self.gt_db = None
        if self.strong:
            self.gt_db = gt_db if gt_db is not None else self._build_gt_db(ann_dir, splits_json)

    def _build_gt_db(self, ann_dir: Path, splits_json: str):
        def frames():
            for sid in json.load(open(splits_json))["split"]["train"]:
                ann = json.load(open(ann_dir / f"{sid}.json"))
                for fid, fd in ann.get("frames", {}).items():
                    objs = _sonar_objects(fd.get("objects", []))
                    if not objs:
                        continue
                    p = _sonar_bin_path(self.points_root, sid, int(fid))
                    if p.exists():
                        yield _load_points(p), objs
        return augment.build_gt_database(frames())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, fi, objs, do_aug = self.samples[idx]
        base_points = _load_points(_sonar_bin_path(self.points_root, sid, fi))
        base_objects = [dict(o) for o in objs]
        points, objects = base_points, base_objects
        if do_aug:
            # seed=None -> 매 호출 새 랜덤(매 epoch 다른 증강)
            points, objects = augment.augment_frame(base_points, [dict(o) for o in objs],
                                                     seed=None, gt_db=self.gt_db, strong=True)
        arrays = cd.build_cache_arrays(points, objects, None, center_only=True)
        # 증강으로 포인트가 range 밖으로 다 나가 빈-voxel이 되면(model.py B=coords.max()+1
        # 배치 크래시 유발) 증강 전 원본으로 fallback. 원본이 빈 프레임은 exclude로 이미 제외됨.
        if arrays["voxel_features"].shape[0] == 0:
            arrays = cd.build_cache_arrays(base_points, base_objects, None, center_only=True)
        sample = {"voxel_features": torch.from_numpy(arrays["voxel_features"]),
                  "num_points": torch.from_numpy(arrays["num_points"]),
                  "coords": torch.from_numpy(arrays["coords"])}
        for k in CENTER_KEYS:
            sample[k] = torch.from_numpy(arrays[k])
        sample["scene_id"], sample["frame_idx"] = f"{sid}/frame_{fi:06d}", -1
        return sample
