"""analyze_intensity_by_range.py - intensity가 reliability_factor 후보로 유효한지
학습 없이 로컬 캐시 데이터로 검증.

compactness와 다른 점: compactness는 fg_gate가 이미 쓰는 것과 같은 "모양" 단서라
순환 논리였다(§7.5). Intensity는 이 프로젝트가 이미 확인한 TVG over-compensation
때문에 range의 단순 대리 신호가 아니고(raw intensity가 range에 따라 오히려 증가),
"이 셀이 diver인지"와 독립적으로 "진짜 단단한 반사체(hard reflector)인지"를 알려줄
잠재력이 있다 - clutter/multipath는 원래 있어야 할 range보다 더 감쇠된(약한)
반사로 나타나는 경우가 많다는 가설.

검증 방법: range bucket마다 GT-positive 셀과 배경 셀의 평균 intensity를 나눠서 본다.
같은 bucket(=같은 TVG 보정 수준) 안에서 pos > neg 격차가 있으면, 그 격차 자체가
range 대리 신호가 아닌 "진짜 정보"라는 뜻 - 특히 2.5-3.5m(compactness가 무너진
구간)에서 이 격차가 살아있는지가 핵심 질문."""

import argparse

import numpy as np
import torch

import foreground_gate as fg
from eval_voxelnet_by_range import BUCKETS
from dataset import CachedVoxelNetDataset


def bev_mean_intensity(voxel_features: torch.Tensor, coords: torch.Tensor, num_points: torch.Tensor,
                        h: int, w: int):
    """voxel_features: (K,T,7) [x,y,z,intensity,...]. coords: (K,3) [z,y,x](단일 프레임,
    collate_fn 이전). 반환: (1,1,H,W) 평균 intensity, (1,1,H,W) count(=density, 참고용)."""
    device = voxel_features.device
    K, T, _ = voxel_features.shape
    valid = (torch.arange(T, device=device).unsqueeze(0) < num_points.unsqueeze(1))  # (K,T)

    y_idx = (coords[:, 1] // 2).long().clamp(0, h - 1)
    x_idx = (coords[:, 2] // 2).long().clamp(0, w - 1)
    flat_idx = (y_idx * w + x_idx).unsqueeze(1).expand(-1, T)[valid]

    intensity = voxel_features[..., 3][valid]

    n_cells = h * w
    count = torch.zeros(n_cells, device=device)
    sum_i = torch.zeros(n_cells, device=device)
    count.scatter_add_(0, flat_idx, torch.ones_like(intensity))
    sum_i.scatter_add_(0, flat_idx, intensity)

    mean_i = sum_i / count.clamp_min(1.0)
    return mean_i.view(1, 1, h, w), count.view(1, 1, h, w)


def bucket_of(r: float) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= r < hi:
            return name
    return BUCKETS[-1][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default="../cache/voxel")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-frames", type=int, default=300)
    parser.add_argument("--min-count", type=int, default=1)
    args = parser.parse_args()

    ds = CachedVoxelNetDataset(args.cache_root, args.split, head="center", load_gt_boxes=True)
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(ds), size=min(args.n_frames, len(ds)), replace=False)

    stats = {b[0]: {"pos": [], "neg": []} for b in BUCKETS}

    for i in idxs:
        sample = ds[i]
        gt_boxes = sample["gt_boxes"]
        if len(gt_boxes) == 0:
            continue
        mean_i, count = bev_mean_intensity(sample["voxel_features"], sample["coords"],
                                            sample["num_points"], h=50, w=60)
        h, w = mean_i.shape[-2:]
        target = fg.fg_target_bev([gt_boxes], h, w)[0, 0]
        xx, yy = fg.cell_centers_xy(h, w)
        rr = torch.sqrt(xx ** 2 + yy ** 2)

        mi_np, count_np = mean_i[0, 0].numpy(), count[0, 0].numpy()
        target_np, rr_np = target.numpy(), rr.numpy()
        mask = count_np >= args.min_count
        for r, v, t, keep in zip(rr_np.flatten(), mi_np.flatten(), target_np.flatten(), mask.flatten()):
            if not keep:
                continue
            b = bucket_of(float(r))
            stats[b]["pos" if t > 0.5 else "neg"].append(float(v))

    print(f"n_frames_used={len(idxs)}  min_count={args.min_count}")
    print(f"{'bucket':10s} {'n_neg':>8s} {'neg_intensity':>14s} {'n_pos':>8s} {'pos_intensity':>14s} {'gap(pos-neg)':>14s}")
    for name, _, _ in BUCKETS:
        neg, pos = stats[name]["neg"], stats[name]["pos"]
        neg_mean = np.mean(neg) if neg else float("nan")
        pos_mean = np.mean(pos) if pos else float("nan")
        gap = pos_mean - neg_mean if (pos and neg) else float("nan")
        print(f"{name:10s} {len(neg):8d} {neg_mean:14.4f} {len(pos):8d} {pos_mean:14.4f} {gap:14.4f}")


if __name__ == "__main__":
    main()
