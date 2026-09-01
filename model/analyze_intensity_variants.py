"""analyze_intensity_variants.py - local_intensity_reliability()가 s0에서 이미 의미 있는
스크리닝 결과(§7.6)를 냈으니, s1/s2 launch 전에 모듈 설계 자체를 더 낫게 바꿀 여지가
있는지 로컬에서 저비용으로 검토한다(학습 없이 캐시 데이터만, 2026-08-20).

현재 설계(foreground_gate.local_intensity_reliability): 셀별 평균(mean) intensity,
배경 셀 평균(mean)을 baseline으로 삼음. 검토 대상 2가지:
1. cell aggregation: mean vs max - voxel(0.1m)->BEV셀(0.2m) 2:1 다운샘플링 과정에서
   강한 반사 포인트 하나가 약한 포인트들에 평균으로 희석될 수 있음(Phase2 VFE
   attention pooling 검토에서 나온 "peak 신호는 max가 유리" 논의와 같은 맥락).
2. baseline aggregation: mean vs median - compactness 검토 때 나온 "edge case엔
   median이 더 이상적" 논의를 intensity baseline에도 적용해볼 가치가 있는지(배경
   셀 중 소수의 강한 clutter가 mean baseline을 밀어올려 gap을 과소평가할 수 있음).

결과(300프레임, test split): max aggregation + median baseline 조합이 모든 bucket에서
gap을 크게 키우고, 특히 mean/mean 설계에서 유일하게 역전(-0.0217)됐던 5m+가
+0.1294로 완전히 뒤집힌다 - mean이 소수 강한 clutter에 끌려 baseline을 과대평가하고,
평균 집계가 peak 반사 신호를 희석시키고 있었다는 뜻. §7.6에 기록."""

import numpy as np
import torch

import foreground_gate as fg
from eval_voxelnet_by_range import BUCKETS
from dataset import CachedVoxelNetDataset


def bev_cell_agg(voxel_features, coords, num_points, h, w, agg: str):
    """agg='mean'|'max' - voxel_features[...,3](intensity)를 BEV 셀로 집계."""
    device = voxel_features.device
    K, T, _ = voxel_features.shape
    valid = (torch.arange(T, device=device).unsqueeze(0) < num_points.unsqueeze(1))
    y_idx = (coords[:, 1] // 2).long().clamp(0, h - 1)
    x_idx = (coords[:, 2] // 2).long().clamp(0, w - 1)
    flat_idx = (y_idx * w + x_idx).unsqueeze(1).expand(-1, T)[valid]
    intensity = voxel_features[..., 3][valid]

    n_cells = h * w
    count = torch.zeros(n_cells)
    count.scatter_add_(0, flat_idx, torch.ones_like(intensity))
    if agg == "mean":
        sum_i = torch.zeros(n_cells)
        sum_i.scatter_add_(0, flat_idx, intensity)
        agg_i = sum_i / count.clamp_min(1.0)
    else:  # max - scatter_reduce amax
        agg_i = torch.full((n_cells,), float("-inf"))
        agg_i.scatter_reduce_(0, flat_idx, intensity, reduce="amax", include_self=True)
        agg_i[count == 0] = 0.0
    return agg_i.view(1, 1, h, w), count.view(1, 1, h, w)


def bucket_of(r: float) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= r < hi:
            return name
    return BUCKETS[-1][0]


def main():
    ds = CachedVoxelNetDataset("../cache/voxel", "test", head="center", load_gt_boxes=True)
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(ds), size=min(300, len(ds)), replace=False)

    stats = {agg: {b[0]: {"pos": [], "neg": []} for b in BUCKETS} for agg in ("mean", "max")}

    for i in idxs:
        sample = ds[i]
        gt_boxes = sample["gt_boxes"]
        if len(gt_boxes) == 0:
            continue
        h, w = 50, 60
        target = fg.fg_target_bev([gt_boxes], h, w)[0, 0]
        xx, yy = fg.cell_centers_xy(h, w)
        rr = torch.sqrt(xx ** 2 + yy ** 2)

        for agg in ("mean", "max"):
            agg_i, count = bev_cell_agg(sample["voxel_features"], sample["coords"], sample["num_points"],
                                          h, w, agg)
            mask = count[0, 0].numpy() >= 1
            agg_np, target_np, rr_np = agg_i[0, 0].numpy(), target.numpy(), rr.numpy()
            for r, v, t, keep in zip(rr_np.flatten(), agg_np.flatten(), target_np.flatten(), mask.flatten()):
                if not keep:
                    continue
                b = bucket_of(float(r))
                stats[agg][b]["pos" if t > 0.5 else "neg"].append(float(v))

    for agg in ("mean", "max"):
        print(f"\n=== cell aggregation = {agg} ===")
        print(f"{'bucket':10s} {'neg_mean':>10s} {'neg_median':>10s} {'pos_mean':>10s} "
              f"{'gap(mean_bg)':>13s} {'gap(median_bg)':>16s}")
        for name, _, _ in BUCKETS:
            neg, pos = stats[agg][name]["neg"], stats[agg][name]["pos"]
            if not neg or not pos:
                print(f"{name:10s}  (데이터 부족)")
                continue
            neg_mean, neg_median = np.mean(neg), np.median(neg)
            pos_mean = np.mean(pos)
            print(f"{name:10s} {neg_mean:10.4f} {neg_median:10.4f} {pos_mean:10.4f} "
                  f"{pos_mean - neg_mean:13.4f} {pos_mean - neg_median:16.4f}")


if __name__ == "__main__":
    main()
