"""analyze_compactness_by_range.py - compactness(BEV 셀 내 point 공간적 응집도)가
실제로 2.5-3.5m에서 튀는 패턴을 보이는지, 학습 없이 로컬 캐시 데이터로 먼저 검증.

가설: fg_gate의 오탐(배경을 foreground로 착각)이 2.5-3.5m에서 집중되는 이유가
"그 range에서 clutter가 우연히 diver처럼 응집된 패턴을 만들기 때문"이라면, **배경 셀의
point spread(분산)가 그 구간에서 유독 작아야 한다**(낮은 spread = 응집됨 = diver처럼
보일 위험). 이게 확인되면 compactness 기반 reliability가 방향이 맞다는 근거가 되고,
반대로 spread가 그 구간에서 딱히 작지 않다면(또는 오히려 커진다면) compactness도
density처럼 range의 대리 신호에 불과할 수 있다.

density처럼 scatter_add(sum, sum_sq)로 분산을 벡터화 - point를 셀별로 그룹화하는
ragged 자료구조 없이 Var = E[X^2] - E[X]^2 공식만으로 계산."""

import argparse

import numpy as np
import torch

import config
import foreground_gate as fg
from eval_voxelnet_by_range import BUCKETS
from dataset import CachedVoxelNetDataset


def bev_point_spread(voxel_features: torch.Tensor, coords: torch.Tensor, num_points: torch.Tensor,
                      h: int, w: int) -> torch.Tensor:
    """voxel_features: (K,T,7) [x,y,z,intensity,...]. coords: (K,3) [z,y,x] - dataset.py의
    단일 샘플 형식(collate_fn 이전, batch_idx 컬럼 없음 - 여긴 프레임 하나씩만 처리).
    num_points: (K,). 반환: (1,1,H,W) BEV(x,y) spread = sqrt(var_x+var_y), 점 0~1개인
    셀은 0(spread 정의 불가 - 별도로 count까지 같이 반환해 호출측에서 걸러야 함).
    density_map(count)도 같이 반환."""
    device = voxel_features.device
    K, T, _ = voxel_features.shape
    valid = (torch.arange(T, device=device).unsqueeze(0) < num_points.unsqueeze(1))  # (K,T)

    y_idx = (coords[:, 1] // 2).long().clamp(0, h - 1)
    x_idx = (coords[:, 2] // 2).long().clamp(0, w - 1)
    flat_idx_per_voxel = y_idx * w + x_idx  # (K,) - single frame이므로 batch=0 고정
    flat_idx = flat_idx_per_voxel.unsqueeze(1).expand(-1, T)[valid]  # (N_valid,)

    x = voxel_features[..., 0][valid]
    y = voxel_features[..., 1][valid]

    n_cells = h * w
    count = torch.zeros(n_cells, device=device)
    sum_x = torch.zeros(n_cells, device=device)
    sum_x2 = torch.zeros(n_cells, device=device)
    sum_y = torch.zeros(n_cells, device=device)
    sum_y2 = torch.zeros(n_cells, device=device)
    count.scatter_add_(0, flat_idx, torch.ones_like(x))
    sum_x.scatter_add_(0, flat_idx, x)
    sum_x2.scatter_add_(0, flat_idx, x * x)
    sum_y.scatter_add_(0, flat_idx, y)
    sum_y2.scatter_add_(0, flat_idx, y * y)

    safe_count = count.clamp_min(1.0)
    var_x = (sum_x2 / safe_count - (sum_x / safe_count) ** 2).clamp_min(0.0)
    var_y = (sum_y2 / safe_count - (sum_y / safe_count) ** 2).clamp_min(0.0)
    spread = torch.sqrt(var_x + var_y)
    return spread.view(1, 1, h, w), count.view(1, 1, h, w)


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
    parser.add_argument("--min-count", type=int, default=3, help="이 개수 미만인 셀은 spread 통계에서 제외(1~2점은 산술적으로 spread가 작게 나오지만 무의미)")
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
        spread, count = bev_point_spread(sample["voxel_features"], sample["coords"],
                                          sample["num_points"], h=50, w=60)
        h, w = spread.shape[-2:]
        target = fg.fg_target_bev([gt_boxes], h, w)[0, 0]
        xx, yy = fg.cell_centers_xy(h, w)
        rr = torch.sqrt(xx ** 2 + yy ** 2)

        spread_np, count_np = spread[0, 0].numpy(), count[0, 0].numpy()
        target_np, rr_np = target.numpy(), rr.numpy()
        mask = count_np >= args.min_count
        for r, s, t, keep in zip(rr_np.flatten(), spread_np.flatten(), target_np.flatten(), mask.flatten()):
            if not keep:
                continue
            b = bucket_of(float(r))
            stats[b]["pos" if t > 0.5 else "neg"].append(float(s))

    print(f"n_frames_used={len(idxs)}  min_count={args.min_count}")
    print(f"{'bucket':10s} {'n_neg':>8s} {'neg_spread_mean':>16s} {'n_pos':>8s} {'pos_spread_mean':>16s}")
    for name, _, _ in BUCKETS:
        neg, pos = stats[name]["neg"], stats[name]["pos"]
        neg_mean = np.mean(neg) if neg else float("nan")
        pos_mean = np.mean(pos) if pos else float("nan")
        print(f"{name:10s} {len(neg):8d} {neg_mean:16.4f} {len(pos):8d} {pos_mean:16.4f}")


if __name__ == "__main__":
    main()
