"""analyze_miss_runs.py - AB3DMOT류 실제 tracker를 만들기 전에 먼저 답해야 할 질문:
"놓친 프레임들이 짧게 하나만 빠지는 건가, 길게 통째로 안 잡히는 건가?" 진단.

진짜 tracker(Kalman+Hungarian)는 필요 없다 - GT의 identity는 이미 알려져 있으므로(한
scene 안에서 라벨된 diver가 사실상 1명), 매 프레임 GT 중심 근처(radius 이내)에
score_thresh 넘는 peak가 있으면 hit, 없으면 miss로 이진 시퀀스를 만들고 miss run
길이를 히스토그램으로 보면 충분하다. 이 결과가 짧은 gap 위주면 Bayesian temporal
fusion(4.9)이 실제로 도움될 여지가 크고, 긴 결측 위주면 그 트랙은 우선순위가 낮아진다."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import config
from analyze_recall_collapse import decode_all_peaks, load_model_for_analysis, match_by_distance

SCORE_THRESH = 0.3
MATCH_RADIUS_M = 1.0


def scene_frame_sequence(cache_root: Path, scene_id: str, splits: tuple = ("val", "test")) -> list:
    """cache manifest에서 이 scene의 프레임을 프레임 인덱스 순으로 정렬해 반환
    (rel_path 리스트). 증강본(_aug)은 제외(연속 시퀀스가 아니라 개별 재샘플이라 시간축
    분석에 안 맞음)."""
    with open(cache_root / "manifest.json") as f:
        m = json.load(f)
    entries = []
    for split in splits:
        for rel in m[split]:
            if rel.startswith(scene_id + "/") and "_aug" not in rel:
                entries.append(rel)
    entries.sort(key=lambda rel: int(Path(rel).stem.replace("frame_", "")))
    return entries


def hit_miss_sequence(model, device, cache_root: Path, entries: list) -> list:
    """entries 순서대로 model을 돌려 프레임별 hit(1)/miss(0)를 낸다. GT가 없는 프레임은
    None(시퀀스에서 제외 - "놓침"의 정의 자체가 성립하지 않음)."""
    seq = []
    for rel in entries:
        with np.load(cache_root / rel) as npz:
            voxel_features, coords, num_points = npz["voxel_features"], npz["coords"], npz["num_points"]
            gt_boxes = npz["gt_boxes"]
        if len(gt_boxes) == 0:
            seq.append(None)
            continue
        vf = torch.from_numpy(voxel_features).to(device)
        npt = torch.from_numpy(num_points).to(device)
        batch_col = torch.zeros((len(coords), 1), dtype=torch.int64)
        crd = torch.cat([batch_col, torch.from_numpy(coords)], dim=1).to(device)
        with torch.no_grad():
            hm, off, z, dim, rot, density = model(vf, npt, crd)
        hm_sig = torch.sigmoid(hm)[0].cpu().numpy()  # (1,H,W) - decode_all_peaks가 [0]으로 (H,W)를 꺼냄
        off_hwc = off[0].permute(1, 2, 0).cpu().numpy()  # (C,H,W) -> (H,W,C)
        z_hwc = z[0].permute(1, 2, 0).cpu().numpy()
        peaks, _n_peak = decode_all_peaks(hm_sig, off_hwc, z_hwc)
        gt_xyz = gt_boxes[:, :3]
        results = match_by_distance(peaks, gt_xyz, SCORE_THRESH, MATCH_RADIUS_M)
        n_tp = sum(1 for _, is_tp in results if is_tp)
        seq.append(1 if n_tp > 0 else 0)
    return seq


def run_length_histogram(seq: list) -> dict:
    """None(GT 없음) 제외한 0/1 시퀀스에서 0(miss)의 연속 run 길이 분포. 반환:
    {run_length: count}, 그리고 hit/miss 총 프레임 수."""
    clean = [v for v in seq if v is not None]
    runs = defaultdict(int)
    i = 0
    while i < len(clean):
        if clean[i] == 0:
            j = i
            while j < len(clean) and clean[j] == 0:
                j += 1
            runs[j - i] += 1
            i = j
        else:
            i += 1
    n_hit = sum(clean)
    n_miss = len(clean) - n_hit
    return dict(runs), n_hit, n_miss, len(clean)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cache-root", default="../cache/voxel")
    parser.add_argument("--scene", default="scene_0049")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int, default=0, help="0=전체, N이면 앞에서부터 N프레임만(연속성 유지, CPU 시간 절약용)")
    args = parser.parse_args()

    model = load_model_for_analysis(Path(args.ckpt))
    model.to(args.device).eval()

    cache_root = Path(args.cache_root)
    entries = scene_frame_sequence(cache_root, args.scene)
    if args.max_frames > 0:
        entries = entries[:args.max_frames]
    print(f"scene={args.scene} n_frames={len(entries)}")

    seq = hit_miss_sequence(model, args.device, cache_root, entries)
    runs, n_hit, n_miss, n_total = run_length_histogram(seq)

    print(f"ckpt={args.ckpt}")
    print(f"total frames(GT 있는)={n_total}  hit={n_hit} ({100*n_hit/n_total:.1f}%)  "
          f"miss={n_miss} ({100*n_miss/n_total:.1f}%)")
    print("miss run length histogram (length: n_occurrences, 그 길이가 차지하는 총 miss 프레임 수):")
    for length in sorted(runs):
        occ = runs[length]
        print(f"  length={length:3d}: {occ:4d}회 발생, 총 {occ*length:4d} miss 프레임")
    isolated = runs.get(1, 0)
    isolated_frames = isolated * 1
    total_miss_frames = sum(k * v for k, v in runs.items())
    frac_isolated = isolated_frames / total_miss_frames if total_miss_frames else 0.0
    print(f"-> 전체 miss 프레임 중 '고립된(앞뒤가 hit인) 1프레임짜리' 비율: {100*frac_isolated:.1f}%")
    print("   (이 비율이 높을수록 Bayesian temporal fusion이 recall 복구에 도움될 여지가 큼)")


if __name__ == "__main__":
    main()
