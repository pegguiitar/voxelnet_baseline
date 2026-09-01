"""analyze_static_scenes.py - 정지(거의 움직이지 않는) 다이버 구간 탐지. Day2/사이드
트랙(project_postmeeting_5day_sprint_plan §4) 첫 단계 - 팀원이 겪은 문제(raw point
시간축 병합 시 움직이는 다이버는 팔다리가 뭉개짐)가 정지 구간에서는 안 생기므로,
이 구간의 다중 프레임을 누적해 "공짜 dense supervision"으로 쓸 수 있는지 확인하기
위해 먼저 그런 구간이 실제로 얼마나 있는지 정량화한다.

방법: link_id로 같은 객체를 프레임 간 추적, 연속(또는 1~2프레임 간격) 라벨된 프레임
사이의 centroid 이동거리가 threshold 미만이면 "정지"로 표시, 연속된 정지 프레임을
하나의 구간(segment)으로 묶는다.

Usage:
    python analyze_static_scenes.py [--threshold 0.1] [--max-frame-gap 2]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Triband_BEV" / "baseline"))
import common  # noqa: E402
import filter_outliers  # noqa: E402


def centroid_of(obj: dict) -> np.ndarray:
    c = obj["centroid"]
    return np.array([c["x"], c["y"], c["z"]])


def analyze_scene(scene_id: str, flagged_keys, threshold: float, max_frame_gap: int):
    """반환: list of dict {link_id, start_frame, end_frame, n_frames, max_step_disp}."""
    # link_id별로 (frame_idx, centroid) 시퀀스를 모음
    tracks = {}  # link_id -> list of (frame_idx, centroid)
    for frame_idx, objects in filter_outliers.iter_filtered_frames(scene_id, flagged_keys):
        for o in objects:
            if not o.get("class", "").endswith("-sonar"):
                continue
            link_id = o.get("link_id")
            if link_id is None:
                continue
            tracks.setdefault(link_id, []).append((frame_idx, centroid_of(o)))

    segments = []
    for link_id, seq in tracks.items():
        seq.sort(key=lambda t: t[0])
        seg_start_idx = 0
        for i in range(1, len(seq) + 1):
            broke = False
            if i < len(seq):
                prev_frame, prev_c = seq[i - 1]
                cur_frame, cur_c = seq[i]
                gap = cur_frame - prev_frame
                step_disp = float(np.linalg.norm(cur_c - prev_c))
                # 누적 표류(slow drift) 방지: 매 스텝은 threshold 이하여도, 구간 시작점
                # 대비 지금까지의 총 이동량이 threshold를 넘으면 더 이상 "정지"가 아님 -
                # 예를 들어 매 프레임 9cm씩 계속 같은 방향으로 움직이면 스텝은 통과하지만
                # 10프레임 뒤엔 90cm 이동한 것이므로 이건 잡아야 함.
                seg_start_c = seq[seg_start_idx][1]
                drift_from_start = float(np.linalg.norm(cur_c - seg_start_c))
                if gap > max_frame_gap or step_disp > threshold or drift_from_start > threshold:
                    broke = True
            else:
                broke = True  # 시퀀스 끝
            if broke:
                seg = seq[seg_start_idx:i]
                if len(seg) >= 3:  # 최소 3프레임은 돼야 "구간"으로 의미 있음
                    disps = [float(np.linalg.norm(seg[j + 1][1] - seg[j][1])) for j in range(len(seg) - 1)]
                    centroids = np.stack([c for _, c in seg])
                    drift_from_start_all = np.linalg.norm(centroids - centroids[0], axis=1)
                    segments.append({
                        "link_id": link_id,
                        "start_frame": seg[0][0], "end_frame": seg[-1][0],
                        "n_frames": len(seg),
                        "trajectory": [{"frame": f, "x": float(c[0]), "y": float(c[1]), "z": float(c[2])}
                                       for f, c in seg],
                        "max_drift_from_start": float(drift_from_start_all.max()),
                        "max_step_disp": max(disps) if disps else 0.0,
                        "mean_step_disp": float(np.mean(disps)) if disps else 0.0,
                    })
                seg_start_idx = i
    return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.1,
                         help="프레임간 centroid 이동거리(m) 이 값 미만이면 '정지'로 취급")
    parser.add_argument("--max-frame-gap", type=int, default=2,
                         help="이 이상 프레임 번호가 벌어지면 연속으로 안 침(라벨 누락 등)")
    args = parser.parse_args()

    flagged_keys = filter_outliers.load_flagged_keys()
    scenes = common.list_scenes(include_excluded=False)

    all_segments = {}
    total_static_frames = 0
    total_labeled_frames = 0

    for scene_id in scenes:
        segs = analyze_scene(scene_id, flagged_keys, args.threshold, args.max_frame_gap)
        n_labeled = sum(1 for _ in filter_outliers.iter_filtered_frames(scene_id, flagged_keys))
        total_labeled_frames += n_labeled
        # link_id 2개 이상이 겹치는 프레임 구간에서 동시에 정지해 있으면 단순 합계는
        # 프레임을 중복으로 센다 - 실제 "정지 상태인 고유 프레임 수"는 프레임 인덱스
        # 집합의 합집합으로 구해야 한다.
        scene_static_raw = sum(s["n_frames"] for s in segs)
        frame_set = set()
        for s in segs:
            frame_set.update(range(s["start_frame"], s["end_frame"] + 1))
        scene_static = len(frame_set)
        total_static_frames += scene_static
        all_segments[scene_id] = segs
        if segs:
            dup = scene_static_raw - scene_static
            dup_note = f", link_id 중복 {dup}개 제외" if dup > 0 else ""
            print(f"{scene_id}: 라벨 {n_labeled}프레임 중 정지구간 {len(segs)}개, "
                  f"정지 프레임(중복제거) {scene_static}개 ({100*scene_static/max(n_labeled,1):.1f}%{dup_note})")
            for s in sorted(segs, key=lambda x: -x["n_frames"])[:3]:
                print(f"    link_id={s['link_id']} frame[{s['start_frame']}-{s['end_frame']}] "
                      f"n={s['n_frames']} mean_step={s['mean_step_disp']*100:.1f}cm max_step={s['max_step_disp']*100:.1f}cm")
        else:
            print(f"{scene_id}: 라벨 {n_labeled}프레임, 정지구간 없음")

    print(f"\n=== 전체 요약 (threshold={args.threshold}m, max_frame_gap={args.max_frame_gap}) ===")
    print(f"총 라벨 프레임: {total_labeled_frames}, 정지 구간에 속한 프레임: {total_static_frames} "
          f"({100*total_static_frames/max(total_labeled_frames,1):.1f}%)")
    n_segs = sum(len(v) for v in all_segments.values())
    seg_lens = [s["n_frames"] for segs in all_segments.values() for s in segs]
    if seg_lens:
        print(f"정지 구간 개수: {n_segs}, 구간 길이(프레임) 평균={np.mean(seg_lens):.1f} "
              f"중앙값={np.median(seg_lens):.0f} 최대={max(seg_lens)}")

    out_path = Path(__file__).resolve().parent.parent / "reports" / "static_scene_segments.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"threshold_m": args.threshold, "max_frame_gap": args.max_frame_gap,
                   "total_labeled_frames": total_labeled_frames, "total_static_frames": total_static_frames,
                   "segments_by_scene": all_segments}, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
