"""prepare_labels_final.py - labels_final(최종 검수 라벨) → data_final 파이프라인 1단계.

(A) 변환: labels_final/*.json -> data_final/annotations/*.json
    각 -sonar 객체에 rotation_x/y/z = rotations.x/y/z (단순 복사, deg) 추가.
    (revised_labels->data_revised/annotations 변환과 동일 확인됨: 8253객체 불일치 0)
    empty(0-객체) 프레임도 그대로 보존한다 - 캐시 빌드 단계에서 pos/full 리스트로 갈린다.
(C) split: data_final/reports/splits.json
    기존 16/7/8 배정 유지 + scene_0081(신규, GT 있음) -> train,
    빈 4 scene(0036/0070/0074/0076, 순수 negative) -> train.
    => train 21 / val 7 / test 8 = 36 scene.
(B) stats: scene별 frames/nonEmpty/GT 집계 출력.

empty 프레임 정책: 표준 detector처럼 GT-없는(포인트-있는) 프레임을 negative supervision으로
쓴다. 캐시 빌드(cache 단계)가 이들을 포함해 {split}_full 리스트를 만들고, {split}는 non-empty만
남겨 backward-compat + test empty 포함 여부를 --split override로 고를 수 있게 한다.

사용: python prepare_labels_final.py
"""
import json, glob, os
from pathlib import Path

ROOT = Path("/home/eugene/Data/APRILab_baseline")
SRC = ROOT / "labels_final"
OUT_ANN = ROOT / "data_final" / "annotations"
OUT_REPORTS = ROOT / "data_final" / "reports"

# 기존 data_revised split 배정(16/7/8) 재사용 + 신규 5개 train으로.
OLD_TRAIN = ["scene_0000", "scene_0021", "scene_0022", "scene_0042", "scene_0044",
             "scene_0048", "scene_0050", "scene_0063", "scene_0065", "scene_0068",
             "scene_0072", "scene_0078", "scene_0079", "scene_0080", "scene_0082", "scene_0091"]
VAL = ["scene_0009", "scene_0019", "scene_0049", "scene_0064", "scene_0075", "scene_0090", "scene_0092"]
TEST = ["scene_0035", "scene_0043", "scene_0055", "scene_0067", "scene_0077", "scene_0088", "scene_0089", "scene_0093"]
NEW_TO_TRAIN = ["scene_0081", "scene_0036", "scene_0070", "scene_0074", "scene_0076"]  # 0081=GT있음, 나머지=순수neg
TRAIN = OLD_TRAIN + NEW_TO_TRAIN


def is_sonar(o):
    return str(o.get("class", "")).endswith("-sonar")


def convert():
    OUT_ANN.mkdir(parents=True, exist_ok=True)
    n_obj = 0
    for f in sorted(glob.glob(str(SRC / "*.json"))):
        d = json.load(open(f))
        for fid, fd in d.get("frames", {}).items():
            for o in fd.get("objects", []):
                r = o.get("rotations")
                if r is not None:
                    o["rotation_x"] = r.get("x", 0.0)
                    o["rotation_y"] = r.get("y", 0.0)
                    o["rotation_z"] = r.get("z", 0.0)
                    n_obj += 1
        json.dump(d, open(OUT_ANN / os.path.basename(f), "w"))
    return n_obj


def write_split():
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    # data_revised와 동일 구조: {"split": {...}}
    split = {"split": {"train": TRAIN, "val": VAL, "test": TEST}}
    json.dump(split, open(OUT_REPORTS / "splits.json", "w"), indent=2)
    # 빈 flagged (cache_revised의 filter_outliers가 기대)
    json.dump({}, open(OUT_REPORTS / "flagged_frames.json", "w"))


def stats():
    assign = {}
    for s in TRAIN: assign[s] = "train"
    for s in VAL: assign[s] = "val"
    for s in TEST: assign[s] = "test"
    agg = {sp: {"scenes": 0, "frames": 0, "nonEmpty": 0, "empty": 0, "gt": 0} for sp in ["train", "val", "test"]}
    for f in sorted(glob.glob(str(OUT_ANN / "*.json"))):
        sc = os.path.basename(f)[:-5]
        sp = assign.get(sc)
        if sp is None:
            print("  WARN unassigned:", sc); continue
        d = json.load(open(f))
        a = agg[sp]; a["scenes"] += 1
        for fid, fd in d.get("frames", {}).items():
            objs = [o for o in fd.get("objects", []) if is_sonar(o)]
            a["frames"] += 1
            if objs:
                a["nonEmpty"] += 1; a["gt"] += len(objs)
            else:
                a["empty"] += 1
    print(f"{'split':<7}{'scenes':>7}{'frames':>8}{'nonEmpty':>9}{'empty':>7}{'GT':>7}{'GT/nonEmpty':>12}")
    for sp in ["train", "val", "test"]:
        a = agg[sp]
        gpf = a["gt"] / a["nonEmpty"] if a["nonEmpty"] else 0
        print(f"{sp:<7}{a['scenes']:>7}{a['frames']:>8}{a['nonEmpty']:>9}{a['empty']:>7}{a['gt']:>7}{gpf:>12.2f}")
    return agg


if __name__ == "__main__":
    n = convert()
    print(f"[A] 변환 완료: {len(list(OUT_ANN.glob('*.json')))} scene, rotation_x/y/z 추가 {n} 객체 -> {OUT_ANN}")
    write_split()
    print(f"[C] split 작성: train {len(TRAIN)} / val {len(VAL)} / test {len(TEST)} -> {OUT_REPORTS/'splits.json'}")
    print("[B] stats:")
    stats()
