"""make_val_subsample.py - 캐시 manifest에 'val_sub'(고정 val 서브샘플) split을 추가한다.

학습 중 best 선정·추세 모니터링용 val을 scene은 전부 유지하되 scene당 프레임을 균등
간격으로 솎아 (기본 1/3) 만든다. scene을 빼지 않아 range/라벨러 대표성은 유지되고,
scene 내 프레임은 매우 중복적(연속 프레임)이라 정보 손실이 작다. 고정 서브셋이라 매
epoch 동일 프레임 → epoch 간 상대 비교(best 선정)가 유효하다. 최종 test는 이걸 안 쓰고
항상 full val/test로 평가한다.

사용: python make_val_subsample.py --cache-root <dir> --every 3
manifest.json의 'val_sub' 키를 새로 채운다(기존 train/val/test는 안 건드림).
"""
import argparse, json, collections
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--every", type=int, default=3, help="scene당 이 간격마다 1프레임 유지")
    ap.add_argument("--src-split", default="val")
    args = ap.parse_args()

    mpath = Path(args.cache_root) / "manifest.json"
    m = json.load(open(mpath))
    by_scene = collections.OrderedDict()
    for rel in m[args.src_split]:
        by_scene.setdefault(rel.split("/")[0], []).append(rel)

    sub = []
    for scene, rels in by_scene.items():
        rels_sorted = sorted(rels)
        kept = rels_sorted[:: args.every]
        if not kept:  # 아주 작은 scene도 최소 1개 유지
            kept = rels_sorted[:1]
        sub.extend(kept)

    m["val_sub"] = sub
    json.dump(m, open(mpath, "w"), indent=2)
    print(f"val: {len(m[args.src_split])} -> val_sub: {len(sub)} 프레임 "
          f"({len(by_scene)} scenes 전부 유지, every={args.every})")


if __name__ == "__main__":
    main()
