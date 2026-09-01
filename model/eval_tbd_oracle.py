"""eval_tbd_oracle.py - TBD 오라클 harness (무료 GO/NO-GO 게이트).

기존 detector(체크포인트)의 test detection을 scene별 프레임 순서로 뽑아(낮은 threshold),
tbd_oracle(KF+존재 log-odds) 후처리를 씌우고, raw vs oracle의 AP3D를 같은 매칭으로 비교한다.
temporal 일관성이 FP↓/원거리 recall↑를 주는지 학습 0원으로 판정.

reports_v2/8_31_roadmap.html 부록 A 스펙 구현. eval_voxelnet.py 기계(decode/IoU/compute_ap)를 재사용.

사용(로컬):
  python eval_tbd_oracle.py --ckpt <path> --cache-root <data_final/cache_strong/voxel> \
      --split test --decode-thresh 0.1 [--limit-scenes 2]
"""
import argparse, json, collections
import numpy as np
import torch

import config
import heatmap_targets as ht
from model import VoxelNet, load_state_dict_compat
import eval_voxelnet as E   # gt_obb_from_row, pred_obb_from_box, iou_3d_obb, compute_ap, IOU_THRESHOLDS, _decode_center_sample, rotated_nms
import tbd_oracle


def _scene_frame(rel):
    # "scene_0043/frame_000012.npz" -> ("scene_0043", 12)
    sc, fr = rel.split("/")
    return sc, int(fr.replace("frame_", "").replace(".npz", ""))


def _range_bucket(cx, cy):
    r = float(np.hypot(cx, cy))
    for lo, hi, name in [(0,2,"0-2m"),(2,3.5,"2-3.5m"),(3.5,5,"3.5-5m"),(5,99,"5m+")]:
        if lo <= r < hi:
            return name
    return "5m+"


def _match(cands, gt_boxes, score_key, nms_iou):
    """cands(box dict)를 score_key로 필터/정렬 -> NMS -> IoU 매칭.
    반환: {iou_thr: [(score, is_tp), ...]} + per-GT recall용 (range, matched@0.4) 리스트."""
    filtered = [b for b in cands if b.get(score_key, b["score"]) >= 0]  # 전부(이미 decode_thresh 적용됨)
    # NMS는 원 score 기준으로(기하 동일), 하지만 순위/AP는 score_key로
    filtered = E.rotated_nms(filtered, iou_thresh=nms_iou)
    gt_list = [E.gt_obb_from_row(row) for row in gt_boxes]
    pred_list = [E.pred_obb_from_box(b) for b in filtered]
    conf = [b.get(score_key, b["score"]) for b in filtered]
    n_p, n_g = len(pred_list), len(gt_list)
    iou_mat = np.zeros((n_p, n_g))
    rng = np.random.default_rng(0)
    for pi,(pc,pd,pR) in enumerate(pred_list):
        for gi,(gc,gd,gR) in enumerate(gt_list):
            iou_mat[pi,gi] = E.iou_3d_obb(pc,pd,pR,gc,gd,gR,rng=rng)
    order = np.argsort(-np.array(conf)) if n_p else np.array([],dtype=int)
    out = {it: [] for it in E.IOU_THRESHOLDS}
    gt_matched_04 = [False]*n_g  # 원거리 recall(0.4)용
    for it in E.IOU_THRESHOLDS:
        matched=set()
        for pi in order:
            cg=[gi for gi in range(n_g) if gi not in matched]; is_tp=False
            if cg:
                ious=iou_mat[pi,cg]; bl=int(np.argmax(ious)); bg,bi=cg[bl],ious[bl]
                if bi>=it:
                    matched.add(bg); is_tp=True
                    if abs(it-0.4)<1e-9: gt_matched_04[bg]=True
            out[it].append((conf[pi], is_tp))
    gt_info=[(E.gt_obb_from_row(row)[0], gt_matched_04[gi]) for gi,row in enumerate(gt_boxes)]  # (center, matched)
    return out, gt_info


@torch.no_grad()
def run(ckpt_path, cache_root, split, decode_thresh, nms_iou, device, limit_scenes=None, limit_frames=None):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = VoxelNet(head="center").to(device)
    load_state_dict_compat(model, ckpt["model"]); model.eval()

    manifest = json.load(open(f"{cache_root}/manifest.json"))
    entries = [e for e in manifest[split] if "_augS" not in e and "_aug1" not in e]
    by_scene = collections.OrderedDict()
    for rel in entries:
        sc,fr=_scene_frame(rel); by_scene.setdefault(sc,[]).append((fr,rel))
    scenes=list(by_scene.keys())
    if limit_scenes: scenes=scenes[:limit_scenes]

    # 프레임별 detection + gt를 scene별 순서로 수집
    det_raw={it:[] for it in E.IOU_THRESHOLDS}; det_tbd={it:[] for it in E.IOU_THRESHOLDS}
    n_gt=0
    recall_raw=collections.Counter(); recall_tbd=collections.Counter(); gt_by_range=collections.Counter()
    empty_raw=[]; empty_tbd=[]; n_empty_frames=0   # test_empty(0-GT) 프레임의 post-NMS detection score
    for si,sc in enumerate(scenes):
        frames_sorted=sorted(by_scene[sc])
        if limit_frames: frames_sorted=frames_sorted[:limit_frames]
        per_frame_cands=[]; per_frame_gt=[]
        for fr,rel in frames_sorted:
            with np.load(f"{cache_root}/{rel}") as npz:
                vf,co,npn,gtb=npz["voxel_features"],npz["coords"],npz["num_points"],npz["gt_boxes"]
            if vf.shape[0]==0:  # 빈-voxel skip
                per_frame_cands.append([]); per_frame_gt.append(gtb); continue
            vt=torch.from_numpy(vf).to(device); nt=torch.from_numpy(npn).to(device)
            bc=torch.zeros((len(co),1),dtype=torch.int64); ct=torch.cat([bc,torch.from_numpy(co)],1).to(device)
            hm,off,z,dim,rot,_=model(vt,nt,ct)
            cands=E._decode_center_sample(hm[0],off[0],z[0],dim[0],rot[0],score_thresh=decode_thresh)
            per_frame_cands.append(cands); per_frame_gt.append(gtb)
        # tbd 오라클용 detection 포맷으로 변환
        tbd_frames=[]
        for cands in per_frame_cands:
            fr_dets=[]
            for b in cands:
                c,d,R=E.pred_obb_from_box(b)
                fr_dets.append({"center":tuple(c),"dims":tuple(d),"R":R,"score":float(b["score"])})
            tbd_frames.append(fr_dets)
        tbd_out=tbd_oracle.run_scene(tbd_frames)
        # score_tbd를 원 cands에 주입
        for cands,tdets in zip(per_frame_cands,tbd_out):
            for b,td in zip(cands,tdets): b["score_tbd"]=td["score_tbd"]
        # 프레임별 매칭(raw & tbd), 누적
        for cands,gtb in zip(per_frame_cands,per_frame_gt):
            n_gt+=len(gtb)
            if len(gtb)==0:  # 순수 배경 프레임: 모든 post-NMS detection = FP
                n_empty_frames+=1
                for b in E.rotated_nms(cands, iou_thresh=nms_iou):
                    empty_raw.append(float(b["score"])); empty_tbd.append(float(b.get("score_tbd",b["score"])))
            mr,gi_r=_match(cands,gtb,"score",nms_iou)
            mt,gi_t=_match(cands,gtb,"score_tbd",nms_iou)
            for it in E.IOU_THRESHOLDS: det_raw[it]+=mr[it]; det_tbd[it]+=mt[it]
            for (c,mtch) in gi_r:
                rb=_range_bucket(c[0],c[1]); gt_by_range[rb]+=1
                if mtch: recall_raw[rb]+=1
            for (c,mtch) in gi_t:
                rb=_range_bucket(c[0],c[1])
                if mtch: recall_tbd[rb]+=1
        print(f"[{si+1}/{len(scenes)}] {sc} done (frames={len(frames_sorted)})", flush=True)

    # AP3D raw vs tbd
    print("\n=== AP3D: raw vs TBD-oracle (n_gt=%d) ==="%n_gt)
    print(f"{'IoU':>5} | {'raw':>8} | {'oracle':>8} | {'Δ':>7}")
    for it in E.IOU_THRESHOLDS:
        ar,_,_=E.compute_ap(det_raw[it],n_gt); at,_,_=E.compute_ap(det_tbd[it],n_gt)
        print(f"{it:>5} | {ar:8.4f} | {at:8.4f} | {at-ar:+7.4f}")
    print("\n=== range별 recall@0.4 (all-candidate, matched GT / GT) ===")
    for rb in ["0-2m","2-3.5m","3.5-5m","5m+"]:
        g=gt_by_range[rb]
        if g==0: continue
        print(f"  {rb:>7}: raw {recall_raw[rb]/g:.3f}  oracle {recall_tbd[rb]/g:.3f}  (GT={g})")

    # ===== operating-point 지표 (오라클의 진짜 가치: 배포 threshold에서 FP↓/recall↑) =====
    def _pr(dets):
        if not dets: return np.array([]),np.array([]),np.array([])
        c=np.array([d[0] for d in dets]); tp=np.array([d[1] for d in dets],dtype=bool)
        o=np.argsort(-c); c=c[o]; tp=tp[o]
        tpc=np.cumsum(tp); fpc=np.cumsum(~tp)
        return c, tpc/max(n_gt,1), tpc/np.maximum(tpc+fpc,1)  # conf, recall, precision (모두 desc-score 순)
    def _p_at_r(c,r,p,tgt):
        if len(r)==0: return 0.0,1.0
        i=int(np.searchsorted(r,tgt))
        if i>=len(r): return 0.0,float(c[-1])
        return float(p[i]),float(c[i])   # precision, threshold@recall=tgt
    def _f1max(c,r,p):
        if len(r)==0: return 0.0,0.0,0.0,1.0
        f1=2*p*r/np.maximum(p+r,1e-9); i=int(np.argmax(f1))
        return float(f1[i]),float(r[i]),float(p[i]),float(c[i])
    cr,rr,pr=_pr(det_raw[0.4]); ct,rt,pt=_pr(det_tbd[0.4])
    print("\n=== ★operating-point @ IoU0.4 (score-scale 무관, 같은 recall에서 precision 비교)★ ===")
    print(f"{'target R':>9} | {'raw P':>7} | {'oracle P':>8} | {'Δ':>7}")
    for tgt in [0.5,0.6,0.7]:   # 모델 max recall~0.78이라 achievable 범위로
        p_r,_=_p_at_r(cr,rr,pr,tgt); p_t,_=_p_at_r(ct,rt,pt,tgt)
        print(f"{tgt:>9.2f} | {p_r:7.3f} | {p_t:8.3f} | {p_t-p_r:+7.3f}")
    f1r=_f1max(cr,rr,pr); f1t=_f1max(ct,rt,pt)
    print(f"F1-max:  raw {f1r[0]:.3f}(P{f1r[2]:.2f}/R{f1r[1]:.2f})  oracle {f1t[0]:.3f}(P{f1t[2]:.2f}/R{f1t[1]:.2f})  Δ={f1t[0]-f1r[0]:+.3f}")
    # ★empty 배경 프레임 FP 억제★: 각 방법의 F1-max threshold(항상 정의됨)에서 배경 detection 수
    thr_r=f1r[3]; thr_t=f1t[3]
    fp_r=sum(1 for s in empty_raw if s>=thr_r); fp_t=sum(1 for s in empty_tbd if s>=thr_t)
    print(f"\n=== ★empty(배경) 프레임 FP @ F1-max operating point★ (n_empty_frames={n_empty_frames}) ===")
    print(f"  raw:    {fp_r} FP  (thr={thr_r:.3f})   →  {fp_r/max(n_empty_frames,1):.3f} FP/frame")
    print(f"  oracle: {fp_t} FP  (thr={thr_t:.3f})   →  {fp_t/max(n_empty_frames,1):.3f} FP/frame")
    print(f"  FP 감소: {fp_r-fp_t} ({(fp_r-fp_t)/max(fp_r,1)*100:+.1f}%)")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt",required=True); ap.add_argument("--cache-root",required=True)
    ap.add_argument("--split",default="test"); ap.add_argument("--decode-thresh",type=float,default=0.1)
    ap.add_argument("--nms-iou",type=float,default=0.2); ap.add_argument("--device",default="cpu")
    ap.add_argument("--limit-scenes",type=int,default=None)
    ap.add_argument("--limit-frames",type=int,default=None)
    a=ap.parse_args()
    run(a.ckpt,a.cache_root,a.split,a.decode_thresh,a.nms_iou,a.device,a.limit_scenes,a.limit_frames)


if __name__=="__main__":
    main()
