# APRILab detection-head kit — PointPillars 비교용

우리(SonarVoxNet) Sec 6.2 ablation의 **세 head config를 PointPillars 인코더+backbone에
그대로 붙여** 공정 비교(head-only swap)할 수 있게 뽑아낸 self-contained kit.

- `aprilab_heads/` — 필요한 소스 전부 (우리 repo에서 그대로 복사 + `heads.py` 분리본)
- `example_usage.py` — 세 config를 target→loss→decode까지 도는 실행 가능한 최소 예제
- 외부 의존성: `numpy`, `torch`, `shapely`만.

```bash
python example_usage.py   # 더미 데이터로 세 경로가 다 도는지 검증
```

---

## 1. 세 config는 무엇이 다른가 (핵심)

| # | 이름 | head 모듈 | rotation TARGET | loss | decode |
|---|------|-----------|-----------------|------|--------|
| 1 | **anchor** | `AnchorHead` | z-yaw only | `loss.voxelnet_loss` | `decode.decode_boxes` + `rotated_nms` |
| 2 | **center** (CenterPoint baseline) | `CenterHead` | z-yaw only | `center_loss.center_voxelnet_loss` | `heatmap_targets.decode_center_boxes` |
| 3 | **ours** | `CenterHead` | **full 3D** | `center_loss.center_voxelnet_loss` | `heatmap_targets.decode_center_boxes` |

> **가장 중요한 점**: config 2(center)와 3(ours)의 **head 코드는 완전히 동일**하다
> (둘 다 `CenterHead`, rot_head 6채널=6D). 유일한 차이는 **rotation target을 만드는 방식**:
> - z-yaw only : GT의 `rotation_x`/`rotation_y`를 0으로 밀어 6D 인코딩 → `heads.zyaw_flatten(objects)`로 감싼 뒤 target 생성
> - full 3D    : GT를 그대로 6D 인코딩 → objects 그대로
>
> anchor(config 1)도 rotation은 항상 6D(reg 12채널 = xyz3 + dim3 + rot6). config 1은
> 우리 ablation에선 z-yaw target으로 돌렸으므로 위 표도 z-yaw로 표기했다.
>
> 즉 **우리 기여의 본질은 새 head 아키텍처가 아니라 "BEV-yaw 관례 → full-3D 회전 확장"**
> 이다(같은 center head, target만 full-3D). 이 kit도 그 사실을 그대로 반영한다.
> config 2 vs 3 비교가 곧 우리 기여의 크기다.

---

## 2. BEV 인터페이스 계약 (PointPillars 쪽을 여기에 맞출 것)

head는 `(B, C, H, W)` BEV feature map을 바로 먹는다. **반드시** 아래 grid를 맞출 것:

| 항목 | 값 | 출처 |
|---|---|---|
| 예측 해상도 (H, W) | **(50, 60)** (row=y, col=x) | `config.ANCHOR_GRID_SIZE[::-1]` |
| cell 크기 | **0.2 m** | `config.ANCHOR_STRIDE` |
| x 범위 (W축, 60셀) | **[0, 12] m** | `config.POINT_CLOUD_RANGE` |
| y 범위 (H축, 50셀) | **[-5, 5] m** | `config.POINT_CLOUD_RANGE` |
| z 범위 | [-2.5, 2.5] m | `config.POINT_CLOUD_RANGE` |
| 채널 C | 자유 | head를 `in_channels=C`로 생성 |

PointPillars의 pillar grid + 2D backbone stride를 조합해 **최종 BEV가 50×60 @0.2m**가
되도록 설정하면, target 생성/decode 좌표계가 그대로 맞아 코드 수정 없이 동작한다.
(예: pillar 0.1m + backbone stride 2, 또는 pillar 0.2m + stride 1 등. 물리 원점은
x0=0, y0=-5.) 채널 C는 무엇이든 되고, head가 `nn.Conv2d(C, ...)`로 받는다.

> 만약 PointPillars 해상도를 우리와 다르게 두고 싶으면, `config.py`의
> `POINT_CLOUD_RANGE`/`VOXEL_SIZE`/`ANCHOR_STRIDE`를 그쪽에 맞춰 바꾸면 target/decode가
> 따라간다(단, 세 config 사이에선 반드시 동일 grid를 쓸 것 — 그래야 공정 비교).

---

## 3. 라벨(object) 스키마

`build_heatmap_targets` / `assign_targets`가 먹는 GT dict 형식:

```python
{
  "centroid":   {"x": float, "y": float, "z": float},        # m
  "dimensions": {"length": float, "width": float, "height": float},  # m (length=x, width=y, height=z)
  "rotation_x": float, "rotation_y": float, "rotation_z": float,     # rad (Euler, Rz·Ry·Rx)
}
```

`points`: `(N, >=3)` numpy, [x,y,z,...]. (density 보조 라벨 계산에만 쓰이며 base 비교엔
불필요 — 없으면 빈 배열 넘겨도 됨. 회전 오일러→행렬은 `rotation3d.euler_to_matrix`.)

---

## 4. 각 config 호출 순서 (shape 포함)

### config 2·3 — center / ours (`CenterHead`)
```python
head = heads.CenterHead(in_channels=C, rotation_repr="6d")
hm, off, z, dim, rot = head(feat)          # feat:(B,C,50,60)
# hm(B,1,H,W) off(B,2,H,W) z(B,1,H,W) dim(B,3,H,W) rot(B,6,H,W)

# target (per-frame, numpy) — ours면 objects 그대로, center면 heads.zyaw_flatten(objects)
t = build_heatmap_targets(objects_or_flattened, points, rotation_repr="6d")
#   t["heatmap"](1,H,W) t["reg_mask"](H,W) t["offset"](H,W,2)
#   t["z"](H,W,1) t["dim"](H,W,3) t["rot"](H,W,6)
# -> collate로 (B,...) stack 후 tensor화

loss, stats = center_voxelnet_loss(hm, off, z, dim, rot,
    heatmap, reg_mask, offset, z_center, dim_center, rot_center,  # (B,...) targets
    rotation_repr="6d")   # rotation_loss_mode 기본 "l1_target"

boxes = decode_center_boxes(sigmoid(hm)_np, off_np, z_np, dim_np, rot_np,
                            score_thresh=0.3, rotation_repr="6d")  # per-frame (H,W,C) numpy
```

### config 1 — anchor (`AnchorHead`)
```python
head = heads.AnchorHead(in_channels=C)     # A = len(config.ANCHOR_ROTATIONS) = 2
cls, reg = head(feat)                       # cls(B,A,H,W)  reg(B,A*12,H,W)

grid = build_anchor_grid()                  # (H,W,A,7)
cls_lab, reg_tgt = assign_targets(grid, objects)   # (H,W,A), (H,W,A,12)
# ⚠️ GOTCHA: loss는 (B,A,H,W)/(B,A,H,W,12) 레이아웃 기대 → 반드시 permute:
cls_lab = np.transpose(cls_lab, (2,0,1))          # (A,H,W)
reg_tgt = np.transpose(reg_tgt, (2,0,1,3))        # (A,H,W,12)

loss, stats = voxelnet_loss(cls, reg, cls_lab_b, reg_tgt_b)   # (B,...) 배치

boxes = decode_boxes(cls[b], reg[b], grid, score_thresh=0.3)  # per-sample (batch dim 제거)
kept  = rotated_nms(boxes, iou_thresh=0.1)
```

---

## 5. 반드시 지킬 것 / 주의

- **focal bias 초기화**: `CenterHead`가 `heatmap_head.bias = -log((1-0.1)/0.1)`로 초기화한다.
  빼면 heatmap이 "전부 배경"으로 붕괴해 아무것도 검출 못 한다. (원 이유는 `heads.py` 주석 참고.)
- **공정 비교 조건**: 세 config가 **같은 BEV grid + 같은 학습 recipe**를 써야 한다. 우리 확정
  recipe: AdamW / lr 0.001 / weight_decay 0.01 / OneCycle / momentum-cycling 0.95↔0.85 /
  batch 4 / 20 epoch. anchor의 IoU 매칭 임계는 `config.POS_IOU_THRESH=0.6`,`NEG_IOU_THRESH=0.45`.
- **anchor 크기/회전**: `config.ANCHOR_SIZE=(1.57,1.02,1.13)`, `ANCHOR_ROTATIONS=(0, π/2)`,
  `ANCHOR_Z_CENTER=0.12` — 우리 데이터 실측값. PointPillars에서도 동일하게 둘 것.
- **평가 지표**: AP는 3D OBB IoU 기반. rotation 품질은 yaw-only AOE만 보지 말고 **tilt/
  full-3D geodesic 오차**도 볼 것(config 2 vs 3의 기여가 거기서 드러난다). 우리
  `eval_extra_metrics.py`의 `tilt_deg`/`AOE3D_deg` 참고(이 kit엔 미포함, 필요하면 요청).

## 6. 이 kit에서 뺀 것 (base 비교엔 불필요)
우리 `RPNCenterHead`의 부가 branch — density-aux(RAANet), foreground gate(Direction 3),
stage-2 refinement — 는 뺐다. 셋 다 opt-in이고 세 config 공정 비교 축과 직교하므로,
넣으면 오히려 비교를 흐린다. 필요하면 별도 제공 가능.

## 파일 대응
| 파일 | 역할 |
|---|---|
| `heads.py` | `AnchorHead`, `CenterHead`(분리본) + `zyaw_flatten` |
| `config.py` | grid/voxel/anchor/IoU 임계 등 상수 |
| `rotation3d.py` | 6D(및 quat/axisangle) 인코딩·디코딩, Gram-Schmidt, euler↔matrix |
| `anchors.py` | `build_anchor_grid`, `assign_targets`, `gt_boxes_from_objects` |
| `heatmap_targets.py` | `build_heatmap_targets`(center target) + `decode_center_boxes`(center decode) |
| `center_loss.py` | `center_voxelnet_loss` (center head loss) |
| `loss.py` | `voxelnet_loss` (anchor head loss, focal + smooth-L1) |
| `decode.py` | `decode_boxes`(anchor decode) + `rotated_nms` |
