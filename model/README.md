# model/ 사용 가이드 — center head + rotation3d 학습

이 폴더는 여러 달치 실험(polar, GRR, fg_gate, stage2, TBD 오라클 등)이 전부 한 디렉토리에
쌓인 연구용 코드입니다. **확정된 최종 레시피(VoxelNet backbone + CenterPoint center head +
rotation3d)를 학습하는 데 필요한 파일은 그중 일부**입니다. 아래 순서대로 보면 됩니다.

## 0. 먼저: 데이터는 이 repo에 없음

이 repo에는 코드만 있고, 학습에 필요한 라벨(`data_final/annotations`)과 원본 소나
데이터(`uScenes`), 그리고 미리 빌드된 voxel 캐시(`data_final/cache_strong/voxel`)는
포함되어 있지 않습니다(용량 문제로 git에 안 올림). 아래 중 하나가 필요합니다.

- 미리 빌드된 캐시(`data_final/cache_strong/voxel`, 또는 Modal volume `voxelnet-data`의
  `cache_parts_final/`)를 공유받기, 또는
- 원본 라벨+포인트클라우드를 받아서 §2의 캐시 빌드 스크립트로 직접 생성

캐시 없이는 `train.py --cache-root ...`가 작동하지 않으니, 진행 전에 확인하세요.

## 1. 확정 레시피 (2026-09-01 기준)

| 항목 | 값 |
|---|---|
| 아키텍처 | VoxelNet backbone + CenterPoint center head + rotation3d(6D) |
| optimizer | AdamW |
| lr schedule | OneCycle |
| weight decay | 0.01 |
| batch size | 4 |
| epochs | 20 |
| eval IoU | AP3D @ 0.3/0.35/0.4 추적, **0.35가 목표(primary) 지표** |
| seed | 2개 기본(s0, s1), 여유 있으면 s2 |
| val split | `val_full` (empty 프레임 포함 - FP 반영, 업계 표준) |
| train split | `train` (positive-only, empty 프레임 섞지 않음 — 섞으면 AP3D -8~-12%로 확인됨) |

## 2. 파이프라인 핵심 파일 (이것만 보면 됨)

| 파일 | 역할 |
|---|---|
| `config.py` | voxel/grid/anchor/loss 전역 하이퍼파라미터 |
| `voxelize.py` | point cloud → voxel feature 변환 |
| `anchors.py` | anchor grid 빌드, GT box 파싱(캐시 빌드 시 내부적으로 필요, center head도 의존) |
| `heatmap_targets.py` | CenterPoint heatmap + rotation3d 타겟 생성 |
| `rotation3d.py` | 6D rotation 인코딩/디코딩 유틸 |
| `augment.py` | GT-sampling(aspect-preserving 내장) + flip + translation 증강 |
| `cache_dataset.py` | voxel 캐시 빌드 공통 함수 (`cache_final.py`가 이걸 import) |
| `cache_final.py` | **실제로 실행할 캐시 빌드 스크립트** (labels_final/data_final 전용, train/val_full/test/test_full 등 split 생성) |
| `dataset.py` | `CachedVoxelNetDataset` — 캐시에서 배치 로딩 |
| `model.py` | VoxelNet backbone + `RPNCenterHead` (rotation3d 포함) |
| `center_loss.py` | center head loss (focal + rotation3d 포함 회귀) — **이걸 씀** (`loss.py`는 anchor head 전용, 안 씀) |
| `decode.py` | heatmap → 3D box 디코딩 (rotation3d 복원 포함) |
| `train.py` | 학습 엔트리포인트 |
| `eval_voxelnet.py` | AP3D 평가 엔트리포인트 |

> `train.py`는 `foreground_gate.py`, `stage2_refine.py`, `loss.py`, `anchors.py`도
> import합니다(코드 실행에 필요). 다만 아래 커맨드처럼 관련 플래그(`--fg-mode`,
> `--use-stage2` 등)를 안 주면 기능은 꺼진 채로 무시되니 신경 쓸 필요 없습니다.

## 3. 실행 순서

### 3-1. 캐시 빌드 (이미 캐시를 받았다면 생략)

```bash
python cache_final.py --strong-aug --out-cache data_final/cache_strong/voxel
```

### 3-2. 학습

```bash
python train.py \
  --cache-root /path/to/data_final/cache_strong/voxel \
  --head center \
  --split train \
  --val-split val_full \
  --epochs 20 \
  --optimizer adamw \
  --lr-schedule onecycle \
  --weight-decay 0.01 \
  --batch-size 4 \
  --seed 0 \
  --run-name voxelnet_cartesian_s0
```

`--seed 1`(`--run-name voxelnet_cartesian_s1`)로 한 번 더 돌려서 2-seed를 채우세요.
`optimizer`/`lr-schedule`/`batch-size`는 이미 기본값과 같지만, 나중에 코드 기본값이
바뀌어도 레시피가 흔들리지 않도록 명시적으로 넣는 걸 권장합니다.

학습 중 `val_every` epoch마다 val AP3D를 재서 IoU 0.3/0.35/0.4 각 기준으로 best
체크포인트 3개(`{run_name}_best_iou30.pt` / `_iou35.pt` / `_iou40.pt`)를 자동
갱신합니다. **보고/배포에는 `_best_iou35.pt`(primary)를 씁니다.**

### 3-3. 평가

```bash
python eval_voxelnet.py \
  --ckpt checkpoints/voxelnet_cartesian_s0_best_iou35.pt \
  --cache-root /path/to/data_final/cache_strong/voxel \
  --split test \
  --iou-thresholds 0.3 0.35 0.4 0.5
```

실전 지표(빈 프레임 FP까지 반영)를 보려면 `--split test_full`로 한 번 더 돌립니다.

## 4. 안 써도 되는 파일들 (실험/폐기 트랙)

아래는 지금까지의 조사 과정에서 나온 실험 스크립트로, **negative 결론이 났거나 아직
채택 전 단계**입니다. center head + rotation3d 기본 학습에는 필요 없습니다.

- `polar_stream.py`, `polar_ga.py`, `partner_grr.py`, `pdconv.py`, `validate_polar_design.py`
  — polar/GRR 좌표계 실험. Cartesian baseline을 이기지 못해 drop 결정.
- `foreground_gate.py`, `patch_density_target.py` — fg_gate 실험(유일하게 긍정적 신호가
  있었지만 아직 정교화 진행 중, 기본 레시피엔 미포함).
- `stage2_refine.py` — 2단계 rescoring, 순효과 불명확(RNG 교란 이슈 있었음).
- `tbd_oracle.py`, `eval_tbd_oracle.py` — 시간축 Bayesian TBD용 오라클 조사(다음 단계
  후보, 아직 미착수).
- `analyze_*.py`, `dump_fp_pattern.py`, `gather_frame.py`, `smoke_center*.py`,
  `cache_revised.py`, `cache_static_segments.py`, `make_val_subsample.py`,
  `patch_heatmap_radius.py`, `eval_old_vs_new_rotation.py`,
  `eval_range_threshold_calibration.py`, `eval_extra_metrics.py`,
  `eval_voxelnet_by_range.py`, `prepare_labels_final.py`, `compare_models.py`,
  `validate_iou_3d_obb.py`, `validate_rotation3d.py`, `analyze_static_scenes.py`
  — 전부 일회성 분석/검증/라벨준비 스크립트. 라벨(`labels_final`)은 이미 확정되어
  재실행 불필요하고, 나머지는 결과 해석/디버깅용이라 학습 자체엔 안 쓰입니다.

## 5. 체크포인트 이름 규칙

`voxelnet_cartesian_s{seed}` 가 baseline(=center head + rotation3d) 이름입니다.
추가 모듈을 얹을 경우에만 `voxelnet_cartesian_{module}_s{seed}` 식으로 토큰을
붙입니다(예: `voxelnet_cartesian_fggate-joint_s0`). center head/rotation3d는 baseline에
내장되어 있으므로 이름에 별도 토큰을 넣지 않습니다.
