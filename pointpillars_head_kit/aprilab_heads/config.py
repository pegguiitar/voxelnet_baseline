"""config.py - VoxelNet(Zhou&Tuzel 2018) 하이퍼파라미터, 우리 소나 데이터 실측치 기반.

핵심 설계 결정 하나만 남긴다: Z축 voxel 크기(VOXEL_SIZE[2]=0.5)를 X/Y(0.1)의 5배로
잡아서 D'(z방향 voxel grid 크기)가 정확히 10이 되게 했다 - 이러면 논문 Table의 car
config(D'=10)와 conv-middle-layer 3개의 커널/스트라이드/패딩을 그대로 재사용해도
출력이 논문과 똑같이 (128, H', W')로 reshape된다(§3.1 계산: 10->5->3->2, C=64*2=128).
RPN 입력 채널 수(128)가 그대로 맞아떨어지므로 RPN 쪽 채널 설계도 그대로 베낄 수 있다 -
직접 구현하면서 conv-middle 출력 채널을 재유도할 필요가 없어진다.

X/Y/Z range와 anchor 크기는 실측(Triband_BEV 데이터, 120 프레임/1500 라벨 샘플)
기반: GT centroid x[1.4,10.5] y[-3.8,3.1] z[-1.55,1.13], dims 평균 l=1.57 w=1.02 h=1.13.
"""

import os
from pathlib import Path

VOXELNET_ROOT = Path(__file__).resolve().parent.parent
TRIBAND_ROOT = VOXELNET_ROOT.parent / "Triband_BEV"

# point cloud range: (x_min,y_min,z_min,x_max,y_max,z_max), 미터.
# GT 박스(x 1.4~10.5, y -3.8~3.1, z -1.55~1.13)를 여유있게 감싸면서, 먼 배경/노이즈
# 포인트(x up to 16, y +-10.9)는 잘라낸다 - 실측 커버리지 89%(model/config 설계 시
# 확인, 조사 세션 기록).
POINT_CLOUD_RANGE = (0.0, -5.0, -2.5, 12.0, 5.0, 2.5)

# (vx, vy, vz). vz=0.5로 D'=5.0/0.5=10 고정 (위 설명 참고).
# VOXEL_Z 환경변수로 z voxel 크기만 오버라이드 가능(기본 0.5) - z-fine 실험(vz=0.25 → D'=20)
# 전용. 캐시 빌드(cache_final.py)·학습·평가가 전부 config.VOXEL_SIZE/GRID_SIZE를 읽으므로
# 같은 env를 주면 D'가 일관되게 바뀐다. z-fine 백본(ZFineMiddle)은 D'=20 입력을 가정하므로
# --use-zfine 런은 반드시 VOXEL_Z=0.25와 짝지어야 함(model.py에서 assert).
VOXEL_SIZE = (0.1, 0.1, float(os.environ.get("VOXEL_Z", "0.5")))

GRID_SIZE = (  # (W', H', D') = (x, y, z) voxel 개수
    round((POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0]) / VOXEL_SIZE[0]),
    round((POINT_CLOUD_RANGE[4] - POINT_CLOUD_RANGE[1]) / VOXEL_SIZE[1]),
    round((POINT_CLOUD_RANGE[5] - POINT_CLOUD_RANGE[2]) / VOXEL_SIZE[2]),
)

MAX_POINTS_PER_VOXEL = 35  # 논문 car config T=35 그대로 (실측 프레임당 6~9k포인트로 스케일 비슷)
# 실측 non-empty voxel 수: mean 2718, p95 4190, max 4686 (80프레임 샘플, xy 0.1m 기준).
# 여유를 크게 둬서 어떤 프레임도 잘리지 않게 한다.
MAX_VOXELS = 8000

INPUT_FEATURE_DIM = 7  # [x,y,z,intensity, x-vx,y-vy,z-vz] (논문 §2.1.1)

# --- anchor (단일 클래스 "diver", 논문처럼 클래스당 anchor 1종 + 회전 2종) ---
ANCHOR_SIZE = (1.57, 1.02, 1.13)  # (length=x, width=y, height=z), 실측 dims 평균
ANCHOR_Z_CENTER = 0.12  # 실측 centroid z 평균
ANCHOR_ROTATIONS = (0.0, 1.5707963267948966)  # 0, pi/2 라디안 (논문과 동일)

# anchor grid는 RPN 최종 출력 해상도(conv-middle 출력의 1/2)에 맞춘다.
ANCHOR_STRIDE = (VOXEL_SIZE[0] * 2, VOXEL_SIZE[1] * 2)  # (x,y) 미터/셀 = 0.2m
ANCHOR_GRID_SIZE = (GRID_SIZE[0] // 2, GRID_SIZE[1] // 2)  # (W'', H'') = (60, 50)

POS_IOU_THRESH = 0.6
NEG_IOU_THRESH = 0.45

# --- loss ---
# 분류: focal loss (Lin et al. 2017, RetinaNet 표준값). 프레임당 positive anchor가
# 평균 2.49개(0.041%)뿐인 극단적 불균형이라 논문 원안(plain weighted BCE)보다 이쪽을
# 채택 - 근거는 VoxelNet/reports/precision_gap_analysis.html.
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
SMOOTH_L1_BETA = 1.0 / 9.0  # 표준 SmoothL1 default(=1)보다 뾰족하게(작은 잔차 민감) - SECOND/OpenPCDet 관례값

# --- RPN 채널 (논문 Fig.4, conv-middle 출력 128채널 기준) ---
RPN_IN_CHANNELS = 128
RPN_BLOCK_CHANNELS = (128, 128, 256)
RPN_BLOCK_LAYERS = (4, 6, 6)  # 각 block의 총 conv 수 (첫 conv가 stride2 downsample)
RPN_UPSAMPLE_CHANNELS = 256  # 각 deconv 출력 채널 (concat 전)

# --- 학습 (확정 recipe: CenterPoint AdamW/OneCycle - project_final_training_recipe) ---
# ★2026-09-06 정정: 이전엔 SGD 시대 값(LR=0.01/NUM_EPOCHS=30/WD=1e-4)이 남아 있어서, Modal
# wrapper의 --lr None default가 이 값을 조용히 fallback시키는 사고가 반복됐다(quat/axisangle
# 학습 등 최근 실험 전부 실제로는 peak_lr=0.01로 돌아감 - canonical recipe 문서의 lr=0.001과
# 불일치). config 자체를 확정 recipe와 맞춰서 문서·wrapper·config 세 곳이 서로 어긋나던
# 상황을 해소한다. train.py CLI를 명시적으로 override하지 않으면 이제 recipe와 정합하게
# 학습된다. 이전 실험 재현할 땐 CLI에 --lr 0.01 등 명시 지정할 것.
BATCH_SIZE = 4
NUM_EPOCHS = 20
LR = 0.001               # CenterPoint peak (--lr, OneCycle 최대치)
LR_DECAY_EPOCH_FRAC = 0.85  # step schedule 전용 (onecycle에선 무시)
LR_DECAY_FACTOR = 0.1
WEIGHT_DECAY = 0.01      # CenterPoint AdamW wd (이전 1e-4는 SGD-tuned)
# OpenPCDet/mmdet3d류 3D detector 관례값(35) - 원래 이 프로젝트엔 clipping이 전혀 없었는데,
# Day2 range-aware loss(k=2/5) 학습이 초반(warmup 없음, LR=0.01 고정)에 cls_head 가중치가
# 폭주해(weight mean -8→-34~-74, bias 더 깊은 음수로) sigmoid 출력이 입력과 무관하게 거의
# 상수(0.027)로 포화되며 붕괴하는 걸 겪은 뒤 추가함 - range weight로 grid의 88%(r>=r0)에서
# loss 크기가 최대 k*(cap_r-r0)+1배 커지는데, 그 큰 loss가 학습 극초반(파라미터가 아직
# 랜덤이라 가장 불안정한 구간) 그대로 큰 gradient로 들어가 나쁜 영역에 갇힌 것으로 진단.
GRAD_CLIP_NORM = 35.0

CHECKPOINT_DIR = VOXELNET_ROOT / "checkpoints"
RUNS_DIR = VOXELNET_ROOT / "runs"

# --- RAANet(arXiv:2111.09515)식 보조 density-level classification head (attention
# 메커니즘이 아니라 range/density-gradient 문제를 보조 supervision으로 직접 가르치는
# 부분만 채택 - attention 쪽은 이미 기각됨, project_gate23_negative_results_k0_final
# 참고). GT 박스 안 실제 point 개수(point_in_obb)를 3클래스로 분류, positive cell에서만
# CE loss로 감독, 추론 시엔 그냥 안 씀(inference 비용 0). 클래스 경계는 전체 16,512개
# GT 박스의 point 개수 tertile 실측값(로컬 분석, range와 피어슨 상관 -0.64로 "원거리=
# 희소" 가정 확인됨) - CenterHead 전용(anchor head는 스코프 밖, 오늘 방침).
DENSITY_THRESH_LOW = 629    # 이하: sparse(class 0)
DENSITY_THRESH_HIGH = 1051  # 초과: dense(class 2), 사이: adequate(class 1)
DENSITY_AUX_WEIGHT = 0.2    # 원 논문 λ_aux

# --- Voxel polarization Phase 1 (Cylinder3D식 cylindrical partitioning, z축은 그대로
# Cartesian 유지, BEV 평면(x,y)만 (r,theta) 극좌표로 재인덱싱 - project_polarization_design
# 메모리 참고). validate_polar_design.py로 정량 검증 완료(item1: r/theta 100% 커버리지,
# item2: non-empty voxel 비율 전 구간 Cartesian보다 높음, item3: heatmap radius가 Cartesian의
# 항상-tau-floor(2.0) 문제를 일부 구간에서 벗어남). --polar로 opt-in(기본 off), Cartesian
# 파이프라인은 그대로 유지 - 두 방식을 나란히 비교해야 하므로 기존 걸 덮어쓰지 않는다.
SONAR_AZIMUTH_LIMIT_DEG = 45.0  # stamp_core.py 실측치(6개 scene 공통, 센서 물리 한계) 재확인
POLAR_R_RANGE = (0.8, 11.0)  # 실측 GT r 1.04~10.60m을 여유있게 감쌈(r_min>0: 극좌표 원점 퇴화 회피)
POLAR_THETA_RANGE_DEG = (-SONAR_AZIMUTH_LIMIT_DEG, SONAR_AZIMUTH_LIMIT_DEG)
POLAR_R_BINS = 102     # raw voxel grid 해상도 - dr ~= 0.1m (VOXEL_SIZE[0]와 동일)
POLAR_THETA_BINS = 90  # dtheta = 1도
# (W',H',D') 관례에 맞춤: theta를 W(가로/열), r을 H(세로/행) 자리에 - 극좌표를 "위에서 아래로
# range가 증가하는 unwrap된 부채꼴"로 보는 배치(range-view 계열 표현과 동일 관례).
POLAR_GRID_SIZE = (POLAR_THETA_BINS, POLAR_R_BINS, GRID_SIZE[2])
# heatmap(center head 타겟)은 RPN 다운샘플 배수(2x)만큼 성긴 별도 해상도를 쓴다 -
# Cartesian이 ANCHOR_STRIDE(voxel 해상도의 2배)를 쓰는 것과 동일 관계.
POLAR_HEATMAP_R_BINS = POLAR_R_BINS // 2
POLAR_HEATMAP_THETA_BINS = POLAR_THETA_BINS // 2

# --- PARTNER(arXiv:2308.03982) Phase2 GRR(Global Representation Re-alignment) ---
# polar 전용, opt-in(VoxelNet(use_grr=True)). N/S/W_a는 논문 원안 기본값 그대로 - 논문은
# R≈1155(Waymo)에서 검증했고 우리 R=102라 압축비가 논문보다 훨씬 낮지만(논문에 N에 대한
# ablation 자체가 없어 우리 스케일에서의 최적값도 미검증), 1차 실험은 원안 기본값으로
# 먼저 보고 결과에 따라 N을 스윕 후보로 남겨둔다.
PARTNER_GRR_N = 4
PARTNER_GRR_FILTER_WINDOW = 3
PARTNER_GRR_WINDOW_A = 8
