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

from pathlib import Path

VOXELNET_ROOT = Path(__file__).resolve().parent.parent
TRIBAND_ROOT = VOXELNET_ROOT.parent / "Triband_BEV"

# point cloud range: (x_min,y_min,z_min,x_max,y_max,z_max), 미터.
# GT 박스(x 1.4~10.5, y -3.8~3.1, z -1.55~1.13)를 여유있게 감싸면서, 먼 배경/노이즈
# 포인트(x up to 16, y +-10.9)는 잘라낸다 - 실측 커버리지 89%(model/config 설계 시
# 확인, 조사 세션 기록).
POINT_CLOUD_RANGE = (0.0, -5.0, -2.5, 12.0, 5.0, 2.5)

# (vx, vy, vz). vz=0.5로 D'=5.0/0.5=10 고정 (위 설명 참고).
VOXEL_SIZE = (0.1, 0.1, 0.5)

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

# --- Sparse 3D backbone (sparse_voxelnet.py) -- BEV로 압축하지 않는 대안 경로.
# ConvMiddleLayers(z를 채널로 눌러 BEV로 만듦) + RPNBackbone(2D CNN)을 통째로 대체한다.
# head(sparse_center_head.py)는 baseline RPNCenterHead와 동일한 목표(heatmap/offset/
# dim/rot6D/density)를 유지하되 dense (B,C,H,W) 대신 sparse (N,C) 위에서 동작하도록
# 차원만 바꾼 버전 -- backbone 쪽만 실험적으로 바꿔가며 비교하기 위한 스캐폴딩.
# 단일 스테이지/stride 2 기본값은 3d_point_cloud(자매 프로젝트)에서 다이버 탐지에
# 검증된 값 그대로 재사용 -- 4단계/stride 16처럼 깊게 다운샘플하면 최종 해상도가
# 다이버 크기보다 커져 precision/recall이 무너지는 걸 그 프로젝트에서 직접 겪었다.
SPARSE_BACKBONE_STAGE_CHANNELS = (128,)      # 단일 스테이지, VFE 출력(128)과 동일 폭 유지
SPARSE_BACKBONE_NUM_BLOCKS_PER_STAGE = 3
SPARSE_BACKBONE_DOWNSAMPLE_KERNEL = 3
SPARSE_BACKBONE_DOWNSAMPLE_STRIDE = 2         # 총 다운샘플 배율 = stride^(len(STAGE_CHANNELS))

# --- sonar_diver_dataset.py 전용 (3d_point_cloud 자매 프로젝트의 diver sonar 데이터) ---
# 이 데이터는 Triband_BEV와 물리적 커버리지 자체가 다르므로(원점 기준 훨씬 넓은 범위),
# 위 POINT_CLOUD_RANGE/VOXEL_SIZE(RPN_IN_CHANNELS 등 dense 파이프라인 전체가 이 값에
# 맞춰 캘리브레이션됨)를 덮어쓰지 않고 별도 상수로 둔다 - sparse 경로(sparse_voxelnet.py/
# sparse_center_head.py/train_sparse.py)만 이걸 읽는다.
# VOXEL_SIZE=0.1 (2026-09-01, was 0.2): dense baseline의 x,y 유효 해상도가 정확히
# 0.2m(VOXEL_SIZE[0]=0.1 x ANCHOR_STRIDE의 다운샘플 2배)였다는 걸 확인한 뒤, sparse
# 경로도 같은 유효 해상도(0.1 x backbone stride 2 = 0.2m)로 맞춰 "해상도는 동일, z를
# 채널로 누르느냐 아니냐만 다른" 공정한 backbone 비교가 되도록 조정. 3d-point-cloud에서
# 이 정확히 같은 변경(0.2->0.1)이 활성 voxel 수를 크게 늘려 안전한 배치 크기가 8~12에서
# 2~4로 떨어진 전례가 있음 - 배치 크기는 이 변경 후 반드시 실측으로 재검증할 것.
SPARSE_POINT_CLOUD_RANGE = (0.0, -11.4, -5.6, 16.2, 11.4, 5.6)
SPARSE_VOXEL_SIZE = (0.1, 0.1, 0.1)
SPARSE_GRID_SIZE = tuple(round((SPARSE_POINT_CLOUD_RANGE[3 + i] - SPARSE_POINT_CLOUD_RANGE[i]) / SPARSE_VOXEL_SIZE[i])
                          for i in range(3))  # (W',H',D')=(x,y,z) counts, config.GRID_SIZE와 동일 관례
SPARSE_MAX_POINTS_PER_VOXEL = 35
SPARSE_MAX_VOXELS = 40000  # 0.1m에서는 프레임당 활성 voxel이 0.2m 대비 최대 8배까지 늘 수 있음

# sparse 백본 출력 위에 얹는 축별(axial) 슬롯 어텐션 refinement -- models/slotformer.py
# (3d_point_cloud 자매 프로젝트에서 검증된 구현 그대로 포팅). dense 경로는 이미 RPNBackbone이
# 전체 그리드를 컨볼브해서 receptive field가 넓으므로 SlotFormer가 필요 없고, sparse 경로만
# 적용 대상.
# WIN_SIZE=24: 3d_point_cloud의 WIN_SIZE=12는 그 프로젝트 유효 해상도 0.4m 기준 물리적
# 윈도우 4.8m -- 이 프로젝트의 sparse 유효 해상도는 0.2m(SPARSE_VOXEL_SIZE 0.1 x backbone
# stride 2)이므로 같은 4.8m 물리적 윈도우를 맞추려면 24칸이 필요 (4.8 / 0.2 = 24).
# NUM_CYCLES=2 (x,y,z 두 바퀴 = 6 레이어, 2026-09-01: 3L에서 변경). 3d_point_cloud의 6L은
# 학습 도중 사용자 요청으로 epoch 2에서 중단돼 3L과의 최종 비교가 안 끝났음 -- 여기서
# 끝까지 학습해서 직접 비교.
SPARSE_SLOTFORMER_ENABLED = True
SPARSE_SLOTFORMER_WIN_SIZE = 24
SPARSE_SLOTFORMER_NUM_CYCLES = 2
SPARSE_SLOTFORMER_NUM_HEADS = 4

# sparse_voxelnet_down_slot_up.py 전용 -- 4단계 다운샘플 encoder -> SlotFormer(bottleneck,
# 가장 적은 active voxel에서 돌려서 제일 쌈) -> 4단계 업샘플 decoder(완전 복원, stride=1).
# 3d_point_cloud 자매 프로젝트의 backbone3d_down_slot_up.py와 동일 설계, 이식.
SPARSE_DOWN_SLOT_UP_STAGE_CHANNELS = (64, 96, 128, 128)
SPARSE_DOWN_SLOT_UP_NUM_BLOCKS_PER_STAGE = 2
SPARSE_DOWN_SLOT_UP_DOWNSAMPLE_KERNEL = 3
SPARSE_DOWN_SLOT_UP_DOWNSAMPLE_STRIDE = 2
SPARSE_DOWN_SLOT_UP_UPSAMPLE_STAGES = 4       # =len(STAGE_CHANNELS) -> 완전 복원 (stride=1)
SPARSE_DOWN_SLOT_UP_DECODER_BLOCKS_PER_STAGE = 2
# WIN_SIZE=3: bottleneck 유효 해상도 = SPARSE_VOXEL_SIZE(0.1) * stride^4(16) = 1.6m/voxel.
# 물리적 윈도우 ~4.8m를 맞추려면 4.8/1.6=3 (기존 단일 스테이지 백본의 WIN_SIZE=24는
# bottleneck 해상도가 0.2m였을 때 값 -- 인코더가 훨씬 깊어져서 voxel 하나가 이미 넓은
# 공간을 커버하므로 같은 물리적 크기를 맞추려면 훨씬 작은 숫자가 필요).
SPARSE_DOWN_SLOT_UP_SLOTFORMER_WIN_SIZE = 3
SPARSE_DOWN_SLOT_UP_SLOTFORMER_NUM_CYCLES = 2  # 6L
SPARSE_DOWN_SLOT_UP_SLOTFORMER_NUM_HEADS = 4
# 경고: 이 조합(4단계 다운 + 4단계 완전 복원 + SlotFormer, VOXEL_SIZE=0.1)은 3d_point_cloud
# 자매 프로젝트의 `unet` 브랜치(디코더만, SlotFormer 없음)가 A100에서도 epoch당 3시간+
# 걸렸던 것과 같은 비용 구조(decoder 얕은 단계가 거의 원본 해상도의 voxel 수를 처리)를
# 그대로 가짐 -- 실제 학습 전에 반드시 실측(메모리/속도) 먼저 할 것.

# experiments/exp2_down_slot_up_bev/voxelnet.py 전용 (2026-09-02, "지금 실험 토대로 구조 변경" 새 실험) -- 위
# SPARSE_DOWN_SLOT_UP_* 백본은 그대로 재사용하되, 그 출력(sparse voxel)을 SparseCenterHead로
# 바로 보내는 대신 dense (B,C,D,H,W)로 scatter한 뒤 z를 채널로 눌러 BEV로 만들고, RPNCenterHead
# (dense 파이프라인의 원래 head, model.py -- 안 바꿈)로 보낸다. VOXEL_SIZE의 z를 0.5로 굵게
# 잡은 이유가 바로 이것: SPARSE_POINT_CLOUD_RANGE의 z 범위(-5.6~5.6, 11.2m)에서 z=0.1이면
# D=112라 4단계 다운+4단계 완전복원을 해도 D가 그대로 112로 남아 채널로 누르면 128*112=14336
# 채널이 되어버림 -- z=0.5로 바꾸면 D=22로 시작해서, 채널 압축 자체는 여전히 크지만
# (128*22=2816) 1x1 conv 하나로 RPN_IN_CHANNELS(128)까지 투영 가능한 수준.
# x,y는 그대로 0.1 유지 -- z만 훨씬 굵게 잡는 게 정확히 원조 VoxelNet의 ConvMiddleLayers가
# 하던 것(z만 빠르게 눌러서 채널로, x,y는 그대로 두고 2D RPN에서 처리)과 같은 철학.
SPARSE_BEV_VOXEL_SIZE = (0.1, 0.1, 0.5)
# 2026-09-02: scan_point_range.py로 labeling-tool-main/dataset 전체(37010 프레임, 2.52억
# point) 스캔해서 실측 범위(x: 0.135~16.017, y: -11.331~11.306, z: -5.478~5.482)를 확인함 --
# 기존 SPARSE_POINT_CLOUD_RANGE가 이미 이걸 여유있게 다 감싸고 있어 정확했음. x=0 근처
# 빈 공간(0~0.135m)만큼 살짝 자르는 것도 검토했으나, 그건 dense grid(모든 셀이 항상
# 계산 비용을 먹음)에서나 의미 있는 최적화 -- sparse는 비어있는 셀이 애초에 비용이
# 거의 없으므로 굳이 좁힐 이유가 없음. 그대로 SPARSE_POINT_CLOUD_RANGE 재사용.
SPARSE_BEV_POINT_CLOUD_RANGE = SPARSE_POINT_CLOUD_RANGE  # unchanged -- only VOXEL_SIZE differs from the sparse_voxelnet.py experiment (fully-sparse head, not part of the experiments/ BEV series)
SPARSE_BEV_GRID_SIZE = tuple(round((SPARSE_BEV_POINT_CLOUD_RANGE[3 + i] - SPARSE_BEV_POINT_CLOUD_RANGE[i]) / SPARSE_BEV_VOXEL_SIZE[i])
                              for i in range(3))  # (W',H',D')=(x,y,z) -- (162,228,22) at this voxel size
SPARSE_BEV_MAX_POINTS_PER_VOXEL = 35
SPARSE_BEV_MAX_VOXELS = 40000

# Same 4-stage backbone shape as SPARSE_DOWN_SLOT_UP_* above, reused verbatim for this
# experiment too (only VOXEL_SIZE/head/data-split differ) -- UPSAMPLE_STAGES=4 (full
# restore) means the backbone's output D equals the *input* D (22 at this voxel size),
# not the deepest-bottleneck D -- see experiments/exp2_down_slot_up_bev/voxelnet.py for
# the exact grid-size bookkeeping (computed once at construction, not re-derived from
# data each forward).
SPARSE_BEV_STAGE_CHANNELS = (64, 96, 128, 128)
SPARSE_BEV_NUM_BLOCKS_PER_STAGE = 2
SPARSE_BEV_DOWNSAMPLE_KERNEL = 3
SPARSE_BEV_DOWNSAMPLE_STRIDE = 2
SPARSE_BEV_UPSAMPLE_STAGES = 4
SPARSE_BEV_DECODER_BLOCKS_PER_STAGE = 2
SPARSE_BEV_SLOTFORMER_WIN_SIZE = 3    # x,y voxel size (0.1) and backbone depth/stride are identical
                                       # to SPARSE_DOWN_SLOT_UP_*, so the bottleneck's x,y effective
                                       # resolution is the same 1.6m/voxel (0.1*2^4) -- reuses that
                                       # experiment's WIN_SIZE=3 derivation unchanged (only z/VOXEL_SIZE
                                       # and what happens after the backbone differ between the two).
SPARSE_BEV_SLOTFORMER_NUM_CYCLES = 2  # 6L, matching the current experiment this is based on
SPARSE_BEV_SLOTFORMER_NUM_HEADS = 4

# experiments/exp3_conv_middle_bev/voxelnet.py 전용 (2026-09-02) -- exp1/exp2와 달리 backbone
# 구조 자체를 바꾸는 실험이 아니라, dense pipeline(model.py)에서 딱 ConvMiddleLayers 한
# 부분만 sparse conv로 바꾸면 어떻게 되는지 보는 실험. VFE/RPNBackbone/RPNCenterHead는
# dense와 완전히 동일(코드 재사용), SlotFormer도 없음 -- ConvMiddleLayers와 똑같은 3-layer
# 모양(channels 128->64->64->64, kernel=3, stride/padding (2,1,1)/(1,1,1) -> (1,1,1)/(0,1,1)
# -> (2,1,1)/(1,1,1))을 SparseConv3dDown으로 그대로 복제한다 -- 채널 폭도 dense와 동일하게
# 64로 고정(별도 STAGE_CHANNELS 없음, 원본 ConvMiddleLayers처럼 하드코딩된 모양).
# (이전엔 여기 exp3가 z만 계속 줄이는 4단계 커스텀 backbone+SlotFormer였으나, "dense에서
# BEV로 누르는 부분만 sparse로 바꾼 것"을 보고 싶다는 요청으로 이 설계로 교체됨.)
SPARSE_BEV_CONVMID_CHANNELS = 64  # ConvMiddleLayers의 고정 채널 폭과 동일

# --- 학습 ---
BATCH_SIZE = 4
NUM_EPOCHS = 30
LR = 0.01
LR_DECAY_EPOCH_FRAC = 0.85  # 마지막 15%는 lr/10 (논문: 160 epoch 중 마지막 10epoch)
LR_DECAY_FACTOR = 0.1
WEIGHT_DECAY = 1e-4
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
