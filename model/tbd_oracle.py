"""tbd_oracle.py - 무료 Track-Before-Detect 오라클 (고전, 비학습).

제안 아키텍처(Differentiable Bayesian TBD, reports_v2/8_31_roadmap.html)의 고전 lower-bound.
기존 detector의 프레임별 detection(낮은 threshold)에 KF + greedy NN association + 존재
log-odds(IPDA-lite) 후처리를 씌워, temporal 일관성이 FP↓/원거리 recall↑를 주는지 학습 0원으로 판정.

입력 detection 포맷(프레임별 list):
    {"center": (3,) xyz, "dims": (3,), "R": (3,3), "score": float}
run_scene()이 프레임 순서대로 track을 돌리고, 각 detection에 존재 posterior를 반영한
"score_tbd"(rescored)와 track_id, confirmed 플래그를 채워 반환한다. 박스 기하(center/dims/R)는
불변 - 시간축은 confidence만 건드린다(기하 blur 회피, per-track size는 참고용으로만 추정).

순수 numpy. 모델/캐시 불필요 - detection dump에만 의존.
"""
import numpy as np

# --- 튜닝 파라미터 (val_sub에서 소규모 그리드, test로 튜닝 금지=leakage) ---
GATE_DIST = 1.0        # m, association gating (다이버 이동 median 4.7cm/frame, 여유크게)
P_BIRTH = 0.05         # Markov birth  Pr(E=1|E=0)
P_DEATH = 0.05         # Markov death  Pr(E=0|E=1)
# IPDA 존재 우도비: matched는 log(score)-log(P_CLUTTER) (score>clutter면 양의 증거),
# missed는 log(1-P_D). PI0(prior)로 빼면 score<prior인 약-지속 타겟이 못 쌓임 → clutter율 사용.
P_CLUTTER = 0.10       # clutter false-alarm 확률(분모) - 실제 detection은 이보다 큼
P_D = 0.60             # 진짜 타겟의 프레임당 detection 확률(miss 페널티 log(1-P_D))
BIRTH_LOGODDS = -3.0   # 새 track 초기 log-odds - 단일 detection으로 confirm 안 되게 낮게
CONFIRM_LOGODDS = 0.0  # 이 이상이면 confirmed(존재확률>0.5)
PRUNE_LOGODDS = -5.0   # 이 이하 track 제거
DT = 1.0               # 프레임 간격(등속모델)
Q_POS, Q_VEL = 0.05, 0.05   # KF process noise
R_MEAS = 0.10          # KF 측정 noise (centroid)
SIZE_LR = 0.2          # per-track size EMA 학습률


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _markov_predict_logodds(L):
    """존재 log-odds에 birth/death Markov 예측 스텝(Chapman-Kolmogorov)."""
    p = _sigmoid(L)
    p_bar = (1 - P_DEATH) * p + P_BIRTH * (1 - p)
    return _logit(p_bar)


class _Track:
    __slots__ = ("x", "P", "L", "dims", "id", "hits")

    def __init__(self, center, dims, score, tid):
        # KF 상태 [x, y, vx, vy] (BEV 등속). z는 상태 밖(측정값 통과).
        self.x = np.array([center[0], center[1], 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([R_MEAS, R_MEAS, 1.0, 1.0]).astype(np.float64)
        self.L = BIRTH_LOGODDS + (np.log(max(score, 1e-6)) - np.log(P_CLUTTER))
        self.dims = np.array(dims, dtype=np.float64)
        self.id = tid
        self.hits = 1

    def predict(self):
        F = np.array([[1, 0, DT, 0], [0, 1, 0, DT], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
        self.x = F @ self.x
        Q = np.diag([Q_POS, Q_POS, Q_VEL, Q_VEL])
        self.P = F @ self.P @ F.T + Q
        self.L = _markov_predict_logodds(self.L)

    def update(self, center, dims, score):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        z = np.array([center[0], center[1]], dtype=np.float64)
        y = z - H @ self.x
        S = H @ self.P @ H.T + np.eye(2) * R_MEAS
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        # 존재 log-odds 갱신(TBD 우도 누적): matched 측정 (IPDA 우도비)
        self.L = self.L + (np.log(max(score, 1e-6)) - np.log(P_CLUTTER))
        # per-track size EMA(참고용; 원거리 fallback에 쓸 수 있음)
        self.dims = (1 - SIZE_LR) * self.dims + SIZE_LR * np.array(dims)
        self.hits += 1

    def miss(self):
        self.L = self.L + np.log(1 - P_D)

    @property
    def pos(self):
        return self.x[:2]


def run_scene(frames):
    """frames: list(프레임 순서) of list of detection dict.
    각 detection에 'score_tbd','track_id','confirmed'를 채워 같은 구조로 반환."""
    tracks = []
    next_id = 0
    out = []
    for dets in frames:
        for t in tracks:
            t.predict()
        # greedy NN association (gating)
        unmatched = list(range(len(dets)))
        assigned = {}
        if tracks and dets:
            pairs = []
            for di, d in enumerate(dets):
                for ti, t in enumerate(tracks):
                    dist = np.linalg.norm(np.array(d["center"][:2]) - t.pos)
                    if dist <= GATE_DIST:
                        pairs.append((dist, di, ti))
            pairs.sort()
            used_t, used_d = set(), set()
            for dist, di, ti in pairs:
                if di in used_d or ti in used_t:
                    continue
                used_d.add(di); used_t.add(ti); assigned[di] = ti
            unmatched = [di for di in range(len(dets)) if di not in used_d]
            matched_t = used_t
        else:
            matched_t = set()
        # update matched
        for di, ti in assigned.items():
            d = dets[di]
            tracks[ti].update(d["center"], d["dims"], d["score"])
        # miss for unmatched tracks
        for ti, t in enumerate(tracks):
            if ti not in matched_t:
                t.miss()
        # spawn new tracks for unmatched detections
        for di in unmatched:
            d = dets[di]
            tracks.append(_Track(d["center"], d["dims"], d["score"], next_id)); next_id += 1
            assigned[di] = len(tracks) - 1
        # rescore: 존재 posterior 반영
        frame_out = []
        for di, d in enumerate(dets):
            ti = assigned[di]
            p = _sigmoid(tracks[ti].L)
            nd = dict(d)
            nd["score_tbd"] = float(np.sqrt(max(d["score"], 1e-6) * p))  # 원 score와 존재의 기하평균
            nd["track_id"] = tracks[ti].id
            nd["confirmed"] = bool(tracks[ti].L >= CONFIRM_LOGODDS)
            frame_out.append(nd)
        out.append(frame_out)
        # prune dead tracks
        tracks = [t for t in tracks if t.L > PRUNE_LOGODDS]
    return out


# ---------------- 유닛 테스트 (합성) ----------------
def _selftest():
    rng = np.random.default_rng(0)
    dims = (0.9, 0.9, 1.7); R = np.eye(3)
    frames = []
    # 약-지속 타겟: score 0.20(threshold 0.3 미만), 천천히 이동
    # + 강-transient 노이즈: score 0.60, 매 프레임 랜덤 위치 1개(한 프레임만 존재)
    for i in range(20):
        dets = []
        cx, cy = 3.0 + 0.05 * i, 0.1 + 0.03 * i   # 약-지속 타겟(이동)
        dets.append({"center": (cx, cy, 0.2), "dims": dims, "R": R, "score": 0.20})
        # transient 노이즈 1개(랜덤)
        nx, ny = rng.uniform(-4, 4), rng.uniform(-4, 4)
        dets.append({"center": (nx, ny, 0.2), "dims": dims, "R": R, "score": 0.60})
        frames.append(dets)
    out = run_scene(frames)
    # 마지막 프레임에서: 타겟(det0)은 confirmed+rescore↑, 노이즈(det1)는 낮게
    last = out[-1]
    tgt = last[0]; noise = last[1]
    print(f"약-지속 타겟(원 score 0.20): score_tbd={tgt['score_tbd']:.3f} confirmed={tgt['confirmed']}")
    print(f"강-transient 노이즈(원 score 0.60): score_tbd={noise['score_tbd']:.3f} confirmed={noise['confirmed']}")
    # 타겟 존재확률 추이
    traj = [out[i][0]["score_tbd"] for i in range(20)]
    print("타겟 score_tbd 추이:", [round(v, 2) for v in traj])
    assert tgt["confirmed"], "약-지속 타겟이 confirm 안 됨(존재 누적 실패)"
    assert tgt["score_tbd"] > noise["score_tbd"], "타겟이 transient 노이즈보다 낮음(rescore 실패)"
    assert not noise["confirmed"], "transient 노이즈가 confirm됨(FP 억제 실패)"
    print("PASS: 약-지속 confirm + 강-transient 억제 확인")


if __name__ == "__main__":
    _selftest()
