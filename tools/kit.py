"""제3회 천문우주 AI 경진대회(태양풍 속도 예측) 실전 키트.

PLAYBOOK.md Phase 0 이 요구하는 것만 담는다:
채점기 · 세그먼트 분해 · 블록 CV · 지속성 기저 · 노이즈 실측 · 예측 자산 저장 · 힐클라이밍.

본선 데이터 스키마는 당일에야 확정되므로 모든 함수는 numpy 배열만 받는다.
y 규격: (n_samples, n_horizon), 열 h = 6*(h+1) 시간 뒤. 본선은 H=12 (72h).

본선 제출은 code.ipynb 단일 파일이라 외부 스크립트를 못 쓴다.
그래서 이 파일은 **셀 하나에 통째로 붙여넣을 수 있는 크기**로 유지한다. 늘리지 마라.

자체 점검: python tools/kit.py
"""
import pathlib

import numpy as np

HORIZON_H = 6            # 관측·예측 간격(시간)
N_HORIZON = 12           # 6h × 12 = 72h (RULES.md §3 확정)
CR_DAYS = 27.2753        # 태양 회합 자전 주기(캐링턴)
# 주의: persistence_27d / carrington_block 은 절대 시각을 요구한다. 본선 제공 입력에는
# 날짜가 없고 test 는 익명이라 **제출 경로에서 쓸 수 없다**(DOMAIN.md §9.1).
# train/validation 에 시각이 남아 있을 때의 분석용으로만 남겨둔다.
PRED_DIR = pathlib.Path("preds")
LOG = pathlib.Path("experiments/log.md")


# ── 1. 채점기 (PLAYBOOK §1.1) ────────────────────────────────────────────────
def _check(y, p):
    """채점 전 방어. 여기서 안 막으면 조용히 잘못된 점수가 나온다.

    · shape 불일치 — 예측이 20열인데 정답이 12열이면 뒤 8열이 무시된다(20→12 미수정 모델)
    · 예측 NaN — 마스크로 빼면 **어려운 셀만 NaN 낸 모델이 유리해진다.** 벌점이어야 한다
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    if y.shape != p.shape:
        raise ValueError(f"shape 불일치: y{y.shape} vs p{p.shape}")
    if not np.isfinite(p[np.isfinite(y)]).all():
        raise ValueError("예측에 NaN/inf 가 있다. 채점 대상 셀은 전부 유한해야 한다")
    return y, p


def rmse_per_horizon(y, p):
    """horizon 별 RMSE. **진단용이다** — 대회 지표는 `score` 다.

    정답 결측(NaN)은 그 셀만 제외한다.
    """
    y, p = _check(y, p)
    out = np.full(y.shape[1], np.nan)
    for h in range(y.shape[1]):
        m = np.isfinite(y[:, h])
        if m.any():
            out[h] = np.sqrt(np.mean((y[m, h] - p[m, h]) ** 2))
    return out


def score(y, p):
    """대회 지표: **전체 셀을 한 번에** 평균 후 제곱근. RULES.md §4 식 그대로. 낮을수록 좋다.

    ⚠ horizon 별 RMSE 를 평균한 값이 아니다. 두 값은 다르다 —
    horizon RMSE 가 [0, 10] 이면 이 식은 sqrt(50)=7.07, 평균식은 5 다.
    2026-08 이전 판이 평균식이었다. 그때의 실험 점수는 지금 값과 비교하지 마라.
    """
    y, p = _check(y, p)
    m = np.isfinite(y)
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))


def sensitivity(y, p):
    """항별 감도. 전체 RMSE 기준 d(score)/d(MSE_h) = 1/(2·H·score).

    **모든 horizon 이 같다.** 전체를 한 번에 평균하므로 어느 horizon 의 MSE 를 1 줄이든
    점수에 주는 효과가 동일하다. ⇒ **개선 여지가 큰 곳(=오차가 큰 장기 horizon·HSS 구간)이
    그대로 이득이 큰 곳이다.** 표를 뽑아 어디에 여지가 남았는지만 보면 된다.
    """
    r = rmse_per_horizon(y, p)
    s = score(y, p)
    return r, np.full(len(r), 1.0 / (2 * len(r) * s))


# ── 2. 세그먼트 분해기 (PLAYBOOK §2.4) ──────────────────────────────────────
def breakdown(y, p, groups):
    """(그룹, 셀수, RMSE, 편향) 목록. 전체 점수 하나만 보는 실험은 시작하지 마라.

    groups 는 행 단위(길이 n) 또는 셀 단위(y 와 같은 shape) 둘 다 받는다.
    편향 = mean(pred − true). HSS 과소예측을 잡는 열이 이거다.
    셀 단위 RMSE 라 `score` 와 같은 형태다. 단 그룹별로 잘라 보는 진단용이다.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    g = np.asarray(groups)
    if g.shape != y.shape:
        g = np.repeat(g.reshape(-1, 1), y.shape[1], axis=1)
    ok = np.isfinite(y) & np.isfinite(p)
    rows = []
    for k in np.unique(g[ok]):
        m = ok & (g == k)
        d = p[m] - y[m]
        rows.append((k, int(m.sum()), float(np.sqrt(np.mean(d ** 2))), float(np.mean(d))))
    return rows


def speed_band(y, edges=(400, 500, 600)):
    """실제 속도 구간 라벨. 300~400 평상시 / 600+ 고속 기류(HSS) 를 갈라 본다."""
    return np.digitize(np.asarray(y, float), edges)


# ── 3. 검증 설계 (PLAYBOOK §2.1) ────────────────────────────────────────────
def carrington_block(times, period_days=CR_DAYS, epoch=None):
    """캐링턴 자전 주기 블록 id. 무작위 셔플은 27일 지속성 누수를 부른다."""
    t = np.asarray(times, dtype="datetime64[s]").astype("float64") / 86400.0
    epoch = t.min() if epoch is None else epoch
    return np.floor((t - epoch) / period_days).astype(int)


def block_folds(blocks, n_folds=5, gap=1):
    """연속 블록 n등분 + 퍼지. 검증 블록 앞뒤 gap 블록은 학습에서 뺀다.

    입력 창(과거 5일)과 출력 창(미래 3일 = 72h)이 경계를 넘어 겹치므로 퍼지 없으면 누수다.
    """
    ub = np.unique(blocks)
    for c in np.array_split(ub, n_folds):
        val = np.isin(blocks, c)
        near = np.isin(blocks, np.arange(c.min() - gap, c.max() + gap + 1))
        yield ~near, val


# ── 4. 기저 (첫날 리더보드 확보용) ──────────────────────────────────────────
def persistence_last(v, n_horizon=N_HORIZON):
    """마지막 관측값을 전 horizon 에 복사. 최하 기저."""
    return np.repeat(np.asarray(v, float).reshape(-1, 1), n_horizon, axis=1)


def persistence_27d(v, idx, lag_steps=109, n_horizon=N_HORIZON):
    """27일 전 같은 위상의 값. **본선 제출 경로에서는 못 쓴다** (DOMAIN.md §9.1).

    입력 창이 120h 뿐이고 샘플에 날짜가 없다. train/validation 에 시각이 남아 있을 때의
    분석·비교용으로만 남긴다. 제출 기저는 `persistence_last` 다.

    v: 6시간 균일 격자 위의 속도 시계열, idx: 각 샘플의 기준 시점 인덱스.
    lag_steps: 27.2753일 = 109.1 스텝. 실효 지연은 태양풍 속도에 따라 달라지니
    106~112 를 스윕해서 로컬 최적을 잡아라.  # ponytail: 균일 격자 가정, 결측 많으면 시각 기준 검색으로 교체
    """
    v = np.asarray(v, float)
    idx = np.asarray(idx, int)
    out = np.full((len(idx), n_horizon), np.nan)
    for h in range(n_horizon):
        src = idx + (h + 1) - lag_steps
        ok = (src >= 0) & (src < len(v))
        out[ok, h] = v[src[ok]]
    return out


# ── 5. 노이즈 실측 (PLAYBOOK §2.3) ──────────────────────────────────────────
def pred_dist(a, b, scale=None):
    """두 예측의 거리. sd(Δ) ≈ k·√dist 적합의 x축."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    scale = np.nanstd(a) if scale is None else scale
    return float(np.nanmean(np.abs(a - b)) / scale)


def private_sigma(k, dist, p=0.5):
    """Public→Private 전이 sd. p = public 비율. 후보 Δ 를 이걸로 나눠 σ 를 낸다.

    ⚠ k 는 **paired 실측으로만** 얻는다. `calibrate` 참조. 감으로 넣지 마라.
    """
    return k * np.sqrt(dist) * np.sqrt(p) * np.sqrt(1 / p + 1 / (1 - p))


def calibrate(cv, lb):
    """제출할 때마다 쌓은 (로컬 CV, 리더보드) 쌍으로 `LB ≈ a·CV + b` 를 적합한다.

    반환 (a, b, resid_sd, n).

    **resid_sd 가 채택선이다.** 로컬에서 이보다 작은 개선은 리더보드에서 구별되지 않는다.
    널 제출 1장으로 σ 를 재려던 옛 방법을 대체한다 — 그건 시드를 바꿔도 모델이 실제로 달라져
    채점 노이즈가 아니었고, 제출 1장을 태웠다 (POSTMORTEM M27).

    a 가 1 에서 크게 벗어나면 **로컬이 판정기가 아니라 탐색기**라는 신호다 (PLAYBOOK §2.1):
    로컬 이득이 리더보드에 그 비율로만 전달된다는 뜻이다.
    n < 4 면 기울기를 믿지 마라 — 절편만 쓰고 resid_sd 는 보수적으로 크게 잡아라.
    """
    x, y = np.asarray(cv, float), np.asarray(lb, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2:
        return np.nan, np.nan, np.nan, len(x)
    a, b = np.polyfit(x, y, 1)
    resid = y - (a * x + b)
    # 자유도 2 를 뺀다. 점 2개면 잔차가 0 이라 σ 를 0 으로 착각하면 안 된다.
    sd = float(np.sqrt(np.sum(resid ** 2) / (len(x) - 2))) if len(x) > 2 else np.nan
    return float(a), float(b), sd, len(x)


# ── 6. 실험 자산 (PLAYBOOK §4.1) ────────────────────────────────────────────
def save_pred(name, version, oof, test):
    """진 실험도 반드시 저장한다. 마지막 이득은 재조합에서 나온다."""
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PRED_DIR / f"oof_{name}_{version}.npy", np.asarray(oof))
    np.save(PRED_DIR / f"test_{name}_{version}.npy", np.asarray(test))


def load_oofs(pattern="oof_*.npy"):
    return {f.stem[4:]: np.load(f) for f in sorted(PRED_DIR.glob(pattern))}


def log_row(exp_id, hypothesis, change, cv, lb, verdict):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if not LOG.exists():
        LOG.write_text("| ID | 가설 | 변경점 | CV | LB | 판단 |\n|---|---|---|---|---|---|\n", encoding="utf-8")
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"| {exp_id} | {hypothesis} | {change} | {cv} | {lb} | {verdict} |\n")


# ── 7. 결합 (PLAYBOOK §5) ───────────────────────────────────────────────────
def hill_climb(oofs, y, n_iter=60):
    """중복 선택 허용 그리디. 수동 가중치 튜닝보다 거의 항상 낫다. 결합 1순위."""
    names = list(oofs)
    picks, cur, best = [], None, np.inf
    for _ in range(n_iter):
        trials = []
        for n in names:
            t = oofs[n] if cur is None else (cur * len(picks) + oofs[n]) / (len(picks) + 1)
            trials.append((score(y, t), n, t))
        s, n, t = min(trials, key=lambda x: x[0])
        if s >= best - 1e-9:
            break
        best, cur = s, t
        picks.append(n)
    return {n: picks.count(n) / len(picks) for n in set(picks)}, best


def resid_corr(a, b, y):
    """잔차 상관 ρ. 높으면(≥0.84) 앙상블 이득이 없다 — 그 자리에서 접어라."""
    ra, rb = (np.asarray(a, float) - y).ravel(), (np.asarray(b, float) - y).ravel()
    m = np.isfinite(ra) & np.isfinite(rb)
    return float(np.corrcoef(ra[m], rb[m])[0, 1])


def decompose(other, components):
    """예측 하나를 내 성분들로 최소자승 분해. 잔차가 0에 가까우면 독립 모델이 아니다.

    반환 (가중치, 상대잔차). 잔차가 작으면 그 혼합비를 이긴 방향으로 외삽하는 축이 열린다.
    ⚠ 외삽은 이긴 방향으로만.

    ⚠⚠ **대상은 우리 `code.ipynb` 안에서 재현되는 우리 모델의 예측뿐이다.**
    공개 커널·다른 팀 제출을 넣지 마라 — 그 예측은 제공 데이터가 아니고(RULES.md §1 1항),
    거기서 뽑은 혼합비는 제출 코드로 재생성할 수 없어 재현 검증에 걸린다(§1 3·4항).
    """
    X = np.column_stack([np.asarray(c, float).ravel() for c in components])
    b = np.asarray(other, float).ravel()
    m = np.isfinite(b) & np.isfinite(X).all(1)
    w, *_ = np.linalg.lstsq(X[m], b[m], rcond=None)
    return w, float(np.linalg.norm(X[m] @ w - b[m]) / np.linalg.norm(b[m]))


# ── 자체 점검 ───────────────────────────────────────────────────────────────
def demo():
    rng = np.random.default_rng(0)
    n, H = 400, N_HORIZON
    y = 400 + 150 * rng.standard_normal((n, H))

    assert score(y, y) == 0.0, "완벽예측은 0"
    assert np.isclose(score(y, y + 10), 10.0), "상수 오프셋 10 이면 RMSE 10"

    # ⚠ 위 두 검사는 「전체 RMSE」와 「horizon RMSE 평균」이 우연히 같은 경우다.
    #    두 식을 실제로 가르는 반례가 있어야 회귀를 잡는다. horizon 오차가 0 과 10 이면
    #    규정식(RULES.md §4) = sqrt((0+100)/2) = 7.071, 옛 평균식 = 5.
    y3 = np.zeros((10, 2))
    p3 = np.zeros((10, 2))
    p3[:, 1] = 10.0
    assert np.isclose(score(y3, p3), np.sqrt(50.0)), "전체 RMSE 여야 한다 (평균식 아님)"
    assert np.allclose(rmse_per_horizon(y3, p3), [0.0, 10.0])

    y2 = y.copy()
    y2[3, 5] = np.nan
    assert np.isclose(score(y2, y2 + 10), 10.0), "정답 NaN 셀은 제외되어야 한다"

    # 예측 NaN 은 제외가 아니라 실패다 — 어려운 셀만 NaN 내면 유리해지는 걸 막는다
    pbad = (y + 10).copy()
    pbad[0, 0] = np.nan
    try:
        score(y, pbad)
        raise AssertionError("예측 NaN 을 통과시켰다")
    except ValueError:
        pass

    # shape 불일치도 실패 — 20→12 미수정 모델이 로컬 PASS 되면 안 된다
    try:
        score(y, np.zeros((n, H + 8)))
        raise AssertionError("shape 불일치를 통과시켰다")
    except ValueError:
        pass

    r, sens = sensitivity(y, y + 10)
    assert np.allclose(sens, sens[0]), "전체 RMSE 기준이면 horizon 간 감도가 같다"

    # 채택선 산출 — 완전한 선형이면 잔차 0, 흔들리면 σ 가 커진다
    slope, icpt, sd, npts = calibrate([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])
    assert np.isclose(slope, 2.0) and abs(icpt) < 1e-9 and sd < 1e-9 and npts == 4
    _, _, sd2, _ = calibrate([1.0, 2.0, 3.0, 4.0], [2.0, 4.1, 5.9, 8.0])
    assert sd2 > sd, "잡음이 있으면 잔차 σ 가 커야 한다"
    assert np.isnan(calibrate([1.0, 2.0], [1.0, 2.0])[2]), "점 2개로 σ 를 0 이라 하면 안 된다"

    rows = breakdown(y, y + 10, speed_band(y))
    assert len(rows) == 4 and all(np.isclose(b, 10.0) for _, _, _, b in rows), "편향 열은 +10"

    t = np.arange("2024-01-01", "2024-07-01", np.timedelta64(6, "h"), dtype="datetime64[s]")
    blk = carrington_block(t)
    assert blk[0] == 0 and np.all(np.diff(blk) >= 0)
    folds = list(block_folds(blk, n_folds=4, gap=1))
    assert len(folds) == 4
    for tr, va in folds:
        assert not (tr & va).any(), "학습/검증 겹치면 안 된다"
        assert set(blk[tr]).isdisjoint(set(blk[va]) | {min(blk[va]) - 1, max(blk[va]) + 1}), "퍼지 실패"

    v = np.arange(500.0)
    p27 = persistence_27d(v, idx=np.array([200, 300]), lag_steps=109, n_horizon=3)
    assert p27[0, 0] == 200 + 1 - 109
    assert np.isnan(persistence_27d(v, idx=np.array([5]), lag_steps=109, n_horizon=1)[0, 0]), "범위 밖은 NaN"
    assert persistence_last(np.array([1.0, 2.0])).shape == (2, N_HORIZON)

    # 힐클라이밍: 각각은 나쁘고 평균이 좋은 두 예측 → 둘 다 잡아야 한다
    e = rng.standard_normal((n, H)) * 30
    a, b = y + e, y - e
    w, s = hill_climb({"a": a, "b": b}, y)
    assert set(w) == {"a", "b"} and s < min(score(y, a), score(y, b))
    assert abs(resid_corr(a, b, y) + 1) < 1e-6, "정반대 잔차 ρ = −1"

    w2, res = decompose(0.8 * a + 0.2 * b, [a, b])
    assert np.allclose(w2, [0.8, 0.2], atol=1e-6) and res < 1e-9

    k = 1.0
    assert np.isclose(private_sigma(k, dist=1.0, p=0.5), np.sqrt(2))
    assert pred_dist(y, y) == 0.0

    print("kit 자체 점검 통과")


if __name__ == "__main__":
    demo()
