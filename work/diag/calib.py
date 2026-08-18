"""고속 기류 과소예측(>600 km/s 구간 편향 −113)을 잡는 보정 계수를 만든다.

채점 분해가 말하는 것: 전체 셀의 7.4% 인 >600 구간이 제곱오차의 34% 를 낸다.
원인은 평균 회귀 — 모델이 극단을 안쪽으로 당긴다. 그래서 horizon 별로
    y ≈ a * pred + b
를 적합해 예측을 도로 펴 준다. a>1 이면 눌린 것을 편 것이다.

**계수는 train 내부 4-fold OOF 예측으로만 적합한다.** validation 은 효과 측정에만 쓴다.
validation 타깃으로 계수를 맞추면 그건 validation 학습이라 규정 위반이다 (RULES.md §1 7항).

산출: work/preds/calib.json  {"a": [...12], "b": [...12], ...}
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, "/home/jovyan/work")
from feats import sample_feats  # noqa: E402  imgfeat2 와 똑같은 피처 (work/feats.py)

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
OUT = pathlib.Path("/home/jovyan/work/preds")
T = [f"target_{i:02d}" for i in range(12)]
FOLDS = 4
ITERS = 300


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)))


tr_x = pd.read_csv(ROOT / "train/inputs.csv")
tr_y = pd.read_csv(ROOT / "train/targets.csv")[T].to_numpy(np.float64)
Xtr = sample_feats("train", tr_x)
print("features", Xtr.shape, flush=True)

oof = np.zeros_like(tr_y)
for fold, (fit, hold) in enumerate(KFold(FOLDS, shuffle=True, random_state=777).split(Xtr)):
    for h in range(12):
        m = HistGradientBoostingRegressor(
            max_iter=ITERS, learning_rate=0.07, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.12, n_iter_no_change=20, random_state=777,
        ).fit(Xtr[fit], tr_y[fit, h])
        oof[hold, h] = m.predict(Xtr[hold])
    print(f"fold{fold} oof_rmse={rmse(tr_y[hold], oof[hold]):.3f}", flush=True)

print(f"OOF 전체 rmse={rmse(tr_y, oof):.3f}", flush=True)

a, b = np.zeros(12), np.zeros(12)
for h in range(12):
    # y = a*p + b 최소자승. 평균 회귀가 있으면 a > 1 이 나온다.
    a[h], b[h] = np.polyfit(oof[:, h], tr_y[:, h], 1)
cal = oof * a + b
print("horizon별 기울기 a:", " ".join(f"{x:.3f}" for x in a), flush=True)
print(f"OOF 보정 전 {rmse(tr_y, oof):.3f} → 보정 후 {rmse(tr_y, cal):.3f}", flush=True)

# 속도대별로도 확인. 고속 구간 편향이 줄어야 의미가 있다.
for lo, hi in ((0, 400), (400, 500), (500, 600), (600, 2000)):
    m = (tr_y >= lo) & (tr_y < hi)
    print(f"  [{lo},{hi}) n={m.sum():6d} "
          f"bias {np.mean(oof[m]-tr_y[m]):8.2f} → {np.mean(cal[m]-tr_y[m]):8.2f}   "
          f"rmse {rmse(tr_y[m], oof[m]):7.2f} → {rmse(tr_y[m], cal[m]):7.2f}", flush=True)

(OUT / "calib.json").write_text(json.dumps(
    {"a": a.tolist(), "b": b.tolist(), "folds": FOLDS, "iters": ITERS,
     "oof_rmse": round(rmse(tr_y, oof), 3), "oof_rmse_calibrated": round(rmse(tr_y, cal), 3)},
    indent=2))
np.save(OUT / "oof_calibsrc.npy", oof)
print("saved", OUT / "calib.json")
