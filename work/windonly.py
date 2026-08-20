"""wind 20개만으로 12 horizon 예측. 영상 없이 어디까지 가는지 재는 기준선.

규정: train(9,607) 으로만 학습한다. validation(1,199) 은 성능 확인 전용 — 학습에 넣지 않는다
(RULES.md §1 7항). 출력은 preds/*.npy 와 로그 한 줄.
"""
import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
OUT = pathlib.Path("/home/jovyan/work/preds")
OUT.mkdir(parents=True, exist_ok=True)
W = [f"wind_{i:02d}" for i in range(20)]
T = [f"target_{i:02d}" for i in range(12)]


def feats(df):
    w = df[W].to_numpy(np.float64)
    d = np.diff(w, axis=1)
    return np.column_stack([
        w, d,
        w.mean(1), w.std(1), w.min(1), w.max(1),
        w[:, -1], w[:, -1] - w[:, -2], w[:, -1] - w[:, -4], w[:, -1] - w[:, 0],
        w[:, -4:].mean(1), w[:, -8:].mean(1),
        d[:, -4:].mean(1), np.abs(d).mean(1),
    ])


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


tr_x = pd.read_csv(ROOT / "train/inputs.csv")
tr_y = pd.read_csv(ROOT / "train/targets.csv")[T].to_numpy(np.float64)
va_x = pd.read_csv(ROOT / "validation/inputs.csv")
va_y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
te_x = pd.read_csv(ROOT / "test/inputs.csv")

Xtr, Xva, Xte = feats(tr_x), feats(va_x), feats(te_x)
persist = np.repeat(va_x[W[-1]].to_numpy(np.float64)[:, None], 12, axis=1)

preds = {}
preds["persist"] = (persist, np.repeat(te_x[W[-1]].to_numpy(np.float64)[:, None], 12, axis=1))

ridge = RidgeCV(alphas=np.logspace(-3, 4, 30)).fit(Xtr, tr_y)
preds["ridge"] = (ridge.predict(Xva), ridge.predict(Xte))

gv, gt = np.zeros_like(va_y), np.zeros((len(te_x), 12))
for h in range(12):
    m = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
        early_stopping=True, validation_fraction=0.12, random_state=777,
    ).fit(Xtr, tr_y[:, h])
    gv[:, h], gt[:, h] = m.predict(Xva), m.predict(Xte)
    print(f"  gbdt h{h:02d} val_rmse={rmse(va_y[:, h], gv[:, h]):.2f}", flush=True)
preds["gbdt"] = (gv, gt)
preds["ridge_gbdt"] = ((preds["ridge"][0] + gv) / 2, (preds["ridge"][1] + gt) / 2)

report = {}
for name, (pv, pt) in preds.items():
    np.save(OUT / f"val_{name}.npy", pv)
    np.save(OUT / f"test_{name}.npy", pt)
    report[name] = {
        "val_rmse": round(rmse(va_y, pv), 3),
        "per_h": [round(rmse(va_y[:, h], pv[:, h]), 1) for h in range(12)],
    }
(OUT / "windonly.json").write_text(json.dumps(report, indent=2))
for name, r in sorted(report.items(), key=lambda kv: kv[1]["val_rmse"]):
    print(f"{name:12s} val_rmse={r['val_rmse']:7.3f}  per_h={r['per_h']}", flush=True)
