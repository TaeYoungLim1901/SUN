"""E018 — 도착 국면만 가중한 v2 GBDT.

HSS 병목은 지속이 아니라 도착(last~494, peak>600).
y>600 전체 가중은 이미 빠른 구간까지 건드리고 졌다(E010w, E011c 단독).
여기서는 last_wind<500 이고 max(y)>600 인 train 행만 가중 3.

산출: preds/val_arrive.npy · test_arrive.npy
채택선: v2 67.022 를 이기거나, 6종 결합에 hill_climb 이 넣을 것.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "/home/jovyan/work")
from feats import sample_feats  # noqa: E402

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
OUT = pathlib.Path("/home/jovyan/work/preds")
T = [f"target_{i:02d}" for i in range(12)]
W = [f"wind_{i:02d}" for i in range(20)]
SEED = 777
TAG = "arrive"


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main():
    xs = {s: pd.read_csv(ROOT / f"{s}/inputs.csv") for s in ("train", "validation", "test")}
    tr_y = pd.read_csv(ROOT / "train/targets.csv")[T].to_numpy(np.float64)
    va_y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
    last = xs["train"][W[-1]].to_numpy(np.float64)
    sw = np.where((last < 500.0) & (tr_y.max(1) > 600.0), 3.0, 1.0)
    print("arrive rows", int((sw > 1).sum()), "/", len(sw), flush=True)

    X = {s: sample_feats(s, x) for s, x in xs.items()}
    print("X", X["train"].shape, flush=True)
    pv, pt = np.zeros_like(va_y), np.zeros((len(xs["test"]), 12))
    for h in range(12):
        m = HistGradientBoostingRegressor(
            max_iter=700, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.12, n_iter_no_change=30,
            random_state=SEED).fit(X["train"], tr_y[:, h], sample_weight=sw)
        pv[:, h] = m.predict(X["validation"])
        pt[:, h] = m.predict(X["test"])
        print(f"  h{h:02d} val={rmse(va_y[:, h], pv[:, h]):.2f}", flush=True)

    np.save(OUT / f"val_{TAG}.npy", pv)
    np.save(OUT / f"test_{TAG}.npy", pt)
    val = rmse(va_y, pv)
    (OUT / f"{TAG}.json").write_text(json.dumps({
        "val_rmse": round(val, 3),
        "n_arrive": int((sw > 1).sum()),
        "per_h": [round(rmse(va_y[:, h], pv[:, h]), 1) for h in range(12)],
    }, indent=2))
    print(f"{TAG} val_rmse={val:.3f}  (v2 67.022 / 결합 hill)", flush=True)


if __name__ == "__main__":
    main()
