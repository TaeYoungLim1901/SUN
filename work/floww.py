"""E022c — 광류 + wind20 잔차 HGBR. 1563차원 omniflow 가 OOM 나는 동안의 중간판.

사용: python work/floww.py
"""
import json
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
OUT = pathlib.Path("/home/jovyan/work/preds")
FCACHE = pathlib.Path("/home/jovyan/work/flow_cache")
W = [f"wind_{i:02d}" for i in range(20)]
T = [f"target_{i:02d}" for i in range(12)]
SEED = 777
TAG = "floww"


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def wind_block(x):
    w = x[W].to_numpy(np.float64)
    d = np.diff(w, axis=1)
    return np.column_stack([
        w, d, w.mean(1), w.std(1), w.min(1), w.max(1),
        w[:, -1] - w[:, -2], w[:, -1] - w[:, -4],
        w[:, -4:].mean(1), w[:, -8:].mean(1),
    ])


def main():
    xs = {s: pd.read_csv(ROOT / f"{s}/inputs.csv") for s in ("train", "validation", "test")}
    tr_y = pd.read_csv(ROOT / "train/targets.csv")[T].to_numpy(np.float64)
    va_y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
    last = {s: x[W[-1]].to_numpy(np.float64) for s, x in xs.items()}
    X = {s: np.column_stack([np.load(FCACHE / f"{s}_X.npy"), wind_block(xs[s])]) for s in xs}
    print("X", X["train"].shape, flush=True)
    pv = np.zeros_like(va_y)
    pt = np.zeros((len(xs["test"]), 12), np.float64)
    for h in range(12):
        m = HistGradientBoostingRegressor(
            max_iter=700, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.12, n_iter_no_change=30,
            random_state=SEED)
        m.fit(X["train"], tr_y[:, h] - last["train"])
        pv[:, h] = m.predict(X["validation"]) + last["validation"]
        pt[:, h] = m.predict(X["test"]) + last["test"]
        print(f"  {TAG} h{h:02d} val={rmse(va_y[:, h], pv[:, h]):.2f}", flush=True)
    assert np.isfinite(pv).all() and np.isfinite(pt).all()
    np.save(OUT / f"val_{TAG}.npy", pv)
    np.save(OUT / f"test_{TAG}.npy", pt)
    val = rmse(va_y, pv)
    (OUT / f"{TAG}.json").write_text(json.dumps({
        "val_rmse": round(val, 3),
        "per_h": [round(rmse(va_y[:, h], pv[:, h]), 1) for h in range(12)],
        "dim": int(X["train"].shape[1]),
    }, indent=2))
    print(f"{TAG} val_rmse={val:.3f}", flush=True)


if __name__ == "__main__":
    main()
