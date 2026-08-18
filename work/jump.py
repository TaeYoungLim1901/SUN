"""E019 — 마지막 풍속 대비 잔차를 v2 피처로 예측. CNN 잔차 타깃을 스칼라에 이식.

산출: preds/val_jump.npy · test_jump.npy
채택: v2 67.022 를 이기거나 hill 이 6종에 넣고 결합이 0.05 이상 내릴 것.
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
TAG = "jump"


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main():
    xs = {s: pd.read_csv(ROOT / f"{s}/inputs.csv") for s in ("train", "validation", "test")}
    tr_y = pd.read_csv(ROOT / "train/targets.csv")[T].to_numpy(np.float64)
    va_y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
    last = {s: x[W[-1]].to_numpy(np.float64) for s, x in xs.items()}
    X = {s: sample_feats(s, x) for s, x in xs.items()}
    print("X", X["train"].shape, flush=True)

    pv, pt = np.zeros_like(va_y), np.zeros((len(xs["test"]), 12))
    models = []
    for h in range(12):
        m = HistGradientBoostingRegressor(
            max_iter=700, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.12, n_iter_no_change=30,
            random_state=SEED).fit(X["train"], tr_y[:, h] - last["train"])
        models.append(m)
        pv[:, h] = m.predict(X["validation"]) + last["validation"]
        pt[:, h] = m.predict(X["test"]) + last["test"]
        print(f"  h{h:02d} val={rmse(va_y[:, h], pv[:, h]):.2f}", flush=True)

    import torch
    mp = pathlib.Path("/home/jovyan/work/jump_models.pth")
    torch.save(models, mp)
    print("saved", mp, mp.stat().st_size / 1e6, "MB", flush=True)

    np.save(OUT / f"val_{TAG}.npy", pv)
    np.save(OUT / f"test_{TAG}.npy", pt)
    val = rmse(va_y, pv)
    (OUT / f"{TAG}.json").write_text(json.dumps({
        "val_rmse": round(val, 3),
        "per_h": [round(rmse(va_y[:, h], pv[:, h]), 1) for h in range(12)],
    }, indent=2))
    print(f"{TAG} val_rmse={val:.3f}  (v2 67.022 / 결합 Δ0.05)", flush=True)


if __name__ == "__main__":
    main()
