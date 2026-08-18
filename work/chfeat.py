"""E017 — CH 궤적을 v2 GBDT 의 추가 피처로만 넣는다. E012 재개 조건.

E012 는 궤적만으로 71.979, hill_climb 이 안 뽑았다. 여기서는 v2 839차원을 유지하고
`chtrack.seq_feats` 를 옆에 붙인다. 창을 좁히지 않는다.

산출: preds/val_chfeat.npy · test_chfeat.npy
사용: python work/chfeat.py
채택선: v2_gbdt 67.022 를 이길 것.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "/home/jovyan/work")
from chtrack import seq_feats  # noqa: E402
from feats import sample_feats  # noqa: E402

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
OUT = pathlib.Path("/home/jovyan/work/preds")
T = [f"target_{i:02d}" for i in range(12)]
SEED = 777
TAG = "chfeat"


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main():
    xs = {s: pd.read_csv(ROOT / f"{s}/inputs.csv") for s in ("train", "validation", "test")}
    tr_y = pd.read_csv(ROOT / "train/targets.csv")[T].to_numpy(np.float64)
    va_y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
    X = {s: np.column_stack([sample_feats(s, x), seq_feats(s, x)]) for s, x in xs.items()}
    print("X", X["train"].shape, flush=True)

    pv, pt = np.zeros_like(va_y), np.zeros((len(xs["test"]), 12))
    for h in range(12):
        m = HistGradientBoostingRegressor(
            max_iter=700, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.12, n_iter_no_change=30,
            random_state=SEED).fit(X["train"], tr_y[:, h])
        pv[:, h] = m.predict(X["validation"])
        pt[:, h] = m.predict(X["test"])
        print(f"  h{h:02d} dim={X['train'].shape[1]} val={rmse(va_y[:, h], pv[:, h]):.2f}", flush=True)

    np.save(OUT / f"val_{TAG}.npy", pv)
    np.save(OUT / f"test_{TAG}.npy", pt)
    val = rmse(va_y, pv)
    (OUT / f"{TAG}.json").write_text(json.dumps(
        {"val_rmse": round(val, 3),
         "per_h": [round(rmse(va_y[:, h], pv[:, h]), 1) for h in range(12)]}, indent=2))
    print(f"{TAG} val_rmse={val:.3f}  (v2_gbdt 67.022 를 이겨야 채택)", flush=True)


if __name__ == "__main__":
    main()
