import pathlib
import sys
import time
import gc

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "/home/jovyan/work")
import eswf  # noqa: E402

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
OUT = pathlib.Path("/home/jovyan/work/preds")
T = [f"target_{i:02d}" for i in range(12)]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)))


def main():
    x = pd.read_csv(ROOT / "train/inputs.csv")
    y = pd.read_csv(ROOT / "train/targets.csv")[T].to_numpy(np.float64)
    fold = np.load(OUT / "train_fold.npy")
    assert fold.shape == (len(x),) and np.array_equal(np.unique(fold), np.arange(5))
    eswf.SEQ = {"train": eswf.sequences("train", x)}
    oof = np.zeros_like(y)
    print("sequence", eswf.SEQ["train"].shape, "folds", np.bincount(fold), flush=True)
    for h in range(12):
        t0 = time.time()
        X = eswf.build("train", x, 6.0 * (h + 1)).astype(np.float32, copy=False)
        for f in range(5):
            train = fold != f
            model = HistGradientBoostingRegressor(
                max_iter=700, learning_rate=0.05, max_leaf_nodes=31,
                l2_regularization=1.0, early_stopping=False, random_state=777,
            ).fit(X[train], y[train, h])
            oof[~train, h] = model.predict(X[~train])
            del model
            gc.collect()
        print(f"h{h:02d} dim={X.shape[1]} rmse={rmse(y[:, h], oof[:, h]):.3f} {time.time() - t0:5.0f}s", flush=True)
    assert oof.shape == (9607, 12) and np.isfinite(oof).all()
    np.save(OUT / "oof_eswf.npy", oof)
    print("total oof rmse", rmse(y, oof), flush=True)


if __name__ == "__main__":
    main()
