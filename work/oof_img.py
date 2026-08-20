import pathlib
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV

sys.path.insert(0, "/home/jovyan/work")
from imgfeat import sample_feats  # noqa: E402

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
OUT = pathlib.Path("/home/jovyan/work/preds")
T = [f"target_{i:02d}" for i in range(12)]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main():
    x = pd.read_csv(ROOT / "train/inputs.csv")
    y = pd.read_csv(ROOT / "train/targets.csv")[T].to_numpy(np.float64)
    fold = np.load(OUT / "train_fold.npy")
    assert fold.shape == (len(x),) and np.array_equal(np.unique(fold), np.arange(5))
    X = sample_feats("train", x)
    assert X.shape[0] == len(x)
    print("features", X.shape, "folds", np.bincount(fold), flush=True)

    ridge = np.zeros_like(y)
    for f in range(5):
        train = fold != f
        ridge[~train] = RidgeCV(alphas=np.logspace(-3, 4, 30)).fit(X[train], y[train]).predict(X[~train])
        print(f"ridge f{f} rmse={rmse(y[~train], ridge[~train]):.3f}", flush=True)
    np.save(OUT / "oof_img_ridge.npy", ridge)

    gbdt = np.zeros_like(y)
    for h in range(12):
        t0 = time.time()
        for f in range(5):
            train = fold != f
            model = HistGradientBoostingRegressor(
                max_iter=500, learning_rate=0.06, max_leaf_nodes=31,
                early_stopping=False, random_state=777,
            ).fit(X[train], y[train, h])
            gbdt[~train, h] = model.predict(X[~train])
        print(f"gbdt h{h:02d} rmse={rmse(y[:, h], gbdt[:, h]):.3f} {time.time() - t0:5.0f}s", flush=True)
    assert np.isfinite(ridge).all() and np.isfinite(gbdt).all()
    np.save(OUT / "oof_img_gbdt.npy", gbdt)
    print("ridge total", rmse(y, ridge), "gbdt total", rmse(y, gbdt), flush=True)


if __name__ == "__main__":
    main()
