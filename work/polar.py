"""Polar longitude-waterfall component."""
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path("/home/jovyan/public_dataset/competition_dataset_6h")
CACHE = Path("/home/jovyan/work/resized_cache/96px")
OUT = Path("/home/jovyan/work/preds")
IMG = [f"image_{i:02d}" for i in range(20)]
WIND = [f"wind_{i:02d}" for i in range(20)]
TARGET = [f"target_{i:02d}" for i in range(12)]
SEED = 777


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def polar_bins(h=96, w=96):
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    radius = 0.47 * h
    dx = xx - (w / 2 - 0.5)
    dy = yy - (h / 2 - 0.5)
    rho = np.sqrt(dx * dx + dy * dy)
    disk = rho <= radius
    theta = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    rbin = np.minimum((rho / radius * 6).astype(np.int8), 5)
    tbin = (theta / (2 * np.pi) * 36).astype(np.int8)
    bid = tbin[disk].astype(np.int16) * 6 + rbin[disk].astype(np.int16)
    counts = np.bincount(bid, minlength=36 * 6).astype(np.float32)
    assert counts.min() > 0
    return disk, bid, counts


def frame_features(arr, disk, bid, counts):
    out = np.empty((arr.shape[0], 36, 2), np.float32)
    for i in range(arr.shape[0]):
        for c in range(2):
            image = arr[i, c]
            med = float(np.median(image[disk]))
            dark = image[disk] < 0.45 * med
            occupied = np.bincount(bid, weights=dark, minlength=216) / counts
            out[i, :, c] = occupied.reshape(36, 6).mean(axis=1)
    return out


def load_split(split, inputs):
    arr = np.load(CACHE / f"{split}_images.npy", mmap_mode="r")
    names = sorted(pd.unique(inputs[IMG].to_numpy().ravel()).tolist())
    assert arr.shape == (len(names), 2, 96, 96), (split, arr.shape, len(names))
    idx = {name: i for i, name in enumerate(names)}
    slots = np.asarray([[idx[name] for name in inputs[col]] for col in IMG], dtype=np.int32).T
    assert slots.shape == (len(inputs), 20)
    disk, bid, counts = polar_bins(arr.shape[2], arr.shape[3])
    frames = frame_features(arr, disk, bid, counts)
    seq = frames[slots]
    stats = np.stack((
        seq[:, -1],
        seq.mean(axis=1),
        seq.std(axis=1),
        seq[:, -4:].mean(axis=1) - seq[:, :4].mean(axis=1),
    ), axis=-1).reshape(len(inputs), -1)
    wind = inputs[WIND].to_numpy(np.float64)
    features = np.column_stack((stats, wind, np.diff(wind, axis=1, prepend=wind[:, :1])))
    assert features.shape == (len(inputs), 328)
    print(f"{split} unique={len(names)} X={features.shape}", flush=True)
    return features


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = {s: pd.read_csv(ROOT / s / "inputs.csv") for s in ("train", "validation", "test")}
    assert len(inputs["train"]) == 9607
    assert len(inputs["validation"]) == 1199
    assert len(inputs["test"]) == 3868
    y_train = np.abs(pd.read_csv(ROOT / "train" / "targets.csv")[TARGET].to_numpy(np.float64))
    y_val = np.abs(pd.read_csv(ROOT / "validation" / "targets.csv")[TARGET].to_numpy(np.float64))
    X = {s: load_split(s, inputs[s]) for s in inputs}
    val_pred = np.empty_like(y_val)
    test_pred = np.empty((len(inputs["test"]), 12), np.float64)
    for h in range(12):
        model = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.12, n_iter_no_change=30, random_state=SEED,
        )
        model.fit(X["train"], y_train[:, h])
        val_pred[:, h] = model.predict(X["validation"])
        test_pred[:, h] = model.predict(X["test"])
        print(f"h{h:02d} val_rmse={rmse(y_val[:, h], val_pred[:, h]):.4f}", flush=True)
    assert val_pred.shape == (1199, 12) and test_pred.shape == (3868, 12)
    assert np.isfinite(val_pred).all() and np.isfinite(test_pred).all()
    np.save(OUT / "val_polar.npy", val_pred)
    np.save(OUT / "test_polar.npy", test_pred)
    score = rmse(y_val, val_pred)
    (OUT / "polar.json").write_text(json.dumps({"val_rmse": score}), encoding="utf-8")
    print(f"polar val_rmse={score:.4f}", flush=True)


if __name__ == "__main__":
    main()
