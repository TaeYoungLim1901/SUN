"""Ballistic ToF occupancy — longitude-strip dark fraction per velocity hypothesis.

SEARCH.md §2-3. Window is not narrowed. Each horizon gathers occupancy at the
longitude that ballistic ToF maps onto the current disk.

Usage: python work/tof.py
"""
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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
PX = 96
REL = 0.45
R_FRAC = 0.47
HALF_DEG = 7.5
LOWLAT = 0.5
AU_KM = 1.496e8
DEG_PER_DAY = 13.2
V_HYP = (350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 700.0)
N_V = len(V_HYP)
N_STRIP = N_V * 2  # all-lat + low-lat per v
CHUNK = 256


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def geom(h=PX, w=PX):
    yy, xx = np.mgrid[:h, :w].astype(np.float64)
    radius = R_FRAC * h
    dx = (xx - (w / 2 - 0.5)) / radius
    dy = (yy - (h / 2 - 0.5)) / radius
    disk = (dx * dx + dy * dy) <= 1.0
    lon = np.degrees(np.arcsin(np.clip(dx, -1.0, 1.0)))
    return disk, lon, dy


def phi_deg(t_h, v):
    t_f = AU_KM / v / 3600.0
    return DEG_PER_DAY * (t_h - t_f) / 24.0


def strip_stack(disk, lon, dy):
    """(12, 14, H, W) bool: horizon × (7v × {all, low})."""
    h, w = disk.shape
    out = np.zeros((12, N_STRIP, h, w), np.bool_)
    for hi in range(12):
        t_h = 6.0 * (hi + 1)
        for vi, v in enumerate(V_HYP):
            phi = phi_deg(t_h, v)
            band = disk & (np.abs(lon - phi) <= HALF_DEG)
            out[hi, 2 * vi] = band
            out[hi, 2 * vi + 1] = band & (np.abs(dy) < LOWLAT)
    return out


def occ_unique(arr, disk, strips):
    """Unique-image occupancy. strips (S,H,W) → (n_unique, 2, S)."""
    n, c, h, w = arr.shape
    S = strips.shape[0]
    sf = strips.reshape(S, -1).astype(np.float32)
    counts = sf.sum(1)
    counts_safe = np.where(counts > 0, counts, 1.0)
    dflat = disk.ravel()
    out = np.empty((n, c, S), np.float32)
    for i0 in range(0, n, CHUNK):
        sl = np.asarray(arr[i0:i0 + CHUNK], dtype=np.float32)
        k = sl.shape[0]
        pix = sl.reshape(k, c, -1)
        med = np.median(pix[:, :, dflat], axis=2)
        dark = pix < (REL * med[:, :, None])
        occ = dark @ sf.T / counts_safe
        occ[:, :, counts <= 0] = np.nan
        out[i0:i0 + k] = occ
        print(f"    unique {min(i0 + k, n)}/{n}", flush=True)
    return out


def load_split(split, inputs, disk, strips):
    arr = np.load(CACHE / f"{split}_images.npy", mmap_mode="r")
    names = sorted(pd.unique(inputs[IMG].to_numpy().ravel()).tolist())
    assert arr.shape == (len(names), 2, PX, PX), (split, arr.shape, len(names))
    idx = {name: i for i, name in enumerate(names)}
    slots = np.asarray([[idx[name] for name in inputs[col]] for col in IMG],
                       dtype=np.int32).T
    assert slots.shape == (len(inputs), 20)
    print(f"{split} unique={len(names)} occupancy…", flush=True)
    occ_u = occ_unique(arr, disk, strips.reshape(-1, PX, PX))
    occ_u = occ_u.reshape(len(names), 2, 12, N_STRIP)
    n = len(slots)
    last = occ_u[slots[:, -1]]
    last4 = np.zeros((n, 2, 12, N_STRIP), np.float32)
    all20 = np.zeros((n, 2, 12, N_STRIP), np.float32)
    for t in range(20):
        fr = occ_u[slots[:, t]]
        all20 += fr
        if t >= 16:
            last4 += fr
    last4 /= 4.0
    all20 /= 20.0
    del occ_u
    img = np.stack((last, last4, all20), axis=1)
    img = np.moveaxis(img, 3, 1)
    img = img.reshape(n, 12, 3 * 2 * N_STRIP)
    del last, last4, all20
    wind = inputs[WIND].to_numpy(np.float64)
    dw = np.diff(wind, axis=1, prepend=wind[:, :1])
    wind_block = np.column_stack((wind, dw))
    print(f"{split} img={img.shape} wind={wind_block.shape}", flush=True)
    return img, wind_block


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = {s: pd.read_csv(ROOT / s / "inputs.csv") for s in ("train", "validation", "test")}
    assert len(inputs["train"]) == 9607
    assert len(inputs["validation"]) == 1199
    assert len(inputs["test"]) == 3868
    y_train = np.abs(pd.read_csv(ROOT / "train" / "targets.csv")[TARGET].to_numpy(np.float64))
    y_val = np.abs(pd.read_csv(ROOT / "validation" / "targets.csv")[TARGET].to_numpy(np.float64))
    disk, lon, dy = geom()
    strips = strip_stack(disk, lon, dy)
    n_pix = strips.reshape(12, N_STRIP, -1).sum(-1)
    print("phi_deg", [[round(phi_deg(6.0 * (h + 1), v), 2) for v in V_HYP]
                      for h in range(12)], flush=True)
    print("strip pixels min/median/max", int(n_pix.min()), int(np.median(n_pix)),
          int(n_pix.max()), flush=True)
    Ximg, Xw = {}, {}
    for s in ("train", "validation", "test"):
        Ximg[s], Xw[s] = load_split(s, inputs[s], disk, strips)
    val_pred = np.empty_like(y_val)
    test_pred = np.empty((len(inputs["test"]), 12), np.float64)
    dim = None
    for h in range(12):
        X = {s: np.column_stack((Ximg[s][:, h], Xw[s])) for s in Ximg}
        dim = int(X["train"].shape[1])
        assert dim == 124, dim
        assert len(X["train"]) == 9607
        model = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.12, n_iter_no_change=30, random_state=SEED,
        )
        model.fit(X["train"], y_train[:, h])
        val_pred[:, h] = model.predict(X["validation"])
        test_pred[:, h] = model.predict(X["test"])
        print(f"h{h:02d} dim={dim} val_rmse={rmse(y_val[:, h], val_pred[:, h]):.4f}",
              flush=True)
    assert val_pred.shape == (1199, 12) and test_pred.shape == (3868, 12)
    assert np.isfinite(val_pred).all() and np.isfinite(test_pred).all()
    np.save(OUT / "val_tof.npy", val_pred)
    np.save(OUT / "test_tof.npy", test_pred)
    score = rmse(y_val, val_pred)
    (OUT / "tof.json").write_text(json.dumps({
        "val_rmse": round(score, 4),
        "dim": dim,
        "pred_mean": round(float(val_pred.mean()), 4),
        "per_h": [round(rmse(y_val[:, h], val_pred[:, h]), 2) for h in range(12)],
    }, indent=2), encoding="utf-8")
    print(f"tof val_rmse={score:.4f} mean={val_pred.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
