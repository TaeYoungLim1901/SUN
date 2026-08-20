"""코로나 홀 기하와 전파 지연 기반 태양풍 GBDT."""
import pathlib
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
WORK = pathlib.Path("/home/jovyan/work")
OUT = WORK / "preds"
IMG = [f"image_{i:02d}" for i in range(20)]
WIND = [f"wind_{i:02d}" for i in range(20)]
TARGET = [f"target_{i:02d}" for i in range(12)]
TH = np.arange(-77, 41, 7.0)
LAG = 3.69 - 0.07386 * TH
HORIZON = 0.25 * (np.arange(12) + 1)
F = 42


def image_features(split, inputs):
    names = sorted(pd.unique(inputs[IMG].to_numpy().ravel()).tolist())
    image_path = WORK / "resized_cache" / "96px" / f"{split}_images.npy"
    if not image_path.exists():
        raise SystemExit(f"missing image cache: {image_path}")
    idx = {name: i for i, name in enumerate(names)}
    path = OUT / f"phys_{split}.npy"
    if path.exists():
        out = np.load(path, mmap_mode="r")
        if out.shape != (len(names), F):
            raise ValueError(f"bad physics cache {path}: {out.shape}")
        return idx, out

    arr = np.load(image_path, mmap_mode="r")
    if arr.shape != (len(names), 2, 96, 96):
        raise ValueError(f"bad image cache {image_path}: {arr.shape}")
    yy, xx = np.meshgrid(np.arange(96), np.arange(96), indexing="ij")
    r = 0.47 * 96
    dx, dy = (xx - 47.5) / r, (yy - 47.5) / r
    disk = dx * dx + dy * dy <= 1
    mu = np.sqrt(np.clip(1 - dx * dx - dy * dy, 1e-3, 1))
    lon = np.degrees(np.arcsin(np.clip(dx, -1, 1)))
    lat = np.degrees(np.arcsin(np.clip(dy, -1, 1)))
    swath = ((np.abs(lon[None] - TH[:, None, None]) < 7) &
             (np.abs(lat)[None] < 60) & disk[None])
    out = np.empty((len(names), F), dtype=np.float32)
    for i, frame in enumerate(arr):
        i193, i211 = frame
        ch = (i193 < 0.35 * np.median(i193[disk])) & disk
        ch2 = ch & (i211 < 0.45 * np.median(i211[disk]))
        den = swath.sum(axis=(1, 2))
        occ = np.divide((swath & ch[None]).sum(axis=(1, 2)), den,
                        out=np.zeros(TH.size, dtype=np.float64), where=den > 0)
        vhat = 330 + 930 * np.interp(HORIZON, LAG[::-1], occ[::-1])
        if ch.any():
            ch_lon = lon[ch]
            phi_co = np.degrees(np.arcsin(np.clip(np.mean(dy[ch]), -1, 1)))
            area = (ch / mu).sum() * (6.96e5 / r) ** 2
            thw, t_pk = ch_lon.max(), 3.69 - 0.07386 * ch_lon.mean()
            t_si = 3.69 - 0.07386 * thw
            west = ch & (lon >= np.percentile(ch_lon, 75))
            edge = i193[west].mean() - i193[ch].mean()
            phi_cm = np.degrees(np.arcsin(np.clip(np.mean(dx[ch]), -1, 1)))
        else:
            area = phi_co = thw = t_si = t_pk = edge = phi_cm = 0.0
        if ch2.any():
            dphi = lon[ch2].max() - lon[ch2].min()
        else:
            dphi = 0.0
        vmax = 330.8 + 5.7 * dphi if dphi < 67 else 710.0
        v_hof = 478 + (2.77e-9 * area) * (1 - abs(phi_co) / 61.4)
        out[i] = np.r_[occ, vhat, area, phi_co, v_hof, dphi, vmax, 0.09 * dphi,
                       thw, t_si, t_pk, edge, ch.mean(), ch2.mean(), phi_cm]
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return idx, out


def row_features(inputs, idx, per):
    seq = np.stack([per[[idx[name] for name in inputs[col]]] for col in IMG], axis=1)
    selected = seq[:, [19, 15, 11, 7, 3]].reshape(len(inputs), -1)
    stat = np.concatenate([seq.mean(axis=1), seq.std(axis=1)], axis=1)
    omega = (seq[:, 19, -1] - seq[:, 0, -1]) / 4.75
    rot = np.column_stack([omega, np.abs(omega - 13.2)])
    wind = inputs[WIND].to_numpy(np.float64)
    diff = np.diff(wind, axis=1)
    wf = np.column_stack([wind, diff, wind.mean(1), wind.std(1),
                           wind.min(1), wind.max(1)])
    return np.column_stack([selected, stat, rot, wf]).astype(np.float32)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def main():
    t0 = time.perf_counter()
    xs = {s: pd.read_csv(ROOT / s / "inputs.csv") for s in ("train", "validation", "test")}
    ys = {s: pd.read_csv(ROOT / s / "targets.csv")[TARGET].to_numpy(np.float64)
          for s in ("train", "validation")}
    feats = {}
    for split, inputs in xs.items():
        idx, values = image_features(split, inputs)
        feats[split] = row_features(inputs, idx, values)
    assert feats["train"].shape[1] > 0 and np.isfinite(feats["train"]).all()
    print(f"feature count: {feats['train'].shape[1]}", flush=True)
    val = np.empty((len(xs["validation"]), 12), dtype=np.float64)
    test = np.empty((len(xs["test"]), 12), dtype=np.float64)
    for h in range(12):
        model = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=20, early_stopping=False, random_state=777)
        model.fit(feats["train"], ys["train"][:, h] - xs["train"][WIND[-1]].to_numpy())
        val[:, h] = model.predict(feats["validation"]) + xs["validation"][WIND[-1]].to_numpy()
        test[:, h] = model.predict(feats["test"]) + xs["test"][WIND[-1]].to_numpy()
        print(f"h{h:02d} val RMSE: {rmse(ys['validation'][:, h], val[:, h]):.4f}", flush=True)
    assert np.isfinite(val).all() and np.isfinite(test).all()
    np.save(OUT / "val_phys.npy", val)
    np.save(OUT / "test_phys.npy", test)
    print(f"total val RMSE: {rmse(ys['validation'], val):.4f}", flush=True)
    print(f"wall time: {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
