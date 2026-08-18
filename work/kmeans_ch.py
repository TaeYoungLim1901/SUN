"""E015 — 193+211 결합 픽셀 k-means(k=3) 로 CH·조용한태양·활성영역을 나눈다. DOMAIN §6 #2·§9.4.

`0.35 × disk median` 임계는 닫힌 축이다 (PNG 가 per-image 재스케일이라 물리 임계가 아니다).
대신 두 채널의 **상대 강도쌍**을 군집해 분할한다. 군집 중심은 **train 이미지에서만** 적합한다.

산출: preds/val_chkm.npy · test_chkm.npy (+ chkm.json)
사용: python work/kmeans_ch.py
"""
import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
CACHE = pathlib.Path("/home/jovyan/work/resized_cache/64px")
OUT = pathlib.Path("/home/jovyan/work/preds")
IMG = [f"image_{i:02d}" for i in range(20)]
W = [f"wind_{i:02d}" for i in range(20)]
T = [f"target_{i:02d}" for i in range(12)]
SEED = 777
GRID = 4
TAG = "chkm"


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def disk_mask(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    return ((yy - h / 2 + .5) ** 2 + (xx - w / 2 + .5) ** 2) <= (0.47 * h) ** 2


def ratios(split):
    """(n, 2, npix) — 디스크 안 픽셀을 자기 이미지 중앙값으로 나눈 상대 강도."""
    a = np.asarray(np.load(CACHE / f"{split}_images.npy", mmap_mode="r"), np.float32)
    n, c, h, w = a.shape
    d = disk_mask(h, w)
    flat = a.reshape(n, c, -1)[:, :, d.ravel()]
    med = np.median(flat, axis=2, keepdims=True)
    return flat / np.maximum(med, 1.0), d


def fit_centers(r):
    """train 상대강도에서만 k=3 중심을 잡는다. 어두운 순으로 정렬 (0=CH)."""
    rng = np.random.default_rng(SEED)
    n = r.shape[0]
    rows = rng.choice(n, size=min(400, n), replace=False)
    px = rng.choice(r.shape[2], size=min(1500, r.shape[2]), replace=False)
    sample = r[rows][:, :, px].transpose(0, 2, 1).reshape(-1, 2)
    km = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(sample)
    ctr = km.cluster_centers_
    return ctr[np.argsort(ctr.sum(1))]


def label(r, ctr):
    """(n, npix) 클러스터 라벨. 거리 계산을 직접 해서 이미지 전체를 한 번에 훑는다."""
    x = r.transpose(0, 2, 1)                                  # (n, npix, 2)
    d = ((x[:, :, None, :] - ctr[None, None]) ** 2).sum(-1)   # (n, npix, 3)
    return d.argmin(-1).astype(np.int8)


def per_image(split, ctr):
    r, dmask = ratios(split)
    n = r.shape[0]
    lab = np.empty((n, r.shape[2]), np.int8)
    for i in range(0, n, 512):                                # 메모리 상한
        lab[i:i + 512] = label(r[i:i + 512], ctr)
    h = w = int(np.sqrt(dmask.size))
    yy, xx = np.mgrid[0:h, 0:w]
    ys, xs = yy[dmask].astype(np.float32), xx[dmask].astype(np.float32)
    ch = lab == 0
    area = ch.mean(1)
    out = [area, (lab == 1).mean(1), (lab == 2).mean(1)]
    out += [np.nanmean(np.where(lab == k, r[:, 0], np.nan), 1) for k in range(3)]     # 193 클래스 평균
    out += [np.nanmean(np.where(lab == k, r[:, 1], np.nan), 1) for k in range(3)]     # 211
    tot = np.maximum(ch.sum(1), 1)
    cx, cy = (ch * xs).sum(1) / tot, (ch * ys).sum(1) / tot                    # CH 무게중심
    out += [cx / w, cy / h, np.sqrt(((ch * (xs - cx[:, None]) ** 2).sum(1)) / tot) / w]
    # 격자별 CH 면적 + 서쪽 경도 스트립 (자전해 지구를 향해 오는 쪽)
    gy, gx = (ys // (h / GRID)).astype(int), (xs // (w / GRID)).astype(int)
    for a in range(GRID):
        for b in range(GRID):
            m = (gy == a) & (gx == b)
            out.append(ch[:, m].mean(1) if m.any() else np.zeros(n, np.float32))
    for lo, hi in ((0.55, 0.75), (0.75, 0.95)):
        m = (xs >= lo * w) & (xs < hi * w)
        out.append(ch[:, m].mean(1))
    f = np.nan_to_num(np.column_stack(out).astype(np.float32))
    print(f"{split} per-image {f.shape} CH면적 평균 {area.mean():.3f}", flush=True)
    return f


def sample_feats(split, inputs, ctr, cache={}):
    if split not in cache:
        cache[split] = per_image(split, ctr)
    per = cache[split]
    names = sorted(pd.unique(inputs[IMG].to_numpy().ravel()).tolist())
    idx = {v: i for i, v in enumerate(names)}
    seq = np.stack([per[[idx[v] for v in inputs[col]]] for col in IMG], axis=1)   # (N,20,F)
    w = inputs[W].to_numpy(np.float64)
    return np.column_stack([
        w, np.diff(w, axis=1), w.mean(1), w.std(1), w.min(1), w.max(1),
        w[:, -1] - w[:, -4], w[:, -4:].mean(1), w[:, -8:].mean(1),
        seq[:, -1], seq.mean(1), seq.std(1),
        seq[:, -4:].mean(1) - seq[:, :4].mean(1),
        seq[:, -8:, :3].reshape(len(w), -1),        # 클래스 면적 3종만 시계열 통째로
    ])


def main():
    tr_x = pd.read_csv(ROOT / "train/inputs.csv")
    va_x = pd.read_csv(ROOT / "validation/inputs.csv")
    te_x = pd.read_csv(ROOT / "test/inputs.csv")
    tr_y = pd.read_csv(ROOT / "train/targets.csv")[T].to_numpy(np.float64)
    va_y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)

    r, _ = ratios("train")
    ctr = fit_centers(r)
    del r
    print("centers (193,211)", np.round(ctr, 3).tolist(), flush=True)

    Xtr = sample_feats("train", tr_x, ctr)
    Xva = sample_feats("validation", va_x, ctr)
    Xte = sample_feats("test", te_x, ctr)
    print("X", Xtr.shape, Xva.shape, Xte.shape, flush=True)

    pv, pt = np.zeros_like(va_y), np.zeros((len(te_x), 12))
    for h in range(12):
        m = HistGradientBoostingRegressor(
            max_iter=700, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.12, n_iter_no_change=30,
            random_state=SEED).fit(Xtr, tr_y[:, h])
        pv[:, h], pt[:, h] = m.predict(Xva), m.predict(Xte)
        print(f"  h{h:02d} val={rmse(va_y[:, h], pv[:, h]):.2f}", flush=True)

    np.save(OUT / f"val_{TAG}.npy", pv)
    np.save(OUT / f"test_{TAG}.npy", pt)
    val = rmse(va_y, pv)
    (OUT / f"{TAG}.json").write_text(json.dumps(
        {"val_rmse": round(val, 3), "centers": ctr.tolist(),
         "per_h": [round(rmse(va_y[:, h], pv[:, h]), 1) for h in range(12)]}, indent=2))
    print(f"{TAG} val_rmse={val:.3f}", flush=True)


if __name__ == "__main__":
    main()
