"""E022 — 픽셀 광류 족. DOMAIN §9.5.

유니크 연속 쌍만 Farneback. disk-median 정규화 프레임에서 계산해
밝기 드리프트가 흐름을 만들지 않게 한다.

물리 축약 (전디스크 + 적도 ±30°):
  어두운 화소 서향 수송속도, 중심자오선 어두운 면적 플럭스,
  최대 암부 구조의 자오선 도달 잔여시간, 창 안 암부 면적 증감률.

(a) flow     — 광류 피처만, last 잔차 HGBR
(b) omniflow — omni 행렬에 열 추가

사용: python work/flow.py
"""
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "/home/jovyan/work")
from align import aligned_block, seq_of  # noqa: E402
from feats import sample_feats  # noqa: E402
from t4ch import t4_block  # noqa: E402

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
CACHE = pathlib.Path("/home/jovyan/work/resized_cache/96px")
OUT = pathlib.Path("/home/jovyan/work/preds")
FCACHE = pathlib.Path("/home/jovyan/work/flow_cache")
OUT.mkdir(parents=True, exist_ok=True)
FCACHE.mkdir(parents=True, exist_ok=True)
IMG = [f"image_{i:02d}" for i in range(20)]
W = [f"wind_{i:02d}" for i in range(20)]
T = [f"target_{i:02d}" for i in range(12)]
SEED = 777
REL = 0.55
# synodic 27.2753 d → 6h 당 3.30°. 구면 z-성분 = 강체자전 시선면 vx.
OMEGA = np.deg2rad(360.0 * 0.25 / 27.2753)
EXCLUDE = {
    "persist", "rotter", "chtrack", "imgpt", "chkm", "submitted",
    "pack63s", "ridge", "ridge_gbdt", "v2_ridge",
}

cv2.setNumThreads(2)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def geom(h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 0.47 * h
    dx = (xx - w / 2 + .5) / r
    dy = (yy - h / 2 + .5) / r
    disk = (dx * dx + dy * dy) <= 1.0
    eq = disk & (np.abs(dy) < np.sin(np.deg2rad(30)))
    cm = disk & (np.abs(dx) < np.sin(np.deg2rad(15)))
    z = np.sqrt(np.maximum(0.0, 1.0 - dx * dx - dy * dy))
    vx_rig = (OMEGA * r * z).astype(np.float32)
    return disk, eq, cm, vx_rig, r, w / 2 - .5


def nrm(img, disk):
    med = float(np.median(img[disk]))
    med = med if med >= 1.0 else 1.0
    return np.clip(img.astype(np.float32) * (128.0 / med), 0, 255).astype(np.uint8)


def _cc(mask, fx, cx0, r):
    """최대 암부 / 동쪽(미통과) 최대 암부의 (t_cm_h, cx_Rsun, vx_px, area_frac)."""
    nan4 = (np.nan, np.nan, np.nan, np.nan)
    if int(mask.sum()) < 8:
        return nan4, nan4
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.view(np.uint8)
                                                     if mask.dtype == np.uint8
                                                     else mask.astype(np.uint8), 8)
    if n <= 1:
        return nan4, nan4
    areas = st[1:, cv2.CC_STAT_AREA].astype(np.float32)
    order = np.argsort(-areas)
    pix = float(mask.size)

    def pack(k):
        m = lab == k
        ys, xs = np.nonzero(m)
        cx = (xs.astype(np.float32).mean() - cx0) / r
        vx = float(fx[m].mean())
        t_h = (-cx * r / vx) * 6.0 if abs(vx) >= 0.05 else np.nan
        return t_h, float(cx), vx, float(m.sum()) / pix

    big = pack(int(order[0]) + 1)
    incoming = nan4
    for j in order:
        k = int(j) + 1
        m = lab == k
        xs = np.nonzero(m)[1].astype(np.float32)
        if xs.mean() < cx0:
            incoming = pack(k)
            break
    return big, incoming


def pair_ch(a, b, disk, eq, cm, vx_rig, r, cx0):
    """한 채널 연속쌍 → 영역 2 × 12 스칼라."""
    med_a = float(np.median(a[disk]))
    med_b = float(np.median(b[disk]))
    med_a = med_a if med_a >= 1.0 else 1.0
    med_b = med_b if med_b >= 1.0 else 1.0
    da = a < REL * med_a
    db = b < REL * med_b
    flow = cv2.calcOpticalFlowFarneback(
        nrm(a, disk), nrm(b, disk), None, 0.5, 3, 11, 3, 5, 1.2, 0)
    fx = flow[..., 0]
    out = []
    for region in (disk, eq):
        m = db & region
        n = int(m.sum())
        if n >= 4:
            vx = fx[m]
            vx_mean = float(vx.mean())
            vx_med = float(np.median(vx))
            vx_res = float((fx[m] - vx_rig[m]).mean())
        else:
            vx_mean = vx_med = vx_res = np.nan
        cm_m = m & cm
        nc = int(cm_m.sum())
        cm_flux = float(fx[cm_m].sum()) / (nc + 1e-6) * nc / (int(region.sum()) + 1e-6)
        cm_area = nc / (int((cm & region).sum()) + 1e-6)
        d_area = float((db & region).mean() - (da & region).mean())
        big, inc = _cc(m.astype(np.uint8), fx, cx0, r)
        out.extend([vx_mean, vx_med, vx_res, cm_flux, cm_area, d_area,
                    big[0], big[1], big[2], big[3],
                    inc[0], inc[1]])
    return np.asarray(out, np.float32)


def phase_shift(a, b, disk):
    fa = np.where(disk, nrm(a, disk), 0).astype(np.float32)
    fb = np.where(disk, nrm(b, disk), 0).astype(np.float32)
    (dx, dy), resp = cv2.phaseCorrelate(fa, fb)
    return np.asarray([dx, dy, resp], np.float32)


def pair_vec(img_a, img_b, g):
    disk, eq, cm, vx_rig, r, cx0 = g
    parts = [pair_ch(img_a[c], img_b[c], disk, eq, cm, vx_rig, r, cx0) for c in (0, 1)]
    parts += [phase_shift(img_a[c], img_b[c], disk) for c in (0, 1)]
    return np.concatenate(parts)


def unique_pairs(inputs):
    names = inputs[IMG].to_numpy()
    seen = {}
    for row in names:
        for t in range(19):
            seen.setdefault((row[t], row[t + 1]), None)
    return list(seen)


def build_pair_table(split, inputs, idx, arr, g):
    path = FCACHE / f"{split}_pairs.npy"
    keys = unique_pairs(inputs)
    key_path = FCACHE / f"{split}_keys.txt"
    F = 2 * 2 * 12 + 2 * 3
    if path.exists() and key_path.exists():
        stored = key_path.read_text(encoding="utf-8").splitlines()
        tab = np.load(path)
        if stored == [f"{a}\t{b}" for a, b in keys] and tab.shape == (len(keys), F):
            print(f"  {split} pair cache hit {tab.shape}", flush=True)
            return {(a, b): tab[i] for i, (a, b) in enumerate(keys)}
    tab = np.full((len(keys), F), np.nan, np.float32)
    t0 = time.perf_counter()
    vx_acc, vx_n = 0.0, 0
    for i, (na, nb) in enumerate(keys):
        tab[i] = pair_vec(arr[idx[na]], arr[idx[nb]], g)
        v = tab[i, 0]
        if np.isfinite(v):
            vx_acc += float(v)
            vx_n += 1
        if (i + 1) % 500 == 0 or i + 1 == len(keys):
            mean_vx = vx_acc / max(vx_n, 1)
            print(f"  {split} pairs {i+1}/{len(keys)}  "
                  f"{time.perf_counter()-t0:.0f}s  <vx193>={mean_vx:.3f}px/fr",
                  flush=True)
    np.save(path, tab)
    key_path.write_text("".join(f"{a}\t{b}\n" for a, b in keys), encoding="utf-8")
    return {(a, b): tab[i] for i, (a, b) in enumerate(keys)}


def sample_flow(inputs, table):
    names = inputs[IMG].to_numpy()
    n = len(names)
    f0 = next(iter(table.values()))
    seq = np.empty((n, 19, f0.shape[0]), np.float32)
    for i in range(n):
        row = names[i]
        for t in range(19):
            seq[i, t] = table[(row[t], row[t + 1])]
    last4 = np.nanmean(seq[:, -4:], axis=1)
    return np.column_stack([
        seq[:, -1],
        np.nanmean(seq, axis=1),
        np.nanstd(seq, axis=1),
        seq[:, -1] - seq[:, 0],
        last4,
    ]).astype(np.float32)


def load_split(split, inputs):
    arr = np.load(CACHE / f"{split}_images.npy", mmap_mode="r")
    names = sorted(pd.unique(inputs[IMG].to_numpy().ravel()).tolist())
    assert arr.shape[0] == len(names), (split, arr.shape, len(names))
    idx = {v: i for i, v in enumerate(names)}
    g = geom(arr.shape[2], arr.shape[3])
    print(f"{split} unique={len(names)} shape={arr.shape}", flush=True)
    table = build_pair_table(split, inputs, idx, arr, g)
    X = sample_flow(inputs, table)
    print(f"{split} flow X {X.shape} finite={np.isfinite(X).mean():.3f}", flush=True)
    return X


def fit_tag(tag, X, last, tr_y, va_y, n_test):
    pv = np.zeros_like(va_y)
    pt = np.zeros((n_test, 12), np.float64)
    for h in range(12):
        m = HistGradientBoostingRegressor(
            max_iter=700, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.12, n_iter_no_change=30,
            random_state=SEED)
        m.fit(X["train"], tr_y[:, h] - last["train"])
        pv[:, h] = m.predict(X["validation"]) + last["validation"]
        pt[:, h] = m.predict(X["test"]) + last["test"]
        print(f"  {tag} h{h:02d} dim={X['train'].shape[1]} "
              f"val={rmse(va_y[:, h], pv[:, h]):.2f}", flush=True)
    assert pv.shape == (len(va_y), 12) and pt.shape == (n_test, 12)
    assert np.isfinite(pv).all() and np.isfinite(pt).all(), tag
    np.save(OUT / f"val_{tag}.npy", pv)
    np.save(OUT / f"test_{tag}.npy", pt)
    val = rmse(va_y, pv)
    (OUT / f"{tag}.json").write_text(json.dumps({
        "val_rmse": round(val, 3),
        "per_h": [round(rmse(va_y[:, h], pv[:, h]), 1) for h in range(12)],
        "dim": int(X["train"].shape[1]),
    }, indent=2))
    print(f"{tag} val_rmse={val:.3f}", flush=True)
    return pv


def score(pv_flow, pv_omni, va_y):
    import kit
    print("== RESID ==", flush=True)
    vals = {}
    for f in sorted(OUT.glob("val_*.npy")):
        name = f.stem[4:]
        if name in EXCLUDE:
            continue
        v = np.load(f)
        if v.shape != va_y.shape or not np.isfinite(v).all():
            print(f"SKIP {name} {getattr(v, 'shape', None)}", flush=True)
            continue
        vals[name] = v.astype(np.float64)
    for n in ("flow", "omniflow"):
        if n not in vals:
            continue
        print(f"  {n:12s} rmse={kit.score(va_y, vals[n]):.3f}", flush=True)
        if "omni" in vals:
            print(f"    rho_omni={kit.resid_corr(vals[n], vals['omni'], va_y):.3f}", flush=True)
        if "img_blend" in vals:
            print(f"    rho_blend={kit.resid_corr(vals[n], vals['img_blend'], va_y):.3f}", flush=True)
    print(f"N_POOL {len(vals)} {sorted(vals)}", flush=True)
    w, s = kit.hill_climb(vals, va_y, n_iter=80)
    print(f"hill_climb {s:.3f}", flush=True)
    for n, wt in sorted(w.items(), key=lambda kv: -kv[1]):
        print(f"  {n:16s} {wt:.3f}", flush=True)
    (OUT / "flow_hill.json").write_text(json.dumps({
        "hill": round(float(s), 3),
        "weights": {k: round(v, 4) for k, v in w.items()},
        "n_pool": len(vals),
    }, indent=2))


def main():
    xs = {s: pd.read_csv(ROOT / f"{s}/inputs.csv") for s in ("train", "validation", "test")}
    tr_y = pd.read_csv(ROOT / "train/targets.csv")[T].to_numpy(np.float64)
    va_y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
    last = {s: x[W[-1]].to_numpy(np.float64) for s, x in xs.items()}

    flow = {s: load_split(s, xs[s]) for s in xs}
    np.save(FCACHE / "train_X.npy", flow["train"])
    np.save(FCACHE / "validation_X.npy", flow["validation"])
    np.save(FCACHE / "test_X.npy", flow["test"])

    print("fit flow…", flush=True)
    pv_flow = fit_tag("flow", flow, last, tr_y, va_y, len(xs["test"]))

    print("omni matrix…", flush=True)
    base = {s: sample_feats(s, x) for s, x in xs.items()}
    seq = {s: seq_of(s, x) for s, x in xs.items()}
    wind = {s: x[W].to_numpy(np.float64) for s, x in xs.items()}
    print("base", base["train"].shape, "flow", flow["train"].shape, flush=True)

    print("fit omniflow…", flush=True)
    pv_of = np.zeros_like(va_y)
    pt_of = np.zeros((len(xs["test"]), 12), np.float64)
    for h in range(12):
        X = {s: np.column_stack([
            base[s], t4_block(seq[s], h), aligned_block(seq[s], wind[s], h), flow[s],
        ]) for s in xs}
        m = HistGradientBoostingRegressor(
            max_iter=700, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.12, n_iter_no_change=30,
            random_state=SEED)
        m.fit(X["train"], tr_y[:, h] - last["train"])
        pv_of[:, h] = m.predict(X["validation"]) + last["validation"]
        pt_of[:, h] = m.predict(X["test"]) + last["test"]
        print(f"  omniflow h{h:02d} dim={X['train'].shape[1]} "
              f"val={rmse(va_y[:, h], pv_of[:, h]):.2f}", flush=True)
    assert np.isfinite(pv_of).all() and np.isfinite(pt_of).all()
    np.save(OUT / "val_omniflow.npy", pv_of)
    np.save(OUT / "test_omniflow.npy", pt_of)
    val = rmse(va_y, pv_of)
    (OUT / "omniflow.json").write_text(json.dumps({
        "val_rmse": round(val, 3),
        "per_h": [round(rmse(va_y[:, h], pv_of[:, h]), 1) for h in range(12)],
    }, indent=2))
    print(f"omniflow val_rmse={val:.3f}", flush=True)

    score(pv_flow, pv_of, va_y)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
