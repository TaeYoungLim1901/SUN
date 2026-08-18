"""타이밍 vs 진폭 분해. DOMAIN §9.6 — kit.breakdown 이 못 가르는 것.

로컬: python work/diag_peak.py --bundle work/local_data/pred_bundle.npz
서버: python work/diag_peak.py
채점 공식은 kit.score 와 같다. 이 파일은 진단만 한다. 학습에 val 을 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
PRED = pathlib.Path("/home/jovyan/work/preds")


def score(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def peak_diag(y, p):
    """행마다 피크 높이 / 피크 시각 / 상승구간."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    iy, ip = y.argmax(1), p.argmax(1)
    n = np.arange(len(y))
    y_pk, p_pk = y[n, iy], p[n, ip]
    height = p_pk - y_pk
    step = ip.astype(float) - iy.astype(float)
    # 상승: 그 행에서 최댓값 이전
    rise_err, fall_err = [], []
    for i in range(len(y)):
        k = int(iy[i])
        rise_err.extend((p[i, : k + 1] - y[i, : k + 1]).ravel())
        if k < 11:
            fall_err.extend((p[i, k + 1 :] - y[i, k + 1 :]).ravel())
    return {
        "rmse": score(y, p),
        "peak_h_bias": float(height.mean()),
        "peak_h_rmse": float(np.sqrt(np.mean(height ** 2))),
        "peak_step_bias": float(step.mean()),
        "peak_step_mae": float(np.mean(np.abs(step))),
        "rise_bias": float(np.mean(rise_err)) if rise_err else np.nan,
        "fall_bias": float(np.mean(fall_err)) if fall_err else np.nan,
        "hss_rmse": score(y[y > 600], p[y > 600]) if (y > 600).any() else np.nan,
        "hss_bias": float((p[y > 600] - y[y > 600]).mean()) if (y > 600).any() else np.nan,
    }


def hill_climb(oofs, y, n_iter=80):
    names = list(oofs)
    picks, cur, best = [], None, np.inf
    for _ in range(n_iter):
        trials = []
        for n in names:
            t = oofs[n] if cur is None else (cur * len(picks) + oofs[n]) / (len(picks) + 1)
            trials.append((score(y, t), n, t))
        s, n, t = min(trials, key=lambda x: x[0])
        if s >= best - 1e-9:
            break
        best, cur = s, t
        picks.append(n)
    return {n: picks.count(n) / len(picks) for n in set(picks)}, best


def load_bundle(path):
    z = np.load(path)
    y = z["y"]
    vals = {k[4:]: z[k] for k in z.files if k.startswith("val_")}
    tests = {k[5:]: z[k] for k in z.files if k.startswith("test_")}
    wind = z["wind"] if "wind" in z.files else None
    return y, vals, tests, wind


def load_server():
    import pandas as pd
    tcols = [f"target_{i:02d}" for i in range(12)]
    y = pd.read_csv(ROOT / "validation/targets.csv")[tcols].to_numpy(np.float64)
    vals, tests = {}, {}
    for f in sorted(PRED.glob("val_*.npy")):
        v = np.load(f)
        if v.shape == y.shape:
            vals[f.stem[4:]] = v
    for f in sorted(PRED.glob("test_*.npy")):
        tests[f.stem[5:]] = np.load(f)
    return y, vals, tests, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="")
    a = ap.parse_args()
    if a.bundle:
        y, vals, tests, wind = load_bundle(a.bundle)
    else:
        y, vals, tests, wind = load_server()

    print(f"{'name':<14}{'RMSE':>8}{'pkH':>8}{'pkS':>7}{'riseB':>8}{'fallB':>8}{'HSS':>8}{'HSSb':>8}")
    rows = []
    for name, v in vals.items():
        if v.shape != y.shape:
            continue
        d = peak_diag(y, v)
        d["name"] = name
        rows.append(d)
        print(f"{name:<14}{d['rmse']:8.2f}{d['peak_h_bias']:8.1f}{d['peak_step_mae']:7.2f}"
              f"{d['rise_bias']:8.1f}{d['fall_bias']:8.1f}{d['hss_rmse']:8.1f}{d['hss_bias']:8.1f}")

    # 결합 후보 몇 갈래
    sets = {
        "core": ["img_gbdt", "eswf", "img_ridge", "v2_gbdt", "cnn96b", "cnn96w3"],
        "w5": ["img_gbdt", "eswf", "img_ridge", "v2_gbdt", "cnn96b", "cnn96w5"],
        "both_cnn_w": ["img_gbdt", "eswf", "img_ridge", "v2_gbdt", "cnn96b", "cnn96w3", "cnn96w5"],
        "plus_imgpt": ["img_gbdt", "eswf", "img_ridge", "v2_gbdt", "cnn96b", "cnn96w3", "imgpt"],
    }
    print("\n== hill ==")
    for nm, keys in sets.items():
        sub = {k: vals[k] for k in keys if k in vals}
        if len(sub) < 2:
            continue
        w, s = hill_climb(sub, y)
        b = sum(w[k] * vals[k] for k in w)
        d = peak_diag(y, b)
        print(f"{nm:14} {s:.3f}  pkH {d['peak_h_bias']:+.1f}  step {d['peak_step_mae']:.2f}  "
              f"HSS {d['hss_rmse']:.1f} {d['hss_bias']:+.1f}  {w}")


if __name__ == "__main__":
    main()
