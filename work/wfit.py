"""광류 시드백이 들어온 뒤의 **결합 가중 재적합**을 정직하게 판정한다.

현 카드는 (E039 시절에 고른 base 7성분 비율)×0.8 + (광류백)×0.2 다. base 비율은 광류가
없던 때 고른 것이라 다시 맞출 여지가 있다. 여기서는 6h 연결요소 2-fold × N 반복으로
**한쪽 반에서 가중을 적합하고 반대쪽 반에서만 채점**한다 (add_honest.py 와 같은 프로토콜).

  python work/wfit.py            # 판정만
  python work/wfit.py --write    # 통과했을 때 blend.json 갱신
"""
import argparse
import glob
import json
import os
import pathlib

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from noise_floor import components

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
P = pathlib.Path("/home/jovyan/work/preds")
T = [f"target_{i:02d}" for i in range(12)]
WCOL = [f"wind_{i:02d}" for i in range(20)]
CUR = {"img_ridge": .075, "img_gbdt": .18, "omni": .19, "eswf": .215,
       "cnn3b": .113, "cnn3c": .113, "cnn3d": .114}      # 광류 몫을 빼기 전 base 비율
REPEATS = 20
RIDGE = 1e-3          # 심플렉스 적합에 거는 아주 약한 수축 (11개 그룹뿐이라 필요하다)


def load(split, tag):
    return np.load(P / f"{split}_{tag}.npy")


def fit_simplex(X, y):
    """비음수 최소제곱 + 합=1 강제. X (n, k), y (n,)."""
    k = X.shape[1]
    A = np.vstack([X, np.full((1, k), 50.0), np.sqrt(RIDGE) * np.eye(k)])
    b = np.concatenate([y, [50.0], np.zeros(k)])
    w, _ = nnls(A, b)
    s = w.sum()
    return w / s if s > 0 else np.full(k, 1.0 / k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wflow", type=float, default=0.20, help="비교 기준 카드의 광류 몫")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    Y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
    wind = pd.read_csv(ROOT / "validation/inputs.csv")[WCOL].to_numpy(np.float64)
    grp = np.asarray(components(wind))

    flow_tags = sorted(os.path.basename(x)[4:-4] for x in glob.glob(str(P / "val_cnn6s6*.npy")))
    bag = np.mean([load("val", t) for t in flow_tags], axis=0)
    names = list(CUR) + ["flow"]
    stack = np.stack([load("val", t) for t in CUR] + [bag])          # (k, n, 12)
    cur_w = np.array([v * (1 - a.wflow) for v in CUR.values()] + [a.wflow])

    rmse = lambda w, m: float(np.sqrt(np.mean((np.tensordot(w, stack[:, m], 1) - Y[m]) ** 2)))
    uniq = np.unique(grp)
    rng = np.random.default_rng(0)
    deltas, picks = [], []
    for r in range(REPEATS):
        order = rng.permutation(uniq)
        half = set(order[: len(order) // 2].tolist())
        for flip in (0, 1):
            fit_m = np.isin(grp, list(half)) if flip == 0 else ~np.isin(grp, list(half))
            sc_m = ~fit_m
            X = stack[:, fit_m].reshape(len(names), -1).T
            w = fit_simplex(X, Y[fit_m].ravel())
            deltas.append(rmse(w, sc_m) - rmse(cur_w, sc_m))
            picks.append(w)
    d = np.array(deltas)
    picks = np.array(picks)
    print(f"기준 카드 (base×{1 - a.wflow:.2f} + flow×{a.wflow:.2f}) 전체 val {rmse(cur_w, np.ones(len(Y), bool)):.4f}")
    print(f"정직 판정 (2-fold×{REPEATS} = {len(d)}판)")
    print(f"  중앙 Δ {np.median(d):+.4f} · 평균 {d.mean():+.4f} · 이득 {(d < 0).sum()}/{len(d)}"
          f" ({100 * (d < 0).mean():.0f}%) · 최악 {d.max():+.4f}")
    print("  적합 가중 중앙:", {n: round(float(v), 4) for n, v in zip(names, np.median(picks, 0))})
    full = fit_simplex(stack.reshape(len(names), -1).T, Y.ravel())
    print("  참고(낙관) 전체 val 적합:", {n: round(float(v), 4) for n, v in zip(names, full)},
          f"→ {rmse(full, np.ones(len(Y), bool)):.4f}")
    ok = np.median(d) < 0 and (d < 0).mean() >= 0.8
    print("판정:", "GO" if ok else "NO-GO", "— 선: 중앙 Δ<0 · 이득 ≥80%")

    if a.write and ok:
        w = np.median(picks, 0)
        w = w / w.sum()
        cfg = json.loads((pathlib.Path("/home/jovyan/work") / "blend.json").read_text())
        base = {k: float(v) for k, v in zip(list(CUR), w[:-1])}
        per = float(w[-1]) / len(flow_tags)
        cfg["weights"] = {**base, **{t: per for t in flow_tags}}
        (pathlib.Path("/home/jovyan/work") / "blend.json").write_text(json.dumps(cfg))
        print("blend.json 갱신")


if __name__ == "__main__":
    main()
