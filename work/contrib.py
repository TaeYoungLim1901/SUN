"""후보 성분을 기존 블렌드에 **추가**했을 때의 기여를 짝지어 판정한다.

단독 val 로 성분을 죽이지 않는다 — 필요한 건 정확한 성분이 아니라 **다르게 틀리는** 성분이다.
  python work/contrib.py cnn4a cnn4l8 cnn3h ...
"""
import pathlib
import sys

import numpy as np
import pandas as pd

from noise_floor import components

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
P = pathlib.Path("/home/jovyan/work/preds")
T = [f"target_{i:02d}" for i in range(12)]
WCOL = [f"wind_{i:02d}" for i in range(20)]
BASE = {"img_ridge": .075, "img_gbdt": .18, "omni": .19, "eswf": .215,
        "cnn3b": .113, "cnn3c": .113, "cnn3d": .114}
GRID = (0.05, 0.10, 0.15, 0.20, 0.25)


def main():
    tags = sys.argv[1:] or sorted(p.stem[4:] for p in P.glob("val_cnn4*.npy"))
    va = pd.read_csv(ROOT / "validation/inputs.csv")
    Y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
    L = lambda t: np.load(P / f"val_{t}.npy")
    lab = components(np.round(va[WCOL].to_numpy(np.float64), 4))
    rows = [np.flatnonzero(lab == c) for c in range(lab.max() + 1)]
    rm = lambda p, ix: float(np.sqrt(np.mean((Y[ix] - p[ix]) ** 2)))
    allix = np.arange(len(va))

    A = sum(w * L(t) for t, w in BASE.items())
    ra = rm(A, allix)
    resid_a = (A - Y).ravel()
    print(f"기준 블렌드 val = {ra:.4f}\n")
    print(f"{'태그':12s} {'단독':>8s} {'ρ(blend)':>9s} {'ρ(잔차)':>8s} {'best w':>7s} {'Δ':>8s} "
          f"{'성분이득':>8s} {'boot σ':>7s} {'95% CI':>18s}")
    rng = np.random.default_rng(0)
    for t in tags:
        if not (P / f"val_{t}.npy").exists():
            print(f"{t:12s}  없음")
            continue
        Pt = L(t)
        rho = np.corrcoef(A.ravel(), Pt.ravel())[0, 1]
        rho_r = np.corrcoef(resid_a, (Pt - Y).ravel())[0, 1]
        best = min(((rm((1 - w) * A + w * Pt, allix), w) for w in GRID))
        rb, w = best
        B = (1 - w) * A + w * Pt
        d = np.array([rm(B, ix) - rm(A, ix) for ix in rows])
        boot = []
        for _ in range(1000):
            pick, tot = [], 0
            while tot < len(va):
                c = rng.integers(len(rows))
                pick.append(rows[c])
                tot += len(rows[c])
            ix = np.concatenate(pick)
            boot.append(rm(B, ix) - rm(A, ix))
        boot = np.array(boot)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"{t:12s} {rm(Pt, allix):8.3f} {rho:9.4f} {rho_r:8.4f} {w:7.2f} {rb-ra:+8.4f} "
              f"{int((d<0).sum()):5d}/{len(d)} {boot.std():7.4f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
