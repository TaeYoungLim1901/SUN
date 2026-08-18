"""HSS 지속 보정 축이 살아있는지 보는 진단. validation 수치는 채택이 아니다."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jovyan/work")
import kit  # noqa: E402

R = Path("/home/jovyan/public_dataset/competition_dataset_6h")
P = Path("/home/jovyan/work/preds")
T = [f"target_{i:02d}" for i in range(12)]
W = [f"wind_{i:02d}" for i in range(20)]
y = pd.read_csv(R / "validation/targets.csv")[T].to_numpy(np.float64)
w = pd.read_csv(R / "validation/inputs.csv")[W].to_numpy(np.float64)
lw = w[:, -1]
vals = {}
for f in sorted(P.glob("val_*.npy")):
    v = np.load(f)
    if v.shape == y.shape and np.isfinite(v).all():
        vals[f.stem[4:]] = v
wts, _ = kit.hill_climb(vals, y, n_iter=80)
blend = sum(wts[n] * vals[n] for n in wts)
base = kit.score(y, blend)
print("base", round(base, 3))


def apply(p, T0, alpha, tau):
    out = p.copy()
    gap = np.maximum(lw - p.T, 0.0).T          # (n,12) 모델이 last_wind 보다 낮게 찍은 양
    on = (lw >= T0).astype(np.float64)[:, None]
    decay = np.exp(-np.arange(12) / tau)
    out = p + alpha * gap * on * decay
    return out


best = (base, None)
print("T alpha tau rmse d  (VAL PEEK)")
for T0 in (500, 550, 580, 600, 650):
    for alpha in (0.15, 0.25, 0.4, 0.6, 0.8):
        for tau in (2.0, 3.0, 4.0, 6.0, 8.0):
            s = kit.score(y, apply(blend, T0, alpha, tau))
            if s < best[0]:
                best = (s, (T0, alpha, tau))
            if s < base - 0.15:
                print(f"  {T0} {alpha} {tau} {s:.3f} {s - base:+.3f}")
print("best peek", best)

# 상승 중일 때만
rise = lw - w[:, -5]
print("rise-gated peek")
best2 = (base, None)
for T0 in (450, 500, 550):
    for D in (20, 40, 60):
        for alpha in (0.25, 0.4, 0.6):
            for tau in (3.0, 5.0, 8.0):
                out = blend.copy()
                gap = np.maximum(lw - blend.T, 0.0).T
                on = ((lw >= T0) | (rise >= D)).astype(np.float64)[:, None]
                out = blend + alpha * gap * on * np.exp(-np.arange(12) / tau)
                s = kit.score(y, out)
                if s < best2[0]:
                    best2 = (s, (T0, D, alpha, tau))
print("best rise peek", best2)

# 고속 셀만 가중 재채점 — 보정 후 HSS RMSE
T0, alpha, tau = best[1] if best[1] else (600, 0.4, 4.0)
g = apply(blend, T0, alpha, tau)
print("HSS rmse base", kit.score(y[y > 600], blend[y > 600]),
      "gated", kit.score(y[y > 600], g[y > 600]))
for row in kit.breakdown(y, g, kit.speed_band(y)):
    print(" gated band", row)
