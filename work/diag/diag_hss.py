"""고속 구간 진단. 학습 없음. validation 확인 전용."""
import json
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
vals = {}
for f in sorted(P.glob("val_*.npy")):
    v = np.load(f)
    if v.shape == y.shape and np.isfinite(v).all():
        vals[f.stem[4:]] = v

wts, best = kit.hill_climb(vals, y, n_iter=80)
blend = sum(wts[n] * vals[n] for n in wts)
hss = y > 600
print("hill", wts, "rmse", round(best, 3))
print("HSS cells", int(hss.sum()), f"{100 * hss.mean():.1f}%",
      "sse", float(((blend - y)[hss] ** 2).sum() / ((blend - y) ** 2).sum()))
print("lw | HSS-row", np.round(np.quantile(w[hss.any(1), -1], [0, .25, .5, .75, 1]), 1))
print("lw | quiet  ", np.round(np.quantile(w[~hss.any(1), -1], [0, .25, .5, .75, 1]), 1))
print("== horizon HSS ==")
for h in range(12):
    m = y[:, h] > 600
    if m.sum() < 5:
        continue
    row = [f"h{h:02d} n={m.sum():4d}"]
    for name in ("blend", "img_blend", "v2_gbdt", "cnn96b"):
        p = blend if name == "blend" else vals[name]
        row.append(f"{name}={kit.score(y[m, h:h + 1], p[m, h:h + 1]):5.1f}")
    row.append(f"y={y[m, h].mean():.0f} p={blend[m, h].mean():.0f}")
    print(" ", *row)

print("== resid corr ==")
names = ["img_blend", "v2_gbdt", "cnn96", "cnn96b", "cnn96s2", "ridge_gbdt"]
for i, a in enumerate(names):
    for b in names[i + 1:]:
        if a in vals and b in vals:
            print(f"  {a:12} {b:12} {kit.resid_corr(vals[a], vals[b], y):.3f}")

print("== jsons ==")
for f in sorted(P.glob("*.json")):
    print(f.name, f.read_text()[:200].replace("\n", " "))
