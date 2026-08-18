"""새 예측의 잔차상관만 찍는다."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, "/home/jovyan/work")
import kit
R = Path("/home/jovyan/public_dataset/competition_dataset_6h")
P = Path("/home/jovyan/work/preds")
T = [f"target_{i:02d}" for i in range(12)]
y = pd.read_csv(R / "validation/targets.csv")[T].to_numpy(np.float64)
vals = {f.stem[4:]: np.load(f) for f in P.glob("val_*.npy") if np.load(f).shape == y.shape}
new = [n for n in vals if n not in
       {"persist", "ridge", "gbdt", "ridge_gbdt", "img_ridge", "img_gbdt",
        "img_blend", "v2_ridge", "v2_gbdt", "v2_blend", "cnn96", "cnn96b", "cnn96s2"}]
print("new", new)
base = ["img_blend", "v2_gbdt", "cnn96b", "cnn96s2"]
for n in new:
    print(n, "val", round(kit.score(y, vals[n]), 3))
    for b in base:
        if b in vals:
            print(f"  ρ {n}~{b} {kit.resid_corr(vals[n], vals[b], y):.3f}")
    hss = y > 600
    print("  HSS rmse", round(kit.score(y[hss], vals[n][hss]), 1),
          "bias", round(float((vals[n][hss] - y[hss]).mean()), 1))
wts, best = kit.hill_climb(vals, y, n_iter=80)
print("hill", wts, round(best, 3))
