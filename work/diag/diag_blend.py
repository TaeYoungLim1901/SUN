"""결합 후보를 몇 갈래로 다시 고른다. validation 모델선택 전용."""
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
print("all", {n: round(kit.score(y, v), 3) for n, v in sorted(vals.items(), key=lambda kv: kit.score(y, kv[1]))})

sets = {
    "all": list(vals),
    "no_img_gbdt": [n for n in vals if n != "img_gbdt"],
    "core": ["img_blend", "v2_gbdt", "eswf", "cnn96b", "cnn96w3"],
    "core_no_w3": ["img_blend", "v2_gbdt", "eswf", "cnn96b"],
    "scalar": ["img_blend", "v2_gbdt", "eswf"],
    "no_eswf": ["img_blend", "v2_gbdt", "cnn96b", "cnn96w3"],
}
for name, keys in sets.items():
    sub = {k: vals[k] for k in keys if k in vals}
    w, s = kit.hill_climb(sub, y, n_iter=80)
    print(f"\n{name} {s:.3f}  {w}")
    b = sum(w[k] * vals[k] for k in w)
    for row in kit.breakdown(y, b, kit.speed_band(y)):
        print(" ", row)
    # HSS
    m = y > 600
    print("  HSS", round(kit.score(y[m], b[m]), 1), "bias", round(float((b[m] - y[m]).mean()), 1))
