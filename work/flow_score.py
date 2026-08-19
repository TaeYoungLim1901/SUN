"""E022 채점만. python work/flow_score.py"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jovyan/work")
import kit  # noqa: E402

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
OUT = pathlib.Path("/home/jovyan/work/preds")
T = [f"target_{i:02d}" for i in range(12)]
EXCLUDE = {
    "persist", "rotter", "chtrack", "imgpt", "chkm", "submitted",
    "pack63s", "ridge", "ridge_gbdt", "v2_ridge",
}


def main():
    y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
    vals = {}
    for f in sorted(OUT.glob("val_*.npy")):
        name = f.stem[4:]
        if name in EXCLUDE:
            continue
        v = np.load(f)
        if v.shape != y.shape:
            print(f"SKIP {name} shape {v.shape}", flush=True)
            continue
        if not np.isfinite(v).all():
            print(f"SKIP {name} nan", flush=True)
            continue
        if name == "omniflow" and (np.abs(v).sum(0) == 0).any():
            print(f"SKIP {name} incomplete", flush=True)
            continue
        vals[name] = v.astype(np.float64)
    print("== SINGLE ==", flush=True)
    for n, s in sorted(((k, kit.score(y, v)) for k, v in vals.items()), key=lambda x: x[1]):
        print(f"  {n:16s} {s:8.3f}", flush=True)
    print("== RESID new ==", flush=True)
    for n in ("flow", "omniflow"):
        if n not in vals:
            continue
        print(f"  {n:12s} {kit.score(y, vals[n]):.3f}", flush=True)
        for o in ("omni", "img_blend"):
            if o in vals:
                print(f"    rho_{o} {kit.resid_corr(vals[n], vals[o], y):.3f}", flush=True)
    print(f"N_POOL {len(vals)} {sorted(vals)}", flush=True)
    w, s = kit.hill_climb(vals, y, n_iter=80)
    print(f"hill_climb {s:.3f}", flush=True)
    for n, wt in sorted(w.items(), key=lambda kv: -kv[1]):
        print(f"  {n:16s} {wt:.3f}", flush=True)
    (OUT / "flow_hill.json").write_text(json.dumps({
        "hill": round(float(s), 3),
        "weights": {k: round(v, 4) for k, v in w.items()},
        "n_pool": len(vals),
        "tags": {n: round(kit.score(y, vals[n]), 3) for n in ("flow", "omniflow") if n in vals},
    }, indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
