"""E022 이어서 — 캐시된 광류 피처로 omniflow 학습 + 채점.

사용: python work/flow_fit.py
"""
import json
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "/home/jovyan/work")
from align import aligned_block, seq_of  # noqa: E402
from feats import sample_feats  # noqa: E402
from t4ch import t4_block  # noqa: E402
import kit  # noqa: E402

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
OUT = pathlib.Path("/home/jovyan/work/preds")
FCACHE = pathlib.Path("/home/jovyan/work/flow_cache")
W = [f"wind_{i:02d}" for i in range(20)]
T = [f"target_{i:02d}" for i in range(12)]
SEED = 777
EXCLUDE = {
    "persist", "rotter", "chtrack", "imgpt", "chkm", "submitted",
    "pack63s", "ridge", "ridge_gbdt", "v2_ridge",
}


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main():
    xs = {s: pd.read_csv(ROOT / f"{s}/inputs.csv") for s in ("train", "validation", "test")}
    tr_y = pd.read_csv(ROOT / "train/targets.csv")[T].to_numpy(np.float64)
    va_y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
    last = {s: x[W[-1]].to_numpy(np.float64) for s, x in xs.items()}
    flow = {s: np.load(FCACHE / f"{s}_X.npy") for s in xs}
    print("flow", {s: flow[s].shape for s in xs}, flush=True)

    pv_flow = np.load(OUT / "val_flow.npy")
    print(f"flow val_rmse={kit.score(va_y, pv_flow):.3f}", flush=True)
    vo = np.load(OUT / "val_omni.npy") if (OUT / "val_omni.npy").exists() else None
    vb = np.load(OUT / "val_img_blend.npy") if (OUT / "val_img_blend.npy").exists() else None
    if vo is not None:
        print(f"  rho_omni  {kit.resid_corr(pv_flow, vo, va_y):.3f}", flush=True)
    if vb is not None:
        print(f"  rho_blend {kit.resid_corr(pv_flow, vb, va_y):.3f}", flush=True)

    print("omni matrix…", flush=True)
    base = {s: sample_feats(s, x) for s, x in xs.items()}
    seq = {s: seq_of(s, x) for s, x in xs.items()}
    wind = {s: x[W].to_numpy(np.float64) for s, x in xs.items()}
    print("base", base["train"].shape, flush=True)

    pv = np.zeros_like(va_y)
    pt = np.zeros((len(xs["test"]), 12), np.float64)
    for h in range(12):
        X = {s: np.column_stack([
            base[s], t4_block(seq[s], h), aligned_block(seq[s], wind[s], h), flow[s],
        ]) for s in xs}
        m = HistGradientBoostingRegressor(
            max_iter=700, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.12, n_iter_no_change=30,
            random_state=SEED)
        m.fit(X["train"], tr_y[:, h] - last["train"])
        pv[:, h] = m.predict(X["validation"]) + last["validation"]
        pt[:, h] = m.predict(X["test"]) + last["test"]
        np.save(OUT / "val_omniflow.npy", pv)
        np.save(OUT / "test_omniflow.npy", pt)
        print(f"  omniflow h{h:02d} dim={X['train'].shape[1]} "
              f"val={rmse(va_y[:, h], pv[:, h]):.2f}", flush=True)

    assert np.isfinite(pv).all() and np.isfinite(pt).all()
    val = rmse(va_y, pv)
    (OUT / "omniflow.json").write_text(json.dumps({
        "val_rmse": round(val, 3),
        "per_h": [round(rmse(va_y[:, h], pv[:, h]), 1) for h in range(12)],
    }, indent=2))
    print(f"omniflow val_rmse={val:.3f}", flush=True)

    vals = {}
    for f in sorted(OUT.glob("val_*.npy")):
        name = f.stem[4:]
        if name in EXCLUDE:
            continue
        v = np.load(f)
        if v.shape != va_y.shape or not np.isfinite(v).all():
            print(f"SKIP {name} {getattr(v, 'shape', None)} finite={np.isfinite(v).all() if hasattr(v, 'shape') else '?'}",
                  flush=True)
            continue
        vals[name] = v.astype(np.float64)
    print("== SINGLE ==", flush=True)
    for n in ("flow", "omniflow"):
        if n not in vals:
            continue
        print(f"  {n:12s} {kit.score(va_y, vals[n]):.3f}", flush=True)
        if "omni" in vals:
            print(f"    rho_omni  {kit.resid_corr(vals[n], vals['omni'], va_y):.3f}", flush=True)
        if "img_blend" in vals:
            print(f"    rho_blend {kit.resid_corr(vals[n], vals['img_blend'], va_y):.3f}", flush=True)
    print(f"N_POOL {len(vals)} {sorted(vals)}", flush=True)
    w, s = kit.hill_climb(vals, va_y, n_iter=80)
    print(f"hill_climb {s:.3f}", flush=True)
    for n, wt in sorted(w.items(), key=lambda kv: -kv[1]):
        print(f"  {n:16s} {wt:.3f}", flush=True)
    (OUT / "flow_hill.json").write_text(json.dumps({
        "hill": round(float(s), 3),
        "weights": {k: round(v, 4) for k, v in w.items()},
        "n_pool": len(vals),
        "flow": round(kit.score(va_y, vals["flow"]), 3) if "flow" in vals else None,
        "omniflow": round(val, 3),
    }, indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
