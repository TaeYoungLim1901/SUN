"""E022 omniflow 단독 학습. 피처는 디스크 캐시. PYTHONUNBUFFERED=1 권장."""
import json
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "/home/jovyan/work")
from align import aligned_block, seq_of  # noqa: E402
from feats import sample_feats  # noqa: E402
from t4ch import t4_block  # noqa: E402

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
OUT = pathlib.Path("/home/jovyan/work/preds")
FCACHE = pathlib.Path("/home/jovyan/work/flow_cache")
W = [f"wind_{i:02d}" for i in range(20)]
T = [f"target_{i:02d}" for i in range(12)]
SEED = 777


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def cached(name, fn):
    p = FCACHE / name
    if p.exists():
        a = np.load(p)
        print(f"  hit {name} {a.shape}", flush=True)
        return a
    a = fn()
    np.save(p, a)
    print(f"  save {name} {a.shape}", flush=True)
    return a


def main():
    xs = {s: pd.read_csv(ROOT / f"{s}/inputs.csv") for s in ("train", "validation", "test")}
    tr_y = pd.read_csv(ROOT / "train/targets.csv")[T].to_numpy(np.float64)
    va_y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
    last = {s: x[W[-1]].to_numpy(np.float64) for s, x in xs.items()}
    flow = {s: np.load(FCACHE / f"{s}_X.npy") for s in xs}
    print("omni feats…", flush=True)
    base = {s: cached(f"omni_base_{s}.npy", lambda s=s: sample_feats(s, xs[s])) for s in xs}
    seq = {s: cached(f"omni_seq_{s}.npy", lambda s=s: seq_of(s, xs[s])) for s in xs}
    wind = {s: x[W].to_numpy(np.float64) for s, x in xs.items()}
    print("base", base["train"].shape, "seq", seq["train"].shape, flush=True)

    pv = np.zeros_like(va_y)
    pt = np.zeros((len(xs["test"]), 12), np.float64)
    for h in range(12):
        Xtr = np.column_stack([base["train"], t4_block(seq["train"], h),
                               aligned_block(seq["train"], wind["train"], h), flow["train"]])
        m = HistGradientBoostingRegressor(
            max_iter=700, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.12, n_iter_no_change=30,
            random_state=SEED)
        m.fit(Xtr, tr_y[:, h] - last["train"])
        del Xtr
        Xva = np.column_stack([base["validation"], t4_block(seq["validation"], h),
                               aligned_block(seq["validation"], wind["validation"], h),
                               flow["validation"]])
        Xte = np.column_stack([base["test"], t4_block(seq["test"], h),
                               aligned_block(seq["test"], wind["test"], h), flow["test"]])
        pv[:, h] = m.predict(Xva) + last["validation"]
        pt[:, h] = m.predict(Xte) + last["test"]
        del Xva, Xte, m
        np.save(OUT / "val_omniflow.npy", pv)
        np.save(OUT / "test_omniflow.npy", pt)
        print(f"  omniflow h{h:02d} val={rmse(va_y[:, h], pv[:, h]):.2f}", flush=True)

    assert np.isfinite(pv).all() and np.isfinite(pt).all()
    val = rmse(va_y, pv)
    (OUT / "omniflow.json").write_text(json.dumps({
        "val_rmse": round(val, 3),
        "per_h": [round(rmse(va_y[:, h], pv[:, h]), 1) for h in range(12)],
    }, indent=2))
    print(f"omniflow val_rmse={val:.3f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
