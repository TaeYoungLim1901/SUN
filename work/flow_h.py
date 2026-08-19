"""omniflow 한 horizon. python work/flow_h.py 0"""
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "/home/jovyan/work")
from align import aligned_block  # noqa: E402
from t4ch import t4_block  # noqa: E402

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
OUT = pathlib.Path("/home/jovyan/work/preds")
FCACHE = pathlib.Path("/home/jovyan/work/flow_cache")
W = [f"wind_{i:02d}" for i in range(20)]
T = [f"target_{i:02d}" for i in range(12)]
SEED = 777


def main():
    h = int(sys.argv[1])
    xs = {s: pd.read_csv(ROOT / f"{s}/inputs.csv") for s in ("train", "validation", "test")}
    tr_y = pd.read_csv(ROOT / "train/targets.csv")[T].to_numpy(np.float64)
    va_y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
    last = {s: x[W[-1]].to_numpy(np.float64) for s, x in xs.items()}
    flow = {s: np.load(FCACHE / f"{s}_X.npy") for s in xs}
    base = {s: np.load(FCACHE / f"omni_base_{s}.npy") for s in xs}
    seq = {s: np.load(FCACHE / f"omni_seq_{s}.npy") for s in xs}
    wind = {s: x[W].to_numpy(np.float64) for s, x in xs.items()}
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
    va_p = m.predict(Xva) + last["validation"]
    te_p = m.predict(Xte) + last["test"]
    vp = OUT / "val_omniflow.npy"
    tp = OUT / "test_omniflow.npy"
    pv = np.load(vp) if vp.exists() else np.zeros_like(va_y)
    pt = np.load(tp) if tp.exists() else np.zeros((len(xs["test"]), 12), np.float64)
    if pv.shape != va_y.shape:
        pv = np.zeros_like(va_y)
    if pt.shape != (len(xs["test"]), 12):
        pt = np.zeros((len(xs["test"]), 12), np.float64)
    pv[:, h] = va_p
    pt[:, h] = te_p
    np.save(vp, pv)
    np.save(tp, pt)
    rmse = float(np.sqrt(np.mean((va_y[:, h] - va_p) ** 2)))
    print(f"omniflow h{h:02d} val={rmse:.2f} finite={np.isfinite(pv).all()}", flush=True)


if __name__ == "__main__":
    main()
