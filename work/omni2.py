"""E021 — 열린 도메인 피처를 *한 모델에 동시에* 넣는다.

  v2 839
  + t−4d CH 블록 (DOMAIN §9.2, 프레임 h+4)
  + 발사시각 j*·τ 블록 (E016, 창은 안 좁힘)
  타깃은 last 대비 잔차 (E019).

하나씩 갈아끼우면 ρ≥0.95 로 v2 자리만 바뀐다. 여기를 새 족으로 본다.
산출: preds/val_omni.npy · test_omni.npy · omni_models.pth
채택: 단독 <66.829(jump) 또는 hill 결합이 63.75 대비 0.08 이상.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "/home/jovyan/work")
from align import aligned_block, seq_of  # noqa: E402
from feats import sample_feats  # noqa: E402
from phys import image_features, row_features  # noqa: E402
from t4ch import t4_block  # noqa: E402

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
OUT = pathlib.Path("/home/jovyan/work/preds")
W = [f"wind_{i:02d}" for i in range(20)]
T = [f"target_{i:02d}" for i in range(12)]
SEED = 777
TAG = "omni2"


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main():
    xs = {s: pd.read_csv(ROOT / f"{s}/inputs.csv") for s in ("train", "validation", "test")}
    tr_y = pd.read_csv(ROOT / "train/targets.csv")[T].to_numpy(np.float64)
    va_y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
    last = {s: x[W[-1]].to_numpy(np.float64) for s, x in xs.items()}
    base = {s: sample_feats(s, x) for s, x in xs.items()}
    seq = {s: seq_of(s, x) for s, x in xs.items()}
    wind = {s: x[W].to_numpy(np.float64) for s, x in xs.items()}
    physics = {}
    for s, x in xs.items():
        cache = OUT / f"phys_{s}.npy"
        if not cache.exists():
            raise FileNotFoundError(cache)
        idx, values = image_features(s, x)
        physics[s] = row_features(x, idx, values)
    print("base", base["train"].shape, "seq", seq["train"].shape, flush=True)

    pv, pt = np.zeros_like(va_y), np.zeros((len(xs["test"]), 12))
    models = []
    for h in range(12):
        core = {s: np.column_stack([base[s], t4_block(seq[s], h), aligned_block(seq[s], wind[s], h)])
                for s in xs}
        X = {s: np.column_stack([core[s], physics[s]]) for s in xs}
        if h == 0:
            print(f"feature count before={core['train'].shape[1]} after={X['train'].shape[1]}", flush=True)
        m = HistGradientBoostingRegressor(
            max_iter=700, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.12, n_iter_no_change=30,
            random_state=SEED).fit(X["train"], tr_y[:, h] - last["train"])
        models.append(m)
        pv[:, h] = m.predict(X["validation"]) + last["validation"]
        pt[:, h] = m.predict(X["test"]) + last["test"]
        print(f"  h{h:02d} dim={X['train'].shape[1]} val={rmse(va_y[:, h], pv[:, h]):.2f}", flush=True)

    torch.save(models, pathlib.Path("/home/jovyan/work/omni_models.pth"))
    np.save(OUT / f"val_{TAG}.npy", pv)
    np.save(OUT / f"test_{TAG}.npy", pt)
    val = rmse(va_y, pv)
    (OUT / f"{TAG}.json").write_text(json.dumps({
        "val_rmse": round(val, 3),
        "per_h": [round(rmse(va_y[:, h], pv[:, h]), 1) for h in range(12)],
    }, indent=2))
    print(f"total val RMSE: {val:.3f}", flush=True)
    print(f"original omni total val RMSE: {rmse(va_y, np.load(OUT / 'val_omni.npy')):.3f}", flush=True)
    print(f"{TAG} val_rmse={val:.3f}  (jump 66.829 / v2 67.022)", flush=True)


if __name__ == "__main__":
    main()
