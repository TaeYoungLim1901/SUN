"""E020 — 문헌의 t−4d CH 를 horizon 마다 *실제로 있는 프레임* 에서 뽑는다.

DOMAIN §9.2: target 기준 t−4d 는 전 horizon 창 안 (프레임 인덱스 = h+4).
v2 는 창 끝(프레임 19) 요약을 모든 horizon 에 같다 쓴다. 여기는 839를 유지하고
해당 프레임의 전역+서쪽 스트립을 *추가*만 한다. 창을 좁히지 않는다 (E007 금지).

산출: preds/val_t4ch.npy · test_t4ch.npy · t4ch_models.pth
채택: v2 67.022 를 이기거나 hill 이 넣고 결합 Δ≥0.08.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "/home/jovyan/work")
from feats import DARK, image_scalars, sample_feats  # noqa: E402

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
OUT = pathlib.Path("/home/jovyan/work/preds")
IMG = [f"image_{i:02d}" for i in range(20)]
T = [f"target_{i:02d}" for i in range(12)]
SEED = 777
TAG = "t4ch"
NGLOBAL = 2 * (4 + len(DARK))


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def seq_of(split, inputs):
    idx, per = image_scalars(split, inputs)
    seq = np.stack([per[[idx[v] for v in inputs[col]]] for col in IMG], axis=1)
    return np.concatenate([seq[:, :, :NGLOBAL], seq[:, :, -32:]], axis=2)


def t4_block(seq, h):
    """target 기준 t−4d = 프레임 h+4. 이웃 ±1 과 창 끝 대비."""
    j = int(np.clip(h + 4, 0, 19))
    lo, hi = max(j - 1, 0), min(j + 1, 19)
    return np.column_stack([
        seq[:, j], seq[:, lo], seq[:, hi],
        seq[:, j] - seq[:, 19], seq[:, j] - seq[:, 0],
    ])


def main():
    xs = {s: pd.read_csv(ROOT / f"{s}/inputs.csv") for s in ("train", "validation", "test")}
    tr_y = pd.read_csv(ROOT / "train/targets.csv")[T].to_numpy(np.float64)
    va_y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
    base = {s: sample_feats(s, x) for s, x in xs.items()}
    seq = {s: seq_of(s, x) for s, x in xs.items()}
    print("base", base["train"].shape, "seq", seq["train"].shape, flush=True)

    pv, pt = np.zeros_like(va_y), np.zeros((len(xs["test"]), 12))
    models = []
    for h in range(12):
        X = {s: np.column_stack([base[s], t4_block(seq[s], h)]) for s in xs}
        m = HistGradientBoostingRegressor(
            max_iter=700, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.12, n_iter_no_change=30,
            random_state=SEED).fit(X["train"], tr_y[:, h])
        models.append(m)
        pv[:, h] = m.predict(X["validation"])
        pt[:, h] = m.predict(X["test"])
        print(f"  h{h:02d} j={h+4} dim={X['train'].shape[1]} val={rmse(va_y[:, h], pv[:, h]):.2f}",
              flush=True)

    torch.save(models, pathlib.Path("/home/jovyan/work/t4ch_models.pth"))
    np.save(OUT / f"val_{TAG}.npy", pv)
    np.save(OUT / f"test_{TAG}.npy", pt)
    val = rmse(va_y, pv)
    (OUT / f"{TAG}.json").write_text(json.dumps({
        "val_rmse": round(val, 3),
        "per_h": [round(rmse(va_y[:, h], pv[:, h]), 1) for h in range(12)],
    }, indent=2))
    print(f"{TAG} val_rmse={val:.3f}  (v2 67.022 / 결합 Δ0.08)", flush=True)


if __name__ == "__main__":
    main()
