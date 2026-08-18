"""E016 — horizon 별 발사시각 정합 프레임을 **추가 피처로만** 준다. E007 재개 조건 그대로.

E007 은 창을 j*±2 로 **좁혀서** 졌다 (전 horizon 악화). 여기서는 v2 피처 839개를 그대로 두고,
horizon h 마다 τ=1AU/v 로 계산한 발사 슬롯 j*(h) 의 스칼라를 선형보간해 **덧붙이기만** 한다.

  τ[h] = 41556 / v  (시간),  Δ = 6(h+1) − τ,  j* = 19 + Δ/6
  v 가설 둘: 마지막 풍속, 최근 8슬롯 평균.

산출: preds/val_align.npy · test_align.npy (+ align.json)
사용: python work/align.py
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "/home/jovyan/work")
from feats import DARK, image_scalars, sample_feats  # noqa: E402

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
OUT = pathlib.Path("/home/jovyan/work/preds")
IMG = [f"image_{i:02d}" for i in range(20)]
W = [f"wind_{i:02d}" for i in range(20)]
T = [f"target_{i:02d}" for i in range(12)]
SEED = 777
TAG = "align"
AU_HOURS = 41556.0        # 1 AU / (1 km/s) 를 시간으로. τ[h] = AU_HOURS / v
NGLOBAL = 2 * (4 + len(DARK))     # 전역 디스크 스칼라 18개 (채널 2 × 9)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def seq_of(split, inputs):
    """(N, 20, F) 시점별 이미지 스칼라. 정합에 쓰는 부분만 잘라 돌려준다."""
    idx, per = image_scalars(split, inputs)
    seq = np.stack([per[[idx[v] for v in inputs[col]]] for col in IMG], axis=1)
    return np.concatenate([seq[:, :, :NGLOBAL], seq[:, :, -32:]], axis=2)   # 전역 + 서쪽 스트립


def gather(seq, j):
    """슬롯 축을 소수 인덱스 j 로 선형보간. seq (N,20,F), j (N,) → (N,F)."""
    j = np.clip(j, 0, 19)
    lo = np.floor(j).astype(int)
    hi = np.minimum(lo + 1, 19)
    t = (j - lo).astype(np.float32)[:, None]
    r = np.arange(len(seq))
    return seq[r, lo] * (1 - t) + seq[r, hi] * t


def aligned_block(seq, w, h):
    """horizon h 의 발사 슬롯 피처. 속도 가설 둘 + τ·j* 자체."""
    cols = []
    for v in (w[:, -1], w[:, -8:].mean(1)):
        tau = AU_HOURS / np.maximum(v, 200.0)
        js = 19 + (6 * (h + 1) - tau) / 6.0
        cols += [gather(seq, js), tau[:, None], js[:, None],
                 gather(seq, js) - gather(seq, np.full_like(js, 19.0))]
    return np.column_stack(cols)


def main():
    xs = {s: pd.read_csv(ROOT / f"{s}/inputs.csv") for s in ("train", "validation", "test")}
    tr_y = pd.read_csv(ROOT / "train/targets.csv")[T].to_numpy(np.float64)
    va_y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)

    base = {s: sample_feats(s, x) for s, x in xs.items()}
    seq = {s: seq_of(s, x) for s, x in xs.items()}
    wind = {s: x[W].to_numpy(np.float64) for s, x in xs.items()}
    print("base", base["train"].shape, "seq", seq["train"].shape, flush=True)

    pv, pt = np.zeros_like(va_y), np.zeros((len(xs["test"]), 12))
    for h in range(12):
        X = {s: np.column_stack([base[s], aligned_block(seq[s], wind[s], h)]) for s in xs}
        m = HistGradientBoostingRegressor(
            max_iter=700, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.12, n_iter_no_change=30,
            random_state=SEED).fit(X["train"], tr_y[:, h])
        pv[:, h] = m.predict(X["validation"])
        pt[:, h] = m.predict(X["test"])
        print(f"  h{h:02d} dim={X['train'].shape[1]} val={rmse(va_y[:, h], pv[:, h]):.2f}", flush=True)

    np.save(OUT / f"val_{TAG}.npy", pv)
    np.save(OUT / f"test_{TAG}.npy", pt)
    val = rmse(va_y, pv)
    (OUT / f"{TAG}.json").write_text(json.dumps(
        {"val_rmse": round(val, 3),
         "per_h": [round(rmse(va_y[:, h], pv[:, h]), 1) for h in range(12)]}, indent=2))
    print(f"{TAG} val_rmse={val:.3f}  (v2_gbdt 67.022 를 이겨야 채택)", flush=True)


if __name__ == "__main__":
    main()
