"""E012 — 창 안 코로나홀 중심 궤적. DOMAIN §9.5.

27일 lag 는 불가. 20프레임(4.75일, ~63°) 안에서 CH 가 동→서로 지나가는
것이 남은 시간 정보다. 전역 요약을 대체하지 않고 궤적 스칼라만 더한다.

산출: val_chtrack.npy / test_chtrack.npy
사용: python work/chtrack.py
"""
import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
CACHE = pathlib.Path("/home/jovyan/work/resized_cache/64px")
OUT = pathlib.Path("/home/jovyan/work/preds")
IMG = [f"image_{i:02d}" for i in range(20)]
W = [f"wind_{i:02d}" for i in range(20)]
T = [f"target_{i:02d}" for i in range(12)]
SEED = 777
TAG = "chtrack"


def per_image(split, inputs):
    arr = np.load(CACHE / f"{split}_images.npy", mmap_mode="r")
    names = sorted(pd.unique(inputs[IMG].to_numpy().ravel()).tolist())
    a = np.asarray(arr, dtype=np.float32)
    n, c, h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 0.47 * h
    dx = (xx - w / 2 + .5) / r
    dy = (yy - h / 2 + .5) / r
    disk = (dx ** 2 + dy ** 2) <= 1.0
    low = np.abs(dy) < np.sin(np.deg2rad(30))
    m = disk & low
    flat = a.reshape(n, c, -1)
    med = np.median(flat[:, :, disk.ravel()], axis=2, keepdims=True)
    dark = (flat < med * 0.55) * m.ravel()
    mass = dark.sum(2) + 1e-6
    # 채널 평균 중심 (태양반지름 단위). x>0 = 서쪽(원반 오른쪽)
    cx = (dark * dx.ravel()).sum(2) / mass
    cy = (dark * dy.ravel()).sum(2) / mass
    area = dark.mean(2)
    return {v: i for i, v in enumerate(names)}, np.column_stack(
        [cx, cy, area]).astype(np.float32)


def seq_feats(split, inputs):
    idx, per = per_image(split, inputs)
    seq = np.stack([per[[idx[v] for v in inputs[col]]] for col in IMG], axis=1)  # (N,20,6)
    w = inputs[W].to_numpy(np.float64)
    dw = np.diff(w, axis=1)
    # 궤적: 최근 x, x 변화(동→서), 자오선 통과(x 부호 변화), 면적 변화
    x193, x211 = seq[:, :, 0], seq[:, :, 1]
    a193, a211 = seq[:, :, 4], seq[:, :, 5]
    cross = ((x193[:, 1:] * x193[:, :-1]) < 0).sum(1)
    return np.column_stack([
        w, dw, w[:, -1], w[:, -1] - w[:, -4],
        seq[:, -1], seq.mean(1), seq[:, -1] - seq[:, 0],
        x193[:, -1], x211[:, -1],
        x193[:, -1] - x193[:, 0], x211[:, -1] - x211[:, 0],
        a193[:, -1] - a193[:, 0], a211[:, -1] - a211[:, 0],
        np.gradient(x193, axis=1).mean(1), np.gradient(x211, axis=1).mean(1),
        cross, (np.abs(x193) < 0.26).mean(1), (np.abs(x211) < 0.26).mean(1),
    ])


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main():
    tr_x = pd.read_csv(ROOT / "train/inputs.csv")
    va_x = pd.read_csv(ROOT / "validation/inputs.csv")
    te_x = pd.read_csv(ROOT / "test/inputs.csv")
    tr_y = pd.read_csv(ROOT / "train/targets.csv")[T].to_numpy(np.float64)
    va_y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
    print("track feats…", flush=True)
    Xtr, Xva, Xte = seq_feats("train", tr_x), seq_feats("validation", va_x), seq_feats("test", te_x)
    print("X", Xtr.shape, flush=True)
    gv, gt = np.zeros_like(va_y), np.zeros((len(te_x), 12))
    for h in range(12):
        m = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.12, n_iter_no_change=20, random_state=SEED,
        ).fit(Xtr, tr_y[:, h])
        gv[:, h], gt[:, h] = m.predict(Xva), m.predict(Xte)
        print(f"  h{h:02d} {rmse(va_y[:, h], gv[:, h]):.2f}", flush=True)
    np.save(OUT / f"val_{TAG}.npy", gv)
    np.save(OUT / f"test_{TAG}.npy", gt)
    (OUT / f"{TAG}.json").write_text(json.dumps({
        "val_rmse": round(rmse(va_y, gv), 3),
        "per_h": [round(rmse(va_y[:, h], gv[:, h]), 1) for h in range(12)],
    }, indent=2))
    print(f"{TAG} val_rmse={rmse(va_y, gv):.3f}", flush=True)


if __name__ == "__main__":
    main()
