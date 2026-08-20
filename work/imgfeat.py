"""영상을 스칼라로 축약해 wind 피처에 붙인다. 코로나홀 = EUV 에서 어두운 영역 (DOMAIN §6-4).

64px 캐시(유니크 이미지당 1장)를 그대로 재사용한다. 유니크 이미지 단위로 한 번만 계산하고
샘플의 20슬롯은 인덱싱만 한다 — 이게 이 데이터셋에서 가장 큰 공짜 절약이다.
규정: train 으로만 학습. validation 은 성능 확인 전용 (RULES.md §1 7항).
"""
import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
CACHE = pathlib.Path("/home/jovyan/work/resized_cache/64px")
OUT = pathlib.Path("/home/jovyan/work/preds")
OUT.mkdir(parents=True, exist_ok=True)
IMG = [f"image_{i:02d}" for i in range(20)]
W = [f"wind_{i:02d}" for i in range(20)]
T = [f"target_{i:02d}" for i in range(12)]
DARK = (20, 40, 60)          # 0-255. 코로나홀은 어둡다
GRID = 2                     # 2×2 격자 + 전체 디스크


def image_scalars(split, inputs):
    """유니크 이미지 → (n_unique, F) 스칼라. 캐시 파일 순서는 샘플코드와 동일하게 sorted(unique)."""
    arr = np.load(CACHE / f"{split}_images.npy", mmap_mode="r")   # (n, 2, 64, 64) uint8
    names = sorted(pd.unique(inputs[IMG].to_numpy().ravel()).tolist())
    assert arr.shape[0] == len(names), (split, arr.shape, len(names))
    a = np.asarray(arr, dtype=np.uint8)
    n, c, h, w = a.shape
    out = [a.reshape(n, c, -1).mean(2), a.reshape(n, c, -1).std(2)]
    for th in DARK:
        out.append((a.reshape(n, c, -1) < th).mean(2))
    s = h // GRID
    for gy in range(GRID):
        for gx in range(GRID):
            cell = a[:, :, gy * s:(gy + 1) * s, gx * s:(gx + 1) * s].reshape(n, c, -1)
            out.append(cell.mean(2))
            for th in DARK:
                out.append((cell < th).mean(2))
    feats = np.concatenate([o.reshape(n, -1) for o in out], axis=1).astype(np.float32)
    return {name: i for i, name in enumerate(names)}, feats


def sample_feats(split, inputs):
    idx, per_image = image_scalars(split, inputs)
    seq = np.stack([per_image[[idx[v] for v in inputs[col]]] for col in IMG], axis=1)  # (N,20,F)
    w = inputs[W].to_numpy(np.float64)
    d = np.diff(w, axis=1)
    return np.column_stack([
        w, d, w.mean(1), w.std(1), w.min(1), w.max(1),
        w[:, -1] - w[:, -2], w[:, -1] - w[:, -4], w[:, -4:].mean(1), w[:, -8:].mean(1),
        seq[:, -1], seq.mean(1), seq[:, -4:].mean(1) - seq[:, :4].mean(1), seq.std(1),
    ])


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main():
    tr_x = pd.read_csv(ROOT / "train/inputs.csv")
    va_x = pd.read_csv(ROOT / "validation/inputs.csv")
    te_x = pd.read_csv(ROOT / "test/inputs.csv")
    tr_y = pd.read_csv(ROOT / "train/targets.csv")[T].to_numpy(np.float64)
    va_y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)

    Xtr = sample_feats("train", tr_x)
    Xva = sample_feats("validation", va_x)
    Xte = sample_feats("test", te_x)
    print("feature dim", Xtr.shape, flush=True)

    ridge = RidgeCV(alphas=np.logspace(-3, 4, 30)).fit(Xtr, tr_y)
    rv, rt = ridge.predict(Xva), ridge.predict(Xte)
    gv, gt = np.zeros_like(va_y), np.zeros((len(te_x), 12))
    for h in range(12):
        m = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.06, max_leaf_nodes=31,
            early_stopping=True, validation_fraction=0.12, random_state=777,
        ).fit(Xtr, tr_y[:, h])
        gv[:, h], gt[:, h] = m.predict(Xva), m.predict(Xte)
        print(f"  h{h:02d} val_rmse={rmse(va_y[:, h], gv[:, h]):.2f}", flush=True)

    report = {}
    for name, (pv, pt) in {"img_ridge": (rv, rt), "img_gbdt": (gv, gt),
                           "img_blend": ((rv + gv) / 2, (rt + gt) / 2)}.items():
        np.save(OUT / f"val_{name}.npy", pv)
        np.save(OUT / f"test_{name}.npy", pt)
        report[name] = {"val_rmse": round(rmse(va_y, pv), 3),
                        "per_h": [round(rmse(va_y[:, h], pv[:, h]), 1) for h in range(12)]}
    (OUT / "imgfeat.json").write_text(json.dumps(report, indent=2))
    for name, r in sorted(report.items(), key=lambda kv: kv[1]["val_rmse"]):
        print(f"{name:12s} val_rmse={r['val_rmse']:7.3f}  per_h={r['per_h']}", flush=True)


if __name__ == "__main__":
    main()
