"""imgfeat v2 — 격자를 4×4 로 늘리고, 서쪽 경도대(지구를 향해 자전해 오는 쪽) 스트립과
최근 8슬롯 시계열을 그대로 GBDT 에 넘긴다. 나머지는 v1 과 같다.

규정: train 으로만 학습. validation 은 확인 전용 (RULES.md §1 7항).
"""
import sys

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 777

import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
CACHE = pathlib.Path("/home/jovyan/work/resized_cache/64px")
OUT = pathlib.Path("/home/jovyan/work/preds")
OUT.mkdir(parents=True, exist_ok=True)
IMG = [f"image_{i:02d}" for i in range(20)]
W = [f"wind_{i:02d}" for i in range(20)]
T = [f"target_{i:02d}" for i in range(12)]
DARK = (15, 25, 40, 60, 80)
GRID = 4


def image_scalars(split, inputs):
    arr = np.load(CACHE / f"{split}_images.npy", mmap_mode="r")
    names = sorted(pd.unique(inputs[IMG].to_numpy().ravel()).tolist())
    assert arr.shape[0] == len(names)
    a = np.asarray(arr, dtype=np.uint8)
    n, c, h, w = a.shape
    # 태양 디스크만 남긴다. 64px 캐시는 원본 512 를 그대로 줄인 것이라 디스크가 프레임에 꽉 찬다.
    yy, xx = np.mgrid[0:h, 0:w]
    disk = ((yy - h / 2 + .5) ** 2 + (xx - w / 2 + .5) ** 2) <= (0.47 * h) ** 2
    flat = a.reshape(n, c, -1)[:, :, disk.ravel()]
    out = [flat.mean(2), flat.std(2), np.percentile(flat, 10, axis=2), np.percentile(flat, 90, axis=2)]
    out += [(flat < th).mean(2) for th in DARK]
    s = h // GRID
    for gy in range(GRID):
        for gx in range(GRID):
            cell = a[:, :, gy * s:(gy + 1) * s, gx * s:(gx + 1) * s].reshape(n, c, -1)
            out.append(cell.mean(2))
            out += [(cell < th).mean(2) for th in (25, 40, 60)]
    # 서쪽(오른쪽) 경도 스트립 — 여기 있는 코로나홀이 며칠 뒤 지구를 때린다
    for lo, hi in ((0.55, 0.75), (0.75, 0.95)):
        strip = a[:, :, :, int(lo * w):int(hi * w)].reshape(n, c, -1)
        out.append(strip.mean(2))
        out += [(strip < th).mean(2) for th in (25, 40, 60)]
    return {v: i for i, v in enumerate(names)}, np.concatenate(
        [o.reshape(n, -1) for o in out], axis=1).astype(np.float32)


def sample_feats(split, inputs):
    idx, per = image_scalars(split, inputs)
    seq = np.stack([per[[idx[v] for v in inputs[col]]] for col in IMG], axis=1)   # (N,20,F)
    w = inputs[W].to_numpy(np.float64)
    d = np.diff(w, axis=1)
    key = seq[:, :, :2 * (4 + len(DARK))]      # 전체 디스크 스칼라만 시계열로 통째 투입
    return np.column_stack([
        w, d, w.mean(1), w.std(1), w.min(1), w.max(1),
        w[:, -1] - w[:, -2], w[:, -1] - w[:, -4], w[:, -4:].mean(1), w[:, -8:].mean(1),
        seq[:, -1], seq.mean(1), seq[:, -4:].mean(1) - seq[:, :4].mean(1), seq.std(1),
        key[:, -8:].reshape(len(w), -1),
    ])


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


tr_x = pd.read_csv(ROOT / "train/inputs.csv")
va_x = pd.read_csv(ROOT / "validation/inputs.csv")
te_x = pd.read_csv(ROOT / "test/inputs.csv")
tr_y = pd.read_csv(ROOT / "train/targets.csv")[T].to_numpy(np.float64)
va_y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)

Xtr, Xva, Xte = sample_feats("train", tr_x), sample_feats("validation", va_x), sample_feats("test", te_x)
print("feature dim", Xtr.shape, flush=True)

fit_x, eval_x, fit_y, eval_y = train_test_split(Xtr, tr_y, test_size=0.2, random_state=SEED)
gv, gt = np.zeros((len(va_x), 12), dtype=np.float64), np.zeros((len(te_x), 12), dtype=np.float64)
for h in range(12):
    m = XGBRegressor(
        n_estimators=3000, learning_rate=0.03, max_depth=6, subsample=0.8,
        colsample_bytree=0.6, min_child_weight=8, reg_lambda=2.0,
        early_stopping_rounds=80, n_jobs=6, tree_method="hist", random_state=SEED,
    ).fit(fit_x, fit_y[:, h], eval_set=[(eval_x, eval_y[:, h])], verbose=False)
    gv[:, h], gt[:, h] = m.predict(Xva), m.predict(Xte)
    print(f"  h{h:02d} val_rmse={rmse(va_y[:, h], gv[:, h]):.2f}", flush=True)

report = {"xgb": {"val_rmse": round(rmse(va_y, gv), 3),
                  "per_h": [round(rmse(va_y[:, h], gv[:, h]), 1) for h in range(12)]}}
np.save(OUT / f"val_xgb{SEED}.npy", gv)
np.save(OUT / f"test_xgb{SEED}.npy", gt)
(OUT / f"xgb{SEED}.json").write_text(json.dumps(report, indent=2))
for name, r in report.items():
    print(f"{name:12s} val_rmse={r['val_rmse']:7.3f}  per_h={r['per_h']}", flush=True)

