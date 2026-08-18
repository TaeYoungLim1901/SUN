"""v3 — horizon 별 '발사시각 정합' 피처.

문제의식(Grok 판정 · DOMAIN §4): 지금까지 모든 horizon 에 **같은 영상 요약**을 먹였다.
그런데 t+h 에 지구에 닿는 태양풍은 h 보다 **전파지연(≈3~5일)만큼 전에** 태양을 떠났다.
그러니 h=6h 를 맞히려면 창의 **앞쪽** 프레임을, h=72h 를 맞히려면 **뒤쪽** 프레임을 봐야 한다.
마지막 프레임과 전체 평균만 주면 도착시각 정보가 뭉개진다.

  전파시간 τ(v) = 1AU / v.  v=400km/s → 103.9h, 600 → 69.3h, 300 → 138.6h
  발사시각 = t + h − τ(v̂),  v̂ = 마지막 관측 풍속 (test 에서도 알 수 있는 값만 쓴다)
  프레임 j 의 시각 = t − (19−j)·6h   →  j* = 19 − (t − 발사시각)/6

horizon 마다 j* 주변 ±2 프레임을 잘라 넣는다. 창 밖이면 마스크(0)와 플래그를 같이 준다.

그 밖에 v2 대비 추가한 것(전부 '다른 족' 기하 피처):
  · 디스크 중앙값 대비 **상대 임계** 암부 비율 (절대 밝기 드리프트에 둔감)
  · 암부 **centroid (x, y)** 와 면적 — 코로나홀이 어디 있는지
  · 적도대 / 극지대 분리 (지구를 때리는 건 저위도 코로나홀이다)
  · 193/211 비 — 온도대가 다른 두 채널의 대비

규정: train 으로만 학습. validation 은 확인 전용 (RULES.md §1 7항).
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
CACHE = pathlib.Path("/home/jovyan/work/resized_cache/64px")
OUT = pathlib.Path("/home/jovyan/work/preds")
OUT.mkdir(parents=True, exist_ok=True)
IMG = [f"image_{i:02d}" for i in range(20)]
W = [f"wind_{i:02d}" for i in range(20)]
T = [f"target_{i:02d}" for i in range(12)]
AU_KM = 1.495978707e8
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 777
TAG = sys.argv[2] if len(sys.argv) > 2 else "v3"


def per_image(split, inputs):
    """유니크 이미지 한 장 → 스칼라 벡터. 캐시 순서는 sorted(unique) 로 다른 스크립트와 동일."""
    arr = np.load(CACHE / f"{split}_images.npy", mmap_mode="r")
    names = sorted(pd.unique(inputs[IMG].to_numpy().ravel()).tolist())
    assert arr.shape[0] == len(names), (arr.shape, len(names))
    a = np.asarray(arr, dtype=np.float32)                    # (n, 2, 64, 64)
    n, c, h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    disk = ((yy - h / 2 + .5) ** 2 + (xx - w / 2 + .5) ** 2) <= (0.47 * h) ** 2
    dm = disk.ravel()
    flat = a.reshape(n, c, -1)[:, :, dm]
    med = np.median(flat, axis=2, keepdims=True)             # 이미지별 디스크 중앙값
    out = [flat.mean(2), flat.std(2), med[:, :, 0],
           np.percentile(flat, 5, axis=2), np.percentile(flat, 95, axis=2)]

    # 상대 임계 암부 — 절대 밝기 드리프트에 둔감하다
    masks = {}
    for f in (0.35, 0.5, 0.65, 0.8):
        m = flat < med * f
        masks[f] = m
        out.append(m.mean(2))

    # 암부 centroid 와 2차 모멘트. 코로나홀의 '위치'가 도착 여부를 가른다.
    ys, xs = yy.ravel()[dm], xx.ravel()[dm]
    for f in (0.5, 0.65):
        m = masks[f].astype(np.float32)
        s = m.sum(2) + 1e-6
        cx = (m * xs).sum(2) / s
        cy = (m * ys).sum(2) / s
        out += [cx / w, cy / h,
                np.sqrt((m * (xs - cx[..., None]) ** 2).sum(2) / s) / w,
                np.sqrt((m * (ys - cy[..., None]) ** 2).sum(2) / s) / h]

    # 위도대 분리: 적도대(중앙 40%) vs 극지대. 지구를 때리는 건 저위도 홀이다.
    lat_eq = (np.abs(yy - h / 2 + .5) < 0.20 * h) & disk
    lat_pl = (~lat_eq) & disk
    for band in (lat_eq, lat_pl):
        b = a.reshape(n, c, -1)[:, :, band.ravel()]
        out.append(b.mean(2))
        out.append((b < med * 0.5).mean(2))
    # 서쪽(오른쪽) 경도대 — 자전으로 곧 지구를 향하는 쪽
    for lo, hi in ((0.55, 0.78), (0.78, 1.0)):
        s_ = a[:, :, :, int(lo * w):int(hi * w)].reshape(n, c, -1)
        out.append(s_.mean(2))
        out.append((s_ < med * 0.5).mean(2))

    f = np.concatenate([o.reshape(n, -1) for o in out], axis=1)
    ratio = (f[:, 0::2] + 1e-3) / (f[:, 1::2] + 1e-3)        # 193 대 211 대비
    return {v: i for i, v in enumerate(names)}, np.concatenate(
        [f, ratio], axis=1).astype(np.float32)


def sequences(split, inputs):
    idx, per = per_image(split, inputs)
    seq = np.stack([per[[idx[v] for v in inputs[col]]] for col in IMG], axis=1)   # (N,20,F)
    assert seq.shape[:2] == (len(inputs), 20)
    return seq


def launch_frame(inputs, horizon_hours):
    """발사 프레임 인덱스 j*. 마지막 관측 풍속으로 전파시간을 추정한다."""
    v = np.clip(inputs[W[-1]].to_numpy(np.float64), 250.0, 900.0)
    tau_h = AU_KM / v / 3600.0
    offset_h = tau_h - horizon_hours          # 현재 시각 t 로부터 몇 시간 전인가
    return 19.0 - offset_h / 6.0, tau_h       # 프레임 인덱스(실수)


def build(split, inputs, horizon_hours):
    seq = SEQ[split]
    n, t, f = seq.shape
    jstar, tau = launch_frame(inputs, horizon_hours)
    rows = np.arange(n)
    win, mask = [], []
    for d in (-2, -1, 0, 1, 2):
        j = np.rint(jstar + d).astype(int)
        ok = (j >= 0) & (j <= 19)
        win.append(seq[rows, np.clip(j, 0, 19)] * ok[:, None])
        mask.append(ok.astype(np.float32))
    w = inputs[W].to_numpy(np.float64)
    dw = np.diff(w, axis=1)
    return np.column_stack([
        w, dw, w.mean(1), w.std(1), w.min(1), w.max(1),
        w[:, -1] - w[:, -2], w[:, -4:].mean(1), w[:, -8:].mean(1),
        tau, jstar, np.column_stack(mask),
        np.concatenate(win, axis=1),                 # 발사시각 정합 창 (5프레임)
        seq[:, -1], seq.mean(1), seq.std(1),          # v2 와 겹치는 전역 요약도 남긴다
        seq[:, -4:].mean(1) - seq[:, :4].mean(1),
    ])


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)))


tr_x = pd.read_csv(ROOT / "train/inputs.csv")
va_x = pd.read_csv(ROOT / "validation/inputs.csv")
te_x = pd.read_csv(ROOT / "test/inputs.csv")
tr_y = pd.read_csv(ROOT / "train/targets.csv")[T].to_numpy(np.float64)
va_y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)

SEQ = {s: sequences(s, x) for s, x in
       (("train", tr_x), ("validation", va_x), ("test", te_x))}
print("per-image feature dim", SEQ["train"].shape, flush=True)

gv, gt = np.zeros_like(va_y), np.zeros((len(te_x), 12))
for h in range(12):
    hh = 6.0 * (h + 1)
    Xtr, Xva, Xte = (build("train", tr_x, hh), build("validation", va_x, hh),
                     build("test", te_x, hh))
    m = HistGradientBoostingRegressor(
        max_iter=700, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.12, n_iter_no_change=30, random_state=SEED,
    ).fit(Xtr, tr_y[:, h])
    gv[:, h], gt[:, h] = m.predict(Xva), m.predict(Xte)
    print(f"  h{h:02d} ({hh:.0f}h) dim={Xtr.shape[1]} val_rmse={rmse(va_y[:, h], gv[:, h]):.2f}",
          flush=True)

np.save(OUT / f"val_{TAG}.npy", gv)
np.save(OUT / f"test_{TAG}.npy", gt)
(OUT / f"{TAG}.json").write_text(json.dumps(
    {"val_rmse": round(rmse(va_y, gv), 3),
     "per_h": [round(rmse(va_y[:, h], gv[:, h]), 1) for h in range(12)]}, indent=2))
print(f"{TAG} val_rmse={rmse(va_y, gv):.3f}", flush=True)
