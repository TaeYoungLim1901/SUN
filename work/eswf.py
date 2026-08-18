"""ESWF 계열 피처 — 중심 자오선 슬라이스 CH 면적 + 속도 가설 다중 투영.

근거 (DOMAIN §1.2, §1.3, §6):
  · Vršnak/Rotter → ESWF 의 핵심 경험 관계는
    **중심 자오선 ±15° 슬라이스 안의 코로나 홀 면적 비율 ↔ 약 4일 뒤 지구 도달 속도** 다.
    지금까지 우리 피처는 **서쪽 림 스트립**을 봤다. 이건 근거가 다른 물건이었다 (E005/E007).
  · 전파 지연은 고정 4일이 아니라 τ = 1AU/v 로 v 에 따라 2.2~5.8일이다.
    고정 지연을 쓰면 HSS 도착 시각을 늦게 찍는다 (DOMAIN §1.3).
  · 그래서 **속도 가설을 여러 개 동시에 투영한다.** 후보 v 마다 발사 프레임이 다르고,
    빠른 스트림이 느린 스트림을 따라잡는 겹침(SIR)은 「여러 가설이 같은 도착시각을 가리키는 것」으로
    자연스럽게 표현된다. 어느 가설이 맞는지는 GBDT 가 고르게 둔다.

E007 의 실패에서 배운 것: 창을 좁혀 **대체**하면 정보가 준다. 여기서는 전역 요약을 **남겨두고**
슬라이스·가설 피처를 **더한다**.

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
TAG = sys.argv[2] if len(sys.argv) > 2 else "eswf"
# 속도 가설. 전파시간 τ=1AU/v 가 2.5일(700) ~ 5.8일(300) 을 덮는다 (DOMAIN §1.3 표).
V_HYP = (300.0, 350.0, 400.0, 450.0, 500.0, 600.0, 700.0)
# CH 임계는 이미지별 디스크 중앙값 대비 상대값. 「35% 를 그대로」는 닫힌 축이라 여러 개를 준다.
REL = (0.35, 0.45, 0.55, 0.70)


def per_image(split, inputs):
    """유니크 이미지 → 슬라이스별 CH 면적 등 스칼라."""
    arr = np.load(CACHE / f"{split}_images.npy", mmap_mode="r")
    names = sorted(pd.unique(inputs[IMG].to_numpy().ravel()).tolist())
    assert arr.shape[0] == len(names), (arr.shape, len(names))
    a = np.asarray(arr, dtype=np.float32)
    n, c, h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 0.47 * h                                  # 디스크 반지름 (64px 캐시 기준)
    dx = (xx - w / 2 + .5) / r                    # −1..1, 태양 반지름 단위
    dy = (yy - h / 2 + .5) / r
    disk = (dx ** 2 + dy ** 2) <= 1.0

    flat = a.reshape(n, c, -1)[:, :, disk.ravel()]
    med = np.median(flat, axis=2, keepdims=True)  # 이미지별 기준 밝기
    out = [flat.mean(2), flat.std(2), med[:, :, 0]]
    out += [(flat < med * f).mean(2) for f in REL]          # 전역 CH 면적비

    # 경도 슬라이스. 원반 위 x = sin(경도) 이므로 ±15° → |x| < sin15° = 0.259
    slices = {"m15": np.abs(dx) < np.sin(np.deg2rad(15)),
              "m30": np.abs(dx) < np.sin(np.deg2rad(30)),
              "m60": np.abs(dx) < np.sin(np.deg2rad(60))}
    # 저위도대(|위도| < 30°) — 지구를 때리는 건 적도 근처 코로나 홀이다
    lowlat = np.abs(dy) < np.sin(np.deg2rad(30))
    for name, sl in slices.items():
        for lat_name, lat in (("all", np.ones_like(lowlat)), ("eq", lowlat)):
            m = sl & lat & disk
            if m.sum() == 0:
                continue
            b = a.reshape(n, c, -1)[:, :, m.ravel()]
            out.append(b.mean(2))
            out += [(b < med * f).mean(2) for f in REL]
            del b
    # 4×3 격자 CH 면적 — Collin 이 실제로 쓴 격자 (DOMAIN §6 #2). 상대 임계로 낸다.
    for gy in range(3):
        for gx in range(4):
            m = (disk & (dy >= -1 + 2 * gy / 3) & (dy < -1 + 2 * (gy + 1) / 3)
                 & (dx >= -1 + 2 * gx / 4) & (dx < -1 + 2 * (gx + 1) / 4))
            if m.sum() == 0:
                continue
            b = a.reshape(n, c, -1)[:, :, m.ravel()]
            out.append(b.mean(2))
            out += [(b < med * f).mean(2) for f in (0.45, 0.7)]
            del b

    # 밝기 히스토그램 분위수 (DOMAIN §6 #7 — 가능, 단 상대량이다)
    out += [np.percentile(flat, q, axis=2) for q in (5, 10, 25, 75, 90, 95)]
    # 활성 영역 대리량: 디스크 중앙값보다 훨씬 밝은 픽셀 비율. 저속풍 국면 지시자.
    # (§6 #8 의 '흑점수 시계열'은 외부 데이터라 불가. 이건 픽셀에서 직접 재는 다른 양이다)
    out += [(flat > med * k).mean(2) for k in (1.5, 2.0, 3.0)]

    # limb 보정: 중심에서 멀수록 시선 방향이 기울어 면적이 줄어 보인다 (DOMAIN §2.1 (a))
    mu = np.sqrt(np.clip(1.0 - dx ** 2 - dy ** 2, 1e-3, 1.0))
    wgt = (1.0 / mu * (np.abs(dx) < np.sin(np.deg2rad(60))) * disk).ravel()
    wgt = wgt[disk.ravel()] / (wgt[disk.ravel()].sum() + 1e-6)
    out += [((flat < med * f) * wgt).sum(2) for f in REL]

    f = np.concatenate([o.reshape(n, -1) for o in out], axis=1).astype(np.float32)
    return {v: i for i, v in enumerate(names)}, f


def sequences(split, inputs):
    idx, per = per_image(split, inputs)
    seq = np.stack([per[[idx[v] for v in inputs[col]]] for col in IMG], axis=1)
    assert seq.shape[:2] == (len(inputs), 20)
    return seq


def build(split, inputs, horizon_hours):
    """전역 요약(그대로 유지) + 속도 가설별 발사 프레임 투영(추가)."""
    seq = SEQ[split]
    n = len(inputs)
    rows = np.arange(n)
    w = inputs[W].to_numpy(np.float64)
    dw = np.diff(w, axis=1)

    proj = []
    for v in V_HYP:
        tau_h = AU_KM / v / 3600.0
        j = int(round(19.0 - (tau_h - horizon_hours) / 6.0))
        ok = 0 <= j <= 19
        proj.append(seq[rows, min(max(j, 0), 19)] * (1.0 if ok else 0.0))
        proj.append(np.full((n, 1), float(ok)))
    # 관측 속도 기반 가설 하나 더 — 지금 부는 바람이 계속된다면 어느 프레임인가
    tau_obs = AU_KM / np.clip(w[:, -1], 250.0, 900.0) / 3600.0
    j_obs = np.clip(np.rint(19.0 - (tau_obs - horizon_hours) / 6.0).astype(int), 0, 19)
    proj.append(seq[rows, j_obs])

    # 고정 지연 (문헌 그대로: target 시각 기준 t−3/4/5일). DOMAIN §9.2 표대로 창 밖이면 마스크한다.
    for lag_d in (3.0, 4.0, 5.0):
        j = int(round(19.0 - (lag_d * 24.0 - horizon_hours) / 6.0))
        ok = 0 <= j <= 19
        proj.append(seq[rows, min(max(j, 0), 19)] * (1.0 if ok else 0.0))
        proj.append(np.full((n, 1), float(ok)))

    # SIR 상승 엣지 (DOMAIN §1.4) — 톱니는 비대칭이라 상승 쪽이 정보가 많다
    rise = np.maximum(dw, 0.0)
    fall = np.minimum(dw, 0.0)
    edge = np.column_stack([
        rise.max(1), fall.min(1), rise.sum(1), fall.sum(1),
        np.argmax(dw, axis=1), np.argmin(dw, axis=1),
        (dw > 0).mean(1), dw[:, -4:].mean(1), dw[:, -4:].max(1),
    ])

    return np.column_stack([
        w, dw, w.mean(1), w.std(1), w.min(1), w.max(1),
        w[:, -1] - w[:, -2], w[:, -4:].mean(1), w[:, -8:].mean(1),
        seq[:, -1], seq.mean(1), seq.std(1),
        seq[:, -4:].mean(1) - seq[:, :4].mean(1),
        edge,
        np.column_stack(proj),
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
print("per-image dim", SEQ["train"].shape, flush=True)

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
    print(f"  h{h:02d} ({hh:3.0f}h) dim={Xtr.shape[1]} val_rmse={rmse(va_y[:, h], gv[:, h]):.2f}",
          flush=True)

np.save(OUT / f"val_{TAG}.npy", gv)
np.save(OUT / f"test_{TAG}.npy", gt)
(OUT / f"{TAG}.json").write_text(json.dumps(
    {"val_rmse": round(rmse(va_y, gv), 3),
     "per_h": [round(rmse(va_y[:, h], gv[:, h]), 1) for h in range(12)]}, indent=2))
print(f"{TAG} val_rmse={rmse(va_y, gv):.3f}", flush=True)
