# %% [markdown]
# # 태양풍 속도 예측 — 제출 파이프라인
#
# 이 노트북 **하나만으로** 학습·추론·제출파일 생성이 전부 된다. 외부 .py import 없음.
#
# - `MODE = "infer"` (기본): `model.pth` 를 불러 test 를 추론하고 `submission.csv` 를 쓴다.
# - `MODE = "train"`: train split 으로 처음부터 학습해 `model.pth` 를 새로 만든다.
#
# 모델은 **영상 스칼라 피처 위의 GBDT/Ridge 3종 결합**이다. 왜 이 구성인가:
#
# | 모델 | validation RMSE |
# |---|---|
# | persistence (마지막 관측 유지) | 87.6 |
# | 대회 공식 베이스라인 3D-CNN+LSTM | 78.1 |
# | 풍속 20개만 사용 | 75.0 |
# | 영상 스칼라 + GBDT + ESWF + CNN 결합 | **63.8** |
#
# 규정 준수: pretrained weight 미사용 · 외부 데이터 미사용 ·
# **validation 은 학습에 쓰지 않는다**(점수 확인 전용) · test target 추정 없음.

# %%
import json
import math
import os
import pathlib
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from torch import nn

# Packed CNN artifact constants: v6 = mean, std, px, norm, wm, fstd; v7 = the same plus smu, ssd; v9 = mean, std, px, norm, wm, fstd.

MODE = os.environ.get("SW_MODE", "infer")   # "infer" (기본) | "train"
SEED = 777
PX = 64                 # 스칼라 캐시. 원본 PNG 는 512×512.
PX96 = 96               # CNN 입력
CLIP = (200.0, 900.0)   # 지구 근처 태양풍의 물리적 범위 (RULES.md §6-4)
DARK_V1, GRID_V1 = (20, 40, 60), 2
DARK_V2, GRID_V2 = (15, 25, 40, 60, 80), 4
AU_KM = 1.495978707e8
AU_HOURS = 41556.0      # 1 AU / (1 km/s) 시간. E016 j* 정합
NGLOBAL_V2 = 2 * (4 + len(DARK_V2))
V_HYP = (300.0, 350.0, 400.0, 450.0, 500.0, 600.0, 700.0)
REL = (0.35, 0.45, 0.55, 0.70)
# 결합 가중치. kit.hill_climb 이 validation 원성분 6개에서 고른 값.
WEIGHTS = {"ridge_v1": 0.10, "gbdt_v1": 0.23, "gbdt_v2": 0.12,
           "eswf": 0.25, "cnn3a": 0.15, "cnn3b": 0.15}
OLD_WEIGHTS = {"ridge_v1": 0.125, "gbdt_v1": 0.25, "gbdt_v2": 0.125,
               "eswf": 0.25, "cnn96b": 0.125, "cnn96w3": 0.125}
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_data_root():
    """노트북이 submission/ 안에서 돌든 /home/jovyan 에서 돌든 데이터를 찾는다."""
    here = pathlib.Path.cwd()
    for base in (here, *here.parents):
        for cand in (base / "public_dataset/competition_dataset_6h",
                     base / "competition_dataset_6h"):
            if (cand / "train/inputs.csv").exists():
                return cand
    raise FileNotFoundError("competition_dataset_6h 를 못 찾았다")


ROOT = find_data_root()
HERE = pathlib.Path.cwd()


def resolve_cache(px):
    for p in (pathlib.Path("/home/jovyan/work/resized_cache") / f"{px}px",
              pathlib.Path.home() / "sw_cache" / f"{px}px"):
        if (p / "train_images.npy").exists() or p == pathlib.Path.home() / "sw_cache" / f"{px}px":
            p.mkdir(parents=True, exist_ok=True)
            return p
    p = pathlib.Path.home() / "sw_cache" / f"{px}px"
    p.mkdir(parents=True, exist_ok=True)
    return p


CACHE = resolve_cache(PX)
CACHE96 = resolve_cache(PX96)
_CACHES = {PX: CACHE, PX96: CACHE96}
_DISK = {}   # med 정규화용 원판 마스크, 해상도별


def cache_for(px):
    """CNN 태그가 96 이 아닌 해상도를 쓸 수 있다 (cnn3c = 128px)."""
    if px not in _CACHES:
        _CACHES[px] = resolve_cache(px)
    return _CACHES[px]
IMG = [f"image_{i:02d}" for i in range(20)]
W = [f"wind_{i:02d}" for i in range(20)]
T = [f"target_{i:02d}" for i in range(12)]
np.random.seed(SEED)
torch.manual_seed(SEED)
print("data:", ROOT)

# %% [markdown]
# ## 1. 이미지 캐시
#
# 샘플들이 같은 PNG 를 공유한다 — train 샘플 9,607개가 쓰는 유니크 이미지는 10,139장뿐이다.
# 그래서 **유니크 파일당 한 번만** 읽어 64×64 uint8 로 캐시한다. 반복 실행은 캐시를 재사용한다.

# %%
def build_cache(split, inputs, px=PX, cache=None):
    cache = CACHE if cache is None else cache
    path = cache / f"{split}_images.npy"
    names = sorted(pd.unique(inputs[IMG].to_numpy().ravel()).tolist())
    if not path.exists():
        tmp = path.with_suffix(".partial")
        arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8,
                                        shape=(len(names), 2, px, px))
        t0 = time.perf_counter()
        for i, name in enumerate(names):
            for c, ch in enumerate(("193", "211")):
                with Image.open(ROOT / split / ch / name) as im:
                    arr[i, c] = np.asarray(
                        im.convert("L").resize((px, px), Image.BILINEAR), dtype=np.uint8)
            if (i + 1) % 2000 == 0:
                print(f"  {split} {i+1}/{len(names)} ({time.perf_counter()-t0:.0f}s)", flush=True)
        arr.flush()
        del arr
        tmp.rename(path)
    a = np.load(path)
    assert a.shape == (len(names), 2, px, px)
    return {v: i for i, v in enumerate(names)}, a


def flow_pairs(split, inputs, idx, arr, px):
    """연속 프레임쌍 Farneback 광류. 파라미터는 work/cnn6.py flow_pairs 와 같다."""
    import cv2
    cache = cache_for(px)
    path = cache / f"flowfield_{px}px.npy"
    kpath = cache / f"flowfield_{px}px_keys.txt"
    tab, nl = chr(9), chr(10)

    def keylines(keys, with_nl=False):
        return [a + tab + b + (nl if with_nl else "") for a, b in keys]

    names = inputs[IMG].to_numpy()
    seen = {}
    for row in names:
        for t in range(19):
            seen.setdefault((row[t], row[t + 1]), len(seen))
    keys = list(seen)
    if path.exists() and kpath.exists() and (
            kpath.read_text(encoding="utf-8").splitlines() == keylines(keys)):
        return seen, np.load(path, mmap_mode="r")
    if path.exists():
        path.unlink()
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                    shape=(len(keys), 2, px, px))
    t0 = time.perf_counter()
    for i, (na, nb) in enumerate(keys):
        a = np.asarray(arr[idx[na], 0], np.uint8)      # 193 채널, 정규화 전 uint8
        b = np.asarray(arr[idx[nb], 0], np.uint8)
        f = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 11, 3, 5, 1.2, 0)
        out[i, 0] = f[..., 0].astype(np.float16)
        out[i, 1] = f[..., 1].astype(np.float16)
        if (i + 1) % 2000 == 0:
            print(f"  {split} flow {i+1}/{len(keys)} ({time.perf_counter()-t0:.0f}s)", flush=True)
    out.flush()
    kpath.write_text("".join(keylines(keys, True)), encoding="utf-8")
    return seen, np.load(path, mmap_mode="r")


# %% [markdown]
# ## 2. 이미지 → 스칼라 피처
#
# 코로나 홀은 EUV 영상에서 **어둡게** 보이고 거기서 고속 태양풍이 나온다.
# 그래서 "어두운 영역이 어디에 얼마나 있나" 를 숫자로 뽑는다. 두 벌을 만든다.
#
# - **v1**: 전체/2×2 격자의 밝기와 암부 비율. 거칠지만 튼튼하다.
# - **v2**: 태양 디스크 마스크 + 4×4 격자 + 서쪽 경도대(자전으로 곧 지구를 향하는 쪽)
#   + 최근 8시점 시계열을 그대로 투입.
#
# 두 벌의 잔차가 완전히 겹치지 않아서 결합에서 이득이 난다.

# %%
def image_scalars_v1(split, inputs):
    """유니크 이미지 → (n_unique, F) 스칼라. 캐시 파일 순서는 샘플코드와 동일하게 sorted(unique)."""
    idx0, arr = build_cache(split, inputs)
    names = list(idx0)
    assert arr.shape[0] == len(names), (split, arr.shape, len(names))
    a = np.asarray(arr, dtype=np.uint8)
    n, c, h, w = a.shape
    out = [a.reshape(n, c, -1).mean(2), a.reshape(n, c, -1).std(2)]
    for th in DARK_V1:
        out.append((a.reshape(n, c, -1) < th).mean(2))
    s = h // GRID_V1
    for gy in range(GRID_V1):
        for gx in range(GRID_V1):
            cell = a[:, :, gy * s:(gy + 1) * s, gx * s:(gx + 1) * s].reshape(n, c, -1)
            out.append(cell.mean(2))
            for th in DARK_V1:
                out.append((cell < th).mean(2))
    feats = np.concatenate([o.reshape(n, -1) for o in out], axis=1).astype(np.float32)
    return {name: i for i, name in enumerate(names)}, feats

def sample_feats_v1(split, inputs):
    idx, per_image = image_scalars_v1(split, inputs)
    seq = np.stack([per_image[[idx[v] for v in inputs[col]]] for col in IMG], axis=1)  # (N,20,F)
    w = inputs[W].to_numpy(np.float64)
    d = np.diff(w, axis=1)
    return np.column_stack([
        w, d, w.mean(1), w.std(1), w.min(1), w.max(1),
        w[:, -1] - w[:, -2], w[:, -1] - w[:, -4], w[:, -4:].mean(1), w[:, -8:].mean(1),
        seq[:, -1], seq.mean(1), seq[:, -4:].mean(1) - seq[:, :4].mean(1), seq.std(1),
    ])


def image_scalars_v2(split, inputs):
    idx0, arr = build_cache(split, inputs)
    names = list(idx0)
    assert arr.shape[0] == len(names)
    a = np.asarray(arr, dtype=np.uint8)
    n, c, h, w = a.shape
    # 태양 디스크만 남긴다. 64px 캐시는 원본 512 를 그대로 줄인 것이라 디스크가 프레임에 꽉 찬다.
    yy, xx = np.mgrid[0:h, 0:w]
    disk = ((yy - h / 2 + .5) ** 2 + (xx - w / 2 + .5) ** 2) <= (0.47 * h) ** 2
    flat = a.reshape(n, c, -1)[:, :, disk.ravel()]
    out = [flat.mean(2), flat.std(2), np.percentile(flat, 10, axis=2), np.percentile(flat, 90, axis=2)]
    out += [(flat < th).mean(2) for th in DARK_V2]
    s = h // GRID_V2
    for gy in range(GRID_V2):
        for gx in range(GRID_V2):
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

def sample_feats_v2(split, inputs):
    idx, per = image_scalars_v2(split, inputs)
    seq = np.stack([per[[idx[v] for v in inputs[col]]] for col in IMG], axis=1)   # (N,20,F)
    w = inputs[W].to_numpy(np.float64)
    d = np.diff(w, axis=1)
    key = seq[:, :, :2 * (4 + len(DARK_V2))]      # 전체 디스크 스칼라만 시계열로 통째 투입
    return np.column_stack([
        w, d, w.mean(1), w.std(1), w.min(1), w.max(1),
        w[:, -1] - w[:, -2], w[:, -1] - w[:, -4], w[:, -4:].mean(1), w[:, -8:].mean(1),
        seq[:, -1], seq.mean(1), seq[:, -4:].mean(1) - seq[:, :4].mean(1), seq.std(1),
        key[:, -8:].reshape(len(w), -1),
    ])


def sample_feats(split, inputs):
    return sample_feats_v2(split, inputs)


def align_seq(split, inputs):
    """v2 시점별 스칼라에서 정합에 쓰는 전역+서쪽 스트립만. E016."""
    idx, per = image_scalars_v2(split, inputs)
    seq = np.stack([per[[idx[v] for v in inputs[col]]] for col in IMG], axis=1)
    return np.concatenate([seq[:, :, :NGLOBAL_V2], seq[:, :, -32:]], axis=2)


def align_gather(seq, j):
    j = np.clip(j, 0, 19)
    lo = np.floor(j).astype(int)
    hi = np.minimum(lo + 1, 19)
    t = (j - lo).astype(np.float32)[:, None]
    r = np.arange(len(seq))
    return seq[r, lo] * (1 - t) + seq[r, hi] * t


def align_block(seq, w, h):
    cols = []
    for v in (w[:, -1], w[:, -8:].mean(1)):
        tau = AU_HOURS / np.maximum(v, 200.0)
        js = 19 + (6 * (h + 1) - tau) / 6.0
        cols += [align_gather(seq, js), tau[:, None], js[:, None],
                 align_gather(seq, js) - align_gather(seq, np.full_like(js, 19.0))]
    return np.column_stack(cols)


def t4_block(seq, h):
    """target 기준 t−4d = 프레임 h+4. DOMAIN §9.2."""
    j = int(np.clip(h + 4, 0, 19))
    lo, hi = max(j - 1, 0), min(j + 1, 19)
    return np.column_stack([
        seq[:, j], seq[:, lo], seq[:, hi],
        seq[:, j] - seq[:, 19], seq[:, j] - seq[:, 0],
    ])


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)))


def eswf_per_image(split, inputs):
    idx0, arr = build_cache(split, inputs, PX, CACHE)
    names = list(idx0)
    a = np.asarray(arr, dtype=np.float32)
    n, c, h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 0.47 * h
    dx, dy = (xx - w / 2 + .5) / r, (yy - h / 2 + .5) / r
    disk = (dx ** 2 + dy ** 2) <= 1.0
    flat = a.reshape(n, c, -1)[:, :, disk.ravel()]
    med = np.median(flat, axis=2, keepdims=True)
    out = [flat.mean(2), flat.std(2), med[:, :, 0]]
    out += [(flat < med * f).mean(2) for f in REL]
    slices = {"m15": np.abs(dx) < np.sin(np.deg2rad(15)),
              "m30": np.abs(dx) < np.sin(np.deg2rad(30)),
              "m60": np.abs(dx) < np.sin(np.deg2rad(60))}
    lowlat = np.abs(dy) < np.sin(np.deg2rad(30))
    for sl in slices.values():
        for lat in (np.ones_like(lowlat), lowlat):
            m = sl & lat & disk
            if m.sum() == 0:
                continue
            b = a.reshape(n, c, -1)[:, :, m.ravel()]
            out.append(b.mean(2))
            out += [(b < med * f).mean(2) for f in REL]
    for gy in range(3):
        for gx in range(4):
            m = (disk & (dy >= -1 + 2 * gy / 3) & (dy < -1 + 2 * (gy + 1) / 3)
                 & (dx >= -1 + 2 * gx / 4) & (dx < -1 + 2 * (gx + 1) / 4))
            if m.sum() == 0:
                continue
            b = a.reshape(n, c, -1)[:, :, m.ravel()]
            out.append(b.mean(2))
            out += [(b < med * f).mean(2) for f in (0.45, 0.7)]
    out += [np.percentile(flat, q, axis=2) for q in (5, 10, 25, 75, 90, 95)]
    out += [(flat > med * k).mean(2) for k in (1.5, 2.0, 3.0)]
    mu = np.sqrt(np.clip(1.0 - dx ** 2 - dy ** 2, 1e-3, 1.0))
    wgt = (1.0 / mu * (np.abs(dx) < np.sin(np.deg2rad(60))) * disk).ravel()
    wgt = wgt[disk.ravel()] / (wgt[disk.ravel()].sum() + 1e-6)
    out += [((flat < med * f) * wgt).sum(2) for f in REL]
    return {v: i for i, v in enumerate(names)}, np.concatenate(
        [o.reshape(n, -1) for o in out], axis=1).astype(np.float32)


def eswf_seq(split, inputs):
    idx, per = eswf_per_image(split, inputs)
    return np.stack([per[[idx[v] for v in inputs[col]]] for col in IMG], axis=1)


def eswf_feats(seq, inputs, horizon_hours):
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
    tau_obs = AU_KM / np.clip(w[:, -1], 250.0, 900.0) / 3600.0
    j_obs = np.clip(np.rint(19.0 - (tau_obs - horizon_hours) / 6.0).astype(int), 0, 19)
    proj.append(seq[rows, j_obs])
    for lag_d in (3.0, 4.0, 5.0):
        j = int(round(19.0 - (lag_d * 24.0 - horizon_hours) / 6.0))
        ok = 0 <= j <= 19
        proj.append(seq[rows, min(max(j, 0), 19)] * (1.0 if ok else 0.0))
        proj.append(np.full((n, 1), float(ok)))
    rise, fall = np.maximum(dw, 0.0), np.minimum(dw, 0.0)
    edge = np.column_stack([
        rise.max(1), fall.min(1), rise.sum(1), fall.sum(1),
        np.argmax(dw, axis=1), np.argmin(dw, axis=1),
        (dw > 0).mean(1), dw[:, -4:].mean(1), dw[:, -4:].max(1),
    ])
    return np.column_stack([
        w, dw, w.mean(1), w.std(1), w.min(1), w.max(1),
        w[:, -1] - w[:, -2], w[:, -4:].mean(1), w[:, -8:].mean(1),
        seq[:, -1], seq.mean(1), seq.std(1),
        seq[:, -4:].mean(1) - seq[:, :4].mean(1), edge, np.column_stack(proj),
    ])


class Encoder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()

        def blk(i, o, s=2):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.SiLU(inplace=True),
                nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.SiLU(inplace=True))

        self.net = nn.Sequential(blk(2, 24), blk(24, 48), blk(48, 96), blk(96, 128),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2),
                                 nn.Linear(128, dim))

    def forward(self, x):
        return self.net(x.contiguous(memory_format=torch.channels_last))


class Net(nn.Module):
    def __init__(self, dim=128, hid=160):
        super().__init__()
        self.enc = Encoder(dim)
        self.gru = nn.GRU(dim, hid, num_layers=1, batch_first=True, bidirectional=True)
        self.wind = nn.Sequential(nn.Linear(40, 128), nn.SiLU(inplace=True), nn.Linear(128, 128))
        self.head = nn.Sequential(nn.Linear(4 * hid + 128, 256), nn.SiLU(inplace=True),
                                  nn.Dropout(0.25), nn.Linear(256, 12))

    def forward(self, img, wind):
        b, t = img.shape[:2]
        e = self.enc(img.flatten(0, 1)).view(b, t, -1)
        g, _ = self.gru(e)
        g = torch.cat([g[:, -1], g.mean(1)], dim=-1)
        return self.head(torch.cat([g, self.wind(wind)], dim=-1))


class EncoderV3(nn.Module):
    def __init__(self, dim=192, wm=1.0):
        super().__init__()
        c1, c2, c3, c4 = (int(32 * wm), int(64 * wm), int(128 * wm), int(256 * wm))

        def blk(i, o, s=2):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.SiLU(inplace=True),
                nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.SiLU(inplace=True))

        self.net = nn.Sequential(blk(2, c1), blk(c1, c2), blk(c2, c3), blk(c3, c4),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2),
                                 nn.Linear(c4, dim))

    def forward(self, x):
        return self.net(x.contiguous(memory_format=torch.channels_last))


class NetV3(nn.Module):
    def __init__(self, dim=192, hid=224, wm=1.0):
        super().__init__()
        self.enc = EncoderV3(dim, wm)
        self.gru = nn.GRU(dim, hid, num_layers=1, batch_first=True, bidirectional=True)
        self.wind = nn.Sequential(nn.Linear(40, 128), nn.SiLU(inplace=True), nn.Linear(128, 128))
        self.head = nn.Sequential(nn.Linear(4 * hid + 128, 256), nn.SiLU(inplace=True),
                                  nn.Dropout(0.25), nn.Linear(256, 12))

    def forward(self, img, wind):
        b, t = img.shape[:2]
        e = self.enc(img.flatten(0, 1)).view(b, t, -1)
        g, _ = self.gru(e)
        g = torch.cat([g[:, -1], g.mean(1)], dim=-1)
        return self.head(torch.cat([g, self.wind(wind)], dim=-1))


class EncoderV4(nn.Module):
    """GAP 제거판 (`work/cnn4.py` Encoder 와 레이어·이름이 같아야 state_dict 가 맞는다)."""

    GRID = 6
    RED = 32
    SHAPE = {"g6": (6, 6), "l8": (1, 8), "l28": (2, 8)}

    def __init__(self, dim=192, wm=1.0, pool="g6"):
        super().__init__()
        c1, c2, c3, c4 = (int(32 * wm), int(64 * wm), int(128 * wm), int(256 * wm))
        self.pool = pool
        hh, ww = self.SHAPE[pool]

        def blk(i, o, s=2):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.SiLU(inplace=True),
                nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.SiLU(inplace=True))

        body = [blk(4, c1), blk(c1, c2), blk(c2, c3), blk(c3, c4), nn.AdaptiveAvgPool2d((hh, ww))]
        if pool == "g6":
            # 6×6 은 셀이 36개라 1×1 로 채널을 줄여야 파라미터가 안 터진다
            red = int(self.RED * wm)
            body += [nn.Conv2d(c4, red, 1, bias=False), nn.BatchNorm2d(red), nn.SiLU(inplace=True)]
            flat = red * hh * ww
        else:
            flat = c4 * hh * ww
        body += [nn.Flatten(), nn.Dropout(0.2), nn.Linear(flat, dim)]
        self.net = nn.Sequential(*body)
        self._coord = None

    def coords(self, x):
        if self._coord is None or self._coord.shape[-1] != x.shape[-1]:
            px = x.shape[-1]
            g = torch.linspace(-1, 1, px, device=x.device, dtype=x.dtype)
            yy, xx = torch.meshgrid(g, g, indexing="ij")
            self._coord = torch.stack([xx, yy])[None]
        return self._coord.expand(x.shape[0], -1, -1, -1)

    def forward(self, x):
        x = torch.cat([x, self.coords(x).to(x.dtype)], dim=1)
        return self.net(x.contiguous(memory_format=torch.channels_last))


class NetV4(nn.Module):
    def __init__(self, dim=192, hid=224, wm=1.0, pool="g6"):
        super().__init__()
        self.enc = EncoderV4(dim, wm, pool)
        self.gru = nn.GRU(dim, hid, num_layers=1, batch_first=True, bidirectional=True)
        self.wind = nn.Sequential(nn.Linear(40, 128), nn.SiLU(inplace=True), nn.Linear(128, 128))
        self.head = nn.Sequential(nn.Linear(4 * hid + 128, 256), nn.SiLU(inplace=True),
                                  nn.Dropout(0.25), nn.Linear(256, 12))

    def forward(self, img, wind):
        b, t = img.shape[:2]
        e = self.enc(img.flatten(0, 1)).view(b, t, -1)
        g, _ = self.gru(e)
        g = torch.cat([g[:, -1], g.mean(1)], dim=-1)
        return self.head(torch.cat([g, self.wind(wind)], dim=-1))


class EncoderV6(nn.Module):
    """2-스트림(영상 2ch + 광류 2ch). work/cnn6.py Encoder 와 레이어·이름이 같다."""

    def __init__(self, dim=192, wm=1.0):
        super().__init__()
        c1, c2, c3, c4 = (int(32 * wm), int(64 * wm), int(128 * wm), int(256 * wm))

        def blk(i, o, s=2):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.SiLU(inplace=True),
                nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.SiLU(inplace=True))

        self.net = nn.Sequential(blk(4, c1), blk(c1, c2), blk(c2, c3), blk(c3, c4),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2),
                                 nn.Linear(c4, dim))

    def forward(self, x):
        return self.net(x.contiguous(memory_format=torch.channels_last))


class NetV6(nn.Module):
    def __init__(self, dim=192, hid=224, wm=1.0):
        super().__init__()
        self.enc = EncoderV6(dim, wm)
        self.hid = hid
        self.gru = nn.GRU(dim, hid, num_layers=1, batch_first=True, bidirectional=True)
        self.wind = nn.Sequential(nn.Linear(40, 128), nn.SiLU(inplace=True), nn.Linear(128, 128))
        self.head = nn.Sequential(nn.Linear(4 * hid + 128, 256), nn.SiLU(inplace=True),
                                  nn.Dropout(0.25), nn.Linear(256, 12))

    def forward(self, img, wind):
        b, t = img.shape[:2]
        e = self.enc(img.flatten(0, 1)).view(b, t, -1)
        g, _ = self.gru(e)
        g = torch.cat([g[:, -1], g.mean(1)], dim=-1)
        return self.head(torch.cat([g, self.wind(wind)], dim=-1))


class NetV7(NetV6):
    def __init__(self, dim=192, hid=224, wm=1.0, sdim=839):
        super().__init__(dim, hid, wm)
        self.wind = nn.Sequential(nn.Linear(40 + sdim, 128), nn.SiLU(inplace=True), nn.Linear(128, 128))


class NetV9(NetV6):
    pass


def wind_cnn(inputs):
    w = inputs[W].to_numpy(np.float32)
    d = np.diff(w, axis=1, prepend=w[:, :1])
    return np.concatenate([(w - 400.0) / 100.0, d / 50.0], axis=1)


def bag_predict(entry, X):
    """horizon 슬롯이 모델 리스트면 시드백이라 평균낸다 (omnibag·eswfbag). 단일 모델은 그대로."""
    ms = entry if isinstance(entry, (list, tuple)) else [entry]
    return np.mean([m.predict(X) for m in ms], axis=0)


def disk_mask(px, dev):
    """cnn3.py 의 med 정규화와 같은 원판 마스크 (중심 반지름 0.47·px)."""
    yy, xx = torch.meshgrid(torch.arange(px, device=dev), torch.arange(px, device=dev), indexing="ij")
    return ((yy - px / 2 + 0.5) ** 2 + (xx - px / 2 + 0.5) ** 2) <= (0.47 * px) ** 2


def stony_grid(px, dev):
    a = torch.linspace(-72.0, 72.0, px, device=dev) * math.pi / 180.0
    lat, lon = torch.meshgrid(a, a, indexing="ij")
    gx = 0.94 * torch.cos(lat) * torch.sin(lon)
    gy = 0.94 * torch.sin(lat)
    return torch.stack([gx, gy], dim=-1)[None]


FLOW_ARCHES = {"v6", "v7", "v9"}


def cnn_predict(state, inputs, split, mean, std, arch="v1", px=PX96, norm="z", wm=1.0,
                pool="g6", fstd=None, smu=None, ssd=None):
    idx, arr = build_cache(split, inputs, px, cache_for(px))
    slots = np.stack([[idx[v] for v in inputs[col]] for col in IMG], axis=1)
    farr = pslots = None
    if arch in FLOW_ARCHES:
        seen, farr = flow_pairs(split, inputs, idx, arr, px)
        names = inputs[IMG].to_numpy()
        pslots = np.stack([[seen[(row[t], row[t + 1])] for t in range(19)] for row in names])
        slots = slots[:, 1:]                                          # 영상 t=1..19
        assert slots.shape == pslots.shape == (len(inputs), 19)
    imgs = torch.from_numpy(np.asarray(arr)).to(DEV)
    slots = torch.from_numpy(slots).long().to(DEV)
    wf = torch.from_numpy(wind_cnn(inputs)).to(DEV)
    if arch in FLOW_ARCHES:
        if fstd is None:
            raise KeyError(f"{arch} CNN artifact missing fstd")
        scale = float(fstd)
        if not np.isfinite(scale) or scale == 0.0:
            raise ValueError(f"{arch} CNN artifact has invalid fstd: {fstd!r}")
    else:
        scale = 1.0
    if arch == "v7":
        if smu is None or ssd is None:
            raise KeyError("v7 CNN artifact missing smu/ssd")
        s = sample_feats(split, inputs).astype(np.float32)
        smu, ssd = np.asarray(smu, np.float32), np.asarray(ssd, np.float32)
        if (s.shape[1:] != (839,) or smu.shape != (839,) or ssd.shape != (839,)
                or not np.isfinite(smu).all() or not np.isfinite(ssd).all() or np.any(ssd == 0)):
            raise ValueError(f"v7 scalar stats shape mismatch: {s.shape}, {smu.shape}, {ssd.shape}")
        scalars = torch.from_numpy(np.clip((s - smu) / ssd, -8, 8).astype(np.float32)).to(DEV)
        wf = torch.cat([wf, scalars], dim=1)
    last = torch.from_numpy(inputs[W[-1]].to_numpy(np.float32)).to(DEV)
    if arch == "v4":
        model = NetV4(int(192 * wm), int(224 * wm), wm, pool)
    elif arch == "v7":
        model = NetV7(int(192 * wm), int(224 * wm), wm)
    elif arch == "v9":
        model = NetV9(int(192 * wm), int(224 * wm), wm)
    elif arch == "v6":
        model = NetV6(int(192 * wm), int(224 * wm), wm)
    elif arch == "v3":
        model = NetV3(int(192 * wm), int(224 * wm), wm)
    else:
        model = Net()
    model = model.to(DEV).to(memory_format=torch.channels_last)
    model.load_state_dict(state)
    model.eval()
    if norm == "med" and px not in _DISK:
        _DISK[px] = disk_mask(px, DEV)
    grid = stony_grid(px, DEV) if arch == "v9" else None
    out = []
    for i in range(0, len(inputs), 96):
        rows = torch.arange(i, min(i + 96, len(inputs)), device=DEV)
        raw = imgs[slots[rows]].float()
        if norm == "med":
            # 절대 밝기 confound 를 이미지별 disk-median 으로 나눈다 (cnn3.py NORM=med 와 동일)
            med = raw[..., _DISK[px]].median(-1).values[..., None, None]
            raw = raw / med.clamp(min=1.0)
        im = (raw - mean) / std
        if arch in FLOW_ARCHES:
            ii = rows.detach().cpu().numpy()
            fl = torch.from_numpy(np.asarray(farr[pslots[ii]], np.float32)).to(DEV) / scale
            im = torch.cat([im, fl], dim=2)                           # (b,19,4,PX,PX)
        if arch == "v9":
            b, t = im.shape[:2]
            im = torch.nn.functional.grid_sample(
                im.flatten(0, 1), grid.expand(b * t, -1, -1, -1), mode="bilinear",
                padding_mode="zeros", align_corners=False).view(b, t, *im.shape[2:])
        with torch.no_grad():
            if arch in FLOW_ARCHES:
                fim = torch.flip(im, dims=[-2])
                fim = torch.cat([fim[:, :, :3], -fim[:, :, 3:]], dim=2)   # fy 부호
                if DEV.type == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out.append((model(im, wf[rows]).float()
                                    + model(fim, wf[rows]).float()) / 2
                                   + last[rows, None])
                else:
                    out.append((model(im, wf[rows]).float()
                                + model(fim, wf[rows]).float()) / 2
                               + last[rows, None])
            elif arch == "v3":
                if DEV.type == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out.append((model(im, wf[rows]).float()
                                    + model(torch.flip(im, dims=[-2]), wf[rows]).float()) / 2
                                   + last[rows, None])
                else:
                    out.append((model(im, wf[rows]).float()
                                + model(torch.flip(im, dims=[-2]), wf[rows]).float()) / 2
                               + last[rows, None])
            elif DEV.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out.append(model(im, wf[rows]).float() + last[rows, None])
            else:
                out.append(model(im, wf[rows]).float() + last[rows, None])
    return torch.cat(out).cpu().numpy()


# %% [markdown]
# ## 3. 학습 (MODE == "train") 또는 모델 적재 (MODE == "infer")
#
# horizon 12개를 각각 따로 적합한다. **train split 9,607행만** 쓴다.

# %%
test_inputs = pd.read_csv(ROOT / "test/inputs.csv")

if MODE == "train":
    train_inputs = pd.read_csv(ROOT / "train/inputs.csv")
    train_targets = pd.read_csv(ROOT / "train/targets.csv")[T].to_numpy(np.float64)
    X1 = sample_feats_v1("train", train_inputs)
    X2 = sample_feats_v2("train", train_inputs)
    last_tr = train_inputs[W[-1]].to_numpy(np.float64)
    aseq_tr = align_seq("train", train_inputs)
    w_tr = train_inputs[W].to_numpy(np.float64)
    seq_tr = eswf_seq("train", train_inputs)
    print("feature dims", X1.shape, X2.shape, seq_tr.shape, flush=True)

    ridge_v1 = RidgeCV(alphas=np.logspace(-3, 4, 30)).fit(X1, train_targets)
    gbdt_v1, gbdt_v2, gbdt_eswf = [], [], []
    for h in range(12):
        gbdt_v1.append(HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.06, max_leaf_nodes=31,
            early_stopping=True, validation_fraction=0.12,
            random_state=SEED).fit(X1, train_targets[:, h]))
        Xo = np.column_stack([X2, t4_block(aseq_tr, h), align_block(aseq_tr, w_tr, h)])
        gbdt_v2.append(HistGradientBoostingRegressor(
            max_iter=700, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.12, n_iter_no_change=30,
            random_state=SEED).fit(Xo, train_targets[:, h] - last_tr))
        Xe = eswf_feats(seq_tr, train_inputs, 6.0 * (h + 1))
        gbdt_eswf.append(HistGradientBoostingRegressor(
            max_iter=700, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.12, n_iter_no_change=30,
            random_state=SEED).fit(Xe, train_targets[:, h]))
        print(f"  h{h:02d} 학습 완료", flush=True)

    cnn = {}
    idx, arr = build_cache("train", train_inputs, PX96, CACHE96)
    cnn["mean"] = float(np.asarray(arr, np.float32).mean())
    cnn["std"] = float(np.asarray(arr, np.float32).std())
    for tag in ("cnn96b", "cnn96w3"):
        p = pathlib.Path("/home/jovyan/work") / f"{tag}.pth"
        assert p.exists(), f"{p} 없음 — 먼저 work/cnn.py 로 학습해라"
        cnn[tag] = torch.load(p, map_location="cpu", weights_only=True)

    torch.save({"ridge_v1": ridge_v1, "gbdt_v1": gbdt_v1, "gbdt_v2": gbdt_v2,
                "gbdt_eswf": gbdt_eswf, "cnn": cnn,
                "weights": WEIGHTS, "clip": CLIP, "seed": SEED, "px": PX,
                "version": "v1-omni-eswf-cnn96bw3"}, HERE / "model.pth")
    print("saved model.pth")

blob = torch.load(HERE / "model.pth", map_location="cpu", weights_only=False)
if blob.get("compressed"):
    # 제출 파일 용량 상한 때문에 sklearn 파트를 zlib 로 담았다 (work/shrink.py). 값은 그대로다.
    import pickle
    import zlib
    blob = {k: (pickle.loads(zlib.decompress(v)) if isinstance(v, (bytes, bytearray)) else v)
            for k, v in blob.items()}
assert len(blob["gbdt_v1"]) == 12 and len(blob["gbdt_eswf"]) == 12
CLIP = blob.get("clip", CLIP)
# pack.py 는 img_ridge/img_gbdt/v2_gbdt 키를 쓰기도 한다.
_w = dict(blob.get("weights") or WEIGHTS)
_alias = {"img_ridge": "ridge_v1", "img_gbdt": "gbdt_v1", "v2_gbdt": "gbdt_v2",
          "omni": "gbdt_v2", "omnibag": "gbdt_v2"}
WEIGHTS = {_alias.get(k, k): float(v) for k, v in _w.items()}
CNN_TAGS = tuple(k for k in WEIGHTS if k.startswith("cnn"))
if not CNN_TAGS:
    CNN_TAGS = ("cnn96b", "cnn96w3")
    WEIGHTS = {**OLD_WEIGHTS, **WEIGHTS}
for _k in ("ridge_v1", "gbdt_v1", "gbdt_v2", "eswf"):
    assert _k in WEIGHTS, f"weights 에 {_k} 없음: {sorted(WEIGHTS)}"
assert all(t in blob["cnn"] for t in CNN_TAGS), (CNN_TAGS, sorted(blob["cnn"]))


def predict(split, inputs):
    f1, f2 = sample_feats_v1(split, inputs), sample_feats_v2(split, inputs)
    last = inputs[W[-1]].to_numpy(np.float64)[:, None]
    aseq = align_seq(split, inputs)
    w = inputs[W].to_numpy(np.float64)
    seq = eswf_seq(split, inputs)
    pe = np.column_stack([
        bag_predict(blob["gbdt_eswf"][h], eswf_feats(seq, inputs, 6.0 * (h + 1)))
        for h in range(12)])
    # gbdt_v2 슬롯 = omni (t−4d + j* + 잔차)
    pj = last + np.column_stack([
        bag_predict(blob["gbdt_v2"][h],
                    np.column_stack([f2, t4_block(aseq, h), align_block(aseq, w, h)]))
        for h in range(12)])
    p = (WEIGHTS["ridge_v1"] * blob["ridge_v1"].predict(f1)
         + WEIGHTS["gbdt_v1"] * np.column_stack([bag_predict(m, f1) for m in blob["gbdt_v1"]])
         + WEIGHTS["gbdt_v2"] * pj
         + WEIGHTS["eswf"] * pe)
    cnn = blob["cnn"]
    for tag in CNN_TAGS:
        entry = cnn[tag]
        if isinstance(entry, dict) and "state" in entry:
            state, arch, px = entry["state"], entry.get("arch", "v1"), entry.get("px", PX96)
            mean, std = entry.get("mean", cnn["mean"]), entry.get("std", cnn["std"])
            norm, wm = entry.get("norm", "z"), float(entry.get("wm", 1.0))
            pool = entry.get("pool", "g6")
            stats = entry.get("stats", {})
            stats = stats if isinstance(stats, dict) else {}
            fstd = entry.get("fstd", stats.get("fstd"))
            smu = entry.get("smu", stats.get("smu"))
            ssd = entry.get("ssd", stats.get("ssd"))
            if arch in FLOW_ARCHES and fstd is None:
                raise KeyError(f"CNN tag {tag} artifact missing fstd")
            if arch == "v7" and (smu is None or ssd is None):
                raise KeyError(f"CNN tag {tag} artifact missing smu/ssd")
        else:
            state, arch, px = entry, "v1", PX96
            mean, std = cnn["mean"], cnn["std"]
            norm, wm, pool, fstd, smu, ssd = "z", 1.0, "g6", None, None, None
        p = p + WEIGHTS[tag] * cnn_predict(state, inputs, split, mean, std, arch, px, norm, wm, pool, fstd, smu, ssd)
    return np.clip(p, *CLIP)


# %% [markdown]
# ## 4. validation 점수 (참고용 — 학습에 쓰지 않는다)

# %%
val_inputs = pd.read_csv(ROOT / "validation/inputs.csv")
val_targets = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
val_pred = predict("validation", val_inputs)
persistence = np.repeat(val_inputs[W[-1]].to_numpy(np.float64)[:, None], 12, axis=1)
print(f"validation RMSE  {rmse(val_targets, val_pred):8.3f}")
print(f"persistence RMSE {rmse(val_targets, persistence):8.3f}")
print("horizon별:", " ".join(f"{rmse(val_targets[:, h], val_pred[:, h]):.0f}" for h in range(12)))

# %% [markdown]
# ## 5. test 추론 → submission.csv

# %%
test_pred = predict("test", test_inputs)
submission = pd.DataFrame(test_pred, columns=T)
submission.insert(0, "sample_id", test_inputs.sample_id.tolist())

test_ids = pd.read_csv(ROOT / "test/test_ids.csv").sample_id.tolist()
assert submission.columns.tolist() == ["sample_id"] + T
assert len(submission) == 3868
assert submission.sample_id.tolist() == test_ids
assert submission.sample_id.is_unique
assert np.isfinite(submission[T].to_numpy()).all()
submission.to_csv(HERE / "submission.csv", index=False)
print("saved submission.csv", submission.shape)
submission.head()
