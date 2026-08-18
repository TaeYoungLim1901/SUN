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
# | 영상 스칼라 + GBDT (아래 구성) | **64.8** |
#
# 규정 준수: pretrained weight 미사용 · 외부 데이터 미사용 ·
# **validation 은 학습에 쓰지 않는다**(점수 확인 전용) · test target 추정 없음.

# %%
import json
import os
import pathlib
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV

MODE = os.environ.get("SW_MODE", "infer")   # "infer" (기본) | "train"
SEED = 777
PX = 64                 # 캐시 해상도. 원본 PNG 는 512×512.
CLIP = (200.0, 900.0)   # 지구 근처 태양풍의 물리적 범위 (RULES.md §6-4)
DARK_V1, GRID_V1 = (20, 40, 60), 2
DARK_V2, GRID_V2 = (15, 25, 40, 60, 80), 4
# 결합 가중치. kit.hill_climb 이 validation 에서 고른 값을 고정해 박아둔 것이다.
WEIGHTS = {"ridge_v1": 0.25, "gbdt_v1": 0.25, "gbdt_v2": 0.50}


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
CACHE = pathlib.Path.home() / "sw_cache" / f"{PX}px"
CACHE.mkdir(parents=True, exist_ok=True)
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
def build_cache(split, inputs):
    path = CACHE / f"{split}_images.npy"
    names = sorted(pd.unique(inputs[IMG].to_numpy().ravel()).tolist())
    if not path.exists():
        tmp = path.with_suffix(".partial")
        arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8,
                                        shape=(len(names), 2, PX, PX))
        t0 = time.perf_counter()
        for i, name in enumerate(names):
            for c, ch in enumerate(("193", "211")):
                with Image.open(ROOT / split / ch / name) as im:
                    arr[i, c] = np.asarray(
                        im.convert("L").resize((PX, PX), Image.BILINEAR), dtype=np.uint8)
            if (i + 1) % 2000 == 0:
                print(f"  {split} {i+1}/{len(names)} ({time.perf_counter()-t0:.0f}s)", flush=True)
        arr.flush()
        del arr
        tmp.rename(path)
    a = np.load(path)
    assert a.shape == (len(names), 2, PX, PX)
    return {v: i for i, v in enumerate(names)}, a


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


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)))


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
    print("feature dims", X1.shape, X2.shape, flush=True)

    ridge_v1 = RidgeCV(alphas=np.logspace(-3, 4, 30)).fit(X1, train_targets)
    gbdt_v1, gbdt_v2 = [], []
    for h in range(12):
        gbdt_v1.append(HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.06, max_leaf_nodes=31,
            early_stopping=True, validation_fraction=0.12,
            random_state=SEED).fit(X1, train_targets[:, h]))
        gbdt_v2.append(HistGradientBoostingRegressor(
            max_iter=700, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.12, n_iter_no_change=30,
            random_state=SEED).fit(X2, train_targets[:, h]))
        print(f"  h{h:02d} 학습 완료", flush=True)

    torch.save({"ridge_v1": ridge_v1, "gbdt_v1": gbdt_v1, "gbdt_v2": gbdt_v2,
                "weights": WEIGHTS, "clip": CLIP, "seed": SEED, "px": PX,
                "version": "gbdt-v1v2-blend"}, HERE / "model.pth")
    print("saved model.pth")

blob = torch.load(HERE / "model.pth", map_location="cpu", weights_only=False)
assert len(blob["gbdt_v1"]) == 12 and len(blob["gbdt_v2"]) == 12 and blob["px"] == PX
WEIGHTS, CLIP = blob["weights"], blob["clip"]


def predict(split, inputs):
    """세 모델을 각각 돌려 고정 가중치로 섞고 물리 범위로 자른다."""
    f1, f2 = sample_feats_v1(split, inputs), sample_feats_v2(split, inputs)
    p = (WEIGHTS["ridge_v1"] * blob["ridge_v1"].predict(f1)
         + WEIGHTS["gbdt_v1"] * np.column_stack([m.predict(f1) for m in blob["gbdt_v1"]])
         + WEIGHTS["gbdt_v2"] * np.column_stack([m.predict(f2) for m in blob["gbdt_v2"]]))
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
