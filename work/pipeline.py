# %% [markdown]
# # 태양풍 속도 예측 — 제출 파이프라인
#
# 이 노트북 **하나만으로** 학습·추론·제출파일 생성이 전부 된다. 외부 .py import 없음.
#
# - `MODE = "infer"` (기본): `model.pth` 를 불러 test 를 추론하고 `submission.csv` 를 쓴다.
# - `MODE = "train"`: train split 으로 처음부터 학습해 `model.pth` 를 새로 만든다.
#
# 규정 준수:
# - pretrained weight 미사용. 모델은 전부 이 데이터로 처음부터 적합한다.
# - 외부 데이터 미사용. 대회가 준 `public_dataset` 만 읽는다.
# - **validation split 은 학습에 쓰지 않는다.** 성능 확인·모델 선택에만 쓴다.
# - test 의 target 을 추정해 학습에 넣는 경로 없음 (pseudo-labeling 없음).

# %%
import json
import pathlib
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.ensemble import HistGradientBoostingRegressor

MODE = "infer"          # "infer" | "train"
SEED = 777
PX = 64                 # 캐시 해상도. 원본 PNG 는 512×512.
AU_KM = 1.495978707e8   # 전파지연 계산용


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
# 샘플들이 같은 PNG 를 공유한다 (train 샘플 9,607개가 쓰는 유니크 이미지는 10,139장뿐).
# 그래서 **유니크 파일당 한 번만** 읽어 64×64 uint8 로 캐시한다. 반복 실행에서는 캐시를 재사용한다.

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
# ## 2. 이미지 → 스칼라
#
# 코로나 홀은 EUV 영상에서 **어둡게** 보이고, 거기서 고속 태양풍이 나온다.
# 그래서 "얼마나 어두운 영역이 어디에 얼마나 있나" 를 숫자로 뽑는다.
#
# - 임계는 **이미지별 디스크 중앙값 대비 상대값**이다. 절대 밝기 드리프트에 둔감하게 하려는 것.
# - 암부 centroid·퍼짐 = 코로나 홀의 위치. 적도대 홀이 지구를 때린다.
# - 적도/극 분리, 서쪽 경도대(자전으로 곧 지구를 향하는 쪽), 193/211 비.

# %%
def per_image_scalars(split, inputs):
    idx, a = build_cache(split, inputs)
    a = a.astype(np.float32)
    n, c, h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    disk = ((yy - h / 2 + .5) ** 2 + (xx - w / 2 + .5) ** 2) <= (0.47 * h) ** 2
    dm = disk.ravel()
    flat = a.reshape(n, c, -1)[:, :, dm]
    med = np.median(flat, axis=2, keepdims=True)
    out = [flat.mean(2), flat.std(2), med[:, :, 0],
           np.percentile(flat, 5, axis=2), np.percentile(flat, 95, axis=2)]

    masks = {}
    for f in (0.35, 0.5, 0.65, 0.8):
        masks[f] = flat < med * f
        out.append(masks[f].mean(2))

    ys, xs = yy.ravel()[dm], xx.ravel()[dm]
    for f in (0.5, 0.65):
        m = masks[f].astype(np.float32)
        s = m.sum(2) + 1e-6
        cx = (m * xs).sum(2) / s
        cy = (m * ys).sum(2) / s
        out += [cx / w, cy / h,
                np.sqrt((m * (xs - cx[..., None]) ** 2).sum(2) / s) / w,
                np.sqrt((m * (ys - cy[..., None]) ** 2).sum(2) / s) / h]

    lat_eq = (np.abs(yy - h / 2 + .5) < 0.20 * h) & disk
    for band in (lat_eq, (~lat_eq) & disk):
        b = a.reshape(n, c, -1)[:, :, band.ravel()]
        out += [b.mean(2), (b < med * 0.5).mean(2)]
    for lo, hi in ((0.55, 0.78), (0.78, 1.0)):
        s_ = a[:, :, :, int(lo * w):int(hi * w)].reshape(n, c, -1)
        out += [s_.mean(2), (s_ < med * 0.5).mean(2)]

    f = np.concatenate([o.reshape(n, -1) for o in out], axis=1)
    ratio = (f[:, 0::2] + 1e-3) / (f[:, 1::2] + 1e-3)
    return idx, np.concatenate([f, ratio], axis=1).astype(np.float32)


def sequences(split, inputs):
    idx, per = per_image_scalars(split, inputs)
    seq = np.stack([per[[idx[v] for v in inputs[col]]] for col in IMG], axis=1)
    assert seq.shape[:2] == (len(inputs), 20)
    return seq


# %% [markdown]
# ## 3. horizon 별 '발사시각 정합'
#
# t+h 에 지구에 닿는 태양풍은 h 시간 전이 아니라 **전파지연 τ 만큼 전에** 태양을 떠났다.
# τ = 1AU / v 이고 v=400 km/s 면 약 104시간이다. 그러니 horizon 마다 봐야 할 프레임이 다르다.
# 마지막 프레임과 전체 평균만 주면 이 도착시각 구조가 뭉개진다.
#
# 프레임 j 의 시각은 t − (19−j)·6h. 발사시각 = t + h − τ. 여기서 j\* 를 얻어 ±2 프레임을 넣는다.
# 창 밖으로 나가면 0 으로 채우고 **마스크 플래그**를 같이 준다.

# %%
def build_features(seq, inputs, horizon_hours):
    n = len(inputs)
    v = np.clip(inputs[W[-1]].to_numpy(np.float64), 250.0, 900.0)
    tau_h = AU_KM / v / 3600.0
    jstar = 19.0 - (tau_h - horizon_hours) / 6.0
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
        tau_h, jstar, np.column_stack(mask),
        np.concatenate(win, axis=1),
        seq[:, -1], seq.mean(1), seq.std(1),
        seq[:, -4:].mean(1) - seq[:, :4].mean(1),
    ])


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)))


# %% [markdown]
# ## 4. 학습 (MODE == "train") 또는 모델 적재 (MODE == "infer")
#
# horizon 12개를 각각 따로 적합한다. horizon 마다 입력 프레임 창이 다르기 때문이다.
# **train split 9,607행만** 쓴다. validation 은 아래 5절에서 점수를 재는 데만 쓴다.

# %%
test_inputs = pd.read_csv(ROOT / "test/inputs.csv")
test_seq = sequences("test", test_inputs)

if MODE == "train":
    train_inputs = pd.read_csv(ROOT / "train/inputs.csv")
    train_targets = pd.read_csv(ROOT / "train/targets.csv")[T].to_numpy(np.float64)
    train_seq = sequences("train", train_inputs)
    models = []
    for h in range(12):
        hh = 6.0 * (h + 1)
        X = build_features(train_seq, train_inputs, hh)
        m = HistGradientBoostingRegressor(
            max_iter=700, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.12, n_iter_no_change=30,
            random_state=SEED).fit(X, train_targets[:, h])
        models.append(m)
        print(f"  h{h:02d} ({hh:.0f}h) 학습 완료 · 트리 {m.n_iter_}", flush=True)
    torch.save({"models": models, "seed": SEED, "px": PX,
                "clip": (200.0, 900.0), "version": "v3-gbdt"}, HERE / "model.pth")
    print("saved model.pth")

blob = torch.load(HERE / "model.pth", map_location="cpu", weights_only=False)
models, CLIP = blob["models"], blob["clip"]
assert len(models) == 12 and blob["px"] == PX

# %% [markdown]
# ## 5. validation 점수 (참고용 — 학습에 쓰지 않는다)

# %%
val_inputs = pd.read_csv(ROOT / "validation/inputs.csv")
val_targets = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
val_seq = sequences("validation", val_inputs)
val_pred = np.column_stack([
    models[h].predict(build_features(val_seq, val_inputs, 6.0 * (h + 1))) for h in range(12)])
val_pred = np.clip(val_pred, *CLIP)
persistence = np.repeat(val_inputs[W[-1]].to_numpy(np.float64)[:, None], 12, axis=1)
print(f"validation RMSE        {rmse(val_targets, val_pred):8.3f}")
print(f"persistence RMSE       {rmse(val_targets, persistence):8.3f}")
print("horizon별:", " ".join(f"{rmse(val_targets[:, h], val_pred[:, h]):.0f}" for h in range(12)))

# %% [markdown]
# ## 6. test 추론 → submission.csv

# %%
test_pred = np.clip(np.column_stack([
    models[h].predict(build_features(test_seq, test_inputs, 6.0 * (h + 1))) for h in range(12)]),
    *CLIP)
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
