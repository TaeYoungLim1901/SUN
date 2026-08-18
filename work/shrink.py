"""제출용 model.pth 를 줄인다. 제출 실패 원인이 용량이라는 운영진 답변에 대응.

두 손잡이만 쓴다.
  1. **zlib** — sklearn 파트를 압축해 담는다. 정확도 손실 0.
  2. **트리 절단** — HistGBDT 의 `_predictors` 를 앞 k 개만 남긴다. 부스팅 라운드를 줄이는 것과 같다.
     손실은 validation 으로 잰다 (규정상 validation 은 모델 선택에 쓸 수 있다, RULES.md §5-1).

    python work/shrink.py              # 측정만 (아무것도 안 쓴다)
    python work/shrink.py --write K    # gbdt 를 k=K 로 절단 + zlib 압축해 submission/model.pth 재작성

`--write` 는 원본을 `submission/model.pth.bak` 로 남긴다.
"""
import argparse
import copy
import pathlib
import pickle
import sys
import zlib

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/jovyan/work")

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
SUB = pathlib.Path("/home/jovyan/submission")
T = [f"target_{i:02d}" for i in range(12)]
GBDT = ("gbdt_v1", "gbdt_v2", "gbdt_eswf")
KS = (600, 400, 300, 200, 150, 100)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def blob_size(v):
    return len(pickle.dumps(v, protocol=4))


def truncate(models, k):
    """부스팅 라운드를 앞 k 개로 자른 사본. HistGBDT 는 `_predictors` 리스트가 라운드다."""
    out = []
    for m in models:
        c = copy.deepcopy(m)
        c._predictors = c._predictors[:k]
        out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", type=int, default=0, help="이 라운드 수로 절단해 model.pth 재작성")
    a = ap.parse_args()

    blob = torch.load(SUB / "model.pth", map_location="cpu", weights_only=False)
    print("현재", (SUB / "model.pth").stat().st_size / 1e6, "MB")
    for k in blob:
        raw = blob_size(blob[k])
        print(f"  {k:12} {raw / 1e6:7.1f} MB  zlib {len(zlib.compress(pickle.dumps(blob[k], 4), 6)) / 1e6:6.1f} MB")
    print("라운드 수", {g: [len(m._predictors) for m in blob[g]] for g in GBDT}, flush=True)

    # validation 피처를 만들어 절단 손실을 잰다
    va_x = pd.read_csv(ROOT / "validation/inputs.csv")
    va_y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
    import eswf
    from feats import sample_feats as feats_v2
    from imgfeat import sample_feats as feats_v1
    F = {"gbdt_v1": feats_v1("validation", va_x), "gbdt_v2": feats_v2("validation", va_x)}
    eswf.SEQ = {"validation": eswf.sequences("validation", va_x)}   # eswf.build 가 이걸 본다
    print("피처 준비 완료", {k: v.shape for k, v in F.items()}, flush=True)

    def predict(models, key, k=None):
        ms = truncate(models, k) if k else models
        if key != "gbdt_eswf":
            return np.column_stack([m.predict(F[key]) for m in ms])
        return np.column_stack([
            ms[h].predict(eswf.build("validation", va_x, 6.0 * (h + 1)))
            for h in range(12)])

    for g in GBDT:
        base = predict(blob[g], g)
        line = [f"{g:10} full {rmse(va_y, base):7.3f} ({blob_size(blob[g]) / 1e6:.0f}MB)"]
        for k in KS:
            t = truncate(blob[g], k)
            line.append(f"k={k}: {rmse(va_y, predict(t, g)):.3f}/{blob_size(t) / 1e6:.0f}MB")
        print("  ".join(line), flush=True)

    if a.write:
        k = a.write
        new = dict(blob)
        for g in GBDT:
            new[g] = zlib.compress(pickle.dumps(truncate(blob[g], k), 4), 6)
        new["ridge_v1"] = zlib.compress(pickle.dumps(blob["ridge_v1"], 4), 6)
        new["compressed"] = True
        bak = SUB / "model.pth.bak"
        if not bak.exists():
            bak.write_bytes((SUB / "model.pth").read_bytes())
            print("원본 백업", bak)
        torch.save(new, SUB / "model.pth")
        print("새 크기", (SUB / "model.pth").stat().st_size / 1e6, "MB")


if __name__ == "__main__":
    main()
