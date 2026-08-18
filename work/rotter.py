"""E013 — Rotter/Vršnak 선형: V ≈ a + b·A_CM. DOMAIN §1.2.

E010 은 같은 면적을 GBDT 에 넣어 평균회귀한다. 문헌의 경험식은 선형이라
A 가 큰 도착 국면에서 더 공격적이다. 전역 요약을 대체하지 않는다.

사용: python work/rotter.py
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jovyan/work")
import eswf  # noqa: E402

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
OUT = pathlib.Path("/home/jovyan/work/preds")
T = [f"target_{i:02d}" for i in range(12)]
TAG = "rotter"


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def cm_area(seq):
    """eswf per-image 벡터에서 자오선 면적 대리를 고른다. 마지막 프레임 전역 CH."""
    # seq: (N,20,F). 마지막 시점 평균을 A 로 쓴다.
    return seq[:, -1].mean(1)


def main():
    tr_x = pd.read_csv(ROOT / "train/inputs.csv")
    va_x = pd.read_csv(ROOT / "validation/inputs.csv")
    te_x = pd.read_csv(ROOT / "test/inputs.csv")
    tr_y = pd.read_csv(ROOT / "train/targets.csv")[T].to_numpy(np.float64)
    va_y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
    eswf.SEQ = {s: eswf.sequences(s, x) for s, x in
                (("train", tr_x), ("validation", va_x), ("test", te_x))}
    Atr, Ava, Ate = cm_area(eswf.SEQ["train"]), cm_area(eswf.SEQ["validation"]), cm_area(eswf.SEQ["test"])
    gv, gt = np.zeros_like(va_y), np.zeros((len(te_x), 12))
    for h in range(12):
        coef = np.polyfit(Atr, tr_y[:, h], 1)
        gv[:, h] = np.polyval(coef, Ava)
        gt[:, h] = np.polyval(coef, Ate)
        print(f"  h{h:02d} a={coef[0]:.1f} b={coef[1]:.1f} val={rmse(va_y[:, h], gv[:, h]):.1f}",
              flush=True)
    np.save(OUT / f"val_{TAG}.npy", gv)
    np.save(OUT / f"test_{TAG}.npy", gt)
    (OUT / f"{TAG}.json").write_text(json.dumps({
        "val_rmse": round(rmse(va_y, gv), 3),
        "per_h": [round(rmse(va_y[:, h], gv[:, h]), 1) for h in range(12)],
    }, indent=2))
    print(f"{TAG} val_rmse={rmse(va_y, gv):.3f}", flush=True)


if __name__ == "__main__":
    main()
