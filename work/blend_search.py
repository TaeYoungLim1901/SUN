"""CNN 시드백 결합 가중을 val npy 로만 고른다. 학습 없음, 초 단위.

가족 안에서는 **균등 평균**(val 에 안 맞춘다). 가족 몫 두 개만 성긴 격자로 고른다:
  w_cnn        = CNN 전체 몫
  med/z128/z96 = CNN 안의 배분 (성긴 심플렉스)
산출: work/blend.json — pack_seedbag.py 가 그대로 먹는다.
"""
import itertools
import json
import pathlib

import numpy as np
import pandas as pd

P = pathlib.Path("/home/jovyan/work/preds")
Y = pd.read_csv("/home/jovyan/public_dataset/competition_dataset_6h/validation/targets.csv")[
    [f"target_{i:02d}" for i in range(12)]].to_numpy(np.float64)
rmse = lambda p: float(np.sqrt(np.mean((Y - p) ** 2)))
L = lambda t: np.load(P / f"val_{t}.npy")

# 스칼라부는 E039(Public 56.8181) 그대로. 비율만 유지하고 CNN 몫에 맞춰 재정규화한다.
SCAL = {"img_ridge": .075, "img_gbdt": .18, "omni": .19, "eswf": .215}
SCAL_SUM = sum(SCAL.values())          # 0.66

# 가족 = (해상도, 정규화). 시드는 있는 것 전부 — 골라내면 val 에 맞추는 것이다.
FAM = {
    "med128": [t for t in ("cnn3h", "cnn3j") if (P / f"val_{t}.npy").exists()]
              + sorted(p.stem[4:] for p in P.glob("val_cnn3med*.npy")),
    "z128":   [t for t in ("cnn3c", "cnn3e", "cnn3g") if (P / f"val_{t}.npy").exists()]
              + sorted(p.stem[4:] for p in P.glob("val_cnn3z*.npy")),
    "z96":    [t for t in ("cnn3b", "cnn3d", "cnn3f") if (P / f"val_{t}.npy").exists()],
}
PX = {"med128": 128, "z128": 128, "z96": 96}
NORM = {"med128": "med", "z128": "z", "z96": "z"}


def main():
    fam = {k: v for k, v in FAM.items() if v}
    avg = {}
    for k, tags in fam.items():
        avg[k] = np.mean([L(t) for t in tags], axis=0)
        print(f"{k:7s} n={len(tags):2d} bag={rmse(avg[k]):.3f}  "
              + " ".join(f"{t}={rmse(L(t)):.2f}" for t in tags))

    scal = sum(w * L(t) for t, w in SCAL.items())
    grid_w = [0.30, 0.34, 0.38, 0.42, 0.46, 0.50]
    grid_s = [0, 1, 2, 3, 4]                    # 성긴 심플렉스 (정수비)
    best = None
    for wc in grid_w:
        for shares in itertools.product(grid_s, repeat=len(fam)):
            if sum(shares) == 0:
                continue
            sh = np.array(shares, float) / sum(shares)
            p = scal * ((1 - wc) / SCAL_SUM) + wc * sum(s * avg[k] for s, k in zip(sh, fam))
            r = rmse(p)
            if best is None or r < best[0]:
                best = (r, wc, dict(zip(fam, sh)))
    r, wc, sh = best
    print(f"\nbest val={r:.4f}  w_cnn={wc}  shares=" + " ".join(f"{k}:{v:.2f}" for k, v in sh.items()))

    weights, meta = {}, {}
    for t, w in SCAL.items():
        weights[t] = w * (1 - wc) / SCAL_SUM
    for k, tags in fam.items():
        if sh[k] <= 0:
            continue
        for t in tags:
            weights[t] = wc * sh[k] / len(tags)
            meta[t] = {"px": PX[k], "norm": NORM[k]}
    s = sum(weights.values())
    weights = {k: v / s for k, v in weights.items()}          # 부동소수 잔차 제거
    out = {"val_rmse": round(r, 4), "w_cnn": wc, "shares": sh, "weights": weights, "cnn": meta}
    pathlib.Path("/home/jovyan/work/blend.json").write_text(json.dumps(out, indent=1))
    print("wrote work/blend.json ·", len(meta), "CNN ·", f"sum={sum(weights.values()):.6f}")

    # 기준선: 현 제출본 E039 성분 (참고)
    base = scal + .113 * L("cnn3b") + .113 * L("cnn3c") + .114 * L("cnn3d")
    print(f"E039 성분 동일 가중 val={rmse(base):.4f} (팩 재추론 실측 63.441 에 대응)")


if __name__ == "__main__":
    main()
