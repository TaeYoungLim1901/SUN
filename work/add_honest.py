"""성분 추가를 **정직하게** 판정한다 — 가중 w 를 반쪽에서 고르고 반대쪽에서만 채점.

`contrib.py` 는 val 전체에서 최적 w 를 골라 Δ 를 보고한다(낙관 편향). 여기서는
6h 연결요소 2-fold × N 반복으로 w 선택과 채점을 분리한다. CHECKLIST.md D 의 채택 판정용.

  python work/add_honest.py cnn5hub                 # 태그 하나 추가
  python work/add_honest.py cnn5hub cnn5h4101 ...   # 여러 태그를 평균(시드백)해 추가
"""
import pathlib
import sys

import numpy as np
import pandas as pd

from noise_floor import components

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
P = pathlib.Path("/home/jovyan/work/preds")
T = [f"target_{i:02d}" for i in range(12)]
WCOL = [f"wind_{i:02d}" for i in range(20)]
BASE = {"img_ridge": .075, "img_gbdt": .18, "omni": .19, "eswf": .215,
        "cnn3b": .113, "cnn3c": .113, "cnn3d": .114}
GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
REPEATS = 20


def group_folds(lab, rng, n_folds=2):
    sizes = np.bincount(lab)
    load = np.zeros(n_folds)
    assign = np.zeros(len(sizes), np.int32)
    for c in rng.permutation(len(sizes)):
        f = int(np.argmin(load))
        assign[c] = f
        load[f] += sizes[c]
    return assign[lab]


def main():
    tags = sys.argv[1:]
    assert tags, "태그를 하나 이상 줘라"
    va = pd.read_csv(ROOT / "validation/inputs.csv")
    Y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
    L = lambda t: np.load(P / f"val_{t}.npy")
    have = [t for t in tags if (P / f"val_{t}.npy").exists()]
    miss = [t for t in tags if t not in have]
    if miss:
        print("없는 태그:", miss)
    assert have
    New = np.mean([L(t) for t in have], axis=0)
    A = sum(w * L(t) for t, w in BASE.items())
    rm = lambda p, m: float(np.sqrt(np.mean((Y[m] - p[m]) ** 2)))
    allm = np.ones(len(va), bool)
    print(f"추가 성분 {len(have)}개 {have}")
    print(f"  단독 {rm(New, allm):.3f} · 기준 블렌드 {rm(A, allm):.4f} · "
          f"ρ(잔차) {np.corrcoef((A-Y).ravel(), (New-Y).ravel())[0,1]:.4f}")

    lab = components(np.round(va[WCOL].to_numpy(np.float64), 4))
    deltas, picks = [], []
    for s in range(REPEATS):
        f = group_folds(lab, np.random.default_rng(s))
        for tr in (0, 1):
            fit, ev = f == tr, f != tr
            w = min(GRID, key=lambda w: rm((1 - w) * A + w * New, fit))
            picks.append(w)
            deltas.append(rm((1 - w) * A + w * New, ev) - rm(A, ev))
    d = np.array(deltas)
    wins = int((d < 0).sum())
    print(f"\n정직 판정 (2-fold×{REPEATS} = {len(d)}판)")
    print(f"  중앙 Δ {np.median(d):+.4f} · 평균 {d.mean():+.4f} · 이득 {wins}/{len(d)} "
          f"({wins/len(d):.0%}) · 최악 {d.max():+.4f}")
    print(f"  고른 w 분포: " + " ".join(f"{w}:{picks.count(w)}" for w in GRID if picks.count(w)))
    w_full = min(GRID, key=lambda w: rm((1 - w) * A + w * New, allm))
    full = rm((1 - w_full) * A + w_full * New, allm) - rm(A, allm)
    print(f"  참고(낙관) 전체 val 최적 w={w_full} Δ={full:+.4f}")
    ok = np.median(d) < 0 and wins >= 0.8 * len(d) and full <= -0.3
    print("판정:", "GO" if ok else "NO-GO",
          "— 선: 중앙 Δ<0 · 이득 ≥80% · 전체 Δ ≤ −0.3")


if __name__ == "__main__":
    main()
