"""노이즈 바닥 교차검증 — 부트스트랩 하나로 결론내지 않는다.

(1) 성분별 Δ 를 직접 본다 (σ 가 큰 이유가 실재하는지)
(2) 성분 절반 분할 σ (복원추출 없음)
(3) 성분 부트스트랩 σ
셋이 같은 자릿수여야 결론을 쓴다.
"""
import itertools
import pathlib

import numpy as np
import pandas as pd

from noise_floor import components

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
P = pathlib.Path("/home/jovyan/work/preds")
T = [f"target_{i:02d}" for i in range(12)]
WCOL = [f"wind_{i:02d}" for i in range(20)]
SCAL = {"img_ridge": .075, "img_gbdt": .18, "omni": .19, "eswf": .215}

va = pd.read_csv(ROOT / "validation/inputs.csv")
Y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
L = lambda t: np.load(P / f"val_{t}.npy")
lab = components(np.round(va[WCOL].to_numpy(np.float64), 4))
rows = [np.flatnonzero(lab == c) for c in range(lab.max() + 1)]

scal = sum(w * L(t) for t, w in SCAL.items())
A = scal + .113 * L("cnn3b") + .113 * L("cnn3c") + .114 * L("cnn3d")
B = scal + .113 * L("cnn3b") + .113 * L("cnn3h") + .114 * L("cnn3d")
rm = lambda p, ix: float(np.sqrt(np.mean((Y[ix] - p[ix]) ** 2)))
allix = np.arange(len(va))
print(f"상관 corr(A,B) = {np.corrcoef(A.ravel(), B.ravel())[0,1]:.6f}")
print(f"전체 Δ = {rm(B, allix) - rm(A, allix):+.4f}\n")

print("성분별:")
for i, ix in enumerate(rows):
    print(f"  c{i:02d} n={len(ix):4d}  A={rm(A,ix):7.3f}  B={rm(B,ix):7.3f}  Δ={rm(B,ix)-rm(A,ix):+.4f}")

# (2) 성분 절반 분할 — 복원추출 없음. 「어느 절반을 받았나」의 변동
half = []
ncomp = len(rows)
combos = list(itertools.combinations(range(ncomp), ncomp // 2))
rng = np.random.default_rng(0)
for c in (combos if len(combos) <= 500 else [tuple(rng.choice(ncomp, ncomp // 2, replace=False)) for _ in range(500)]):
    ix = np.concatenate([rows[j] for j in c])
    half.append(rm(B, ix) - rm(A, ix))
half = np.array(half)
print(f"\n(2) 성분 절반 분할 {len(half)}개 · 평균 행수 {len(va)//2}: Δ σ = {half.std():.4f}  "
      f"평균 {half.mean():+.4f}  범위 [{half.min():+.4f}, {half.max():+.4f}]")

# (3) 부트스트랩 (복원추출)
boot = []
for _ in range(2000):
    pick, tot = [], 0
    while tot < len(va):
        c = rng.integers(ncomp)
        pick.append(rows[c])
        tot += len(rows[c])
    ix = np.concatenate(pick)
    boot.append(rm(B, ix) - rm(A, ix))
boot = np.array(boot)
print(f"(3) 성분 부트스트랩 2000회 · 목표 {len(va)}행: Δ σ = {boot.std():.4f}")

pub = 3868 // 2
print(f"\nPublic({pub}행) 환산 — 블록 구조 같다고 보고 σ ∝ 1/sqrt(행수):")
for name, sd, n in (("절반분할", half.std(), len(va) // 2), ("부트스트랩", boot.std(), len(va))):
    print(f"  {name:6s} σ={sd:.4f} @ {n}행  →  Public σ ≈ {sd*np.sqrt(n/pub):.4f}  (2σ={2*sd*np.sqrt(n/pub):.4f})")
