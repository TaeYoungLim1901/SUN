"""두 예측 배열을 6h 연결요소 단위로 짝지어 판정한다. 채택·**기각 둘 다** 이걸로 한 번 더 본다.

  python work/paired.py A.npy B.npy [이름A 이름B]

내는 것: 전체 Δ · 성분별 Δ 와 부호 일치 · 절반분할 σ · 부트스트랩 CI · Public 환산.
숫자 하나(전체 Δ)로 축을 닫지 말라는 규칙의 구현이다 (val 은 연결요소가 11개뿐이다).
"""
import itertools
import pathlib
import sys

import numpy as np
import pandas as pd

from noise_floor import components

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
T = [f"target_{i:02d}" for i in range(12)]
WCOL = [f"wind_{i:02d}" for i in range(20)]


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    na, nb = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else ("A", "B")
    A, B = np.load(a_path), np.load(b_path)
    va = pd.read_csv(ROOT / "validation/inputs.csv")
    Y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
    assert A.shape == B.shape == Y.shape, (A.shape, B.shape, Y.shape)

    lab = components(np.round(va[WCOL].to_numpy(np.float64), 4))
    rows = [np.flatnonzero(lab == c) for c in range(lab.max() + 1)]
    rm = lambda p, ix: float(np.sqrt(np.mean((Y[ix] - p[ix]) ** 2)))
    allix = np.arange(len(va))
    full = rm(B, allix) - rm(A, allix)
    print(f"{na}={rm(A, allix):.4f}  {nb}={rm(B, allix):.4f}  전체 Δ={full:+.4f}  "
          f"corr={np.corrcoef(A.ravel(), B.ravel())[0,1]:.5f}")

    d = np.array([rm(B, ix) - rm(A, ix) for ix in rows])
    print("성분별 Δ: " + " ".join(f"{x:+.3f}" for x in d))
    print(f"  이득 {int((d < 0).sum())}/{len(d)} · 중앙 {np.median(d):+.4f} · 평균 {d.mean():+.4f}")

    ncomp = len(rows)
    combos = list(itertools.combinations(range(ncomp), ncomp // 2))
    half = np.array([rm(B, np.concatenate([rows[j] for j in c])) -
                     rm(A, np.concatenate([rows[j] for j in c])) for c in combos])
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(2000):
        pick, tot = [], 0
        while tot < len(va):
            c = rng.integers(ncomp)
            pick.append(rows[c])
            tot += len(rows[c])
        boot.append(rm(B, np.concatenate(pick)) - rm(A, np.concatenate(pick)))
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"절반분할 {len(half)}조합: σ={half.std():.4f} · 이득 {(half < 0).mean():.0%}")
    print(f"부트스트랩 2000: σ={boot.std():.4f} · 95% CI [{lo:+.4f}, {hi:+.4f}] · 이득 {(boot < 0).mean():.0%}")
    print(f"전체 Δ / 로컬 σ = {full / boot.std():+.2f}σ")

    verdict = ("채택 후보" if full <= -0.3 and (d < 0).mean() >= 0.8 and hi < 0 else
               "판정 불가 (σ 안)" if abs(full) < 2 * boot.std() else "악화")
    print("판정:", verdict, "— 채택선: 전체 Δ ≤ −0.3 · 성분 이득 ≥80% · CI 상한 < 0")


if __name__ == "__main__":
    main()
