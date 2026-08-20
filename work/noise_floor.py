"""Public 차이의 노이즈 바닥 (하한) — 팀원 관측 NEXT §7.4.

Public 은 test 3,868 의 절반(~1,934행)이고 6h 이웃이 입력창 20개 중 19개를 공유한다.
그래서 「관측된 Public Δ 하나」를 믿으려면 **짝지은 ΔRMSE 의 표집 σ** 를 알아야 한다.
val 을 6h 연결요소 단위로 부트스트랩해 그 σ 를 잰다.

가정: val 과 Public 이 같은 분포. 실제 절대값이 5점 벌어지므로 **하한**으로만 읽어라.
"""
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
P = pathlib.Path("/home/jovyan/work/preds")
T = [f"target_{i:02d}" for i in range(12)]
WCOL = [f"wind_{i:02d}" for i in range(20)]
NBOOT = 2000
TARGET_FRAC = 1.0        # 표본 크기를 val 대비 몇 배로 뽑을지
SCAL = {"img_ridge": .075, "img_gbdt": .18, "omni": .19, "eswf": .215}


def components(w):
    n = len(w)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    head = {}
    for j in range(n):
        head.setdefault(w[j, :19].tobytes(), []).append(j)
    for i in range(n):
        for j in head.get(w[i, 1:].tobytes(), []):
            if i != j:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    _, lab = np.unique([find(i) for i in range(n)], return_inverse=True)
    return lab


def main():
    va = pd.read_csv(ROOT / "validation/inputs.csv")
    Y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
    L = lambda t: np.load(P / f"val_{t}.npy")
    lab = components(np.round(va[WCOL].to_numpy(np.float64), 4))
    rows = [np.flatnonzero(lab == c) for c in range(lab.max() + 1)]
    sizes = np.array([len(r) for r in rows])
    print(f"val {len(va)}행 · 연결요소 {len(rows)}개 · 최대 {sizes.max()} · 중앙 {np.median(sizes):.0f}")

    scal = sum(w * L(t) for t, w in SCAL.items())
    A = scal + .113 * L("cnn3b") + .113 * L("cnn3c") + .114 * L("cnn3d")   # 현 카드 성분
    B = scal + .113 * L("cnn3b") + .113 * L("cnn3h") + .114 * L("cnn3d")   # cnn3c→cnn3h
    full = lambda p, m: float(np.sqrt(np.mean((Y[m] - p[m]) ** 2)))
    allm = np.ones(len(va), bool)
    print(f"전체 val: A={full(A, allm):.4f}  B={full(B, allm):.4f}  Δ={full(B, allm)-full(A, allm):+.4f}")

    rng = np.random.default_rng(0)
    n_test = len(pd.read_csv(ROOT / "test/inputs.csv"))
    target = int(TARGET_FRAC * len(va))
    dA, dB, dd = [], [], []
    for _ in range(NBOOT):
        pick, tot = [], 0
        while tot < target:
            c = rng.integers(len(rows))
            pick.append(rows[c])
            tot += sizes[c]
        ix = np.concatenate(pick)
        ra = np.sqrt(np.mean((Y[ix] - A[ix]) ** 2))
        rb = np.sqrt(np.mean((Y[ix] - B[ix]) ** 2))
        dA.append(ra)
        dB.append(rb)
        dd.append(rb - ra)
    dA, dd = np.array(dA), np.array(dd)
    print(f"\n절대 RMSE 의 표집 σ (반쪽 표본)      = {dA.std():.4f}")
    print(f"**짝지은 ΔRMSE 의 σ (같은 표본 비교)** = {dd.std():.4f}  "
          f"[2.5%, 97.5%] = [{np.percentile(dd,2.5):+.4f}, {np.percentile(dd,97.5):+.4f}]")
    print(f"판정 가능 최소 Δ (2σ)                 = {2*dd.std():.4f}")
    # Public 은 test 의 절반. 같은 블록 구조라면 σ ∝ 1/sqrt(블록 수)
    pub_rows = n_test // 2
    scale = np.sqrt(target / pub_rows)
    print(f"Public 환산 (test {n_test} 의 절반 {pub_rows}행): σ ≈ {dd.std()*scale:.4f}  "
          f"2σ ≈ {2*dd.std()*scale:.4f}")
    for name, obs in (("k 300→200 (Public −0.113)", -0.1134),
                      ("k 200→150 (Public −0.057)", -0.0571),
                      ("k 150→100 (Public +0.008)", +0.0076),
                      ("omnibag 순수 (Public +0.041)", +0.041),
                      ("3위↔4위 간격 0.0057", 0.0057)):
        print(f"  {name:32s} |Δ|/σ = {abs(obs)/dd.std():.2f}")


if __name__ == "__main__":
    main()
