"""결합 가중 채택 판정 — sol 프로토콜: 6h 연결요소 그룹 2-fold × 20 반복.

한쪽 반에서 (w_cnn, 가족 배분) 을 고르고 **반대쪽 반에서만 채점**한다.
val 전체 RMSE 격자탐색(= E037 을 태운 방식)은 진단용으로만 출력한다.

  python work/blend_group.py            # 판정만
  python work/blend_group.py --write    # 통과했을 때 work/blend.json 갱신
"""
import argparse
import itertools
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
P = pathlib.Path("/home/jovyan/work/preds")
T = [f"target_{i:02d}" for i in range(12)]
WCOL = [f"wind_{i:02d}" for i in range(20)]
SEEDS = list(range(20))
GRID_W = [0.30, 0.34, 0.38, 0.42, 0.46, 0.50]
GRID_S = [0, 1, 2, 3, 4]

SCAL = {"img_ridge": .075, "img_gbdt": .18, "omni": .19, "eswf": .215}
SCAL_SUM = sum(SCAL.values())
BASE_CNN = {"cnn3b": .113, "cnn3c": .113, "cnn3d": .114}      # 현 제출본 E039 성분
PX = {"med128": 128, "z128": 128, "z96": 96, "wm2": 128, "px256": 256,
      "gap128med": 128, "gap128z": 128, "gap96med": 96}
NORM = {"med128": "med", "z128": "z", "z96": "z", "wm2": "med", "px256": "med",
        "gap128med": "med", "gap128z": "z", "gap96med": "med"}
WM = {k: 2.0 if k == "wm2" else 1.0 for k in PX}
ARCH = {k: "v4" if k.startswith("gap") else "v3" for k in PX}


def families():
    fam = {
        "med128": [t for t in ("cnn3h", "cnn3j") if (P / f"val_{t}.npy").exists()]
                  + sorted(p.stem[4:] for p in P.glob("val_cnn3med*.npy"))
                  + sorted(p.stem[4:] for p in P.glob("val_cnn3p20med*.npy")),
        "z128":   [t for t in ("cnn3c", "cnn3e", "cnn3g") if (P / f"val_{t}.npy").exists()]
                  + sorted(p.stem[4:] for p in P.glob("val_cnn3z*.npy"))
                  + sorted(p.stem[4:] for p in P.glob("val_cnn3p20z*.npy")),
        "z96":    [t for t in ("cnn3b", "cnn3d", "cnn3f") if (P / f"val_{t}.npy").exists()],
        "wm2":    sorted(p.stem[4:] for p in P.glob("val_cnn3wm2*.npy")),
        "px256":  sorted(p.stem[4:] for p in P.glob("val_cnn3px256*.npy")),
        "gap128med": sorted(p.stem[4:] for p in P.glob("val_cnn4a*.npy")),
        "gap128z":   sorted(p.stem[4:] for p in P.glob("val_cnn4b*.npy")),
        "gap96med":  sorted(p.stem[4:] for p in P.glob("val_cnn4c*.npy")),
    }
    return {k: v for k, v in fam.items() if v}


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def components(w):
    """같은 20창이 1스텝 밀려 겹치면 같은 성분 (diag_calib_refute.py 의 g6 그래프)."""
    n = len(w)
    uf = UF(n)
    head = {}
    for j in range(n):
        head.setdefault(w[j, :19].tobytes(), []).append(j)
    for i in range(n):
        for j in head.get(w[i, 1:].tobytes(), []):
            if i != j:
                uf.union(i, j)
    _, lab = np.unique([uf.find(i) for i in range(n)], return_inverse=True)
    return lab


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    va = pd.read_csv(ROOT / "validation/inputs.csv")
    Y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
    L = lambda t: np.load(P / f"val_{t}.npy")
    rmse = lambda p, m: float(np.sqrt(np.mean((Y[m] - p[m]) ** 2)))

    fam = families()
    avg = {k: np.mean([L(t) for t in v], axis=0) for k, v in fam.items()}
    scal = sum(w * L(t) for t, w in SCAL.items())
    base = scal + sum(w * L(t) for t, w in BASE_CNN.items())
    for k, v in fam.items():
        print(f"{k:7s} n={len(v):2d} bag={np.sqrt(np.mean((Y - avg[k])**2)):.3f}  " + " ".join(v))

    def blend(wc, sh):
        return scal * ((1 - wc) / SCAL_SUM) + wc * sum(s * avg[k] for k, s in sh.items())

    combos = []
    for wc in GRID_W:
        for sv in itertools.product(GRID_S, repeat=len(fam)):
            if sum(sv) == 0:
                continue
            sh = {k: s / sum(sv) for k, s in zip(fam, sv)}
            combos.append((wc, sh, blend(wc, sh)))
    print(f"격자 {len(combos)}개 · 성분 {len(fam)}가족")

    lab = components(np.round(va[WCOL].to_numpy(np.float64), 4))
    sizes = np.bincount(lab)
    print(f"연결요소 {len(sizes)}개 · 최대 {sizes.max()}행 ({sizes.max()/len(va):.1%})")

    all_rows = np.ones(len(va), bool)
    deltas = []
    for s in SEEDS:
        f = group_folds(lab, np.random.default_rng(s))
        for tr in (0, 1):
            m_fit, m_ev = f == tr, f != tr
            wc, sh, _ = min(combos, key=lambda c: rmse(c[2], m_fit))
            deltas.append(rmse(blend(wc, sh), m_ev) - rmse(base, m_ev))
    d = np.array(deltas)
    wins = int((d < 0).sum())
    print(f"\n그룹 2-fold×{len(SEEDS)} ({len(d)}판): 중앙 Δ={np.median(d):+.4f}  "
          f"평균 {d.mean():+.4f}  이득 {wins}/{len(d)}  최악 {d.max():+.4f}")

    wc, sh, p_full = min(combos, key=lambda c: rmse(c[2], all_rows))
    full = rmse(p_full, all_rows) - rmse(base, all_rows)
    print(f"전체 val: base={rmse(base, all_rows):.4f} → {rmse(p_full, all_rows):.4f} (Δ={full:+.4f})  "
          f"w_cnn={wc} " + " ".join(f"{k}:{v:.2f}" for k, v in sh.items()))

    ok = np.median(d) < 0 and wins >= 0.8 * len(d) and full <= -0.3
    print("판정:", "GO" if ok else "NO-GO",
          "(채택선: 중앙 Δ<0 · 이득 ≥80% · 전체 Δ ≤ −0.3)")

    if a.write:
        weights = {t: w * (1 - wc) / SCAL_SUM for t, w in SCAL.items()}
        meta = {}
        for k, tags in fam.items():
            if sh[k] <= 0:
                continue
            for t in tags:
                weights[t] = wc * sh[k] / len(tags)
                meta[t] = {"px": PX[k], "norm": NORM[k], "wm": WM[k], "arch": ARCH[k]}
        ssum = sum(weights.values())
        weights = {k: v / ssum for k, v in weights.items()}
        pathlib.Path("/home/jovyan/work/blend.json").write_text(json.dumps(
            {"val_rmse": round(rmse(p_full, all_rows), 4), "w_cnn": wc, "shares": sh,
             "group_median_delta": round(float(np.median(d)), 4), "wins": f"{wins}/{len(d)}",
             "weights": weights, "cnn": meta}, indent=1))
        print("wrote work/blend.json ·", len(meta), "CNN")


if __name__ == "__main__":
    main()
