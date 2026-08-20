"""검증 분포를 테스트 분포로 재가중해 카드 델타를 비교한다."""
import pathlib

import numpy as np
import pandas as pd

from noise_floor import components

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
PRED = pathlib.Path("/home/jovyan/work/preds")
T = [f"target_{i:02d}" for i in range(12)]
W = [f"wind_{i:02d}" for i in range(20)]


def qbins(x, n, edges=None):
    edges = np.quantile(x, np.linspace(0, 1, n + 1)) if edges is None else edges
    return np.searchsorted(edges[1:-1], x, side="right")


def one_d_weights(val, test):
    pool = np.r_[val, test]
    edges = np.quantile(pool, np.linspace(0, 1, 21))
    iv, it = qbins(val, 20, edges), qbins(test, 20, edges)
    nv = np.bincount(iv, minlength=20)
    nt = np.bincount(it, minlength=20)
    ratio = ((nt + 1.0) / (len(test) + 20)) / ((nv + 1.0) / (len(val) + 20))
    w = np.clip(ratio, 0.2, 5.0)[iv]
    return w / w.mean()


def two_d_weights(val, test):
    pool = np.vstack((val, test))
    _, ex, ey = np.histogram2d(pool[:, 0], pool[:, 1], bins=10)
    def cells(z):
        ix = np.clip(np.searchsorted(ex, z[:, 0], side="right") - 1, 0, 9)
        iy = np.clip(np.searchsorted(ey, z[:, 1], side="right") - 1, 0, 9)
        return ix, iy
    ivx, ivy = cells(val)
    itx, ity = cells(test)
    hv = np.zeros((10, 10), dtype=float)
    ht = np.zeros((10, 10), dtype=float)
    np.add.at(hv, (ivx, ivy), 1)
    np.add.at(ht, (itx, ity), 1)
    ratio = ((ht + 1.0) / (len(test) + 100)) / ((hv + 1.0) / (len(val) + 100))
    w = np.clip(ratio, 0.2, 5.0)[ivx, ivy]
    return w / w.mean()


def score(pred, y, w, ix=None):
    e = np.mean((pred - y) ** 2, axis=1)
    if ix is not None:
        e, w = e[ix], w[ix]
    return float(np.sqrt(np.sum(w * e) / np.sum(w)))


def main():
    vi = pd.read_csv(ROOT / "validation/inputs.csv")
    ti = pd.read_csv(ROOT / "test/inputs.csv")
    y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
    vw = vi[W].to_numpy(np.float64)
    tw = ti[W].to_numpy(np.float64)
    tags = ["img_ridge", "img_gbdt", "omni", "eswf", "cnn3b", "cnn3c", "cnn3d", "BAGsmall32"]
    p = {tag: np.load(PRED / f"val_{tag}.npy").astype(np.float64, copy=False) for tag in tags}
    assert all(x.shape == y.shape for x in p.values())

    scal = .075 * p["img_ridge"] + .18 * p["img_gbdt"] + .19 * p["omni"] + .215 * p["eswf"]
    e039 = scal + .113 * p["cnn3b"] + .113 * p["cnn3c"] + .114 * p["cnn3d"]
    cards = {
        "E039": e039,
        "MAX": .8 * e039 + .2 * p["BAGsmall32"],
        "N50": .5 * scal / .66 + .5 * p["BAGsmall32"],
    }
    v1 = one_d_weights(vw[:, 19], tw[:, 19])
    v2 = two_d_weights(np.c_[vw[:, 19], vw[:, 19] - vw[:, 15]],
                       np.c_[tw[:, 19], tw[:, 19] - tw[:, 15]])
    print(f"ESS 1D={v1.sum() ** 2 / np.sum(v1 ** 2):.2f}  2D={v2.sum() ** 2 / np.sum(v2 ** 2):.2f}")

    metrics = (np.ones(len(y)), v1, v2)
    public = {"E039": 56.8181, "MAX": 56.2743, "N50": 55.8298}
    names = ("unweighted", "1D", "2D")
    values = {}
    print("\nCARD RMSE")
    for name, pred in cards.items():
        values[name] = np.array([score(pred, y, w) for w in metrics])
        print(f"{name:5s} " + "  ".join(f"{n}={v:.4f}" for n, v in zip(names, values[name])) +
              f"  public={public[name]:.4f}")

    pairs = (("E039", "MAX"), ("MAX", "N50"), ("E039", "N50"))
    ratios = []
    print("\nDELTAS (second - first)")
    for a, b in pairs:
        pdlt = public[b] - public[a]
        d = values[b] - values[a]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = pdlt / d
        ratios.append(r)
        print(f"{a}->{b} public={pdlt:+.4f}  " + "  ".join(
            f"{n} d={x:+.4f} ratio={z:+.4f}" for n, x, z in zip(names, d, r)))
    ratios = np.asarray(ratios)
    gap, spread = np.mean(np.abs(ratios - 1), axis=0), np.ptp(ratios, axis=0)
    print("SUMMARY " + " | ".join(f"{n}:gap={g:.4f},spread={s:.4f}" for n, g, s in zip(names, gap, spread)) +
          f" | closest_to_1={names[gap.argmin()]} smallest_spread={names[spread.argmin()]}")

    lab = components(np.round(vw, 4))
    print("\nN50 CONNECTED COMPONENTS")
    print("component n_rows unweighted 1D_weighted 2D_weighted")
    for c in range(lab.max() + 1):
        ix = lab == c
        print(f"{c:9d} {ix.sum():7d} " + " ".join(f"{score(cards['N50'], y, w, ix):11.4f}" for w in metrics))


if __name__ == "__main__":
    main()
