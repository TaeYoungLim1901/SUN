"""N50 후보 스윕과 두 판정을 한 번에 출력한다."""
import itertools
import pathlib

import numpy as np
import pandas as pd

from noise_floor import components

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
P = pathlib.Path("/home/jovyan/work/preds")
T = [f"target_{i:02d}" for i in range(12)]
WCOL = [f"wind_{i:02d}" for i in range(20)]
GRID = tuple(np.round(np.arange(0.05, 0.5001, 0.05), 2))
HONEST_GRID = tuple(np.round(np.arange(0.0, 0.5001, 0.05), 2))


def load(split, tag):
    return np.load(P / f"{split}_{tag}.npy")


def score(pred, y, w=None, ix=None):
    e = np.mean((pred - y) ** 2, axis=1)
    if ix is not None:
        e = e[ix]
        w = w[ix] if w is not None else None
    return float(np.sqrt(np.mean(e) if w is None else np.sum(w * e) / np.sum(w)))


def corr(a, b, w=None):
    a, b = a.ravel(), b.ravel()
    if w is None:
        return float(np.corrcoef(a, b)[0, 1])
    ww = np.repeat(w, len(a) // len(w))
    ww = ww / ww.sum()
    a, b = a - np.sum(ww * a), b - np.sum(ww * b)
    return float(np.sum(ww * a * b) / np.sqrt(np.sum(ww * a * a) * np.sum(ww * b * b)))


def one_d_weights(val, test):
    pool = np.r_[val, test]
    edges = np.quantile(pool, np.linspace(0, 1, 21))
    iv = np.searchsorted(edges[1:-1], val, side="right")
    it = np.searchsorted(edges[1:-1], test, side="right")
    nv = np.bincount(iv, minlength=20)
    nt = np.bincount(it, minlength=20)
    ratio = ((nt + 1.0) / (len(test) + 20)) / ((nv + 1.0) / (len(val) + 20))
    return np.clip(ratio, 0.2, 5.0)[iv] / np.mean(np.clip(ratio, 0.2, 5.0)[iv])


def make_rows(families, cached, n50, bag0, y, weights=None):
    rows = []
    for k in (1, 2, 3):
        for names in itertools.combinations(families, k):
            paths = [p for name in names for p in families[name]]
            bag = np.mean([cached[p] for p in paths], axis=0)
            vals = [score((1 - w) * n50 + w * bag, y, weights) for w in GRID]
            j = int(np.argmin(vals))
            rows.append({"name": "+".join(names), "k": k, "n": len(paths), "bag": bag,
                         "solo": score(bag, y, weights), "rb": corr(bag - y, bag0 - y, weights),
                         "rn": corr(bag - y, n50 - y, weights), "w": GRID[j],
                         "rm": vals[j], "delta": vals[j] - score(n50, y, weights)})
    return sorted(rows, key=lambda r: r["delta"])


def print_table(title, rows, weighted=False):
    print(f"\n{title}")
    print("name | n | solo RMSE | rho-BAG | rho-N50 | best w | RMSE | " + ("wdelta" if weighted else "delta"))
    for r in rows:
        print(f"{r['name']:16s} | {r['n']:2d} | {r['solo']:10.4f} | {r['rb']:+.4f} | "
              f"{r['rn']:+.4f} | {r['w']:.2f} | {r['rm']:.4f} | {r['delta']:+.4f}")


def group_folds(lab, rng, n_folds=2):
    sizes = np.bincount(lab)
    load = np.zeros(n_folds)
    assign = np.zeros(len(sizes), np.int32)
    for c in rng.permutation(len(sizes)):
        f = int(np.argmin(load))
        assign[c] = f
        load[f] += sizes[c]
    return assign[lab]


def paired(a, b, va, y, lab):
    rows = [np.flatnonzero(lab == c) for c in range(lab.max() + 1)]
    full = score(b, y) - score(a, y)
    d = np.array([score(b[ix], y[ix]) - score(a[ix], y[ix]) for ix in rows])
    combos = itertools.combinations(range(len(rows)), len(rows) // 2)
    half = np.array([score(b[np.concatenate([rows[j] for j in c])], y[np.concatenate([rows[j] for j in c])]) -
                     score(a[np.concatenate([rows[j] for j in c])], y[np.concatenate([rows[j] for j in c])])
                     for c in combos])
    rng = np.random.default_rng(0)
    boot = []
    while len(boot) < 2000:
        pick, total = [], 0
        while total < len(va):
            ix = rows[rng.integers(len(rows))]
            pick.append(ix)
            total += len(ix)
        ix = np.concatenate(pick)
        boot.append(score(b[ix], y[ix]) - score(a[ix], y[ix]))
    boot = np.asarray(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\npaired overall_delta={full:+.4f} gain={int((d < 0).sum())}/{len(d)} "
          f"deltas=" + " ".join(f"{x:+.3f}" for x in d))
    print(f"paired half_sigma={half.std():.4f} bootstrap_sigma={boot.std():.4f} "
          f"CI95=[{lo:+.4f},{hi:+.4f}] delta/sigma={full / boot.std():+.2f}")
    return full


def honest(a, new, va, y, lab):
    deltas, picks = [], []
    for s in range(20):
        f = group_folds(lab, np.random.default_rng(s))
        for tr in (0, 1):
            fit, ev = f == tr, f != tr
            w = min(HONEST_GRID, key=lambda z: score((1 - z) * a + z * new, y, ix=fit))
            picks.append(w)
            deltas.append(score((1 - w) * a + w * new, y, ix=ev) - score(a, y, ix=ev))
    d = np.asarray(deltas)
    print(f"honest median_delta={np.median(d):+.4f} mean_delta={d.mean():+.4f} "
          f"gain={(d < 0).mean():.0%} w_hist=" + " ".join(
              f"{w:.2f}:{picks.count(w)}" for w in HONEST_GRID if picks.count(w)))
    return float(np.median(d)), float((d < 0).mean())


def main():
    vi = pd.read_csv(ROOT / "validation/inputs.csv")
    ti = pd.read_csv(ROOT / "test/inputs.csv")
    y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
    base4 = (.075 * load("val", "img_ridge") + .18 * load("val", "img_gbdt") +
             .19 * load("val", "omni") + .215 * load("val", "eswf")) / .66
    n50 = .5 * base4 + .5 * load("val", "BAGsmall32")
    base4t = (.075 * load("test", "img_ridge") + .18 * load("test", "img_gbdt") +
              .19 * load("test", "omni") + .215 * load("test", "eswf")) / .66
    n50t = .5 * base4t + .5 * load("test", "BAGsmall32")
    np.save(P / "val_N50.npy", n50)
    np.save(P / "test_N50.npy", n50t)
    bag0 = load("val", "BAGsmall32")
    print(f"N50 val RMSE={score(n50, y):.4f}")

    patterns = {"cnn7": "val_cnn7s*.npy", "cnn7p": "val_cnn7p*.npy", "cnn9": "val_cnn9s*.npy"}
    families = {name: sorted(P.glob(pattern)) for name, pattern in patterns.items()}
    for name in ("phys", "omni2"):
        path = P / f"val_{name}.npy"
        if path.exists():
            families[name] = [path]
    families = {name: paths for name, paths in families.items() if paths}
    cached = {p: np.load(p) for paths in families.values() for p in paths}
    rows = make_rows(families, cached, n50, bag0, y)
    print_table("UNWEIGHTED", rows)
    unions = [r for r in rows if r["k"] >= 2]
    if not unions:
        print("NO 2/3-family candidate")
        return
    best = unions[0]
    cand = (1 - best["w"]) * n50 + best["w"] * best["bag"]
    np.save(P / "val_CAND.npy", cand)
    lab = components(np.round(vi[WCOL].to_numpy(np.float64), 4))
    overall = paired(n50, cand, vi, y, lab)
    median, gain = honest(n50, best["bag"], vi, y, lab)
    print(f"ACCEPT={'PASS' if overall <= -.3 and median < 0 and gain >= .8 else 'FAIL'} "
          f"bar=delta<=-0.3 median<0 gain>=80% best={best['name']} w={best['w']:.2f}")

    weights = one_d_weights(vi["wind_19"].to_numpy(np.float64), ti["wind_19"].to_numpy(np.float64))
    weighted = make_rows(families, cached, n50, bag0, y, weights)
    print_table("IMPORTANCE-WEIGHTED (1D wind_19)", weighted, True)


if __name__ == "__main__":
    main()
