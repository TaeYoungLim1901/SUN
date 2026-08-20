"""검증과 테스트 입력의 공변량 이동을 측정한다."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors

ROOT = Path("/home/jovyan/public_dataset/competition_dataset_6h")
PCTS = np.arange(10, 100, 10)


def ks(a, b):
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    x = np.sort(np.r_[a, b])
    return float(np.max(np.abs(np.searchsorted(a, x, "right") / a.size -
                               np.searchsorted(b, x, "right") / b.size)))


def show_metric(name, values):
    print(f"\n{name}")
    print("decile       train validation      test")
    q = [np.percentile(values[s], PCTS) for s in ("train", "validation", "test")]
    for p, a, b, c in zip(PCTS, *q):
        print(f"{p:>3d}% {a:12.3f} {b:12.3f} {c:10.3f}")
    print(f"KS validation-test {ks(values['validation'], values['test']):.4f}"
          f"  train-test {ks(values['train'], values['test']):.4f}")


def main():
    from feats import sample_feats

    splits = ("train", "validation", "test")
    inputs = {s: pd.read_csv(ROOT / s / "inputs.csv") for s in splits}
    wind = {s: inputs[s][[f"wind_{i:02d}" for i in range(20)]].to_numpy(float)
            for s in splits}
    metrics = {
        "wind_19": {s: wind[s][:, 19] for s in splits},
        "mean(wind)": {s: wind[s].mean(1) for s in splits},
        "std(wind)": {s: wind[s].std(1) for s in splits},
        "wind_19-wind_18": {s: wind[s][:, 19] - wind[s][:, 18] for s in splits},
    }
    for name, values in metrics.items():
        show_metric(name, values)

    feats = {s: np.asarray(sample_feats(s, inputs[s]), dtype=np.float64) for s in splits}
    assert all(x.shape[1] == 839 and np.isfinite(x).all() for x in feats.values())
    v19, t19 = wind["validation"][:, 19], wind["test"][:, 19]
    vmin, vmax = v19.min(), v19.max()
    print(f"\nwind_19 test outside validation min/max: "
          f"{np.mean((t19 < vmin) | (t19 > vmax)):.4f}")
    flo, fhi = feats["validation"].min(0), feats["validation"].max(0)
    oor = (feats["test"] < flo) | (feats["test"] > fhi)
    print(f"feature block test rows with any out-of-range feature: {oor.any(1).mean():.4f}")
    print(f"median out-of-range features per test row: {np.median(oor.sum(1)):.1f}")

    X = np.vstack((feats["validation"], feats["test"]))
    y = np.r_[np.zeros(len(feats["validation"])), np.ones(len(feats["test"]))]
    clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=.05,
                                         max_leaf_nodes=31, random_state=0)
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    auc = roc_auc_score(y, prob)
    print(f"\nadversarial validation ROC AUC: {auc:.4f}")
    if auc > .6:
        clf.fit(X, y)
        pi = permutation_importance(clf, X, y, scoring="roc_auc", n_repeats=3,
                                    random_state=0, n_jobs=1)
        print("top 15 permutation-importance features")
        for i in np.argsort(pi.importances_mean)[-15:][::-1]:
            print(f"f{i:03d} {pi.importances_mean[i]:.6f}")

    mean, std = feats["train"].mean(0), feats["train"].std(0)
    std[std == 0] = 1
    z = {s: (feats[s] - mean) / std for s in splits}
    rng = np.random.default_rng(0)
    sample = z["test"][rng.choice(len(z["test"]), min(500, len(z["test"])), replace=False)]
    d_test = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=1).fit(z["validation"])
    d_val = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=1).fit(z["train"])
    d_test = d_test.kneighbors(sample, return_distance=True)[0][:, 0]
    d_val = d_val.kneighbors(z["validation"], return_distance=True)[0][:, 0]
    print("\nnearest-neighbor distance deciles (test->validation, validation->train)")
    for p, a, b in zip(PCTS, np.percentile(d_test, PCTS), np.percentile(d_val, PCTS)):
        print(f"{p:>3d}% {a:12.3f} {b:12.3f}")


if __name__ == "__main__":
    main()
