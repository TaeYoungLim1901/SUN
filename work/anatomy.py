"""검증 예측의 오차 구조를 표로 해부한다."""
import pathlib
import sys

import numpy as np
import pandas as pd

from noise_floor import components

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
T = [f"target_{i:02d}" for i in range(12)]
W = [f"wind_{i:02d}" for i in range(20)]
TB = ["<350", "350-450", "450-550", "550-650", ">650"]
rmse = lambda a: float(np.sqrt(np.mean(np.square(a))))


def report(title, groups, err, rows=False):
    print(f"\n{title}")
    print(f"{'bin':<24} {'n_rows' if rows else 'n_cells':>7} {'RMSE':>10} {'bias':>10}")
    for name, mask in groups:
        mask = np.asarray(mask)
        z = err[mask] if rows else err.ravel()[mask]
        print(f"{str(name):<24} {int(mask.sum()):7d} {rmse(z):10.4f} {z.mean():+10.4f}")


def dist(title, values, keys):
    print(f"  {title}")
    for key in keys:
        n = int(np.sum(values == key))
        print(f"    {str(key):<20} {n:6d} {100*n/len(values):7.2f}%")


def main():
    pred = np.load(sys.argv[1] if len(sys.argv) > 1 else "/home/jovyan/work/preds/val_N50.npy")
    inp = pd.read_csv(ROOT / "validation/inputs.csv")
    y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
    wind = inp[W].to_numpy(np.float64)
    assert pred.shape == y.shape == (len(inp), 12), (pred.shape, y.shape)

    e = pred - y
    s = e * e
    total = s.sum()
    lab = components(np.round(wind, 4))
    last = wind[:, 19]
    yflat, sflat = y.ravel(), s.ravel()
    target_bin = np.select([yflat < 350, yflat < 450, yflat < 550, yflat < 650],
                           TB[:4], default=TB[4])

    print(f"TOTAL RMSE {rmse(e):.4f}   SSE {total:.4f}")
    print(f"{'horizon':>8} {'RMSE':>10} {'SSE share':>12}")
    for h in range(12):
        print(f"{h:8d} {rmse(e[:, h]):10.4f} {100*s[:, h].sum()/total:11.2f}%")

    print("\nCONNECTED COMPONENTS")
    print(f"{'component':>10} {'n_rows':>7} {'RMSE':>10} {'SSE share':>12} {'wind_19':>10} {'target':>10} {'bias':>10}")
    for c in np.unique(lab):
        ix = lab == c
        print(f"{c:10d} {ix.sum():7d} {rmse(e[ix]):10.4f} {100*s[ix].sum()/total:11.2f}% "
              f"{last[ix].mean():10.3f} {y[ix].mean():10.3f} {e[ix].mean():+10.3f}")

    qwind = pd.qcut(last, 5, duplicates="drop")
    wind_groups = [(str(c), np.asarray(qwind == c)) for c in qwind.categories]
    report("LAST WIND QUINTILES", wind_groups, e, True)

    tgroups = [(b, target_bin == b) for b in TB]
    report("TRUE TARGET BINS", tgroups, e)
    far_bin = target_bin.reshape(y.shape)[:, 6:].ravel()
    report("TRUE TARGET BINS HORIZONS 6-11", [(b, far_bin == b) for b in TB], e[:, 6:])

    change = np.abs(y - last[:, None]).ravel()
    qchange = pd.qcut(change, 5, duplicates="drop")
    change_groups = [(str(c), np.asarray(qchange == c)) for c in qchange.categories]
    report("ABS TARGET-WIND_19 QUINTILES", change_groups, e)

    order = np.argsort(sflat)[::-1]
    print("\nCONCENTRATION")
    for pct in (.01, .05, .10, .25):
        k = max(1, int(np.ceil(len(order) * pct)))
        print(f"worst {pct:.0%}: {100*sflat[order[:k]].sum()/total:7.2f}% of SSE ({k} cells)")
    k = max(1, int(np.ceil(len(order) * .01)))
    wi = order[:k]
    dist("horizon", wi % 12, range(12))
    dist("target bin", target_bin[wi], TB)
    dist("component", np.repeat(lab, 12)[wi], np.unique(lab))

    row_rmse = np.sqrt(np.mean(s, axis=1))
    print("\n15 WORST ROWS BY ROW-RMSE")
    print(f"{'row':>6} {'component':>10} {'wind_19':>10} {'target min':>11} {'target max':>11} "
          f"{'pred min':>10} {'pred max':>10} {'row RMSE':>10}")
    for i in np.argsort(row_rmse)[-15:][::-1]:
        print(f"{i:6d} {lab[i]:10d} {last[i]:10.3f} {y[i].min():11.3f} {y[i].max():11.3f} "
              f"{pred[i].min():10.3f} {pred[i].max():10.3f} {row_rmse[i]:10.4f}")


if __name__ == "__main__":
    main()
