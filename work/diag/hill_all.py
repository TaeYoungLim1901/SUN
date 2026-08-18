"""Numpy-only hill_climb over every val_*.npy. No training."""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jovyan/work")
import kit  # noqa: E402

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
PRED = pathlib.Path("/home/jovyan/work/preds")
T = [f"target_{i:02d}" for i in range(12)]


def main():
    y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
    vals = {}
    for f in sorted(PRED.glob("val_*.npy")):
        name = f.stem[4:]
        v = np.load(f)
        if v.shape != y.shape:
            print(f"SKIP {name} shape {v.shape}")
            continue
        if not np.isfinite(v).all():
            print(f"SKIP {name} nan")
            continue
        vals[name] = v.astype(np.float64)
    print("N_TAGS", len(vals), sorted(vals))
    print()
    print("== SINGLE ==")
    rows = sorted(((n, kit.score(y, v)) for n, v in vals.items()), key=lambda x: x[1])
    for n, s in rows:
        print(f"  {n:16s} {s:8.3f}")

    print()
    print("== HILL ALL ==")
    w, best = kit.hill_climb(vals, y, n_iter=80)
    print(f"blend {best:.3f}")
    for n, wt in sorted(w.items(), key=lambda kv: -kv[1]):
        print(f"  {n:16s} {wt:.3f}")
    used = set(w)
    unused = [n for n, _ in rows if n not in used]

    print()
    print("== RESID unused vs img_blend / v2_gbdt / align ==")
    print(f"{'tag':16s} {'rmse':>8s} {'rho_blend':>10s} {'rho_v2':>8s} {'rho_align':>10s}")
    for n in unused:
        rb = kit.resid_corr(vals[n], vals["img_blend"], y)
        rv = kit.resid_corr(vals[n], vals["v2_gbdt"], y)
        ra = kit.resid_corr(vals[n], vals["align"], y) if "align" in vals else float("nan")
        print(f"  {n:16s} {kit.score(y, vals[n]):8.3f} {rb:10.3f} {rv:8.3f} {ra:10.3f}")

    print()
    print("== RESID used vs img_blend / v2_gbdt / align ==")
    for n in sorted(used):
        if n in ("img_blend", "v2_gbdt", "align"):
            continue
        rb = kit.resid_corr(vals[n], vals["img_blend"], y)
        rv = kit.resid_corr(vals[n], vals["v2_gbdt"], y)
        ra = kit.resid_corr(vals[n], vals["align"], y) if "align" in vals else float("nan")
        print(f"  {n:16s} {kit.score(y, vals[n]):8.3f} {rb:10.3f} {rv:8.3f} {ra:10.3f}")

    print()
    print("== KEY PAIRWISE ==")
    if "align" in vals:
        print("align~v2", round(kit.resid_corr(vals["align"], vals["v2_gbdt"], y), 3))
        print("align~img_blend", round(kit.resid_corr(vals["align"], vals["img_blend"], y), 3))
    if "hssq65" in vals:
        print("hssq65~v2", round(kit.resid_corr(vals["hssq65"], vals["v2_gbdt"], y), 3))
        if "align" in vals:
            print("hssq65~align", round(kit.resid_corr(vals["hssq65"], vals["align"], y), 3))
            print("hssq65~img_blend", round(kit.resid_corr(vals["hssq65"], vals["img_blend"], y), 3))

    print()
    print("== TARGETED BLENDS ==")
    base6 = ["img_ridge", "img_gbdt", "v2_gbdt", "eswf", "cnn96b", "cnn96w3"]
    specs = [
        ("submit6", base6),
        ("align_for_v2", ["img_ridge", "img_gbdt", "align", "eswf", "cnn96b", "cnn96w3"]),
        ("submit6+align", base6 + ["align"]),
        ("submit6+hssq65", base6 + ["hssq65"]),
        ("align+hss", ["img_ridge", "img_gbdt", "align", "eswf", "cnn96b", "cnn96w3", "hssq65"]),
        ("all_open", [n for n in vals if n not in (
            "persist", "rotter", "imgpt", "eswfw3", "v2_ridge", "ridge",
            "gbdt", "ridge_gbdt", "v2_blend")]),
    ]
    if "chfeat" in vals:
        specs += [
            ("chfeat_for_v2", ["img_ridge", "img_gbdt", "chfeat", "eswf", "cnn96b", "cnn96w3"]),
            ("chfeat_for_align", ["img_ridge", "img_gbdt", "chfeat", "eswf", "cnn96b", "cnn96w3"]),
            ("align+chfeat", ["img_ridge", "img_gbdt", "align", "chfeat", "eswf", "cnn96b", "cnn96w3"]),
            ("six+chfeat", base6 + ["chfeat"]),
        ]
    for tag, names in specs:
        sub = {n: vals[n] for n in names if n in vals}
        ww, ss = kit.hill_climb(sub, y, n_iter=80)
        pretty = {k: round(v, 3) for k, v in sorted(ww.items(), key=lambda kv: -kv[1])}
        print(f"{tag:24s} {ss:.3f}  {pretty}")


if __name__ == "__main__":
    main()
