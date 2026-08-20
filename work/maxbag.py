"""광류 시드백 크기를 키우며 판정. npy 기준, base = E039 성분 가중합."""
import glob, os, pathlib
import numpy as np, pandas as pd

P = "/home/jovyan/work/preds"
ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
T = [f"target_{i:02d}" for i in range(12)]
BASE = {"img_ridge":.075, "img_gbdt":.18, "omni":.19, "eswf":.215,
        "cnn3b":.113, "cnn3c":.113, "cnn3d":.114}

Y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
load = lambda s, t: np.load(f"{P}/{s}_{t}.npy")
rmse = lambda p: float(np.sqrt(np.mean((p - Y) ** 2)))
tags_of = lambda pref: sorted(os.path.basename(x)[4:-4] for x in glob.glob(f"{P}/val_{pref}*.npy"))

if __name__ == "__main__":
    base = sum(w * load("val", k) for k, w in BASE.items())
    print("base(E039 npy) val", round(rmse(base), 4))
    fams = {k: tags_of(p) for k, p in
            (("small","cnn6s6"), ("full","cnn6f5"), ("dual","cnn8d8"), ("g","cnn6g5"))}
    print({k: len(v) for k, v in fams.items()})
    combos = {"small32": fams["small"], "full24": fams["full"], "dual11": fams["dual"],
              "s+f": fams["small"] + fams["full"],
              "all69": sum(fams.values(), [])}
    for name, tg in combos.items():
        b = np.mean([load("val", t) for t in tg], axis=0)
        line = [f"{name:8s} n={len(tg):3d} solo={rmse(b):7.3f}"]
        for w in (0.15, 0.20, 0.25, 0.30, 0.35):
            line.append(f"w{w:.2f}={rmse(base*(1-w)+b*w):7.3f}")
        print(" ".join(line))
        np.save(f"{P}/val_BAG{name}.npy", b)
        te = np.mean([load("test", t) for t in tg], axis=0)
        np.save(f"{P}/test_BAG{name}.npy", te)
    np.save(f"{P}/val_BASE039.npy", base)
    np.save(f"{P}/test_BASE039.npy", sum(w * load("test", k) for k, w in BASE.items()))
