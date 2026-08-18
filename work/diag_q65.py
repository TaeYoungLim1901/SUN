"""hssq65 를 결합에 넣을지 로컬에서 판정. 학습 없음. NEXT §4 레버 2."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import kit  # noqa: E402

D = Path(__file__).parent / "local_data"
z = np.load(D / "pred_bundle.npz")
y = z["y"]
BASE = ["img_ridge", "img_gbdt", "v2_gbdt", "eswf", "cnn96b", "cnn96w3"]
vals = {n: z[f"val_{n}"].astype(np.float64) for n in BASE}
tests = {n: z[f"test_{n}"].astype(np.float64) for n in BASE}
for n in ("hssq65", "hssq65b"):
    vals[n] = np.load(D / f"val_{n}.npy").astype(np.float64)
    tests[n] = np.load(D / f"test_{n}.npy").astype(np.float64)

hss = y > 600
def report(tag, w):
    b = sum(w[n] * vals[n] for n in w)
    bias = float((b - y)[hss].mean())
    print(f"{tag:14} rmse {kit.score(y, b):7.3f}  HSS편향 {bias:7.1f}  w {dict(sorted(w.items()))}")
    return b

print("== 단독 ==")
for n in vals:
    print(f"  {n:9} {kit.score(y, vals[n]):7.3f}  HSS편향 {float((vals[n]-y)[hss].mean()):7.1f}")

print("== resid_corr(hssq65, ·)  ≥0.84 면 접는다 ==")
for n in BASE + ["hssq65b"]:
    print(f"  hssq65 ~ {n:9} {kit.resid_corr(vals['hssq65'], vals[n], y):.3f}")

print("== hill_climb ==")
w6, _ = kit.hill_climb({n: vals[n] for n in BASE}, y, n_iter=80)
report("base6", w6)
for extra in (["hssq65"], ["hssq65b"], ["hssq65", "hssq65b"]):
    w, _ = kit.hill_climb({n: vals[n] for n in BASE + extra}, y, n_iter=80)
    report("+" + "+".join(extra), w)

# 제출본 고정 가중치 대조군
sub = {"img_ridge": .125, "img_gbdt": .25, "v2_gbdt": .125, "eswf": .25,
       "cnn96b": .125, "cnn96w3": .125}
report("submitted", sub)

print("== 전체 후보 hill_climb ==")
allv = {k[4:]: z[k].astype(np.float64) for k in z.files if k.startswith("val_")}
allv["hssq65"] = vals["hssq65"]
for drop in ("persist", "imgpt", "eswfw3", "hssq65b"):
    allv.pop(drop, None)
w, _ = kit.hill_climb(allv, y, n_iter=80)
b = sum(w[n] * allv[n] for n in w)
print(f"  rmse {kit.score(y, b):7.3f}  HSS편향 {float((b-y)[hss].mean()):7.1f}")
print("  w", {k: round(v, 3) for k, v in sorted(w.items())})
