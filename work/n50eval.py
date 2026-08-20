"""cnn7(하이브리드 스칼라) 시드가 N50 카드에 새 정보를 주는지 본다.

  python work/n50eval.py cnn7s7101 cnn7s7102 ...

내는 것: 시드 단독 val · 시드백 단독 · **잔차 ρ**(광류백·N50 대비) · 추가 w 곡선.
ρ≥0.99 면 닫힌 축 재탕이다 (NEXT.md). 채택 판정은 paired.py + add_honest.py 로 따로.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

P = pathlib.Path("/home/jovyan/work/preds")
ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
T = [f"target_{i:02d}" for i in range(12)]
BASE4 = {"img_ridge": .075, "img_gbdt": .18, "omni": .19, "eswf": .215}   # N50 = 이 넷 ×0.5 + 광류백 ×0.5

Y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
load = lambda s, t: np.load(P / f"{s}_{t}.npy")
rmse = lambda p: float(np.sqrt(np.mean((p - Y) ** 2)))
rho = lambda a, b: float(np.corrcoef((a - Y).ravel(), (b - Y).ravel())[0, 1])

tot = sum(BASE4.values())
base4 = sum(w / tot * load("val", k) for k, w in BASE4.items())
bag = load("val", "BAGsmall32")
n50 = 0.5 * base4 + 0.5 * bag
print(f"base4={rmse(base4):.4f}  flowbag32={rmse(bag):.4f}  N50={rmse(n50):.4f}")

tags = sys.argv[1:]
new = np.mean([load("val", t) for t in tags], axis=0)
for t in tags:
    p = load("val", t)
    print(f"  {t:12s} solo={rmse(p):7.3f}  rho(bag)={rho(p, bag):.4f}  rho(N50)={rho(p, n50):.4f}")
print(f"new bag n={len(tags)} solo={rmse(new):.4f}  rho(bag)={rho(new, bag):.4f}  rho(N50)={rho(new, n50):.4f}")

print("추가 (N50 ×(1-w) + new ×w):")
for w in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
    print(f"  w={w:.2f}  {rmse(n50 * (1 - w) + new * w):.4f}")
print("광류백 자리 일부를 새 백으로 (base4 0.5 고정, 광류몫 0.5 안에서 f 만큼 교체):")
for f in (0.25, 0.5, 0.75, 1.0):
    print(f"  f={f:.2f}  {rmse(0.5 * base4 + 0.5 * ((1 - f) * bag + f * new)):.4f}")
np.save(P / "val_N50.npy", n50)
