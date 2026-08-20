"""base4 / 광류백 / 하이브리드백 3그룹 몫을 굵은 격자로 훑고, 정직 2-fold 로 고른다."""
import itertools, pathlib
import numpy as np, pandas as pd
import sys
sys.path.insert(0, '/home/jovyan/work')
from noise_floor import components

P = pathlib.Path('/home/jovyan/work/preds')
ROOT = pathlib.Path('/home/jovyan/public_dataset/competition_dataset_6h')
T = [f'target_{i:02d}' for i in range(12)]
WCOL = [f'wind_{i:02d}' for i in range(20)]
va = pd.read_csv(ROOT / 'validation/inputs.csv')
Y = pd.read_csv(ROOT / 'validation/targets.csv')[T].to_numpy(np.float64)
L = lambda t: np.load(P / f'val_{t}.npy')
B4 = {'img_ridge': .075, 'img_gbdt': .18, 'omni': .19, 'eswf': .215}
base4 = sum(w / 0.66 * L(k) for k, w in B4.items())
flow = L('BAGsmall32')
h13 = np.mean([L(t) for t in [f'cnn7s71{i:02d}' for i in range(1, 11)] + ['cnn9s9101', 'cnn9s9102', 'cnn9s9103']], axis=0)
h24 = np.mean([L(t) for t in sorted(x.name[4:-4] for x in P.glob('val_cnn7s*.npy')) +
               sorted(x.name[4:-4] for x in P.glob('val_cnn9s*.npy'))], axis=0)
lab = components(np.round(va[WCOL].to_numpy(np.float64), 4))
rows = [np.flatnonzero(lab == c) for c in range(lab.max() + 1)]
rm = lambda p, ix: float(np.sqrt(np.mean((Y[ix] - p[ix]) ** 2)))
allix = np.arange(len(va))

grid = [(a, f, h) for a in np.arange(0.35, 0.66, 0.05) for f in np.arange(0.10, 0.56, 0.05)
        for h in [round(1 - a - f, 2)] if 0.0 <= h <= 0.45]
for name, H in (('h13', h13), ('h24', h24)):
    scored = sorted(((rm(a * base4 + f * flow + h * H, allix), a, f, h) for a, f, h in grid))[:6]
    print(name, ' '.join(f'[{a:.2f}/{f:.2f}/{h:.2f}]={v:.4f}' for v, a, f, h in scored))

# 정직: 성분 절반에서 고르고 반대쪽 채점 (2-fold x 20)
rng = np.random.default_rng(0)
for name, H in (('h13', h13), ('h24', h24)):
    ds = []
    for _ in range(20):
        perm = rng.permutation(len(rows))
        for half in (perm[:len(rows) // 2], perm[len(rows) // 2:]):
            fit = np.concatenate([rows[j] for j in half])
            ev = np.concatenate([rows[j] for j in perm if j not in half])
            best = min(grid, key=lambda g: rm(g[0] * base4 + g[1] * flow + g[2] * H, fit))
            n50 = 0.5 * base4 + 0.5 * flow
            ds.append(rm(best[0] * base4 + best[1] * flow + best[2] * H, ev) - rm(n50, ev))
    ds = np.array(ds)
    print(f'{name} 정직 중앙 {np.median(ds):+.4f} 평균 {ds.mean():+.4f} 이득 {(ds<0).mean():.0%}')
