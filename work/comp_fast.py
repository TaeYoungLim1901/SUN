"""성분별 구간 성적. 고속 구간에서 누가 이기는지 본다 (전체 val 로는 안 보인다)."""
import pathlib
import numpy as np, pandas as pd
P = pathlib.Path('/home/jovyan/work/preds')
ROOT = pathlib.Path('/home/jovyan/public_dataset/competition_dataset_6h')
T = [f'target_{i:02d}' for i in range(12)]
Y = pd.read_csv(ROOT / 'validation/targets.csv')[T].to_numpy(np.float64)
lw = pd.read_csv(ROOT / 'validation/inputs.csv')['wind_19'].to_numpy(np.float64)
L = lambda t: np.load(P / f'val_{t}.npy')
rm = lambda p, m: float(np.sqrt(np.mean((p[m] - Y[m]) ** 2)))
M = (('전체', lw > 0), ('lw<450', lw < 450), ('lw>=450', lw >= 450), ('lw>=550', lw >= 550))
tags = ['img_ridge', 'img_gbdt', 'omni', 'eswf', 'cnn3b', 'cnn3c', 'cnn3d', 'BAGsmall32', 'phys', 'omni2']
print(f"{'성분':12s} " + ' '.join(f'{n:>9s}' for n, _ in M))
for t in tags:
    if (P / f'val_{t}.npy').exists():
        p = L(t)
        print(f'{t:12s} ' + ' '.join(f'{rm(p, m):9.3f}' for _, m in M))
c9 = np.mean([L(x.name[4:-4]) for x in P.glob('val_cnn9s*.npy')], axis=0)
print(f"{'cnn9bag':12s} " + ' '.join(f'{rm(c9, m):9.3f}' for _, m in M))
# base4 안에서 eswf 몫만 올려본다 (닫힌 자유적합이 아니라 한 축 스칼라)
B = {'img_ridge': .075, 'img_gbdt': .18, 'omni': .19, 'eswf': .215}
flow = L('BAGsmall32')
hyb = np.mean([L(t) for t in [f'cnn7s71{i:02d}' for i in range(1, 11)] + ['cnn9s9101', 'cnn9s9102', 'cnn9s9103']], axis=0)
print('\neswf 몫만 올린 base4 (나머지 셋은 비례 축소), 카드2 배치 0.30/0.50/0.20:')
for e in (0.215, 0.30, 0.40, 0.50, 0.66):
    rest = {k: v for k, v in B.items() if k != 'eswf'}
    sr = sum(rest.values())
    b4 = e / 0.66 * L('eswf') + sum(v / sr * (0.66 - e) / 0.66 * L(k) for k, v in rest.items())
    p = 0.30 * b4 + 0.50 * flow + 0.20 * hyb
    print(f'  eswf={e:.3f} ' + ' '.join(f'{rm(p, m):9.4f}' for _, m in M))
