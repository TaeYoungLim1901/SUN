"""느린 풍속 구간(test 쪽 분포)에서 각 축의 Δ 를 따로 본다. 전이비가 축마다 다른 이유를 찾는다."""
import pathlib
import numpy as np, pandas as pd
P = pathlib.Path('/home/jovyan/work/preds')
ROOT = pathlib.Path('/home/jovyan/public_dataset/competition_dataset_6h')
T = [f'target_{i:02d}' for i in range(12)]
Y = pd.read_csv(ROOT / 'validation/targets.csv')[T].to_numpy(np.float64)
lw = pd.read_csv(ROOT / 'validation/inputs.csv')['wind_19'].to_numpy(np.float64)
L = lambda t: np.load(P / f'val_{t}.npy')
B = {'img_ridge': .075, 'img_gbdt': .18, 'omni': .19, 'eswf': .215, 'cnn3b': .113, 'cnn3c': .113, 'cnn3d': .114}
e039 = sum(v * L(k) for k, v in B.items())
bag = L('BAGsmall32')
base4 = sum(v / 0.66 * L(k) for k, v in B.items() if k in ('img_ridge', 'img_gbdt', 'omni', 'eswf'))
n50 = 0.5 * base4 + 0.5 * bag
card1 = L('CARD1')
mx = 0.8 * e039 + 0.2 * bag
rm = lambda p, m: float(np.sqrt(np.mean((p[m] - Y[m]) ** 2)))
bins = [('전체', lw > 0), ('lw<350', lw < 350), ('350-450', (lw >= 350) & (lw < 450)),
        ('450-550', (lw >= 450) & (lw < 550)), ('lw>=550', lw >= 550)]
pairs = [('E039->MAX', e039, mx, -0.5438), ('MAX->N50', mx, n50, -0.4445), ('N50->CARD1', n50, card1, -0.0695)]
print(f"{'쌍':12s} {'public':>8s} " + ' '.join(f'{n:>9s}' for n, _ in bins))
for name, a, b, pub in pairs:
    print(f'{name:12s} {pub:+8.4f} ' + ' '.join(f'{rm(b,m)-rm(a,m):+9.4f}' for _, m in bins))
print('\n구간별 행수: ' + ' '.join(f'{n}={int(m.sum())}' for n, m in bins))
# flow 몫 곡선 (느린 구간 따로)
print('\nflow 몫 곡선 (base4 나머지):')
for f in (0.40, 0.50, 0.55, 0.60, 0.70):
    p = (1 - f) * base4 + f * bag
    print(f'  f={f:.2f} 전체={rm(p, lw>0):.4f} lw<450={rm(p, lw<450):.4f} lw>=450={rm(p, lw>=450):.4f}')
