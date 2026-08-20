"""세 쌍의 Δ 를 '변화 크기'(|target-lw|) 구간으로 쪼갠다. lw 구간과 어느 쪽이 Public 을 설명하나."""
import pathlib
import numpy as np, pandas as pd
P = pathlib.Path('/home/jovyan/work/preds')
ROOT = pathlib.Path('/home/jovyan/public_dataset/competition_dataset_6h')
T = [f'target_{i:02d}' for i in range(12)]
Y = pd.read_csv(ROOT / 'validation/targets.csv')[T].to_numpy(np.float64)
lw = pd.read_csv(ROOT / 'validation/inputs.csv')['wind_19'].to_numpy(np.float64)[:, None]
L = lambda t: np.load(P / f'val_{t}.npy')
B = {'img_ridge': .075, 'img_gbdt': .18, 'omni': .19, 'eswf': .215, 'cnn3b': .113, 'cnn3c': .113, 'cnn3d': .114}
e039 = sum(v * L(k) for k, v in B.items()); flow = L('BAGsmall32')
base4 = sum(v / 0.66 * L(k) for k, v in B.items() if k in ('img_ridge', 'img_gbdt', 'omni', 'eswf'))
n50 = 0.5 * base4 + 0.5 * flow; mx = 0.8 * e039 + 0.2 * flow
hyb = np.mean([L(t) for t in [f'cnn7s71{i:02d}' for i in range(1, 11)] + ['cnn9s9101', 'cnn9s9102', 'cnn9s9103']], axis=0)
card1 = 0.40 * base4 + 0.40 * flow + 0.20 * hyb
card2 = 0.30 * base4 + 0.50 * flow + 0.20 * hyb
chg = np.abs(Y - lw)                      # 셀 단위 변화 크기
q = np.percentile(chg, [60, 80, 95])
bins = [('|Δ|<p60', chg < q[0]), ('p60-80', (chg >= q[0]) & (chg < q[1])),
        ('p80-95', (chg >= q[1]) & (chg < q[2])), ('상위5%', chg >= q[2])]
rm = lambda p, m: float(np.sqrt(np.mean((p[m] - Y[m]) ** 2)))
pairs = [('E039->MAX', e039, mx, -0.5438), ('MAX->N50', mx, n50, -0.4445),
         ('N50->CARD1', n50, card1, -0.0695), ('CARD1->CARD2', card1, card2, None)]
print(f"{'쌍':14s} {'public':>8s} " + ' '.join(f'{n:>9s}' for n, _ in bins) + f"  {'SSE몫':>8s}")
for name, a, b, pub in pairs:
    line = ' '.join(f'{rm(b, m) - rm(a, m):+9.4f}' for _, m in bins)
    print(f'{name:14s} {("%+.4f" % pub) if pub else "     ?":>8s} {line}')
tot = ((card1 - Y) ** 2).sum()
print('\n각 구간의 SSE 몫(card1): ' + ' '.join(f'{n}={((card1-Y)**2)[m].sum()/tot:.1%}' for n, m in bins))
print('셀 수: ' + ' '.join(f'{n}={int(m.sum())}' for n, m in bins))
print('\n성분별 상위5% 구간 RMSE:')
for t in ('BAGsmall32', 'omni', 'eswf', 'img_gbdt', 'cnn3d', 'phys'):
    if (P / f'val_{t}.npy').exists():
        print(f'  {t:12s} {rm(L(t), bins[3][1]):8.2f}')
c9 = np.mean([L(x.name[4:-4]) for x in P.glob('val_cnn9s*.npy')], axis=0)
c7 = np.mean([L(f'cnn7s71{i:02d}') for i in range(1, 11)], axis=0)
for nm, p in (('cnn9bag', c9), ('cnn7bag', c7), ('card1', card1), ('card2', card2)):
    print(f'  {nm:12s} {rm(p, bins[3][1]):8.2f}')
