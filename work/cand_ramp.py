"""후보 카드들을 '변화 크기' 구간으로 판정한다. 카드1 대비 Δ."""
import pathlib
import numpy as np, pandas as pd
P = pathlib.Path('/home/jovyan/work/preds')
ROOT = pathlib.Path('/home/jovyan/public_dataset/competition_dataset_6h')
T = [f'target_{i:02d}' for i in range(12)]
Y = pd.read_csv(ROOT / 'validation/targets.csv')[T].to_numpy(np.float64)
lw = pd.read_csv(ROOT / 'validation/inputs.csv')['wind_19'].to_numpy(np.float64)[:, None]
L = lambda t: np.load(P / f'val_{t}.npy')
B = {'img_ridge': .075, 'img_gbdt': .18, 'omni': .19, 'eswf': .215}
base4 = sum(v / 0.66 * L(k) for k, v in B.items()); flow = L('BAGsmall32')
c9 = np.mean([L(x.name[4:-4]) for x in P.glob('val_cnn9s*.npy')], axis=0)
c7 = np.mean([L(f'cnn7s71{i:02d}') for i in range(1, 11)], axis=0)
hyb = np.mean([L(t) for t in [f'cnn7s71{i:02d}' for i in range(1, 11)] + ['cnn9s9101', 'cnn9s9102', 'cnn9s9103']], axis=0)
card1 = 0.40 * base4 + 0.40 * flow + 0.20 * hyb
cands = {
    'card2 0.30/0.50/hyb0.20': 0.30 * base4 + 0.50 * flow + 0.20 * hyb,
    'A 0.30/0.50/c9 0.20': 0.30 * base4 + 0.50 * flow + 0.20 * c9,
    'E 0.30/0.45/c9.15/c7.10': 0.30 * base4 + 0.45 * flow + 0.15 * c9 + 0.10 * c7,
    'G 0.35/0.35/c7.15/c9.15': 0.35 * base4 + 0.35 * flow + 0.15 * c7 + 0.15 * c9,
    'H 0.30/0.40/c7.15/c9.15': 0.30 * base4 + 0.40 * flow + 0.15 * c7 + 0.15 * c9,
    'I 0.35/0.40/c7.10/c9.15': 0.35 * base4 + 0.40 * flow + 0.10 * c7 + 0.15 * c9,
}
chg = np.abs(Y - lw); q = np.percentile(chg, [60, 80, 95])
bins = [('bulk|Δ|<p60', chg < q[0]), ('p60-80', (chg >= q[0]) & (chg < q[1])),
        ('p80-95', (chg >= q[1]) & (chg < q[2])), ('상위5%', chg >= q[2])]
rm = lambda p, m: float(np.sqrt(np.mean((p[m] - Y[m]) ** 2)))
print(f"{'후보':26s} {'전체':>8s} " + ' '.join(f'{n:>11s}' for n, _ in bins))
print(f"{'card1(제출본) 절대값':26s} {rm(card1, chg >= 0):8.4f} " + ' '.join(f'{rm(card1, m):11.3f}' for _, m in bins))
for n, p in cands.items():
    print(f'{n:26s} {rm(p, chg>=0)-rm(card1, chg>=0):+8.4f} ' + ' '.join(f'{rm(p,m)-rm(card1,m):+11.4f}' for _, m in bins))
