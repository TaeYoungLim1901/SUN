"""cnn7 을 빼고 cnn9·광류 위주로 짠 후보들을 구간별로 판정한다."""
import pathlib, sys
import numpy as np, pandas as pd
sys.path.insert(0, '/home/jovyan/work')
from noise_floor import components
P = pathlib.Path('/home/jovyan/work/preds')
ROOT = pathlib.Path('/home/jovyan/public_dataset/competition_dataset_6h')
T = [f'target_{i:02d}' for i in range(12)]
WCOL = [f'wind_{i:02d}' for i in range(20)]
va = pd.read_csv(ROOT / 'validation/inputs.csv')
Y = pd.read_csv(ROOT / 'validation/targets.csv')[T].to_numpy(np.float64)
lw = va['wind_19'].to_numpy(np.float64)
L = lambda t: np.load(P / f'val_{t}.npy')
B = {'img_ridge': .075, 'img_gbdt': .18, 'omni': .19, 'eswf': .215}
base4 = sum(v / 0.66 * L(k) for k, v in B.items()); flow = L('BAGsmall32')
c7 = np.mean([L(f'cnn7s71{i:02d}') for i in range(1, 11)], axis=0)
c9 = np.mean([L(t) for t in sorted(x.name[4:-4] for x in P.glob('val_cnn9s*.npy'))], axis=0)
hyb13 = np.mean([L(t) for t in [f'cnn7s71{i:02d}' for i in range(1, 11)] + ['cnn9s9101', 'cnn9s9102', 'cnn9s9103']], axis=0)
card1 = 0.40 * base4 + 0.40 * flow + 0.20 * hyb13
cands = {
    'card1(제출됨)': card1,
    'card2 0.30/0.50/hyb0.20': 0.30 * base4 + 0.50 * flow + 0.20 * hyb13,
    'A 0.30/0.50/c9 0.20': 0.30 * base4 + 0.50 * flow + 0.20 * c9,
    'B 0.35/0.45/c9 0.20': 0.35 * base4 + 0.45 * flow + 0.20 * c9,
    'C 0.30/0.55/c9 0.15': 0.30 * base4 + 0.55 * flow + 0.15 * c9,
    'D 0.35/0.50/c9 0.15': 0.35 * base4 + 0.50 * flow + 0.15 * c9,
    'E 0.30/0.45/c9 0.15/c7 0.10': 0.30 * base4 + 0.45 * flow + 0.15 * c9 + 0.10 * c7,
    'F 0.40/0.60(광류만)': 0.40 * base4 + 0.60 * flow,
}
lab = components(np.round(va[WCOL].to_numpy(np.float64), 4))
comps = [np.flatnonzero(lab == c) for c in range(lab.max() + 1)]
rng = np.random.default_rng(0)
rm = lambda p, ix: float(np.sqrt(np.mean((p[ix] - Y[ix]) ** 2)))
print(f"{'후보':30s} {'전체':>8s} {'lw<450':>8s} {'lw>=450':>8s} {'lw>=550':>8s} {'fast CI상한':>11s} {'fast이득':>8s}")
for name, p in cands.items():
    ix = np.flatnonzero(lw >= 450)
    rows = [r for r in (np.intersect1d(c, ix) for c in comps) if len(r) > 5]
    boot = []
    for _ in range(1000):
        pick = np.concatenate([rows[rng.integers(len(rows))] for _ in rows])   # 같은 표본으로 둘 다 잰다
        boot.append(rm(p, pick) - rm(card1, pick))
    boot = np.array(boot)
    print(f'{name:30s} {rm(p, np.arange(len(Y))):8.4f} {rm(p, lw < 450):8.4f} {rm(p, lw >= 450):8.4f} '
          f'{rm(p, lw >= 550):8.4f} {np.percentile(boot, 97.5):+11.3f} {(boot < 0).mean():8.0%}')
