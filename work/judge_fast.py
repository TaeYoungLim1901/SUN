"""고속 구간(lw>=450)에서 짝지은 판정. 성분 단위로 부트스트랩해 카드2가 카드1보다 나은지 본다."""
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
base4 = sum(v / 0.66 * L(k) for k, v in B.items()); bag = L('BAGsmall32')
hyb = np.mean([L(t) for t in [f'cnn7s71{i:02d}' for i in range(1, 11)] + ['cnn9s9101', 'cnn9s9102', 'cnn9s9103']], axis=0)
card1 = 0.40 * base4 + 0.40 * bag + 0.20 * hyb
card2 = 0.30 * base4 + 0.50 * bag + 0.20 * hyb
card3 = 0.25 * base4 + 0.55 * bag + 0.20 * hyb
lab = components(np.round(va[WCOL].to_numpy(np.float64), 4))
comps = [np.flatnonzero(lab == c) for c in range(lab.max() + 1)]
rng = np.random.default_rng(0)
rm = lambda p, ix: float(np.sqrt(np.mean((p[ix] - Y[ix]) ** 2)))
for name, cand in (('card2(0.30/0.50/0.20)', card2), ('card3(0.25/0.55/0.20)', card3)):
    print(f'--- {name} vs card1 ---')
    for mname, mask in (('전체', np.ones(len(va), bool)), ('lw>=450', lw >= 450), ('lw>=550', lw >= 550)):
        ix = np.flatnonzero(mask)
        d = rm(cand, ix) - rm(card1, ix)
        rows = [np.intersect1d(c, ix) for c in comps]
        rows = [r for r in rows if len(r) > 5]
        dc = np.array([rm(cand, r) - rm(card1, r) for r in rows])
        boot = []
        for _ in range(2000):
            pick = np.concatenate([rows[rng.integers(len(rows))] for _ in range(len(rows))])
            boot.append(rm(cand, pick) - rm(card1, pick))
        boot = np.array(boot)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f'  {mname:8s} Δ={d:+.4f} 성분이득 {int((dc<0).sum())}/{len(rows)} '
              f'σ={boot.std():.4f} CI[{lo:+.3f},{hi:+.3f}] 이득 {(boot<0).mean():.0%}')
