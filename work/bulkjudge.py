"""bulk 구간에서 연결요소 짝지어 카드 후보를 판정한다 (paired.py 의 구간판). 카드2(55.5866)를 고른 도구.

  python work/bulkjudge.py

card1 기준으로 광류 몫 램프를 전체/bulk 두 지표로 낸다. 전체 val 은 카드2 쌍에서 부호가 틀렸다 —
bulk Δ 만 Public 과 부호가 맞는다 (전이비 1.12).
"""
import sys, pathlib, itertools
import numpy as np, pandas as pd
sys.path.insert(0, '/home/jovyan/work')
from noise_floor import components
P = pathlib.Path('/home/jovyan/work/preds')
ROOT = pathlib.Path('/home/jovyan/public_dataset/competition_dataset_6h')
T = [f'target_{i:02d}' for i in range(12)]; WCOL = [f'wind_{i:02d}' for i in range(20)]
va = pd.read_csv(ROOT/'validation/inputs.csv')
Y = pd.read_csv(ROOT/'validation/targets.csv')[T].to_numpy(np.float64)
lw = va['wind_19'].to_numpy(np.float64)[:,None]
L = lambda t: np.load(P/f'val_{t}.npy')
B = {'img_ridge': .075, 'img_gbdt': .18, 'omni': .19, 'eswf': .215}
base4 = sum(v/0.66*L(k) for k,v in B.items()); flow = L('BAGsmall32')
hyb = np.mean([L(t) for t in [f'cnn7s71{i:02d}' for i in range(1,11)]+['cnn9s9101','cnn9s9102','cnn9s9103']], axis=0)
mk = lambda f: (0.80-f)*base4 + f*flow + 0.20*hyb
card1 = mk(0.40)
cands = {'card2 flow.50': mk(0.50), 'flow.55': mk(0.55), 'flow.60': mk(0.60)}
lab = components(np.round(va[WCOL].to_numpy(np.float64), 4))
rows = [np.flatnonzero(lab == c) for c in range(lab.max()+1)]
chg = np.abs(Y-lw); q60 = np.percentile(chg, 60); bulk = chg < q60
rng = np.random.default_rng(0)
def rm(p, ix, mask):
    m = mask[ix]
    return float(np.sqrt(np.mean((Y[ix][m]-p[ix][m])**2)))
allix = np.arange(len(va))
print(f'연결요소 {len(rows)}개, 행수 {[len(r) for r in rows]}')
for name, C in cands.items():
    for tag, mask in [('전체', chg >= 0), ('bulk', bulk)]:
        d = np.array([rm(C, ix, mask) - rm(card1, ix, mask) for ix in rows])
        full = rm(C, allix, mask) - rm(card1, allix, mask)
        combos = list(itertools.combinations(range(len(rows)), len(rows)//2))
        half = np.array([rm(C, np.concatenate([rows[j] for j in c]), mask) -
                         rm(card1, np.concatenate([rows[j] for j in c]), mask) for c in combos])
        boot = []
        for _ in range(2000):
            pick, tot = [], 0
            while tot < len(va):
                j = rng.integers(len(rows)); pick.append(rows[j]); tot += len(rows[j])
            ix = np.concatenate(pick)
            boot.append(rm(C, ix, mask) - rm(card1, ix, mask))
        boot = np.array(boot)
        print(f'{name:14s} {tag:5s} Δ={full:+.4f} 이득 {int((d<0).sum())}/{len(d)} 중앙 {np.median(d):+.4f} '
              f'절반σ={half.std():.4f} boot σ={boot.std():.4f} CI95=[{np.percentile(boot,2.5):+.3f},{np.percentile(boot,97.5):+.3f}] '
              f'P(Δ<0)={float((boot<0).mean()):.2f}')
    print()
