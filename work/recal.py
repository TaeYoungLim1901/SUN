"""카드1 채점(55.7603)을 넣어 전이비를 다시 적합한다."""
import pathlib
import numpy as np, pandas as pd
P = pathlib.Path('/home/jovyan/work/preds')
ROOT = pathlib.Path('/home/jovyan/public_dataset/competition_dataset_6h')
T = [f'target_{i:02d}' for i in range(12)]
Y = pd.read_csv(ROOT / 'validation/targets.csv')[T].to_numpy(np.float64)
va = pd.read_csv(ROOT / 'validation/inputs.csv')['wind_19'].to_numpy(np.float64)
te = pd.read_csv(ROOT / 'test/inputs.csv')['wind_19'].to_numpy(np.float64)
edges = np.percentile(np.r_[va, te], np.linspace(0, 100, 21)); edges[0] -= 1; edges[-1] += 1
cv, ct = np.histogram(va, edges)[0] + 1.0, np.histogram(te, edges)[0] + 1.0
w = ((ct / ct.sum()) / (cv / cv.sum()))[np.clip(np.digitize(va, edges) - 1, 0, 19)]
w = np.clip(w, 0.2, 5.0); w /= w.mean()
L = lambda t: np.load(P / f'val_{t}.npy')
rm = lambda p: float(np.sqrt(np.mean((p - Y) ** 2)))
wrm = lambda p: float(np.sqrt((w * ((p - Y) ** 2).mean(1)).sum() / w.sum()))
B = {'img_ridge': .075, 'img_gbdt': .18, 'omni': .19, 'eswf': .215, 'cnn3b': .113, 'cnn3c': .113, 'cnn3d': .114}
e039 = sum(v * L(k) for k, v in B.items())
bag = L('BAGsmall32')
base4 = sum(v / 0.66 * L(k) for k, v in B.items() if k in ('img_ridge', 'img_gbdt', 'omni', 'eswf'))
n50 = 0.5 * base4 + 0.5 * bag
card1 = L('CARD1')
cards = {'E039': (e039, 56.8181), 'MAX': (0.8 * e039 + 0.2 * bag, 56.2743),
         'N50': (n50, 55.8298), 'CARD1': (card1, 55.7603)}
for k, (p, lb) in cards.items():
    print(f'{k:6s} val={rm(p):.4f} wval={wrm(p):.4f} public={lb}')
seq = ['E039', 'MAX', 'N50', 'CARD1']
print('\n쌍 (연속):')
for a, b in zip(seq, seq[1:]):
    pa, la = cards[a]; pb, lb = cards[b]
    dl, dw, dp = rm(pb) - rm(pa), wrm(pb) - wrm(pa), lb - la
    print(f'{a}->{b}: public={dp:+.4f} 무가중={dl:+.4f}(비 {dp/dl:.2f}) 가중={dw:+.4f}(비 {dp/dw:.2f})')
cv_ = np.array([rm(p) for p, _ in cards.values()]); lb_ = np.array([l for _, l in cards.values()])
wv_ = np.array([wrm(p) for p, _ in cards.values()])
for name, x in (('무가중', cv_), ('가중', wv_)):
    a, b = np.polyfit(x, lb_, 1)
    r = lb_ - (a * x + b)
    print(f'{name} 적합 a={a:.3f} b={b:.3f} resid σ={r.std(ddof=1):.3f} 채택선 σ/a={r.std(ddof=1)/a:.3f}')
