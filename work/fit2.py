"""Public Δ 를 (느린 구간 Δ, 빠른 구간 Δ) 두 항으로 적합하고 후보 카드를 예측한다."""
import pathlib
import numpy as np, pandas as pd
P = pathlib.Path('/home/jovyan/work/preds')
ROOT = pathlib.Path('/home/jovyan/public_dataset/competition_dataset_6h')
T = [f'target_{i:02d}' for i in range(12)]
Y = pd.read_csv(ROOT / 'validation/targets.csv')[T].to_numpy(np.float64)
lw = pd.read_csv(ROOT / 'validation/inputs.csv')['wind_19'].to_numpy(np.float64)
L = lambda t: np.load(P / f'val_{t}.npy')
B = {'img_ridge': .075, 'img_gbdt': .18, 'omni': .19, 'eswf': .215, 'cnn3b': .113, 'cnn3c': .113, 'cnn3d': .114}
e039 = sum(v * L(k) for k, v in B.items()); bag = L('BAGsmall32')
base4 = sum(v / 0.66 * L(k) for k, v in B.items() if k in ('img_ridge', 'img_gbdt', 'omni', 'eswf'))
n50 = 0.5 * base4 + 0.5 * bag; mx = 0.8 * e039 + 0.2 * bag; card1 = L('CARD1')
hyb = np.mean([L(t) for t in [f'cnn7s71{i:02d}' for i in range(1, 11)] + ['cnn9s9101', 'cnn9s9102', 'cnn9s9103']], axis=0)
S, F = lw < 450, lw >= 450
rm = lambda p, m: float(np.sqrt(np.mean((p[m] - Y[m]) ** 2)))
pairs = [(e039, mx, -0.5438), (mx, n50, -0.4445), (n50, card1, -0.0695)]
A = np.array([[rm(b, S) - rm(a, S), rm(b, F) - rm(a, F)] for a, b, _ in pairs])
y = np.array([p for _, _, p in pairs])
coef, *_ = np.linalg.lstsq(A, y, rcond=None)
print('적합: publicΔ = %.3f·slowΔ + %.3f·fastΔ' % tuple(coef))
print('잔차', np.round(y - A @ coef, 4))
cands = {'N50(현행-1)': n50, 'CARD1(제출됨)': card1}
for f in (0.55, 0.60, 0.65, 0.70):
    cands[f'flow{f:.2f}'] = (1 - f) * base4 + f * bag
for f, h in ((0.55, 0.10), (0.60, 0.10), (0.55, 0.15), (0.50, 0.20)):
    cands[f'flow{f:.2f}+hyb{h:.2f}'] = (1 - f - h) * base4 + f * bag + h * hyb
print(f"\n{'카드':20s} {'전체val':>8s} {'slow':>8s} {'fast':>8s} {'예상 publicΔ(vs CARD1)':>22s}")
for k, p in cands.items():
    ds, df = rm(p, S) - rm(card1, S), rm(p, F) - rm(card1, F)
    print(f'{k:20s} {rm(p, lw>0):8.4f} {rm(p, S):8.4f} {rm(p, F):8.4f} {coef[0]*ds + coef[1]*df:+22.4f}')
