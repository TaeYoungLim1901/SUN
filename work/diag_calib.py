"""val 에서 '보정 상한' 을 잰다. 채택 아님 — 잠재 이득 크기만 본다."""
import numpy as np, pandas as pd, pathlib, json
ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
P = pathlib.Path("/home/jovyan/work/preds")
T = [f"target_{i:02d}" for i in range(12)]
y = pd.read_csv(ROOT/"validation/targets.csv")[T].to_numpy(np.float64)
xin = pd.read_csv(ROOT/"validation/inputs.csv")
lw = xin[[f"wind_{i:02d}" for i in range(20)]].to_numpy(np.float64)[:, -1]

W = dict(img_ridge=0.125, img_gbdt=0.25, omni=0.125, eswf=0.25, cnn96b=0.125, cnn96w3=0.125)
def load(n):
    return np.load(P/f"val_{n}.npy").astype(np.float64)
blend = sum(w*load(n) for n, w in W.items())
rmse = lambda a: float(np.sqrt(np.mean((a-y)**2)))
print("blend", round(rmse(blend), 4))

# 1) horizon 별 affine  y = a*p + b   (val 적합 = 상한 측정용)
cal = blend.copy()
A, B = [], []
for h in range(12):
    a, b = np.polyfit(blend[:, h], y[:, h], 1)
    A.append(a); B.append(b); cal[:, h] = a*blend[:, h] + b
print("affine(h)", round(rmse(cal), 4), "a=", np.round(A, 3).tolist())

# 2) 전역 분산 확장  p -> m + s*(p-m)
m = blend.mean()
best = min(((rmse(m + s*(blend-m)), s) for s in np.arange(1.0, 1.61, 0.05)))
print("global scale", round(best[0], 4), "s=", round(best[1], 3))

# 3) 분위수 매핑 (horizon 별, val 자기 분포로 — 상한)
qm = blend.copy()
for h in range(12):
    o = np.argsort(blend[:, h]); qm[o, h] = np.sort(y[:, h])
print("quantile map", round(rmse(qm), 4))

# 4) 구간별 편향
for lo, hi in [(0, 400), (400, 500), (500, 600), (600, 700), (700, 2000)]:
    m_ = (y >= lo) & (y < hi)
    if m_.sum():
        print(f"  y[{lo},{hi}) n={m_.sum():6d} bias={np.mean(blend[m_]-y[m_]):8.2f} rmse={np.sqrt(np.mean((blend[m_]-y[m_])**2)):7.2f} share={np.sum((blend[m_]-y[m_])**2)/np.sum((blend-y)**2)*100:5.1f}%")

# 5) 2D 보정 상한: (lw, pred) 로 y 예측하는 작은 GBDT — 잠재 신호 확인
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
oof = np.zeros_like(blend)
for h in range(12):
    X = np.c_[blend[:, h], lw, blend.mean(1), blend.max(1), blend[:, h]-lw]
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        g = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06)
        g.fit(X[tr], y[tr, h]); oof[te, h] = g.predict(X[te])
print("2d-gbdt oof(val,shuffle)", round(rmse(oof), 4))
