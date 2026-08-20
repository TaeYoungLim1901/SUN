"""족별 기여를 구간별로 본다. 지금까지는 전체 val 하나로만 봤다."""
import pathlib, glob
import numpy as np, pandas as pd
P = pathlib.Path('/home/jovyan/work/preds')
ROOT = pathlib.Path('/home/jovyan/public_dataset/competition_dataset_6h')
T = [f'target_{i:02d}' for i in range(12)]
Y = pd.read_csv(ROOT / 'validation/targets.csv')[T].to_numpy(np.float64)
lw = pd.read_csv(ROOT / 'validation/inputs.csv')['wind_19'].to_numpy(np.float64)
L = lambda t: np.load(P / f'val_{t}.npy')
B = {'img_ridge': .075, 'img_gbdt': .18, 'omni': .19, 'eswf': .215}
base4 = sum(v / 0.66 * L(k) for k, v in B.items()); bag = L('BAGsmall32')
card1 = 0.40 * base4 + 0.40 * bag + 0.20 * np.mean(
    [L(t) for t in [f'cnn7s71{i:02d}' for i in range(1, 11)] + ['cnn9s9101', 'cnn9s9102', 'cnn9s9103']], axis=0)
fams = {'cnn7': sorted(x[len(str(P))+5:-4] for x in glob.glob(str(P/'val_cnn7s*.npy'))),
        'cnn9': sorted(x[len(str(P))+5:-4] for x in glob.glob(str(P/'val_cnn9s*.npy'))),
        'cnn7q': sorted(x[len(str(P))+5:-4] for x in glob.glob(str(P/'val_cnn7q*.npy'))),
        'cnn7z': sorted(x[len(str(P))+5:-4] for x in glob.glob(str(P/'val_cnn7z*.npy'))),
        'phys': ['phys'], 'omni2': ['omni2'], 'flowbag': None}
rm = lambda p, m: float(np.sqrt(np.mean((p[m] - Y[m]) ** 2)))
M = (('전체', np.ones(len(Y), bool)), ('lw<450', lw < 450), ('lw>=450', lw >= 450), ('lw>=550', lw >= 550))
print(f"{'족':8s} {'n':>3s} " + ' '.join(f'{n:>10s}' for n, _ in M) + '   (카드1에 w=0.10 추가시 Δ)')
for name, tags in fams.items():
    if tags is None:
        new = bag
    else:
        tags = [t for t in tags if (P / f'val_{t}.npy').exists()]
        if not tags:
            continue
        new = np.mean([L(t) for t in tags], axis=0)
    p = 0.9 * card1 + 0.1 * new
    print(f'{name:8s} {len(tags) if tags else 32:3d} ' +
          ' '.join(f'{rm(p, m) - rm(card1, m):+10.4f}' for _, m in M))
