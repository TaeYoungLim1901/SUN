import numpy as np, pathlib, subprocess
P = pathlib.Path('/home/jovyan/work/preds')
tags = [f'cnn7s71{i:02d}' for i in range(1, 11)] + ['cnn9s9101', 'cnn9s9102', 'cnn9s9103']
n50 = np.load(P / 'val_N50.npy')
bag = np.mean([np.load(P / f'val_{t}.npy') for t in tags], axis=0)
np.save(P / 'val_CARD1.npy', 0.8 * n50 + 0.2 * bag)
print(subprocess.run(['bash', '-lc', 'cd /home/jovyan/work && python paired.py preds/val_N50.npy preds/val_CARD1.npy N50 CARD1 2>&1 | tail -7'], capture_output=True, text=True).stdout)
print(subprocess.run(['bash', '-lc', 'cd /home/jovyan/work && python add_honest.py --base N50 ' + ' '.join(tags) + ' 2>&1 | tail -6'], capture_output=True, text=True).stdout)
