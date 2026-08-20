"""cnn7 시드가 쓴 train 통계(스칼라 표준화 · 광류 std)를 복원해 저장한다. 학습 코드와 같은 경로로 다시 계산한다."""
import pathlib, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, "/home/jovyan/work")
from feats import sample_feats

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
WORK = pathlib.Path("/home/jovyan/work")
tr = pd.read_csv(ROOT / "train/inputs.csv")
S = sample_feats("train", tr).astype(np.float32)
smu, ssd = S.mean(0), S.std(0)
ssd[ssd < 1e-6] = 1.0
farr = np.load(WORK / "flowfield_cache/train_96px.npy", mmap_mode="r")
# 학습은 캐시를 GPU 로 올린 뒤 앞 2000쌍의 std 를 쟀다. 같은 순서·같은 개수.
fstd = float(torch.from_numpy(np.asarray(farr[:2000], np.float16)).float().std().item())
np.savez(WORK / "v7_stats.npz", smu=smu, ssd=ssd, fstd=np.float64(fstd))
print("dim", S.shape, "fstd", fstd)
print("smu[:3]", smu[:3], "ssd[:3]", ssd[:3])
