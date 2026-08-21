"""Polar longitude-waterfall CNN from provided 193/211 EUV only."""
import json
import math
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
WORK = pathlib.Path("/home/jovyan/work")
OUT = WORK / "preds"
TAG = sys.argv[1] if len(sys.argv) > 1 else "polarcnn"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 19101
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 60
PX = 96
IMG = [f"image_{i:02d}" for i in range(20)]
W = [f"wind_{i:02d}" for i in range(20)]
T = [f"target_{i:02d}" for i in range(12)]
DEV = torch.device("cuda")
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True
torch.cuda.set_per_process_memory_fraction(0.45)
torch.set_num_threads(2)


def polar_bins():
    yy, xx = np.mgrid[:PX, :PX].astype(np.float32)
    r = 0.47 * PX
    dx, dy = xx - (PX / 2 - 0.5), yy - (PX / 2 - 0.5)
    rho = np.sqrt(dx * dx + dy * dy)
    disk = rho <= r
    lon = np.clip(((np.arctan2(dy, dx) + np.pi) / (2 * np.pi) * 36).astype(np.int16), 0, 35)
    rad = np.clip((rho / r * 6).astype(np.int16), 0, 5)
    bid = lon[disk] * 6 + rad[disk]
    counts = np.bincount(bid, minlength=216).astype(np.float32)
    assert counts.min() > 0
    return disk, bid, counts


def frame_maps(split, inputs):
    cache = WORK / "polar_map_cache"
    cache.mkdir(exist_ok=True)
    path = cache / f"{split}.npy"
    names = sorted(pd.unique(inputs[IMG].to_numpy().ravel()).tolist())
    arr = np.load(WORK / "resized_cache" / f"{PX}px" / f"{split}_images.npy", mmap_mode="r")
    assert arr.shape == (len(names), 2, PX, PX)
    if not path.exists():
        disk, bid, counts = polar_bins()
        out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                        shape=(len(names), 2, 36, 6))
        for i, frame in enumerate(arr):
            for c in range(2):
                x = frame[c]
                dark = x[disk] < 0.5 * np.median(x[disk])
                out[i, c] = (np.bincount(bid, weights=dark, minlength=216) / counts).reshape(36, 6)
        out.flush()
    maps = np.load(path)
    assert maps.shape == (len(names), 2, 36, 6)
    idx = {name: i for i, name in enumerate(names)}
    slots = np.stack([[idx[v] for v in inputs[col]] for col in IMG], axis=1)
    assert slots.shape == (len(inputs), 20)
    return torch.from_numpy(maps).to(DEV), torch.from_numpy(slots).long().to(DEV)


def wind_feats(inputs):
    w = inputs[W].to_numpy(np.float32)
    d = np.diff(w, axis=1, prepend=w[:, :1])
    return np.concatenate([(w - 400.0) / 100.0, d / 50.0], axis=1)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(2, 32, 3, padding=1, bias=False), nn.BatchNorm3d(32), nn.SiLU(inplace=True),
            nn.Conv3d(32, 64, 3, stride=(2, 2, 1), padding=1, bias=False), nn.BatchNorm3d(64), nn.SiLU(inplace=True),
            nn.Conv3d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm3d(128), nn.SiLU(inplace=True))
        self.wind = nn.Sequential(nn.Linear(40, 128), nn.SiLU(inplace=True), nn.Linear(128, 128))
        self.head = nn.Sequential(nn.Linear(128 * 5 * 9 * 3 + 128, 512), nn.SiLU(inplace=True),
                                  nn.Dropout(0.3), nn.Linear(512, 12))

    def forward(self, x, wind):
        z = self.conv(x).flatten(1)
        return self.head(torch.cat([z, self.wind(wind)], dim=1))


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)))


xs = {s: pd.read_csv(ROOT / s / "inputs.csv") for s in ("train", "validation", "test")}
tr_y = pd.read_csv(ROOT / "train/targets.csv")[T].to_numpy(np.float32)
va_y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float32)
data = {s: frame_maps(s, xs[s]) for s in xs}
wf = {s: torch.from_numpy(wind_feats(xs[s])).to(DEV) for s in xs}
last = {s: torch.from_numpy(xs[s][W[-1]].to_numpy(np.float32)).to(DEV) for s in xs}
ytr = torch.from_numpy(tr_y).to(DEV)
model = Net().to(DEV)
print("params", sum(p.numel() for p in model.parameters()), flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.05)
BS = 128
steps = math.ceil(len(xs["train"]) / BS)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 8e-4, total_steps=EPOCHS * steps, pct_start=0.25)


def batch(split, rows):
    maps, slots = data[split]
    return maps[slots[rows]].float().permute(0, 2, 1, 3, 4), wf[split][rows]


@torch.no_grad()
def infer(split):
    model.eval()
    out = []
    for i in range(0, len(xs[split]), 256):
        rows = torch.arange(i, min(i + 256, len(xs[split])), device=DEV)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(*batch(split, rows)).float() + last[split][rows, None])
    return torch.cat(out)


best, state, bad = 1e9, None, 0
for ep in range(1, EPOCHS + 1):
    model.train(); perm = torch.randperm(len(xs["train"]), device=DEV); total = 0.0; t0 = time.perf_counter()
    for i in range(0, len(perm), BS):
        rows = perm[i:i + BS]; opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(*batch("train", rows)) + last["train"][rows, None]
            loss = ((pred - ytr[rows]) ** 2).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        total += loss.item() * len(rows)
    v = rmse(va_y, infer("validation").cpu().numpy())
    print(f"ep{ep:03d} loss={total/len(perm):8.2f} val_rmse={v:7.3f} lr={sched.get_last_lr()[0]:.2e} {time.perf_counter()-t0:.1f}s", flush=True)
    if v < best - 1e-4:
        best, bad = v, 0; state = {k: p.detach().clone() for k, p in model.state_dict().items()}
    else:
        bad += 1
        if bad >= 12: break

model.load_state_dict(state)
vp = infer("validation").cpu().numpy(); tp = infer("test").cpu().numpy()
assert vp.shape == (1199, 12) and tp.shape == (3868, 12)
assert np.isfinite(vp).all() and np.isfinite(tp).all()
OUT.mkdir(exist_ok=True)
np.save(OUT / f"val_{TAG}.npy", vp); np.save(OUT / f"test_{TAG}.npy", tp)
torch.save(state, WORK / f"{TAG}.pth")
(OUT / f"{TAG}.json").write_text(json.dumps({"val_rmse": round(rmse(va_y, vp), 3), "seed": SEED}, indent=2))
print(f"{TAG} best val_rmse={best:.3f}", flush=True)
