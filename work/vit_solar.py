"""Random-init spatial ViT on the latest 193/211 EUV frame plus wind history."""
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
TAG = sys.argv[1] if len(sys.argv) > 1 else "vitsolar"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 20101
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 80
PX = 128
PATCH = 8
IMG = [f"image_{i:02d}" for i in range(20)]
W = [f"wind_{i:02d}" for i in range(20)]
T = [f"target_{i:02d}" for i in range(12)]
DEV = torch.device("cuda")
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True
torch.cuda.set_per_process_memory_fraction(0.45)
torch.set_num_threads(2)


def cache(split, inputs):
    path = WORK / "resized_cache" / f"{PX}px" / f"{split}_images.npy"
    names = sorted(pd.unique(inputs[IMG].to_numpy().ravel()).tolist())
    arr = np.load(path)
    assert arr.shape == (len(names), 2, PX, PX)
    idx = {v: i for i, v in enumerate(names)}
    slots = torch.tensor([idx[v] for v in inputs[IMG[-1]]], device=DEV)
    return torch.from_numpy(arr).to(DEV), slots


def wind_feats(inputs):
    w = inputs[W].to_numpy(np.float32)
    d = np.diff(w, axis=1, prepend=w[:, :1])
    return np.concatenate([(w - 400.0) / 100.0, d / 50.0], axis=1)


class Net(nn.Module):
    def __init__(self, dim=192, depth=4, heads=6):
        super().__init__()
        self.patch = nn.Conv2d(2, dim, PATCH, PATCH)
        n = (PX // PATCH) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=2 * dim, dropout=0.1,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, norm=nn.LayerNorm(dim))
        self.wind = nn.Sequential(nn.Linear(40, 128), nn.SiLU(inplace=True), nn.Linear(128, 128))
        self.head = nn.Sequential(nn.Linear(dim + 128, 256), nn.SiLU(inplace=True),
                                  nn.Dropout(0.25), nn.Linear(256, 12))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, image, wind):
        x = self.patch(image).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1) + self.pos
        x = self.encoder(x)[:, 0]
        return self.head(torch.cat([x, self.wind(wind)], 1))


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)))


xs = {s: pd.read_csv(ROOT / s / "inputs.csv") for s in ("train", "validation", "test")}
tr_y = pd.read_csv(ROOT / "train/targets.csv")[T].to_numpy(np.float32)
va_y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float32)
data = {s: cache(s, xs[s]) for s in xs}
yy, xx = torch.meshgrid(torch.arange(PX, device=DEV), torch.arange(PX, device=DEV), indexing="ij")
disk = ((yy - PX / 2 + 0.5) ** 2 + (xx - PX / 2 + 0.5) ** 2) <= (0.47 * PX) ** 2
for s, (arr, slots) in data.items():
    a = arr.float(); med = a[:, :, disk].median(-1).values[:, :, None, None]
    data[s] = ((a / med.clamp_min(1.0) - 1.0) / 0.35, slots)
wf = {s: torch.from_numpy(wind_feats(xs[s])).to(DEV) for s in xs}
last = {s: torch.from_numpy(xs[s][W[-1]].to_numpy(np.float32)).to(DEV) for s in xs}
ytr = torch.from_numpy(tr_y).to(DEV)
model = Net().to(DEV)
print("params", sum(p.numel() for p in model.parameters()), flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
BS = 64
steps = math.ceil(len(xs["train"]) / BS)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 5e-4, total_steps=EPOCHS * steps, pct_start=0.2)


def batch(split, rows, train=False):
    arr, slots = data[split]
    im = arr[slots[rows]]
    if train:
        if torch.rand(()) < 0.5: im = torch.flip(im, [-2])
        im = im * (1 + 0.08 * torch.randn(len(rows), 1, 1, 1, device=DEV))
    return im, wf[split][rows]


@torch.no_grad()
def infer(split):
    model.eval(); out = []
    for i in range(0, len(xs[split]), 128):
        rows = torch.arange(i, min(i + 128, len(xs[split])), device=DEV)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            im, w = batch(split, rows)
            pred = (model(im, w).float() + model(torch.flip(im, [-2]), w).float()) / 2
            out.append(pred + last[split][rows, None])
    return torch.cat(out)


best, state, bad = 1e9, None, 0
for ep in range(1, EPOCHS + 1):
    model.train(); perm = torch.randperm(len(xs["train"]), device=DEV); total = 0.0; t0 = time.perf_counter()
    for i in range(0, len(perm), BS):
        rows = perm[i:i + BS]; opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(*batch("train", rows, True)) + last["train"][rows, None]
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
