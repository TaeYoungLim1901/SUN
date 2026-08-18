"""단일 이미지 사전학습. Encoder+MLP, GRU 없음. pretrained weight 미사용 (RULES.md §1 5항).

각 train 샘플에서 프레임 t∈0..19 를 무작위로 뽑아 그 한 장만 인코딩한다.
타깃은 cnn.py 와 같이 마지막 관측 대비 잔차 (pred + wind_19).
validation 은 early stopping·모델선택 전용. train 에 합치지 않는다 (RULES.md §1 7항).

사용: python work/pretrain.py
"""
import json
import math
import pathlib
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
WORK = pathlib.Path("/home/jovyan/work")
OUT = WORK / "preds"
PX = 96
TAG = "imgpt"
SEED = 781
EPOCHS = 30
IMG = [f"image_{i:02d}" for i in range(20)]
W = [f"wind_{i:02d}" for i in range(20)]
T = [f"target_{i:02d}" for i in range(12)]


class Encoder(nn.Module):
    """작은 residual CNN. 랜덤 초기화, 제공 데이터로만 학습."""

    def __init__(self, dim=128):
        super().__init__()

        def blk(i, o, s=2):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.SiLU(inplace=True),
                nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.SiLU(inplace=True))

        self.net = nn.Sequential(blk(2, 24), blk(24, 48), blk(48, 96), blk(96, 128),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, dim))

    def forward(self, x):
        return self.net(x.contiguous(memory_format=torch.channels_last))


class Net(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.enc = Encoder(dim)
        self.wind = nn.Sequential(nn.Linear(40, 128), nn.SiLU(inplace=True), nn.Linear(128, 128))
        self.head = nn.Sequential(nn.Linear(dim + 128, 256), nn.SiLU(inplace=True),
                                  nn.Dropout(0.25), nn.Linear(256, 12))

    def forward(self, img, wind):
        e = self.enc(img)
        return self.head(torch.cat([e, self.wind(wind)], dim=-1))


def cache(split, inputs):
    d = WORK / "resized_cache" / f"{PX}px"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{split}_images.npy"
    names = sorted(pd.unique(inputs[IMG].to_numpy().ravel()).tolist())
    if not path.exists():
        tmp = path.with_suffix(".partial")
        arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8,
                                        shape=(len(names), 2, PX, PX))
        for i, name in enumerate(names):
            for c, ch in enumerate(("193", "211")):
                with Image.open(ROOT / split / ch / name) as im:
                    arr[i, c] = np.asarray(im.convert("L").resize((PX, PX), Image.BILINEAR),
                                           dtype=np.uint8)
            if (i + 1) % 2000 == 0:
                print(f"  {split} {i+1}/{len(names)}", flush=True)
        arr.flush()
        del arr
        tmp.rename(path)
    a = np.load(path)
    assert a.shape == (len(names), 2, PX, PX), (a.shape, len(names))
    return {v: i for i, v in enumerate(names)}, a


def pack(split, inputs, device):
    idx, arr = cache(split, inputs)
    slots = np.stack([[idx[v] for v in inputs[col]] for col in IMG], axis=1)
    assert slots.shape == (len(inputs), 20)
    return torch.from_numpy(arr).to(device), torch.from_numpy(slots).long().to(device)


def wind_feats(inputs):
    w = inputs[W].to_numpy(np.float32)
    d = np.diff(w, axis=1, prepend=w[:, :1])
    return np.concatenate([(w - 400.0) / 100.0, d / 50.0], axis=1)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.40)
    torch.set_num_threads(4)

    tr_x = pd.read_csv(ROOT / "train/inputs.csv")
    va_x = pd.read_csv(ROOT / "validation/inputs.csv")
    te_x = pd.read_csv(ROOT / "test/inputs.csv")
    tr_y = pd.read_csv(ROOT / "train/targets.csv")[T].to_numpy(np.float32)
    va_y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float32)

    data = {s: pack(s, x, device) for s, x in (("train", tr_x), ("validation", va_x), ("test", te_x))}
    mean = data["train"][0].float().mean().item()
    std = data["train"][0].float().std().item()
    print(f"{PX}px cache ready · pixel mean={mean:.1f} std={std:.1f}", flush=True)

    wf = {s: torch.from_numpy(wind_feats(x)).to(device)
          for s, x in (("train", tr_x), ("validation", va_x), ("test", te_x))}
    last = {s: torch.from_numpy(x[W[-1]].to_numpy(np.float32)).to(device)
            for s, x in (("train", tr_x), ("validation", va_x), ("test", te_x))}
    ytr = torch.from_numpy(tr_y).to(device)

    model = Net().to(device).to(memory_format=torch.channels_last)
    print("params", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.05)
    bs = 128
    steps = math.ceil(len(tr_x) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=EPOCHS * steps, pct_start=0.15)

    def batch(split, rows, train=False):
        arr, slots = data[split]
        if train:
            t = torch.randint(0, 20, (len(rows),), device=device)
        else:
            t = torch.full((len(rows),), 19, device=device, dtype=torch.long)
        im = (arr[slots[rows, t]].float() - mean) / std
        if train:
            if torch.rand(1).item() < 0.5:
                im = torch.flip(im, dims=[-2])
            sy, sx = int(torch.randint(-6, 7, (1,))), int(torch.randint(-6, 7, (1,)))
            im = torch.roll(im, shifts=(sy, sx), dims=(-2, -1))
            im = (im * (1 + 0.12 * torch.randn(len(rows), 1, 1, 1, device=im.device))
                  + 0.12 * torch.randn(len(rows), 1, 1, 1, device=im.device))
        return im, wf[split][rows]

    @torch.no_grad()
    def infer(split, n):
        model.eval()
        out = []
        for i in range(0, n, 256):
            rows = torch.arange(i, min(i + 256, n), device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out.append(model(*batch(split, rows)).float() + last[split][rows, None])
        return torch.cat(out)

    best, best_state, bad = 1e9, None, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(tr_x), device=device)
        t0, tot = time.perf_counter(), 0.0
        for i in range(0, len(perm), bs):
            rows = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(*batch("train", rows, train=True)) + last["train"][rows, None]
                loss = ((pred - ytr[rows]) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(rows)
        v = rmse(va_y, infer("validation", len(va_x)).cpu().numpy())
        print(f"ep{ep:03d} loss={tot / len(perm):8.2f} val_rmse={v:7.3f} "
              f"lr={sched.get_last_lr()[0]:.2e} {time.perf_counter() - t0:5.1f}s", flush=True)
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: p.detach().clone() for k, p in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                print("early stop", flush=True)
                break

    model.load_state_dict(best_state)
    vp = infer("validation", len(va_x)).cpu().numpy()
    tp = infer("test", len(te_x)).cpu().numpy()
    assert np.isfinite(vp).all() and np.isfinite(tp).all()
    assert vp.shape == (len(va_x), 12) and tp.shape == (len(te_x), 12)
    np.save(OUT / f"val_{TAG}.npy", vp)
    np.save(OUT / f"test_{TAG}.npy", tp)
    torch.save(best_state, WORK / f"{TAG}.pth")
    (OUT / f"{TAG}.json").write_text(json.dumps(
        {"val_rmse": round(rmse(va_y, vp), 3), "seed": SEED, "tag": TAG, "px": PX,
         "per_h": [round(rmse(va_y[:, h], vp[:, h]), 1) for h in range(12)]}, indent=2))
    print(f"{TAG} best val_rmse={best:.3f}", flush=True)


if __name__ == "__main__":
    main()
