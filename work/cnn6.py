"""2-스트림: 영상 + 광류장을 채널로 함께 먹인다 (SEARCH.md §2-4).

`flow.py` 는 광류를 **요약 스칼라 60여개**로 눌러 GBDT 에 넣었고 단독 84.6 으로 죽었다.
그런데 그 잔차 상관은 0.73 으로 우리 앙상블에서 가장 낮다 — 정보는 있는데 표현이 죽인 것이다.
여기서는 프레임쌍마다 Farneback 광류장(fx, fy)을 그대로 채널로 준다.

원본 주석:
영상 인코더를 처음부터 학습한다. pretrained weight 미사용 (RULES.md §1 5항).

핵심 설계 셋:
  1. 유니크 이미지만 96px uint8 로 캐시하고 **통째로 GPU 에 올린다** (train 187MB).
     디스크·DataLoader 병목이 사라져서 epoch 이 초 단위로 떨어진다.
  2. 타깃은 절대 풍속이 아니라 **마지막 관측 대비 잔차**. persistence 를 공짜로 깔고 간다.
  3. validation 은 early stopping·모델선택 전용. train 에 절대 합치지 않는다 (RULES.md §1 7항).

사용: python work/cnn.py [PX] [TAG] [SEED] [EPOCHS]
"""
import json
import math
import pathlib
import sys
import time
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
WORK = pathlib.Path("/home/jovyan/work")
OUT = WORK / "preds"
OUT.mkdir(parents=True, exist_ok=True)
PX = int(sys.argv[1]) if len(sys.argv) > 1 else 96
TAG = sys.argv[2] if len(sys.argv) > 2 else "cnn96"
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 777
EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 40
HSS_W = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0  # y>600 셀 가중. 1 이면 기존과 동일
LAST = int(sys.argv[6]) if len(sys.argv) > 6 else 20     # 최근 N 프레임만. 20 이면 기존과 동일
NORM = sys.argv[7] if len(sys.argv) > 7 else "z"         # z=전역 zscore · med=이미지 disk-median
WM = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0    # 폭 배수. 1 이면 기존 아키텍처 그대로
IMG = [f"image_{i:02d}" for i in range(20)]
W = [f"wind_{i:02d}" for i in range(20)]
T = [f"target_{i:02d}" for i in range(12)]
DEV = torch.device("cuda")
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True
# VM 보호: GPU 메모리는 절반 아래로 묶는다. 같은 노드에 웹서버가 산다.
torch.cuda.set_per_process_memory_fraction(0.45)
torch.set_num_threads(4)


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


TAB = chr(9)
NL = chr(10)


def KEYLINES(keys, nl=False):
    """키 목록을 텍스트 줄로 (탭/개행 리터럴을 코드에 직접 쓰지 않는다)."""
    return [a + TAB + b + (NL if nl else "") for a, b in keys]


def flow_pairs(split, inputs, idx, arr):
    """연속 프레임쌍의 Farneback 광류장을 유니크 쌍 단위로 캐시한다 (flow.py 와 같은 파라미터)."""
    import cv2
    fdir = WORK / "flowfield_cache"
    fdir.mkdir(parents=True, exist_ok=True)
    path = fdir / f"{split}_{PX}px.npy"
    kpath = fdir / f"{split}_{PX}px_keys.txt"
    names = inputs[IMG].to_numpy()
    seen = {}
    for row in names:
        for t in range(19):
            seen.setdefault((row[t], row[t + 1]), len(seen))
    keys = list(seen)
    if path.exists() and kpath.exists() and (
            kpath.read_text(encoding="utf-8").splitlines() == KEYLINES(keys)):
        print(f"  {split} flow cache hit {len(keys)}", flush=True)
        return seen, np.load(path, mmap_mode="r")
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                    shape=(len(keys), 2, PX, PX))
    t0 = time.time()
    for i, (na, nb) in enumerate(keys):
        a = np.asarray(arr[idx[na], 0], np.uint8)      # 193 채널만으로 광류를 낸다
        b = np.asarray(arr[idx[nb], 0], np.uint8)
        f = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 11, 3, 5, 1.2, 0)
        out[i, 0] = f[..., 0].astype(np.float16)
        out[i, 1] = f[..., 1].astype(np.float16)
        if (i + 1) % 2000 == 0:
            print(f"  {split} flow {i+1}/{len(keys)} {time.time()-t0:.0f}s", flush=True)
    out.flush()
    kpath.write_text("".join(KEYLINES(keys, nl=True)), encoding="utf-8")
    return seen, np.load(path, mmap_mode="r")


def pack(split, inputs):
    idx, arr = cache(split, inputs)
    slots = np.stack([[idx[v] for v in inputs[col]] for col in IMG], axis=1)   # (N,20)
    seen, farr = flow_pairs(split, inputs, idx, arr)
    names = inputs[IMG].to_numpy()
    pslots = np.stack([[seen[(row[t], row[t + 1])] for t in range(19)] for row in names])
    # 광류는 19쌍이므로 영상도 t=1..19 로 맞춘다
    slots = slots[:, 1:]
    assert slots.shape == pslots.shape == (len(inputs), 19)
    return (torch.from_numpy(arr).to(DEV), torch.from_numpy(slots).long().to(DEV),
            torch.from_numpy(np.asarray(farr, np.float16)).to(DEV),
            torch.from_numpy(pslots).long().to(DEV))


class Encoder(nn.Module):
    """작은 residual CNN. 랜덤 초기화, 제공 데이터로만 학습."""

    def __init__(self, dim=192, wm=1.0):
        super().__init__()
        c1, c2, c3, c4 = (int(32 * wm), int(64 * wm), int(128 * wm), int(256 * wm))

        def blk(i, o, s=2):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.SiLU(inplace=True),
                nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.SiLU(inplace=True))

        self.net = nn.Sequential(blk(4, c1), blk(c1, c2), blk(c2, c3), blk(c3, c4),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(c4, dim))

    def forward(self, x):
        return self.net(x.contiguous(memory_format=torch.channels_last))


class Net(nn.Module):
    def __init__(self, dim=192, hid=224, wm=1.0):
        super().__init__()
        self.enc = Encoder(dim, wm)
        self.hid = hid
        self.gru = nn.GRU(dim, hid, num_layers=1, batch_first=True, bidirectional=True)
        self.wind = nn.Sequential(nn.Linear(40, 128), nn.SiLU(inplace=True), nn.Linear(128, 128))
        self.head = nn.Sequential(nn.Linear(4 * hid + 128, 256), nn.SiLU(inplace=True),
                                  nn.Dropout(0.25), nn.Linear(256, 12))

    def forward(self, img, wind):
        b, t = img.shape[:2]
        e = self.enc(img.flatten(0, 1)).view(b, t, -1)
        g, _ = self.gru(e)
        g = torch.cat([g[:, -1], g.mean(1)], dim=-1)
        return self.head(torch.cat([g, self.wind(wind)], dim=-1))


def wind_feats(inputs):
    w = inputs[W].to_numpy(np.float32)
    d = np.diff(w, axis=1, prepend=w[:, :1])
    return np.concatenate([(w - 400.0) / 100.0, d / 50.0], axis=1)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)))


tr_x = pd.read_csv(ROOT / "train/inputs.csv")
va_x = pd.read_csv(ROOT / "validation/inputs.csv")
te_x = pd.read_csv(ROOT / "test/inputs.csv")
tr_y = pd.read_csv(ROOT / "train/targets.csv")[T].to_numpy(np.float32)
va_y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float32)

data = {s: pack(s, x) for s, x in (("train", tr_x), ("validation", va_x), ("test", te_x))}
FSTD = 1.0     # 광류 스케일 (px/frame). 학습 직전 train 에서 실측해 덮어쓴다
mean = data["train"][0].float().mean().item()
std = data["train"][0].float().std().item()
if NORM == "med":
    # DOMAIN §9.3: 절대 스케일 confound 를 disk-median 으로 상쇄. 디스크 근사 = 중앙 원.
    yy, xx = torch.meshgrid(torch.arange(PX, device=DEV), torch.arange(PX, device=DEV), indexing="ij")
    disk = ((yy - PX / 2 + 0.5) ** 2 + (xx - PX / 2 + 0.5) ** 2) <= (0.47 * PX) ** 2
    for s in data:
        arr, slots, farr, pslots = data[s]
        a = arr.float()
        med = a[:, :, disk].median(-1).values[:, :, None, None]
        a = a / med.clamp(min=1.0)
        data[s] = (a.to(memory_format=torch.channels_last), slots, farr, pslots)
    mean, std = 1.0, 0.35
print(f"{PX}px cache ready · pixel mean={mean:.1f} std={std:.1f} last={LAST} norm={NORM}", flush=True)

wf = {s: torch.from_numpy(wind_feats(x)).to(DEV)
      for s, x in (("train", tr_x), ("validation", va_x), ("test", te_x))}
last = {s: torch.from_numpy(x[W[-1]].to_numpy(np.float32)).to(DEV)
        for s, x in (("train", tr_x), ("validation", va_x), ("test", te_x))}
ytr = torch.from_numpy(tr_y).to(DEV)

FSTD = float(data["train"][2][:2000].float().std().item()) or 1.0
print(f"flow std={FSTD:.3f} px/frame", flush=True)
model = Net(int(192 * WM), int(224 * WM), WM).to(DEV).to(memory_format=torch.channels_last)
print("params", sum(p.numel() for p in model.parameters()), "hss_w", HSS_W, flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
BS = 32
steps = math.ceil(len(tr_x) / BS)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=EPOCHS * steps, pct_start=0.3)


def batch(split, rows, train=False):
    arr, slots, farr, pslots = data[split]
    im = (arr[slots[rows]].float() - mean) / std              # (b,19,2,PX,PX)
    fl = farr[pslots[rows]].float() / FSTD                    # (b,19,2,PX,PX) fx,fy
    if train:
        if torch.rand(1).item() < 0.5:
            im = torch.flip(im, dims=[-2])                    # 남북 반전만
            fl = torch.flip(fl, dims=[-2])
            fl = torch.stack([fl[:, :, 0], -fl[:, :, 1]], dim=2)   # 뒤집으면 fy 부호가 바뀐다
        sy, sx = (int(torch.randint(-6, 7, (1,))), int(torch.randint(-6, 7, (1,))))
        im = torch.roll(im, shifts=(sy, sx), dims=(-2, -1))
        fl = torch.roll(fl, shifts=(sy, sx), dims=(-2, -1))
        # 밝기/대비 지터는 영상에만. 광류에 곱하면 속도를 왜곡한다
        im = im * (1 + 0.12 * torch.randn(len(rows), 1, 1, 1, 1, device=im.device))              + 0.12 * torch.randn(len(rows), 1, 1, 1, 1, device=im.device)
        keep = (torch.rand(len(rows), im.shape[1], 1, 1, 1, device=im.device) > 0.15).float()
        im, fl = im * keep, fl * keep
    return torch.cat([im, fl], dim=2), wf[split][rows]


@torch.no_grad()
def infer(split, n):
    model.eval()
    out = []
    for i in range(0, n, 96):
        rows = torch.arange(i, min(i + 96, n), device=DEV)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            im, w = batch(split, rows)
            fim = torch.flip(im, dims=[-2])
            fim = torch.cat([fim[:, :, :3], -fim[:, :, 3:]], dim=2)   # fy 부호
            o = model(im, w).float() + model(fim, w).float()
            out.append(o / 2 + last[split][rows, None])
    return torch.cat(out)


best, best_state, bad = 1e9, None, 0
for ep in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(tr_x), device=DEV)
    t0, tot = time.perf_counter(), 0.0
    for i in range(0, len(perm), BS):
        rows = perm[i:i + BS]
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(*batch("train", rows, train=True)) + last["train"][rows, None]
            err = (pred - ytr[rows]) ** 2
            if HSS_W != 1.0:
                err = err * torch.where(ytr[rows] > 600.0, HSS_W, 1.0)
            loss = err.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        tot += loss.item() * len(rows)
    v = rmse(va_y, infer("validation", len(va_x)).cpu().numpy())
    print(f"ep{ep:03d} loss={tot/len(perm):8.2f} val_rmse={v:7.3f} "
          f"lr={sched.get_last_lr()[0]:.2e} {time.perf_counter()-t0:5.1f}s", flush=True)
    if v < best - 1e-4:
        best, bad = v, 0
        best_state = {k: p.detach().clone() for k, p in model.state_dict().items()}
    else:
        bad += 1
        if bad >= 15:
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
    {"val_rmse": round(rmse(va_y, vp), 3), "hss_w": HSS_W, "seed": SEED,
     "per_h": [round(rmse(va_y[:, h], vp[:, h]), 1) for h in range(12)]}, indent=2))
print(f"{TAG} best val_rmse={best:.3f}", flush=True)
