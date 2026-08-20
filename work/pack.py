"""63.76 결합을 submission/ 3종으로 묶는다.

  python work/pack.py --from-npy     # 저장된 val/test npy 로 csv 만 (검증용)
  python work/pack.py --train        # sklearn 재학습 + CNN pth 적재 → model.pth + csv

가중치(공식 val hill_climb, 원성분 6개):
  img_gbdt 0.25 + eswf 0.25 + img_ridge 0.125 + v2_gbdt 0.125 + cnn96b 0.125 + cnn96w3 0.125
"""
import argparse
import pathlib
import pickle
import sys
import zlib

import numpy as np
import pandas as pd

import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from torch import nn

sys.path.insert(0, "/home/jovyan/work")
import eswf  # noqa: E402
import kit  # noqa: E402
from feats import sample_feats as feats_v2  # noqa: E402
from imgfeat import sample_feats as feats_v1  # noqa: E402

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
PRED = pathlib.Path("/home/jovyan/work/preds")
WORK = pathlib.Path("/home/jovyan/work")
SUB = pathlib.Path("/home/jovyan/work/pack_out")  # 제출 폴더를 덮지 않는다. 3종이 모이면 그때 복사.
T = [f"target_{i:02d}" for i in range(12)]
W = [f"wind_{i:02d}" for i in range(20)]
CLIP = (200.0, 900.0)
WEIGHTS = {
    "ridge_v1": 0.125, "gbdt_v1": 0.25, "gbdt_v2": 0.125,
    "eswf": 0.25, "cnn96b": 0.125, "cnn96w3": 0.125,
}
# npy 파일 이름 ↔ 노트북 키
NPY = {
    "ridge_v1": "img_ridge", "gbdt_v1": "img_gbdt", "gbdt_v2": "v2_gbdt",
    "eswf": "eswf", "cnn96b": "cnn96b", "cnn96w3": "cnn96w3",
}
IMG = [f"image_{i:02d}" for i in range(20)]
PX96 = 96
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()

        def blk(i, o, s=2):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.SiLU(inplace=True),
                nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.SiLU(inplace=True))

        self.net = nn.Sequential(blk(2, 24), blk(24, 48), blk(48, 96), blk(96, 128),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2),
                                 nn.Linear(128, dim))

    def forward(self, x):
        return self.net(x.contiguous(memory_format=torch.channels_last))


class Net(nn.Module):
    def __init__(self, dim=128, hid=160):
        super().__init__()
        self.enc = Encoder(dim)
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


def blend_npy():
    va = {k: np.load(PRED / f"val_{NPY[k]}.npy") for k in WEIGHTS}
    te = {k: np.load(PRED / f"test_{NPY[k]}.npy") for k in WEIGHTS}
    y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
    vb = sum(WEIGHTS[k] * va[k] for k in WEIGHTS)
    tb = sum(WEIGHTS[k] * te[k] for k in WEIGHTS)
    print("from-npy val", kit.score(y, vb), "clipped", kit.score(y, np.clip(vb, *CLIP)))
    return np.clip(vb, *CLIP), np.clip(tb, *CLIP)


def write_csv(pred):
    te = pd.read_csv(ROOT / "test/inputs.csv")
    ids = pd.read_csv(ROOT / "test/test_ids.csv").sample_id.tolist()
    sub = pd.DataFrame(pred, columns=T)
    sub.insert(0, "sample_id", te.sample_id.tolist())
    assert sub.sample_id.tolist() == ids
    assert np.isfinite(sub[T].to_numpy()).all()
    SUB.mkdir(parents=True, exist_ok=True)
    sub.to_csv(SUB / "submission.csv", index=False)
    print("wrote", SUB / "submission.csv", sub.shape)


def train_sklearn():
    tr_x = pd.read_csv(ROOT / "train/inputs.csv")
    va_x = pd.read_csv(ROOT / "validation/inputs.csv")
    te_x = pd.read_csv(ROOT / "test/inputs.csv")
    tr_y = pd.read_csv(ROOT / "train/targets.csv")[T].to_numpy(np.float64)
    va_y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)

    print("v1 feats", flush=True)
    X1tr, X1va, X1te = feats_v1("train", tr_x), feats_v1("validation", va_x), feats_v1("test", te_x)
    print("v2 feats", flush=True)
    X2tr, X2va, X2te = feats_v2("train", tr_x), feats_v2("validation", va_x), feats_v2("test", te_x)

    print("eswf seq", flush=True)
    eswf.SEQ = {s: eswf.sequences(s, x) for s, x in
                (("train", tr_x), ("validation", va_x), ("test", te_x))}

    ridge = RidgeCV(alphas=np.logspace(-3, 4, 30)).fit(X1tr, tr_y)
    gbdt_v1, gbdt_v2, gbdt_es = [], [], []
    pv = {k: np.zeros_like(va_y) for k in ("img_ridge", "img_gbdt", "v2_gbdt", "eswf")}
    pt = {k: np.zeros((len(te_x), 12)) for k in pv}
    pv["img_ridge"], pt["img_ridge"] = ridge.predict(X1va), ridge.predict(X1te)
    print("ridge val", kit.score(va_y, pv["img_ridge"]), flush=True)

    for h in range(12):
        m1 = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.06, max_leaf_nodes=31,
            early_stopping=True, validation_fraction=0.12, random_state=777,
        ).fit(X1tr, tr_y[:, h])
        m2 = HistGradientBoostingRegressor(
            max_iter=700, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.12, n_iter_no_change=30, random_state=777,
        ).fit(X2tr, tr_y[:, h])
        hh = 6.0 * (h + 1)
        Xe_tr = eswf.build("train", tr_x, hh)
        Xe_va = eswf.build("validation", va_x, hh)
        Xe_te = eswf.build("test", te_x, hh)
        me = HistGradientBoostingRegressor(
            max_iter=700, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.12, n_iter_no_change=30, random_state=777,
        ).fit(Xe_tr, tr_y[:, h])
        gbdt_v1.append(m1); gbdt_v2.append(m2); gbdt_es.append(me)
        pv["img_gbdt"][:, h], pt["img_gbdt"][:, h] = m1.predict(X1va), m1.predict(X1te)
        pv["v2_gbdt"][:, h], pt["v2_gbdt"][:, h] = m2.predict(X2va), m2.predict(X2te)
        pv["eswf"][:, h], pt["eswf"][:, h] = me.predict(Xe_va), me.predict(Xe_te)
        print(f"  h{h:02d} v1={kit.score(va_y[:, h:h+1], pv['img_gbdt'][:, h:h+1]):.1f} "
              f"v2={kit.score(va_y[:, h:h+1], pv['v2_gbdt'][:, h:h+1]):.1f} "
              f"es={kit.score(va_y[:, h:h+1], pv['eswf'][:, h:h+1]):.1f}", flush=True)

    return dict(ridge=ridge, gbdt_v1=gbdt_v1, gbdt_v2=gbdt_v2, gbdt_eswf=gbdt_es,
                pv=pv, pt=pt, va_y=va_y, te_x=te_x, va_x=va_x)


def load_96(split, inputs):
    path = WORK / "resized_cache" / "96px" / f"{split}_images.npy"
    names = sorted(pd.unique(inputs[IMG].to_numpy().ravel()).tolist())
    a = np.load(path)
    assert a.shape == (len(names), 2, PX96, PX96)
    idx = {v: i for i, v in enumerate(names)}
    slots = np.stack([[idx[v] for v in inputs[col]] for col in IMG], axis=1)
    return torch.from_numpy(a).to(DEV), torch.from_numpy(slots).long().to(DEV)


def load_cnn(tag, va_x, te_x):
    tr_x = pd.read_csv(ROOT / "train/inputs.csv")
    data = {s: load_96(s, x) for s, x in
            (("train", tr_x), ("validation", va_x), ("test", te_x))}
    mean = data["train"][0].float().mean().item()
    std = data["train"][0].float().std().item()
    wf = {s: torch.from_numpy(wind_feats(x)).to(DEV) for s, x in (("validation", va_x), ("test", te_x))}
    last = {s: torch.from_numpy(x[W[-1]].to_numpy(np.float32)).to(DEV)
            for s, x in (("validation", va_x), ("test", te_x))}
    model = Net().to(DEV).to(memory_format=torch.channels_last)
    raw = torch.load(WORK / f"{tag}.pth", map_location=DEV, weights_only=True)
    model.load_state_dict(raw)
    model.eval()

    def infer(split, n):
        out = []
        arr, slots = data[split]
        for i in range(0, n, 96):
            rows = torch.arange(i, min(i + 96, n), device=DEV)
            im = (arr[slots[rows]].float() - mean) / std
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out.append(model(im, wf[split][rows]).float() + last[split][rows, None])
        return torch.cat(out).cpu().numpy()

    vp, tp = infer("validation", len(va_x)), infer("test", len(te_x))
    print(tag, "val", kit.score(
        pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64), vp), flush=True)
    return vp, tp, {k: v.detach().cpu() for k, v in model.state_dict().items()}, mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-npy", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--cnn-only", action="store_true", help="GPU: CNN 두 개 추론만 저장")
    ap.add_argument("--sklearn-only", action="store_true", help="CPU: GBDT/Ridge 만. 이후 --merge")
    ap.add_argument("--merge", action="store_true", help="cnn_blob + sklearn_blob → pth+csv")
    a = ap.parse_args()
    SUB.mkdir(parents=True, exist_ok=True)
    if a.from_npy:
        _, tb = blend_npy()
        write_csv(tb)
        return

    va_x = pd.read_csv(ROOT / "validation/inputs.csv")
    te_x = pd.read_csv(ROOT / "test/inputs.csv")
    va_y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)

    if a.cnn_only:
        cnn_blob, pv, pt = {}, {}, {}
        for tag in ("cnn96b", "cnn96w3"):
            vp, tp, state, mean, std = load_cnn(tag, va_x, te_x)
            pv[tag], pt[tag] = vp, tp
            cnn_blob[tag] = state
            cnn_blob["mean"], cnn_blob["std"] = mean, std
        torch.save({"cnn": cnn_blob, "pv": pv, "pt": pt}, SUB / "cnn_blob.pt")
        print("wrote", SUB / "cnn_blob.pt")
        return

    if a.sklearn_only:
        sk = train_sklearn()
        torch.save({k: sk[k] for k in ("ridge", "gbdt_v1", "gbdt_v2", "gbdt_eswf", "pv", "pt")},
                   SUB / "sklearn_blob.pt")
        print("wrote", SUB / "sklearn_blob.pt")
        return

    if a.merge or a.train:
        if a.train:
            sk = train_sklearn()
            cnn_blob = {}
            for tag in ("cnn96b", "cnn96w3"):
                vp, tp, state, mean, std = load_cnn(tag, va_x, te_x)
                sk["pv"][tag], sk["pt"][tag] = vp, tp
                cnn_blob[tag] = state
                cnn_blob["mean"], cnn_blob["std"] = mean, std
        else:
            sk = torch.load(SUB / "sklearn_blob.pt", weights_only=False)
            cb = torch.load(SUB / "cnn_blob.pt", weights_only=False)
            cnn_blob = cb["cnn"]
            sk["pv"].update(cb["pv"])
            sk["pt"].update(cb["pt"])
        va_y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
    else:
        sys.exit(" --from-npy | --cnn-only | --sklearn-only | --merge | --train")

    def take(d, k):
        return d[k] if k in d else d[NPY[k]]
    vb = sum(WEIGHTS[k] * take(sk["pv"], k) for k in WEIGHTS)
    tb = sum(WEIGHTS[k] * take(sk["pt"], k) for k in WEIGHTS)
    vb, tb = np.clip(vb, *CLIP), np.clip(tb, *CLIP)
    print("RETRAIN blend val", kit.score(va_y, vb), flush=True)
    for row in kit.breakdown(va_y, vb, kit.speed_band(va_y)):
        print(" ", row, flush=True)

    SUB.mkdir(parents=True, exist_ok=True)
    # sklearn 파트는 zlib 로 담는다 — 제출 파일 용량 상한 때문 (work/shrink.py, pipeline.py 가 푼다)
    def z(v):
        return zlib.compress(pickle.dumps(v, protocol=4), 6)

    torch.save({
        "ridge_v1": z(sk["ridge"]), "gbdt_v1": z(sk["gbdt_v1"]), "gbdt_v2": z(sk["gbdt_v2"]),
        "gbdt_eswf": z(sk["gbdt_eswf"]), "cnn": cnn_blob, "weights": WEIGHTS, "clip": CLIP,
        "compressed": True, "version": "v1v2-eswf-cnn96bw3",
    }, SUB / "model.pth")
    print("wrote", SUB / "model.pth", (SUB / "model.pth").stat().st_size)
    write_csv(tb)
    np.save(PRED / "test_submitted.npy", tb)
    print("blend val", kit.score(va_y, vb), flush=True)


if __name__ == "__main__":
    main()
