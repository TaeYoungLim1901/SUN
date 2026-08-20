"""E011 — 고속 지속 보정 + (선택) 고속 가중 GBDT. 정직한 그룹 fold 의 OOF 로만 맞춘다.

진단이 말한 것: last_wind 가 이미 빠른데 모델이 horizon 5 에서 ~510 으로 내려앉는다.
전체 persist 혼합은 끝난 HSS 를 붙잡아 RMSE 가 오른다. 그래서
    pred += alpha * max(last_wind - pred, 0) * 1[last_wind>=T] * exp(-h/tau)
만 허용한다 — 과소예측일 때만, 그리고 감쇠한다.

계수는 train OOF 로만 고른다. validation 은 확인 전용 (RULES.md §1 7항, SUN_PLAN Day2-2).

사용: python work/hss.py            # OOF + 보정 탐색 + val/test 저장
      python work/hss.py --w 3      # 같은 피처, y>600 가중 학습까지
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "/home/jovyan/work")
from feats import sample_feats  # noqa: E402
from groups import GAP_STEPS, iter_folds  # noqa: E402
import kit  # noqa: E402

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
OUT = pathlib.Path("/home/jovyan/work/preds")
T = [f"target_{i:02d}" for i in range(12)]
W = [f"wind_{i:02d}" for i in range(20)]
ITERS = 300
SEED = 777
TAG = "hss"


def rmse(a, b):
    return kit.score(a, b)


def fit_one(X, y, sw=None, quantile=None):
    kw = dict(max_iter=ITERS, learning_rate=0.07, max_leaf_nodes=31, l2_regularization=1.0,
              early_stopping=True, validation_fraction=0.12, n_iter_no_change=20, random_state=SEED)
    if quantile is not None:
        kw.update(loss="quantile", quantile=quantile)
    m = HistGradientBoostingRegressor(**kw)
    m.fit(X, y, sample_weight=sw)
    return m


def apply_boost(p, last, T0, alpha, tau):
    gap = np.maximum(last[:, None] - p, 0.0)
    on = (last >= T0).astype(np.float64)[:, None]
    return p + alpha * gap * on * np.exp(-np.arange(p.shape[1]) / tau)


def search_boost(y, p, last):
    base = rmse(y, p)
    best = (base, 600.0, 0.0, 4.0)
    for T0 in (500.0, 530.0, 550.0, 580.0, 600.0, 630.0):
        for alpha in (0.1, 0.2, 0.3, 0.4, 0.55, 0.7):
            for tau in (2.0, 3.0, 4.0, 6.0, 8.0, 12.0):
                s = rmse(y, apply_boost(p, last, T0, alpha, tau))
                if s < best[0] - 1e-6:
                    best = (s, T0, alpha, tau)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=float, default=1.0, help="y>600 셀 가중. 1 이면 비가중")
    ap.add_argument("--skip-oof", action="store_true",
                    help="전 train 만 학습. 보정 계수는 안 맞추고 raw val 만 본다")
    ap.add_argument("--q", type=float, default=None, help="quantile loss (예: 0.65)")
    args = ap.parse_args()
    tag = TAG if args.w == 1.0 else f"{TAG}w{args.w:g}"
    if args.q is not None:
        tag += f"q{int(round(args.q * 100))}"

    tr_x = pd.read_csv(ROOT / "train/inputs.csv")
    va_x = pd.read_csv(ROOT / "validation/inputs.csv")
    te_x = pd.read_csv(ROOT / "test/inputs.csv")
    tr_y = pd.read_csv(ROOT / "train/targets.csv")[T].to_numpy(np.float64)
    va_y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
    last_tr = tr_x[W[-1]].to_numpy(np.float64)
    last_va = va_x[W[-1]].to_numpy(np.float64)
    last_te = te_x[W[-1]].to_numpy(np.float64)

    print("features…", flush=True)
    Xtr = sample_feats("train", tr_x)
    Xva = sample_feats("validation", va_x)
    Xte = sample_feats("test", te_x)
    print("X", Xtr.shape, Xva.shape, Xte.shape, flush=True)

    comp = np.load(OUT / "train_comp.npy")
    tstep = np.load(OUT / "train_tstep.npy")
    fold = np.load(OUT / "train_fold.npy")

    T0, alpha, tau, oof_s = 600.0, 0.0, 4.0, None
    oof = None
    if not args.skip_oof:
        oof = np.zeros_like(tr_y)
        used = np.zeros(len(tr_y), bool)
        for k, (tr, va) in enumerate(iter_folds(comp, tstep, fold, GAP_STEPS)):
            used[va] = True
            for h in range(12):
                sw = None
                if args.w != 1.0:
                    sw = np.where(tr_y[tr, h] > 600.0, args.w, 1.0)
                m = fit_one(Xtr[tr], tr_y[tr, h], sw, args.q)
                oof[va, h] = m.predict(Xtr[va])
            print(f"fold{k} n_tr={tr.sum()} n_va={va.sum()} oof={rmse(tr_y[va], oof[va]):.3f}",
                  flush=True)
        assert used.all(), f"OOF 구멍 {(~used).sum()}"
        print(f"OOF {rmse(tr_y, oof):.3f}", flush=True)
        for row in kit.breakdown(tr_y, oof, kit.speed_band(tr_y)):
            print("  oof band", row, flush=True)
        oof_s, T0, alpha, tau = search_boost(tr_y, oof, last_tr)
        oof_b = apply_boost(oof, last_tr, T0, alpha, tau)
        print(f"boost T={T0} a={alpha} tau={tau}  OOF {rmse(tr_y, oof):.3f} → {oof_s:.3f}",
              flush=True)
        for row in kit.breakdown(tr_y, oof_b, kit.speed_band(tr_y)):
            print("  oof+boost band", row, flush=True)
        np.save(OUT / f"oof_{tag}.npy", oof)

    # 전 train 재학습 → val/test. 보정 계수는 위 OOF 값 고정.
    gv, gt = np.zeros_like(va_y), np.zeros((len(te_x), 12))
    for h in range(12):
        sw = None
        if args.w != 1.0:
            sw = np.where(tr_y[:, h] > 600.0, args.w, 1.0)
        m = fit_one(Xtr, tr_y[:, h], sw, args.q)
        gv[:, h] = m.predict(Xva)
        gt[:, h] = m.predict(Xte)
        print(f"  full h{h:02d} val={rmse(va_y[:, h:h+1], gv[:, h:h+1]):.2f}", flush=True)

    gv_b, gt_b = apply_boost(gv, last_va, T0, alpha, tau), apply_boost(gt, last_te, T0, alpha, tau)
    print(f"val raw {rmse(va_y, gv):.3f}  boosted {rmse(va_y, gv_b):.3f}", flush=True)
    for row in kit.breakdown(va_y, gv_b, kit.speed_band(va_y)):
        print("  val+boost band", row, flush=True)

    np.save(OUT / f"val_{tag}.npy", gv)
    np.save(OUT / f"test_{tag}.npy", gt)
    np.save(OUT / f"val_{tag}b.npy", gv_b)
    np.save(OUT / f"test_{tag}b.npy", gt_b)
    (OUT / f"{tag}.json").write_text(json.dumps({
        "oof_rmse": None if oof is None else round(rmse(tr_y, oof), 3),
        "oof_boosted": None if oof_s is None else round(oof_s, 3),
        "val_rmse": round(rmse(va_y, gv), 3),
        "val_boosted": round(rmse(va_y, gv_b), 3),
        "T": T0, "alpha": alpha, "tau": tau, "w": args.w,
        "per_h": [round(rmse(va_y[:, h:h+1], gv_b[:, h:h+1]), 1) for h in range(12)],
    }, indent=2))
    print("saved", tag, flush=True)


if __name__ == "__main__":
    main()
