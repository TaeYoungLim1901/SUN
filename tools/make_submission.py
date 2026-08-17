"""예측 배열 → 제출 csv. 컬럼명은 인자라 공식 샘플이 오기 전에 horizon 개수를 박지 않는다.

    python tools/make_submission.py --selftest
    python tools/make_submission.py --pred preds/test_x.npy --ids ids.csv --out submission.csv --horizons h1,h2,...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def write_submission(path, ids, pred, horizon_names=None, id_name="id"):
    pred = np.asarray(pred, float)
    if pred.ndim != 2:
        raise ValueError(f"pred must be (n, H), got {pred.shape}")
    n, h = pred.shape
    ids = np.asarray(ids)
    if len(ids) != n:
        raise ValueError(f"ids {len(ids)} != pred rows {n}")
    if horizon_names is None:
        horizon_names = [f"h{i+1}" for i in range(h)]
    if len(horizon_names) != h:
        raise ValueError("horizon_names length must equal H")
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(id_name + "," + ",".join(horizon_names) + "\n")
        for i in range(n):
            row = ",".join(f"{pred[i, j]:.6f}" for j in range(h))
            f.write(f"{ids[i]},{row}\n")
    return path, n, h


def selftest(tmp=Path("_submit_selftest.csv")):
    ids = np.array(["a", "b"])
    pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    p, n, h = write_submission(tmp, ids, pred, horizon_names=["t6", "t12"])
    text = p.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "id,t6,t12"
    assert n == 2 and h == 2
    tmp.unlink()
    print("make_submission selftest 통과")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pred", type=Path)
    ap.add_argument("--ids", type=Path, help="한 줄에 id 하나, 또는 헤더 있는 csv 첫 열")
    ap.add_argument("--out", type=Path, default=Path("submission.csv"))
    ap.add_argument("--horizons", type=str, default="", help="쉼표 구분. 비면 h1..hH")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.pred and args.ids:
        pred = np.load(args.pred)
        lines = args.ids.read_text(encoding="utf-8").splitlines()
        if "," in lines[0]:
            ids = [ln.split(",")[0] for ln in lines[1:] if ln.strip()]
        else:
            ids = [ln.strip() for ln in lines if ln.strip()]
        names = [s for s in args.horizons.split(",") if s] or None
        path, n, h = write_submission(args.out, ids, pred, names)
        print(f"wrote {path} n={n} H={h}")
    else:
        ap.print_help()
