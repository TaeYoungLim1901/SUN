"""본선 데이터 도착 후 첫 30분. 출력으로 kit 상수·RULES 를 덮는다.

경로를 모르면 --root 에 데이터가 풀린 폴더를 준다.
모델은 돌리지 않는다.

    python tools/data_probe.py --root data
    python tools/data_probe.py --help
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LIKELY = (".npy", ".npz", ".csv", ".parquet", ".fits", ".fts", ".png", ".jpg", ".h5", ".hdf5")


def _stat_numeric(a):
    x = np.asarray(a, float).ravel()
    finite = x[np.isfinite(x)]
    out = {
        "shape": list(np.shape(a)),
        "dtype": str(np.asarray(a).dtype),
        "nan": int(np.isnan(x).sum()) if x.size else 0,
        "n": int(x.size),
    }
    if finite.size:
        out.update(
            min=float(finite.min()),
            max=float(finite.max()),
            mean=float(finite.mean()),
            p50=float(np.median(finite)),
        )
        # 채움값 후보: 9 로만 이뤄진 큰 수
        uniq, cnt = np.unique(np.round(finite, 4), return_counts=True)
        suspects = []
        for u, c in zip(uniq, cnt):
            if u >= 99 and set(str(int(abs(u))).replace("0", "")) <= {"9"}:
                suspects.append({"value": float(u), "count": int(c)})
        out["fill_suspects"] = suspects[:20]
    return out


def probe_array(path: Path):
    rec = {"path": str(path), "bytes": path.stat().st_size}
    suf = path.suffix.lower()
    try:
        if suf == ".npy":
            rec["array"] = _stat_numeric(np.load(path, allow_pickle=False))
        elif suf == ".npz":
            z = np.load(path)
            rec["keys"] = list(z.files)
            rec["arrays"] = {k: _stat_numeric(z[k]) for k in z.files[:12]}
        elif suf == ".csv":
            rec["head"] = path.read_text(encoding="utf-8", errors="replace").splitlines()[:8]
            rec["n_lines"] = sum(1 for _ in path.open(encoding="utf-8", errors="replace"))
        else:
            rec["note"] = "meta only"
    except Exception as e:
        rec["error"] = repr(e)
    return rec


def walk(root: Path, limit=80):
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in LIKELY]
    files = sorted(files, key=lambda p: p.stat().st_size, reverse=True)[:limit]
    return [probe_array(p) for p in files]


def infer_cadence_hours(times):
    t = np.asarray(times, dtype="datetime64[s]")
    if t.size < 3:
        return None
    dt = np.diff(t.astype("datetime64[m]").astype(np.int64))
    dt = dt[dt > 0]
    if dt.size == 0:
        return None
    med = float(np.median(dt)) / 60.0
    return round(med, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data"))
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    if not args.root.exists():
        print(f"NO_DATA {args.root.resolve()}")
        print("폴더를 만들거나 --root 를 지정. 이 스크립트는 데이터를 다운로드하지 않는다.")
        return
    report = {
        "root": str(args.root.resolve()),
        "files": walk(args.root),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str)[:8000])
    print("\n--- overwrite these in RULES / kit call args ---")
    print("H / cadence / channels / fill_suspects / time_min / time_max / train_vs_test_gap")
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
