"""horizon 별로 flow_h.py 를 서브프로세스로. 죽은 열만 다시."""
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
OUT = pathlib.Path("/home/jovyan/work/preds")
T = [f"target_{i:02d}" for i in range(12)]


def done_mask():
    p = OUT / "val_omniflow.npy"
    if not p.exists():
        return [False] * 12
    v = np.load(p)
    return [bool(np.abs(v[:, h]).sum() > 0) and np.isfinite(v[:, h]).all() for h in range(12)]


def main():
    for h in range(12):
        mask = done_mask()
        if mask[h]:
            print(f"skip h{h:02d}", flush=True)
            continue
        print(f"run h{h:02d}", flush=True)
        for attempt in range(4):
            r = subprocess.run(
                [sys.executable, "work/flow_h.py", str(h)],
                cwd="/home/jovyan")
            print(f"exit h{h:02d} try{attempt} {r.returncode}", flush=True)
            if r.returncode == 0 and done_mask()[h]:
                break
            time.sleep(15)
    mask = done_mask()
    print("mask", mask, flush=True)
    if not all(mask):
        print("INCOMPLETE", flush=True)
        return
    y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
    pv = np.load(OUT / "val_omniflow.npy")
    val = float(np.sqrt(np.mean((y - pv) ** 2)))
    per = [round(float(np.sqrt(np.mean((y[:, h] - pv[:, h]) ** 2))), 1) for h in range(12)]
    (OUT / "omniflow.json").write_text(json.dumps({"val_rmse": round(val, 3), "per_h": per}, indent=2))
    print(f"omniflow val_rmse={val:.3f}", flush=True)
    subprocess.run([sys.executable, "work/flow_score.py"], cwd="/home/jovyan")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
