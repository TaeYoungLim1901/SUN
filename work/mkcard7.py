"""새 성분(cnn7=v7 스칼라블록 · cnn9=v9 리맵)을 현행 제출카드에 얹어 다음 카드를 굽는다.

  python work/mkcard7.py --w 0.20 --out /home/jovyan/work/card1 --tags cnn7s7101 ... cnn9s9101 ...

기준 카드는 `n50_stage/model.pth` (Public 55.8298). 기존 가중을 (1-w) 로 눌러 새 태그에 w 를 균등 배분한다.
학습 통계(smu/ssd/fstd)는 **팩에 실어 보낸다** — 추론 때 다시 계산하지 않는다 (과거 fstd 1.368배 버그).
"""
import argparse
import json
import pathlib

import numpy as np
import torch

WORK = pathlib.Path("/home/jovyan/work")
BASECARD = WORK / "n50_stage" / "model.pth"
PIPE = WORK / "pipeline7.py"
MED = (1.0, 0.35)


def nb_from_py(path):
    """`# %%` 퍼센트 스크립트를 노트북으로. jupytext 없이 (제출 노트북은 셀 하나여도 되지만 원본 구조를 지킨다)."""
    cells, cur, kind = [], [], "code"
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("# %%"):
            if "".join(cur).strip():
                cells.append((kind, "".join(cur)))
            cur, kind = [], ("markdown" if "[markdown]" in line else "code")
            continue
        cur.append(line[2:] if kind == "markdown" and line.startswith("# ") else line)
    if "".join(cur).strip():
        cells.append((kind, "".join(cur)))
    out = []
    for k, src in cells:
        c = {"cell_type": k, "metadata": {}, "source": src.rstrip("\n").splitlines(keepends=True)}
        if k == "code":
            c["outputs"], c["execution_count"] = [], None
        out.append(c)
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
            "name": "python3"}, "language_info": {"name": "python", "version": "3.11.11"}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=float, required=True, help="새 백 전체 몫")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--base", default=str(BASECARD))
    a = ap.parse_args()

    blob = torch.load(a.base, map_location="cpu", weights_only=False)
    stats = np.load(WORK / "v7_stats.npz")
    fstd = float(stats["fstd"])
    w_each = a.w / len(a.tags)
    blob["weights"] = {k: float(v) * (1.0 - a.w) for k, v in blob["weights"].items()}
    for t in a.tags:
        arch = "v7" if t.startswith("cnn7") else "v9"
        p = WORK / f"{t}.pth"
        assert p.exists(), f"{p} 없음"
        state = {k: v.half() for k, v in torch.load(p, map_location="cpu", weights_only=True).items()}
        e = {"state": state, "arch": arch, "px": 96, "norm": "med", "wm": 0.7,
             "mean": MED[0], "std": MED[1], "fstd": fstd}
        if arch == "v7":
            e["smu"], e["ssd"] = stats["smu"], stats["ssd"]
        blob["cnn"][t] = e
        blob["weights"][t] = w_each
    tot = sum(blob["weights"].values())
    assert abs(tot - 1.0) < 1e-6, tot
    blob["version"] = f"v3-n50+new{len(a.tags)}w{int(a.w*100)}"

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(blob, out / "model.pth")
    (out / "code.ipynb").write_text(json.dumps(nb_from_py(PIPE), ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    mb = (out / "model.pth").stat().st_size / 1e6
    print(f"{out}  model.pth {mb:.1f}MB  tags {len(a.tags)} w_each {w_each:.5f} sum {tot:.6f}")
    print("weights", {k: round(v, 5) for k, v in blob["weights"].items() if not k.startswith("cnn6")})


if __name__ == "__main__":
    main()
