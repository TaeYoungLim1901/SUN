"""k=300 zlib + jump 가 v2 자리. 제출 폴더는 건드리지 않는다.

  python work/pack_jump.py
입력: work/model.pth.bak + work/jump_models.pth
출력: work/pack_out/model.pth  (~70 MB 목표)
"""
import copy
import pathlib
import pickle
import zlib

import torch

BAK = pathlib.Path("/home/jovyan/work/model.pth.bak")
JUMP = pathlib.Path("/home/jovyan/work/jump_models.pth")
OUT = pathlib.Path("/home/jovyan/work/pack_out/model.pth")
K = 300
GBDT = ("gbdt_v1", "gbdt_v2", "gbdt_eswf")


def truncate(models, k):
    out = []
    for m in models:
        c = copy.deepcopy(m)
        c._predictors = c._predictors[:k]
        out.append(c)
    return out


def z(v):
    return zlib.compress(pickle.dumps(v, 4), 6)


def main():
    blob = torch.load(BAK, map_location="cpu", weights_only=False)
    if blob.get("compressed"):
        raise SystemExit("bak 가 압축됨")
    jump = torch.load(JUMP, map_location="cpu", weights_only=False)
    assert len(jump) == 12
    new = dict(blob)
    new["gbdt_v2"] = jump
    for g in GBDT:
        t = truncate(new[g], K)
        new[g] = z(t)
        print(g, "zlib", len(new[g]) / 1e6)
    new["ridge_v1"] = z(blob["ridge_v1"])
    new["compressed"] = True
    new["version"] = "v1-jump-eswf-cnn96bw3"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new, OUT)
    print("wrote", OUT, OUT.stat().st_size / 1e6, "MB")


if __name__ == "__main__":
    main()
