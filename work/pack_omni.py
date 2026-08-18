"""k=300 zlib + omni 가 v2 자리. 제출 폴더는 완성 후에만 복사.

  python work/pack_omni.py
입력: work/model.pth.bak + work/omni_models.pth
출력: work/pack_out/model.pth
"""
import copy
import pathlib
import pickle
import zlib

import torch

BAK = pathlib.Path("/home/jovyan/work/model.pth.bak")
OMNI = pathlib.Path("/home/jovyan/work/omni_models.pth")
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
    omni = torch.load(OMNI, map_location="cpu", weights_only=False)
    assert len(omni) == 12
    new = dict(blob)
    new["gbdt_v2"] = omni
    for g in GBDT:
        new[g] = z(truncate(new[g], K))
        print(g, "zlib", len(new[g]) / 1e6)
    new["ridge_v1"] = z(blob["ridge_v1"])
    new["compressed"] = True
    new["version"] = "v1-omni-eswf-cnn96bw3"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new, OUT)
    print("wrote", OUT, OUT.stat().st_size / 1e6, "MB")


if __name__ == "__main__":
    main()
