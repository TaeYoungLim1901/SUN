"""측정 없이 k=300 + zlib 로 model.pth 만 다시 쓴다."""
import copy
import pathlib
import pickle
import zlib

import torch

SUB = pathlib.Path("/home/jovyan/submission")
GBDT = ("gbdt_v1", "gbdt_v2", "gbdt_eswf")
K = 300

blob = torch.load(SUB / "model.pth", map_location="cpu", weights_only=False)
print("in", (SUB / "model.pth").stat().st_size / 1e6, "MB  compressed=", blob.get("compressed"))
if blob.get("compressed"):
    raise SystemExit("이미 압축됨. bak 에서 되돌리고 다시 돌려라.")


def truncate(models, k):
    out = []
    for m in models:
        c = copy.deepcopy(m)
        c._predictors = c._predictors[:k]
        out.append(c)
    return out


new = dict(blob)
for g in GBDT:
    t = truncate(blob[g], K)
    new[g] = zlib.compress(pickle.dumps(t, 4), 6)
    print(g, "rounds", [len(m._predictors) for m in t], "zlib", len(new[g]) / 1e6)
new["ridge_v1"] = zlib.compress(pickle.dumps(blob["ridge_v1"], 4), 6)
new["compressed"] = True
bak = SUB / "model.pth.bak"
if not bak.exists():
    bak.write_bytes((SUB / "model.pth").read_bytes())
    print("bak", bak.stat().st_size / 1e6)
torch.save(new, SUB / "model.pth")
print("out", (SUB / "model.pth").stat().st_size / 1e6, "MB")
