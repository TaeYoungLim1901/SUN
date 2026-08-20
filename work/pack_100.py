"""풀트리 zlib + E016 align 으로 v2 자리를 교체. 제출 폴더는 건드리지 않는다.

  python work/pack_100.py     # work/model.pth.bak + work/align_models.pth → pack_out/model.pth
목표 크기 ~90-100 MB. 70.8 MB 승리팩은 work/pack_win_57/ 에 있다.
"""
import pathlib
import pickle
import zlib

import torch

BAK = pathlib.Path("/home/jovyan/work/model.pth.bak")
ALIGN = pathlib.Path("/home/jovyan/work/align_models.pth")
OUT = pathlib.Path("/home/jovyan/work/pack_out/model.pth")
SK = ("ridge_v1", "gbdt_v1", "gbdt_v2", "gbdt_eswf")


def z(v):
    return zlib.compress(pickle.dumps(v, 4), 6)


def main():
    assert BAK.exists(), BAK
    assert ALIGN.exists(), ALIGN
    blob = torch.load(BAK, map_location="cpu", weights_only=False)
    if blob.get("compressed"):
        raise SystemExit("bak 가 이미 압축됨")
    align = torch.load(ALIGN, map_location="cpu", weights_only=False)
    assert len(align) == 12, len(align)
    new = dict(blob)
    new["gbdt_v2"] = align
    print("raw", {k: len(pickle.dumps(new[k], 4)) / 1e6 for k in SK})
    for k in SK:
        new[k] = z(new[k])
        print("zlib", k, len(new[k]) / 1e6)
    new["compressed"] = True
    new["version"] = "v1-align-eswf-cnn96bw3"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new, OUT)
    mb = OUT.stat().st_size / 1e6
    print("wrote", OUT, mb, "MB")
    if mb > 105:
        raise SystemExit("100MB 목표를 넘었다. k 절단을 넣어라.")


if __name__ == "__main__":
    main()
