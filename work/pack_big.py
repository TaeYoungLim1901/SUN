"""제출 팩(model.pth)을 만든다.

  python work/pack_big.py                # 풀트리 (채점 확인 안 됨 — 227MB 판이 점수 안 떴다)
  python work/pack_big.py --k 300        # k=300 절단 + zlib = 채점이 확인된 70~76MB 포맷
"""
import argparse
import copy
import pathlib
import pickle
import zlib

import torch


WORK = pathlib.Path("/home/jovyan/work")
BAK = WORK / "model.pth.bak"
OMNI = {False: WORK / "omni_models.pth", True: WORK / "omnibag_models.pth"}   # E033: omni 단일시드 -> 시드백(101/202/303)
OUT = WORK / "pack_big" / "model.pth"
# E034: cnn3a(67.485) 를 cnn3d(66.881) 로 갈고 CNN 몫을 0.34 로. val 63.1794
WEIGHTS = {
    "ridge_v1": 0.075,
    "gbdt_v1": 0.18,
    "gbdt_v2": 0.19,                    # = omnibag
    "eswf": 0.215,
    "cnn3b": 0.113,
    "cnn3c": 0.113,
    "cnn3d": 0.114,
}
# CNN 태그별 입력 해상도와 그 해상도의 train 캐시 픽셀 통계 (cnn3.py 와 같은 값)
CNN_PX = {"cnn3b": 96, "cnn3c": 128, "cnn3d": 96}
PX_STATS = {96: (65.29798889160156, 56.472267150878906),
            128: (65.31739807128906, 56.68568420410156)}


def tagged(state, arch, px=96):
    mean, std = PX_STATS[px]
    return {"state": state, "arch": arch, "px": px, "mean": mean, "std": std}


def truncate(entry, k):
    """부스팅 라운드를 앞 k 개만 남긴다 (work/shrink.py 와 같은 손잡이). 시드백이면 각 모델에."""
    ms = entry if isinstance(entry, (list, tuple)) else [entry]
    out = []
    for m in ms:
        c = copy.deepcopy(m)
        c._predictors = c._predictors[:k]
        out.append(c)
    return out if isinstance(entry, (list, tuple)) else out[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=0, help="GBDT 라운드 절단 수 (0 = 풀트리)")
    ap.add_argument("--no-zlib", action="store_true", help="k 를 줘도 압축은 안 한다")
    ap.add_argument("--fp16", action="store_true",
                    help="CNN state 를 fp16 으로 담는다. 추론이 이미 bf16 autocast 라 손실 ~0, 절반 크기")
    ap.add_argument("--bag", action="store_true",
                    help="gbdt_v2 슬롯에 omnibag(시드 3개) 을 넣는다. 없으면 omni 단일시드")
    a = ap.parse_args()

    blob = torch.load(BAK, map_location="cpu", weights_only=False)
    omni = torch.load(OMNI[a.bag], map_location="cpu", weights_only=False)
    assert len(omni) == 12
    if a.bag:
        assert all(len(mh) == 3 for mh in omni), [len(mh) for mh in omni]
    assert len(blob["gbdt_v1"]) == len(blob["gbdt_eswf"]) == 12
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, sum(WEIGHTS.values())

    # 가중치에 없는 CNN 은 담지 않는다 (예전 cnn96b/w3 는 7.5MB 죽은 무게였다)
    cnn = {"mean": blob["cnn"]["mean"], "std": blob["cnn"]["std"]}
    for tag, px in CNN_PX.items():
        state = torch.load(WORK / f"{tag}.pth", map_location="cpu", weights_only=True)
        if a.fp16:
            state = {k: (v.half() if v.is_floating_point() else v) for k, v in state.items()}
        cnn[tag] = tagged(state, "v3", px)
    assert set(CNN_PX) == {k for k in WEIGHTS if k.startswith("cnn")}

    new = dict(blob)
    new["gbdt_v2"] = omni
    new["cnn"] = cnn
    new["weights"] = WEIGHTS
    new.pop("compressed", None)
    new["version"] = "v1-omnibag-eswf-cnn3bcd"

    if a.k:
        for g in ("gbdt_v1", "gbdt_v2", "gbdt_eswf"):
            new[g] = [truncate(e, a.k) for e in new[g]]
        new["version"] += f"-k{a.k}"
    if a.fp16:
        new["version"] += "-fp16"
    if a.k and not a.no_zlib:
        # pipeline.py 가 blob["compressed"] 를 보고 되돌린다. 값 손실 0.
        for g in ("ridge_v1", "gbdt_v1", "gbdt_v2", "gbdt_eswf"):
            new[g] = zlib.compress(pickle.dumps(new[g], 4), 6)
        new["compressed"] = True
        new["version"] += "-zlib"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new, OUT)
    size = OUT.stat().st_size
    print(f"wrote {OUT} {size / 1e6:.3f} MB")
    if size > 700_000_000:
        raise SystemExit("ERROR: model.pth exceeds 700 MB; inference stopped")


if __name__ == "__main__":
    main()
