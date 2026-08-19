"""blend.json 이 고른 CNN 시드백 카드를 model.pth 로 굽는다.

  python work/pack_seedbag.py            # k=150 절단 + zlib + CNN fp16 (E039 와 같은 GBDT 부)
  python work/pack_seedbag.py --k 0      # 풀트리 (700MB 상한 안에는 들어가지만 Public 근거는 k=150)
"""
import argparse
import copy
import json
import pathlib
import pickle
import zlib

import torch

WORK = pathlib.Path("/home/jovyan/work")
BAK = WORK / "model.pth.bak"
OMNI = WORK / "omni_models.pth"          # 단일 시드. omnibag 은 Public 에서 반증됐다 (E037)
OUT = WORK / "pack_seedbag" / "model.pth"
ALIAS = {"img_ridge": "ridge_v1", "img_gbdt": "gbdt_v1", "omni": "gbdt_v2", "omnibag": "gbdt_v2"}
Z_STATS = {96: (65.29798889160156, 56.472267150878906),
           128: (65.31739807128906, 56.68568420410156)}
MED_STATS = (1.0, 0.35)                  # cnn3.py NORM=med 가 쓰는 고정값


def truncate(entry, k):
    ms = entry if isinstance(entry, (list, tuple)) else [entry]
    out = []
    for m in ms:
        c = copy.deepcopy(m)
        c._predictors = c._predictors[:k]
        out.append(c)
    return out if isinstance(entry, (list, tuple)) else out[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=150, help="GBDT 라운드 절단 (0 = 풀트리)")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--bak", default=str(BAK), help="GBDT 원본 blob (기본 model.pth.bak)")
    ap.add_argument("--out", default=str(OUT), help="쓸 경로")
    ap.add_argument("--no-omni", action="store_true",
                    help="gbdt_v2 를 omni_models.pth 로 갈지 않고 blob 것을 그대로 쓴다 (refit 평가용)")
    a = ap.parse_args()

    cfg = json.loads((WORK / "blend.json").read_text())
    weights = {ALIAS.get(k, k): float(v) for k, v in cfg["weights"].items()}
    meta = cfg["cnn"]
    assert abs(sum(weights.values()) - 1.0) < 1e-6, sum(weights.values())
    for need in ("ridge_v1", "gbdt_v1", "gbdt_v2", "eswf"):
        assert need in weights, (need, sorted(weights))

    blob = torch.load(a.bak, map_location="cpu", weights_only=False)
    assert len(blob["gbdt_v1"]) == len(blob["gbdt_eswf"]) == 12
    if not a.no_omni:
        omni = torch.load(OMNI, map_location="cpu", weights_only=False)
        assert len(omni) == 12

    cnn = {"mean": blob["cnn"]["mean"], "std": blob["cnn"]["std"]}
    tags = [t for t in weights if t.startswith("cnn")]
    for tag in tags:
        m = meta[tag]
        state = torch.load(WORK / f"{tag}.pth", map_location="cpu", weights_only=True)
        if not a.no_fp16:
            state = {k: (v.half() if v.is_floating_point() else v) for k, v in state.items()}
        mean, std = MED_STATS if m["norm"] == "med" else Z_STATS[m["px"]]
        cnn[tag] = {"state": state, "arch": m.get("arch", "v3"), "px": m["px"], "norm": m["norm"],
                    "wm": float(m.get("wm", 1.0)), "mean": mean, "std": std,
                    "fstd": float(m.get("fstd", 1.0))}

    new = dict(blob)
    if not a.no_omni:
        new["gbdt_v2"] = omni
    new["cnn"] = cnn
    new["weights"] = weights
    new.pop("compressed", None)
    new["version"] = f"v2-cnnbag{len(tags)}-omni-eswf"
    if a.k:
        for g in ("gbdt_v1", "gbdt_v2", "gbdt_eswf"):
            new[g] = [truncate(e, a.k) for e in new[g]]
        new["version"] += f"-k{a.k}"
        for g in ("ridge_v1", "gbdt_v1", "gbdt_v2", "gbdt_eswf"):
            new[g] = zlib.compress(pickle.dumps(new[g], 4), 6)
        new["compressed"] = True
    if not a.no_fp16:
        new["version"] += "-fp16"

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new, out)
    size = out.stat().st_size
    print(f"wrote {out} {size / 1e6:.2f} MB · {new['version']} · CNN {len(tags)}개")
    print("weights:", json.dumps({k: round(v, 4) for k, v in weights.items()}))
    if size > 700_000_000:
        raise SystemExit("ERROR: model.pth exceeds 700 MB")


if __name__ == "__main__":
    main()
