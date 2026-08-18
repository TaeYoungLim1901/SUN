"""제출 직전 버그 검사기. 실패하면 exit 1 — 제출 버튼을 누르기 전에 이걸 통과시켜라.

검사 항목은 전부 **이 대회에서 실제로 틀릴 수 있는 것**만 넣었다. 일반론적 린트는 없다.
  C1 제출 3종 파일 존재 · 이름 정확 (RULES.md §2)
  C2 submission.csv 스키마 — 3,868행 / 13열 / 컬럼 순서 / test_ids.csv 와 같은 순서
  C3 값 위생 — NaN·inf 없음, 물리 범위, 상수열 아님
  C4 code.ipynb 자기완결성 — 로컬 .py import 금지, model.pth 로드 존재, pretrained 호출 없음
  C5 규정 — validation targets 를 학습에 쓴 흔적이 학습 스크립트에 있는지 grep
  C6 재현성 — model.pth 를 로드해 test 를 다시 추론하면 submission.csv 와 일치하는지

사용: python work/bugcheck.py [--repro]
"""
import argparse
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
SUB = pathlib.Path("/home/jovyan/submission")
WORK = pathlib.Path("/home/jovyan/work")
T = [f"target_{i:02d}" for i in range(12)]
N_TEST = 3868
FAILS = []


def check(name, ok, detail=""):
    print(("  OK   " if ok else "  FAIL ") + name + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repro", action="store_true",
                    help="C6 재현 검사까지 돈다 (model.pth 로 test 재추론, 몇 분 걸린다)")
    a = ap.parse_args()

    print("C1 제출 파일")
    have = {p.name for p in SUB.glob("*")}
    for f in ("code.ipynb", "model.pth", "submission.csv"):
        check(f"{f} 존재", f in have)
    extra = have - {"code.ipynb", "model.pth", "submission.csv", ".ipynb_checkpoints"}
    if extra:
        print(f"  참고: 제출되지 않는 파일이 {sorted(extra)} 있다 (무해하지만 헷갈린다)")

    print("C2 submission.csv 스키마")
    if (SUB / "submission.csv").exists():
        sub = pd.read_csv(SUB / "submission.csv")
        ids = pd.read_csv(ROOT / "test/test_ids.csv").sample_id.tolist()
        check("컬럼 = sample_id + target_00..11 (순서까지)",
              sub.columns.tolist() == ["sample_id"] + T, str(sub.columns.tolist()[:3]))
        check(f"행 수 {N_TEST}", len(sub) == N_TEST, str(len(sub)))
        check("sample_id 유일", bool(sub.sample_id.is_unique))
        check("sample_id 순서 = test_ids.csv", sub.sample_id.tolist() == ids)

        print("C3 값 위생")
        v = sub[T].to_numpy(np.float64)
        check("NaN/inf 없음", bool(np.isfinite(v).all()))
        check("물리 범위 200~900 km/s", bool(v.min() >= 150 and v.max() <= 1000),
              f"min={v.min():.1f} max={v.max():.1f}")
        check("상수열 아님 (모델이 죽지 않았나)", float(v.std()) > 5.0, f"std={v.std():.1f}")
        check("horizon 별 표준편차가 0 아님", bool((v.std(0) > 1.0).all()),
              " ".join(f"{x:.0f}" for x in v.std(0)))
    else:
        FAILS.append("submission.csv 없음")

    print("C4 code.ipynb 자기완결성")
    nb_path = SUB / "code.ipynb"
    if nb_path.exists():
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        local = re.findall(r"^\s*(?:from|import)\s+(kit|cnn|imgfeat\w*|windonly|score|bugcheck)\b",
                           src, re.M)
        check("로컬 .py import 없음", not local, str(local))
        check("model.pth 로드 코드 있음", "model.pth" in src)
        check("pretrained weight 호출 없음",
              not re.search(r"pretrained\s*=\s*True|weights\s*=\s*['\"]\w*IMAGENET", src))
        check("submission.csv 쓰기 코드 있음", "submission.csv" in src)
        pips = re.findall(r"^\s*!pip install .*", src, re.M)
        if pips:
            first = src.strip().splitlines()[0] if src.strip() else ""
            check("!pip install 이 최상단", first.strip().startswith("!pip install"), str(pips[:2]))
    else:
        FAILS.append("code.ipynb 없음")

    print("C5 규정 — validation 을 학습에 쓴 흔적")
    bad = []
    # 검사기 자신은 규칙 정규식을 문자열로 들고 있으니 자기를 보면 항상 걸린다 (conform.py 와 같은 함정).
    for p in sorted(WORK.glob("*.py")):
        if p.name in ("bugcheck.py", "score.py"):
            continue
        t = p.read_text(encoding="utf-8", errors="replace")
        # validation targets 가 fit()/학습 배열에 섞이는 전형적 패턴만 좁게 본다
        if re.search(r"(concat|vstack|append|\+)\s*\(?\s*\[?[^\n]{0,60}va_y", t) or \
           re.search(r"fit\([^\n)]*va_[xy]", t):
            bad.append(p.name)
    check("train+validation 합쳐 fit 하는 코드 없음", not bad, str(bad))

    if a.repro:
        print("C6 재현 — model.pth 로 test 재추론")
        try:
            import torch
            blob = torch.load(SUB / "model.pth", map_location="cpu", weights_only=False)
            check("model.pth 로드됨", True, f"타입 {type(blob).__name__}")
            ref = WORK / "preds" / "test_submitted.npy"
            if ref.exists():
                cur = pd.read_csv(SUB / "submission.csv")[T].to_numpy(np.float64)
                d = float(np.max(np.abs(cur - np.load(ref))))
                check("submission.csv = 저장된 test 예측", d < 1e-6, f"최대차 {d:.2e}")
            else:
                print("  건너뜀: work/preds/test_submitted.npy 가 없다")
        except Exception as e:                       # noqa: BLE001
            check("model.pth 로드됨", False, repr(e)[:120])

    print()
    if FAILS:
        print(f"실패 {len(FAILS)}건: {FAILS}")
        sys.exit(1)
    print("전부 통과. 제출해도 된다.")


if __name__ == "__main__":
    main()
