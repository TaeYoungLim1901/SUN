#!/bin/sh
# 카드 폴더를 빈 폴더에 복사해 nbconvert 로 실제 실행하고, 원본 CSV 와 최대 절대차를 잰다.
#
#   bash submissions/repro.sh /home/jovyan/work/card2 /home/jovyan/work/card2_run
#
# 통과 기준은 최대 절대차 ≤ 1e-6 (CHECKLIST.md D). 이건 예외 없이 필수다 —
# 제출은 파일 업로드가 아니라 폴더 스냅샷이라, 채점 서버는 code.ipynb 를 빈 환경에서 다시 돌린다.
# 우리 work/ 에만 있는 .py 를 import 하면 여기서 걸린다.
set -e
CARD=$1
RUN=$2
rm -rf "$RUN"; mkdir -p "$RUN"
cp "$CARD/code.ipynb" "$CARD/model.pth" "$RUN/"
cd "$RUN"
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=5400 code.ipynb
CARD="$CARD" python - <<'PY'
import os, pathlib
import numpy as np, pandas as pd
card = pathlib.Path(os.environ["CARD"])
a = pd.read_csv(card / "submission.csv")
b = pd.read_csv("submission.csv")
cols = [c for c in a.columns if c != "sample_id"]
assert list(a.columns) == list(b.columns), "컬럼이 다르다"
assert (a.sample_id == b.sample_id).all(), "sample_id 순서가 다르다"
d = float(np.abs(a[cols].to_numpy() - b[cols].to_numpy()).max())
print("rows", b.shape, "최대 절대차", d)
assert d <= 1e-6, f"재현 실패: {d}"
print("REPRO_OK")
PY
