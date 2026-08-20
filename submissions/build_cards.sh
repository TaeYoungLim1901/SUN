#!/bin/sh
# 최근 제출 카드 5장을 굽는 명령 그대로. 서버(/home/jovyan) 에서 돈다.
# 하나씩 골라 실행해라 — 0 번(성분 학습)은 GPU 를 몇 시간 잡는다.
# 각 카드가 무엇이고 왜 그 가중인지는 README.md 를 봐라.
set -e
W=/home/jovyan/work

# ── 0. 성분 학습 (한 번만. preds/ 에 npy 가 이미 있으면 건너뛴다) ─────────
# 광류 2-스트림 CNN 32 시드 — 카드 가중의 40~60% 를 차지하는 주력 성분.
#   인자: PX TAG SEED EPOCHS HSS_W LAST NORM WM
#   (HSS_W=y>600 셀 가중 · LAST=쓸 최근 프레임수 · NORM=med(디스크중앙값)/z · WM=폭배수)
for s in $(seq 6101 6132); do python $W/cnn6.py 96 cnn6s$s $s 60 1.0 20 med 0.7; done
# 하이브리드 = 스칼라블록(cnn7) 10 시드 + Stonyhurst 리맵(cnn9) 3 시드
for s in $(seq 7101 7110); do python $W/cnn7.py 96 cnn7s$s $s 60 1.0 20 med 0.7; done
for s in 9101 9102 9103;   do python $W/cnn9.py 96 cnn9s$s $s 60 1.0 20 med 0.7; done

# ── 1. MAX (Public 56.2743) · 2. N50 (Public 55.8298) ────────────────────
# pack_seedbag.py 는 가중을 인자로 받지 않는다 — `work/blend.json` 의 weights 를 읽는다.
# 굽기 전에 blend.json 의 weights 를 아래로 맞춰라 (합 1.0, 실측값은 model.pth 에서 뽑은 것):
#
#   MAX  : ridge_v1 .060 · gbdt_v1 .144 · gbdt_v2 .152 · eswf .172
#          + cnn3b/c/d 합 .272 + cnn6s6101~6132 합 .200 (시드당 .00625)
#   N50  : ridge_v1 .05682 · gbdt_v1 .13636 · gbdt_v2 .14394 · eswf .16288
#          + cnn3 제거 + cnn6s6101~6132 합 .4998 (시드당 .01562)
#
# --k 0 은 GBDT 풀트리(절단 없음). 용량 700MB·검증시간 무제한이 실증돼서 절단은 순손해다
# (k=150 절단하면 val +0.140).
python $W/pack_seedbag.py --k 0 --out $W/max/model.pth   # 285.4MB
python $W/pack_seedbag.py --k 0 --out $W/n50/model.pth   # 272.9MB

# ── 3~5. CARD1/2/3 — N50 팩을 기준으로 하이브리드 13 시드를 0.20 얹는다 ───
# --b4 가 base4(ridge+gbdt×2+eswf) 총합, --w 가 새 백 총합, 광류 몫은 1-b4-w 로 자동.
# 기준 팩 경로는 mkcard7.py 의 BASECARD (= work/n50_stage/model.pth).
TAGS="cnn7s7101 cnn7s7102 cnn7s7103 cnn7s7104 cnn7s7105 cnn7s7106 cnn7s7107 cnn7s7108 cnn7s7109 cnn7s7110 cnn9s9101 cnn9s9102 cnn9s9103"
python $W/mkcard7.py --b4 0.40 --w 0.20 --out $W/card1 --tags $TAGS   # 광류 .40 → Public 55.7603
python $W/mkcard7.py --b4 0.30 --w 0.20 --out $W/card2 --tags $TAGS   # 광류 .50 → Public 55.5866 ← 최고
python $W/mkcard7.py --b4 0.20 --w 0.20 --out $W/card3 --tags $TAGS   # 광류 .60 → 미제출 예비

# ── 6. 굽고 나면 반드시: 노트북 실행 → 빈 폴더 재현 → 버그체크 ───────────
CARD=$W/card2
cd $CARD && jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=7200 code.ipynb
bash $W/submissions/repro.sh $CARD ${CARD}_run   # 최대 절대차 ≤ 1e-6 아니면 assert 로 죽는다
python $W/bugcheck.py --dir $CARD                # C1~C5

# ── 7. 제출칸에 올리기 (버튼은 사람이 누른다) ───────────────────────────
cp $CARD/code.ipynb $CARD/model.pth $CARD/submission.csv /home/jovyan/submission/
md5sum /home/jovyan/submission/* $CARD/*
