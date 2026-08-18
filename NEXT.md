# 다음 세션 시작점 — 모델이 누구든 이 파일부터

**작성:** 2026-08-18 밤, Day 1 종료 직후. 메인 모델이 Opus 5 → **Grok-4.6** 으로 교체되는 시점.
이 파일이 인수인계 단일본이다. `GROK_REVIEW.md` §3 의 문서 스텝 A–F 는 **전부 끝났다 — 다시 하지 마라.**

**읽기 예산:** 이 파일 + `RULES.md` §1·§8 + `DOMAIN.md` §9. 그 외는 필요할 때 지정 줄만.
DOMAIN/PLAN/ONBOARDING/PLAYBOOK/POSTMORTEM 전문 금지. 「전체 재검토」 금지 — 지난 실패가 전부 그거였다.

---

## 0. 지금 성적

| | validation RMSE | Public LB |
|---|---|---|
| persistence (마지막 관측 유지) | 87.59 | - |
| 대회 공식 베이스라인 3D-CNN+LSTM | 78.12 | - |
| 풍속 20개만 | 74.99 | - |
| **제출본** (영상 스칼라 GBDT/Ridge 3종 결합) | **64.76** | **59.6608** |
| 로컬 최고 결합 (cnn96 포함, 미제출) | 64.525 | - |

공개 LB 상위: 57.61 / 59.34 / 59.55 / **우리 59.66** / 60.20 / 62.34 / 72.98 …
6위(72.98)와 10점 절벽. 우리는 절벽 안쪽 4~5위권.

**Public 이 로컬 val 보다 5점 낮다.** 두 숫자를 직접 비교하지 마라. 채택 판단은 **같은 split 안의 Δ** 로만 한다.

---

## 1. 서버에 이미 있는 것

접속·쿠키 유효 확인됨 (`python tools/remote.py check` → A100-SXM4-40GB, 디스크 66.8/95.8 GB free).
**서버 시각은 UTC.** KST = UTC+9 (서버 로그 09:36 = 18:36 KST).

`/home/jovyan/work/preds/` 에 `val_*.npy` (1199×12) + `test_*.npy` (3868×12) 쌍으로 저장돼 있다:

| 태그 | val RMSE | |
|---|---|---|
| `persist` | 87.588 | 기저 |
| `ridge` / `gbdt` / `ridge_gbdt` | 76.0 / 75.277 / 74.986 | wind-only (E003) |
| `img_ridge` / `img_gbdt` / `img_blend` | 69.441 / 68.628 / **66.796** | 영상 스칼라 v1, 2×2 격자 (E004) |
| `v2_ridge` / `v2_gbdt` / `v2_blend` | 83.459 / **67.022** / 69.703 | v2, 4×4 격자 (E005). Ridge 는 839차원에서 붕괴 |
| `cnn96` (+`cnn96b`, `cnn96s2`) | 70.249 | 96px residual CNN+GRU (E006) |

`kit.hill_climb` 가 고른 최고 결합 = `cnn96 0.2 + img_blend 0.4 + v2_gbdt 0.4` → **64.525**.
**제출 노트북은 이게 아니다.** `submission/code.ipynb` 는 외부 import 없이 자립해야 해서
CNN 을 빼고 `ridge_v1 0.25 + gbdt_v1 0.25 + gbdt_v2 0.50` = 64.76 을 박아 넣었다 (`work/pipeline.py`).
→ **cnn96 을 노트북 안으로 넣기만 해도 −0.24 는 공짜다.** 단 `model.pth` 크기·추론 시간 확인할 것.

`/home/jovyan/submission/` = code.ipynb / model.pth (87MB) / submission.csv, 08-18 09:36 UTC 판.
로컬 `submit/` 은 그 백업 사본이다. 업로드용이 아니다.

---

## 2. 안 돌아간 것 (다음 세션이 주울 것)

- **E010 ESWF 미실행.** `work/eswf.py` 는 다 짜여 서버에 올라가 있는데 (`/home/jovyan/work/eswf.py`,
  09:20 UTC) **실행 로그가 없다.** 중심 자오선 ±15/30/60° 슬라이스 CH 면적 + 속도가설 7개 다중투영.
  근거는 DOMAIN §1.2·§1.3. 제일 싼 다음 카드다.
- **E009 중단.** horizon별 affine 보정을 하려다 `KFold(shuffle=True)` 가 닫힌 축(창 중첩·PNG 공유
  누수)이라 멈췄다. 정직한 그룹 분할이 먼저다.

---

## 3. 다음 레버 (이 순서로)

1. **정직한 분할부터.** train 은 인접 창이 겹치고 PNG 를 공유한다. 랜덤 KFold 는 누수다.
   블록/그룹 분할이 없으면 아래 2·3의 숫자를 믿을 수 없다. 이게 1번인 이유다.
2. **고속구간 편향.** `>600 km/s` 가 셀의 7.4% 인데 제곱오차의 34%, 편향 **−113 km/s**.
   여기를 못 건드리면 59 아래로 못 간다. (분포 보정·분위수 회귀·고속 전용 헤드 중 택1,
   단 Box-Cox 는 timeline RMSE 를 +7 시킨 전력 있음 — `GROK_REVIEW.md` §2.)
3. **E010 실행** (위 §2).
4. **다양성은 CNN 쪽에만 남았다.** GBDT 변형끼리 잔차상관 0.87~0.97 = 고갈.
   `kit.resid_corr ≥ 0.84` 면 그 계열은 즉시 접는다. 토론하지 않는다.

새 실험 전에 `CLOSED_AXES.md` 를 **사람이 직접 grep** 한다. 자동 중복검사 믿지 마라.
실험 하나 = `oof_*.npy` + `test_*.npy` + `experiments/log.md` 한 줄. 글 말고 파일이 판정이다.

---

## 4. 명령 모음

```bash
python tools/remote.py check                      # 서버·GPU·쿠키 상태
python tools/remote.py put work/x.py work/x.py    # 올리기
python tools/remote.py get work/x.log .           # 내리기
python tools/remote.py run -f work/x.py           # 실행 (짧은 것만)
# 학습은 붙잡지 말고 백그라운드로:
python tools/remote.py run -c "import subprocess; subprocess.Popen('cd /home/jovyan && nohup python work/eswf.py > work/eswf.log 2>&1 &', shell=True)"

python tools/submit.py     # ★ 제출 버튼 누르기 전 필수. 서버 버그검사 15항목
python tools/conform.py    # 문서 회귀 검사 (1초, 무료)
python tools/kit.py        # 채점기 자체 점검
python tools/push.py       # main → 개인 레포. --share 로만 팀 공개
```

쿠키 만료되면 `CONNECT.md` 의 1–4 (사람이 30초) → `python tools/remote.py auth`.
**같은 계정으로 다른 기기에서 로그인하면 이 쿠키가 죽는다** (RULES §1).

---

## 5. 하지 마라

규정 (어기면 실격):
- `validation` 배열이 `fit()` 근처에 오는 것 — 검증·모델선택·하이퍼파라미터 조정 **전용**
- pseudo-labeling (test target 추정·획득)
- pretrained weight, 외부 데이터/FITS/OMNI

닫힌 축 (이미 죽었다):
- 27일 지속성·lag 피처 (날짜가 없다 — DOMAIN §9.1), B0·흑점
- 서쪽 림 스트립 (E005/E007 — 근거 없던 피처였다), 창을 좁혀 전역요약을 **대체**하기 (E007)
- 시드만 바꾼 제출, 좌우 플립, test 해시로 시계열 복원
- Public 최고점 단독으로 최종 제출 고르기

절차:
- **새 검토서·새 요약 문서 만들지 마라.** 문서가 갈라진 게 이 레포의 주 실패원인이다 (`POSTMORTEM.md`).
  고칠 게 있으면 해당 파일을 고치고 `python tools/conform.py`.
- `GROK_REVIEW.md` §3 문서 스텝 A–F 는 끝났다. 다시 돌지 마라.
- 제출 하루 5회. 학습이나 노트북 실행이 도는 중에 제출 버튼을 누르면 깨진 제출이 나간다.
