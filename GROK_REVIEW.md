# Grok 통합 검토서 — Opus 5 내일용

**작성:** Grok 4.6 (2026-08-17)  
**작업본:** `https://github.com/andy6609/sungnam-to-kaist` (private, WRITE). 로컬 `D:\hackathon\SUN` 은 구버전·부분본. **내일은 깃허브를 clone/pull 한 디렉터리에서만 일한다.**  
**이 파일이 검토 단일본이다.** `GROK_REVIEW_SUNGNAM.md` 는 폐기. 레포 `REVIEW.md`·`POSTMORTEM.md` 는 이미 반영된 감사 로그 — 다시 읽지 마라.

내일 시작: 깃허브를 pull/clone 한 뒤 **이 파일과 `ARENA.md`를 그 루트에 복사**하고, Claude Code 첫 메시지에 `GROK_REVIEW.md §0–§3 과 ARENA.md 만 읽고 스텝 A부터. DOMAIN 전문 금지.` 라고 적어라.

---

## 0. 토큰을 아끼는 방법 (창을 열자마자)

**읽는 것 (이것만)**

1. 이 파일 §0–§3. 본선 당일에만 §4.
2. 그 스텝이 지정한 파일의 **지정 줄 ±15줄**.

**읽지 마라 (다시 감사하면 하루가 사라진다)**

- `DOMAIN.md` 전문, `SUN_PLAN.md` 전문, `PLAYBOOK.md`, `notesC.md`, `notesD.md`, `ONBOARDING.md` 전문
- `ML_Week1-6_정리.md`, PDF, `audio/`, `transcript/`, `REVIEW.md`, `POSTMORTEM.md`
- 로컬 SUN 의 옛 `RULES.md` 스텁, `tools/data_probe.py` 등 (깃허브에 없음)
- 서브에이전트 Explore 「전체 재검토」

**쓰는 방식**

- 한 스텝 = **luna xhigh** 한 호출 = 파일 1–2개 = 검색·치환. 장문 리라이트 금지. sol·Grok은 사람이 그 이름을 말하기 전에는 쓰지 마라.
- Opus는 **diff만** 보고 `DONE/SKIP` 한 줄. 요약을 다시 쓰지 마라.
- `python tools/conform.py` · `python tools/kit.py` 가 해당 스텝의 수락 명령이면 그걸 실행하고 숫자만 본다.
- 스텝을 건너뛰거나 합치지 마라. 합치면 컨텍스트가 커진다.

**보고 형식 (매 스텝 끝)**

```
DONE: path:line — 한 줄
SKIP: id — 이유
```

---

## 1. 이미 끝난 것 — 다시 손대지 마라

깃허브가 SUN 초안·1차 Grok 검토·레포 `REVIEW.md` 를 이미 흡수했다.

| 주제 | 현재 상태 |
|---|---|
| H, 출력 | **12 horizon / 72h**. `N_HORIZON=12` |
| 지표 | **전체 셀 단일 RMSE** `sqrt(mean((y-p)²))`. 반례 `sqrt(50)` 있음 |
| 예측 NaN / shape | `ValueError`. 20열 예측 침묵 통과 불가 |
| 27일 기저·lag 피처 | **제출 불가** (`DOMAIN` §9.1). E000 폐기 |
| 사전학습 weight | 금지. 아키텍처 차용+랜덤 초기화 허용 |
| 제출 | `code.ipynb` + `submission.csv` + `model.pth`, 하루 5회 |
| 입력 | 20시점×193/211 PNG 512² + wind 20. `inputs` 41열, target 13열 |
| `decompose` | 우리 모델만. 공개 커널 금지 (docstring) |
| `calibrate` | 시드-널 제출 대체. kit에 있음 |
| 설명회 규정 | `RULES.md` 가 단일 기준 (`df.pdf`) |
| kit `score`를 horizon-평균으로 되돌리기 | **하지 마라.** 옛 SUN 검토 §5-A 는 폐기 |

`REVIEW.md` 치명 4건 중 kit 쪽은 끝. 남은 것은 아래 §3 뿐이다.

---

## 2. 현재 사실 (외울 것. 파일 다시 열지 마라)

- 본선 2026-08-18–21. **A100-SXM4-40GB 1대**(측정 확인). test 3,868행. Public/Private = test 절반(명목 12.5+12.5).
- **2026-08-18 본선 공지(`ADDRULES.md`) 반영 완료.** 델타는 `RULES.md` §8. 핵심 셋:
  validation 을 train 에 합치면 **실격** / pseudo-labeling **NO-GO**(「추정」) / 샘플코드는 `/home/jovyan/sample_code.ipynb` Duplicate.
- **Day 1(08-18) 종료 상태.** 제출 1장 준비 완료(서버 `/home/jovyan/submission/`, 재현 비트일치 확인).
  validation RMSE: persistence 87.59 / 공식 베이스라인 78.12 / 풍속만 74.99 / 결합 **64.76**.
  LB 1위 57.61 · 5위 62.34 · 6위 72.98. 다음 순서는 `SUN_PLAN.md` Day 2 절.
  서버 접속은 `CONNECT.md`, 제출 전 검사는 `python tools/submit.py`.
- Persistence 보조지표 = **last-value 72h 유지** (`persistence_last`). 27일 지속성은 입력에 없다.
- 영상 창 span = `(20-1)×6h = 114h` → 자전 ≈ **63.3°**. 66.7°는 21프레임일 때.
- CH 피처는 **target 시각 기준** t−4..7d. t−7d는 단기 horizon에서 창 밖 (`DOMAIN` §9.2 표, 산술 맞음).
- PNG에 FITS 헤더 없음 → aiapy 노출·열화 **불가**.
- Box-Cox는 피크 RMSE −22%, **timeline RMSE +7**. 대회 지표는 timeline.
- Swin이 timeline에서 poly를 이김. 「선형만」으로 접지 마라.
- 문헌 68/87은 ICME 제거·다른 horizon. 눈금은 본선 `persistence_last` 실측만.

---

## 3. 내일 작업 순서 — 6스텝 후 본선

각 스텝: **읽기 예산 → 패치 → 수락**. 다음 스텝으로 넘어가기 전에 수락을 통과시켜라.

### 스텝 A — 널 제출 모순 (제출 예산 보호)

**읽기:** `SUN_PLAN.md` 128–129, 136–139, 255. `ONBOARDING.md` 288–291. `REPOS.md` 126.  
**읽지 말 것:** PLAN 나머지, ONBOARDING 파트1.

| 위치 | 동작 |
|---|---|
| `SUN_PLAN.md:128-129` | 「널 조작 제출로 σ 실측」→ **삭제**. 이미 아래 136행에 `calibrate` 가 있다 |
| `SUN_PLAN.md:255` | 「널 조작(시드만) 제출…」→ `kit.calibrate(CV, Public) 잔차 σ = 채택선. 시드 변동은 로컬 OOF` |
| `ONBOARDING.md:288-291` | 시드-널 제출 문단 → `채택선은 kit.calibrate. 시드는 로컬 OOF` |
| `REPOS.md:126` | 「널 조작 제출로 실측」→ `calibrate 로 실측` |

**수락:** 위 네 위치의 **권고 문장만** 사라졌는지 확인. `calibrate` 설명에 남은 「시드만 바꾼 제출은 널이 아니다」는 지우지 마라.

**luna 스펙 한 줄:** `SUN_PLAN.md, ONBOARDING.md, REPOS.md 에서 시드-널 제출 권고만 교체. 다른 문장 금지.`

### 스텝 B — REPOS가 죽은 피처를 다시 연다

**읽기:** `REPOS.md` 120–131만.

- 129: CHIMERApy 즉시 조달 → `제출 불가 — 171Å 미제공. 193/211 PNG 임계·k-means만`
- 130: HUXt 기저 1장 → `제출 불가 — 외부 WSA/FITS`

**수락:** `REPOS.md` §5에 `즉시 조달`·`HUXt` 실행 지시 0건.

### 스텝 C — RULES 주최 팁 주석 (원문은 지우지 마라)

**읽기:** `RULES.md` 201–207만.

`§6-4` 풍속 200·27.27 피처 문장 **아래**에만 추가:

```
⚠ 슬라이드 원문이다. 평소 전형은 300–400 km/s. clip 하한은 제공 wind의 관측 min/max.
⚠ 27.27일 위상 피처는 날짜가 없어 계산 불가(DOMAIN §9.1). 상수 27.27을 열로 넣지 마라.
```

**수락:** 원문 200·27.27은 남아 있고, 바로 다음 줄에 ⚠ 두 개.

### 스텝 D — 잔존 문장 6개 + conform이 다시 물도록

**읽기:** 아래 줄만. DOMAIN 절 전체 금지.

| 위치 | 교체 |
|---|---|
| `SUN_PLAN.md:230` `66.7°` | `63.3°(114h)` |
| `SUN_PLAN.md:234` aiapy annulus + 1/cosθ | `PNG라 aiapy 불가. disk-relative / 방사 프로파일 근사만 (DOMAIN §9.3)` |
| `DOMAIN.md` §10 표 「사실상 같은 설정」 | `영상 6h(대회) vs 12h(Son). 출력 72h/12스텝은 같음` |
| `DOMAIN.md:233` 27일 기저 85–100 첫날 실측 | `제출 기저는 persistence_last 뿐 (§9.1)` |
| `DOMAIN.md:310` 27일 대비로만 평가 | `persistence_last 및 V-only 모델 대비로 평가` |
| `DOMAIN.md:387` Box-Cox 「가장 싼 레버가 살았다」 | `가능. 단 OOF에서 kit.score가 내려갈 때만` |

**같은 luna 호출에** `tools/conform.py` `RULES` 추가:

```
("M29", "창 자전각 66.7", r"66\.7\s*°", "63.3° (114h, 19간격)")
("M07b", "Son=대회 동일", r"사실상 같은 설정", "영상 cadence 다름")
```

M07 `{0,20}` 은 건드리지 말고 리터럴 규칙을 더하라.

**수락:**

```
python tools/conform.py --selftest
python tools/conform.py          # 위반 0
```

고친 6문장이 다시 들어오면 1이 나와야 한다.

### 스텝 E — factcheck 빈 판정 PASS 봉인

**읽기:** `tools/factcheck.py` 233–241, 276–280.

`if rows is None:` 을 `if not rows:` 로 (None과 `[]` 둘 다 실패, 캐시 안 함).  
selftest에 `assert parse_verdicts("```verdicts\\nz\\t미확정\\n```") == []` 다음에  
「`[]` 는 main에서 failed」주석 + 빈 리스트를 성공으로 치지 않는다는 assert가 이미 파서에 있으면 main 경로만 고치면 된다.

**수락:** `python tools/factcheck.py --selftest`

### 스텝 F — 리허설/외부데이터 한 줄

**읽기:** `SUN_PLAN.md` 33–35.

Surya-bench 문장 끝에: `로컬 손연습만. VM dataset/ 및 정규화 통계에 섞지 마라 (§1 1항). 리허설 스키마는 41열/13열.`

**수락:** 해당 절에 `VM`·`41열` 존재.

---

## 4. 본선 당일 (스텝 A–F 끝난 뒤. 문서 작업과 섞지 마라)

순서 고정. 모델은 4번 이후.

```
1. dataset/ 로 스키마 확인 (행 수 9607/1199/3868, 열 41/13, PNG 512)
2. wind_XX 히스토그램 → 채움값이면 그 값만 NaN. 외부 OMNI 센티널 이식 금지
3. persistence_last 로컬 RMSE + horizon별 갭. 제출은 한도 보고 0~1장
4. train 인접창 겹침으로 블록 CV (시각 없으면 carrington_block 포기). 「탐색기」로 적어라
5. V-only GBDT/1D 가 persist를 긴 horizon에서 이기는지
6. 이기면 193/211 상대 CH(4×3, target 기준 지연, 창 밖은 마스크) + 소형 CNN 한 계열
7. 모든 실험 oof_*.npy + test_*.npy + log 한 줄
8. 채택선 = calibrate. Public 최고점 단독 최종 금지
```

없는 지연을 0으로 채우지 마라. 좌우 플립 금지. test 해시로 시계열 복원 금지.

---

## 5. 하지 말 것 (내일 전체)

- DOMAIN/PLAN 장문 다듬기, 톤, 「더 정확한 표」재작성 (`notesC` 반영은 본선 후)
- kit에 함수 추가. `score` 를 horizon-평균으로 되돌리기
- Surya weight, CHIMERA, HUXt, WindNet 스크래치, 고해상도 분할기
- Box-Cox/분위수 매핑을 기본 후처리로
- 27일 lag 피처, B0, 흑점, OMNI N/T/B
- 시드만 바꾼 리더보드 제출
- 로컬 SUN 의 `data_probe.py`/`omni_fill.py`/`make_submission.py`/`AGENTS.md` 를 깃허브에 복사하며 문서 체계를 두 벌로 만들기
- 이 검토를 다시 쓰거나 세 번째 검토서 만들기

---

## 6. P1 — 코드 쓰지 말고 기억만 (파일 열지 마라)

- Ω 14.713 = 적도율. 캐링턴 항성 14.184, 회합 13.199. 오늘 영상에 Δφ를 **서쪽**으로 밀지 마라. CH는 **t−τ 영상의 CM**.
- 193Å ≈ 1.6 MK (1.2 아님).
- z-score는 꼬리를 안 뭉갠다.
- PNG에서 `0.35×median` 은 오프셋 있으면 원래 임계가 아님. rank/k-means.
- `private_sigma(p=0.5)` 는 무작위 분할 전제. 기간 분할이면 `calibrate`만.
- 주최 Persistence 한 숫자로 「모델 무의미」판정 금지 — 단기만 잘하는 모델이 통과한다.
- WSA 식 `V_min+V_max/(1+fs)` 코딩 금지. 안 쓴다.
- 1AU SIR 충격이 없는 것은 평균일 뿐.

---

## 7. 한 줄

깃허브가 규정·지표·27일 사망은 맞다.  
내일은 **문서가 그 결론을 뒤집는 여섯 군데**만 고치고, 본선은 persist → V-only → CH/CNN 이다.
