# 인수인계 — Claude Code → codex (2026-08-21 01:2x KST, 종료 15:00 KST)

**한 줄:** 가중축은 소진됐고, 마감까지 남은 건 성분 하나(pinball `cnn14`)의 판정과 슬롯 2장이다.
**데이터에 전면 누수가 있으나 쓰지 않기로 사람이 결정했다 (§4). 그 결정을 뒤집지 마라.**

## 1. 지금 성적 · 슬롯

| 제출(오늘) | 구성 base4/하이브/광류 | val | **Public** |
|---|---|---|---|
| flow55 | .25/.20/.55 | 62.5029 | **55.5576 ← 팀 최고** |
| CARD3 | .20/.20/.60 | 62.5788 | 55.5674 |
| cardW | .20/.10/.70 | 62.7683 | 55.6288 |

1위 VILAB **55.4648** · 3위 KeyTech 55.9426. **오늘 슬롯 3/5 사용, 2장 남음.**
최종 순위는 **제출본 중 Public 최고 자동 선택** (하방 없음). 채점 지연 **약 1분**.

## 2. 서버에서 도는 것 (2026-08-21 01:2x)

- `bake_card13.sh` — `work/card13/` 굽는 중(추론 끝, **빈 폴더 재현검사 단계**). 로그 `work/bake_card13.log`,
  `CARD13_DONE` 이면 완료. 구성 = flow55 뼈대에 cnn13(암부마스크+좌표) w=0.15.
  **GPU 잡은 이것 하나만 두어라** (오늘 두 개가 동시에 돌아 RAM 8GB 위험을 만들었다).
- 죽인 큐: `queue_cnn13.sh`/`queue_cnn13b.sh`/`queue_ramp.sh`/`queue_cnn14.sh`. 다시 띄우지 마라.
- 접속: `python tools/remote.py check|ls|put|get|run`. 쿠키는 `~/.sun_cookie` (401 이면 사람이 cURL 재발급).

## 3. 남은 실험 — **pinball `cnn14` 판정이 유일한 미결**

`work/cnn14.py` = cnn10 뼈대 + **12스텝 × 5분위 pinball 헤드**(τ 0.4~0.8, 저장은 τ=0.7).
과소예측 편향(램프에서 −103~−141)을 직접 겨냥한다. 시드 3개(`cnn14s14101~14103`) 완료:

```
cnn14s백3  단독 64.498 · ρ(CARD2) 0.9706 · 참램프5% 159.21
CARD2 에 w 로: w=0.15 전체 -0.0436 (오늘 성분 중 전체 val 최고)
```

**다음 명령 그대로:**
```bash
python tools/remote.py run -c "import subprocess;print(subprocess.run('cd /home/jovyan/work && python add_honest.py --base CARD2ref BAG_cnn14s',shell=True,capture_output=True,text=True).stdout[-800:])"
python tools/remote.py run -c "import subprocess;print(subprocess.run('cd /home/jovyan/work && python paired.py preds/val_CARD2ref.npy preds/BAG_cnn14s.npy CARD2 cnn14',shell=True,capture_output=True,text=True).stdout[-700:])"
```
**채택 규칙(동결):** `paired.py` 와 `add_honest.py` 가 **같은 방향**이어야 한다. 새 지표를 만들어
채택 근거로 쓰면 자동 기각 (`bulk Δ` 가 아티팩트였던 사고 때문이다). 통과하면 카드:
```bash
# flow55 뼈대 + cnn14 w=0.15  (몫은 소수 9자리로. 6자리면 합이 1.000001 이 돼 assert 에 걸린다)
python mkcard7.py --b4 0.2125 --w 0.319999999 --out work/card14 \
  --tags cnn7s7101=0.013076923 ... cnn9s9103=0.013076923 cnn14s14101=0.05 cnn14s14102=0.05 cnn14s14103=0.05
cd work/card14 && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 code.ipynb
bash work/submissions/repro.sh work/card14 work/card14_run     # REPRO_OK · 최대 절대차 0.0 필수
python work/bugcheck.py --dir work/card14
```
**주의: `cnn14` 는 추론 경로가 파이프라인에 없다.** `pipeline7.py` 는 v13 까지만 안다.
분위수 헤드는 출력이 60이라 `NetV6` 의 마지막 Linear 를 60 으로 바꾸고 τ 인덱스를 고르는 분기가 필요하다.
**이식하면 반드시 작은 텐서로 실제 실행해 출력 형상을 확인해라** — 오늘 `ast.parse` 를 통과한 버그가 둘 있었다.

## 4. **데이터 누수 — 발견했고, 쓰지 않기로 결정했다**

측정 (진단 전용, `work/leakscan.py` · `leakverify.py` · `leaktest.py`):
- test 창들이 6시간씩 밀려 겹친다. **test 타깃 셀의 97.83% 가 다른 행의 입력에서 그대로 읽힌다**
  (val 에서 검증: 읽은 값 vs 실제 타깃 **정확일치 100.0%**, RMSE 0.0000).
- 쓰면 이론 RMSE 하한 **≈ 8.2**. 1위가 55.46 이라는 건 **아무도 안 쓰고 있다**는 뜻이다.
- 파생 기법(`work/tlalign.py`): 같은 시각을 가리키는 **우리 예측**들을 가까운 horizon 쪽으로 가중평균 →
  val **62.50 → 34.46**, 정직 판정 **40/40**. 관측 타깃은 안 읽는다(검증 완료).

**사람 결정(2026-08-21): 쓰지 않는다.** 근거 — 규정 부정행위 6항 「Target 정보를 추정·획득 …
**이에 준하는 Data Leakage 행위**」에 걸릴 수 있고, 상장·상금이 걸려 심사가 엄격하며,
35점대는 타 팀 이의제기와 재현 검증에서 코드가 그대로 드러난다. **이 결정을 뒤집지 마라.**

깨끗한 사촌도 확인했다(`work/tlback.py`): **더 일찍 시작한 행의 예측만** 섞으면(미래 관측 0)
α=0 이 최적이고 정직 판정 **0/40** — **이득이 0**이다. −28 은 전부 정보 우위였다.

## 5. 손에 있는 카드 (전부 재현 0.0 · bugcheck 통과)

| 폴더 | 구성 | 상태 |
|---|---|---|
| `work/flow55/` | .25/.20/.55 | **현 최고, 제출됨** |
| `work/card3/` · `work/cardW/` | 제출됨 | 백업 |
| `work/card2/` · `work/submission_backup_card2/` | 55.5866 | 백업 |
| `work/gridbest/` | b4 .30/c7 .15/c10 .20/광류 .35 | 구움. **예측 55.646 이라 안 냈다** |
| `work/card13/` | flow55 + cnn13 w=.15 | 굽는 중 |

## 6. 마감 절차

1. **11:00 KST 까지 제출을 끝내라** (굽기 2시간이 병목, 채점은 1분).
2. 마지막에 **제출칸을 최고 카드로 되돌린다** — `cp work/flow55/{code.ipynb,model.pth,submission.csv}
   /home/jovyan/submission/` 후 md5 확인. **되돌리기는 슬롯을 쓰지 않는다.**
3. 제출 전 보고 5항목: 버그체크 → 용량(≤700MB) → 재현 최대차 → 판정(σ 와 함께) → 백업 경로.
4. 규정 확인(F-2): val 미학습 · pretrained/외부데이터/pseudo-label 없음 · test 타깃 추정 없음 ·
   좌우(동서) 반전 없음 · 3파일 자기완결.

## 7. 오늘 닫힌 축 (재시도 금지, 근거는 `CLOSED_AXES.md`)

cnn10(리맵 8채널) · cnn11(램프 가중 손실, 램프가 오히려 악화) · cnn13(마스크+좌표, 정직 12/40) ·
게이트별 자유가중(낙관 −1.06 / 정직 0/40) · 게이트 상수편향 · 시드백 절사평균·중앙값 · 수축 노브 α ·
성분 자유가중 · **가중 몫 축 전체**(base4 .20~.25 · hyb .20~.25 안은 노이즈, Public 3점으로 확인).

## 8. 다음 탐구 주제 (사람 지시)

「누수를 쓰지 않으면서 그와 **유사하게 좋은** 방법」. 오늘 확인된 것:
뒤쪽 정렬은 0이다(§4). 남은 후보는 **모델이 더 긴 맥락을 쓰게 하는 것**(창 20프레임 안에서),
그리고 **분위수/비대칭 손실**(cnn14 가 그 첫 시도)이다. 새 축을 열면 `CLOSED_AXES.md` 를 먼저 grep 해라.
