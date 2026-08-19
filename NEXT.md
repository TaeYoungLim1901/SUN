# 다음 세션 시작점 — 모델이 누구든 이 파일부터

**작성:** 2026-08-19, Grok → **Opus가 이어감.**
`GROK_REVIEW.md` A–F 끝. 다시 하지 마라.

**읽기 예산:** 이 파일 + `RULES.md` §1·§8 + `DOMAIN.md` §9. 그 외는 지정 줄만.
DOMAIN/PLAN/ONBOARDING/PLAYBOOK/POSTMORTEM 전문 금지.

---

## 0. 지금 성적

| | validation RMSE | Public LB |
|---|---|---|
| persistence | 87.59 | - |
| 대회 베이스라인 3D-CNN+LSTM | 78.12 | - |
| 풍속 20개만 | 74.99 | - |
| Day1 스칼라 3종 | 64.76 | 59.6608 |
| 풀트리 6종 197 MB | 63.751 | 용량 거부 |
| 이전 최고 k=300 zlib 6종 | 63.784 | 57.2951 |
| pack100 풀트리+align→v2 | 63.730 | 57.4725 |
| E019 jump→v2 k=300 | 63.722 | 57.2985 |
| **최고 제출** E021 omni k=300 | **63.702** | **57.2905** |
| E032 cnn3a/b 교체+풀트리 복원 (8/19 제출, 사람이 눌렀음) | **63.5328** | 대기 |

공개 (사람, 8/18 밤): 56.9342 / 56.9949 / 57.1681 / 57.2215 / 57.2951이었으나 **6위까지 밀림**.
8/19: E021 **57.2905** (직전 최고 57.2951 대비 −0.0046, 갱신). **1위 56.7776, 갭 0.51.**
8/19 저녁: E032 제출(로컬 63.5328, 직전 infer 63.702 대비 −0.169). 점추정 Public 57.22 (범위 57.15~57.30). 점수는 다음 세션 확인.

**교훈:** 로컬 0.05~0.08 의 v2 자리 교체는 Public을 안 움직이거나 역전된다.
- pack100 로컬 −0.054 → Public **+0.177**
- jump 로컬 −0.062 → Public **+0.003** (동률)
채택은 같은 split Δ 만. Public과 로컬 val을 직접 비교하지 마라.
**채택선: 로컬 Δ ≥ 0.3.** 그 미만은 Public 노이즈에 묻힌다 (63.784→63.674 로컬 −0.11 이 Public −0.005).
보정(조각선형) 축은 기각 — validation 행이 시간순이 아니라 6h 이웃이 섞여 행 단위 분할이 전부 누수였다.
6h 이웃 연결요소로 분할해 재측정하면 전체·GBDT부 모두 음수(−0.118/−0.127).

용량: **제출 상한 700MB**(2026-08-19 주최측 답변, 기존에 알던 50MiB/76MB 는 틀렸다). k=300 트리 절단·zlib 압축은 이제 불필요.

---

## 1. 제출 폴더 — E036 경량 카드, 사람이 누르면 된다

`/home/jovyan/submission/` = **E036**. `python work/pack_big.py --k 200 --fp16` 로 만든다.

| 파일 | 크기 | |
|---|---|---|
| `code.ipynb` | 0.03MB | `work/pipeline.py` → `tools/mknb.py` |
| `model.pth` | **73.99MB** | k=200 절단 + zlib + CNN fp16, CNN 3개(cnn3b·cnn3c 128px·cnn3d) |
| `submission.csv` | 0.94MB | 팩 재추론이 직접 쓴 것 |

합계 **74.96MB** — 채점 이력이 있는 최대 크기(76.5MB)보다 작다. 재추론 약 70초.
bugcheck C1–C5 **전부 통과**. csv 3,868행·13열, min 271.0 / max 724.3, NaN 0.

가중치(고정, val 재적합 없음):
`img_ridge .075 + img_gbdt .18 + omni .19 + eswf .215 + cnn3b .113 + cnn3c .113 + cnn3d .114`
로컬 val **63.350** (팩 재추론 실측). 마지막으로 **채점된** 카드는 63.702 → Public 57.2905, 로컬 −0.35.
같은 족 미세 변화 전이비(로컬 −0.11 → Public −0.005)면 Public **57.26~57.29**, 점추정 57.28.

### 1.2 용량 손잡이 실측 (2026-08-19, fp16 공통)

| k | model.pth | val | |
|---|---|---|---|
| 풀트리 | 226.9MB | 63.26 | E032. **점수 안 뜸** |
| 300 | 81.5MB | 63.282 | fp16 비용 0.001 = 무손실 (추론이 이미 bf16 autocast) |
| **200** | **74.0MB** | **63.350** | 현 제출본 |
| 150 | 70.2MB | 63.441 | +0.159, 굳이 |

무게 분해(k=300 기준): `gbdt_eswf` 23.9 / `cnn` 32.5 / `gbdt_v2(omni)` 15.2 / `gbdt_v1` 6.7 MB.
공짜 절감 둘 — 가중치에 없던 `cnn96b`/`cnn96w3` 7.5MB 제거(손실 0), CNN fp16 12.5MB(손실 0.001).

백업: E035 102MB 판 `work/pack_e035_102mb/`, E032 227MB 판 `work/pack_e032_227mb/`,
Public 57.2905 판 `work/pack_win_5729/`, 57.2951 판 `work/pack_win_57/`.

pth 키 alias (`img_ridge`/`ridge_v1`) 유지. **갈라지지 마라.**
`model.pth` 의 `cnn` dict 는 태그별 `{"state","arch","px","mean","std"}` 를 받는다 (v3 = cnn3 계열 + 상하반전 TTA, px 96/128, state 는 fp16 가능).
`gbdt_v1`/`gbdt_v2`/`gbdt_eswf` 는 horizon 슬롯이 모델 리스트면 시드백으로 평균낸다 (`bag_predict`).

### 1.0 점수 미표시 진단 (2026-08-19, 두 건 연속)

8/19 제출 두 건(E032 227MB 풀트리, E035 102MB k=300+zlib)이 **둘 다 점수가 안 떴다**.
같은 날 E021(76MB)은 정상 채점됐다(57.2905). 우리 쪽 원인은 아래로 전부 배제됐다:

| 가설 | 실측 | 판정 |
|---|---|---|
| 용량 | 102MB 판도 안 뜸. 채점된 판은 70.8·76.5MB, 상한은 700MB | 기각 |
| 채점 런타임 | 콜드 캐시 빌드 해상도당 1.8분(64/96/128 합계 5.4분) + 추론 64초 | 기각 |
| csv 스키마·값 | bugcheck C1–C5 통과. 3,868행·13열·순서 test_ids 일치·NaN 0·271~724 km/s | 기각 |
| 노트북 자체 실행 | 빈 디렉터리에서 `jupyter nbconvert --execute` rc=0, 60~90초, csv md5 가 제출본과 동일(`d99405fa…`) | 기각 |

**남은 것은 플랫폼 쪽 채점 적체·실패다.** 주최측에 문의할 사실:
8/19 제출 2건 점수 미표시(227MB·102MB), 같은 날 76MB 건은 정상 채점, 제출 횟수 복구 요청.
전례 있음 — 주최측이 이미 「시스템 부하로 중복 처리된 제출 기록 삭제 + 횟수 복구」를 해줬다.
**답을 받기 전에 카드를 더 만들지 마라.** 슬롯만 태운다.

### 1.1 용량 — 상한 정정(2026-08-19)

**제출 용량 상한은 700MB.** 기존에 알던 50MiB/76MB 는 틀렸다. k=300 트리 절단·zlib 압축은 더 이상 필요 없다 — 위 카드는 둘 다 제거하고 풀트리로 복원했다.

---

## 2. 서버 예측

| 태그 | val RMSE | 비고 |
|---|---|---|
| `omni` | **66.476** | E021. 고정가중 v2 교체 63.674. hill 재가중은 63.761로 실패 |
| `jump` | 66.829 | E019. Public 57.2985 |
| `hssq65` | 66.543 | ρ(v2)=0.965. 재가중 축 고갈. 안 넣음 |
| `align` | 66.730 | E016. pack100 으로 Public 짐 |
| `v2_gbdt` / `eswf` | 67.022 / 67.028 | |
| `cnn96b` / `w3` / `s2` / `cnnmed` | 68.55 / 68.91 / 68.35 / **68.108** | med는 단독 이김, 결합 −0.021 기각 |
| `chfeat` / `arrive` / `cnnlate` | 67.249 / 67.886 / 69.461 | 기각 |
| `chtrack` / `rotter` / `imgpt` | 71.98 / 96.20 / 71.78 | 기각 |
| `cnn3a` / `cnn3b` / `cnn3c` | 67.485 / 67.119 / 67.028 | 폭2배 인코더+상하반전TTA, seed 811/823/829. cnn3c 는 128px |
| **`cnn3d`** | **66.881** | 96px seed 837 60ep, 랜덤초기화(SSL 안 물림). 단독 CNN 최고. E034 로 결합 −0.353 |
| **`omnibag`** | **65.607** | omni 시드 101·202·303 평균. omni(66.476) 대비 **−0.869**. 단독 스칼라 최고 |
| `omniflow` | 66.741 | omni 839 + 광류 270 = 1563차원. omni 보다 나빠 단독 기각, 결합 기여 미측정 |
| `flow` | 84.645 | 광류만 단독. 약하지만 잔차상관 0.721, 결합 5% 넣으면 −0.066 |

정직 분할: `preds/train_{comp,tstep,fold}.npy`. gap=109, 공유PNG=0.

---

## 3. 닫힌 것 (다시 하지 마라)

| 축 | 결과 |
|---|---|
| v2 자리만 갈아끼우기 (align / jump) | Public 무이득 또는 악화. ρ≥0.95 |
| last_wind persist / 부스트 | 악화 또는 도착에서 0 |
| ESWF·CNN y>600 가중 | 기각 |
| 예측 시프트 1–3 | 기각. peak 편향 +0.11 |
| chfeat / arrive / cnnlate / cnnmed 결합 | 기각 또는 −0.03 |
| 재가중만 (E014b) | 상한 63.655 |
| rotter / chtrack 단독 / k-means 분할 | 기각 |
| 197 MB pth | 용량 거부 |
| horizon별 결합 가중 | 홀드아웃 63.718 vs 공통가중 63.673, 악화 |
| 분위수 매핑 사후보정 | 69.00, 대폭 악화 |
| (pred,lw,통계) 2D GBDT 재보정 | 66.25, 대폭 악화 |
| horizon별 affine 보정(전 행) | 홀드아웃 +0.013/−0.004. 조각선형은 별개 축, 열려있음 |
| 조각선형 사후 보정(전체·GBDT부) | 6h 이웃 성분 분할 재측정 −0.118/−0.127(행 단위 분할 +0.390/+0.284 는 누수). 성분 2-fold×20 중앙 −0.458/−0.385, 이득 0/20. `pipeline.py` 원복, 제출 안 함 |

HSS 병목은 **도착**(lw 중앙 494, train 713행 lw<500 & peak>600)이지 지속이 아니다.
도메인은 맞았고, 구현이 839차원 클론을 양산했다.

---

## 4. Opus가 할 일 (이 순서로)

「고속구간 편향 −113」 가설은 **착시로 판명**되었다 — pred 로 조건한 회귀 기울기 1.001~1.033, E[y|pred]≈pred.
편향 보정 축이 아니라 도착 시각 정보 부족이다. 폐기.
조각선형 보정도 기각 확정(§3, `CLOSED_AXES.md`) — 재개 조건 없이는 다시 열지 마라.

1. **제출 점수 확인** 후 `NEXT.md` §0 갱신 (2026-08-19 카드, 점추정 57.22).
2. **`eswf` 시드백 완성** → `eswfbag` 성분으로 결합 재측정. omnibag 이 단독 −0.87 났으니 기대값 있음(재료: `preds/val_eswf_s{404,505,606}.npy`).
3. **다음 카드 = E034**(고정 가중, 로컬 **63.1794**, 제출본 대비 **−0.353** = 채택선 통과):
   `img_ridge .075 / img_gbdt .18 / omnibag .19 / eswf .215 / cnn3b .113 / cnn3c .113 / cnn3d .114`.
   `pipeline.py`·`pack_big.py` 는 이미 이 구성을 안다 (태그별 PX·시드백 평균). **막힌 것 하나: `omnibag_models.pth`** —
   `omnibag.py` 가 모델을 안 저장했다. 저장 코드 넣어 eswf 시드백 뒤에 재적합 중(대기열, PPID=1).
   그게 떨어지면 `python work/pack_big.py` → `pack.py --from-npy` 로 csv → `stage.py` → `bugcheck.py --repro`.
   cnn3e/f/g(128/96/128px 새 시드)가 끝나면 CNN 3개 자리를 더 좋은 시드로 갈아 재측정하고 나서 팩을 만들어라.
4. **SSL 사전학습은 보류.** `cnn3.py` 에 SSL 초기화가 아예 안 물려 있었다 (cnn3d 는 랜덤초기화).
   `ssl_enc_96.pth`(masked_mse 0.0626) 는 있지만 자체 사전학습의 규정 적합성은 `RULES.md` §7-4 미회답 상태다.
   주최측 답을 받기 전엔 제출 카드에 넣지 마라. 측정만 할 거면 `cnn3.py` 에 `enc.load_state_dict` 한 줄.

닫힌 것 재탕 금지. 실험 하나 = val+test npy + `experiments/log.md` 한 줄. 전 `CLOSED_AXES.md` grep.

---

## 5. 명령

```bash
python tools/remote.py check
python tools/submit.py
python tools/conform.py
python tools/push.py                        # 개인
python tools/push.py --share NEXT.md ...    # 팀 andy6609/sungnam-to-kaist
```

쿠키: 2026-08-18 12:14 UTC 갱신, refresh **8/19 11:56 UTC** 까지 — **오늘 만료.**
죽으면 `CONNECT.md` 1–4 → `remote.py auth`.
**`lab` 문서(302) 말고 Fetch/XHR `api/...` 를 Copy as cURL.** `copy as fetch` 는 쿠키 없음.

서버 실자원은 **CPU 2코어·RAM 8GB**(`cpu.max`=200000/100000, `memory.max`=8589934592). `nproc`/`free` 의 12코어·83GB 는 호스트 값이라 거짓 — 오늘 gtune2·cnn3b·OOF 가 무에러로 죽은 건 전부 이 8GB OOM.
백그라운드 잡은 **`setsid` + `start_new_session=True`** 로 띄우고 PPID=1 을 확인해라. `remote.py run --new` 반복으로 커널이 113개까지 샌 적 있고, 커널을 지우면 그 커널이 띄운 잡까지 같이 죽는다.

### 지금 서버에서 도는 것 (PPID=1 분리됨, 노트북 꺼도 계속)

- GPU: `cnn3.py 128 cnn3e 853` → `96 cnn3f 867` → `128 cnn3g 881` (각 60ep, 약 30분씩). ssl_mae·cnn3d 는 끝났다
- CPU: `eswf.py` 시드 404·505·606 순차 → eswf 시드백 재료 (`preds/val_eswf_s{404,505,606}.npy`)
- CPU 대기열: eswf 끝나면 `omnibag.py`(모델 저장판) 자동 시작 → `omnibag_models.pth`, 약 50분

백업 경로: 직전 제출본(57.2905) `work/pack_win_5729/`, 그 전 판(57.2951) `work/pack_win_57/`.

---

## 6. 하지 마라

규정: val을 train에 합침, pseudo-label, pretrained weight, 외부 데이터.
[공지3] ZIP/TAR·불필요 복사·외부 학습 금지. [공지2] 캐시는 `/home/jovyan` 하위.
닫힌 축: 27일 lag, 창 좁히기, 좌우 플립, test 시계열 복원, Public 단독 최종 선택.
**새 검토서 만들지 마라.** 이 파일만 고치고 `python tools/conform.py`.
