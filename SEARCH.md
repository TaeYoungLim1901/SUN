# 압도적 개선 서치 · 모델 검토 · 근거 감사

**작성:** 2026-08-19 23:40 KST, Grok. 채팅에만 있던 결과를 파일로 옮긴 것.
**팀 `--share` 금지.** `ARENA.md`·`GROK_REVIEW.md` 와 같은 개인 문서.
`GROK_REVIEW.md` A–F 재검토가 아니다. 실험·제출·서버 잡은 이 파일로 시작하지 마라 — 레버 순서는 [NEXT.md](NEXT.md) §4.

기록 시점 서버 (읽기만, 안 죽임): GPU `cnn3wm2`(128px) + `cnn3px256` 동시, CPU `work/refit150/pipeline_refit.py`.

---

## 0. 스냅샷 (기록 시점)

| | val | Public |
|---|---|---|
| 1위 | - | 56.5416 |
| **우리 최고** E039 같은 7성분 k=150 | 63.441 | **56.8181** |
| E040 같은 성분 k=100 | 63.602 | 56.8257 |
| E036 같은 성분 k=200 (제출 폴더·Private 헤지) | 63.350 | 56.8752 |
| E038 k=300 | 63.282 | 56.9886 |
| E037 k=300+omnibag | 63.207 | 57.0295 |

갭 1위까지 **0.276**. 채택선: 성분 교체 로컬 Δ ≥ 0.3 (2σ). k 축은 별 회귀 — 로컬이 역지표 (r=−0.79). 섞지 마라.

고정 가중 (E036/E039, 합 1.000):
`ridge .075 + gbdt_v1 .18 + omni .19 + eswf .215 + cnn3b .113 + cnn3c(128) .113 + cnn3d .114`

---

## 1. 이미 서치·실험된 것

| 상태 | 축 | 숫자 |
|---|---|---|
| Public 최고 | 같은 7성분, k=150 (E039) | 56.8181 |
| 헤지 | k=200 (E036) | 56.8752 |
| **닫힘** | k 절단 | 300→150 단조, 100에서 반전(+0.0076). GBDT 부 k=150 고정 |
| **닫힘** | v2 자리만 교체 | align +0.177, jump +0.003, omnibag +0.154 |
| **닫힘** | 재가중만 / 조각선형 / 분위수 / 2D재보정 / 고정 시프트 / Rotter / 창 좁히기 | [CLOSED_AXES.md](CLOSED_AXES.md) |
| 대기 (학습 없음) | cnn3h 블렌드 **추가** → eswfbag → flow 진단 | 채택: 6h 2-fold×20, Δ≥0.3, 이득 16/20 |
| GPU 기록 시점 | 20ep 스케줄 `p20med/p20z`, 폭2배 `wm2`, 256px | **GAP+GRU 같은 족** — 아래 §2가 아님 |
| 보류 | SSL (`RULES.md` §7-4 미회답), GAP 위 DILATE | 2순위 |

---

## 2. 압도적 새 족 (병렬 서치 합의)

구멍: `EncoderV3` 가 `AdaptiveAvgPool2d(1)` 로 **경도를 지운다.** 창 안 자전 63° (DOMAIN §9.5). HSS 오차는 진폭이 아니라 **도착**(lw~494, peak>600 713행). 시프트·조각선형·재가중이 실패한 이유.

채택은 전부 **추가 자리**. v2 자리 교체 금지. ρ(v2)<0.9 일 때만 예외.

| # | 새 족 (cnn3/omni/eswf 클론 아님) | 기대 로컬 Δ | 왜 ≥0.3 | 첫 측정 (GPU/CPU 끝난 뒤) |
|---|---|---|---|---|
| 1 | **공간 유지 CNN** — GAP 제거, 6×6 맵 / ConvGRU / (2+1)D, horizon별 헤드 | −0.35~−0.70 | 공식 3D-CNN 78은 64px·용량 부족이지 3D 기각이 아님. cnn3 교체만으로 Public −0.415 | `AdaptiveAvgPool2d(6)` + 좌표채널, 128px med 20ep 한 시드 → `blend_group.py`. GAP 남기면 클론 |
| 2 | **극/캐링턴 스트립** — 20프레임을 경도×위도 워터폴 | −0.40~−0.80 | 남은 시계. chtrack 70스칼라 71.98로 죽음, **픽셀 궤적**은 미측정 | 64px polar 캐시 → occupancy GBDT 프록시. Δ>0.12 이고 ρ(eswf,cnn3)<0.9 이면 CNN |
| 3 | **탄도 쿼리 / ToF occupancy** — j* 에서 공간 gather, **창은 안 좁힘** | −0.30~−0.50 | E007은 프레임 삭제라 기각. ESWF는 이 물리의 스칼라(결합 −0.44) | 전체 프레임 유지. ρ(eswf)>0.93 이면 ESWF 클론 → 중단 |
| 4 | **2스트림** — 밝기 + CH 마스크 또는 광류를 **채널**로 | −0.30~−0.55 | SpeedNet은 CH 바이너리 맵. 우리 k-means/flow는 요약 GBDT로만 죽음(ρ 0.72) | 마스크는 rank/Otsu. `0.35×disk median` 금지 (DOMAIN §9.4). omniflow 1563차원 재탕 금지 |
| 5 | **seq2seq 디코더** — 창 안 wind 20 재구성 + 12스텝 | −0.30~−0.60 | NYUAD 이득의 핵심은 디코더. **27d·GoogleNet 가중은 불법** | 이력 인코더는 wind 20만. 27일 lag 넣으면 실격 |

문헌 (규정 필터 후): WindNet 2020SW002478, SpeedNet-EUV 2025EA004523, Son 2023 ApJS, Collin 2410.05068 (Box-Cox는 peak↑ timeline↓ 68.1→75.1), NYUAD 2507.17298 (27/54d·ImageNet 경로 제거한 디코더만), DILATE 1909.09020.

**1위가 가진 것 (추론):** 7개 고상관 전문가가 아니라 **경도를 텐서로 쓰는 한 족.** 27일 lag는 전 팀 불가.

---

## 3. 지금 돌리면 안 되는 것

cnn3 폭/시드/256px (이미 GPU) · k 추가 제출 · v2 자리 교체 · omniflow 단독 · Box-Cox 를 timeline RMSE 레버로 · GAP 남긴 채 DILATE · ImageNet/timm 가중 · 27d/B0/흑점/OMNI N·T·B · val을 train에 합침 · pseudo-label · test 시계열 복원 · 좌우 플립 · aiapy · 제출 카드에 SSL.

`cnn3h` 추가·`eswfbag`은 NEXT §4.4 측정이지 이 표의 새 족이 아니다. 기대 블렌드 <0.3.

---

## 4. 현재 모델 검토 (E036/E039)

같은 7자리, k만 200 vs 150. 둘 다 **omni 단일시드**, omnibag 아님. E034 npy 63.1794는 omnibag 가중이므로 팩 숫자로 쓰지 마라.

| 슬롯 | w | X | 타깃 | TTA |
|---|---|---|---|---|
| ridge_v1 | .075 | v1 스칼라 215d | 절대, 12h 공동 RidgeCV | 없음 |
| gbdt_v1 | .18 | **ridge와 동일 X** | 절대, horizon별 | 없음 |
| omni (gbdt_v2) | .19 | 839+t−4d+j* = 1293d | 잔차 y−last | 없음 |
| eswf | .215 | CM±15/30/60 + 속도가설 7 + ok 플래그 | 절대 | 없음 |
| cnn3b | .113 | 96px, NetV3 GAP→biGRU | 잔차 + last | 상하 평균 |
| cnn3c | .113 | 128px, 같은 망 | 잔차 + last | 상하 |
| cnn3d | .114 | 96px seed 837, SSL 안 물림 | 잔차 + last | 상하 |

포화: **omni 자리** (Public 3연속). ridge+gbdt_v1은 피처가 같다. CNN 3개는 한 족 (E035: e/f가 b+c+d를 못 이김).

이 코드가 못 나타내는 것: 디스크 위 CH 이동(GAP 1×1), 피크 스텝(셀 MSE만), SIR 압축(N·T·B 없음. ESWF 7가설은 같은 요약의 다른 j).

### 결함 — 새 족 없이 Public ≥0.1 가능

1. HistGBDT `early_stopping=True, validation_fraction=.12` 행 무작위. 6h 이웃 누수. 전량 재적합 없음. k=150 이 사고 정규화였을 가능성. **`refit150`이 이 축.**
2. 팩 CNN이 OneCycle 정점(ep18) 전 스냅샷(ep9–12). 20ep 프로브가 이 축.
3. `pipeline.py` 소스 기본 가중은 E032 (`cnn3a/b`). NEXT §0.3은 `norm="med"`·`wm` 을 받는다고 썼는데 **로컬 `pipeline.py` 에는 없다.** cnn3h를 이 노트북으로 팩하면 z-score를 med 네트에 씌운다.
4. `pack_big.py` 주석·version 이 omnibag인데 기본은 단일 omni. `--bag` 한 번이면 E037 (+0.154).
5. C6는 재추론이 아니다. `stage.py`가 같은 csv를 npy로 저장한 뒤 비교. `pack.py --from-npy` 는 구형 E021 가중 — 새 카드 금지.

재현 GO: 현 `pipeline.py` → `mknb.py`, 빈 폴더에서 `code.ipynb+model.pth` 실행, 후보 csv 최대차 ≤1e-6.

---

## 5. 기각·채택 근거

8/19 규칙: 닫으려면 (a) 부분집합 표 + (b) 홀드아웃 분할 2종 부호 일치 + (c) 6h 이웃 연결요소. 행 셔플/5-fold는 누수 (E024 +0.39 vs 성분 −0.12).
프로토콜 완결은 **E024b뿐.** 아래 KEEP은 「숫자가 안 뒤집힘」이지 「(a)(b)(c) 완료」가 아니다. luna는 후자라 거의 전부 WEAK로 표기 — 방향 판정과 혼동하지 마라.

### KEEP (뒤집히지 않음)

| 축 | 근거 |
|---|---|
| v2 자리 클론 교체 | Public 3회: +0.177 / +0.003 / +0.154 |
| 조각선형 사후 보정 | 성분 5-fold −0.118/−0.127, 2-fold×20 중앙 −0.458, 이득 0/20, 구간 전부 ≤0 |
| Rotter 선형 | 96.20 > persist 87.59 |
| 창 좁히기 E007 | 전 horizon 악화. 재개는 j* 추가 피처 → 이미 omni |
| Ridge 839차원 | val 83.5 붕괴 |
| E004 영상 스칼라 | wind-only 74.99 → 66.80 (−8.2) |
| E036 팩 > E021 | 로컬 −0.352, Public −0.415 (CNN 족 교체. 1.18:1 기울기는 쓰지 마라) |
| k 곡선 최소 150 | 4점 Public 56.989 / 56.875 / **56.818** / 56.826 |

### WEAK (방향은 맞음, 6h 미측정)

chtrack 단독 71.98 · 재가중 천장 63.655 · y>600 가중 · 고정 시프트 E011e · omniflow **단독** 66.741 · E030 단독 −0.869 · E021 자리 교체 Δ 0.077 (0.3 미달, Public −0.0046 = 노이즈)

### WRONG / 너무 넓게 닫힘

| 판정 | 이유 | 재개 |
|---|---|---|
| E023 horizon 가중 기각 | 「val 절반×40」= 행 분할. E024와 같은 누수 | `blend_group.py` 로 공통 vs horizon별, 6h 2-fold×20 |
| cnnmed = med 사망 | 측정은 옛 96px CNN 결합 −0.021, ρ=0.920. cnn3h 128px med **66.384**은 다른 축 | NEXT §4.4 그대로. 옛 슬롯은 닫힌 채 |
| E034를 한 장으로 채택 | npy 63.1794는 **omnibag .19**. 팩 E037은 Public +0.154. CNN 자리와 bag을 섞어 적음 | CNN 자리 / bag 을 분리해서만 인용 |
| E011e로 타이밍 보조손실 기각 | 닫힌 것은 **고정 출력 시프트**. peak 편향 +0.11, MAE 3.1은 분산 | DILATE/peak-step은 별 축. GAP 제거 다음 |
| E028 Δ0.14 미제출 | 그 카드(cnn96→cnn3a/b)에는 맞음. E036은 더 큰 족 교체(−0.352)라 0.3을 넘고 Public −0.415 | 0.17로 낮춘 뒤 E037을 낸 것이 축 혼동 |

E030 단독 순위는 산다. **v2 자리에 넣는 채택은 죽었다.** 추가 자리는 ρ(omni)<0.9 일 때만.

### 문서 버그 (감사 추적용)

| 혼선 | 사실 |
|---|---|
| NEXT §0 / CHECKLIST 「E034 로컬 −0.143, Public +0.154」 | 그건 **E037** (63.207 / 57.0295). E034는 npy 63.1794, Public 없음 |
| `experiments/log.md` E036 두 줄 | 용량 스윕 vs 채점된 75MB 팩 |
| E040 ID 두 개 | k=100 팩 (56.8257) vs cnn3h med (66.384) |
| NEXT §0.1 vs §0.4·§4 | k 「닫힘·150 고정」 vs 「E039는 진단, prefix 곡선 먼저」 |

---

## 6. GPU/CPU 끝난 뒤 첫 손

서버 한도: GPU 1개, CPU 잡 2개, RAM 8GB. 한도를 넘겨 죽이지 마라.

1. `refit150` 숫자 확인. 새 k 제출 금지.
2. 20ep 프로브(`p20med/z`, `wm2`, `px256`) 단독 val. 같은 족이면 블렌드만 `blend_group.py`.
3. **GAP 제거 프로브 하나.** 128px med, `AdaptiveAvgPool2d(6)` + 좌표채널, 20ep, 상하만. `preds/val_*.npy` + `test_*.npy` + `experiments/log.md` 한 줄. 채택은 6h 2-fold×20, Δ≥0.3, 16/20.
4. 그 전엔 cnn3h 추가(§4.4)가 학습 0 인 다음 카드.

닫힌 축 재탕 금지. 실험 하나 = val+test npy + 로그 한 줄. 전 `CLOSED_AXES.md` grep.
