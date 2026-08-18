# 다음 세션 시작점 — 모델이 누구든 이 파일부터

**작성:** 2026-08-18 밤, Grok. **최고 Public 57.2951** (5→6위). jump 57.2985 동률.
**다음 버튼 후보:** omni 팩 (t−4d+j*+잔차 동시, k=300, 76.5 MB, infer **63.702**). `GROK_REVIEW.md` A–F 끝.

**읽기 예산:** 이 파일 + `RULES.md` §1·§8 + `DOMAIN.md` §9. 그 외는 필요할 때 지정 줄만.
DOMAIN/PLAN/ONBOARDING/PLAYBOOK/POSTMORTEM 전문 금지.

---

## 0. 지금 성적

| | validation RMSE | Public LB |
|---|---|---|
| persistence | 87.59 | - |
| 대회 베이스라인 3D-CNN+LSTM | 78.12 | - |
| 풍속 20개만 | 74.99 | - |
| **어제 제출본** (스칼라 3종) | 64.76 | 59.6608 |
| 풀트리 6종 (197 MB) | 63.751 | 용량 거부 |
| **최고** (zlib+k=300 6종) | **63.784** | **57.2951** |
| pack100 (풀트리 zlib + align→v2) | 63.730 | 57.4725 |
| E019 jump→v2 (k=300 zlib) | 63.722 | **57.2985** |

공개 상위: 56.9342 / 56.9949 / 57.1681 / 57.2215 / **우리 57.2951** / 57.5318 / 57.6114 / 57.7067
pack100 은 로컬 −0.054, Public **+0.177**. 풀트리·align 교체는 일반화가 아니었다.
**마지막 제출은 57.2951 본(k=300)을 바닥으로.** 로컬 0.05짜리 재탕 금지.
QnA 원 한도 50 MiB(52,428,800). 주최가 올렸고 70.8 MB 는 통과. 197 MB·149 MB 는 예전에 거부.
70.8 MB 사본: 서버 `work/pack_win_57/`. `submission/` 은 pack100 으로 교체됨 (bugcheck 통과).

**Public 과 로컬 val 을 직접 비교하지 마라.** 채택은 같은 split 의 Δ 만.

---

## 1. 제출 폴더 (버튼만 누르면 됨)

`/home/jovyan/submission/` = **omni 팩** (bugcheck C1–C5 통과). 57.2951 본은 `work/pack_win_57/`.

| 파일 | 크기 | |
|---|---|---|
| `code.ipynb` | 28,191 | t−4d + j* + 잔차 + zlib |
| `model.pth` | 76,516,122 | k=300 zlib, v2 자리 = omni |
| `submission.csv` | 943,541 | 3,868행, min 272.3 max 721.8 |

가중치 그대로. `gbdt_v2` 키 = omni (839 + t−4d + j*, 타깃=잔차). infer val **63.702**.

원본 197 MB 는 `work/model.pth.bak`. 팩 백업 `work/pack_out/` + 로컬 `submit/`.
pth 키 alias (`img_ridge`/`ridge_v1`) 유지. **다시 갈라지지 마라.**
**pth 포맷은 pipeline/pack 양쪽이 같이 알아야 한다 — 한쪽만 고치지 마라.**

### 1.1 용량 — **닫힘**

운영진: 모델 파일 줄여보라. 측정 결과 (결합 RMSE, 손실선 0.05):

| | v1 | v2 | eswf | **blend** | pth |
|---|---|---|---|---|---|
| full | 68.628 | 67.022 | 67.028 | **63.751** | 197 MB |
| zlib only | 같음 | 같음 | 같음 | 63.751 | ~65 MB 예상 |
| **k=300+zlib** | 68.549 | 66.798 | 66.865 | **63.784** (+0.033) | **70.8 MB** |
| k=200 | 68.643 | 66.685 | 66.692 | 63.859 (+0.108) | 기각 |

단독은 라운드를 자를수록 좋아졌고 결합은 나빠졌다. k=300 이 선 안의 최소.
`pipeline.py:402` / `pack.py` 는 zlib. `work/shrink.py` 는 측정, `shrink_write.py` 가 실제 기록.

---

## 2. 서버에 있는 예측

GPU 유휴. pack100 infer 끝 (val 63.730). `align_models.pth` 63.8 MB 저장됨.

| 태그 | val RMSE | 비고 |
|---|---|---|
| `img_blend` | 66.796 | v1 ridge+gbdt 평균 |
| `eswf` | **67.028** | E010. 장기 horizon 이김. 결합 0.25 |
| `v2_gbdt` | 67.022 | |
| `hssq65` | **66.543** | 분위수 0.65, v2 피처. HSS 편향 −98. E014b 로 판정 — ρ(v2_gbdt)=0.965, 결합 이득 0.09. 안 넣는다 |
| `cnn96b` / `cnn96w3` / `cnn96s2` | 68.55 / 68.91 / 68.35 | w3 가 결합에 있음 |
| `cnn96w5` | 69.695 | ρ(img_blend)=0.815. w3 대체 시 63.751→63.695. 미적용 |
| `eswfw3` | 67.930 | 기각 |
| `chtrack` | 71.979 | E012. hill_climb 이 안 뽑음. 기각 |
| `rotter` | 96.197 | E013. persistence 미만. 기각 |
| `imgpt` | 71.783 | 단일프레임 사전학습. 기각(단독) |
| `persist` | 87.588 | |

정직 분할: `preds/train_{comp,tstep,fold}.npy`. 28성분, gap=109, 공유PNG=0.

로컬에 받은 것: `work/local_data/pred_bundle.npz` + 개별 `val|test_*.npy` (우리 산출물만). 원본 csv 사본은 [공지3] 때문에 지웠고 `work/local_data/` 는 `.gitignore` 에 있다. 학습은 서버에서만.

---

## 3. 이 세션에서 닫은 것 (다시 하지 마라)

| 축 | 결과 |
|---|---|
| last_wind 높으면 persist | 악화 (65.4) |
| last_wind 부스트 | −0.12, 도착 국면에선 0 |
| ESWF × y>600 가중 3 | 67.93, 기각 |
| 예측 1–3스텝 당김 | 64.01/65.20/66.97, 기각. peak_step **편향 +0.11** (늦는 게 아님) |
| CNN 시드만 / w=5 단독 | 고갈 또는 −0.06 |

HSS 병목은 **도착**(last wind 중앙값 494, train 713행이 lw<500 & peak>600)이지 지속이 아니다.

---

## 4. 다음 레버 — 마지막 제출 1회 (11:40)

1. ~~k=300~~ **57.2951 (5위).** `work/pack_win_57/`.
2. ~~pack100~~ **57.4725. 기각(Public).** 로컬 이득이 Public 에서 뒤집힘.
3. ~~E017 chfeat~~ **기각.** 67.249, hill 이 안 뽑음. ρ(v2)=0.980.
4. ~~E018 arrive~~ **기각 67.886.** hill 안 뽑음.
5. ~~cnnlate~~ **기각 69.461.** 결합 −0.028 뿐. 안 냄.
6. ~~E019 jump~~ **Public 57.2985.** 동률. 최고 57.2951.
7. **E021 omni 팩 준비됨.** 단독 66.476, 고정가중 63.674, infer k=300 **63.702**, 76.5 MB.
   버튼은 사람이. Public 이 57.2951 을 못 이기면 이 축(스칼라 동시투입)도 접는다.
   cnnmed 68.108 단독 이김, 결합 −0.021 기각.
8. 내일: 픽셀/모션(광류) 또는 Hofmeister CH **폭→지속시간**. 요약통계 동시투입은 여기까지.
9. 깃허브 3회 솔루션 0. 1·2회는 탐지. 닫힌 것: align/jump 재제출, hssq65 재가중, rotter, chtrack 단독, k-means, persist/시프트, 197 MB.

Box-Cox 금지. persist/시프트/가중 ESWF 재개 금지.

새 실험 전 `CLOSED_AXES.md` 사람 grep. 실험 하나 = val+test npy + `experiments/log.md` 한 줄.

---

## 5. 명령

```bash
python tools/remote.py check
python tools/submit.py          # 버튼 전 버그검사
python work/bugcheck.py         # 서버에서 직접
python tools/conform.py
python tools/push.py
```

분위수 로그: `work/hss_q65.log`. 예측: `work/preds/val_hssq65.npy` + `test_hssq65.npy`.

쿠키: 2026-08-18 12:14 UTC 갱신, refresh 토큰 기준 **8/19 11:56 UTC** 까지. 죽으면 `CONNECT.md` 1–4 → `remote.py auth`.
⚠ 복사할 때 **`lab` 문서 요청(302) 말고 Fetch/XHR 의 `api/...` 요청**을 Copy as cURL 해라. `oauth2_proxy` 쿠키가 없으면 401 만 본다.
`remote.py auth` 가 그 붙여넣기를 이제 저장 전에 막는다. `copy as fetch` 는 쿠키를 안 담아서 쓸 수 없다.

---

## 6. 하지 마라

규정: val 을 train 에 합침, pseudo-label, pretrained weight, 외부 데이터.
**[공지3]** 데이터셋 ZIP/TAR·분할·불필요 복사 금지, 외부 환경 학습 금지(재현성 검증에서 불리). 대용량 삭제하면 `~/.local/share/Trash` 까지 비워라. 2026-08-18 확인: 디스크 31%, Trash 20K, 압축파일 0.
**[공지2]** 캐시·출력은 `/home/jovyan` 하위에만. 우리는 `work/resized_cache` + `~/sw_cache` 폴백 — 조치 불필요.
닫힌 축: 27일 lag, 서쪽 림, 창 좁히기, 좌우 플립, test 시계열 복원, Public 단독 최종 선택.
**새 검토서 만들지 마라.** 이 파일만 고치고 `python tools/conform.py`.
