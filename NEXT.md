# 다음 세션 시작점 — 모델이 누구든 이 파일부터

**작성:** 2026-08-18 12:20 UTC, Opus 세션 종료. Grok 이 이어감.
레버 2·3·4 전부 기각. 공지 2·3 반영 완료. **제출 실패 원인은 model.pth 용량이다 — §1.1 부터 해라.**
`GROK_REVIEW.md` §3 문서 스텝 A–F 는 끝났다. 다시 하지 마라.

**읽기 예산:** 이 파일 + `RULES.md` §1·§8 + `DOMAIN.md` §9. 그 외는 필요할 때 지정 줄만.
DOMAIN/PLAN/ONBOARDING/PLAYBOOK/POSTMORTEM 전문 금지.

---

## 0. 지금 성적

| | validation RMSE | Public LB |
|---|---|---|
| persistence | 87.59 | - |
| 대회 베이스라인 3D-CNN+LSTM | 78.12 | - |
| 풍속 20개만 | 74.99 | - |
| **어제 제출본** (스칼라 3종) | 64.76 | **59.6608** |
| **제출 폴더에 대기 중** (ESWF+CNN 6종) | **63.751** | 아직 안 누름 |

공개 LB 상위: 57.61 / 59.34 / 59.55 / **우리 59.66** / 60.20 / 62.34 / 72.98 …
**제출 상태 (2026-08-18 11:49 UTC 확인):** 사람이 오늘 버튼을 **2회** 눌렀으나 점수가 안 뜬다.
어제치 59.6608 이 **2줄** 올라가 있는데 이것도 랙으로 인한 중복 제출로 보인다 — 하루 5회에서 이미 깎였을 수 있다.
누를 당시 서버는 유휴였다 (load 0.23, GPU 6 MiB, 학습 프로세스 0). **「작업 중이라 실패」가 아니다.**
파일 3종 무결: `submission.csv` 3,869줄(헤더+3,868), md5 `1674f097…`, `code.ipynb` `e578d39b…`, `model.pth` `cb183c0d…` (197 MB).
`RULES.md` §2 에 "평가 완료까지 시간이 걸릴 수 있음" 이 공식 문구로 있다. **더 누르지 마라 — 한도만 태운다.** 운영진 문의가 먼저다.

**Public 과 로컬 val 을 직접 비교하지 마라.** 채택은 같은 split 의 Δ 만.

---

## 1. 제출 폴더 (버튼만 누르면 됨)

`/home/jovyan/submission/` 2026-08-18 11:31 UTC 판. bugcheck C1–C5 **통과**.

| 파일 | 크기 | |
|---|---|---|
| `code.ipynb` | 25,497 | `work/pipeline.py` → `mknb`. infer 기본. ESWF+CNN 포함 |
| `model.pth` | 197,254,746 | ridge_v1+gbdt_v1+gbdt_v2+gbdt_eswf+cnn96b+cnn96w3 |
| `submission.csv` | 943,594 | 3,868행, min 275 max 715 |

가중치 (pth 키 = 노트북 키, **맞춰 둠**):
`ridge_v1 0.125 + gbdt_v1 0.25 + gbdt_v2 0.125 + eswf 0.25 + cnn96b 0.125 + cnn96w3 0.125`

한 번 키가 `img_ridge`/`ridge_v1` 로 갈라져 추론이 죽을 뻔했다. `pipeline.py` 에 alias 를 넣었고 pth 키를 고쳤다. **다시 갈라지지 마라.**

백업: `/home/jovyan/work/pack_out/` 동일 3종.

### 1.1 제출이 안 나가는 이유 = **model.pth 197 MB** (다음 세션 1순위)

운영진 QNA 답변: 제출 오류에 **모델 파일 용량을 줄여보라**. 우리 pth 구성(pickle 기준):
`gbdt_eswf 82.2` + `gbdt_v2 47.5` + `gbdt_v1 25.3` + `cnn 7.5` = 162.5 MB → 디스크 197 MB.
어제 성공한 제출은 스칼라 3종이라 훨씬 작았다. 상한 수치는 공지에 없다 — 측정해서 줄인다.

**준비 끝난 것 (로컬에 있음, 서버 업로드만 남음):**

| 파일 | 상태 |
|---|---|
| `work/shrink.py` | **업로드 필요.** 측정 전용 실행 + `--write K` 로 재작성 |
| `work/pipeline.py:402` | `compressed` 플래그면 zlib 해제. **수정 완료** |
| `work/pack.py` | 앞으로 만드는 pth 도 zlib 저장. **수정 완료** |

손잡이 둘뿐이다. ① sklearn 파트 **zlib** (정확도 손실 0) ② HistGBDT `_predictors` 앞 k 개 절단
(부스팅 라운드 축소, 손실은 validation 으로 잰다 — 모델 선택 용도는 규정상 허용, RULES §5-1).

```bash
python tools/remote.py put work/shrink.py work/shrink.py
python tools/remote.py run -c "import subprocess; subprocess.Popen('cd /home/jovyan/work && nohup python shrink.py > shrink.log 2>&1 &', shell=True)"
# 표 보고 손실 0.05 이내 최소 k 선택 → --write K → mknb 재생성 → tools/submit.py → 사람이 버튼
```

`--write` 는 원본을 `submission/model.pth.bak` 로 남긴다. **pth 포맷은 pipeline/pack 양쪽이 같이 알아야 한다 — 한쪽만 고치지 마라.**

---

## 2. 서버에 있는 예측

GPU 비어 있음. 학습 프로세스 없음.

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

## 4. 다음 레버 (이 순서로)

1. **사람이 제출 버튼을 누른다.** 학습 중 금지 — 지금 학습 없음. 누른 뒤 Public 을 §0 에 적어라.
2. ~~`hssq65` 결합~~ **판정 끝(E014b).** 63.751 → 63.657 (−0.09). ρ(v2_gbdt)=0.965 로 v2 자리를 갈아끼울 뿐이고,
   로컬 후보 전부 넣은 상한도 63.655 다. **재가중 축은 고갈됐다.** 파이프라인·pth 키를 건드릴 값이 아니다.
3. ~~`rotter.py` / `chtrack.py`~~ **둘 다 기각(E013 96.197 / E012 71.979).** 실행했고 hill_climb 이 안 뽑았다.
4. ~~193+211 k-means CH 분할~~ **기각(E015).** 71.432 · ρ(img_blend) 0.875. 군집 자체는 물리적으로 맞았다
   (CH 0.20/0.42 · 조용 0.99/0.99 · AR 1.77/1.50, CH 면적 20%). **분할 방식이 아니라 디스크 요약통계 족이 고갈이다.**
5. **E016 = `work/align.py` (업로드만 하면 됨).** E007 재개 조건 그대로 — 창을 좁히지 말고 j*·τ 를 **추가 피처로만**.
   v2 839피처 유지 + horizon 마다 τ=1AU/v 로 계산한 발사 슬롯 j*(h) 스칼라를 선형보간해 덧붙인다
   (속도 가설 둘: 마지막 풍속 / 최근 8슬롯 평균). **채택선: v2_gbdt 67.022 를 이길 것** — 같은 피처의 상위집합이라
   ρ 는 볼 필요 없고, 이기면 v2 자리를 대체한다. 제출 팩이 끝난 **뒤에** 던져라 (학습 중 제출 금지).

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
