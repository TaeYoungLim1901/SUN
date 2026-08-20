# `work/` 코드 지도 — 88개 .py 가 무엇인가

제출 카드를 굽는 경로만 알고 싶으면 [README.md](README.md) 를 봐라. 여기는 **전부**의 목록이다.
살아남은 것과 죽은 것을 같이 둔다 — 죽은 축을 다시 여는 게 이 팀의 가장 흔한 낭비였다
(`CLOSED_AXES.md` 를 먼저 grep 해라).

경로는 전부 서버 `/home/jovyan/work/` 기준이고, 데이터는 `/home/jovyan/public_dataset/` 에 있다.

## 1. 제출 카드에 실제로 들어간 성분 ★

| 파일 | 무엇 |
|---|---|
| `imgfeat.py` `imgfeat2.py` `imgfeat3.py` `feats.py` | 영상 → 스칼라 피처. 코로나홀 = EUV 에서 어두운 영역. v2 는 격자 4×4 + 서쪽 경도대 |
| `omni.py` | omni 1293차원 GBDT. 표 모델 중 가장 세다 |
| `eswf.py` | 경험적 태양풍 예보식 기반 피처 |
| **`cnn6.py`** ★ | **광류 2-스트림 CNN.** Farneback 광류장(fx,fy)을 채널로. 32 시드 = 카드 가중의 40~60% |
| `cnn7.py` | 위 + 스칼라 839차원 블록. 잔차 ρ 0.877 로 갈라진 유일한 족 |
| `cnn9.py` | 위 + Stonyhurst 좌표 리맵. 3 시드만 |
| `cnn.py` `cnn3.py` `cnn4.py` | 영상 단독 CNN 1~3세대. cnn3b/c/d 는 MAX 카드까지 실렸고 N50 에서 빠졌다 |
| `pack_seedbag.py` `mkcard7.py` | 성분 → `model.pth` 로 굽는다 (MAX/N50 은 앞, CARD1/2/3 은 뒤) |
| `pipeline7.py` | 제출 노트북의 원본. 자기완결이어야 한다 (`work/*.py` import 금지) |
| `bugcheck.py` | 제출 전 C1~C5 (스키마·순서·NaN·물리범위·validation 오염) |

## 2. 판정 도구 — 채택/기각은 여기를 통과해야 한다

| 파일 | 무엇 |
|---|---|
| `noise_floor.py` | 6시간 창이 겹치는 행을 **연결요소 11개**로 묶는다. 아래 전부의 토대 |
| `paired.py` | 연결요소 짝지음 + 절반분할 σ + 부트스트랩 CI. 숫자 하나로 닫지 않기 위한 도구 |
| `bulkjudge.py` | 위를 bulk 구간(`\|target−wind_19\|`<p60)에서. **카드2 를 고른 도구** |
| `add_honest.py` | 가중을 반쪽에서 고르고 반대쪽에서 채점 (2-fold×20). 낙관 편향 제거 |
| `blend_group.py` `groups.py` | 정직한 그룹 분할 (공유 PNG 로 상대시각을 잇는다) |
| `contrib.py` | 후보 성분의 추가 기여. **낙관치다 — 이것만으로 채택·기각 금지** |
| `blend_search.py` `wfit.py` `grid3.py` | 결합 가중 탐색 (npy 만, 학습 없음, 초 단위) |
| `maxbag.py` `n50eval.py` | 시드백 크기·새 시드의 추가 정보량 판정 |
| `noise_check.py` | 노이즈 바닥 교차검증. 부트스트랩 하나로 결론 안 낸다 |
| `ramp.py` `cand_ramp.py` `fastfam.py` `comp_fast.py` `slow.py` `judge_fast.py` | 구간별(램프/고속/저속) Δ. **구간을 나눠 보는 습관이 카드2 를 만들었다** |
| `mix.py` `sweep.py` `anatomy.py` `score.py` | 짝 부트스트랩 · 족 한눈에 · 오차 해부 · 채점기 |
| `recal.py` `tweight.py` `fit2.py` | 로컬 Δ → Public 전이비 적합 |

## 3. 시도했고 안 된 것 (재개 조건은 `CLOSED_AXES.md`)

| 파일 | 결과 |
|---|---|
| `flow.py` `floww.py` `flow_omni.py` `flow_fit.py` `flow_h.py` `flow_runh.py` `flow_score.py` `flow_wait.py` | 광류를 **스칼라로 요약해** GBDT 에 넣은 판. 단독 84.6 으로 죽었다. 같은 정보를 채널로 준 `cnn6.py` 가 살았다 — **표현이 정보를 죽이고 있었다** |
| `phys.py` `omni2.py` | 도메인 물리 스칼라. ρ 0.875 로 확실히 다른 족인데 어느 가중에서도 손해 |
| `cnn7p.py` | 피크 보조손실. λ 를 두 자릿수 바꿔도 무이득 |
| `cnn8.py` | 3-스트림(6채널). 이득 0.005 |
| `pretrain.py` | 단일 이미지 사전학습 (외부 가중치 아님 — 규정상 pretrained 금지라 자체 학습) |
| `diag_calib.py` `diag_calib_refute.py` `patch_calib.py` `calib_chain.py` `recal.py` `test_calib_apply.py` | 사후 보정 전반. 행 단위 분할에서 보이던 +0.39 는 **누수**였다 |
| `shrink.py` `shrink_write.py` `pack_100.py` `pack_jump.py` `pack_big.py` `pack.py` `pack_omni.py` | 용량 줄이기 세대. 700MB 상한이 확인되면서 전부 순손해로 판명 (트리 절단 = val +0.140) |
| `hss.py` `rotter.py` `chtrack.py` `kmeans_ch.py` `polar.py` `tof.py` `chfeat.py` `t4ch.py` | 물리 기반 축 — 고속 지속 보정 · Rotter/Vršnak 선형 · 코로나홀 궤적/군집 · 경도 워터폴 · 탄도 ToF |
| `xgb.py` `jump.py` `align.py` `shift.py` `arrive.py` `windonly.py` | 표 모델·정렬 축. `windonly.py` 는 영상 없이 wind 20개만 쓴 기준선(74.99) |

## 4. 운영

| 파일 | 무엇 |
|---|---|
| `stage.py` | `submission/` 으로 복사 + bugcheck. **제출 폴더는 이 시점에만 바뀐다** |
| `mkstats.py` | 시드가 쓴 train 통계(스칼라 표준화·광류 std)를 복원해 팩에 싣는다. 추론 때 다시 계산하면 값이 어긋난다 (과거 fstd 1.368배 버그) |
| `oof_img.py` `oof_eswf.py` | OOF 예측 생성 |
| `pipeline.py` `pipeline_eval.py` | 이전 세대 제출 노트북과 그 평가판 |
| `judge1.py` `judge_fast.py` | 초기 판정 스크립트 |
