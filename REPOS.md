# 깃허브 레포 조사 — 내용 기준

이름이 아니라 **파일 내용**으로 찾았다 (`gh search code`). 각 항목은 트리·소스 확인분.
「무엇을 가져올 것인가」만 적는다. 물리 배경은 [DOMAIN.md](DOMAIN.md).

---

## 0. 대회 자체 — 확인된 운영 사실

| 레포 | 내용 |
|---|---|
| [LeeChangmin0310/2024_KASI_KAIST_Space_AI_Competition](https://github.com/LeeChangmin0310/2024_KASI_KAIST_Space_AI_Competition) | **제1회 종합 1위(태양상)** 우쭈쭈팀. 코드 노트북 한글·영문 2본 |
| [maeng00/2024_KASI_KAIST_Space_AI_Competition](https://github.com/maeng00/2024_KASI_KAIST_Space_AI_Competition) | 위 팀원 미러 |
| [tmdekd/Astrospace-AI-Competition](https://github.com/tmdekd/Astrospace-AI-Competition) | 다른 참가자. `code.ipynb` / `code_final.ipynb` / `submission_final.csv` / `0.86.csv` — **제출 3파일 구조 실물** |

**README 에서 확인된 사실 (추정 아님):**

- **제1회 = 2024.07~08**, 주제 «AI기반 지구 영향 태양 이벤트 자동화 탐지»
  (코로나홀 · 흑점 · 홍염 **탐지**). ⚠ 딥리서치는 이걸 「2회차 CME 탐지」로 적었다 — **회차·주제가 어긋난다.**
- 플랫폼: **`https://kaist-kasiai.elice.io/explore`**
- 평가: **F1 @ IoU 0.5**, **Public : Private = 50 : 50**
- **🔴 사전학습 모델 사용 불가** — 1회차 규정에 명시돼 있었다.
  ⇒ 본선 규정도 동일하게 **가중치 사용 금지**다. Surya·GoogLeNet **사전학습 가중치 전략은 제외**하고,
  아키텍처만 차용해 랜덤 초기화로 학습한다 ([RULES.md](RULES.md) §1 5항).
- 1위 팀 최종: Public **0.884474** / Private **0.884492** / Total 0.884483
  ⇒ **Public–Private 차이가 2e-5.** 분할이 무작위이고 평가 노이즈가 매우 작았다는 실증.
  같은 구조라면 이번에도 **널 조작 σ 가 작을 것**이고, 그러면 작은 개선도 순위에 반영된다 **[추론]**
- 예선: 천문·AI 퀴즈 + 데이터 라벨링 점수
- 1위 팀 기법: YOLOv5n~v10s 전수 비교 → **의외로 YOLOv5s 가 최고**(데이터가 작고 단순해서),
  상하·좌우 flip 증강, image 1024 / batch 64 / epoch 70

> **제3회(풍속 회귀) 공개 솔루션은 2026-08-18 기준 0개다.** 1·2회는 탐지 과제라 코드를 베낄 수 없다.
> 공개 레포는 위 3개가 전부다. 공개 커널 벤치마크가 없다.
> 성분 분해(PLAYBOOK §5.5)의 대상은 **팀원 제출과 리더보드뿐**이다.

---

## 1. Tier 1 — 직접 이식 대상

### [eunsu-park/geoindex-model](https://github.com/eunsu-park/geoindex-model) 🔥 **2026-08-07 커밋 (현재 진행형)**
한국 우주환경 ML 연구자(박은수)의 **다중 horizon 시계열 예측 프레임워크**. 우리 과제와 구조가 동형이다.

- **입력 = 태양풍 시계열, 출력 = 다중 스텝 지수** — `configs/io/in6h_out6h.yaml` ~ `in3d_out24h.yaml` **24종**
  → 우리의 「과거 120시간 → 미래 72시간」이 이 축에 그대로 들어간다
- `src/pipeline/datasets_hdf5.py` — **SDO + OMNI 다중 모달** 로더 (README 는 "future" 라 표기)
- `configs/cv/fold1~5.yaml` + `scripts/make_cv_indices.py` + `scripts/aggregate_cv_results.py`
  → **폴드 인덱스를 파일로 굳히는 방식.** PLAYBOOK §2.1 이 요구하는 「CV 를 먼저 못 박는다」의 구현체
- **`src/losses.py` ← 이 레포에서 가장 값나가는 파일.** DOMAIN §4 가 필요하다고 한 것들이 이미 다 있다:
  - `QuantileLoss` — 분위수 손실
  - `GradientBasedWeightLoss` — **급변 구간 강조**. SIR 상승 엣지가 여기 걸린다
  - `MAEOutlierFocusedLoss` — 이상치 집중. HSS 피크
  - `WeightedMSELoss` — **horizon 별 시간 가중**. 대회 지표는 12 horizon 균등 RMSE 라 여기 손댈 축이 열린다
  - `AdaptiveWeightLoss`, `MultiTaskLoss`, `MultiModalContrastiveLoss`
  - `SolarWindWeightedLoss` (NOAA G-Scale 기반 폭풍 가중)
- `src/networks/` — linear · lstm · transformer · tcn · patchtst · timesnet · convlstm · gnn · **fusion**
  → PLAYBOOK §3 「다양한 기저」를 하루에 세울 수 있는 재료
- `analysis/evaluate_storm_performance.py` — **이벤트 구간 별도 평가.** 우리 세그먼트 분해기의 참고형
- `src/uncertainty.py` (MCD), `analysis/recalibrate_mcd.py` — 보정

> **가져올 것:** `losses.py` 통째로, `configs/io` 의 in/out 윈도 설계, CV 인덱스 파일화 패턴,
> `evaluate_storm_performance.py` 의 이벤트 분해.
> 같은 저자의 [geoindex-data](https://github.com/eunsu-park/geoindex-data) (OMNI 수집),
> [geoindex-realtime](https://github.com/eunsu-park/geoindex-realtime), [hp-prediction](https://github.com/eunsu-park/hp-prediction),
> [publication-code](https://github.com/eunsu-park/publication-code) 도 같은 계열.

### [NASA-IMPACT/Surya](https://github.com/NASA-IMPACT/Surya) — `downstream_examples/solar_wind_forcasting/`
경로 오타(`forcasting`)까지 확인. 안에 `README.md`, `download_data.sh`, `config.yaml`,
**`solar_wind_tutorial.ipynb`** 가 있다. 태양풍 다운스트림 파인튜닝 완제품.

- 데이터셋: [Surya-bench-solarwind](https://huggingface.co/datasets/nasa-ibm-ai4science/Surya-bench-solarwind) (DOMAIN §2.3 에 규격 정리)
- 파생: [NASA-IMPACT/SuryaBench](https://github.com/NASA-IMPACT/SuryaBench) (`solarwind/`),
  [SwRI-IDEA-Lab/surya_workshop](https://github.com/SwRI-IDEA-Lab/surya_workshop) (워크숍 실습),
  [HungrySemiconductor/SYN-SolarImage](https://github.com/HungrySemiconductor/SYN-SolarImage) (동일 트리 파생)
- ⚠ **사전학습 가중치 금지면 전부 무효.** 1회차 규정이 금지였다(§0). 규정 확인이 최우선

### [Vishal-Upendran/WindNet](https://github.com/Vishal-Upendran/WindNet)
GoogLeNet + LSTM, AIA 전면 원반. 성능은 이제 낡았지만(cc 0.55) **구조가 Son et al. 2023 의 원형**이고
사전학습 없이 자작 가능한 형태다. 규정이 pretrained 금지일 때의 **기본 설계도.**

---

## 2. Tier 2 — 피처·전처리 부품

| 레포 | 쓸 곳 |
|---|---|
| [TCDSolar/CHIMERApy](https://github.com/TCDSolar/CHIMERApy) | **CHIMERA 파이썬 구현.** 171/193/211 삼중 비율 코로나 홀 분할. `chimerapy/chimera.py` + 테스트 포함, 2026-07 갱신 |
| [shibaji7/pyCHIPS](https://github.com/shibaji7/pyCHIPS) | OpenCV 기반 코로나 홀 자동 탐지. `chips/chips.py`, `fetch.py`. 임계 기반이라 가볍다 |
| [samaloney/CHIMERA](https://github.com/samaloney/CHIMERA), [Vexvare/Solar_Analysis_Toolkit](https://github.com/Vexvare/Solar_Analysis_Toolkit) | 같은 계열 대안 구현 (`CoronalHoleBoundaries.py`) |
| [University-of-Reading-Space-Science/HUXt](https://github.com/University-of-Reading-Space-Science/HUXt) | **경량 태양풍 전파 모델**(61★). WSA 입력 fits 예제 동봉. 물리 기반 기저 1장을 싸게 만드는 길. 순수 파이썬이라 A100 없이 돈다 |
| [helioforecast/Predstorm](https://github.com/helioforecast/Predstorm) | L1/L5 실시간 태양풍 예보(17★). `predstorm/data.py` = OMNI/ACE/DSCOVR 로더, `predict.py` = Dst 경험식(Burton/OBrien/Temerin-Li). **in-situ 파생 피처 카탈로그로 읽어라** |
| [cmoestl/heliocats](https://github.com/cmoestl/heliocats) | **ICME 카탈로그 제작·분석**(15★). `icme_rate.py`, HELCATS ARRCAT 동봉. ICME 구간 분리(DOMAIN §1.5)용 |
| [University-of-Reading-Space-Science/SURF](https://github.com/University-of-Reading-Space-Science/SURF) | `surf/surf_insitu.py` — in-situ 태양풍 처리 |
| [aiapy](https://aiapy.readthedocs.io/en/stable/preparing_data.html) | level 1 → 1.5. 열화 보정 · 노출 정규화 · **annulus limb brightening 보정** (DOMAIN §2.1) |

---

## 3. Tier 3 — 참고·데이터

| 레포 | 내용 |
|---|---|
| [FrontierDevelopmentLab/2025-HL-Solar-Wind](https://github.com/FrontierDevelopmentLab/2025-HL-Solar-Wind) | FDL 2025 태양풍 프로젝트. `solar_phenomena_prediction/configs/finetune_solarwind_config*.yaml` + AIA/HMI **MAE 사전학습** 설정 + iSAX 클러스터링. **다중 모달 파인튜닝 설정값 참고** |
| [Joaggi/sdofmv2](https://github.com/Joaggi/sdofmv2) | SDO 다중기기 파운데이션 모델 v2. `src/sdofmv2/tasks/solar_wind/` 존재 |
| [HelioML/HelioML](https://github.com/HelioML/HelioML) | 태양물리 ML 교재(80★). 실행 가능한 노트북 10여 장 |
| [bairaelyn/ambsowi-ml](https://github.com/bairaelyn/ambsowi-ml) | Ambient Solar Wind ML. 노트북 1개 + **적합된 모델 pickle**. 오래됐지만(2021) 피처 설계가 직설적 |
| [Gupta-Anit/SolarWind_Prediction](https://github.com/Gupta-Anit/SolarWind_Prediction) | PSI MHD 시뮬 데이터에 **ConvLSTM 2/3층** 다중 노트북 + `DataAugmentation.ipynb`, `correlation_check.ipynb` |
| [FrontierDevelopmentLab/2025-HL-Ionosphere](https://github.com/FrontierDevelopmentLab/2025-HL-Ionosphere) | `scripts/dataset_omniweb.py` — OMNI 로더 단품 |
| [chickens5/SUN](https://github.com/chickens5/SUN) | Kp 예보. 우리 과제는 아니지만 `ggsp/features.py` 에 **27일 재현 지연 피처(`kp_lag_27d`)** 구현이 있다. 캐시 전략(연도별 parquet)도 참고 |

---

## 4. 못 찾은 것 (없다고 보는 게 맞다)

- **제2·3회 대회 참가자 공개 레포** — 검색상 존재하지 않는다
- **arXiv 2410.05068**(다항회귀가 DL 을 이긴 논문)의 **공개 코드** — 못 찾았다.
  DOMAIN §3.4 의 피처 정의(4×3 격자 CH 면적 @ t−4~7d, V @ t−26~28d)를 보고 직접 구현해야 한다.
  분량은 작다 — 그게 그 논문의 요지다
- **ESWF 공식 구현** — 공개 코드 없음. 관계식만 논문에 있다
- **Son et al. 2023 (대회 원형 논문) 코드** — 공개 안 됨. WindNet 이 가장 가까운 대체

---

## 5. 이 조사가 계획을 바꾸는 지점

1. **현재 규정상 pretrained weight 사용 금지** — Surya/LoRA 전략은 **제외**. 단, 아키텍처 차용 + 랜덤 초기화는 허용. 계획 A 는
   WindNet/Son 구조를 **처음부터 학습**하는 것 + 격자 CH 면적 피처 + GBDT/선형으로 잡아라
2. **`eunsu-park/geoindex-model` 의 `losses.py`** 를 첫날 확보해라. HSS 대응 손실이 이미 다 구현돼 있다
3. **Public–Private 차이 2e-5 (1회차)** — 노이즈가 작을 가능성. 그래도 `kit.calibrate` 로 **실측**해라.
   지표가 F1 → RMSE 로 바뀌었으므로 그 값을 그대로 옮기지 마라
4. **공개 커널 벤치마크가 없다** — 성분 분해 대상은 팀원 제출뿐. 리더보드 격차 해석에 더 신중해야 한다
5. **CHIMERApy / pyCHIPS 는 제출 불가** — 171Å 미제공. CH 분할은 **193/211 PNG 임계·k-means** 만 쓴다
6. **HUXt 기저는 제출 불가** — 외부 WSA/FITS 입력이 필요하다 (DOMAIN §9.1)
