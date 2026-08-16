# `DOMAIN.md` PDF 검증 및 §9 보완

## 판독 기준

검증 대상은 디스크에 있는 네 PDF와 repo의 `RULES.md`, `DOMAIN.md`뿐이다. 아래 `p.`는 PDF 파일 쪽수이며, `출처없음`은 네 PDF에서 확인되지 않았다는 뜻이지 해당 명제가 물리적으로 거짓이라는 뜻은 아니다.

두 번째 논문(`Forecasting High-Speed Solar Wind Streams from Solar Images`)은 이미지로만 된 스캔 PDF였다. `pdftotext`와 PyMuPDF는 본문 대신 Wiley/AGU 문구만 추출했다. 따라서 해당 논문의 표·본문은 렌더한 페이지 이미지를 판독했고, 판독할 수 없는 내용은 추측하지 않았다.

## 1. 검증 결과

### 1.1–1.5 물리·전처리 주장

| `DOMAIN.md` 주장 | PDF 실제 내용 | 판정 |
|---|---|---|
| §1.1 저속풍 300–400 km/s, 고속풍 600–800 km/s | Son은 저속풍을 `v ≤ 400 km/s`, 고속풍을 `v ≥ 650 km/s`로 정의한다. 600–800의 범위·이중 봉우리·250–300 극저속 빈도는 네 PDF에서 확인되지 않는다. (Son p.1, Introduction) | **불일치** |
| §1.2 193 Å = Fe XII 약 1.2 MK, 211 Å = Fe XIV 약 2 MK | Collin은 193 Å Fe XII를 약 **1.6 MK**, 211 Å Fe XIV를 약 **2.0 MK**라고 쓴다. Son은 두 채널을 사용하지만 온도를 제시하지 않는다. (Collin p.4, §2.1.1; Son p.2, §2) | **불일치** |
| §1.2 코로나 홀은 193/211 Å에서 어둡다 | Cranmer는 코로나 홀이 UV·X-ray 방출이 가장 낮은 영역이라고 설명하고, Son은 193/211 Å가 코로나 홀 특성을 나타낸다고 설명한다. (Cranmer pp.4–5, §3; Son p.2, §2) | **일치** |
| §1.3 태양–지구 전파에 약 4일, 고정 4일 지연 | Son은 1 AU 비교에서 3–4일이 걸린다고 하고, Collin은 4일 앞 예측을 목표로 한다. (Son p.4, §4; Collin pp.2, 21, §§1–2, 7) | **일치** |
| §1.3 속도별 2.2–5.8일 표, `Ω=14.713°/일`, `v=400 → Δφ≈64°` | 네 PDF에서 이 계산표와 각도 수치를 찾지 못했다. | **출처없음** |
| §1.4 SIR의 순서가 압축 → 계면 → 고속 고원 → 희박/완만 하강 | Richardson의 1 AU 도식은 압축된 느린 바람 `S′`, 압축·감속된 빠른 바람 `F′`, 비 perturb된 빠른 바람 `F`, 그리고 하강부의 rarefaction `R`을 구분하며, 계면은 `S′–F′` 경계다. ‘고원’은 논문의 용어는 아니지만 비 perturb된 빠른 바람 구간에 대응한다. (Richardson pp.15–16, §3, Fig.14) | **일치** |
| §1.4 속도 급상승 400 → 700+ km/s, 상승이 하강보다 훨씬 빠름 | Richardson은 양의 속도 gradient와 뒤쪽 rarefaction을 정성적으로 지지하지만 이 숫자와 ‘훨씬’이라는 정량 비교는 제시하지 않는다. (Richardson pp.15–17, §3) | **출처없음** |
| §1.4 전·후방 충격파는 보통 약 2 AU 이후 형성되고 1 AU에는 충격 없는 압축이 대부분 | Richardson은 충격파가 2 AU 이후 형성되는 경향과 3–5 AU에서 높은 검출률을 보고한다. 동시에 1 AU에서도 전방 17%, 후방 6%, 쌍 1.4%가 보고되어 예외가 분명하다. (Richardson pp.20, 28–29, §§3, 5) | **일치** *(단, “보통”은 “대부분이나 1 AU에도 존재”로 읽어야 함)* |
| §1.5 Son의 한계: CME 기인 증가를 못 맞추고 코로나 홀 HSS의 약 70%만 포착 | Son은 시험 중 Jian 기준 HSS **36건 중 25건(약 70%)**의 속도 증가를 포착했다고 한다. 별도로 속도 증가를 낳은 ICME **12건**을 언급하며 이를 모두 포착하지 못했다고 쓴다. 70%를 ‘코로나 홀 기인 HSS만의 분모’라고 정의하지는 않는다. (Son p.5, §4; p.6, §5) | **불일치** |
| §1.5 CME 속도 800+ km/s, EUV 원반에서 며칠 전 예고되지 않음, ICME는 환원 불가능한 오차 바닥 | Son이 명시한 것은 입력에 CME 정보가 없어 CME 증가를 예측하지 못한다는 점과 한 사례의 halo CME 속도 **1179 km/s**다. ‘800+’를 일반값으로 쓰거나 ‘환원 불가능한 오차 바닥’이라고 정식화하지 않는다. (Son pp.5–7, §§4–5) | **불일치** |

### 2.2 코로나 홀 35% 임계

| `DOMAIN.md` 주장 | PDF 실제 내용 | 판정 |
|---|---|---|
| 193 Å 원반 **중앙값**의 35% 미만을 CH로 분할; Vršnak+2007/Krista & Gallagher 2009의 표준 | 네 PDF 어느 곳에도 35% 수치, 중앙값/평균 구분, Vršnak·Krista의 해당 결과가 없다. Cranmer는 UV/X-ray 최저 방출이라는 정성 정의와 필라멘트 제외 필요성만 말한다. (Cranmer pp.4–5, §3) | **출처없음** |
| 이 네 논문이 35%가 median이라고 뒷받침한다 | Son은 임계 분할을 하지 않고 보정한 193/211 원 영상을 사용한다. Collin은 193/211 두 채널을 결합한 **k-means, k=3**으로 CH·활성영역·조용한 태양을 나눈다. (Son p.2, §2; Collin p.4, §2.1.1) | **출처없음** |

따라서 이 자료만으로는 “median인가 mean인가”, “누구의 결과인가”를 확정할 수 없다. `DOMAIN.md`의 Vršnak+2007 / Krista & Gallagher 2009 귀속은 네 PDF 검증 범위를 넘어선다.

### 3.1 동일 조건 비교표

`DOMAIN.md`의 2012–2020 표 전체는 네 PDF에서 확인되지 않는다.

| `DOMAIN.md` 행 | 네 PDF에서 확인되는 가장 가까운 실제 표 | 판정 |
|---|---|---|
| DL: MAE 51.4, RMSE 68.6, 상관 .69, HSS .82, CSI .35 | Son의 24 h 표는 RMSE **57.6**, CC **.78**뿐이며 MAE/HSS/CSI가 없다. Collin의 4×3 표는 timeline RMSE **68.1**, CC **.70**이다. (Son p.3 Table 1; Collin p.13 Table 2) | **출처없음** |
| 27일 지속성: 65.2 / 87.0 / .57 | Collin의 27-day baseline은 timeline RMSE **89.1**, CC **.56**이고 HSS peak RMSE는 **78.4**다. 같은 표의 수치가 아니다. (Collin p.17 Table 3) | **출처없음** |
| WSA–ENLIL: 86.9 / 116.3 / .41 | Collin의 WSA 행은 timeline RMSE **86.5**, CC **.38**이다. Son은 한 CR 2209 사례와 별도의 1995–2001 연구 RMSE **97.2**를 언급할 뿐이다. (Collin p.17 Table 3; Son p.5, §4) | **출처없음** |
| ESWF 3.2: 100.4 / 130.9 / .08 | Collin의 ESWF 행은 timeline RMSE **91.8**, CC **.43**이다. (Collin p.17 Table 3) | **출처없음** |
| 표의 기간은 2012–2020 | Son의 전체 자료는 2010년 5월–2020년 12월이고, Collin의 주 개발 자료는 2010년 6월–2019년 12월이다. 2012–2020 표라는 locator는 네 PDF에 없다. (Son p.2; Collin p.3) | **출처없음** |

결론: 이 표는 네 PDF 중 어느 논문의 표도 아니며, 이 자료만으로 원 논문을 식별할 수 없다. `DOMAIN.md`의 “가장 신뢰할 표”라는 표현은 제거해야 한다.

### 3.2 Son et al. 2023

| `DOMAIN.md` 주장 | PDF 실제 내용 | 판정 |
|---|---|---|
| 193+211 Å, 64×64, 과거 5일, 풍속 6시간 간격 | Son은 193/211 Å 영상을 원본 4k×4k에서 64×64 PNG로 줄이고, 풍속은 과거 5일 동안 6시간 간격으로 사용한다고 한다. (Son p.2, §2) | **일치** |
| 영상 cadence는 12시간 | Son은 풍속은 6시간, 영상은 **12시간** cadence라고 명시한다. (Son p.2, §2) | **일치** |
| 대회와 논문이 “사실상 동일”; 대회 영상도 6시간 격자 | `RULES.md`는 대회 영상과 풍속 모두 6시간 간격이라고 한다. 따라서 입력 풍속·출력 cadence는 유사하지만 영상 cadence는 다르다. (Son p.2; `RULES.md` §3) | **불일치** |
| 출력 6시간 간격 12스텝, 72시간 | Son은 3일 앞 풍속을 6시간 cadence로 순차 예측한다. (Son pp.1–2, Abstract/§2) | **일치** |
| 3D conv 1층 → Inception 3개 → LSTM 1층; 풍속 dense 2층; concat 후 dense 1층 | 논문의 구조 설명과 일치한다. 활성화는 conv/LSTM ReLU, dense SELU, loss MSE, Adam learning rate `10^-4`다. (Son pp.2–3, §3) | **일치** |
| 6 h: CC .92/RMSE 37.4; 24 h: .78/57.6; 72 h: .67/68.2 | 24 h 값은 Table 1, 양 끝값과 전 범위는 Figure 2·Abstract에 있다. (Son pp.1, 3, Abstract/Fig.2/Table 1) | **일치** |
| 기간·분할: 2010-05–2020-12; train 9,992 / val 1,321 / test 3,890; 1–8월/9월/10–12월 | Son이 그대로 명시한다. 결측·품질 불량을 제외한 총 pair 수는 **15,203**이다. (Son p.2, §2) | **일치** |

핵심 정정은 영상 cadence다. Son 설정은 “영상 12 h + 풍속 6 h”이고, 본선 규정은 “영상 6 h + 풍속 6 h”다. 두 설정을 동일하다고 쓰면 안 된다.

### 3.3 기타 성능 행

| `DOMAIN.md` 행 | 실제 locator | 판정 |
|---|---|---|
| Upendran+2020 76.4/.55, Raju & Das 76.3/.57, Sun+2021 62.8/.79; 24 h 조건 | Son Table 1에 각각 24 h (Upendran/Raju는 1-day average, Sun은 1 h average)와 위 수치가 있다. (Son p.3 Table 1) | **일치** |
| 다항회귀 3차+Lasso: 4일, SC24, 68.1/.70 | Collin의 4×3 polynomial model이 timeline RMSE **68.1**, CC **.70**이고, 개발 자료는 2010–2019 solar cycle 24다. (Collin pp.3, 13, §§2.1, Table 2) | **일치** |
| 같은 모델 SC25 상승기 2020–2023: RMSE 80.3 | 80.3은 2020을 분포변환 재적합에 사용한 뒤 **2021–2023**을 평가한 운용 설정의 timeline RMSE다. 단순한 “2020–2023 평가”로 쓰면 기간 설명이 틀린다. (Collin pp.19–21, §5–7) | **불일치** |
| Multi-modal encoder–decoder Day1/Day4 46.9/53.3, .78/.55 | 해당 논문은 네 PDF 묶음에 없다. | **출처없음** |

### 3.4 Collin 다항회귀 논문

| `DOMAIN.md` 주장 | PDF 실제 내용 | 판정 |
|---|---|---|
| 3차 다항회귀+Lasso, 계수 약 165개 | 논문은 165 coefficients 모델을 명시한다. 8개 주요 feature로 성능–복잡도 절충을 얻었다. (Collin p.18, §4) | **일치** |
| CH area lag `t−4,5,6,7 d`, V·density·pressure·temperature·B lag `t−26,27,28 d` | Table 1이 CH area/asymmetry를 4–7일, in-situ 변수를 26–28일로 정의한다. (Collin p.4, Table 1; p.5, §2.1.1–2.1.2) | **일치** |
| 4×3은 timeline, 10×10은 HSS peak용 | 4×3은 timeline RMSE 최적의 저복잡도 선택이고, 10×10은 HSS peak RMSE 최적 선택이다. (Collin pp.9, 22, §3.1, Appendix A) | **일치** |
| timeline RMSE 68.1, CC .70, SC24 | 4×3 polynomial model의 값이다. (Collin p.13 Table 2; p.21 §7) | **일치** |
| CV에 CME 제거와 180일 buffer | Richardson–Cane ICME list로 CME 관련 구간을 제거하고, 27일 반복 때문에 fold 경계에 **180일**을 버린다. (Collin pp.6, 8, §§2.1.4, 2.3.2) | **일치** |
| Swin Transformer는 29M parameter | 비교 문단이 Swin을 29 million parameters로 보고한다. (Collin p.18, §4) | **일치** |
| “gradient boosting, Swin, WSA/ESWF, EUHFORIA를 이겼다” | EUHFORIA는 비교 표에서 ours가 모든 주요 지표에서 우세하다. WSA는 논문 서술상 모든 범주에서 우세하다. 반면 ESWF는 HSS peak가 더 정확하고 ours는 timeline·일부 event 지표가 낫다. Gradient boosting과 Swin은 HSS peak CC에서 ours가 각각 **.59 대 .60**, **.63 대 .71**로 지며, Swin의 timeline RMSE도 **68.5 대 69.8**로 더 낮다. (Collin pp.17–18, Table 3/§4) | **불일치** |

따라서 “딥러닝을 전부 이겼다”가 아니라, 평가 대상·지표에 따라 단순 모델이 경쟁적이었다고 써야 한다. 특히 Collin 표의 각 `Ours` 행은 비교 모델별 available-date intersection이 달라질 수 있다. (Collin p.17, Table 3 주석)

### 4.1–4.2 HSS peak와 분포변환

| `DOMAIN.md` 주장 | PDF 실제 내용 | 판정 |
|---|---|---|
| MSE가 peak를 낮추고 peak timing shift가 RMSE를 키운다 | Collin은 MSE가 큰 편차를 억제해 HSS peak를 체계적으로 낮추며, 많은 peak가 시간상 이동되어 RMSE가 커진다고 분석한다. 다만 “timing이 항상 amplitude보다 더 큰 원인”이라는 일반화는 수치 분해로 입증하지 않는다. (Collin pp.11, 20, §§3.2–3.3, 6) | **일치** *(강한 인과 표현은 완화)* |
| Box–Cox/분위수 매핑으로 HSS peak RMSE 22–23% 개선 | 논문에서 실제 사용한 것은 **Box–Cox 기반 distribution transformation**이다. 4×3 HSS peak RMSE는 **113.3→87.5 km/s, 23%**, 10×10은 **98.9→76.8 km/s, 22%**다. (Collin pp.7, 13, §2.2.3/Table 2) | **일치** *(범위 보완)* |
| 이 개선은 전체 HSS/RMSE 개선이다 | 값은 전체 timeline이 아니라 관측·예측 peak를 매칭한 HSS peak subset의 RMSE다. 변환 후 timeline RMSE는 4×3에서 **68.1→75.1**, 10×10에서 **72.1→79.0**으로 오히려 증가한다. (Collin p.13, Table 2 및 본문) | **불일치** |
| “분위수 매핑”도 논문이 검증했다 | 논문은 histogram matching을 유사 방법으로 언급하지만 실험한 변환은 Box–Cox다. (Collin p.7, §2.2.3) | **불일치** |

### 5. 27일 주장

| `DOMAIN.md` 주장 | PDF 실제 내용 | 판정 |
|---|---|---|
| 회합 자전 주기 27.2753일, 항성 기준 25.38일 | Richardson은 지구에서 본 반복을 약 **27일**, 태양의 sidereal rotation을 **25.38일**이라고 쓴다. 정확한 27.2753일은 쓰지 않는다. (Richardson pp.1, 4, Abstract/§2.1) | **출처없음** *(27일 근사는 지지)* |
| 저위도 코로나 홀이 27일보다 약간 빠르게 재현된다 | Cranmer는 저위도 코로나 홀의 존재와 수명이 수 회전일 수 있음을 말하지만 그런 recurrence period 비교를 하지 않는다. Richardson의 약 26일 언급도 Ulysses 관측의 회전 간격이지 저위도 CH의 “27일보다 빠른 주기” 결론이 아니다. (Cranmer pp.4–5, 8; Richardson p.38, §6) | **출처없음** |

## 2. `DOMAIN.md` §9의 공백 보완

### 2.1 보정되지 않은 intensity에서 CH segmentation

- **임계 민감도·affine rescaling 실험:** 네 논문에는 없다. 35% 임계의 rescaling 민감도, per-image normalization, rank/quantile 대안의 비교표가 없다. 이 질문에 대한 결론은 **근거없음**이다.
- **논문에서 실제로 한 것:** Son은 FITS 단계에서 회전·중심정렬·노출 보상·감도 열화 보상을 SolarSoft로 수행한 뒤 64×64 PNG로 저장했다. 이는 calibration drift를 다루는 전처리이지만, 잔여 drift나 threshold sensitivity의 실험은 아니다. (Son p.2, §2)
- Collin은 193/211 두 채널을 결합해 k-means `k=3`으로 CH·활성영역·quiet Sun을 분할하고, binary map을 256×256으로 downsample했다. 고정 intensity threshold와의 민감도 비교는 없다. (Collin p.4, §2.1.1)
- Cranmer는 필라멘트를 CH에서 제외해야 한다고 하고, 서로 다른 장비의 absolute calibration uncertainty가 density 비교를 넓힐 수 있다고 경고하지만, 영상 segmentation용 보정법을 제시하지 않는다. (Cranmer pp.5, 10, §3/§4.2)

**판정:** 이 네 편이 주는 것은 “Son은 보정된 원자료를 사용했다”와 “Collin은 adaptive한 k-means를 사용했다”까지다. PNG의 per-image affine 변환이 35% 임계를 깨뜨리는지, 어떤 상대 임계가 안전한지는 **근거없음**.

### 2.2 HSS peak timing 대 amplitude

- Collin은 관측·예측 enhancement를 연결할 때 peak 시각 차이가 **3일 이내**면 match로 간주하고, hit/miss/false alarm을 계산한다. 하지만 ±3일 안의 실제 timing error 분포, amplitude error와 timing error의 RMSE 분해는 보고하지 않는다. (Collin p.6, §2.1.4; p.7, §2.3.1)
- 같은 논문은 matched HSS peak의 속도 RMSE/MAE/CC를 별도로 계산하고, fixed forecast horizon 때문에 peak가 시간상 이동한다고 논의한다. 따라서 amplitude 쪽 지표와 timing의 **거친 match 규칙**은 있으나, 요청한 정량 split은 없다. (Collin pp.7–8, §§2.3.1–2.3.2; p.15, §3.4)
- Son은 특정 equatorial CH 사례에서 상승 시각도 잘 맞았다고 그림으로 보이지만, 전 test set의 arrival-step error를 별도 보고하지 않는다. (Son p.3, §4, Fig.4; p.6, §5)
- Richardson은 다른 SIR 예측 연구의 interface arrival offset 평균·절대값을 소개하지만, Son/Collin 영상 모델의 HSS peak amplitude와 결합한 분해가 아니다. (Richardson p.80, §12, Fig.76)

**판정:** 네 논문에는 `peak height error / arrival-step error / width error`의 독립 진단이 없다. `3일`은 오차 측정값이 아니라 event matching tolerance다.

### 2.3 image branch가 실제로 학습한 것

- Son의 근거는 정성적이다. 잘 맞은 상승 사례에서 동쪽 limb에서 서쪽 limb로 이동하는 뚜렷한 equatorial CH가 공통으로 보이고, 저자들은 이를 image branch가 CH source를 학습한 이유로 해석한다. ablation, saliency, pixel attribution은 없다. (Son pp.3–4, §4, Figs.4/6)
- Collin의 permutation importance는 이미지 자체가 아니라 segmentation에서 나온 CH area를 분석한다. timeline에서는 중앙 부근 CH area, 특히 4–5일 전, HSS peak에서는 위치·위도 구분이 중요했고, 27일 전 풍속 `v^(27)`도 강한 feature였다. (Collin pp.10–11, §3.2, Fig.5)
- 그러므로 “영상이 위치·면적·시간 변화를 준다”는 근거는 있지만, Son의 두 branch 중 image와 과거 wind 중 어느 쪽이 더 기여했는지는 실험하지 않았다. Collin의 `v^(27)` 중요도는 과거 풍속 history 의존성에 대한 간접 증거지만, 본선의 5일 wind branch와 같은 설정이 아니며 27일 lag를 직접 입력으로 쓴다. (Collin pp.4, 10–11; Son p.2)

**판정:** image saliency/ablation은 **근거없음**. “past wind history에 더 의존한다”도 Son 설정에 대해서는 **근거없음**, Collin의 별도 모델에 대해서만 부분적 근거가 있다.

### 2.4 72 h 예측 가능성의 바닥

- Son의 72 h 결과는 RMSE **68.2 km/s**, CC **.67**이지만 이는 달성 성능이지 irreducible floor가 아니다. CME 정보가 입력에 없어 CME 기인 증가를 놓친다는 한계는 명시하지만 CME가 RMSE에 기여한 양을 분해하지 않는다. (Son pp.1, 5–7, Abstract/§4–5)
- Collin은 cycle phase가 바뀌면 성능이 악화되는 것을 수치로 보인다. cycle 25 운용에서 첫 설정의 timeline RMSE는 **74.5**, HSS peak RMSE는 **131.6**이고, 다른 재학습·분포변환 설정은 timeline **80.3**, HSS peak **92.2**다. 이것은 distribution/cycle shift의 관측값이지 irreducible error의 증명은 아니다. (Collin pp.19–21, §5–7)
- Richardson과 Cranmer는 SIR 형성·진화와 CH–fast-wind 관계를 설명하지만, 영상만으로 72 h에 달성 가능한 error ceiling을 수치화하지 않는다. (Richardson pp.14–16, 28–32; Cranmer pp.4–8)

**판정:** CME 기여량, 물리적 irreducible floor, 72 h ceiling은 네 논문에서 **근거없음**. 인용 가능한 것은 Son의 72 h 성능과 CME blind spot, Collin의 cycle-shift 성능 저하뿐이다.

### 2.5 `RULES.md` 제약과 전략의 충돌

1. **Son 설정을 “동일”이라고 쓰면 안 된다.** 본선은 영상 6 h, Son은 영상 12 h다. 풍속 6 h와 72 h 출력만 일치한다. (Son p.2; `RULES.md` §3)
2. **Collin의 핵심 신호는 그대로 제출 feature가 될 수 없다.** 논문은 27일 전 V, N, pressure, T, B, 태양위도·sunspot, Richardson–Cane CME list와 날짜 기반 180일 CV를 사용한다. 본선 입력은 20개 과거 V와 193/211 PNG뿐이고 외부 데이터는 금지다. (`RULES.md` §§1, 3; Collin pp.3–6, 8)
3. **27일 persistence baseline을 명시적으로 계산할 수 없다.** Richardson·Collin이 약 27일 recurrence를 사용해도, 익명 test의 시계열 복원은 `RULES.md` §5-3에서 하지 말 것으로 되어 있고, 현재 행에는 120 h만 제공된다. `DOMAIN.md` §9.1의 “27일 lag 사망” 판정이 맞다. (Richardson p.1; Collin pp.4–5, 11; `RULES.md` §§3, 5-3)
4. **외부 CME 제거를 대회 성능으로 재현할 수 없다.** Collin의 68.1/76.8 등은 Richardson–Cane 목록으로 CME를 제거한 curated evaluation이다. 외부 ICME 목록을 쓸 수 없는 본선과 같은 평가가 아니다. (Collin pp.5–8, 13; `RULES.md` §1)
5. **살아남는 문헌 근거는 제한적이다.** 제공된 영상·풍속으로 Son식 multimodal 입력을 학습하는 발상과, Collin의 “CH 위치·면적이 peak에 중요”라는 물리적 가설은 남는다. 그러나 35% threshold, 27일 lag, 외부 calibration/CME/OMNI feature, 180일 날짜 CV를 논문 성능 수치와 함께 가져오는 것은 이 contest 제약과 충돌한다. (Son pp.2–3; Collin pp.4, 10–11; `RULES.md` §§1, 3)

## 3. 문서에 바로 반영할 정정

- §1.5의 “코로나 홀 HSS 약 70%”를 **“Son 시험에서 Jian 기준 HSS 36건 중 25건을 포착(약 70%); CME 기인 증가를 예측하지 못함”**으로 고친다. (Son p.5)
- §2.2의 35% 행에는 **출처없음**을 붙이고, median/mean과 Vršnak/Krista 귀속을 네 PDF 근거로 단정하지 않는다.
- §3.1의 2012–2020 비교표는 네 PDF에서 검증되지 않으므로 삭제하거나 원출처 PDF를 별도로 확보한다.
- §3.2의 “대회와 동일”을 삭제하고, **Son 영상 12 h 대 본선 영상 6 h**를 명시한다.
- §3.4의 “모든 모델을 이김”을 지표별 비교로 바꾸고, Box–Cox 개선은 **CME 제거·matched HSS peak subset**에 한정한다.
- §5의 27.2753일 및 저위도 CH의 “27일보다 빠른 재현”은 이 네 PDF에서 검증하지 못했으므로 출처를 보류한다.

