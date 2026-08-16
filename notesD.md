# 세 교과서에서 추린 contest 물리

페이지는 PDF 페이지가 아니라 책에 인쇄된 쪽수다. 목차와 본문이 확인되는 범위만 썼다. `Space Physics`는 OCR/텍스트 추출이 비어 있는 스캔본이라 관련 페이지를 렌더링해 읽었다.

## 1. Nicole Meyer-Vernet, *Basics of the Solar Wind*

### 1) 실제 목차 확인

저작권면의 초판 표기는 Cambridge University Press, 2007이다. 실제 목차의 장 구성은 다음과 같다 (Meyer-Vernet 2007, 목차 pp. vii–xii).

| 장 | 실제 제목 | 책의 쪽수 |
|---|---|---:|
| 1 | The wind from the Sun: an introduction | 1–37 |
| 2 | Tool kit for space plasma physics | 41–110 |
| 3 | Anatomy of the Sun | 113–160 |
| 4 | The outer solar atmosphere | 165–218 |
| 5 | How does the solar wind blow? | 223–286 |
| 6 | Structure and perturbations | 291–330 |
| 7 | Bodies in the wind | 335–413 |
| 8 | The solar wind in the Universe | 419–449 |

DOMAIN.md §10의 `4장·5장·6장`이라는 큰 방향은 맞지만, 이 PDF의 판본은 2007년 초판이며 §10이 제시한 세부 절 제목·쪽수는 목차와 다시 맞춰야 한다.

### 2) 읽을 가치가 있는 장·절

| 절 | 쪽수 | 이 contest에서 답하는 질문 |
|---|---:|---|
| 1.1–1.3 | 1–37 | fast/slow wind가 어떤 큰 구조와 태양활동 상태에 붙는가? |
| 3.2.5, 문제 3.4.3 | 목차상 135, 158 | 태양회전과 위도별 차이가 5일 EUV 시퀀스에 어떤 각도 불확실성을 주는가? |
| 4.1–4.3, 4.5–4.7 | 166–217 | coronal hole·closed loop·open field가 EUV에서 어떻게 보이는가? |
| 5.1–5.5 | 225–273 | 열린 자기장, 압력, 가속이 wind speed를 어떻게 제한하는가? |
| 6.1–6.3 | 291–314 | stream interface와 SIR/CIR의 상승·감쇠 비대칭은 왜 생기는가? |

### 3) 추출

- 1 AU의 대표값은 fast wind 약 **750 km/s**, slow wind 약 **400 km/s**다. fast wind의 전자밀도는 약 2.5 cm⁻³, slow wind는 약 7 cm⁻³이며, 같은 표의 ram pressure는 각각 약 2.6×10⁻⁹ Pa와 2.1×10⁻⁹ Pa다. 출력 clipping의 중심·꼬리를 `600–800` 대 `300–500 km/s`로 넓게 잡는 물리적 근거가 된다 (Meyer-Vernet 2007, p. 33).
- 활동 극소기에는 큰 극 coronal hole과 밝은 적도 streamer가 공존하고, 빠른 바람은 적도 ±20°의 좁은 띠를 제외하면 넓게 나타난다. 극대기에는 fast stream이 좁아지고 slow wind가 더 넓은 위도에 퍼진다. 이미지에서 CH의 존재만 세기보다, 위도·크기·경계와 활동 상태를 함께 보아야 한다 (Meyer-Vernet 2007, pp. 32–34).
- CH는 영상에서 상대적으로 어둡다. 지배적인 자기장은 멀리 뻗어 되돌아오지 않는 반면, 작은 closed loop만 coronal height까지 도달한다. 따라서 dark region의 면적뿐 아니라 loop-like bright structure와 경계의 복잡도를 분리하는 것이 결정에 직접 연결된다 (Meyer-Vernet 2007, p. 191).
- fast/slow 경계에서는 fast flow가 앞선 slow flow를 압축한다. 선행면은 slow wind 쪽, 후행면은 fast wind 쪽이며, 후방 압력 교란은 fast flow에 대해 태양 쪽으로 전파되어 fast stream을 감속한다. 그래서 속도는 경계에서 빠르게 올라가고, 뒤의 rarefaction에서는 느리게 풀리는 비대칭이 생긴다 (Meyer-Vernet 2007, pp. 308–309).
- frozen-in 조건 때문에 두 흐름은 자유롭게 섞이지 않는다. 압축과 자기장 구조는 거리와 함께 발달하고, 압력파가 shock으로 가파르게 되는 것은 대체로 수 AU 바깥이다. 따라서 1 AU에서는 SIR의 ramp/압축은 featureable하지만 shock을 항상 가정하면 안 된다 (Meyer-Vernet 2007, p. 309).
- CH 중심에서 나오는 fast wind와 CH 경계·streamer 부근의 slower wind는 같은 CH mask 안에서도 다르다. 자기장 선의 발산이 빠르면 작은 source가 큰 flux를 낼 수 있다는 해석과, 더 넓은 영역에서 거의 방사형으로 나온다는 해석이 함께 존재하며 source mapping 자체가 논쟁적이다. CH 면적→속도 선형 매핑은 피해야 한다 (Meyer-Vernet 2007, pp. 305–307).
- CME는 coronal에서 수천 km/s까지 가능하지만, 지구 궤도에서는 ambient wind에 의해 가속·감속되어 속도가 보통 ambient 범위에 가까워질 수 있다. CME의 질량은 약 10¹²–10¹³ kg, 평균 ecliptic mass flux 기여는 한 cycle 평균 약 1/10이나 변동이 크다. disk EUV만으로 CME의 시작·기하·도착 효과를 안정적으로 결정할 수 없다는 결론을 유지한다 (Meyer-Vernet 2007, pp. 309–310).
- 5일 동안 보이는 회전량은 고정된 한 값이 아니다. Meyer의 그림과 문제는 태양회전·differential rotation을 별도로 다루므로, 영상 정렬은 위도에 따른 drift를 허용해야 한다 (Meyer-Vernet 2007, 목차 p. 135 및 p. 158).

### 4) 버릴 장

- 2장: plasma 도구·수식은 이번 입력에서 직접 관측되는 EUV와 wind speed 이상의 제약을 주지 않는다.
- 3장: 태양 내부·핵·발전기 세부는 120 h 영상 feature와 무관하다. 단, 회전 절만 남긴다.
- 7장: 행성·달·혜성의 wind interaction은 태양 wind 예측 입력이 아니다.
- 8장: 다른 별의 wind는 contest의 태양 관측과 직접 연결되지 않는다.

## 2. Mark Moldwin, *An Introduction to Space Weather*

### 1) 실제 목차 확인

저작권면은 Cambridge University Press, **2008년 초판**이며 ISBN 978-0-521-86149-6이다. 따라서 DOMAIN.md §10의 “2판 CUP 2022/23”은 이 PDF와 다르다. 실제 목차는 다음과 같다 (Moldwin 2008, 목차 pp. vii–ix).

| 장 | 실제 제목 | 책의 쪽수 |
|---|---|---:|
| 1 | What is space weather? | 1–16 |
| 2 | The variable Sun | 17–36 |
| 3 | The heliosphere | 37–49 |
| 4 | Earth’s space environment | 50–67 |
| 5 | Earth’s upper atmosphere | 68–77 |
| 6 | Technological impacts | 79–94 |
| 7 | Perils living in space | 95–111 |
| 8 | Other space weather phenomena | 112–121 |

DOMAIN.md §10의 `2.6–2.7`은 실제 목차상 각각 solar rotation과 sunspot/solar cycle 부분으로 확인할 수 있지만, 이 책에서 contest에 직접 유용한 heliosphere material은 **3장 3.4–3.8**, 특히 pp. 39–41이다. §10의 “2판” 표기는 정정해야 한다.

### 2) 읽을 가치가 있는 장·절

| 절 | 쪽수 | 이 contest에서 답하는 질문 |
|---|---:|---|
| 2.6–2.7 | 27–35 | 회전·활동주기의 물리와, 날짜 없는 입력에서 무엇을 버려야 하는가? |
| 3.4–3.8 | 38–49 | open/closed field, CH, Parker spiral, 1 AU wind의 대표 범위는 무엇인가? |

### 3) 추출

- 태양 적도 회전은 약 25일, 극은 약 36일, 지구에서 중요한 평균은 약 27일이다. 따라서 5일의 nominal 66.7° 회전은 적도·평균값 근사일 뿐이고, feature의 위도에 따라 달라진다 (Moldwin 2008, p. 25).
- 적도 쪽 closed field는 느린 wind와 연결되고, 극 쪽 open field는 빠른 wind와 연결된다. coronal hole은 X-ray에서 어둡게 보인다 (Moldwin 2008, p. 40).
- 1 AU 평균은 density 약 5 cm⁻³, temperature 약 10⁶ K, speed 약 400 km/s이다. 이 값은 이번 입력에 없는 density/temperature를 복원하라는 뜻이 아니라, 속도 출력의 중심값과 low-speed tail의 물리적 기준이다 (Moldwin 2008, p. 41).
- CME는 질량이 10¹² kg보다 크고 속도가 1000 km/s보다 클 수 있으며, 빠른 CME는 shock을 만든다. 이는 CME-driven enhancement가 큰 오차 요인임을 보여 주지만, `800+ km/s`를 모든 CME의 대표값으로 쓰는 근거는 아니다 (Moldwin 2008, p. 41).
- Parker spiral의 태양-지구 방향에 대한 각도는 solar-wind speed에 의존하고, 빠른 wind일수록 spiral angle이 작다. 다만 이 관계는 source의 절대 위치·시간과 speed가 있을 때의 전파 물리이지, 날짜 없는 PNG에서 ballistic back-mapping feature를 만들 근거가 아니다 (Moldwin 2008, p. 39).

### 4) 버릴 장

- 1장: space-weather 정의와 역사적 도입은 물리 feature를 추가하지 않는다.
- 4–5장: 지구 자기권·전리층은 입력에 없는 지구 환경 자료를 요구한다.
- 6–8장: 기술 피해·우주인 위험·다른 현상은 target speed 예측에 직접 쓰이지 않는다.

## 3. Russell, Luhmann & Strangeway, *Space Physics: An Introduction*

### 1) 실제 목차 확인

저작권면은 Cambridge University Press, **2016년 초판**이며 ISBN 978-1-107-09882-4이다. 이 PDF는 텍스트층이 없는 스캔본이다. 실제 목차는 다음과 같다 (Russell, Luhmann & Strangeway 2016, 목차 pp. ix–xiii).

| 장 | 실제 제목 | 책의 쪽수 |
|---|---|---:|
| 1 | Solar-terrestrial physics: evolution of a discipline | 1–25 |
| 2 | The upper atmosphere and ionosphere | 26–52 |
| 3 | Physics of magnetized plasmas | 54–91 |
| 4 | The Sun and its atmosphere | 94–144 |
| 5 | Solar wind and heliosphere | 145–195 |
| 6 | Collisionless shocks | 199–221 |
| 7 | Solar-wind interaction with magnetized obstacles | 222–245 |
| 8 | Solar-wind interaction with unmagnetized bodies | 248–274 |
| 9 | Solar-wind–magnetosphere coupling | 275–312 |
| 10 | The terrestrial magnetosphere | 315–343 |
| 11 | Auroras | 346–371 |
| 12 | Planetary magnetospheres | 372–395 |
| 13 | Plasma waves | 396–432 |

DOMAIN.md §10의 `4.2–4.5, 5.2–5.3`이라는 장 선택은 맞지만, 이 PDF의 실제 판본은 2016년이고 §10에 없는 핵심은 5.1 및 5.3의 SIR/CIR 세부다.

### 2) 읽을 가치가 있는 장·절

| 절 | 쪽수 | 이 contest에서 답하는 질문 |
|---|---:|---|
| 4.2–4.6 | 95–142 | 태양회전, EUV 구조, open/closed corona, CME의 관측 한계는 무엇인가? |
| 5.1–5.3 | 145–183 | Parker wind, speed distribution, source dependence, SIR/CIR와 전파 지연은 어떻게 연결되는가? |
| 3.5, 3.7 | 70–90 | frozen-in·MHD 압축을 SIR 해석에 필요한 만큼만 확인한다. |
| 6.1–6.5 | 199–212 | shock edge를 별도 feature로 믿을 수 있는 범위는 어디까지인가? |

### 3) 추출

- 태양의 적도 회전은 약 25일, 극은 대략 30일이며 지구에서 본 Carrington형 평균은 약 27.3일이다. 책은 위도별 differential rotation을 명시하므로 66.7°/5일은 nominal 값으로만 써야 한다 (Russell, Luhmann & Strangeway 2016, pp. 100–101).
- EUV에서 Fe IX는 17.1 nm, Fe XII는 19.5 nm, Fe XV는 28.4 nm의 예로 제시된다. active-region loop는 밝고 coronal hole은 어둡다. 따라서 AIA intensity를 고유 temperature의 직접 측정값으로 쓰면 안 되며, DOMAIN의 193/211 channel-temperature 대응은 이 책만으로 211 Å/Fe XIV까지 검증되지 않는다 (Russell, Luhmann & Strangeway 2016, pp. 112–113, 117–120).
- open field의 corona는 약 1 MK·약 10⁸ cm⁻³, loop corona는 약 2 MK·약 10⁹–10¹¹ cm⁻³의 대표 규모로 설명된다. 밝기는 temperature 하나가 아니라 density와 loop length/자기구조에도 좌우된다. 그러므로 두 채널의 절대 밝기보다 dark/open 후보, loop texture, 경계 변화를 feature로 우선한다 (Russell, Luhmann & Strangeway 2016, pp. 121–123).
- 1 AU speed plot은 대략 **300–800 km/s** 범위를 보이고, 고속 구간은 대체로 **600 km/s 초과**, 저속 구간은 **350 km/s 미만**이며, 두 흐름의 경계는 자주 약 450 km/s 부근으로 나뉜다. 고속 stream은 활동이 낮을 때 약 27일 재현성이 더 뚜렷하고, 활동이 높을 때 흐려진다 (Russell, Luhmann & Strangeway 2016, pp. 151–155).
- source 쪽에서도 CH 중심의 큰 영역은 `>600 km/s`, CH boundary/streamer는 `<400 km/s` 쪽이다. 이는 단일 CH 면적보다 중심성·경계 거리·형태 안정성을 별도 signal로 보라는 근거다 (Russell, Luhmann & Strangeway 2016, pp. 156–157).
- SIR/CIR는 고속 stream이 앞선 저속 stream을 따라잡아 만드는 압축 구조다. 1 AU에서 density와 magnetic enhancement는 한 day 미만으로 비교적 대칭적일 수 있지만, velocity는 ramp-up을 보이고 뒤따르는 저속 stream이 있으면 뒤쪽에 rarefaction이 생긴다. 이 조합이 “빠른 상승·느린 감쇠” feature의 물리적 근거다 (Russell, Luhmann & Strangeway 2016, pp. 158–159).
- 책은 1 AU의 SIR/CIR에 약한 forward/reverse shock pair가 경계를 낄 수 있고 Mach number가 약 2인 예를 제시한다. 동시에 shock은 거리와 함께 더 흔해지고, stream 간 속도 차는 거리와 함께 흐려진다. 그러므로 1 AU shock은 금지된 feature가 아니라 낮은 신뢰도의 optional edge signal이다 (Russell, Luhmann & Strangeway 2016, pp. 158–159).
- CME의 coronagraph 시야에서 대부분 속도는 수백 km/s이고 수천 km/s는 꼬리다. 넓은 사건일수록 빠른 경향은 있지만, 속도·폭·방향은 projection 때문에 어렵다. 활동영역/filament의 전조는 위험 flag가 될 수 있어도, 72 h target speed의 크기·도착시각을 결정하지는 못한다 (Russell, Luhmann & Strangeway 2016, pp. 124–132).
- Parker spiral의 기하에는 `β = R₀Ω/V`가 나오므로 speed가 빠를수록 각도가 작다. 그러나 이번 contest에서 source longitude와 절대 시간이 없으므로 이 식은 물리 설명이지 usable back-mapping feature가 아니다 (Russell, Luhmann & Strangeway 2016, p. 149).

### 4) 버릴 장

- 1장: 역사·발견사는 입력 feature를 추가하지 않는다.
- 2장: 지구 상층대기·전리층은 target과 직접 연결되지 않는다.
- 7–12장: magnetosphere, 행성, aurora는 태양 원격영상만 있는 문제의 관측 범위를 벗어난다.
- 13장: plasma wave 이론은 20개 wind speed와 두 EUV 채널의 제한을 넘는다.

## 세 책이 DOMAIN.md를 정정하는 지점

| DOMAIN 위치 | 책·쪽수 | 정정 |
|---|---|---|
| §1의 “fast/slow 대표 범위” | Meyer-Vernet 2007, p. 33; Russell et al. 2016, pp. 151–160 | `fast=600–800`, `slow=300–500`은 대표 구간으로 타당하지만, 단일 경계나 `400→700+`를 보편 법칙으로 고정하면 안 된다. 책의 표와 그림은 저속 tail과 고속 tail이 겹칠 수 있음을 보여 준다. |
| §1의 “1 AU SIR은 대체로 shockless” | Meyer-Vernet 2007, p. 309; Russell et al. 2016, pp. 158–159 | Meyer는 shock이 수 AU에서 더 발달한다고 하지만, Russell 등은 1 AU에서도 약한 forward/reverse shock pair 사례를 든다. “대부분은 압축/ramp, shock은 가능하지만 약한 보조 signal”로 바꿔야 한다. |
| §1의 CME `800+ km/s` 및 EUV 비예측성 | Meyer-Vernet 2007, pp. 309–310; Moldwin 2008, p. 41; Russell et al. 2016, pp. 124–132 | CME는 수백 km/s가 흔하고 수천 km/s가 꼬리다. active-region/filament의 전조와 source flag는 있을 수 있지만, 속도·폭·방향·도착은 projection과 initiation 불확실성 때문에 disk EUV만으로 확정할 수 없다. `800+`를 정의값으로 쓰지 말아야 한다. |
| §2의 EUV channel을 온도계처럼 해석하는 부분 | Russell et al. 2016, pp. 112–113, 121–123 | 책은 EUV ion 예와 open/loop corona의 온도·밀도 규모를 주지만, 밝기가 density와 loop geometry에도 좌우됨을 명시한다. 193/211의 단일 temperature feature보다 morphology와 상대 intensity 변화를 우선한다. 211 Å/Fe XIV의 정확한 대응은 이 책에서 확인되지 않는다. |
| §5의 단일 `25.38일` 회전값 | Moldwin 2008, p. 25; Russell et al. 2016, pp. 100–101 | 적도 약 25일, 극 약 30–36일로 값이 위도에 따라 달라진다. 25.38일은 전 태양 feature에 적용할 보편 주기가 아니다. 5일의 66.7°도 latitude-dependent nominal value로만 써야 한다. |
| §1의 정확한 SIR 상승·감쇠 서술 | Meyer-Vernet 2007, pp. 308–309; Russell et al. 2016, pp. 158–159 | 비대칭은 물리적으로 지지되지만, 책이 보장하는 것은 fast-flow compression, velocity ramp, trailing rarefaction이다. 영상만으로 모든 event의 동일한 rise/decay template을 강제하면 안 된다. |

## 책에는 있는데 DOMAIN.md에 없는 것

- **CH 중심 대 경계:** CH 중심은 `>600 km/s`, 경계/streamer는 `<400 km/s` 쪽이므로 면적·중심성·경계까지 분리할 이유가 있다 (Russell et al. 2016, pp. 156–157).
- **밝기의 원인 분해:** open-field corona와 loop corona의 density/temperature 규모가 다르고, EUV brightness는 density·loop length·자기구조의 함수다. threshold 하나보다 두 채널의 texture/gradient/loop 후보와 지속성을 보존해야 한다 (Russell et al. 2016, pp. 117–123).
- **SIR event shape:** 1 AU에서 density/B enhancement는 짧고 비교적 대칭적일 수 있지만 velocity ramp와 후방 rarefaction은 비대칭이다. 최근 past-speed 20점에는 level보다 slope, ramp width, trailing decay를 넣을 물리적 근거가 있다 (Russell et al. 2016, pp. 158–159).
- **거리 효과:** SIR compression은 거리와 함께 커지고, 속도 차는 멀리서 흐려진다. 같은 CH signal이라도 speed-dependent delay와 event broadening을 분리해야 한다 (Russell et al. 2016, pp. 158–159; Meyer-Vernet 2007, p. 309).
- **CME는 이진 사건이 아님:** 대부분 수백 km/s, 빠른 수천 km/s는 꼬리이며, 폭이 넓을수록 빠른 경향과 projection 문제가 함께 있다. active-region brightening은 불확실성 flag로만 쓰는 편이 맞다 (Russell et al. 2016, pp. 124–132).

## 세 책으로도 안 되는 것

- headerless 193/211 PNG만으로 open magnetic topology, polarity, PFSS source footpoint를 확정할 수 없다. 책의 CH/open-field 설명은 자기장 지도·모델과 함께 읽어야 한다 (Russell et al. 2016, pp. 117–120; Meyer-Vernet 2007, pp. 305–307).
- 입력에 density·magnetic field·temperature·absolute time·ephemeris가 없으므로 SIR/CIR와 CME를 확정 분류하거나 event별 ballistic arrival을 식별할 수 없다. Parker spiral과 propagation delay는 물리적 prior이지 관측 가능한 정답 feature가 아니다 (Moldwin 2008, pp. 39–41; Russell et al. 2016, p. 149).
- 120 h window 밖의 CH lifetime, recurrence, launch history는 세 책이 채워 주지 않는다. 27일 recurrence는 책에 있어도 이번 입력 제약 아래 feature로 부활하지 않는다 (Moldwin 2008, p. 25; Russell et al. 2016, pp. 151–155).
- 세 책에는 AIA 193/211의 headerless PNG affine scale에서 재현 가능한 threshold, contest-specific image-to-speed mapping, 6 h별 72 h conditional error가 없다 (Meyer-Vernet 2007, pp. 32–34, 191; Moldwin 2008, pp. 39–41; Russell, Luhmann & Strangeway 2016, pp. 112–123, 151–159). 따라서 morphology의 방향과 출력 clipping 범위는 얻지만, 최적 threshold·lag·RMSE는 책만으로 결정되지 않는다.
