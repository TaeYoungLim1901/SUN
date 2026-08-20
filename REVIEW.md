# 저장소 적대적 정확성 검토

## 심각도별 집계

- 치명: 4건
- 높음: 8건
- 중간: 2건
- 낮음: 없음

### [치명] `kit.score`가 대회 RMSE가 아닌 horizon별 RMSE의 평균을 계산한다
- **위치**: tools/kit.py:29-53, tools/kit.py:202-210, SUN_PLAN.md:48-53, RULES.md:95-97
- **문제**: 규정의 지표는 모든 오차 제곱을 한 번 평균한 뒤 제곱근을 취하는 단일 RMSE인데, `score`는 horizon별 RMSE를 먼저 구한 뒤 산술평균한다. 이에 기대어 `sensitivity`와 SUN_PLAN도 단기 horizon의 같은 MSE 감소가 더 비싸고 horizon별 점수가 완전히 가법적이라고 잘못 결론 낸다. 힐클라이밍까지 이 점수로 후보를 고르므로 모든 모델 선택이 다른 목적함수를 최적화한다.
- **근거**: 두 horizon의 RMSE가 0과 10이면 RULES 식은 `sqrt((0^2+10^2)/2)=7.0711`이지만 현재 코드는 `(0+10)/2=5`를 반환한다. 현재 self-check의 완벽예측·전 셀 +10 검사는 두 식이 우연히 같은 경우만 검사해 이 차이를 잡지 못한다.
- **수정**: `score`를 `return float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))` 형태의 단일 RMSE로 교체한다. 결측 target을 허용한다면 `m=np.isfinite(y)`만 사용한다. `sensitivity`는 동일 표본 수에서 모든 horizon에 대해 `1/(2*H*score)`로 바꾸고, SUN_PLAN.md:48-53의 단기 horizon 우대·완전 가법성 문장을 삭제한다. self-check에는 horizon RMSE가 `[0, 10]`인 반례와 기대값 `sqrt(50)`을 추가한다.

### [치명] 채점기가 NaN 예측과 20열 예측을 정상 결과로 받아들인다
- **위치**: tools/kit.py:29-42, tools/kit.py:159-173, tools/kit.py:202-228
- **문제**: `rmse_per_horizon`은 prediction의 NaN도 target 결측처럼 제외한다. 따라서 어려운 셀만 NaN인 모델이 부당하게 좋아지고 `hill_climb`이 이를 선택할 수 있다. 또한 `y.shape == p.shape` 검사가 없어 target이 12열이어도 prediction의 앞 12열만 읽고 뒤의 8열을 조용히 무시한다. 20→12 수정이 덜 된 모델도 로컬 PASS가 가능하다.
- **근거**: 마스크가 `isfinite(y) & isfinite(p)`라 NaN prediction은 벌점이 아니라 표본 삭제다. 반복문도 `y.shape[1]`까지만 돌아 `(n,20)` prediction의 13~20번째 열을 보지 않는다. self-check는 target 쪽 NaN만 시험하고 prediction NaN·shape 불일치를 시험하지 않는다.
- **수정**: 계산 전에 `if y.shape != p.shape: raise ValueError(...)`와 `if not np.isfinite(p[np.isfinite(y)]).all(): raise ValueError(...)`를 넣는다. self-check에 `(n,20)` prediction과 prediction NaN이 각각 실패하는 assert를 추가한다.

### [치명] 공개 커널·남의 제출 예측을 최종 혼합 축으로 쓰도록 지시한다
- **위치**: PLAYBOOK.md:209-214, SUN_PLAN.md:179-180, tools/kit.py:183-193, RULES.md:28-33, RULES.md:47-50
- **문제**: SUN_PLAN은 공개 커널 예측을 회귀 분해한 뒤 혼합비를 외삽하도록 하고, `decompose`도 대상을 “남의 제출”이라고 명시한다. 공개/타 팀의 test prediction은 대회 제공 데이터도 아니고 제출한 `model.pth`와 `code.ipynb`로 생성한 결과도 아니다. 이를 최종 예측에 반영하면 외부 데이터와 재현 검증 양쪽에 걸린다.
- **근거**: RULES §1 1항은 대회 제공 외 데이터 사용을 금지하고 3·4항은 제출 코드의 동작·성능 재현을 요구한다. 외부 prediction에서 추정한 혼합비는 그 외부 prediction 없이는 재생성할 수 없다. REPOS.md:32,128은 현재 공개 커널 벤치마크가 없다고도 적어 실행 계획 자체와 충돌한다.
- **수정**: PLAYBOOK과 SUN_PLAN의 “공개 커널·남의 제출”을 “동일 팀이 제출 notebook에서 재현 가능한 자체 모델의 OOF/test prediction”으로 교체한다. `decompose` docstring도 같은 범위로 제한하고 외부 prediction을 사용한 결과는 최종 제출에 쓰지 않는다고 명시한다.

### [치명] REPOS의 직접 이식 목록이 금지·미제공 입력을 다시 연다
- **위치**: REPOS.md:85-92, REPOS.md:129-130, DOMAIN.md:372-383, RULES.md:28-33
- **문제**: CHIMERApy는 없는 171Å를 요구하고, heliocats의 ICME 카탈로그와 Predstorm/OMNI 로더는 외부 데이터를 요구하며, aiapy level 1→1.5는 FITS 헤더·시각이 필요하다. HUXt의 WSA FITS 기반 기저도 제공 입력만으로 만들 수 없다. 그런데 §5는 CHIMERApy를 “즉시 조달”, HUXt를 “기저 1장 확보”라고 실행 항목으로 남겼다.
- **근거**: RULES §3의 제공 채널은 193/211뿐이고 §1 1항은 외부 데이터를 금지한다. DOMAIN §9.1은 OMNI·ICME 카탈로그·aiapy·ephemeris 경로를 각각 불가로 판정했다.
- **수정**: REPOS.md:85,88-92에 각각 `제출 사용 불가 — 171 미제공`, `외부 WSA/OMNI/ICME 데이터 필요`, `PNG 헤더 부재`를 표시한다. REPOS.md:129-130의 즉시 조달·기저 확보 지시는 삭제하고, 제공 193/211 PNG만 받는 코드 조각만 후보로 남긴다.

### [높음] `factcheck.py`가 무응답·오류·malformed 결과를 깨끗한 PASS로 바꾼다
- **위치**: tools/factcheck.py:120-159, tools/factcheck.py:189-243, POSTMORTEM.md:109-117
- **문제**: subprocess return code와 stderr를 버리고 stdout만 반환한다. 판정 블록이 없으면 `continue`, tagged block이 있으나 유효 행이 없으면 빈 목록을 정상 결과로 캐시한다. 두 경우 모두 거짓·과장 0건으로 exit 0이다. 2열뿐인 잘린 행도 5열로 패딩해 받아들인다.
- **근거**: 임시 corpus/target에서 `run_luna`를 “판정 블록 없음”과 malformed fenced block으로 대체해 `main`을 실행한 결과 둘 다 `거짓 0 · 과장 0 · 미확정 0`, return code 0을 재현했다. POSTMORTEM도 같은 실패가 아직 `continue`로 남았다고 기록한다.
- **수정**: subprocess return code가 0이 아니면 즉시 실패시키고 stderr를 raw에 남긴다. `_rows`는 정확히 5열만 허용한다. 빈 결과는 프롬프트에 별도 `NO_CLAIMS` marker를 강제해 명시적 무주장과 파싱 실패를 구분한다. `failed_sections`를 세어 `wrong or failed_sections`이면 exit 1로 반환한다.

### [높음] `factcheck.py` 캐시가 PDF·프롬프트·모델 변경 후에도 옛 판정을 재사용한다
- **위치**: tools/factcheck.py:47-102, tools/factcheck.py:181-199
- **문제**: 캐시 digest는 절 본문과 PDF **경로 문자열**만 해시한다. 같은 경로의 PDF를 교체하거나 판정 프롬프트·모델·reasoning 설정을 고쳐도 hash가 같아 stale verdict가 살아남는다. cache key도 절 제목만 써 같은 제목이 중복되면 충돌한다.
- **근거**: digest 식은 `(body + corpus)`이고 `corpus`는 파일 내용이 아니라 경로 join 결과다. `PROMPT`, `MODEL`, PDF bytes, 절 순번이 모두 빠져 있다.
- **수정**: 실행 시작 시 각 PDF bytes의 SHA-256을 합친 `corpus_digest`를 만들고, cache digest에 `target.resolve()`, 절 순번·제목·본문, `PROMPT`, `MODEL`, `corpus_digest`를 모두 포함한다.

### [높음] `conform.py`의 줄 전체 면제가 실제 위반을 숨긴다
- **위치**: tools/conform.py:24-31, tools/conform.py:35-40, tools/conform.py:78-91, tools/kit.py:7
- **문제**: 한 줄 어디에든 `아니다`, `불가`, `금지`, `~~` 등이 있으면 줄 전체를 건너뛴다. 서로 다른 절의 정상 정정 문구와 되살아난 오류가 같은 줄에 있으면 false PASS다. M03 정규식도 `H=20`을 못 잡아 현재 `kit.py`의 stale “H=20이면 5일” 문장을 놓친다.
- **근거**: `이 문장은 거짓이 아니다. 출력은 20개 horizon 이다`와 `~~옛 기록~~ 출력은 20개 horizon 이다`를 임시 파일로 스캔했을 때 둘 다 hit 0이었다. 실제 저장소에서도 `python tools/conform.py`는 위반 0으로 끝났지만 tools/kit.py:7에는 H=20이 남아 있다.
- **수정**: `EXEMPT` 기반 줄 전체 skip을 없앤다. 취소선 구간만 `re.sub(r"~~.*?~~", "", line)`로 제거하고 `[기록]`은 줄 맨 앞의 기록 전용 문장만 면제한다. M03에 `r"(?:N_HORIZON|\bH)\s*=\s*20"`을 추가하고, 같은 줄에 정정문과 위반문이 함께 있는 self-test를 추가한다.

### [높음] Box-Cox 정정 후에도 실행표가 다시 “피크 −22%”를 권고한다
- **위치**: SUN_PLAN.md:195, DOMAIN.md:276-291, POSTMORTEM.md:33-34, POSTMORTEM.md:48-50
- **문제**: SUN_PLAN은 피크 시간 어긋남을 “실제 원인”으로 단정하고 Box-Cox/분위수 매핑의 `피크 RMSE −22%`를 대응책으로 제시한다. 이는 POSTMORTEM M06에서 이미 고친 부분집합→전체 일반화의 재발이다. 본선 지표는 전체 timeline RMSE다.
- **근거**: Collin p.13에서 Box-Cox는 matched HSS peak subset을 22~23% 개선했지만 전체 timeline RMSE는 4×3에서 68.1→75.1, 10×10에서 72.1→79.0으로 악화됐다. DOMAIN §4.1은 timing이 amplitude보다 항상 큰 원인이라는 수치 분해도 없다고 명시한다. 분위수 매핑은 해당 논문이 실험한 방법도 아니다.
- **수정**: SUN_PLAN.md:195를 `피크 timing은 원인 후보 중 하나이며 미분해. Box-Cox는 peak subset만 개선하고 timeline RMSE를 악화시켰으므로 대회 score의 OOF 검증 통과 때만 채택`으로 교체하고 `분위수 매핑 −22%` 표현을 삭제한다.

### [높음] 헤더 없는 PNG에 불가능한 aiapy 열화·노출 보정을 필수로 계획한다
- **위치**: SUN_PLAN.md:196,198, CLOSED_AXES.md:25, REPOS.md:92, DOMAIN.md:380, DOMAIN.md:411-424
- **문제**: DOMAIN §9에서 aiapy level 1→1.5와 AIA 열화 보정을 불가로 판정했는데 SUN_PLAN은 aiapy annulus·열화 보정을 실행/필수 항목으로 유지하고 CLOSED_AXES는 열화 보정 생략을 무조건 실패로 닫았다. 8-bit PNG에는 보정에 필요한 exposure time·관측시각·포인팅 헤더가 없다.
- **근거**: DOMAIN §9.3은 노출 정규화와 시간 의존 열화 보정을 불가, limb brightening은 채널별 방사 profile을 이용한 근사만 가능하다고 구분한다. PNG가 per-image 재스케일됐으면 전역 감도 저하는 이미 대부분 상쇄될 수도 있어 “생략하면 무조건 깨진다”도 성립하지 않는다.
- **수정**: SUN_PLAN의 aiapy/열화 보정 지시를 `PNG에서 물리 보정 불가; disk-relative/rank 표현과 방사 profile 근사만 OOF 비교`로 교체한다. CLOSED_AXES.md:25는 삭제하거나 `FITS/aiapy 물리 보정 시도 — 헤더·시각 부재로 불가`로 반대로 닫는다.

### [높음] “시드만 바꾼 널 제출 1장”으로 σ와 Public→Private 전이를 추정할 수 없다
- **위치**: PLAYBOOK.md:101-114, SUN_PLAN.md:99-109, CLOSED_AXES.md:27, REPOS.md:25-27,126-127
- **문제**: 학습 seed를 바꾸면 모델과 prediction이 실제로 바뀌므로 null 조작이 아니다. 그 public 점수 차이는 평가기의 랜덤 노이즈가 아니라 두 모델의 실제 차이다. 더구나 한 쌍으로 `sd(Δ)≈k√dist`의 k를 적합할 수 없고, public 점수만으로 숨겨진 private 전이 분산을 관측할 수도 없다. 제출 1장을 태우고 근거 없는 채택선을 얻게 된다.
- **근거**: 리더보드 RMSE는 같은 prediction에 결정론적이다. REPOS의 이전 대회 Public/Private 점수 한 쌍이 가깝다는 사실도 split이 무작위라는 증거나 분산 추정치가 아니다. 평균 하나가 우연히 가까운 것과 분산이 작은 것은 다른 명제다.
- **수정**: seed 변동은 여러 seed의 OOF/block-bootstrap으로 로컬 추정하고 제출을 쓰지 않는다. `private_sigma`의 k는 과거의 다수 paired Public/Private 변화량으로 보정된 경우에만 사용하며, 그런 자료가 없으면 수치 σ·2σ 채택선 주장을 삭제하고 휴리스틱으로 명시한다.

### [높음] 실제 sample count와 67/8/25 분할 비율이 동시에 성립하지 않는다
- **위치**: RULES.md:70-72, RULES.md:89-90, SUN_PLAN.md:90-92, POSTMORTEM.md:92-94
- **문제**: RULES는 실제 행 수와 분할 비율을 모두 확정값처럼 쓰지만 산술이 맞지 않는다. 후속 문서는 이 비율로 public/private 표본 수와 σ 논리를 전개한다.
- **근거**: 총 행 수는 `9,607+1,199+3,868=14,674`. 실제 비율은 train `65.47%`, validation `8.17%`, test `26.36%`이며 67/8/25가 아니다. test 절반 1,934개도 전체의 `13.18%`이지 12.5%가 아니다.
- **수정**: RULES.md:89-90을 `설명회 명목 비율 67/8/25; 공개 디렉터리 실현 비율은 65.47/8.17/26.36이며 차이 원인은 미확인`으로 교체한다. Public/Private 각각 1,934개라는 값은 실제 split count가 확인되기 전에는 추정으로 표시하고 통계 계산에는 실제 index 수를 사용한다.

### [높음] 20개 영상 프레임 안의 자전량은 66.7°가 아니라 약 63.3°다
- **위치**: DOMAIN.md:442-447, ONBOARDING.md:104-108, SUN_PLAN.md:192, RULES.md:60-64
- **문제**: 문서는 “5일 창”의 경계 폭 120h를 첫 프레임과 마지막 프레임 사이 시간으로 사용한다. 6시간 간격 20시점에는 간격이 19개뿐이므로 실제 관측 프레임 중심 간 span은 114h다. 광류·위치 궤적의 예상 이동량이 과대 계산됐다.
- **근거**: `(20-1)×6h=114h=4.75일`, 따라서 `360°×4.75/27=63.33°`. `360°×5/27=66.67°`는 120h 양 끝에 모두 프레임이 있을 때의 값으로, 그러려면 6시간 간격 21시점이 필요하다.
- **수정**: 세 문서의 `66.7°`를 `프레임 중심 기준 약 63.3°(114h); 120h window edge 기준 명목값은 66.7°`로 교체한다.

### [중간] 폐쇄축이 논문이 시험하지 않은 “고해상도 분할 무가치”까지 결론 낸다
- **위치**: CLOSED_AXES.md:24, DOMAIN.md:155-168, DOMAIN.md:248-268
- **문제**: Collin의 4×3 결과는 k-means로 만든 CH binary map을 어느 격자로 집계할지 비교한 결과다. 원 PNG 해상도나 분할 정교도를 낮춰도 된다는 실험이 아니다. 그런데 CLOSED_AXES는 이를 근거로 고해상도 CH 분할 전체가 값을 못 낸 축이라고 닫았다.
- **근거**: DOMAIN §2.2는 “정교한 분할에 시간 태우지 말라”는 결론이 PDF로 검증되지 않아 별도 검증이 필요하다고 이미 정정했다. 4×3 aggregation과 segmentation fidelity는 다른 축이다.
- **수정**: CLOSED_AXES.md:24를 삭제한다. 닫아야 한다면 실제 contest OOF에서 동일 mask의 aggregation 해상도만 비교한 실험 ID와 재개 조건을 근거로 적고, 분할 정교도와 합치지 않는다.

### [중간] 실험 로그의 첫 활성 항목이 이미 불가능 판정된 27일 기저다
- **위치**: experiments/log.md:3, CLOSED_AXES.md:17-18, DOMAIN.md:372-373
- **문제**: 로그의 유일한 실험 E000이 `27일 지속성 | 기록 시작`으로 남아 있어 현재 실행 후보처럼 보인다. DOMAIN §9와 CLOSED_AXES는 같은 축을 제출 불가로 닫았다. 늦은 규정 수정이 실험 자산까지 전파되지 않은 class D 사례다.
- **근거**: 입력은 20개/120h뿐이고 test 시간순 복원은 금지라 27일 전 값을 얻을 수 없다.
- **수정**: E000의 판단을 `폐기 — DOMAIN §9.1, 제출 계산 불가`로 바꾸고 변경점을 `기록용(실행 금지)`로 명시한다.

## 확인했으나 문제 없음

- **규정 상수**: 출력 12개, 6시간 간격, 미래 72h, 입력 20시점·2채널·풍속 20개, `inputs.csv` 41열=`1+20+20`, target/submission 13열=`1+12`, 193Å/211Å, PNG 512×512, 제출 파일 `code.ipynb`·`submission.csv`·`model.pth`, 하루 5회·자정 초기화는 지정 문서 전반에서 일치했다. `N_HORIZON` 실행 상수도 12다. 표본 비율 산술만 위 finding과 같이 불일치한다.
- **탄도 지연표**: `1 AU=149,597,870.7 km`로 재계산한 DOMAIN의 300/400/500/700/800 km/s 행은 각각 5.77/4.33/3.46/2.47/2.16일로 반올림값이 맞다. ONBOARDING의 공통 행은 일치하고 추가 600/750 km/s 행도 2.89/2.31일로 맞다.
- **DOMAIN §9.2 정합표**: `target_00`, `target_05`, `target_11`의 t−4d/t−7d 시각 산술은 각각 −90/−162h, −60/−132h, −24/−96h로 맞다.
- **블록 CV 핵심 gap**: `block_folds`는 validation 블록과 앞뒤 `gap` 블록을 train에서 제외하며 현재 self-check의 train/validation 비중첩·인접 블록 제외 조건도 논리상 맞다. 단, 대회 test에는 절대 시각이 없어 제출 피처로 쓰지 않는다는 제한도 문서에 반영돼 있다.
- **지속성 기저**: `persistence_last` 기본 출력은 12열이다. `persistence_27d`의 `idx+(h+1)-lag_steps` horizon offset과 범위 밖 NaN 처리는 분석용 균일 시계열이라는 전제에서는 맞고, 제출 불가 주석도 있다.
- **힐클라이밍 본체**: 중복 선택으로 정수 가중 평균을 만드는 갱신식과 반환 가중치는 맞다. 잘못된 결과는 알고리즘 자체가 아니라 위에서 지적한 `score`·NaN 검증을 그대로 호출하는 데서 생긴다.
- **규정 충돌 차단**: pretrained weight 금지, test 시계열 복원 금지, **pseudo-labeling 은 [공지] 「추정」으로 NO-GO**(회색 아님), validation target의 최종 학습 사용 금지, 외부 데이터 금지는 RULES와 SUN_PLAN의 핵심 실행 경로에서 일치한다. PLAYBOOK의 일반적인 pretrained 권고는 SUN_PLAN에서 명시적으로 제외돼 있어 별도 finding으로 잡지 않았다. **제출본 선택은 Public 최고 자동**(2026-08-21 사람 확인).
- **도구 실행**: `factcheck.py --selftest`, `conform.py --selftest`는 통과했고 현재 `conform.py` 전체 스캔은 exit 0이었다. 다만 그 PASS가 놓치는 반례는 위 finding에서 별도로 재현했다. 현재 Python에는 NumPy가 없어 `python tools/kit.py`의 demo 자체는 실행하지 못했고, kit 검증은 코드 추적과 독립 산술 반례로 수행했다.
