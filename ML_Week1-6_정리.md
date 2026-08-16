# 기계학습 개론 (KAIST, Il-Chul Moon) — Week 1~6 정리

원본: `Week 1.pdf` ~ `Week 6.pdf` (Copyright © 2010 Il-Chul Moon, Dept. of Industrial and Systems Engineering, KAIST)
슬라이드 수: W1=37, W2=32, W3=26, W4=27, W5=33, W6=28 (총 183장)

## 전체 흐름 한눈에

| 주차 | 제목 | 핵심 |
|---|---|---|
| 1 | Motivation and Basics | ML 동기, 압정 이야기, MLE/MAP, PAC, 확률·분포 기초 |
| 2 | Fundamentals of Machine Learning | Rule based(Find-S, Version Space, Candidate Elimination), Decision Tree(Entropy/IG/ID3), Linear Regression |
| 3 | Naïve Bayes Classifier | Optimal classification, Bayes risk, decision boundary, 조건부 독립, NB, bag-of-words |
| 4 | Logistic Regression | Logistic/logit function, MCLE, Taylor expansion, gradient descent/ascent, NB vs LR (generative-discriminative) |
| 5 | Support Vector Machine | Max margin, hard/soft margin, hinge loss, Lagrange 쌍대, KKT, kernel trick |
| 6 | Training/Testing and Regularization | Bias-variance 분해, Occam's razor, cross validation, 성능지표, L1/L2 regularization |

**전 주차 관통 주제**: "머신러닝 = 함수 근사(function approximation)" → 함수 유형을 정하고 **파라미터 θ를 찾는 문제**로 환원 → θ를 찾는 도구가 MLE/MAP → 닫힌 해가 없으면 gradient method → 과적합을 막기 위해 regularization

> 표기: 슬라이드 번호는 `[p.N]`. 원문 오탈자/오류는 각 주차 말미 **검토 노트**에 별도 기재(본문은 원문 그대로 유지 + 정정 병기).

---

# Week 1 — Motivation and Basics

## 1. 주차 목표 [p.2]

- Machine learning, AI, Datamining 분야에 대한 동기 부여: Why? What? 분야 개관
- 하나의 이야기(story)를 통한 짧은 문답
  - What consists of machine learning?
  - MLE
  - MAP
- 기초(Some basics): Probability, Distribution, and some rules

## 2. Motivation

### 2.1 Keywords [p.4]

- 떠도는 키워드가 많음: Data-mining, Knowledge discovery, Machine Learning, Artificial Intelligence…
- 키워드는 영역(territory), 관점(perspectives), 문제 유형(types of problems), 연구자(researchers) 등에서 유래
- 라벨링이 아니라 **실체(substance)** 에 집중한다
- 본 강의에서는 그냥 **"Machine Learning"** 이라 부른다 (뭐라 부르든 상관없음)
- 관련 학문 영역: Statistics, AI in CS, Database in CS, Management, Industrial Engineering, …

### 2.2 Abundance of Data [p.5]

데이터는 어디서나 수집되고 있음:

- Text Data (News Articles, 10K Rep., NYT, WSJ, 아고라, Daum)
- Network Data, Machine Logs, Social Networks / Social Media (Facebook, Twitter, Blogosphere)
- Image Data, Surveillance Data
- Trajectory Data, Vehicle Trajectory, Time Series Data
- Disease Outbreak Data
- Purchase + Review Data (Amazon)

### 2.3 Machine Learning Applications [p.6]

| 응용 | 설명 |
|---|---|
| Document Classification | 문서 → Sports / Science / News |
| Stock Market Prediction | X = Feb01 → Y = ? (DJIA 예측) |
| Plate Num. Recognition | 차량 번호판 인식 |
| SNS Recommendation | People You May Know, Recommended Pages |
| Helicopter Control | 헬리콥터 제어 |

### 2.4 Spam Filtering and more [p.7]

- Spam filter, 그 이상: **Importance vs. Urgency**
- 중요한 이메일을 어떻게 예측? → Social networks, Contents
- Features: 이메일의 from, to, cc, title, body text 토큰 → v-dimensional **basic feature (BF)** sub-vector
- 표현 스킴(BF, BF+NC, BF+SI, BF+SIP, BF+SI+NC, BF+SI+NC+SIP)별 SVM 평가(Macro-averaged MAE, p-value)
- Clusters(Newman Cluster) 도 머신러닝 기법인가? 라는 질문
- 출처: Shinjae Yoo, Yiming Yang, Frank Lin, and Il-Chul Moon, *Mining Social Networks for Personalized Email Prioritization*, ACM SIGKDD Conference, Paris, France, Jun 28, 2009

### 2.5 Opinion Mining and more [p.8]

- 출처: Il-Chul Moon, Alice H. Oh, and Kathleen M. Carley, *Analyzing Social Media in Escalating Crisis Situations*, IEEE Conference on Intelligence and Security Informatics (ISI 2011), pp. 71-76, Beijing, China, Jul 10-12, 2011
- 사례: PCC 천안함 침몰 (2010-03-26)
- 목표 — 집단의 합의(consensus) 찾기
  - **Mining population's perception of the event**
  - Mining key opinion buried in a data chunk
  - Estimating future polarity of the population
  - **Strategy to maintain the unity of the population**
- Observable System(텍스트 수/사용자 수 시계열) ↔ Implicit System(Author-to-Author Virtual Influence Network, Topic Evolution Network)
- SNS와 텍스트에 대한 Data-mining

### 2.6 Stock Market Prediction and more [p.9]

- 출처: W.S. Lee, J.Y. Bak, A.H. Oh, and I.-C. Moon. *Media Analysis to Estimate Stock Market Movement*. Information Fusion. Major Revision
- 파이프라인:
  - Economic Indicators → Indicators
  - Bloomberg / Dealbreaker → Word filtering & counting(Word count), Latent Dirichlet Allocation(Topic weight), Aspect Sentiment Unification Model(Senti-topic weight)
  - DOW/NASDAQ → 3 trading days, +1 day price (종속변수)
  - Sliding window approach (Averaged of *k* lag days → Prediction for target day)
  - Support Vector Regression → output → MAPE
- 예측 계수가 높은 토픽: TopicWeight 26 (-0.609), TopicWeight 1 (0.520), TopicWeight 7 (0.508)
- **"swiss"와 DJIA 간 강한 음의 상관**
- 이벤트 구간: US Credit Downgrade, QE2, Europe Debt Crisis

## 3. Types of Machine Learning [p.10]

```
                Machine Learning
      ┌──────────────┬──────────────────┬────── …
Supervised      Unsupervised      Reinforcement
 Learning         Learning          Learning
```

| 유형 | 정의 |
|---|---|
| Supervised Learning | **You** know the true answers of some of instances |
| Unsupervised Learning | **You** do not know the true answers of instances |
| Reinforcement Learning | **You** do know the objective, but you do not know how to achieve |

- "You"가 될 수 있는 주체: Machine learning(연구자), Dataset provider, Machine learning users 등
- 교수마다 다양한 분류가 존재(목적, 데이터 타입 등). 다른 학습 분류법도 존재
- 본 강의는 Supervised / Unsupervised 에 초점(빨간 박스)

### 3.1 Supervised Learning [p.11]

- 참값을 알고, 참값의 예시를 제공할 수 있음
- 사례: Spam filtering, Automatic grading, Automatic categorization
- Classification 또는 Regression의 형태
  - Hit or Miss: 질병 유/무
  - Ranking: A+, B, C, F 중 하나
  - Types: 기사가 긍정/부정
  - Value prediction: 이 물건 가격은 X
- 방법론
  - **Classification**: 관측치로부터 이산(discrete) 종속값 추정
  - **Regression**: 관측치로부터 (연속) 종속값 추정

### 3.2 Unsupervised Learning [p.12]

- 참값을 모르고, 참값 예시를 제공할 수 없음
- 사례: Discovering clusters, Discovering latent factors, Discovering graph structures
- Clustering / filtering / completing
  - 텍스트에서 **대표 토픽 단어** 찾기
  - 얼굴 데이터에서 **잠재 이미지** 찾기
  - 제품-리뷰 점수의 **불완전 행렬 완성**
  - 궤적 데이터에서 **노이즈 필터링**
- 방법론
  - **Clustering**: 인스턴스의 집합과 소속(affiliation) 추정
  - **Filtering**: 신호+노이즈 혼합에서 근본 신호 추정

## 4. Warming Up: A Short Episode

### 4.1 Thumbtack Question [p.14]

- 압정(thumbtack) 던지기 도박 사이트: 못이 위(nail's up)에 걸고 맞히면 2배, 아래(nail's down)도 동일
- 억만장자가 과학적·공학적 지원을 받아 참가하려 함(당신에게 큰 돈 지불)
- 질문: "압정을 던지면 못이 위로 떨어질 확률은?"

### 4.2 Experience from trials [p.15]

- 나의 답: "몇 번 던져 보세요"
- 억만장자가 5번 시도 → nail's up 3/5
- 나의 답: nail's up에 3/5, nail's down에 2/5 투자
- 억만장자: "왜?"

### 4.3 Binomial Distribution [p.16]

- 이항분포는 **이산확률분포**
  - n번의 독립적인 yes/no 실험에서 성공 횟수의 분포, 각 성공확률은 **θ**
  - Bernoulli experiment 라고도 함
- Flips are i.i.d.
  - Independent events
  - Identically distributed according to binomial distribution
- $P(H)=\theta,\ P(T)=1-\theta$
- $P(HHTHT)=\theta\theta(1-\theta)\theta(1-\theta)=\theta^3(1-\theta)^2$
- D = Data = H,H,T,H,T → n=5, k=a_H=3, p=θ
- $P(D|\theta)=\theta^{a_H}(1-\theta)^{a_T}$
- $f(k;n,p)=P(K=k)=\binom{n}{k}p^k(1-p)^{n-k},\quad \binom{n}{k}=\frac{n!}{k!(n-k)!}$
  - n, p는 파라미터로 주어지고 k를 바꿔가며 값 계산
  - 이항계수는 **순서를 무관(order insensitive)하게** 만듦

### 4.4 Maximum Likelihood Estimation [p.17]

- $P(D|\theta)=\theta^{a_H}(1-\theta)^{a_T}$
- Data: a_H, a_T를 가진 시퀀스 D를 관측
- 가설(hypothesis): 압정 도박 결과는 θ의 이항분포를 따른다
- 가설을 강하게 만드는 방법
  - 관측에 대한 더 나은 분포 찾기 (가능하지만 더 많은 근거 필요)
  - **θ의 최적 후보** 찾기 → θ를 가장 그럴듯하게 만드는 조건은?
- 한 가지 후보가 **MLE**: 관측 데이터의 확률을 최대화하는 θ 선택
  $$\hat{\theta}=\arg\max_\theta P(D|\theta)$$

### 4.5 MLE Calculation [p.18]

$$\hat{\theta}=\arg\max_\theta P(D|\theta)=\arg\max_\theta \theta^{a_H}(1-\theta)^{a_T}$$

이대로는 진행이 어려우므로 **log 함수** 트릭 사용:

$$\hat{\theta}=\arg\max_\theta \ln P(D|\theta)=\arg\max_\theta \ln\{\theta^{a_H}(1-\theta)^{a_T}\}=\arg\max_\theta\{a_H\ln\theta+a_T\ln(1-\theta)\}$$

최대화 문제이므로 미분 = 0:

$$\frac{d}{d\theta}(a_H\ln\theta+a_T\ln(1-\theta))=0$$
$$\frac{a_H}{\theta}-\frac{a_T}{1-\theta}=0$$
$$\theta=\frac{a_H}{a_T+a_H}$$

$$\boxed{\hat{\theta}=\frac{a_H}{a_H+a_T}}$$

### 4.6 Number of Trials [p.19]

- 시행 관측 + MLE 관점 + 이항분포 가정 → **θ = 0.6**
- 따라서 head에 걸면 이길 확률이 더 높음
- 억만장자: "계산하는 동안 더 던져봤는데 30 heads, 20 tails다. 뭔가 달라지나?"
- 나: "아니오, 그대로 0.6"
- 억만장자: "그럼 시간 낭비였나?"
- 나: "아니오, 추가 시행은 …에 가치가 있습니다"

### 4.7 Simple Error Bound [p.20]

- 추가 시행은 **추정 오차를 줄인다**
- 현재 $\hat\theta=\frac{a_H}{a_H+a_T}$, $N=a_H+a_T$
- $\theta^*$를 압정 던지기의 참 파라미터, 임의의 오차 $\varepsilon>0$
- **Hoeffding's inequality** 에 의한 단순 상한:
  $$P(|\hat\theta-\theta^*|\ge\varepsilon)\le 2e^{-2N\varepsilon^2}$$
  (수학과 친구에게서 얻어온 식)
- 억만장자: "필요한 시행 횟수 N을 계산해 줄 수 있나? ε=0.1, 0.01% case로"
- 교수님 등장: 이것이 **Probably Approximate Correct (PAC) learning**
  - Probably? (0.01% case)
  - Approximately? (ε = 0.1)

### 4.8 Probably Approximately Correct Learning [p.21]

- ML 커뮤니티의 PAC learning theory에서, 학습자(learner)는
  - 표본 데이터를 받고
  - 가능한 함수 클래스 중에서 일반화 함수(가설)를 선택
  - 선택된 함수가 **낮은 일반화 오차**를 **높은 확률로** 갖도록
- 머신러닝의 정의 — 다양한 관점 존재. **Tom Mitchell**:
  > A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E.
- 정리 박스: **Machine Learning = 데이터셋이 주어졌을 때, 가장 확률 높고(most probable) 가장 근사한(most approximate) 답을 주는 일반화 함수를 선택하는 것**
- 질문: PAC Learning은 machine learning인가?

### 4.9 Wait!!! — Bayes 등장 [p.22]

Bayes라는 학생이 손을 들었다.

### 4.10 Incorporating Prior Knowledge [p.23]

- Bayes: "압정이 정말 head 60%인가? 50 vs 50 아닌가?"
- 억만장자: "그렇게 생각했지만, 이전 지식을 내 시행에 어떻게 합치나?"
- Bayes: **정리(theorem)** 제공
  $$P(\theta|D)=\frac{P(D|\theta)P(\theta)}{P(D)}$$
  $$Posterior=\frac{Likelihood \times Prior\ Knowledge}{Normalizing\ Constant}$$
- 이미 다룬 $P(D|\theta)=\theta^{a_H}(1-\theta)^{a_T}$
- $P(\theta)$가 prior knowledge 부분
- 나: "그럼 $P(\theta|D)$는 데이터와 사전지식이 함께 영향을 준 결론인가?"
- Bayes: "그렇고, 그것이 우리의 **미래의 사전지식**이 된다!"

### 4.11 More Formula from Bayes Viewpoint [p.24]

- $P(\theta|D)\propto P(D|\theta)P(\theta)$
  - $P(D|\theta)=\theta^{a_H}(1-\theta)^{a_T}$
  - $P(\theta)=????$
- 사전지식을 잘 표현해야 곱이 매끄럽고 수식이 복잡해지지 않음
- Bayes: **Beta distribution** 쓰면 어떤가?
  $$P(\theta)=\frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha,\beta)},\quad B(\alpha,\beta)=\frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)},\quad \Gamma(\alpha)=(\alpha-1)!$$
  (범위 [0,1]에 잘 맞음)
- 결과:
  $$P(\theta|D)\propto P(D|\theta)P(\theta)\propto \theta^{a_H}(1-\theta)^{a_T}\theta^{\alpha-1}(1-\theta)^{\beta-1}=\theta^{a_H+\alpha-1}(1-\theta)^{a_T+\beta-1}$$
- 위 식에서 흥미로운 사실 하나를 알게 됨…

### 4.12 Maximum a Posteriori Estimation [p.25]

- 억만장자: "나는 가장 확률 높고 근사한 θ를 원한다"
- MLE에서는 $\hat\theta=\arg\max_\theta P(D|\theta)$ 로부터
  - $P(D|\theta)=\theta^{a_H}(1-\theta)^{a_T}$
  - $\hat\theta=\frac{a_H}{a_H+a_T}$
- MAP에서는 $\hat\theta=\arg\max_\theta P(\theta|D)$ 로부터
  - $P(\theta|D)\propto\theta^{a_H+\alpha-1}(1-\theta)^{a_T+\beta-1}$
  - $$\hat\theta=\frac{a_H+\alpha-1}{a_H+\alpha+a_T+\beta-2}$$
- 어차피 최대화이므로 계산 방식은 동일

### 4.13 Conclusion from Anecdote [p.26]

| MLE | MAP |
|---|---|
| $\hat\theta=\dfrac{a_H}{a_H+a_T}$ | $\hat\theta=\dfrac{a_H+\alpha-1}{a_H+\alpha+a_T+\beta-2}$ |

- 억만장자: "누가 맞나? 숫자가 다르다!"
- Bayes: "게임을 반복할 충분한 돈을 준다면 사실상 같다"
- 나: "$a_H$, $a_T$가 커지면 $\alpha,\beta$는 무의미해진다"
- 억만장자: "그래도 시행을 더 못 주면 α, β가 중요하다. 누가 α, β를 정하나?"
- Bayes와 나: "글쎄… 대학원생?"

## 5. Basics

### 5.1 What we just saw is… [p.28]

- 하나의 **투쟁(struggle)**
  - 억만장자: 거대한 가능성 중 작은 데이터셋을 분석해 돈을 벌기
  - 나: 작은 데이터셋으로 가장 확률 높고 근사한 답 주기
  - Bayes: 사전지식을 답에 반영할 수 있음을 설득
- 결국: 압정 게임의 본질 찾기 → 핵심은 head/tail 결과의 확률
- 문제 해결의 기반 지식: **Probability, Distribution, Some mathematical tricks**
- (Bayes의 "Beta distribution 쓰자"는 확률·분포·통계 지식에서 나온 것)

### 5.2 Probability [p.29]

- 철학적으로 둘 중 하나
  - **Objectivists**: 사건의 상태를 기술하기 위해 숫자를 부여 (counting)
  - **Subjectivists**: 자신의 믿음으로 사건에 숫자 부여 (betting)
- 수학적으로: 아래 성질을 갖는 함수

$$P(E)\in\mathbb{R},\quad P(E)\ge 0,\quad \forall E\in F,\quad P(\Omega)=1$$
$$P(E_1\cup E_2\cup\cdots)=\sum_{i=1}^{\infty}P(E_i)\ \text{(mutually exclusive 수열일 때)}$$

- 파생 성질

$$A\subseteq B \Rightarrow P(A)\le P(B),\quad P(\emptyset)=0,\quad 0\le P(E)\le 1$$
$$P(A\cup B)=P(A)+P(B)-P(A\cap B),\quad P(E^C)=1-P(E)$$

### 5.3 Conditional Probability [p.30]

- 우리는 보통 전체 표본공간 Ω를 다루지 않고, 항상 조건을 건다
  - 파라미터가 X, Y, Z…라 가정
  - 사건이 X, Y, Z…의 범위(scope) 안에 있다고 가정
- $$P(A|B)=\frac{P(A\cap B)}{P(B)}$$ (B가 주어졌을 때 A의 조건부 확률)
- 유용한 공식
  $$P(B|A)=\frac{P(A|B)P(B)}{P(A)}$$ ← 조건과 대상 사건을 뒤집을 수 있음
  $$P(A)=\sum_n P(A|B_n)P(B_n)$$ ← 조건부 확률과 prior들을 모두 더해 대상 사건을 복원
- $Posterior=\frac{Likelihood\times Prior\ Knowledge}{Normalizing\ Constant}$ 와 연결됨

### 5.4 Probability Distribution [p.31]

- 확률분포는 확률 시행/실험/설문의 잠재적 사건의 부분집합에 확률을 부여
- 사건을 확률로 매핑하는 함수
  - 확률이라 부르므로 확률의 성질(공리)을 지켜야 함
  - 사건은 연속 수치값일 수도, 이산 범주값일 수도 있음
- 예: $f(x)=\dfrac{e^{-\frac{1}{2}x^2}}{\sqrt{2\pi}}$
  - f: probability distribution function, x: 연속값, f(x): 부여된 확률
- **PDF** = $f(x)$, **CDF** = $\int_{-\infty}^{x}f(x)\,dx$

### 5.5 Normal Distribution [p.32]

- 매우 흔히 관측되는 분포, 연속 수치값
- $$f(x;\mu,\sigma)=\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$
- Notation: $N(\mu,\sigma^2)$ / Mean: $\mu$ / Variance: $\sigma^2$

### 5.6 Beta Distribution [p.33]

- 닫힌 구간 [0,1]을 지지(support), 연속 수치값 → 확률의 성질과 잘 맞음
- $$f(\theta;\alpha,\beta)=\frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha,\beta)},\quad B(\alpha,\beta)=\frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)},\quad \Gamma(\alpha)=(\alpha-1)!$$
- Notation: $Beta(\alpha,\beta)$
- Mean: $\dfrac{\alpha}{\alpha+\beta}$ / Variance: $\dfrac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$

### 5.7 Binomial Distribution [p.34]

- 이산값에 대한 가장 단순한 분포: Bernoulli trial(yes/no), 0 or 1, Selection, switch…
- $$f(\theta;n,p)=\binom{n}{k}p^k(1-p)^{n-k},\quad \binom{n}{k}=\frac{n!}{k!(n-k)!}$$
- Notation: $B(n,p)$ / Mean: $np$ / Variance: $np(1-p)$

### 5.8 Multinomial Distribution [p.35]

- 이항분포의 일반화: yes/no를 넘어 A, B, C, D, E, …, Z 선택 (단어 선택, 클러스터 선택 등)
- $$f(x_1,\dots,x_k;n,p_1,\dots,p_k)=\frac{n!}{x_1!\cdots x_k!}p_1^{x_1}\cdots p_k^{x_k}$$
- Notation: $Mult(P),\ P=\langle p_1,\dots,p_k\rangle$
- Mean: $E(x_i)=np_i$ / Variance: $Var(x_i)=np_i(1-p_i)$

## 6. Acknowledgement & Further Readings [p.36–37]

- 영향: Prof. Carlos Guestrin (CMU), Prof. Eric P. Xing (CMU) — 일부 이미지도 두 교수의 강의자료에서 인용
- Further readings: **Bishop Chapter 1 and 2**

## Week 1 검토 노트

| 위치 | 내용 | 판정 |
|---|---|---|
| p.18 | $\theta=\frac{a_H}{a_T+a_H}$ 와 최종 $\hat\theta=\frac{a_H}{a_H+a_T}$ | 동일 식(분모 순서만 다름). 문제 없음 |
| p.34 | $f(\theta;n,p)$ 로 표기했으나 우변은 $k$의 함수 | 원문 표기 오류. $f(k;n,p)$가 맞음 (p.16에서는 올바르게 표기) |
| p.33 | $\Gamma(\alpha)=(\alpha-1)!$ | 정수 α에 한해 성립. 일반 실수 α에서는 $\Gamma(\alpha)=\int_0^\infty t^{\alpha-1}e^{-t}dt$ |
| p.20 | Hoeffding bound $2e^{-2N\varepsilon^2}$ | 표준식과 일치 |
| p.16 | "Also called a Bernoulli experiment" | 엄밀히는 n회 Bernoulli trial의 합이 이항분포. 표현상 축약 |

---

# Week 2 — Fundamentals of Machine Learning

## 1. 주차 목표 [p.2]

가장 고전적인 머신러닝 방법 학습:

- Rule based approach / Classical statistics approach / Information theory approach
- **Rule based machine learning**: 특수화·일반화된 규칙을 찾는 법, 규칙이 쉽게 깨지는 이유
- **Decision Tree**: 학습 데이터로 트리 생성하는 법, 새 데이터에서 약한 학습기가 되는 이유
- **Linear Regression**: 학습 데이터로 파라미터 집합 추론하는 법, feature engineering의 한계

## 2. Rule Based Machine Learning

### 2.1 From the Last Week [p.4]

- ML 정의(Tom Mitchell) 재확인
- More experience → 더 많은 압정 던지기, 더 많은 사전지식
- Data: a_H, a_T를 가진 시퀀스 D 관측 / 가설: 결과가 θ의 이항분포를 따름
- 압정 외 첫 시도: **Rule based learning** — 여전히 더 나은 가설을 고르는 문제

### 2.2 A Perfect World for Rule Based Learning [p.5]

완벽한 세계의 가정:

| 가정 | 의미 |
|---|---|
| No observation errors, No inconsistent observations | Training data is error-free, noise-free |
| No stochastic elements in the system we observe | Target function is deterministic |
| Full information in the observations to regenerate the system | Target function is contained in hypotheses set |

**EnjoySport 데이터셋** — 관측: Sky, Temp, Humid, Wind, Water, Forecast → EnjoySport

| Sky | Temp | Humid | Wind | Water | Forecst | EnjoySpt |
|---|---|---|---|---|---|---|
| Sunny | Warm | Normal | Strong | Warm | Same | Yes |
| Sunny | Warm | High | Strong | Warm | Same | Yes |
| Rainy | Cold | High | Strong | Warm | Change | No |
| Sunny | Warm | High | Strong | Cool | Change | Yes |

### 2.3 Function Approximation [p.6]

- Machine Learning = 더 나은 근사 함수를 만드는 노력 (PAC Learning Theory 상기)
- EnjoySport의 완벽 세계에서
  - **Instance X**: Features O: `<Sunny, Warm, Normal, Strong, Warm, Same>`, Label Y: `<Yes>`
  - **Training Dataset D**: 인스턴스에 대한 관측 모음
  - **Hypotheses H**: X를 Y로 바꾸는 잠재적 가능 함수. 예) $h_i$: `<Sunny, Warm, ?, ?, ?, Same>` → Yes. 몇 개의 가설이 존재하는가?
  - **Target Function c**: features와 label 사이의 미지의 목표 함수
- 목표 전이: **모든 x∈X에 대해 h(x)=c(x)인 h∈H를 결정** → **모든 x∈D에 대해 h(x)=c(x)인 h∈H를 결정**

### 2.4 Graphical Representation of Function Approximation [p.7]

- Instances X 와 Hypotheses H 를 대응시켜 표현. H는 General(아래) ↔ Specific(위) 축을 가짐
- $x_1$: `<Sunny, Warm, Normal, Strong, Warm, Same>`
- $x_2$: `<Sunny, Warm, Normal, Light, Warm, Same>`
- $x_3$: `<Sunny, Warm, Normal, Strong, Warm, Change>`
- $h_1$: `<Sunny, ?, ?, ?, Warm, ?>`
- $h_2$: `<Sunny, ?, ?, ?, Warm, Same>`
- $h_3$: `<Sunny, ?, ?, Strong, Warm, ?>`
- 더 나은 함수 근사는? → **Generalization vs. Specialization**

### 2.5 Find-S Algorithm [p.8]

```
Find-S Algorithm
  Initialize h to the most specific in H
  For instance x in D
      if x is positive
          For feature f in O
              If f_i in h == f_i in x
                  Do nothing
              Else
                  f_i in h = f_i in h ∪ f_i in x
  Return h
```

- Instances
  - $x_1$: `<Sunny, Warm, Normal, Strong, Warm, Same>`
  - $x_2$: `<Sunny, Warm, Normal, Light, Warm, Same>`
  - $x_4$: `<Sunny, Warm, Normal, Strong, Warm, Change>`
- Hypotheses 진행
  - $h_0$ = `<∅, ∅, ∅, ∅, ∅, ∅>`
  - $h_1$ = `<Sunny, Warm, Normal, Strong, Warm, Same>`
  - $h_{1,2,3}$ = `<Sunny, Warm, Normal, ?, Warm, Same>`
  - $h_{1,2,3,4}$ = `<Sunny, Warm, Normal, ?, Warm, ?>`
- 문제점: 가능한 h가 많고 **수렴을 결정할 수 없음**

### 2.6 Version Space [p.9]

- 가능한 가설이 많고 수렴을 찾을 방법이 없음 → 가능 가설의 **경계(perimeter)** 설정 필요
- 가능한 가설의 집합 = **Version Space, VS**
  - **General Boundary, G**: version space의 최대 일반 가설 집합
  - **Specific Boundary, S**: version space의 최대 특수 가설 집합
- 모든 가설 h는 다음을 만족:
  $$VS_{H,D}=\{h\in H\mid \exists s\in S, \exists g\in G,\ g\ge h\ge s\}$$
  여기서 $x\ge y$ 는 x가 y보다 더 일반적이거나 같음을 의미
- EnjoySport 결과
  - S: `{<Sunny, Warm, ?, Strong, ?, ?>}`
  - 중간: `<Sunny,?,?,Strong,?,?>`, `<Sunny,Warm,?,?,?,?>`, `<?,Warm,?,Strong,?,?>`
  - G: `{<Sunny,?,?,?,?,?>, <?,Warm,?,?,?,?>}`

### 2.7 Candidate Elimination Algorithm [p.10]

```
Candidate Elimination Algorithm
  Initialize S to maximally specific h in H     S0: {<∅,∅,∅,∅,∅,∅>}
  Initialize G to maximally general h in H      G0: {<?,?,?,?,?,?>}
  For instance x in D
      If y of x is positive
          Generalize S as much as needed to cover o in x
          Remove any h in G, for which h(o)≠y
      If y of x is negative
          Specialize G as much as needed to exclude o in x
          Remove any h in S, for which h(o)=y
  Generate h that satisfies ∃s∈S, ∃g∈G, g ≥ h ≥ s
```

### 2.8 Progress of Candidate Elimination Algorithm [p.11–13]

**1~2번째 인스턴스 (Yes, Yes) 처리 후** [p.11]

- S0: `{<∅,∅,∅,∅,∅,∅>}`
- S1: `{<Sunny, Warm, Normal, Strong, Warm, Same>}`
- S2: `{<Sunny, Warm, ?, Strong, Warm, Same>}`
- G0, G1, G2: `{<?,?,?,?,?,?>}`

**3번째 인스턴스 (Rainy, Cold, High, Strong, Warm, Change → No) 처리 후** [p.12]

- S2, S3: `{<Sunny, Warm, ?, Strong, Warm, Same>}`
- G3: `{<Sunny,?,?,?,?,?>, <?,Warm,?,?,?,?>, <?,?,?,?,?,Same>}`
- (G0, G1, G2: `{<?,?,?,?,?,?>}` 로부터 특수화)

**4번째 인스턴스 (Sunny, Warm, High, Strong, Cool, Change → Yes) 처리 후** [p.13]

- S4: `{<Sunny, Warm, ?, Strong, ?, ?>}`
- G4: `{<Sunny,?,?,?,?,?>, <?,Warm,?,?,?,?>}`
- S와 G 사이에 **여전히 많은 h**가 존재

### 2.9 How to classify the next instance? [p.14]

- Version space = D를 만족하는 H의 부분집합
  - S: `{<Sunny, Warm, ?, Strong, ?, ?>}`
  - $h_1$ = `<Sunny, ?,?,Strong,?,?>`, $h_2$ = `<Sunny, Warm,?,?,?,?>`, $h_3$ = `<?, Warm,?,Strong,?,?>`
  - G: `{<Sunny, ?,?,?,?,?>, <?,Warm, ?, ?, ?, ?>}`
- 새 인스턴스가 들어오면?
  - `<Sunny, Warm, Normal, Strong, Cool, Change>`
  - `<Rainy, Cold, Normal, Light, Warm, Same>`
  - `<Sunny, Warm, Normal, Light, Warm, Same>`
- 어떻게 분류? 부분집합에서 **어떤 h를 적용?** 아니면 **모든 h로 분류?** 만족되는 h는 몇 개?

### 2.10 Is this working? [p.15]

- candidate-elimination이 올바른 가설로 수렴하는가?
  - Converge? → 가설을 선택할 수 있는가
  - Correct? → 그 가설이 관측 시스템에서 참인가
- **가정이 주어지면 yes and yes** (error-free/noise-free, deterministic target, target ∈ H)
- 그러나 우리는 완벽한 세계에 살지 않음
  - D의 x의 o에 노이즈 존재
  - x의 o 이외의 결정 요인 존재
  - → **올바른 h가 노이즈에 의해 제거될 수 있음** → **yes and no 라고 말할 수 없음**

## 3. Decision Tree

### 3.1 Because we live with noises… [p.17]

- 더 나은 학습 방법 필요: 노이즈에 강건(robust), 가설의 더 간결한(concise) 표현
- 대안: **decision tree**
- 예) 가설 `<Sunny, ?,?,?,?,?>` ↔ Sky(Rainy→No / Sunny→Yes) 트리
- 예) 가설 `<Sunny, Warm, ?, Strong, ?, ?>` ↔ Sky→Temp→Wind 트리 (해당 가설에 대응하는 **유일한** 잠재 decision tree)

### 3.2 Credit Approval Dataset [p.18]

- http://archive.ics.uci.edu/ml/datasets/Credit+Approval
- 기밀 보호를 위해 데이터셋 익명화(피처 이름과 값 모두)
- 속성: A1: b, a / A2: continuous / A3: continuous / A4: u, y, l, t / A5: g, p, gg / A6: c, d, cc, i, j, k, m, r, q, w, x, e, aa, ff / A7: v, h, bb, j, n, z, dd, ff, o / A8: continuous / A9: t, f / A10: t, f / A11: continuous / A12: t, f / A13: g, p, s / A14: continuous / A15: continuous / **C: +,- (class attribute)**

**Counting Result**

- 690 instances total, 307 positive instances (→ 383 negative)
- A1: a일 때 98 positive / 112 negative, b일 때 206 positive / 262 negative, ?일 때 3 positive / 9 negative
- A9: t일 때 284 positive / 77 negative, f일 때 23 positive / 306 negative

```
A1 (307+,383-)          A9 (307+,383-)
 a→98+,112-              t→284+,77-
 b→206+,262-             f→23+,306-
 ?→3+,9-
```

- 어떤 속성을 가설의 피처 집합에 넣는 것이 더 나은가?

### 3.3 Entropy [p.19]

- 더 나은 확인 속성 = **불확실성을 가장 많이 줄이는** 속성
- 그럼 피처 변수의 불확실성을 어떻게 측정?
- 확률변수의 **엔트로피** (피처는 확률변수, 엔트로피가 클수록 불확실성 큼)
  $$H(X)=-\sum_X P(X=x)\log_b P(X=x)$$
  - $P(X=1)=0.5$ 일 때 최대(most random case), $P(X=1)=0$ 또는 $1$ 일 때 0
- **Conditional Entropy**: 피처 변수가 주어졌을 때 클래스의 엔트로피
  $$H(Y|X)=\sum_X P(X=x)H(Y|X=x)=\sum_X P(X=x)\Big\{-\sum_Y P(Y=y|X=x)\log_b P(Y=y|X=x)\Big\}$$

### 3.4 Information Gain [p.20]

- Entropy Before Decision Node ↔ Entropy After Decision Node
- $H(Y)=-\sum_{Y\in\{+,-\}}P(Y=y)\log_2 P(Y=y)$
- $H(Y|A1)=\sum_{X\in\{a,b,?\}}\sum_{Y\in\{+,-\}}P(A1=x,Y=y)\log_2\frac{P(A1=x)}{P(A1=x,Y=y)}$
- $H(Y|A9)=\sum_{X\in\{t,f\}}\sum_{Y\in\{+,-\}}P(A9=x,Y=y)\log_2\frac{P(A9=x)}{P(A9=x,Y=y)}$
- 전후 차이:
  $$IG(Y,A_i)=H(Y)-H(Y|A_i)$$
- 승자는? → (A9)

### 3.5 Top-Down Induction Algorithm [p.21]

- decision tree 학습에는 많은 변형이 존재: **ID3, C4.5, CART**…
- 한 예: **ID3 algorithm**

```
ID3 algorithm
  Create an initial open node
  Put instances in the initial node
  Repeat until no open node
      Select an open node to split
      Select a best variable to split
      For values of the selected variable
          Sort instances with the value of the selected variable
          Put the sorted items under the branch of the value of the variable
          If the sorted items are all in one class
              Close the leaf node of the branch
```

**A1과 A9만 사용한 트리**

```
A9 (307+,383-)
├─ t → A1 (284+,77-)
│       ├─ a → 93+,21-   → Class+
│       ├─ b → 189+,56-  → Class+
│       └─ ? → 2+,0-     → Class+
└─ f → A1 (23+,306-)
        ├─ a → 5+,91-    → Class-
        ├─ b → 17+,206-  → Class-
        └─ ? → 1+,9-     → Class-
```

- 오분류: 21+56+0+5+17+1 = **100 instances** → **85.5% Accuracy**

### 3.6 If you want more…. [p.22]

A10이 그 다음으로 information gain이 높다고 가정하면:

```
A9 (307+,383-)
├─ t → A1 (284+,77-)
│       ├─ a → A10 (93+,21-)
│       │       ├─ t → 69+,7-   → Class+
│       │       └─ f → A12 (24+,14-)
│       │               ├─ t → 14+,5-  → Class+
│       │               └─ f → A5 (10+,9-)
│       │                       ├─ gg → 0+,0-
│       │                       ├─ g  → 9+,7-  → Class+
│       │                       └─ p  → 1+,2-  → Class-
│       ├─ b → 189+,56-  → Class+
│       └─ ? → 2+,0-     → Class+
└─ f → A1 (23+,306-)
        ├─ a → 5+,91-    → Class-
        ├─ b → 17+,206-  → Class-
        └─ ? → 1+,9-     → Class-
```

- 오분류: 5+17+1+56+7+5+7+1 = **99 instances** → **85.6% Accuracy**

### 3.7 Problem of Decision Tree [p.23]

- 주어진 데이터셋에서는 더 잘함 — 그러나 **주어진 경험(=Training dataset)에서만**
- 만든 트리를 현장에 배포하면?
  - 세상에는 노이즈와 비일관성이 매우 많음
  - Training dataset은 실세계의 완벽한 표본이 아님 (Noise, Inconsistencies)
- 전형적 결과: 트리 크기(노드 수)가 커질수록 training accuracy는 상승하나 test accuracy는 하락 → **"Should have stopped here!"**
- 언제 멈출지 아는 것은 어려운 과제. 어떻게?
  - Pruning by divided dataset?
  - Path length penalty?

### 3.8 Why we are not interested in these? [p.24]

- Rule based ML 알고리즘의 장점
  - 구현이 쉬움
  - 해석이 쉬움 (특히 decision tree)
- 약점
  - **Fragile**: 데이터셋의 완벽한 세계를 가정. 학습과 모순되는 새 관측이 문제를 일으킴
  - **Convergence**: 완벽한 데이터셋에서만 수렴 보장. 노이즈가 있으면 참 가설이 배제될 가능성. 언제 멈출지도 매우 어려움
- 여전히 많이 쓰임: 쉬움 → 넓은 사용자층 → 많은 응용 → 더 나은 결과???
- **백기사(white knight)가 필요**: 노이즈 데이터셋 처리 가능, 오류에 강건
- (작은 데이터셋을 믿을 것인가? 5/6 → head 83.3% 확률?)

## 4. Linear Regression

### 4.1 How about statistical approach? [p.26]

- http://archive.ics.uci.edu/ml/datasets/Housing
- Housing dataset: 13개 수치 독립변수, 1개 수치 종속변수
- 근사 함수를 어떻게 만드나? (ML은 함수 근사 과정)
- 여기서 우리의 가설: **집값은 피처 값들의 선형 가중합**
  $$h:\ \hat f(x;\theta)=\theta_0+\sum_{i=1}^{n}\theta_i x_i=\sum_{i=0}^{n}\theta_i x_i$$
  - n은 피처 값의 개수
  - 두 측면: 선형 가중합(모델), 파라미터 θ
- 첫 과제는 압정 때와 마찬가지로 **더 나은 θ 찾기**

### 4.2 Finding θ in Linear Regression [p.27]

$$h:\ \hat f(x;\theta)=\sum_{i=0}^{n}\theta_i x_i \rightarrow \hat f = X\theta$$

$$X=\begin{pmatrix}1&\cdots&x_n^D\\ \vdots&\ddots&\vdots\\ 1&\cdots&x_n^D\end{pmatrix},\quad \theta=\begin{pmatrix}\theta_0\\ \theta_1\\ \cdots\\ \theta_n\end{pmatrix}$$

현실은 노이즈가 있으므로:

$$f(x;\theta)=\sum_{i=0}^{n}\theta_i x_i + e = y \rightarrow f = X\theta + e = Y$$

차이가 노이즈에 의한 오차이므로 이를 최소화:

$$\hat\theta=\arg\min_\theta (f-\hat f)^2=\arg\min_\theta (Y-X\theta)^2 = \arg\min_\theta (Y-X\theta)^T(Y-X\theta)$$
$$=\arg\min_\theta(\theta^TX^TX\theta-2\theta^TX^TY+Y^TY)=\arg\min_\theta(\theta^TX^TX\theta-2\theta^TX^TY)$$

### 4.3 Optimized θ [p.28]

$$\hat\theta=\arg\min_\theta(\theta^TX^TX\theta-2\theta^TX^TY)$$

압정 때와 같은 기법(미분=0):

$$\nabla_\theta(\theta^TX^TX\theta-2\theta^TX^TY)=0$$
$$2X^TX\theta-2X^TY=0$$
$$\boxed{\theta=(X^TX)^{-1}X^TY}$$

- X와 Y를 알므로 θ를 계산 가능
- 시연 목적으로 차원을 상수항과 첫 번째 피처로 제한 (Red: True, Blue: Estimated)
- **오차가 정규분포를 따르면 이는 MLE와 같다**

### 4.4 If you want more…. [p.29]

- 피처 수(차원)를 늘릴 수 있음: x에 $x^2, x^3, x^4$… 추가
- 결과를 개선할 수 있음! — **정말 그런가?**
  $$h:\ \hat f(x;\theta)=\sum_{i=0}^{n}\sum_{j=1}^{m}\theta_{i,j}\phi_j(x_i)$$
- "Better Fitting?" 영역과 그렇지 않은 영역이 동시에 발생 → 나중에 다시 다룸

### 4.5 Too Brittle to Be Used Naively [p.30]

- 우리가 하는 일: 데이터셋에 함수를 근사
  - 함수는 **Discrete logics** 또는 **Statistical model** 일 수 있음
  - 함수 유형이 주어지는 경우가 많고, 그 함수의 **파라미터**가 분석 대상
- 파라미터를 찾는 대안들
  - MLE or MAP
  - Engineering the features
  - Setting the generalization and the specialization level
- 대안 중 최선은? 완벽한 데이터가 있으면 알 수 있지만, 없다

## 5. Acknowledgement & Further Readings [p.31–32]

- 영향: Prof. Tom Mitchell (CMU), Prof. Eric P. Xing (CMU)
- Further readings: **Bishop Chapter 14.4**, **Mitchell Chapter 1, 2, 3**

## Week 2 검토 노트

| 위치 | 내용 | 판정 |
|---|---|---|
| p.18 | 690 = 307(+) + 383(-); A1: (98+112)+(206+262)+(3+9)=690, 양성합 98+206+3=307, 음성합 112+262+9=383 | **정합** |
| p.18 | A9: (284+77)+(23+306)=690, 양성합 284+23=307, 음성합 77+306=383 | **정합** |
| p.21 | 오분류 21+56+0+5+17+1=100, (690-100)/690=85.51% | **정합** |
| p.22 | 오분류 5+17+1+56+7+5+7+1=99, (690-99)/690=85.65% | **정합** |
| p.8 | Find-S에서 인스턴스를 $x_1,x_2,x_4$로 쓰면서 가설은 $h_{1,2,3}$, $h_{1,2,3,4}$로 표기 | 원문 인덱스 불일치(경미). 슬라이드 그림상 $x_3$는 negative라 건너뛴 것으로 읽힘 |
| p.8 | `f_i in h = f_i in h ∪ f_i in x` | 실제로는 "일치하지 않으면 `?`(don't care)로 대체"를 의미 |
| p.20 | $H(Y\|A1)=\sum\sum P(x,y)\log_2\frac{P(x)}{P(x,y)}$ | 조건부 엔트로피의 양수형 표현으로 정확 ($-\sum P(x,y)\log\frac{P(x,y)}{P(x)}$와 동일) |
| p.27 | 행렬 X의 첫 행과 마지막 행이 모두 $x_n^D$로 표기 | 원문 표기 오류. 각 행은 서로 다른 인스턴스여야 함 |
| p.16 vs p.34(W1) | 이항분포 표기 일관성 | W1 검토 노트 참조 |
| p.2 | "Information theory appraoch" | 오타 (approach) |

---

# Week 3 — Naïve Bayes Classifier

## 1. 주차 목표 [p.2]

- **최적 분류(optimal classification) 개념** 학습: optimal predictor, Bayes risk, decision boundary
- **naïve Bayes classifier** 학습: 분류기 이해, 선형 분류기의 베이지안 버전, 조건부 독립, naïve 가정
- **텍스트 마이닝 사례 적용**: bag-of-words 개념, 문서 분류에 적용

## 2. Optimal Classification and Decision Boundary

### 2.1 Supervised Learning 복습 [p.4]

(Week 1 p.11과 동일 내용 — 참값을 알고 예시 제공 가능, Classification/Regression, 방법론 구분)

### 2.2 Optimal Classification [p.5]

- Bayes classifier의 **optimal predictor**
  $$f^*=\arg\min_f P(f(X)\ne Y)$$
  - 오차 최소화의 함수 근사
- Y의 클래스가 둘뿐이라 가정하면
  $$f^*(x)=\arg\max_{Y=y}P(Y=y|X=x)$$
- $\sum_{y\in Y}P(Y=y|X=x)=?$ (=1)

### 2.3 Detour: Thumbtack MLE and MAP [p.6]

- MLE: $\hat\theta=\arg\max_\theta P(D|\theta)$, $P(D|\theta)=\theta^{a_H}(1-\theta)^{a_T}$, $\hat\theta=\frac{a_H}{a_H+a_T}$
- MAP: $\hat\theta=\arg\max_\theta P(\theta|D)$, $P(\theta|D)\propto\theta^{a_H+\alpha-1}(1-\theta)^{a_T+\beta-1}$, $\hat\theta=\frac{a_H+\alpha-1}{a_H+\alpha+a_T+\beta-2}$
- 어차피 최대화이므로 계산은 동일
- 가정: Y={H,T}이면 θ는 head를 볼 확률 값 / X=D는 이전 시행, 데이터셋
- $\hat\theta=\arg\max_\theta P(\theta|D)$ → **사용자가 $\hat\theta>0.5$이면 Y=H로 가정**
- $f^*(x)=\arg\max_{Y=y}P(Y=y|X)$ → **분류기가 Y=H인지 아닌지를 알려줌**

### 2.4 Optimal Classification and Bayes Risk [p.7]

- 최적 분류기도 실수를 한다: $R(f^*)>0$ (**Bayes Risk**)
- 왜? **결합확률(joint probability)의 정보가 충분하지 않기 때문**
- $$P(Y=y|X=x)=\frac{P(X=x|Y=y)P(Y=y)}{P(X=x)}$$
- $$f^*(x)=\arg\max_{Y=y}P(Y=y|X=x)=\arg\max_{Y=y}P(X=x|Y=y)P(Y=y)$$
  - $P(X=x|Y=y)$: **Class Conditional Density**
  - $P(Y=y)$: **Class Prior**

### 2.5 Decision Boundary [p.8]

- **Decision Boundary**: 분류 결과가 바뀌는 경계
- $f^*(x)=\arg\max_{Y=y}P(Y=y|X=x)=\arg\max_{Y=y}P(X=x|Y=y)P(Y=y)$
- Gaussian class conditional density라면?
  $$P(X=x;\mu,\sigma)=\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$
- $P(X=x|Y=y)P(Y=y)$ 두 곡선의 교점이 경계

### 2.6 Decision Boundary in Two Dimension [p.9]

- 클래스 조건부 밀도로 **두 개의 다변량 정규분포** 사용
- Decision boundary → **직선 (linear decision boundary)**
- 실세계 응용에서의 문제? → $x_1$과 $x_2$의 조합을 관측하는 것
- 1차원 → 2차원 확장:
  $$P(X=x;\mu,\sigma)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$
  $$P(X=(x_1,x_2)|Y=y)=\frac{1}{\sqrt{2\pi|\Sigma_y|}}\exp\left(-\frac{(x-\mu_y)\Sigma_y^{-1}(x-\mu_y)'}{2}\right)$$

### 2.7 Learning the Optimal Classifier [p.10]

$$f^*(x)=\arg\max_{Y=y}P(Y=y|X=x)=\arg\max_{Y=y}\underbrace{P(X=x|Y=y)}_{\text{Class Conditional Density}}\underbrace{P(Y=y)}_{\text{Class Prior}}$$

- 알아야 할 것
  - Prior = Class Prior = $P(Y=y)$
  - Likelihood = Class Conditional Density = $P(X=x|Y=y)$
- 어떻게 값을 아는가? → 데이터셋 D의 관측을 통해
- 그런데 D가 모든 X와 Y를 갖고 있는가? 특히 **X의 모든 조합**을?

## 3. Naïve Bayes Classifier

### 3.1 Dataset for Optimal Classifier Learning [p.12]

EnjoySport 데이터셋 재사용.

- $f^*(x)=\arg\max_{Y=y}P(X=x|Y=y)P(Y=y)$
  - $P(X=x|Y=y)=P(x_1=sunny, x_2=warm, x_3=normal, x_4=strong, x_5=warm, x_6=same\mid y=Yes)$
  - $P(Y=y)=(y=Yes)$
- 필요한 파라미터/관측 수는?
  - 모든 x,y에 대한 $P(X=x|Y=y)$ → $(2^d-1)k$
  - 모든 y에 대한 $P(Y=y)$ → $k-1$
  - 흔히 벌어지는 일: $N \gg (2^d-1)k \gg |D|$
- 완벽한 세계가 아님을 기억! 노이즈가 있으므로 확률변수+분포로 모델링해야 하고, **반복 관측(replications)이 필요**

### 3.2 Why need an additional assumption? [p.13]

- 위 모델을 학습하려면 **얻을 수 없을 만큼 거대한 데이터셋**이 필요
- 비현실적 가정들을 완화했더니, 이제 모델이 학습 불가능해짐
  - 다른 가정을 추가할 때
  - 완화한 가정들만큼 치명적이지 않은 가정
- 데이터 수요의 주요 원인은?
  - $P(X=x|Y=y)$ for all x,y → $(2^d-1)k$
  - x는 벡터값이고 길이가 d → **d가 수요의 원천**
  - 그럼 d를 줄일까? 아니면 ????

### 3.3 Conditional Independence [p.14]

- 지나가던 통계학자: "이러면 어때?"
  $$P(X=\langle x_1,\dots,x_i\rangle|Y=y)\rightarrow \prod_i P(X_i=x_i|Y=y)$$
- "가능한가요?" → "예! $x_1,\dots,x_i$가 y가 주어졌을 때 조건부 독립이라면"
- **Conditional Independence**
  - $x_1$이 y가 주어졌을 때 $x_2$와 조건부 독립
  - $(\forall x_1,x_2,y)\ P(x_1|x_2,y)=P(x_1|y)$
  - 따라서 $P(x_1,x_2|y)=P(x_1|y)P(x_2|y)$
- 예: $P(Thunder|Rain, Lightning)=P(Thunder|Lightning)$
  - 번개가 있으면 비와 무관하게 확률 p로 천둥이 발생

### 3.4 Conditional vs. Marginal Independence [p.15]

Commander → OfficerA, OfficerB (Commander가 "Go!"를 지시)

- **Marginal independence**
  - $P(OfficerA=Go|OfficerB=Go) > P(OfficerA=Go)$
  - → **주변 독립이 아님!**
  - X와 Y가 독립 ⟺ $P(X)=P(X|Y)$, 따라서 $P(X,Y)=P(X)P(Y)$
- **Conditional independence**
  - $P(OfficerA=Go|OfficerB=Go, Commander=Go)=P(OfficerA=Go|Commander=Go)$
  - → **조건부 독립임!**

### 3.5 Dataset for Optimal Classifier Learning with Conditional Independent Assumption [p.16]

- 이전: $f^*(x)=\arg\max_{Y=y}P(X=x|Y=y)P(Y=y)$, $P(X=x|Y=y)$는 $(2^d-1)k$ 가지
- X의 모든 피처에 조건부 독립 가정 적용:
  $$f^*(x)=\arg\max_{Y=y}P(X=x|Y=y)P(Y=y)\approx\arg\max_{Y=y}P(Y=y)\prod_{1\le i\le d}P(X_i=x_i|Y=y)$$
- 가정 채택 후 파라미터 수는? → $P(X_i=x_i|Y=y)$ 는 **$(2-1)dk$ 가지**
- "잠깐! 지나가던 통계학자! 이게 맞나요????!!!!"

### 3.6 Naïve Bayes Classifier [p.17]

- 통계학자: "그래, 가정이 naïve한 건 안다. 그러니 **naïve Bayes classifier**라고 부르지 그래?"
- Given
  - Class Prior **P(Y)**
  - 클래스 Y가 주어졌을 때 조건부 독립인 **d개의 피처 X**
  - 각 $X_i$에 대해 likelihood $P(X_i|Y)$
- **Naïve Bayes Classifier Function**
  $$f_{NB}(x)=\arg\max_{Y=y}P(Y=y)\prod_{1\le i\le d}P(X_i=x_i|Y=y)$$
- naïve Bayes classifier는 다음 조건에서 **최적 분류기**
  - X에 대한 조건부 독립 가정이 성립할 때
  - prior가 옳을 때
- 문제는 없나????

### 3.7 Problem of Naïve Bayes Classifier [p.18]

- **Problem 1: Naïve assumption**
  - 매우 많은 경우 X의 변수들이 상관되어 있음
  - 왜? **Multi-collinearity**
- **Problem 2: Incorrect Probability Estimations**
  - 억만장자: Head, Head, Head…
  - 데이터 부족 상태의 MLE → Tail의 가능성이 전혀 없음! $P(Y=tail)=0$
  - 어리석은 prior의 MAP → 우리 데이터셋이나 지식이 prior를 추정할 만큼 좋은가?
- **Problem 2는 항상 존재. Problem 1은 우리의 가정이 도입한 것.**

## 4. Text Mining Application: Simple Sentiment Classification

### 4.1 Product Review and Sentiment Analysis [p.20]

- Amazon: 제품 정보 + 제품 리뷰
- 리뷰에는 긍정도 있고 부정도 있음
- **10,000개의 리뷰에서 부정 리뷰를 찾으려면?**

### 4.2 Why simple word searching doesn't work [p.21]

- 보편적으로 좋은/나쁜 단어가 존재: Excellent, good, super… / Horrible, worst, never…
- 그런데 이런 건? — **Cool?** (Cool Beer) / **Hot?** (Hot Pizza) / **Big?** (Big LCD) / **Small?** (Small Size)
- 검색과 카운팅 → **확률적 접근(Probabilistic approach)**

### 4.3 Bag Of Words [p.22]

- 통계 분석을 위해 리뷰 텍스트를 **벡터**로 변환
- 벡터 `<1,0,0,1>`
- 단어 목록 `<I, cool, lcd, reliant>`
- 합치면: 이 리뷰는 "I"와 "reliant"를 포함

### 4.4 Sample Dataset [p.23]

- Bag of words: **198 documents**, **29,717 unique words**
- Classes: Positive Sentiment / Negative Sentiment
- Naïve Bayes Classifier 적용:
  $$f_{NB}(x)=\arg\max_{Y=y}P(Y=y)\prod_{1\le i\le d}P(X_i=x_i|Y=y)$$
- 계산해야 할 것: $P(Y=y)$, $P(X_i=x_i|Y=y)$

### 4.5 Matlab Exercise! [p.24]

Let's do some coding…

## 5. Acknowledgement & Further Readings [p.25–26]

- 영향: Prof. Eric P. Xing (CMU)
- Further readings: **Bishop Chapter 1, 8.2**

## Week 3 검토 노트

| 위치 | 내용 | 판정 |
|---|---|---|
| p.12/13/16 | $P(X=x\|Y=y)$ 파라미터 수 $(2^d-1)k$ | 이진 피처 d개, 클래스 k개 가정 시 정확. 단 EnjoySport는 실제로 이진 피처가 아님(Sky 3값 등) → 설명용 단순화 |
| p.16 | 조건부 독립 후 $(2-1)dk$ | 이진 피처 가정 하 정확 |
| p.12 | "N >> (2^d-1)k >> |D|" | N(필요 관측 수)이 파라미터 수보다, 파라미터 수가 실제 데이터 크기보다 훨씬 큼 — 학습 불가 상황 설명 |
| p.18 | "$P(Y=tail)=0$" | **표기 오류**. Y는 클래스, tail은 관측값. 문맥상 $\hat\theta_{tail}=P(X=tail\|Y)=0$ 또는 $P(tail)=0$을 의미 |
| p.9 | $P(X=(x_1,x_2)\|Y=y)=\frac{1}{\sqrt{2\pi\|\Sigma_y\|}}\exp(\cdot)$ | 2차원 다변량 정규의 정규화 상수는 $\frac{1}{2\pi\|\Sigma\|^{1/2}}$ (일반 d차원: $(2\pi)^{-d/2}\|\Sigma\|^{-1/2}$). **원문 상수 부정확** |
| p.9 | $(x-\mu_y)\Sigma_y^{-1}(x-\mu_y)'$ | 행/열 벡터 규약에 따른 표기. 통상 $(x-\mu)^T\Sigma^{-1}(x-\mu)$ |
| p.5 | $f^*=\arg\min_f P(f(X)\ne Y)$ → $f^*(x)=\arg\max_y P(Y=y\|X=x)$ | 0-1 손실 하에서 정확 (두 클래스 제한 없이 일반적으로 성립) |
| p.15 | Commander/Officer 예시 | explaining-away의 반대 구조(common cause). 조건부 독립 O, 주변 독립 X — **정확** |

---

# Week 4 — Logistic Regression

## 1. 주차 목표 [p.2]

- **Logistic regression classifier** 학습
  - 분류 과제에서 linear regression보다 logistic regression이 왜 더 적합한지
  - logistic function 이해
  - logistic regression classifier 이해
  - open form solution에 대한 근사 접근 이해
- **Gradient descent algorithm** 학습: Taylor expansion, gradient descent/ascent
- **naïve Bayes와 logistic regression의 차이** 학습: 유사점, 차이점, 성능 차이

## 2. Logistic Regression

### 2.1 Optimal Classification and Bayes Risk 재확인 [p.4]

- $P(Y|X)$의 **선형 함수 vs. 비선형 함수** — 어느 쪽이 나은가?
- 선형 함수의 문제
  - **Range** (확률 공리 위반)
  - **Risk optimization**
- 어떤 함수를 쓸까? → **S-curve가 필요!**

### 2.2 Detour: Credit Approval Dataset [p.5]

(Week 2 p.18과 동일 — 690 instances, 307 positive, A1/A9 카운팅, 속성 목록)

### 2.3 Classification with One Variable [p.6]

- 속성 **A15**로 클래스 C를 예측해 보자
- Y축이 $P(Y|X)$를 나타낸다고 상상 → decision boundary가 직관적으로 보임
- X=A15 vs. X=log(A15) 두 가지 스케일로 표시
- 그럼 그 경계를 **어떻게 찾을까?**

### 2.4 Linear Function vs. Non-Linear Function [p.7]

- 선형 함수 적합의 문제
  - **확률 공리 위반**
  - 예제에 대한 **느린 반응(slow response)**
- **logistic function** 적합이 더 나음
  - 확률 공리 유지
  - decision boundary 근처에서 **빠른 반응**
- 표기: Blue = $(X, Y_{true})$, Red = $(X, P_{lin}(Y|X))$, Green = $(X, P_{log}(Y|X))$

### 2.5 Logistic function [p.8]

- **Sigmoid function** 은
  - Bounded / Differentiable / Real function / 모든 실수 입력에 정의 / 양의 도함수
- **Logistic function**
  $$f(x)=\frac{1}{1+e^{-x}}$$
- 인구 성장과 관련
- 왜 좋은가? sigmoid function이고, 특히 **도함수 계산이 쉬움**
- **Logit function** (역함수)
  $$f(x)=\log\left(\frac{x}{1-x}\right)$$
- 다양한 sigmoid 함수들: $\mathrm{erf}(\frac{\sqrt\pi}{2}x)$, $\tanh(x)$, $\frac{2}{\pi}\mathrm{gd}(\frac{\pi}{2}x)$, $\frac{x}{\sqrt{1+x^2}}$, $\frac{2}{\pi}\arctan(\frac{\pi}{2}x)$, $\frac{x}{1+|x|}$

### 2.6 Logistic Function Fitting [p.9]

$$f(x)=\log\left(\frac{x}{1-x}\right)\rightarrow x=\log\left(\frac{p}{1-p}\right)\rightarrow ax+b=\log\left(\frac{p}{1-p}\right)\rightarrow X\theta=\log\left(\frac{p}{1-p}\right)$$

- 1단계 Logit→Logistic: X와 Y를 역으로. Logit에서의 X가 확률
- 2단계: 더 나은 함수 적합을 위한 **선형 이동(linear shift)**
- 3단계: linear regression과 매우 유사 → **다변량(multivariate) 경우로 전환**
- Linear Regression 참조: $\hat f = X\theta$, $\theta=(X^TX)^{-1}X^TY$
- 비교
  - linear regression으로 $P(Y|X)$ 근사: $X\theta=P(Y|X)$ — 확률 공리를 지키지 못함
  - logistic function으로 $P(Y|X)$ 근사: $X\theta=\log\left(\frac{P(Y|X)}{1-P(Y|X)}\right)$ — **from linear to logistic**

### 2.7 Logistic Regression [p.10]

- **Logistic regression은 이항 또는 다항 결과를 예측하는 확률적 분류기**로, 조건부 확률을 logistic function에 적합시킴
- 다른 관점 — 이쪽이 사실 형식적 정의에 더 가까움
- Bernoulli experiment가 주어졌을 때
  $$P(y|x)=\mu(x)^y(1-\mu(x))^{1-y}$$
  $$\mu(x)=\frac{1}{1+e^{-\theta^Tx}}=P(y=1|x)$$
  여기서 $\mu(x)$가 logistic function
- 이전 슬라이드로부터
  $$X\theta=\log\left(\frac{P(Y|X)}{1-P(Y|X)}\right)\rightarrow P(Y|X)=\frac{e^{X\theta}}{1+e^{X\theta}}$$
- 결국 목표는 다시 **θ 찾기**

### 2.8 Finding the Parameter, θ [p.11]

- **MLE of θ**: 관측 데이터의 확률을 최대화하는 θ 선택, $\hat\theta=\arg\max_\theta P(D|\theta)$
- 이것은 **Maximum Conditional Likelihood Estimation (MCLE)**

$$\hat\theta=\arg\max_\theta P(D|\theta)=\arg\max_\theta\prod_{1\le i\le N}P(Y_i|X_i;\theta)$$
$$=\arg\max_\theta\log\left(\prod_{1\le i\le N}P(Y_i|X_i;\theta)\right)=\arg\max_\theta\sum_{1\le i\le N}\log(P(Y_i|X_i;\theta))$$

$$P(Y_i|X_i;\theta)=\mu(X_i)^{Y_i}(1-\mu(X_i))^{1-Y_i}$$

$$\begin{aligned}
\log(P(Y_i|X_i;\theta))&=Y_i\log(\mu(X_i))+(1-Y_i)\log(1-\mu(X_i))\\
&=Y_i\{\log(\mu(X_i))-\log(1-\mu(X_i))\}+\log(1-\mu(X_i))\\
&=Y_i\log\left(\frac{\mu(X_i)}{1-\mu(X_i)}\right)+\log(1-\mu(X_i))\\
&=Y_iX_i\theta+\log(1-\mu(X_i))=Y_iX_i\theta-\log(1+e^{X_i\theta})
\end{aligned}$$

### 2.9 Finding the Parameter, θ, contd. [p.12]

$$\hat\theta=\arg\max_\theta\sum_{1\le i\le N}\log(P(Y_i|X_i;\theta))=\arg\max_\theta\sum_{1\le i\le N}\{Y_iX_i\theta-\log(1+e^{X_i\theta})\}$$

θ의 특정 원소를 찾기 위한 편미분:

$$\begin{aligned}
\frac{\partial}{\partial\theta_j}\Big\{\sum_{1\le i\le N}Y_iX_i\theta-\log(1+e^{X_i\theta})\Big\}
&=\Big\{\sum_{1\le i\le N}Y_iX_{i,j}\Big\}+\Big\{\sum_{1\le i\le N}-\frac{1}{1+e^{X_i\theta}}\times e^{X_i\theta}\times X_{i,j}\Big\}\\
&=\sum_{1\le i\le N}X_{i,j}\left(Y_i-\frac{e^{X_i\theta}}{1+e^{X_i\theta}}\right)=\sum_{1\le i\le N}X_{i,j}(Y_i-P(Y_i=1|X_i;\theta))=0
\end{aligned}$$

- **더 이상 전개할 방법이 없음** → **closed form solution이 없음!**
- **Open form solution → 근사(approximate)!**
- (linear regression은 closed form: $\nabla_\theta(\theta^TX^TX\theta-2\theta^TX^TY)=0 \Rightarrow 2X^TX\theta-2X^TY=0 \Rightarrow \theta=(X^TX)^{-1}X^TY$)

## 3. Gradient Method

### 3.1 Taylor Expansion [p.14]

- Taylor series는 함수의 표현으로, 한 고정점에서의 도함수 값들로 계산된 항들의 무한합
  $$f(x)=f(a)+\frac{f'(a)}{1!}(x-a)+\frac{f''(a)}{2!}(x-a)^2+\cdots=\sum_{n=0}^{\infty}\frac{f^{(n)}(a)}{n!}(x-a)^n$$
  - a = 상수값
- Taylor series는 a에서 무한번 미분 가능할 때 가능 (실수 또는 복소수 a)
- Taylor expansion은 Taylor series를 생성하는 과정
- 예시
  - a=0일 때: $e^x=1+\frac{e^0}{1!}(x-0)^1+\frac{e^0}{2!}(x-0)^2+\cdots$ (n=2, n=4 근사 그래프)
  - a=0.5일 때: $\log x=\log(0.5)+\frac{1/0.5}{1!}(x-0.5)^1+\frac{1/0.5^2}{2!}(x-0.5)^2+\cdots$ (n=2, n=100)

### 3.2 Gradient Descent/Ascent [p.15]

- Gradient descent/ascent method 는
  - 미분 가능한 함수 $f(x)$와 초기 파라미터 $x_1$이 주어졌을 때
  - 파라미터를 $f(x)$의 더 낮은/높은 값으로 반복 이동
  - $f(x)$의 음/양 gradient 방향을 취함으로써
- 왜 작동하나?
  $$f(x)=f(a)+\frac{f'(a)}{1!}(x-a)+O(\|x-a\|^2)$$
  - $a=x_1$, $x=x_1+h\mathbf{u}$ 로 두면 (**u**는 편미분의 단위 방향 벡터)
  $$f(x_1+h\mathbf{u})=f(x_1)+hf'(x_1)\mathbf{u}+h^2O(1)$$
  $$f(x_1+h\mathbf{u})-f(x_1)\approx hf'(x_1)\mathbf{u}$$
  $$\mathbf{u}^*=\arg\min_\mathbf{u}\{f(x_1+h\mathbf{u})-f(x_1)\}=\arg\min_\mathbf{u}hf'(x_1)\mathbf{u}=-\frac{f'(x_1)}{|f'(x_1)|}$$
  - ∵ $f(x_1+h\mathbf{u})\le f(x_1)$, $\vec a\cdot\vec b=|\vec a||\vec b|\cos\alpha$
  $$x_{t+1}\leftarrow x_t+h\mathbf{u}^*=x_t-h\frac{f'(x_t)}{|f'(x_t)|}\quad\textbf{(Gradient Descent)}$$
- $\hat\theta=\arg\max_\theta\sum_{1\le i\le N}\log(P(Y_i|X_i;\theta))$ 에 완벽히 적용 가능 **(Gradient Ascent)**
  - $f(\theta)=\sum_{1\le i\le N}\log(P(Y_i|X_i;\theta))$
  - 초기 파라미터 $\theta_1$ 설정
  - $\theta_t$를 $f(\theta_t)$의 더 높은 값으로 반복 이동
  - $f(\theta_t)$의 **양(positive)** gradient 방향을 취함

### 3.3 How Gradient Descent Works [p.16]

**예제 함수: Rosenbrock function**

$$f(x_1,x_2)=(1-x_1)^2+100(x_2-x_1^2)^2$$
$$\frac{\partial}{\partial x_1}f(x_1,x_2)=-2(1-x_1)-400x_1(x_2-x_1^2)$$
$$\frac{\partial}{\partial x_2}f(x_1,x_2)=200(x_2-x_1^2)$$

- Global Minimum = 0 at (1,1)
- 초기점 $\mathbf{x}^0=(x_1^0,x_2^0)=(-1.3, 0.9)$
- 그 점에서의 편미분 벡터
  $$\mathbf{f}'(\mathbf{x}^0)=\left(\frac{\partial}{\partial x_1}f,\frac{\partial}{\partial x_2}f\right)=(-415.4, -158)$$
- 음의 편미분으로 작은 스케일 h=0.001 만큼 갱신
  $$\mathbf{x}^1\leftarrow\mathbf{x}^0-h\frac{\mathbf{f}'(\mathbf{x}^0)}{|\mathbf{f}'(\mathbf{x}^0)|}$$
  $$\mathbf{x}^1=\begin{pmatrix}-1.3-0.001\times(-415.4/444.4335)\\ 0.9-0.001\times(-158/444.4335)\end{pmatrix}=(-1.2991, 0.9004)$$
- 수렴할 때까지 갱신 반복

### 3.4 Finding θ with Gradient Ascent [p.17]

$$\hat\theta=\arg\max_\theta\sum_{1\le i\le N}\log(P(Y_i|X_i;\theta))$$
$$f(\theta)=\sum_{1\le i\le N}\log(P(Y_i|X_i;\theta))$$
$$\frac{\partial f(\theta)}{\partial\theta_j}=\frac{\partial}{\partial\theta_j}\Big\{\sum_{1\le i\le N}\log(P(Y_i|X_i;\theta))\Big\}=\sum_{1\le i\le N}X_{i,j}(Y_i-P(y=1|x;\theta))$$

- gradient method를 쓰려면 $f'(x)$를 알아야 하고, 위에서 구했음
- Ascent의 경우: $x_{t+1}\leftarrow x_t+h\mathbf{u}^*=x_t+h\frac{f'(x_t)}{|f'(x_t)|}$
- 파라미터 θ의 반복 갱신
  $$\theta_j^{t+1}\leftarrow\theta_j^t+h\frac{\partial f(\theta^t)}{\partial\theta_j^t}=\theta_j^t+h\Big\{\sum_{1\le i\le N}X_{i,j}(Y_i-P(Y=1|X_i;\theta^t))\Big\}$$
  $$=\theta_j^t+\frac{h}{C}\Big\{\sum_{1\le i\le N}X_{i,j}\Big(Y_i-\frac{e^{X_i\theta^t}}{1+e^{X_i\theta^t}}\Big)\Big\}$$
  - C = 단위 벡터로의 정규화(Normalization to the unit vector)
- $\theta_j^0$는 임의로 선택 가능

### 3.5 Logistic Regression Matlab Exercise [p.18]

Let's do some coding…

### 3.6 Linear Regression Revisited [p.19]

이전:

$$\hat\theta=\arg\min_\theta(f-\hat f)^2=\arg\min_\theta(Y-X\theta)^2=\arg\min_\theta(Y-X\theta)^T(Y-X\theta)$$
$$=\arg\min_\theta(\theta^TX^TX\theta-2\theta^TX^TY+Y^TY)=\arg\min_\theta(\theta^TX^TX\theta-2\theta^TX^TY)$$
$$\nabla_\theta(\theta^TX^TX\theta-2\theta^TX^TY)=0 \Rightarrow 2X^TX\theta-2X^TY=0 \Rightarrow \theta=(X^TX)^{-1}X^TY$$

**문제는 없나???** → **Gradient descent가 해결책이 될 수 있음**

$$\hat\theta=\arg\min_\theta(f-\hat f)^2=\arg\min_\theta(Y-X\theta)^2=\arg\min_\theta\sum_{1\le i\le N}\Big(Y^i-\sum_{1\le j\le d}X_j^i\theta_j\Big)^2$$
$$\frac{\partial}{\partial\theta_k}\sum_{1\le i\le N}\Big(Y^i-\sum_{1\le j\le d}X_j^i\theta_j\Big)^2=-\sum_{1\le i\le N}2\Big(Y^i-\sum_{1\le j\le d}X_j^i\theta_j\Big)X_k^i$$
$$\theta_k^{t+1}\leftarrow\theta_k^t-h\frac{\partial f(\theta^t)}{\partial\theta_k^t}=\theta_k^t+h\sum_{1\le i\le N}2\Big(Y^i-\sum_{1\le j\le d}X_j^i\theta_j\Big)X_k^i$$

## 4. Naïve Bayes vs. Logistic Regression

### 4.1 Gaussian Naïve Bayes [p.21]

- 두 분류기의 성능을 비교하고 싶다
  - logistic regression은 연속 피처를 다룸
  - naïve Bayes는 왜 안 되나?
- Naïve Bayes Classifier Function
  $$f_{NB}(x)=\arg\max_{Y=y}P(Y=y)\prod_{1\le i\le d}P(X_i=x_i|Y=y)$$
- 피처가 **연속 확률변수**라면? → 평균 μ, 분산 $\sigma^2$의 **가우시안 분포**를 따른다고 가정
  $$P(X_i|Y,\mu,\sigma^2)=\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(X_i-\mu)^2}{2\sigma^2}}$$
- 축약 표기: $P(Y=y)=\pi_1$
- $$P(Y)\prod_{1\le i\le d}P(X_i|Y)=\pi_k\prod_{1\le i\le d}\frac{1}{\sigma_k^i C}\exp\left(-\frac{1}{2}\left(\frac{X_i-\mu_k^i}{\sigma_k^i}\right)^2\right)$$

### 4.2 Derivation to Logistic Regression (1) [p.22]

naïve Bayes 가정 하에서:

$$P(Y=y|X)=\frac{P(X|Y=y)P(Y=y)}{P(X)}=\frac{P(X|Y=y)P(Y=y)}{P(X|Y=y)P(Y=y)+P(X|Y=n)P(Y=n)}$$
$$=\frac{P(Y=y)\prod_{1\le i\le d}P(X_i|Y=y)}{P(Y=y)\prod_{1\le i\le d}P(X_i|Y=y)+P(Y=n)\prod_{1\le i\le d}P(X_i|Y=n)}$$

가우시안을 대입하면:

$$P(Y=y|X)=\frac{\pi_1\prod_i\frac{1}{\sigma_1^iC}\exp\left(-\frac12\left(\frac{X_i-\mu_1^i}{\sigma_1^i}\right)^2\right)}{\pi_1\prod_i\frac{1}{\sigma_1^iC}\exp\left(-\frac12\left(\frac{X_i-\mu_1^i}{\sigma_1^i}\right)^2\right)+\pi_2\prod_i\frac{1}{\sigma_2^iC}\exp\left(-\frac12\left(\frac{X_i-\mu_2^i}{\sigma_2^i}\right)^2\right)}$$

$$=\cfrac{1}{1+\cfrac{\pi_2\prod_i\frac{1}{\sigma_2^iC}\exp\left(-\frac12\left(\frac{X_i-\mu_2^i}{\sigma_2^i}\right)^2\right)}{\pi_1\prod_i\frac{1}{\sigma_1^iC}\exp\left(-\frac12\left(\frac{X_i-\mu_1^i}{\sigma_1^i}\right)^2\right)}}$$

### 4.3 Derivation to Logistic Regression (2) [p.23]

**두 클래스의 분산이 같다고 가정**: $\sigma_2^i=\sigma_1^i$

$$P(Y=y|X)=\cfrac{1}{1+\cfrac{\pi_2\prod_i\exp\left(-\frac12\left(\frac{X_i-\mu_2^i}{\sigma_2^i}\right)^2\right)}{\pi_1\prod_i\exp\left(-\frac12\left(\frac{X_i-\mu_1^i}{\sigma_1^i}\right)^2\right)}}$$

$$=\cfrac{1}{1+\cfrac{\pi_2\exp\left(-\sum_i\left\{\frac12\left(\frac{X_i-\mu_2^i}{\sigma_2^i}\right)^2\right\}\right)}{\pi_1\exp\left(-\sum_i\left\{\frac12\left(\frac{X_i-\mu_1^i}{\sigma_1^i}\right)^2\right\}\right)}}
=\cfrac{1}{1+\cfrac{\exp\left(-\sum_i\left\{\frac12\left(\frac{X_i-\mu_2^i}{\sigma_2^i}\right)^2\right\}+\log\pi_2\right)}{\exp\left(-\sum_i\left\{\frac12\left(\frac{X_i-\mu_1^i}{\sigma_1^i}\right)^2\right\}+\log\pi_1\right)}}$$

$$=\cfrac{1}{1+\exp\left(-\sum_i\left\{\frac12\left(\frac{X_i-\mu_2^i}{\sigma_2^i}\right)^2\right\}+\log\pi_2+\sum_i\left\{\frac12\left(\frac{X_i-\mu_1^i}{\sigma_1^i}\right)^2\right\}-\log\pi_1\right)}$$

$$\text{(원문)}=\cfrac{1}{1+\exp\left(\sum_i-\frac{1}{2(\sigma_1^i)^2}\left(\left\{(X_i-\mu_1^i)^2-(X_i-\mu_2^i)^2\right\}+\log\pi_2-\log\pi_1\right)\right)}$$

$$\text{(원문)}=\cfrac{1}{1+\exp\left(\sum_i-\frac{1}{2(\sigma_1^i)^2}\left(\left\{2(\mu_2^i-\mu_1^i)X_i+{\mu_2^i}^2-{\mu_2^i}^2\right\}++\log\pi_2-\log\pi_1\right)\right)}$$

> ⚠️ 마지막 두 줄은 원문 그대로이며 **부호·항 오류**가 있다. 정정본은 아래 Week 4 검토 노트 참조.

### 4.4 Naïve Bayes vs. Logistic Regression [p.24]

**Naïve Bayes classifier** (원문 표기)

$$P(Y|X)=\cfrac{1}{1+\exp\left(-\frac{1}{2(\sigma_1^i)^2}\sum_{1\le i\le d}\left\{2(\mu_2^i-\mu_1^i)X_i+{\mu_2^i}^2-{\mu_2^i}^2\right\}+\log\pi_2-\log\pi_1\right)}$$

이 공식을 얻기 위한 가정

- Naïve Bayes assumption, 클래스 간 **동일 분산 가정**
- $P(X|Y)$에 대한 **가우시안 분포**
- $P(Y)$에 대한 **베르누이 분포**
- (합쳐서 **결합확률(joint prob.)을 모델링**)
- 추정할 파라미터 수 = $2\times2\times d+1=4d+1$ (클래스별 분산이 다를 때)

**Logistic Regression**

$$P(Y|X)=\frac{1}{1+e^{-\theta^Tx}}$$

- 가정: **logistic function에 적합**
- 추정할 파라미터 수 = $d+1$

**누가 승자인가?** → 정말로… 승자는 없다. 왜?

### 4.5 Generative-Discriminative Pair [p.25]

**Generative model**, $P(Y|X)=P(X,Y)/P(X)=P(X|Y)P(Y)/P(X)$

- 모든 변수의 완전한 확률 모델
  - 데이터로부터 $P(X|Y)$, $P(Y)$의 파라미터 추정
- 특징: Bayesian, Prior, **결합확률 모델링**
- 예: **Naïve Bayes Classifier**

**Discriminative model**, $P(Y|X)$

- 관측 변수의 분포를 모델링할 필요 없음
  - 데이터로부터 $P(Y|X)$의 파라미터 추정
- 특징: **조건부확률 모델링**
- 예: **Logistic Regression**

**Pros and Cons [Ng & Jordan, 2002]**

- Logistic regression is **less biased**
- Probably approximately correct learning: **Naïve Bayes learns faster**

## 5. Acknowledgement & Further Readings [p.26–27]

- 영향: Prof. Carlos Guestrin (CMU), Prof. Eric Xing (CMU), Prof. Tom Mitchell (CMU)
- Further readings: **Bishop Chapter 4.3, 5.2.1-5.2.4**

## Week 4 검토 노트

### ⚠️ p.23–24: Gaussian NB → Logistic Regression 유도의 부호/항 오류

원문 p.23 3번째 줄까지는 정확하다:

$$P(Y=y|X)=\cfrac{1}{1+\exp\left(-\sum_i\frac12\left(\frac{X_i-\mu_2^i}{\sigma^i}\right)^2+\log\pi_2+\sum_i\frac12\left(\frac{X_i-\mu_1^i}{\sigma^i}\right)^2-\log\pi_1\right)}$$

이를 정리하면 지수부는

$$\sum_i\frac{1}{2(\sigma^i)^2}\left\{(X_i-\mu_1^i)^2-(X_i-\mu_2^i)^2\right\}+\log\pi_2-\log\pi_1$$

$$(X_i-\mu_1^i)^2-(X_i-\mu_2^i)^2 = 2(\mu_2^i-\mu_1^i)X_i+{\mu_1^i}^2-{\mu_2^i}^2$$

**정정된 최종 형태**

$$\boxed{P(Y=y|X)=\cfrac{1}{1+\exp\left(\displaystyle\sum_{1\le i\le d}\frac{1}{2(\sigma^i)^2}\left\{2(\mu_2^i-\mu_1^i)X_i+{\mu_1^i}^2-{\mu_2^i}^2\right\}+\log\pi_2-\log\pi_1\right)}}$$

원문 대비 오류 3가지:

1. **부호**: 원문의 $-\frac{1}{2(\sigma_1^i)^2}$ 는 $+\frac{1}{2(\sigma^i)^2}$ 이어야 함
2. **상수항**: 원문의 ${\mu_2^i}^2-{\mu_2^i}^2$ (= 0) 은 ${\mu_1^i}^2-{\mu_2^i}^2$ 의 오타
3. **괄호 위치(p.23)**: $\log\pi_2-\log\pi_1$ 이 $-\frac{1}{2\sigma^2}$ 계수 안에 들어가 있으나, 실제로는 밖에 있어야 함 (p.24에서는 밖으로 나와 있어 p.23과 불일치)

이 형태는 $\theta^Tx$ 꼴의 선형식이므로 **Gaussian NB(동일 분산 가정) = logistic regression과 같은 함수 형태**라는 결론 자체는 유효하다 (Mitchell, *Generative and Discriminative Classifiers* 참조).

### 기타 검토

| 위치 | 내용 | 판정 |
|---|---|---|
| p.2 | "Know the tailor expansion" | 오타 (Taylor) |
| p.11 | $\log(1-\mu(X_i))=-\log(1+e^{X_i\theta})$ | $1-\mu=\frac{1}{1+e^{X\theta}}$ 이므로 **정확** |
| p.12 | 편미분 결과 $\sum X_{i,j}(Y_i-P(Y_i=1\|X_i;\theta))=0$ | **정확** |
| p.15 | $\mathbf u^*=-\frac{f'(x_1)}{\|f'(x_1)\|}$ 후 $x_{t+1}=x_t-h\frac{f'(x_t)}{\|f'(x_t)\|}$ | **정확** (descent) |
| p.16 | $\|\mathbf f'(\mathbf x^0)\|=\sqrt{415.4^2+158^2}=444.43$ | **정합** |
| p.16 | $\frac{\partial f}{\partial x_1}$ at (-1.3,0.9): $-2(1+1.3)-400(-1.3)(0.9-1.69)=-4.6-410.8=-415.4$ | **정합** |
| p.16 | $\frac{\partial f}{\partial x_2}$ at (-1.3,0.9): $200(0.9-1.69)=-158$ | **정합** |
| p.19 | $\theta_k^{t+1}=\theta_k^t-h\frac{\partial f}{\partial\theta_k}=\theta_k^t+h\sum 2(\cdot)X_k^i$ | **정확** (도함수가 음수형이므로 부호 반전) |
| p.24 | NB 파라미터 수 $2\times2\times d+1=4d+1$ | 클래스 2개 × (μ, σ) 2개 × d + prior 1 = **정합** |
| p.21 | $P(Y=y)=\pi_1$ 로 두고 다음 줄에서 $\pi_k$ 사용 | 표기 혼용(경미) |
| p.10 | $\mu(x)=\frac{1}{1+e^{-\theta^Tx}}$ vs $P(Y\|X)=\frac{e^{X\theta}}{1+e^{X\theta}}$ | 동일 식 (분자·분모에 $e^{X\theta}$ 곱) |

---

# Week 5 — Support Vector Machine

## 1. 주차 목표 [p.2]

- **Support vector machine classifier** 학습
  - SVM의 **maximum margin** 아이디어 이해
  - 최적화 문제의 **정식화(formulation)** 이해
- **Soft-margin과 penalization** 학습
  - penalization 항을 추가하는 법
  - **log-loss와 hinge-loss의 차이** 이해
- **Kernel trick** 학습
  - SVM의 **primal problem과 dual problem** 이해
  - 커널의 종류
  - SVM과 logistic regression에 kernel trick을 적용하는 법

## 2. Support Vector Machine

### 2.1 Detour: Decision Boundary [p.4] / Decision Boundary in Two Dimension [p.5]

(Week 3 p.8, p.9와 동일 내용 재확인 — Bayes risk, Gaussian class conditional density, 두 다변량 정규분포에 의한 선형 decision boundary)

### 2.2 Decision Boundary without Prob. [p.6]

- **확률분포를 고려하지 않고** 어느 것이 더 나은 decision boundary인가?
- **어느 점들이 최전선(front line)에 있는가?**
- (2차원 산점도에 여러 후보 직선이 그려져 있음)

### 2.3 Decision Boundary with Margin [p.7]

- **최대 margin을 갖는 decision boundary**
  - 경계에 가까운 점들 사이에서
  - 점은 몇 개나 필요한가?
- Decision boundary line
  - $\mathbf{w}\cdot\mathbf{x}+b=0$
  - Positive case: $\mathbf{w}\cdot\mathbf{x}+b>0$
  - Negative case: $\mathbf{w}\cdot\mathbf{x}+b<0$
  - **Confidence level**: $(\mathbf{w}\cdot\mathbf{x_j}+b)y_j$
- **Margin?** → 가장 가까운 점에서 decision boundary까지의 **수직 거리**
- (파라미터는 몇 개인가?)

### 2.4 Margin Distance [p.8]

- $f(\mathbf{x})=\mathbf{w}\cdot\mathbf{x}+b$
- 경계 위의 점 x: $f(\mathbf{x})=\mathbf{w}\cdot\mathbf{x}+b=0$
- positive 점 x: $f(\mathbf{x})=\mathbf{w}\cdot\mathbf{x}+b=a,\ a>0$
- 임의의 점 **x** 와, 경계 위에 있으면서 **x**에서 경계로 내린 수선의 발인 점 $\mathbf{x_p}$ 사이의 거리를 측정
  $$x=x_p+r\frac{w}{\|w\|},\quad f(x_p)=0$$
  $$f(x)=w\cdot x+b=w\left(x_p+r\frac{w}{\|w\|}\right)+b=wx_p+b+r\frac{w\cdot w}{\|w\|}=r\|w\|$$
- 따라서 **거리**는
  $$r=\frac{f(x)}{\|w\|}$$
- (예시: $x_2=x_1-1.5 \rightarrow x_1-x_2-1.5=0$, $w=(1,-1)$, 점 (2,2), (3,3), (3,0))

### 2.5 Maximizing the Margin [p.9]

- 좋은 decision boundary? → **Maximum margin!**
  - $r=\dfrac{a}{\|w\|}$
  - **양쪽 모두**를 고려해야 함
- 최적화 문제
  $$\max_{w,b} 2r=\frac{2a}{\|w\|}\quad s.t.\ (wx_j+b)y_j\ge a,\ \forall j$$
- $a$는 임의의 수이므로 **정규화 가능** (a=1)
  $$\boxed{\min_{w,b}\|w\|\quad s.t.\ (wx_j+b)y_j\ge 1,\ \forall j}$$
- 이것은 **quadratic optimization problem**이 된다. 왜?
- 경계선들: $f(x)=w\cdot x+b=-a$ / $=0$ / $=a$

### 2.6 Support Vector Machine with Hard Margin [p.10]

- **SVM**은 어떤 클래스의 가장 가까운 학습 데이터점까지의 거리가 최대가 되도록 **초평면(hyperplane) 집합을 구성**한다
- **Hard margin**
  - **오류 케이스를 전혀 허용하지 않음**
  - 오류 케이스가 있으면? → 새 decision boundary가 필요해지고, **Difficult Problem!!!**
- Hard margin SVM 구현:
  $$\min_{w,b}\|w\|\quad s.t.\ (wx_j+b)y_j\ge 1,\ \forall j$$

## 3. Soft Margin

### 3.1 "Error" Cases in SVM [p.12]

- **선형 decision boundary로 분류가 불가능한** 데이터점들 → 이른바 "error" 케이스
- 어떻게 관리하나?
  - **Option 1**: decision boundary를 더 복잡하게 만든다 → 비선형으로 간다. 문제는?
  - **Option 2**: "error"가 있을 것임을 인정하고, 문제 정식화에 오류를 표현하며, 오류도 함께 줄이려 한다. 문제는?
- (그림 주석: Non-linear decision boundary / No way to represent this with linear decision boundary / Allow some error in observations)

### 3.2 "Error" Handling in SVM [p.13]

- **Option 1)** 오류 케이스를 **세어서** 그 수를 줄인다
  $$\min_{w,b}\|w\|+C\times \#_{error}\quad s.t.\ (wx_j+b)y_j\ge 1,\ \forall j$$
  - 문제는? (0-1 loss는 미분 불가/비볼록)
- **Option 2)** **slack variable** 도입
  - 오분류일 때 $\xi_j>1$
  $$\min_{w,b}\|w\|+C\sum_j\xi_j\quad s.t.\ (wx_j+b)y_j\ge 1-\xi_j,\ \forall j,\quad \xi_j\ge 0,\ \forall j$$
- **C = trade-off parameter**
- 손실함수 비교: **0-1 Loss** (계단형) vs **Hinge Loss** (가로축 $(wx_j+b)y_j$, 1에서 꺾임)

### 3.3 Soft-Margin SVM [p.14]

$$\min_{w,b}\|w\|+C\sum_j\xi_j\quad s.t.\ (wx_j+b)y_j\ge 1-\xi_j,\ \forall j,\quad \xi_j\ge 0,\ \forall j$$

- slack variable을 추가하여 **제약을 완화(soften)**
- 대신 목적함수에서 **오분류 케이스에 페널티**를 부과: $C\sum_j\xi_j$
- **hard-margin SVM은 어떻게 복구하나?** (→ C를 무한대로)
- Hinge Loss:
  $$\xi_j=(1-(wx_j+b)y_j)_+$$
- 점들의 상태
  - 마진 밖 올바른 분류: $\xi_j=0$
  - 마진 안쪽이지만 올바른 쪽: $1>\xi_j>0$
  - 오분류: $\xi_j>1$

### 3.4 Comparison to Logistic Regression [p.15]

- Loss function: $\xi_j=loss(f(x_j),y_j)$
- **SVM loss function: Hinge Loss**
  $$\xi_j=(1-(wx_j+b)y_j)_+$$
- **Logistic Regression loss function: Log Loss**
  $$\hat\theta=\arg\max_\theta\sum_{1\le i\le N}\log(P(Y_i|X_i;\theta))=\arg\max_\theta\sum_{1\le i\le N}\{Y_iX_i\theta-\log(1+e^{X_i\theta})\}$$
  $$\xi_j=-\log(P(Y_j|X_j,w,b))=\log\left(1+e^{(wx_j+b)y_j}\right)\quad\text{(원문; 부호 오류 — 검토 노트 참조)}$$
- **어느 손실함수가 더 선호되나?**
  - decision boundary 근처에서는?
  - 전체 영역에서는?
- (그래프: 빨강=hinge loss는 1에서 0이 됨, 파랑=log loss는 0으로 수렴하지만 절대 0이 되지 않음)

### 3.5 Strength of the Loss Function [p.16]

$$\min_{w,b,\xi_j}\|w\|+C\sum_j\xi_j\quad s.t.\ (wx_j+b)y_j\ge 1-\xi_j,\ \forall j,\quad \xi_j\ge 0,\ \forall j$$

- 모델을 구현하고, **C의 변화에 따라 decision boundary가 어떻게 진화하는지** 관찰
- 실험한 C 값: **C = 0.1, 0.3, 0.5, 1, 10, 10000** (각 점의 $\xi_j$ 값이 표시됨)
  - C가 작을수록 마진이 넓고 slack이 크게 허용됨 (경계가 평평)
  - C가 커질수록 hard margin에 가까워짐

## 4. Kernel Trick

### 4.1 Enough of Studying SVM? [p.18]

- "error" 케이스가 있어도 **soft-margin**으로 SVM을 학습할 수 있다
- 그러나 이것이 **decision boundary의 복잡도(complexity)를 바꾸지는 않는다**
- 실세계에는 **복잡한 decision boundary가 필요한 상황**이 있다
  - **Option 1**: decision boundary를 더 복잡하게 → 비선형으로
  - **Option 2**: "error"를 인정하고 문제 정식화에 표현

### 4.2 Feature Mapping to Expand Dim. [p.19]

$$\min_{w,b,\xi_j}\|w\|+C\sum_j\xi_j\quad s.t.\ (w\varphi(x_j)+b)y_j\ge 1-\xi_j,\ \forall j,\quad \xi_j\ge 0,\ \forall j$$

- 예시 매핑
  $$\varphi(\langle x_1,x_2\rangle)=\langle x_1,x_2,x_1^2,x_2^2,x_1x_2,x_1^3,x_2^3,x_1^2x_2,x_1x_2^2\rangle$$
- 고차원에서 **Linearly Separable Dataset**이 되고, 원 공간으로 되돌리면 비선형 경계
- **문제는???** → **파라미터 수(# of Params), 표현(Representation), 계산(Computation)**…

### 4.3 Rethinking the Formulation [p.20]

- SVM은 **분류 → 제약 조건부 이차계획법(Constrained quadratic programming)** 으로 바뀐다
- **Constrained optimization**
  $$\min_x f(x)\quad s.t.\ g(x)\le 0,\ h(x)=0$$
- **Lagrange method**
  - Lagrange Prime Function: $L(x,\alpha,\beta)=f(x)+\alpha g(x)+\beta h(x)$
  - Lagrange Multiplier: $\alpha\ge 0,\ \beta$
  - Lagrange Dual Function: $d(\alpha,\beta)=\inf_{x\in X}L(x,\alpha,\beta)=\min_x L(x,\alpha,\beta)$
    - *inf*: infimum "Greatest Lower Bound", $\inf\{1,2,3\}=1$
  $$\max_{\alpha\ge 0,\beta}L(x,\alpha,\beta)=\begin{cases}f(x): & x\text{가 feasible할 때}\\ \infty: & \text{그 외}\end{cases}$$
  $$\min_x f(x)\rightarrow \min_x\max_{\alpha\ge0,\beta}L(x,\alpha,\beta)$$
- 제약 최적화의 정식화 기법을 활용 → **Primal and Dual Problems!**

### 4.4 Primal and Dual Problem [p.21]

| Primal Problem | Lagrange Dual Problem |
|---|---|
| $\min_x f(x)$ s.t. $g(x)\le0,\ h(x)=0$ | $\max_{\alpha>0,\beta}d(\alpha,\beta)$ s.t. $\alpha>0$ |
| ↓ | ↓ |
| $\min_x\max_{\alpha\ge0,\beta}L(x,\alpha,\beta)$ | $\max_{\alpha\ge0,\beta}\min_x L(x,\alpha,\beta)$ |

- **Weak duality theorem**
  - $d(\alpha,\beta)\le f(x^*)$ for $\forall\alpha,\forall\beta$
  - $d^*=\max_{\alpha\ge0,\beta}\min_x L(x,\alpha,\beta)\le \min_x\max_{\alpha\ge0,\beta}L(x,\alpha,\beta)=p^*$
  - 쌍대함수를 최대화하면 $f(x^*)$의 **하한(lower bound)** 을 얻는다
  - **Duality gap** = $f(x^*)-d(\alpha^*,\beta^*)$
- **Strong duality**
  - $d^*=\max_{\alpha\ge0,\beta}\min_x L(x,\alpha,\beta)=\min_x\max_{\alpha\ge0,\beta}L(x,\alpha,\beta)=p^*$
  - **KKT(Karush-Kuhn-Tucker) 조건이 만족될 때**

### 4.5 KKT Condition and Strong Duality [p.22]

**Strong duality**: $d^*=\max_{\alpha\ge0,\beta}\min_x L=\min_x\max_{\alpha\ge0,\beta}L=p^*$

**KKT 조건이 충족될 때 성립**

1. $\nabla L(x^*,\alpha^*,\beta^*)=0$
2. $\alpha^*\ge 0$
3. $g(x^*)\le 0$
4. $h(x^*)=0$
5. $\alpha^*g(x^*)=0$  ← **Complementary Slackness**

- Active Constraint: $\alpha^*=0 \Rightarrow g(x^*)=0$ *(원문 표기; 아래 검토 노트 참조)*
- Inactive Constraint: $g(x^*)<0 \Rightarrow \alpha^*=0$
- 관계도
  - Strong Duality $d^*=p^*$ **→ Always →** KKT Condition
  - KKT Condition **→ For convex optimization →** Strong Duality
  - **제약 볼록 최적화에서는 primal과 dual 문제가 동등하다**

### 4.6 Dual Problem of SVM [p.23]

**Primal Problem of Linearly Separable SVM**

$$\min_{w,b}\|w\|\quad s.t.\ (wx_j+b)y_j\ge 1,\ \forall j$$
$$\rightarrow \min_{w,b}\max_{\alpha\ge0,\beta}\frac12 w\cdot w-\sum_j\alpha_j[(wx_j+b)y_j-1]\quad s.t.\ \alpha_j\ge0,\ \forall j$$

**Lagrange Prime Function**

$$L(w,b,\alpha)=\frac12 w\cdot w-\sum_j\alpha_j[(wx_j+b)y_j-1],\quad \alpha_j\ge0,\ \forall j$$

**Dual Problem of Linearly Separable SVM**

$$\max_{\alpha\ge0}\min_{w,b}\frac12 w\cdot w-\sum_j\alpha_j[(wx_j+b)y_j-1]\quad s.t.\ \alpha_j\ge0,\ \forall j$$

**KKT Condition to Eliminate the Duality Gap**

$$\frac{\partial L(w,b,\alpha)}{\partial w}=0,\quad \frac{\partial L(w,b,\alpha)}{\partial b}=0$$
$$\alpha_i\ge0,\ \forall i$$
$$\alpha_i\big((wx_j+b)y_j-1\big)=0,\ \forall i$$

### 4.7 Dual Representation of SVM [p.24]

$$\begin{aligned}
L(w,b,\alpha)&=\frac12 w\cdot w-\sum_j\alpha_j[(wx_j+b)y_j-1]\\
&=\frac12 ww-\sum_j\alpha_jy_jwx_j-b\sum_j\alpha_jy_j+\sum_j\alpha_j\\
&=\frac12\sum_i\sum_j\alpha_i\alpha_jy_iy_jx_ix_j-\sum_i\sum_j\alpha_i\alpha_jy_iy_jx_ix_j-b\times0+\sum_j\alpha_j\\
&=\sum_j\alpha_j-\frac12\sum_i\sum_j\alpha_i\alpha_jy_iy_jx_ix_j
\end{aligned}$$

- 역시 **quadratic programming**
- $\alpha_j$를 알게 되면
  $$\mathbf{w}=\sum_{i=1}^{N}\alpha_iy_i\mathbf{x}_i,\qquad \sum_{i=1}^{N}\alpha_iy_i=0$$
  $$\alpha_i\big((wx_j+b)y_j-1\big)=0$$
- 이제 **w와 b를 다시 구할 수 있다**
  - 왜 이게 더 나은가?
  - $\alpha_j$의 **대부분은** … (0이다 → support vector만 남음)
  - x의 위치는 … (내적으로만 등장)

### 4.8 Mapping Functions [p.25]

- 비선형 분리 데이터셋이 있다고 가정
- **기저 공간(basis space)을 늘리면** 비선형 분리 케이스가 선형 분리 가능해진다
  - 표준 기저: $e_1, e_2, e_3, \dots, e_n$ → 선형 독립이고 $R^n$을 생성
- **공간 매핑 함수** $\phi: L\rightarrow H$ 를 통해 기저 확장 (또는 transformation function 등)
- **문제는????** → **feature space가 점점 커진다….**
- 예시
  $$\phi(x)=\begin{pmatrix}x_1^2\\ \sqrt2 x_1x_2\\ x_2^2\end{pmatrix}$$
  | 원 공간 L | 매핑 공간 H |
  |---|---|
  | a = (0,0) | $\hat a=(0,0,0)^T$ |
  | b = (1,0) | $\hat b=(1,0,0)^T$ |
  | c = (0,1) | $\hat c=(0,0,1)^T$ |
  | d = (1,1) | $\hat d=(1,\sqrt2,1)^T$ |

### 4.9 Kernel Function [p.26]

- 커널은 **다른 공간에서 두 벡터의 내적을 계산**한다 (되도록 그 공간에서 두 벡터를 명시적으로 표현하지 않고)
  $$K(\mathbf{x_i},\mathbf{x_j})=\varphi(\mathbf{x_i})\cdot\varphi(\mathbf{x_j})$$
- 흔한 커널들

| 커널 | 식 | 비고 |
|---|---|---|
| Polynomial (homogeneous) | $k(\mathbf{x_i},\mathbf{x_j})=(\mathbf{x_i}\cdot\mathbf{x_j})^d$ | |
| Polynomial (inhomogeneous) | $k(\mathbf{x_i},\mathbf{x_j})=(\mathbf{x_i}\cdot\mathbf{x_j}+1)^d$ | |
| Gaussian kernel (a.k.a. Radial Basis Function) | $k(\mathbf{x_i},\mathbf{x_j})=\exp(-\gamma\|\mathbf{x_i}-\mathbf{x_j}\|^2)$ | $\gamma>0$. 때때로 $\gamma=\frac{1}{2\sigma^2}$ 로 파라미터화 |
| Hyperbolic tangent (a.k.a. Sigmoid Function) | $k(\mathbf{x_i},\mathbf{x_j})=\tanh(\kappa\mathbf{x_i}\cdot\mathbf{x_j}+c)$ | 일부(전부는 아님) $\kappa>0$, $c<0$ 에 대해 |

### 4.10 Polynomial Kernel Function [p.27]

$\mathbf{x}=\langle x_1,x_2\rangle$, $\mathbf{z}=\langle z_1,z_2\rangle$ 라 하면

- **degree 1**
  $$K(\langle x_1,x_2\rangle,\langle z_1,z_2\rangle)=\langle x_1,x_2\rangle\cdot\langle z_1,z_2\rangle=x_1z_1+x_2z_2=\mathbf{x}\cdot\mathbf{z}$$
- **degree 2**
  $$K(\langle x_1,x_2\rangle,\langle z_1,z_2\rangle)=\langle x_1^2,\sqrt2x_1x_2,x_2^2\rangle\cdot\langle z_1^2,\sqrt2z_1z_2,z_2^2\rangle$$
  $$=x_1^2z_1^2+2x_1x_2z_1z_2+x_2^2z_2^2=(x_1z_1+x_2z_2)^2=(\mathbf{x}\cdot\mathbf{z})^2$$
- **degree 3**: $K=(\mathbf{x}\cdot\mathbf{z})^3$
- **degree n**: $K=(\mathbf{x}\cdot\mathbf{z})^n$
- 커널 $K$를 알기 위해 **x와 z의 변환된 좌표값을 표현하고 계산할 필요가 있는가?**
  - 고차에서의 선형 분리를 이용하기 위해 feature space를 실제로 변환해야 하는가?
  - **조건: 이 트릭이 통하려면 내적(inner product)만 계산 가능하면 된다**

### 4.11 Dual SVM with Kernel Trick [p.28]

$$\max_{\alpha\ge0}\sum_j\alpha_j-\frac12\sum_i\sum_j\alpha_i\alpha_jy_iy_j\varphi(x_i)\varphi(x_j)$$
$$\Rightarrow \max_{\alpha\ge0}\sum_j\alpha_j-\frac12\sum_i\sum_j\alpha_i\alpha_jy_iy_jK(x_i,x_j)$$

- $\alpha_i\big((wx_j+b)y_j-1\big)=0,\ C>\alpha_i>0$
  - $\mathbf{w}=\sum_{i=1}^{N}\alpha_iy_i\varphi(\mathbf{x}_i)$
  - $b=y_j-\sum_{i=1}^{N}\alpha_iy_i\varphi(\mathbf{x}_i)\varphi(\mathbf{x}_j)$
- $\sum_{i=1}^{N}\alpha_iy_i=0$
- $C\ge\alpha_i\ge0,\ \forall i$
- **Dual 정식화가 SVM에 주는 이점**
  - **Kernel trick** — 추정할 파라미터 감소
  - w 대신 **alpha 값만 저장**
    - 몇 개의 alpha가 필요한가?
    - 의미 있는 alpha(≠0)만 고려 → **support vector**

### 4.12 Classification with SVM Kernel Trick [p.29]

| 케이스 | 판별식 | 최적화 |
|---|---|---|
| **Linear case** | $sign(w\cdot x+b)$ | $\min_{w,b}\|w\|$ s.t. $(wx_j+b)y_j\ge1,\forall j$ |
| **Transformed case** | $sign(w\cdot\varphi(x)+b)$ | $\min_{w,b,\xi_j}\|w\|+C\sum_j\xi_j$ s.t. $(w\varphi(x_j)+b)y_j\ge1-\xi_j,\ \xi_j\ge0$ |
| **Kernel trick case** | $sign(w\cdot\varphi(x)+b)$ | $\max_{\alpha\ge0}\sum_j\alpha_j-\frac12\sum_i\sum_j\alpha_i\alpha_jy_iy_jK(x_i,x_j)$ |

Kernel trick case의 부수 조건

- $\mathbf{w}=\sum_{i=1}^{N}\alpha_iy_i\varphi(\mathbf{x}_i)$
- $b=y_j-w\varphi(x_j)$ when $0<\alpha_j<C$
- $\sum_{i=1}^{N}\alpha_iy_i=0$
- $0\le\alpha_i\le C,\ \forall i$

**최종 판별식**

$$\begin{aligned}
sign(w\cdot\varphi(x)+b)&=sign\left(\sum_{i=1}^{N}\alpha_iy_i\varphi(\mathbf{x}_i)\cdot\varphi(x)+y_j-\sum_{i=1}^{N}\alpha_iy_i\varphi(\mathbf{x}_i)\varphi(\mathbf{x}_j)\right)\\
&=sign\left(\sum_{i=1}^{N}\alpha_iy_iK(\mathbf{x}_i,x)+y_j-\sum_{i=1}^{N}\alpha_iy_iK(\mathbf{x}_i,\mathbf{x}_j)\right),\quad 0<\alpha_j<C
\end{aligned}$$

### 4.13 SVM with Various Kernels [p.30]

- SVM은 kernel trick으로 **비선형 분리 케이스에 매우 잘 적응**한다
  - 고차원 피처로 **손쉽게(공짜로) 확장**
- 예: **Polynomial Kernel with Degree 4**, **RBF Kernel with $\gamma=4$**

### 4.14 Logistic Regression with Kernel [p.31]

- Logistic regression: $P(Y|X)=\dfrac{1}{1+e^{-\theta^Tx}}$, θ의 MLE를 찾음
- **logistic regression도 커널화할 수 있는가?**
  - $\mathbf{w}=\sum_{i=1}^{N}\alpha_iy_i\varphi(\mathbf{x}_i)$
  $$P(Y|X)=\frac{1}{1+e^{-\theta^Tx}}=\frac{1}{1+e^{\sum_{i=1}^N\alpha_iy_i\varphi(x_i)\varphi(x)+b}}=\frac{1}{1+e^{\sum_{i=1}^N\alpha_iy_iK(x_i,x)+b}}$$
- **문제가 바뀐다**: θ를 찾는 것에서 **$\alpha_i$를 찾는 것**으로
  - 이 문제를 어떻게 푸나?
  - 다시 말해… 이것은 제약 최적화인가? 아니라면 그것이 시사하는 바는?

## 5. Acknowledgement & Further Readings [p.32–33]

- 영향: Prof. Carlos Guestrin (CMU), Prof. Eric Xing (CMU)
- Further readings: **Bishop Chapter 7 → 6**

## Week 5 검토 노트

| 위치 | 내용 | 판정 |
|---|---|---|
| p.8 | $f(x)=wx_p+b+r\frac{w\cdot w}{\|w\|}=r\|w\|$ | $wx_p+b=0$, $\frac{w\cdot w}{\|w\|}=\|w\|$ 이므로 **정확** |
| p.9 | $\max 2a/\|w\|$ → a=1 정규화 → $\min\|w\|$ | **정확**. 통상 $\min\frac12\|w\|^2$로 쓰나 최소화 지점은 동일 |
| p.15 | $\xi_j=-\log P(Y_j\|X_j,w,b)=\log(1+e^{(wx_j+b)y_j})$ | **부호 오류**. $P=\frac{1}{1+e^{-(wx_j+b)y_j}}$ 이므로 $-\log P=\log(1+e^{-(wx_j+b)y_j})$ 가 맞음. 같은 슬라이드의 그래프(감소 곡선)는 정정본과 일치 |
| p.22 | "Active Constraint $\alpha^*=0\Rightarrow g(x^*)=0$" | **오류**. Active constraint는 $\alpha^*>0\Rightarrow g(x^*)=0$. ($\alpha^*=0$이면 오히려 inactive) |
| p.21 | Lagrange Dual Problem에 $\max_{\alpha>0}$, s.t. $\alpha>0$ | 통상 $\alpha\ge0$. 바로 아래 줄에서는 $\alpha\ge0$을 사용 → **원문 내 표기 불일치(경미)** |
| p.23 | $\min\|w\|$ 가 Lagrange에서 $\frac12 w\cdot w$ 로 바뀜 | 설명 없이 전환. 동일 최소해를 주므로 결과는 정확 |
| p.23 | $L=\frac12w\cdot w-\sum\alpha_j[(wx_j+b)y_j-1]$ | $g(x)=1-(wx_j+b)y_j\le0$ 이므로 $+\alpha g = -\alpha[(wx+b)y-1]$. **정확** |
| p.23 | KKT 나열에서 $\alpha_i$와 $x_j,y_j$ 인덱스 혼용 | 원문 인덱스 표기 오류(경미). $\alpha_j((wx_j+b)y_j-1)=0,\forall j$ 가 맞음 |
| p.24 | $\frac12\sum\sum - \sum\sum = -\frac12\sum\sum$ | **정확** |
| p.27 | degree 2 전개 $(x_1z_1+x_2z_2)^2$ | **정확** |
| p.28 | $b=y_j-\sum\alpha_iy_i\varphi(x_i)\varphi(x_j)$ | margin 위의 support vector($0<\alpha_j<C$)에 대해 성립. **정확** |
| p.19 | $\varphi$ 예시에 $x_1,x_2$ 1차항 포함 | 2차 다항 커널의 명시적 전개와는 다르나 예시로서 문제 없음 |
| p.29 | 판별식에 $y_j$와 $\sum\alpha_iy_iK(x_i,x_j)$ 가 남아 있음 | 이 둘이 곧 $b$. 표현이 장황하나 **정확** |

---

# Week 6 — Training/Testing and Regularization

## 1. 주차 목표 [p.2]

- **bias와 variance 개념** 이해
  - over-fitting과 under-fitting 개념
  - 오차의 두 원천(bias, variance)을 분리할 수 있을 것
- **bias-variance trade-off** 이해
  - **Occam's razor** 개념
  - **cross-validation** 수행 능력
  - 지도학습을 위한 다양한 성능 지표
- **regularization** 개념 이해
  - Linear regression / Logistic regression / Support vector machine 에 적용하는 법

## 2. Concept of Bias and Variance

### 2.1 Up To This Point… [p.4]

- 이제 분류(classification)에 대한 지식을 갖추었다: **Naïve Bayes, Logistic Regression, Support Vector Machine**
- SVM은 여전히 분류에 흔히 쓰이는 알고리즘
- **기능(Functioning)은 어느 정도 완료** — 이제 **효율성과 정확도가 문제**가 된다

### 2.2 Better Machine Learning Approach? [p.5]

- 정확한 예측 결과 — 예) "이 NB 분류기로 스팸을 **95% 정확도**로 거를 수 있다!"
- **이 주장이 옳은가?**
  - **정확도의 타당성**
    - 명확한 정의가 없음
    - Precision/Recall, F-Measure 같은 다른 지표는 왜 안 쓰나?
  - **데이터셋의 타당성**
    - 스팸이란?? 스팸이 몇 개나?
    - 어디서 수집했나?
    - 스팸 안의 분산이 큰가?
    - **스팸 메일이 진화하고 있는가?** (Nigerian prince scheme에서 다른 무언가로?)

### 2.3 Training and Testing [p.6]

**Training**

- 파라미터 추론 절차
- 사전지식, 과거 경험
- **이것이 미래에도 통한다는 보장은 없다**
  - ML의 아킬레스건은 **학습 대상의 안정적/정적 분포** 가정
- ML이 미래에 통하지 않는 이유
  - 도메인이 변하거나, 현재 도메인이 충분한 분산을 보이지 않음
  - ML 알고리즘 자체에 내재적 문제가 있음

**Testing**

- 학습된 ML 알고리즘/추론된 파라미터를 테스트
- 학습 과정과 **무관한 새 데이터셋**
- **미래 인스턴스를 모방** — 관측의 부분집합을 따로 떼어놓음으로써
- (그림: Training set = patients/healthy subjects, Test set = normal/abnormal, separating plane)

### 2.4 Over-Fitting and Under-Fitting [p.7]

- 시나리오: N개의 점으로 ML 알고리즘 학습, 단순 다항 회귀 함수 $Y=F(x)$ 를 학습
  - **F의 차수는 미정** — 선형일 수도, 비선형일 수도
- 아래 세 F 중 어느 것이 더 나아 보이는가?
  - **Underfitting** (직선) / **Just right!** (2차) / **overfitting** (고차 진동)

### 2.5 Tuning Model Complexity [p.8]

- 1차, 2차, N차 학습 함수 — 차수가 올라갈수록 모델이 복잡해짐
- **복잡한 모델이 더 나은가?**
- 그렇다면 복잡한 모델을 개발하다가 **어디서 멈춰야 하나?**
  - 복잡도와 일반성을 계산할 척도가 있는가?
- **모델의 복잡도와 데이터셋의 일반성 사이에는 trade-off가 있다**

### 2.6 Sources of Error in ML [p.9]

- 오차의 원천은 **두 갈래**: **approximation과 generalization**
  $$E_{out}\le E_{in}+\Omega$$
  - $E_{out}$: 학습된 ML 알고리즘의 추정 오차 (회귀 케이스 기준)
  - $E_{in}$: 학습 알고리즘의 **근사(approximation)** 에서 오는 오차
  - $\Omega$: **관측의 분산(variance)** 에 의해 야기되는 오차
- 기호 정의
  - **f**: 학습할 목표 함수
  - **g**: ML의 학습 함수
  - **$g^{(D)}$**: 데이터셋 D를 사용해 학습된 함수, 또는 가설의 한 인스턴스
  - **D**: 실세계에서 뽑은 가용 데이터셋
  - **$\bar g$**: 무한개의 D가 주어졌을 때의 **평균 가설**
    - 형식적으로 $\bar g(x)=E_D[g^{(D)}(x)]$

### 2.7 Bias and Variance [p.10]

데이터셋 D의 한 인스턴스에 대한 오차:

$$E_{out}(g^{(D)}(x))=E_X\left[\left(g^{(D)}(x)-f(x)\right)^2\right]$$

무한개 데이터셋 D에 대한 기대 오차:

$$E_D[E_{out}(g^{(D)}(x))]=E_D\left[E_X\left[\left(g^{(D)}(x)-f(x)\right)^2\right]\right]=E_X\left[E_D\left[\left(g^{(D)}(x)-f(x)\right)^2\right]\right]$$

내부 항 정리:

$$\begin{aligned}
E_D\left[\left(g^{(D)}(x)-f(x)\right)^2\right]&=E_D\left[\left(g^{(D)}(x)-\bar g(x)+\bar g(x)-f(x)\right)^2\right]\\
&=E_D\left[(g^{(D)}(x)-\bar g(x))^2+(\bar g(x)-f(x))^2+2(g^{(D)}(x)-\bar g(x))(\bar g(x)-f(x))\right]\\
&=E_D\left[(g^{(D)}(x)-\bar g(x))^2\right]+(\bar g(x)-f(x))^2+E_D\left[2(g^{(D)}(x)-\bar g(x))(\bar g(x)-f(x))\right]
\end{aligned}$$

$$E_D\left[2(g^{(D)}(x)-\bar g(x))(\bar g(x)-f(x))\right]=0\quad (\because \bar g(x)\text{의 정의})$$

따라서 최종적으로:

$$\boxed{E_D[E_{out}(g^{(D)}(x))]=E_X\left[E_D\left[\left(g^{(D)}(x)-\bar g(x)\right)^2\right]+\left(\bar g(x)-f(x)\right)^2\right]}$$

### 2.8 Bias and Variance Dilemma [p.11]

정의:

$$\text{Variance}(x)=E_D\left[\left(g^{(D)}(x)-\bar g(x)\right)^2\right]$$
$$\text{Bias}^2(X)=\left(\bar g(x)-f(x)\right)^2$$

**의미**

- **Variance**: 데이터셋의 한계 때문에 모델을 **평균 가설로 학습시키지 못하는 무능력**
- **Bias**: 평균 가설이 **실세계에 부합하도록 학습시키지 못하는 무능력**

**bias와 variance를 줄이는 법**

- Variance 줄이기 → **데이터를 더 모은다**
- Bias 줄이기 → **더 복잡한 모델**

**그러나 bias를 줄이면 variance가 커지고, 그 반대도 성립** → **Bias and Variance Dilemma**

## 3. Performance Measurement

### 3.1 Empirical Bias and Variance Trade-off [p.13]

- 설정
  - $f(x)=\sin(2\pi x)$
  - $D=\{$두 점 $\mid$ point$=(x,\sin(2\pi x))$, $0\le x\le 1\}$
  - 두 개의 $g(x)$: **Zero degree**(짙은 회색 선), **One degree**(옅은 회색 선)
  - 두 개의 $\bar g(x)$: **Zero degree**(빨간 선), **One degree**(초록 선)
- **1차와 0차 중 어느 쪽이 bias가 크고, 어느 쪽이 variance가 큰가?**

### 3.2 Bias and Variance of Two Hypotheses [p.14]

| 모델 | Bias | Var. |
|---|---|---|
| Zero degree (단순 모델) | **0.5051** | **0.2410** |
| One degree (복잡 모델) | **0.3092** | **2.0708** |

- **복잡한 모델은 variance가 높고 bias가 낮다**
- **단순한 모델은 variance가 낮고 bias가 높다**
- **ML 알고리즘의 복잡도에서 균형이 필요하다**

### 3.3 Occam's Razor [p.15]

- **Occam's Razor**: 경쟁하는 가설 중 **가장 적은 가정을 하는 것**을 선택해야 한다
- **Competing?** → 예측 오차가 상당히 유사함
- **Fewest assumption?** → 덜 복잡한 모델
- **대략 같은 오차라면 단순한 모델을 선택해야 한다**
- **Bias and Variance tradeoff의 반영!**
  - 그런데, 실세계 환경에서 bias와 variance를 계산하는 것이 가능한가?

### 3.4 Cross Validation [p.16]

- 목표 함수로부터 관측된 **무한개의 표본은 없다**
- 무한 표본추출을 **모방**해야 한다
  - bias-variance tradeoff에서 표본추출 횟수가 쓰인 곳은? → $\bar g$: 무한개 D의 평균 가설, $\bar g(x)=E_D[g^{(D)}(x)]$
- **고정된 개수의 데이터셋으로부터 많은 데이터셋을 만들어야 한다**
- **N-fold cross validation**
  - 주어진 인스턴스 집합을 **N개의 배타적 부분집합**으로 나눈다
  - **(N-1)개 부분집합을 training에 사용**
  - **1개 부분집합을 testing에 사용**
- 특수 케이스: **LOOCV (Leave One Out Cross Validation)** — N-fold cross validation의 극단적 경우

### 3.5 Performance Measure of ML [p.17]

- bias와 variance를 계산할 수 있는가?
  - **목표 함수 f(X)를 모른다!**
  - **평균 가설 $\bar g(x)$를 계산할 수 없다!**
- 따라서 **bias와 variance를 성능 척도로 쓸 수 없다**
- 그럼 어떤 척도를 쓰나?
  - **Accuracy = (TP+TN) / (TP+FP+FN+TN)**
  - **Precision and Recall**
  - **F-Measure**
  - **ROC curve**

**Confusion Matrix**

| | | Actual Value: True | Actual Value: False |
|---|---|---|---|
| **Estimated Value** | Positive | ① True Positive | ② False Positive |
| | Negative | ③ False Negative | ④ True Negative |

### 3.6 Precision and Recall [p.18]

두 가지 케이스 — 분류기 구축: **Spam filter**, **CRM**

**목표가 조금씩 다르다**

- **Spam filter**: 스팸 분류
  - **안전이 우선**. 유효한 이메일이 스팸으로 추정되어 버려지는 것을 원치 않음
  - **FP를 줄이는 것이 우선**
- **CRM**: VIP 고객 분류
  - **접촉(reaching out)이 우선**. VIP 고객을 일반 고객으로 놓치는 것을 원치 않음
  - **FN을 줄이는 것이 우선**

$$\textbf{Precision}=\frac{TP}{TP+FP},\qquad \textbf{Recall}=\frac{TP}{TP+FN}$$

- 각 케이스에 어떤 지표를 쓸 것인가? (Spam filter → Precision, CRM → Recall)

### 3.7 F-Measure [p.19]

- Precision과 recall은 인기 있는 지표지만 응용에서 문제가 있다
  - **가장 안전한 스팸 필터 == 항상 '스팸 아님'이라고 말하는 것**
  - **가장 잘 접촉하는 고객 필터 == 항상 'VIP'라고 말하는 것**
- precision과 recall 성능을 **균형** 잡는 척도가 필요
- **F-Measure**는 precision과 recall로부터 유도된 지표

$$F_b\text{-Measure}=(1+b^2)\times\frac{Precision\times Recall}{b^2\times Precision+Recall}$$
$$F_1\text{-Measure}=2\times\frac{Precision\times Recall}{Precision+Recall}$$

- $F_{0.5}$와 $F_2$도 사용됨
  - **$F_2$는 recall을 강조**
  - **$F_{0.5}$는 precision을 강조**

## 4. Model Regularization

### 4.1 Concept of Regularization [p.21]

- **분산 측면에서의 재앙(Disaster in terms of variance)**
- **regularization을 하면**
  - **완벽한 적합을 희생** → 학습 정확도를 낮춤
  - **테스트에서의 잠재적 적합도를 높임**
  - 모델 복잡도 때문에 **bias가 조금 증가하는 경향**
  - 결국 regularization은 **모델에 대한 또 하나의 제약**
    - 기존 제약은? → **학습 셋에서의 오차 최소화**
- **MSE에 새로운 항을 추가한다**

### 4.2 Formal Definition of Regularization [p.22]

- Regularization은 회귀에 대한 **또 하나의 제약**
  - 아래의 J(B)가 최소화할 regularization function
  - B는 **상수항을 제외한** 회귀 모델의 가중치
- 다양한 regularization이 존재
  - **L1 Regularization == Lasso regularization** — the first order
  - **L2 Regularization == Ridge regularization** — the second order
  - regularization 항의 **차수(order)** 에 따라 달라지며, 차수가 **손실함수의 모양**을 결정
- 식

$$\textbf{L2 (Ridge)}:\quad E(w)=\frac12\sum_{n=0}^{N}(train_n-g(x_n,w))^2+\frac{\lambda}{2}\|w\|^2$$
$$\textbf{L1 (Lasso)}:\quad E(w)=\frac12\sum_{n=0}^{N}(train_n-g(x_n,w))^2+\lambda|w|$$

- 제약 영역의 모양: **Lasso**(마름모) / **Ridge**(원) / **Bridge**(오목 별 모양)

### 4.3 Regularization of Linear Regression [p.23]

$$E(w)=\frac12\sum_{n=0}^{N}(train_n-g(x_n,w))^2+\frac{\lambda}{2}\|w\|^2$$

**w를 closed form으로 계산 가능**

$$\frac{d}{dw}E(w)=0$$
$$\begin{aligned}
\frac{d}{dw}E(w)&=\frac{d}{dw}\left(\frac12\|train-Xw\|^2+\frac{\lambda}{2}\|w\|^2\right)\\
&=\frac{d}{dw}\left(\frac12\|train-Xw\|^T\|train-Xw\|+\frac{\lambda}{2}w^Tw\right)\\
&=\frac{d}{dw}\left(\frac12(train^Ttrain-2X^Tw\cdot train+X^TXw^Tw)+\frac{\lambda}{2}w^Tw\right)\\
&=\frac{d}{dw}\left(train^Ttrain-X^Tw\cdot train+\frac12X^TXw^Tw+\frac{\lambda}{2}w^Tw\right)\\
&=-X^T\cdot train+X^TXw+\lambda w=0
\end{aligned}$$

$$-X^T\cdot train+X^TXw+\lambda Iw=0$$
$$-X^T\cdot train+(X^TX+\lambda I)w=0$$
$$(X^TX+\lambda I)w=X^T\cdot train$$
$$\boxed{w=(X^TX+\lambda I)^{-1}X^T\cdot train}$$

### 4.4 Effect of Regularization [p.24]

| | Bias | Var. |
|---|---|---|
| regularization 없음 (1차 모델) | 0.3092 | **2.0708** |
| **λ = 1** 적용 | **0.4372** | **0.1167** |

- **λ = 1일 때**
  - **bias는 조금 증가**
  - **variance는 크게 감소**

### 4.5 Optimizing the Regularization [p.25]

λ = 0 / λ = 1 / λ = 100 의 세 경우 비교

- **λ를 최적화해야 한다**
  - **너무 낮은 λ**: variance가 너무 높음 → **정규화 안 한 모델처럼 동작**
  - **너무 높은 λ**: variance가 너무 낮음 → **덜 복잡한 모델처럼 동작**, **1차 모델을 상수 모델로 바꿔버림**
- **λ를 어떻게 최적화하나?** (→ cross validation)

### 4.6 Regularization of Logistic Regression [p.26]

- Regularization은 **logistic regression 등 다른 모델에도 적용 가능**
- θ를 찾는 **closed form과 approximate form**을 탐색할 수 있음

$$\arg\max_\theta\sum_{i=1}^{m}\log p(y_i|x_i,\theta)-\alpha R(\theta)$$

$$\textbf{L1}:\ R(\theta)=\|\theta\|_1=\sum_{i=1}^{n}|\theta_i|$$
$$\textbf{L2}:\ R(\theta)=\|\theta\|_2^2=\sum_{i=1}^{n}\theta_i^2$$

### 4.7 Regularization and SVM [p.27]

$$f=\arg\min_{f\in\mathcal{H}}\left\{\frac1n\sum_{i=1}^{n}V(y_i,f(x_i))+\lambda\|f\|_{\mathcal{H}}^2\right\}$$

hinge loss를 대입하면:

$$V(y_i,f(x_i))=(1-yf(x))_+,\qquad (s)_+=\max(s,0)$$
$$f=\arg\min_{f\in\mathcal{H}}\left\{\frac1n\sum_{i=1}^{n}(1-yf(x))_++\lambda\|f\|_{\mathcal{H}}^2\right\}$$
$$f=\arg\min_{f\in\mathcal{H}}\left\{C\sum_{i=1}^{n}(1-yf(x))_++\frac12\|f\|_{\mathcal{H}}^2\right\},\qquad C=\frac{1}{2\lambda n}$$

- **Support vector (machine)는 hinge loss를 쓰는 regularization의 특수한 경우이다**
- (그래프: misclassification loss(파랑, 0-1) vs hinge loss(초록))

## 5. Acknowledgement [p.28]

- 영향: Prof. Eric Xing (CMU)
- (Week 6에는 Further Readings 슬라이드 없음)

## Week 6 검토 노트

| 위치 | 내용 | 판정 |
|---|---|---|
| p.10 | 교차항 $E_D[2(g^{(D)}-\bar g)(\bar g-f)]=0$ | $\bar g=E_D[g^{(D)}]$ 이므로 $E_D[g^{(D)}-\bar g]=0$. **정확** |
| p.10 | 최종 분해 $E_D[E_{out}]=E_X[\text{Var}+\text{Bias}^2]$ | **정확** (노이즈 항이 없는 것은 $f$를 결정론적 목표로 두었기 때문) |
| p.9 | $E_{out}\le E_{in}+\Omega$ (부등식) vs p.10의 등식 분해 | 두 식은 서로 다른 층위(VC-style bound vs bias-variance 분해). 원문은 이를 명시적으로 구분하지 않아 **다소 혼동 소지** |
| p.14 | 0차: Bias 0.5051 / Var 0.2410, 1차: Bias 0.3092 / Var 2.0708 | 본문 주장("복잡=고분산·저편향")과 **수치 정합** |
| p.24 | λ=1 → Bias 0.3092→0.4372, Var 2.0708→0.1167 | 본문 주장("bias 소폭↑, variance 대폭↓")과 **수치 정합** |
| **p.21** | "**Because of the increased model complexity**, the bias tends to increase a little bit" | ⚠️ **원문 서술 오류**. regularization은 유효 모델 복잡도를 **낮추므로** bias가 증가한다. p.25의 "Too high λ → works like a less complex model"과 **모순**. → "**decreased** model complexity" 로 읽어야 함 |
| p.22 | "B is the weight of the regression model **except the constant term**" | 정확한 관행 (절편은 보통 정규화하지 않음). 단 p.23 유도에서는 $\|w\|^2$ 전체에 적용 |
| p.22 | "The below J(B) is the regularization function" | 슬라이드에 J(B)라는 기호가 실제로 등장하지 않음. **표기 불일치(경미)** |
| p.23 | $\|train-Xw\|^2 = \|train-Xw\|^T\|train-Xw\|$ | 노름(스칼라)에 전치를 쓴 **표기 오류**. $(train-Xw)^T(train-Xw)$ 가 맞음 |
| p.23 | 전개 중 $X^TXw^Tw$ | $w^TX^TXw$ 의 표기 오류. 최종 결과 $w=(X^TX+\lambda I)^{-1}X^T train$ 는 **정확** |
| p.27 | $C=\frac{1}{2\lambda n}$ | $\frac1n\sum V+\lambda\|f\|^2$ 의 양변에 $\frac{1}{2\lambda}$ 를 곱하면 $\frac{1}{2\lambda n}\sum V+\frac12\|f\|^2$. **정확** |
| p.27 | "Support vector is a special case of regularization" | "Support vector **machine**"의 축약. 의미상 정확 |
| p.19 | $F_b=(1+b^2)\frac{PR}{b^2P+R}$ | 표준 $F_\beta$ 정의와 **일치**. $b=1$ 대입 시 $F_1$ 도 일치 |
| p.19 | "$F_2$는 recall 강조, $F_{0.5}$는 precision 강조" | **정확** |
| p.17 | Confusion matrix (행=추정, 열=실제) | TP/FP/FN/TN 배치 **정확**. Accuracy 식도 정확 |

---

# 전체 통합 검토 요약

## 1) 주차 간 일관성 — 모순 없음 확인

| 개념 | 재등장 위치 | 판정 |
|---|---|---|
| MLE/MAP 압정 공식 | W1 p.18/p.25 → W3 p.6 | **완전 일치** |
| Supervised Learning 정의 슬라이드 | W1 p.11 → W3 p.4 | **동일** |
| EnjoySport 데이터셋 | W2 p.5,9,11–14,17 → W3 p.12,16 | **동일 (4행)** |
| Credit Approval 카운팅 | W2 p.18 → W4 p.5 | **동일 수치** |
| Decision boundary / Bayes risk | W3 p.7–9 → W4 p.4 → W5 p.4–5 | **동일** |
| 선형회귀 closed form $\theta=(X^TX)^{-1}X^TY$ | W2 p.28 → W4 p.9,12,19 → W6 p.23(정규화판) | **일관** |
| Logistic 식 $P(Y\|X)=\frac{1}{1+e^{-\theta^Tx}}$ | W4 p.10,24 → W5 p.31 → W6 p.26 | **일관** |
| Hinge loss $\xi_j=(1-(wx_j+b)y_j)_+$ | W5 p.13,14,15 → W6 p.27 | **일관** |
| 함수 근사 관점의 ML | W1 p.21 → W2 p.6,26,30 → W6 전반 | **일관** |

## 2) 발견된 오류 — 심각도 순

| # | 위치 | 유형 | 내용 | 정정 |
|---|---|---|---|---|
| 1 | **W4 p.23–24** | **수식 오류(중대)** | GNB→LR 유도에서 지수부 **부호**가 반대이고, 상수항이 ${\mu_2}^2-{\mu_2}^2$(=0) | $+\frac{1}{2(\sigma^i)^2}\sum\{2(\mu_2^i-\mu_1^i)X_i+{\mu_1^i}^2-{\mu_2^i}^2\}+\log\pi_2-\log\pi_1$ |
| 2 | **W5 p.22** | **개념 오류** | "Active Constraint: $\alpha^*=0\Rightarrow g(x^*)=0$" | $\alpha^*>0\Rightarrow g(x^*)=0$ |
| 3 | **W5 p.15** | **부호 오류** | $\xi_j=\log(1+e^{(wx_j+b)y_j})$ | $\xi_j=\log(1+e^{-(wx_j+b)y_j})$ (같은 슬라이드 그래프는 정정본과 일치) |
| 4 | **W6 p.21** | **서술 모순** | "increased model complexity 때문에 bias 증가" | regularization은 유효 복잡도를 **낮춤** → "decreased". p.25와 모순 |
| 5 | **W3 p.9 / W5 p.5** | 상수 부정확 | 2차원 다변량 정규 정규화 상수 $\frac{1}{\sqrt{2\pi\|\Sigma\|}}$ | $\frac{1}{2\pi\|\Sigma\|^{1/2}}$ (일반형 $(2\pi)^{-d/2}\|\Sigma\|^{-1/2}$) |
| 6 | **W3 p.18** | 표기 오류 | $P(Y=tail)=0$ | $P(X=tail\|Y)=0$ (Y는 클래스) |
| 7 | **W6 p.23** | 표기 오류 | $\|train-Xw\|^T\|train-Xw\|$, $X^TXw^Tw$ | $(train-Xw)^T(train-Xw)$, $w^TX^TXw$ — **최종 답은 정확** |
| 8 | **W2 p.27** | 표기 오류 | 행렬 X의 모든 행이 $x_n^D$ | 각 행은 서로 다른 인스턴스 |
| 9 | **W1 p.34** | 표기 오류 | $f(\theta;n,p)$ | $f(k;n,p)$ (W1 p.16은 올바름) |
| 10 | **W2 p.8** | 인덱스 불일치 | 인스턴스 $x_1,x_2,x_4$ ↔ 가설 $h_{1,2,3}$, $h_{1,2,3,4}$ | 슬라이드 그림상 $x_3$(negative)는 건너뜀 |
| 11 | 다수 | 단순 오타 | W2 p.2 "appraoch", W4 p.2 "tailor expansion", W4 p.23 "++logπ2", W5 p.21 $\alpha>0$ vs $\alpha\ge0$ | — |

## 3) 수치 검증 통과 항목

- W2 p.18 Credit Approval 카운팅: 690 = 307(+) + 383(-), A1/A9 분할 합계 모두 정합
- W2 p.21 오분류 100건 → 85.5% (590/690 = 85.51%)
- W2 p.22 오분류 99건 → 85.6% (591/690 = 85.65%)
- W4 p.16 Rosenbrock: $\partial f/\partial x_1=-415.4$, $\partial f/\partial x_2=-158$, $\|f'\|=444.4335$, $\mathbf x^1=(-1.2991, 0.9004)$ 모두 정합
- W4 p.24 NB 파라미터 수 $2\times2\times d+1=4d+1$ 정합
- W6 p.14 / p.24 bias·variance 수치와 본문 주장 정합
- W6 p.27 $C=1/(2\lambda n)$ 정합
- W5 p.8 마진 거리 $r=f(x)/\|w\|$ 유도 정합

## 4) 참고 문헌 총괄

| 주차 | Further Readings |
|---|---|
| Week 1 | Bishop Ch. 1, 2 |
| Week 2 | Bishop Ch. 14.4 / Mitchell Ch. 1, 2, 3 |
| Week 3 | Bishop Ch. 1, 8.2 |
| Week 4 | Bishop Ch. 4.3, 5.2.1–5.2.4 |
| Week 5 | Bishop Ch. 7 → 6 |
| Week 6 | (없음) |

**Acknowledgement 총괄**: Prof. Carlos Guestrin (CMU), Prof. Eric P. Xing (CMU), Prof. Tom Mitchell (CMU)

**인용 논문**

- Shinjae Yoo, Yiming Yang, Frank Lin, Il-Chul Moon, *Mining Social Networks for Personalized Email Prioritization*, ACM SIGKDD, Paris, France, Jun 28, 2009
- Il-Chul Moon, Alice H. Oh, Kathleen M. Carley, *Analyzing Social Media in Escalating Crisis Situations*, IEEE ISI 2011, pp. 71–76, Beijing, China, Jul 10–12, 2011
- W.S. Lee, J.Y. Bak, A.H. Oh, I.-C. Moon, *Media Analysis to Estimate Stock Market Movement*, Information Fusion (Major Revision)
- Ng & Jordan, 2002 (generative-discriminative pair)

**데이터셋**

- UCI Credit Approval: http://archive.ics.uci.edu/ml/datasets/Credit+Approval
- UCI Housing: http://archive.ics.uci.edu/ml/datasets/Housing
- Amazon 리뷰 bag-of-words: 198 documents, 29,717 unique words
