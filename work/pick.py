"""flow55 냐 cardF 냐 — Private 을 모르는 채로 고르는 계산.

합산 = Public(관측·확정) + Private(미지). 고르는 행위는 곧 Private 차이에 거는 것이다.
Private 의 유일한 동일분포 표본이 Public 이다. val 은 test 와 adversarial AUC 1.0 —
완전히 갈라지는 다른 분포라, 사전으로 쓰되 폭을 실측 오차로 넓혀야 한다.
"""
from math import sqrt, erf

def phi(z): return 0.5 * (1 + erf(z / sqrt(2)))

PUB = {"flow55": 55.5576, "cardF": 55.5632}       # 낮을수록 좋다
d_obs = PUB["cardF"] - PUB["flow55"]              # +0.0056 = cardF 가 Public 에서 나쁘다
d_pred = 55.49 - PUB["flow55"]                    # 굽기 때 val 로 예측한 차이 −0.0676
SP = 0.031                                        # 근친 카드 Public 짝잡음 (실측)

# val→Public 예측 오차 실측. 둘 다 「실제가 예측보다 나쁘다」 방향이다.
ERR = [55.7603 - 55.53,   # CARD1  +0.230
       55.5632 - 55.49]   # cardF  +0.073
BIAS = sum(ERR) / len(ERR)
RMS = sqrt(sum(e * e for e in ERR) / len(ERR))

def decide(mu, sd, tag):
    w1, w2 = 1 / sd**2, 1 / SP**2
    post_mu, post_sd = (mu * w1 + d_obs * w2) / (w1 + w2), 1 / sqrt(w1 + w2)
    tot = d_obs + post_mu
    p = phi(-tot / sqrt(post_sd**2 + SP**2))
    print(f"{tag:34s} E[합산 차]={tot:+.4f}  P(cardF 승)={p*100:.0f}%  → {'cardF' if p>0.5 else 'flow55'}")

print(f"val→Public 예측 오차: {ERR} → 편향 {BIAS:+.3f}, RMS {RMS:.3f}\n")
decide(d_pred, 0.048, "굽기 예측을 액면 그대로")
decide(d_pred, 0.070, "절반분할 sigma 로 넓혀")
decide(d_pred, RMS,   "실측 예측오차 폭으로 넓혀")
decide(d_pred + BIAS, RMS, "폭 + 낙관편향까지 보정 (최선)")

if __name__ == "__main__":
    assert d_obs > 0 and RMS > 0.15, "관측 부호와 오차 폭이 바뀌면 결론을 다시 계산해라"
