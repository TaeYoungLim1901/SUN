from math import sqrt, erf
def phi(z): return 0.5*(1+erf(z/sqrt(2)))
pub_f55, pub_cf = 55.5576, 55.5632
d_obs = pub_cf - pub_f55                 # +면 cardF 가 Public 에서 나쁘다
pred  = 55.49 - pub_f55                  # 카드 굽기 때 예측한 delta
sp    = 0.031                            # 근친 카드 Public 짝잡음 (실측)
for prior_sd in (0.048, 0.070, 0.10, 0.20):
    w1, w2 = 1/prior_sd**2, 1/sp**2
    post_mu = (pred*w1 + d_obs*w2)/(w1+w2)
    post_sd = 1/sqrt(w1+w2)
    e_total = d_obs + post_mu            # Public+Private 합의 차이 (cardF - flow55)
    sd_total = sqrt(post_sd**2 + sp**2)
    print(f"prior_sd={prior_sd:.3f}  post_delta={post_mu:+.4f}  E[sum diff]={e_total:+.4f}"
          f"  P(cardF 승)={phi(-e_total/sd_total)*100:.0f}%")
