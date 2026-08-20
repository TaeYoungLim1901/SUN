# Q3 last-0.5 RMSE. CLOSED: 32-seed OF-CNN ρ≥0.988, seed-bag, 128px. Public n=1934, 8 slots, 273/700MB.
# r | move | what it changes | ΔRMSE this blend | cost 1×A100+2CPU | URL
1 NEW ARCH/TARGET (scalar-in-CNN or new loss; NOT another OF-CNN). Swap 1 member iff val ρ<0.95.
  Blend r√((1+ρ)/2): ρ=0.988→−0.2; ρ=0.90→−1.6. Need ρ<0.95. Cost: 8seed×8min GPU, +<50MB, infer≈same.
  https://arxiv.org/abs/1912.02757 Fort: random-init modes >> same-traj (SWA/snapshot/dropout).
  https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/ Deotte 75/500 L1, 4 targets, CV11.54/priv11.44.
  https://www.kaggle.com/competitions/cassava-leaf-disease-classification/writeups/golddiggaz-1st-place-solution 4 diverse; submitted higher-CV not higher-public.
2 HILL-CLIMB BY RESIDUAL ρ. Keep weaker member iff OOF Δ<0; drop ρ≥0.95. No free-weight re-fit (CLOSED).
  Caruana greedy+replacement=weights. Deotte GPU-HC 1st. S5E12: HC raised CV, Public stuck. Cost: 2CPU min, 0 GPU/MB.
  https://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml04.icdm06long.pdf
  https://www.kaggle.com/competitions/playground-series-s5e5/writeups/chris-deotte-1st-place-gpu-hill-climbing
  https://www.kaggle.com/competitions/playground-series-s5e12/writeups/1st-place-solution-hill-climbing-ridge-ensembl
3 SWA (avg last 25% of 1 train; replace last ckpt). 0 extra infer/disk. Subspace of ONE mode (Fort).
  C100 WRN 80.82→81.46 / R164 78.49→79.77 @1 budget; ImageNet +0.6–0.8pp/10ep. Here −0.02..−0.08 blend.
  Cost: +0.25×train/seed, 0 MB, 0 infer. https://arxiv.org/abs/1803.05407 https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/
4 MULTI-SAMPLE DROPOUT M=8 at GRU/head (train only). Infer=1 forward.
  C10 7.95→7.38; C100 30.9→29.8; ImageNet ns 27.5→27.4. +2%iter VGG / +17% small. Here −0.05 CNN if dropout else 0.
  Cost: +<20% train, 0 MB, 0 infer. https://arxiv.org/abs/1905.09788
5 NORM/TARGET DIVERSITY of same 96px (per-seq vs per-frame vs log-EUV). NOT 128px (CLOSED ρ≥0.988).
  APTOS 1st: 4arch×2seed 384+512; ens 0.844 vs 0.831 pub; kept 512 though public unrewarded.
  Landmark 1st: 384–896; ens 0.513 vs 0.476 priv. Clouds multi-scale TTA +0.004 cv. Here −0.1 iff ρ drops.
  Cost: 4–8 seeds×8min, +≤50MB. https://www.kaggle.com/competitions/aptos2019-blindness-detection/writeups/guanshuo-xu-1st-place-solution-summary
  https://github.com/ChristofHenkel/kaggle-landmark-2021-1st-place https://www.kaggle.com/c/understanding_cloud_organization/discussion/117987
6 SNAPSHOT/FGE (cosine cycles, save M minima, avg preds). Same-mode vs independent seeds (Fort).
  Huang: DenseNet C100 19.25→17.41 @1 budget; ImageNet R50 24.01→23.33 (M=2). Snapshot-6 ≈ 2 true models.
  32 independent seeds already dominate. M×infer+M×weights: 273MB → M≥3 blows 700MB. Δ~0. Skip unless replacing bag.
  https://arxiv.org/abs/1704.00109 https://www.kaggle.com/c/bengaliai-cv19/discussion/136815 snapshot beat last-cycle private.
7 SEED-BAG 32→64. Var=r²(ρ+(1−ρ)/k); ρ=0.988: k=32→64 Δ√≈9e-5 → −0.006 RMSE. CLOSED.
  30dML 1st: 20 XGB seeds/HP (tabular; NN seeds saturate faster). Cost: +4.3h GPU, ×2 disk. Do not.
  https://www.kaggle.com/competitions/30-days-of-ml/writeups/kaggle-swags-1st-place-solution https://arxiv.org/abs/1612.01474 Deep Ensembles: 5–10 nets capture most.
8 PUBLIC n=1934. iid SE(RMSE)≈σ/√(2n)≈56/62.2≈0.90. 11-pair transfer resid σ=0.199 (time-corr).
  −0.1 public=noise; −0.5≈2.5σ on transfer. 8 slots: ≤1 trust-LB, rest trust CV. Spend 0 slots on 0.1 probes.
  Blum-Hardt holdout-overfit. Roelofs: avg Kaggle shakeup small, small-n still bites. Polymer/MCTS: 1 CV + 1 mild-LB.
  https://arxiv.org/abs/1502.04585 https://arxiv.org/abs/1907.06721
  https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/writeups/1st-place-solution
  https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/writeups/james-day-1st-place-solution
# Verdict: only #1 with ρ<0.95 buys −0.4 public. #2 gates it. #3–4 cheap/tiny. #5–7 closed/waste. #8 slot policy.
