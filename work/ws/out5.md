# Q5 HSS PEAK amplitude from 193/211 only. CLOSED: posthoc q-map/affine/piecewise, huber, HSS-w. Metric=timeline RMSE. Fast (lw≥550) tracks public.
# 1 GEOMETRY → PEAK (not mean). Best legal: Δθ then A×colat. CM-slice A is a 4d SPEED profile.

Vršnak 2007: CH area in CM ±10° vs HSS *peak* cc=0.62 (100d 2005). Polar CH inclusion changes cc. EUV, no mag. LEGAL https://doi.org/10.1007/s11207-007-0285-8
Rotter 2015 AIA-193 ±7.5° slice, 156 peaks cc≈0.60, 80% arrival ±0.5d. Quote: OLS “underestimate and smooth out peak values”; they fit 3-CR min/max not LS. LEGAL 193 https://arxiv.org/abs/1501.06697
Rotter 2012: predicted vs observed *peak* v cc=0.78 (area-based). LEGAL https://doi.org/10.1007/s11207-012-0101-y
Hofmeister 2018 n=115, 193Å: *peak* v=478+(2.77e-9 A_km2)(1−|φ_co|/61.4). A-only Spearman 0.50; A×colat Pearson 0.70 / RMSE 70 / MAE 57. |φ|≳61.4° HSS misses Earth. LEGAL (pixel colat). B0/φ_sat ILLEGAL https://arxiv.org/abs/1804.09579
Garton 2018: *peak* vmax=330.8+5.7 Δθ (Δθ<67°) else ~710 (σ~50). Duration Δt≈0.09Δθ[d]. **Δθ is the PEAK predictor** (area saturates). CHIMERA=171/193/211 → 171 **ILLEGAL**; compute Δθ from 193∧211. https://arxiv.org/abs/1811.11605
Reiss 2016 ESWF 193: RMSE 108, CC 0.31; hits Δv=32±118, Δt=0.08±0.89d; both ESWF+WSA **under-rep v>350 by ~100–200**. WSA=mag **ILLEGAL**. https://arxiv.org/abs/1607.07048
Samara 2022 CATCH 193 n=45: PEAK vs **long-ext cc=0.57** (best), area 0.50, lat 0.41. Area saturates ≳6.4e10 km² at ~630. Aspect<1 long-ext cc=0.77. LEGAL 193. IMF-polarity pairing / B0 **ILLEGAL** as feats. https://arxiv.org/abs/2204.00368
Collin 2025: 4×3 grid best *timeline* RMSE 68.1; **10×10 (lat cells) best peak** RMSE 98.9. Peak needs location not just A. 27d-V + SSN **ILLEGAL**. https://arxiv.org/abs/2410.05068
**PEAK-best legal quantity = CH longitudinal width Δθ, then A×colat.** CM-slice A (Vršnak/Rotter/ESWF) tracks the lagged *profile*, not peak height. Lat extent is weak. Scatter floor ~70–118 km/s even with isolated-CH pairing — larger than our −0.4 public target, so geometry must be a *new CNN channel/head*, not a GBDT scalar (scalars hurt fast).

# 2 LOSS / OUTPUT (no posthoc rescale). MSE = E[Y|X] → slow-wind mean; peaks are the rare tail.

Collin: MSE “systematically underestimates HSS peak velocities”; Box-Cox *posthoc* peak RMSE 113.3→87.5 (4×3) / 98.9→76.8 (10×10) but **timeline RMSE 68.1→75.1 / 72.1→79.0**. Posthoc CLOSED. Train-time Box-Cox/log *target* is open; same timeline↑ expected. https://arxiv.org/abs/2410.05068
Rotter: do **not** use LS for peaks; 3-CR min/max of (A,v) instead. LEGAL, no extra data. https://arxiv.org/abs/1501.06697
Pinball q>0.5 / expectile τ>0.5: in-training tail lift, not q-map. No solar-wind RMSE paper. TFT P50. LEGAL https://arxiv.org/abs/1912.09363 https://doi.org/10.2307/1912613
MDN / 2-Gaussian (slow+fast): take μ_fast not mixture mean. Bishop 1994. LEGAL, **no published HSS RMSE**. https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-ncrg-94-004.pdf
Two-head = peak scalar + 12-step shape (Collin’s 10×10 vs 4×3 split, in one net). DILATE/onset aux: out2. LEGAL. https://arxiv.org/abs/2410.05068 https://arxiv.org/abs/1909.09020
**Skip/ILLEGAL:** 27d-V, SSN, mag-WSA, 171, OMNI N/T/B, B0. **Closed:** huber, HSS-weight, q-map, affine.

# 3 ENSEMBLE SMOOTHS PEAKS. Mean of bag = same MSE conditional mean.

Reiss: WSA ensemble-like kinematic run has std 70 vs ACE 84 — damped variance; high-v under 100–200. https://arxiv.org/abs/1607.07048
Riley 2013 / Owens 2017: SW ensemble *mean* is the skillful RMSE forecast; members keep amplitude, mean damps. Use spread as σ, not as the point. https://doi.org/10.1002/jgra.50156 https://doi.org/10.1002/2017SW001679
Hegde 2025 MHD ensemble: “HSS leading edge and peak were underestimated”. https://doi.org/10.3847/1538-4357/addf33
Edward-Inatimi 2024: raw SW ensembles underdispersed; they *calibrate/dress* (posthoc family — closed for us). https://doi.org/10.1029/2023SW003851
Weather MoE: mean mixture loses near-term peaks; learned gate beats mean. https://arxiv.org/abs/2509.09052
Owens 2005: RMSE hides peak skill if timing slips → score *event peaks*, not the blend. https://doi.org/10.1029/2005JA011343
**Counter (legal):** (a) do not mean-blend OF-CNN/Stonyhurst with tabular GBDT on fast cells; (b) bag → MDN/max/q>0.5 on CH-wide rows, mean elsewhere; (c) two-head peak+shape so the blend cannot average the peak away.
**Do:** Δθ (or lon-extent map) as extra CNN channel + expectile/MDN/two-head. **Don’t:** more GBDT scalars, posthoc stretch, 27d, mag, 171.
