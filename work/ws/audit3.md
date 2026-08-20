READ-ONLY review. Do not modify files. Do not run the remote server. Reply in the chat only (no file writes).
Be terse: a numbered list of concrete findings, each `file:line - SEV - claim - fix`. Max 25 lines total.

CONTEXT (all measured today on the SUN solar-wind competition; validation = 1199 rows x 12 horizons, RMSE):
- Current submitted card (CARD1) scored public 55.7603. Previous cards: N50 55.8298, MAX 56.2743, E039 56.8181.
- Card compositions (weights over groups): E039 = 7 legacy components; MAX = 0.8*E039 + 0.2*flowbag32;
  N50 = 0.5*base4 + 0.5*flowbag32; CARD1 = 0.40*base4 + 0.40*flowbag + 0.20*hybrid13;
  proposed CARD2 = 0.30*base4 + 0.50*flowbag + 0.20*hybrid13.
  base4 = renormalised {img_ridge .075, img_gbdt .18, omni .19, eswf .215}/0.66.
  hybrid13 = mean of 10 cnn7 seeds (flow CNN + 839 scalar features) + 3 cnn9 seeds (Stonyhurst remap).
- Local val: N50 62.6315, CARD1 62.4274, CARD2 62.4524.
- Pair transfer (public delta / local delta): E039->MAX 1.39, MAX->N50 1.87, N50->CARD1 **0.34**.
- Split by |target - wind_19| (ramp size), pair deltas: winners improved the bulk (|d|<p60) and LOST the
  top-5% ramp cells. SSE shares: bulk 24.9%, p60-80 18.6%, p80-95 24.9%, top5% 31.6% (724 cells).
- In the top-5% ramp cells every component is RMSE 152-176 (best eswf 151.8) - looks irreducible.
- Per-family contribution when added at w=0.10 to CARD1, measured on lw>=550 rows: flowbag -0.433,
  cnn9 -0.369, phys +0.117, omni2 +0.479, cnn7 +0.514.

FILES TO REVIEW (read them; they are the basis of the above): work/ramp.py, work/mix.py, work/fastfam.py,
work/comp_fast.py, work/fit2.py, work/recal.py, work/n50eval.py, work/sweep.py, work/mkcard7.py,
work/pipeline7.py (diff vs work/pipeline.py), CHECKLIST.md section C-2.

ANSWER THREE THINGS:
1. LOGIC/BUG/MISMATCH: any error in those scripts or in the inferences drawn from them. Specifically check:
   is the bulk-vs-ramp conclusion an artifact of conditioning on the true target? is the paired bootstrap
   in mix.py correct now? does mkcard7.py's group re-weighting preserve the intended ratios? does
   pipeline7.py reproduce training-time inference for arch v7 and v9 exactly?
2. IS CARD2 WORTH A SUBMISSION SLOT? We have ~2 slots today (expire) and a deadline tomorrow 15:00 KST.
   CARD2 differs from CARD1 only by shifting 0.10 weight from base4 to flowbag. Judge from the numbers above.
3. BIGGER LEVER: given the SSE decomposition and that the top-5% cells look irreducible, what single change
   has the best expected public gain in under 20 hours on 1 A100 + 2 CPU cores? Rank at most 3, with the
   arithmetic for why. Assume no external data, no pretrained weights, validation may not be trained on.
