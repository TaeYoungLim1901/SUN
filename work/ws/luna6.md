Write ONE new file: D:\hackathon\sun\work\sweep.py . Do not modify any other file. No prose in the answer.
Written here, RUN on the server: `python work/sweep.py` from /home/jovyan/work. It must be a single
self-contained report so a human runs it once instead of running work/n50eval.py by hand ten times.

Read work/n50eval.py, work/paired.py and work/add_honest.py first and reuse their conventions exactly
(same rmse, same component split via `from noise_floor import components`, same npy locations).

WHAT IT DOES
1. Rebuild the current submitted card N50 from npys, exactly as work/n50eval.py does:
   base4 = (0.075*img_ridge + 0.18*img_gbdt + 0.19*omni + 0.215*eswf)/0.66 ; N50 = 0.5*base4 + 0.5*BAGsmall32.
   Save it to preds/val_N50.npy and preds/test_N50.npy (test_ files use the test_ prefix npys).
2. Discover candidate FAMILIES by glob on preds/val_*.npy:
     cnn7  -> val_cnn7s*.npy      (image+flow+scalar hybrid seeds)
     cnn7p -> val_cnn7p*.npy      (same plus peak auxiliary loss)
     cnn9  -> val_cnn9s*.npy      (Stonyhurst remap seeds)
     solo tags: phys, omni2 (each is one npy; skip if absent)
   A family's bag = the plain mean of its member npys.
3. For every family bag AND for every union of 2 and 3 families (bag = mean of all member npys in the
   union, not the mean of bags), print one line:
     name | n members | solo val RMSE | rho(residual) vs BAGsmall32 | rho vs N50 | best w on a grid
     0.05..0.50 step 0.05 | val RMSE at that w | delta vs N50
   Sort the lines by delta ascending. Keep it to one screen.
4. Take the best union by delta, save preds/val_CAND.npy = (1-w)*N50 + w*bag at its best w, and then
   run BOTH judges on it in-process (do not shell out):
     a) the paired-component judge from work/paired.py: overall delta, per-component deltas and the
        gain count, half-split sigma, 2000-sample bootstrap sigma and 95% CI, delta/sigma.
     b) the honest judge from work/add_honest.py with N50 as the base: 2-fold x 20 repeats over the
        connected components, choosing w on one half and scoring the other, print median delta,
        mean delta, gain fraction, and the histogram of chosen w.
   Print the pass/fail line for our acceptance bar: overall delta <= -0.3 AND honest median delta < 0
   AND honest gain >= 80%.
5. Finally print the same table as (3) but computed on IMPORTANCE-WEIGHTED validation, where the weight
   of a validation row is the ratio of test to validation density of wind_19, estimated with 20
   equal-count bins on the pooled wind_19, add-one smoothed, clipped to [0.2,5], normalised to mean 1
   (this is work/tweight.py's 1-D weighting; read it and reuse the same numbers). Label the column
   "wdelta". Measured today: public delta ~= 1.48 * wdelta with spread 0.08 over 3 pairs.

RULES: never fit anything on validation targets, no external data, numpy/pandas only, Korean one-line
docstring, no argparse, no classes, under 200 lines. Must run in under 3 minutes on 2 CPU cores.

ACCEPTANCE (run and paste):
  python -c "import ast;ast.parse(open(r'D:\hackathon\sun\work\sweep.py',encoding='utf-8').read());print('OK')"
