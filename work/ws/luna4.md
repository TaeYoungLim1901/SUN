Write ONE new file: D:\hackathon\sun\work\tweight.py . Do not modify any other file. No prose in the answer.
Written here, RUN on the server: `python work/tweight.py` from /home/jovyan/work.

BACKGROUND (measured today): validation and test are NOT exchangeable. wind_19 KS validation-vs-test
= 0.148 while train-vs-test = 0.049; validation median wind_19 = 415, test = 386, train = 394.
So our blend weights are chosen on a distribution that is faster than the test distribution.

GOAL: recompute known card-vs-card deltas under IMPORTANCE-WEIGHTED validation, where validation rows
are reweighted to match the TEST distribution of wind_19, and report whether the reweighted delta lines
up better with the actual public-leaderboard delta than the unweighted one.

INPUTS on the server:
- /home/jovyan/work/preds/val_<tag>.npy, each (1199,12) float; tags available include
  img_ridge, img_gbdt, omni, eswf, cnn3b, cnn3c, cnn3d, BAGsmall32, BASE039, N50.
- targets: /home/jovyan/public_dataset/competition_dataset_6h/validation/targets.csv (target_00..11)
- inputs: validation/inputs.csv and test/inputs.csv (wind_00..wind_19)

CARDS to reconstruct from those npys (weights are exact, do not re-fit anything):
  E039  = 0.075*img_ridge + 0.18*img_gbdt + 0.19*omni + 0.215*eswf + 0.113*cnn3b + 0.113*cnn3c + 0.114*cnn3d
  MAX   = 0.8*E039 + 0.2*BAGsmall32
  N50   = 0.5*(0.075*img_ridge + 0.18*img_gbdt + 0.19*omni + 0.215*eswf)/0.66 + 0.5*BAGsmall32
Known public leaderboard scores: E039 56.8181, MAX 56.2743, N50 55.8298.

WEIGHTS: estimate w(row) = p_test(wind_19) / p_val(wind_19) by histogram ratio with 20 equal-count
bins built on the pooled (validation+test) wind_19, add-one smoothing, clip w to [0.2, 5], normalise
to mean 1. Also produce a second weighting using a 2-D histogram on (wind_19, wind_19 - wind_15),
10x10 bins, same clipping/normalising.

PRINT (plain text, no files):
1. weight diagnostics: effective sample size sum(w)^2/sum(w^2) for both weightings.
2. For each card: unweighted validation RMSE, 1-D weighted RMSE, 2-D weighted RMSE, and the public score.
3. For each of the 3 pairs (E039->MAX, MAX->N50, E039->N50): delta in each of the three local metrics
   next to the actual public delta, and the ratio public_delta / local_delta for each metric.
4. A one-line summary table of which local metric has the ratio closest to 1 and the smallest spread
   across the 3 pairs.
5. Same as (2) but restricted to the 6h connected components: use `from noise_floor import components`
   exactly as work/paired.py does, and print per-component weighted vs unweighted RMSE for N50 only.

Weighted RMSE means sqrt( sum_i w_i * mean_h (pred-y)^2 / sum_i w_i ).
Never fit on validation targets. numpy/pandas only. Korean one-line docstring, no classes, no argparse,
under 120 lines.

ACCEPTANCE (run and paste):
  python -c "import ast;ast.parse(open(r'D:\hackathon\sun\work\tweight.py',encoding='utf-8').read());print('OK')"
