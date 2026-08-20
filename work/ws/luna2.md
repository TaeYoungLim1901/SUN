Write ONE new file: D:\hackathon\sun\work\shift.py . Do not modify any other file. No prose in the answer.

Context: solar wind competition. train 9607 / validation 1199 / test 3868 rows. Inputs csv at
/home/jovyan/public_dataset/competition_dataset_6h/<split>/inputs.csv with columns wind_00..wind_19
(past speeds, km/s) and image_00..image_19 (PNG filenames). The script is written here but RUN on the
remote server: `python work/shift.py`. It must be import-clean; stdlib + numpy + pandas + sklearn only.
You may import `from feats import sample_feats` (feats.sample_feats(split, inputs_df) -> (N,839) float
features, uses a disk cache; it is slow the first time, ~1 min per split).

Purpose: quantify covariate shift between validation and test, to know whether validation-selected
blend weights transfer to the public leaderboard.

Print (plain text, no plots, no files written):
1. For wind_19 and for mean(wind), std(wind), wind_19-wind_18: the deciles for train / validation / test
   side by side, plus a two-sample KS statistic validation-vs-test and train-vs-test.
2. Fraction of test rows whose wind_19 falls outside the validation min/max, and same for the 839-dim
   feature block: fraction of test rows with any feature outside the validation per-feature [min,max],
   and the median count of out-of-range features per test row.
3. An adversarial-validation check: HistGradientBoostingRegressor is not right — use
   sklearn.ensemble.HistGradientBoostingClassifier on label (0=validation, 1=test) over the 839 features,
   3-fold cross_val_predict, print ROC AUC. AUC ~0.5 means validation and test are exchangeable; print
   the 15 most important features by permutation importance ONLY if AUC > 0.6 (else skip, it is noise).
4. Nearest-neighbour distance: for a random 500-row sample of test, the distance to the nearest
   validation row in standardized 839-space (mean/std from TRAIN only), vs the same statistic for
   validation-to-train. Print the deciles of both.

Rules: never fit anything on validation targets, never merge validation into train, no external data.
Under 120 lines, Korean one-line module docstring, no argparse, no classes.

ACCEPTANCE (run and paste output):
  python -c "import ast;ast.parse(open(r'D:\hackathon\sun\work\shift.py',encoding='utf-8').read());print('OK')"
