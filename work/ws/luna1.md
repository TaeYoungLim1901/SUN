Write ONE new file: D:\hackathon\sun\work\anatomy.py . Do not modify any other file. No prose in the answer.

Context: solar wind competition. Validation has 1199 rows x 12 horizons. Predictions live in
/home/jovyan/work/preds/val_<tag>.npy on a remote server (this machine has no data — the script is
written here and run there; it must be import-clean and runnable with `python work/anatomy.py <val_pred.npy>`).
Existing helpers in the same directory you may import: `from noise_floor import components`
(components(wind_matrix)->int labels for 6h connected components; see work/paired.py lines 1-40 for exact
usage), targets at /home/jovyan/public_dataset/competition_dataset_6h/validation/targets.csv columns
target_00..target_11, inputs.csv columns wind_00..wind_19 and image_00..image_19.

The script must print (plain text, aligned columns, no plots, no files written):
1. total RMSE, and per-horizon RMSE h=0..11 plus each horizon's share of total squared error (%).
2. per connected component: n rows, RMSE, share of total squared error (%), mean last wind (wind_19),
   mean target, mean bias (pred-target).
3. binned by last wind (wind_19) in quintiles: n, RMSE, mean bias.
4. binned by true target value in the bins [<350,350-450,450-550,550-650,>650]: n, RMSE, mean bias,
   and the same table restricted to horizons 6-11 (the far horizons).
5. binned by |target - wind_19| (the size of the change the model must predict) in quintiles: n, RMSE, mean bias.
6. concentration: sort all 1199*12 squared errors descending, print what fraction of total squared error
   comes from the worst 1%, 5%, 10%, 25% of cells, and how those worst 1% cells distribute over
   (horizon, target bin, component).
7. the 15 worst rows by row-RMSE: row index, component id, wind_19, target profile min/max, pred profile
   min/max, row RMSE.

Style: match work/paired.py — module docstring in Korean one line, stdlib+numpy+pandas only, lambdas for
rmse, no argparse, no classes, under 120 lines. Use `sys.argv[1]` for the prediction file, default
"/home/jovyan/work/preds/val_N50.npy".

ACCEPTANCE (run and paste output):
  python -c "import ast;ast.parse(open(r'D:\hackathon\sun\work\anatomy.py',encoding='utf-8').read());print('OK')"
