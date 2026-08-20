Write ONE new file: D:\hackathon\sun\work\omni2.py . Do not modify any other file. No prose in the answer.
Written here, RUN on the server as `python work/omni2.py` from /home/jovyan/work.

GOAL: our `omni` GBDT component (work/omni.py, val 66.476) plus the new physics scalars from
work/phys.py, refit as ONE model. The physics scalars alone are a weak standalone component
(val 68.82) but their residual correlation with our blend is only 0.87, so they may carry
information the omni features lack. Re-fitting one GBDT on the union is the test.

DO THIS:
- Copy work/omni.py exactly (read it — it builds features from `align.aligned_block`, `feats.sample_feats`,
  `t4ch.t4_block`, fits one HistGradientBoostingRegressor per horizon on the residual vs wind_19,
  writes preds/val_omni.npy, preds/test_omni.npy and omni_models.pth).
- Change ONLY: TAG = "omni2", and append the physics feature matrix to the feature matrix of every split.
- The physics features come from work/phys.py. Do NOT re-implement them and do NOT re-run the image loop:
  import the row-feature builder from phys (read work/phys.py first and reuse its function; if its row
  features are only built inside its main block, import its per-unique-image cache
  /home/jovyan/work/preds/phys_<split>.npy and rebuild the row features with the SAME code path by
  importing the function - if that is impossible, add nothing to phys.py, instead copy the minimal
  row-assembly lines into omni2.py with a comment naming the source).
- Keep every hyperparameter, seed and split convention identical to omni.py. Train on TRAIN ONLY.
- Print: feature count before and after the append, per-horizon val RMSE, total val RMSE, and the
  total val RMSE of the original omni component for comparison (load preds/val_omni.npy).

ACCEPTANCE (run and paste):
  python -c "import ast;ast.parse(open(r'D:\hackathon\sun\work\omni2.py',encoding='utf-8').read());print('OK')"
