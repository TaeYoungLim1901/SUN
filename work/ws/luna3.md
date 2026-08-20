Write ONE new file: D:\hackathon\sun\work\phys.py . Do not modify any other file. No prose in the answer.
It is written here but RUN on a remote server: `python work/phys.py` from /home/jovyan/work.

GOAL: a new GBDT component built from EMPIRICAL SOLAR-WIND PHYSICS scalars (coronal-hole geometry +
transit lag), so it fails differently from our image-GBDT and optical-flow CNN components.

DATA ACCESS (copy these conventions from work/cnn6.py `cache()` and work/feats.py):
- inputs: /home/jovyan/public_dataset/competition_dataset_6h/<split>/inputs.csv,
  columns image_00..image_19 (filenames) and wind_00..wind_19 (km/s, 6h cadence, wind_19 = latest).
- targets (train, validation only): <split>/targets.csv columns target_00..target_11 (+6h..+72h).
- images: preprocessed cache /home/jovyan/work/resized_cache/96px/<split>_images.npy,
  shape (n_unique, 2, 96, 96) uint8, channel 0 = 193A, channel 1 = 211A.
  Row order = `sorted(pd.unique(inputs[IMG].to_numpy().ravel()).tolist())`. Build {name: row} from that.
  DO NOT read the original PNGs. DO NOT rebuild the cache. If the file is missing, exit with a message.

PER-FRAME GEOMETRY (96px, so r = 0.47*96, centre 47.5):
  dx=(xx-47.5)/r, dy=(yy-47.5)/r, disk = dx^2+dy^2 <= 1, mu = sqrt(clip(1-dx^2-dy^2,1e-3,1))
  lon = degrees(arcsin(clip(dx,-1,1))), lat = degrees(arcsin(clip(dy,-1,1)))   # west = +lon
  ch  = (I193 < 0.35*median(I193[disk])) & disk          # coronal-hole mask
  ch2 = ch & (I211 < 0.45*median(I211[disk]))            # stricter 193 AND 211 mask

SCALARS PER FRAME (all from the published empirical relations, constants are literature values):
  1. Robbins occupancy vector: th = arange(-77, 41, 7.0) (17 swaths). For each th:
       Is[k] = mean(ch[(abs(lon-th[k])<7) & (abs(lat)<60) & disk])  (0 if the swath has no disk pixels)
       lag d[k] = 3.69 - 0.07386*th[k]  (days)
     store the 17 Is values, and also the interpolated V estimate per forecast horizon h=0..11:
       Vhat[h] = 330 + 930*interp(0.25*(h+1), d[::-1], Is[::-1])       (12 numbers)
  2. Hofmeister: A_km2 = (ch/mu).sum() * (6.96e5/r)**2 ;
     phi_co = degrees(arcsin(clip(mean(dy[ch]) if ch.any() else 0, -1, 1))) ;
     v_hof = 478 + (2.77e-9*A_km2)*(1 - abs(phi_co)/61.4)
  3. Garton: dphi = lon[ch2].max()-lon[ch2].min() (0 if empty);
     vmax = 330.8 + 5.7*dphi if dphi < 67 else 710.0 ; dt_sw = 0.09*dphi
  4. West edge / SIR: thW = lon[ch].max() (0 if empty); t_SI = 3.69-0.07386*thW ;
     t_pk = 3.69-0.07386*mean(lon[ch]) ; west = ch & (lon >= percentile(lon[ch],75)) ;
     edge_sharp = mean(I193[west]) - mean(I193[ch])
  5. area fraction ch.mean() and ch2.mean().
  Compute these ONCE PER UNIQUE IMAGE and cache to /home/jovyan/work/preds/phys_<split>.npy
  (shape (n_unique, F)); if that file exists, load it instead of recomputing.

ROW FEATURES: for each row, take the per-frame scalars of frames 19 (latest), 15, 11, 7, 3
  plus mean and std over the 20 frames, plus the rotation-rate proxy
  omega = (phi_cm[19]-phi_cm[0])/4.75 and rec = abs(omega-13.2) where
  phi_cm[t] = degrees(arcsin(clip(mean(dx[ch_t]),-1,1))), plus wind_00..wind_19 and
  their diffs, wind mean/std/min/max. Concatenate to one float32 matrix.

MODEL: sklearn HistGradientBoostingRegressor, one per horizon h (12 models), target =
  target_h MINUS wind_19 (residual vs persistence, same convention as our other components),
  params: max_iter=400, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=20,
  early_stopping=False, random_state=777. Train on TRAIN ROWS ONLY — never on validation.
  Predict validation and test, add wind_19 back.

OUTPUT: save /home/jovyan/work/preds/val_phys.npy (1199,12) and test_phys.npy (3868,12), float64,
  and print: feature count, per-horizon val RMSE, total val RMSE, and wall time.

STYLE: match work/feats.py + work/omni.py — Korean one-line docstring, numpy/pandas/sklearn only,
no argparse, no classes, under 160 lines. Vectorise over unique images; the loop must be over
unique images (~13k per split), not over rows.

ACCEPTANCE (run and paste):
  python -c "import ast;ast.parse(open(r'D:\hackathon\sun\work\phys.py',encoding='utf-8').read());print('OK')"
