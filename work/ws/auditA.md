# auditA — cnn7 / cnn7p / cnn9 vs cnn6 (read-only)

## 1 LEAKAGE
cnn7/cnn7p/cnn9: clean — pixel mean/std, FSTD[:2000], smu/ssd from train only; sample_feats is per-split no-fit; val only for ES as in cnn6.

## 2 TRAIN/INFERENCE
cnn7.py:264 / cnn7p.py:277: clean — same disk NS-flip + fy-negate TTA as cnn6; 839-d scalars not flipped.
cnn9.py:253 vs 267: clean — remap-then-TTA ≡ train-flip-then-remap (gy odd in lat, gx even in lat; fy stays disk-px so fy negate is still the NS operation).

## 3 cnn7p AUX
cnn7p.py:235-237: clean — pstep/pmag from tr_y−wind_19 only; argmax/max of that residual.
cnn7p.py:279,293: clean — infer uses [0]; train unpacks (out, aux); no other call sites; aux unused at infer.
cnn7p.py:300 - MED - LAM*(CE+L1) is O(1) vs MSE O(4e3); aux grads are a no-op - scale aux by ~1e3 or divide MSE by ~4000.

## 4 cnn9 GRID
cnn9.py:180: clean — 0.94 = 2*(0.47*PX)/PX for align_corners=False; x=0 maps to PX/2−0.5, same center as the disk mask.
cnn9.py:179-181: clean — lon along W (gx), lat along H (gy); not swapped.
cnn9.py:181 - HIGH - gy=+0.94*sin(lat) samples +lat at grid y=+1 (image bottom); PNGs are north-up/west-right (feats.py:48) so this swaps hemispheres - set gy=-0.94*sin(lat).
cnn9.py:255 - LOW - zeros-pad would inject 0 (=disk-median after med-norm, bright vs off-limb ~−2.9) but lim=72 keeps |g|≤0.894<1 so it never fires - use padding_mode="border" if lim rises.

## 5 DTYPE/SCALE
cnn7.py:218,311 - HIGH - smu/ssd (839) and FSTD are printed not written to {TAG}.json/pth; pack_seedbag defaults fstd=1.0 and has no scalar-stat field (same class as the 1.368× flow bug) - dump smu,ssd,FSTD next to the weights.
cnn7p.py:226,325 - HIGH - same missing smu/ssd/FSTD; plus aux.* keys vs NetV6.
cnn9.py:158,253 - HIGH - Net keys identical to cnn6/v6 so pack cnn_predict(arch=v6) loads the pth and skips grid_sample; copy the same GRID+grid_sample into pack infer (GRID is a function of PX, not a weight).
cnn7/cnn7p/cnn9 fp16 flow cache + FSTD[:2000].float(): same as cnn6 — no new in-process scale hole.
