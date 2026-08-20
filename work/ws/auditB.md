# auditB (read-only). format: file:line - SEV - wrong - fix. max 30.

1 n50eval/N50: work/n50eval.py:24-27 - CLEAN - tot=sum(BASE4)=0.66; n50=0.5*base4+0.5*bag ≡ 0.5*(0.075*ridge+0.18*gbdt+0.19*omni+0.215*eswf)/0.66+0.5*BAGsmall32; loads same `/preds/val_BAGsmall32.npy` maxbag.py:31 writes (mean of glob `val_cnn6s6*.npy`).
work/n50eval.py:28 vs NEXT.md:56 - LOW - npy RMSE is 62.6315 not pack re-inference 62.633 (gap 0.0015, same class as MAX 62.869 vs 62.870). No formula fix; do not use 62.633 as the npy number.
work/maxbag.py:19-31 - MED - `small32` is whatever `cnn6s6*` exists; no `len==32` assert, so extra seeds silently redefine N50 and every judge that loads the bag. Fix: `assert len(tg)==32` before save.
work/n50eval.py:43 - MED - eval always `np.save(val_N50.npy)` which anatomy.py:34 and `add_honest --base N50` consume; a later glob-bag rewrite retargets judges off the submitted pack. Fix: do not save, or write `val_N50_npy.npy`.

2 tweight: work/tweight.py:20-28,31-46,49-54 - CLEAN - w=(test dens)/(val dens) with +1 Laplace; 1D/2D edges from pooled val+test; score=sqrt(sum w*mse_row/sum w).
work/tweight.py:66-71 vs n50eval.py:24-27 vs maxbag.py:8-9,29 - CLEAN - E039=scal+0.113/0.113/0.114 cnn3; MAX=0.8*E039+0.2*bag; N50=0.5*scal/0.66+0.5*bag. No disagreement.
work/tweight.py:32-34 - LOW - 2D uses equal-width `histogram2d`, not the equal-count quantiles of 1D. Fix: pooled 2D quantiles if 2D weights decide a slot.

3 anatomy: work/anatomy.py:17-23,61-73 - CLEAN - `rows=True` → `err[mask]` over (n,12) labelled n_rows; `rows=False` → `err.ravel()[mask]` labelled n_cells; far-horizon mask is `target_bin.reshape(y.shape)[:,6:].ravel()` vs `e[:,6:]` (C-order match).
work/anatomy.py:42-43,51-58,79 - CLEAN - SSE share is `s.sum()` fractions (horizon, component, worst-k); not n*rmse^2 with mixed n.
work/anatomy.py:75-79 - CLEAN - worst-1% of SSE is a multiset of per-cell squares so a transpose/F-ravel would not change 18.6%; `assert pred.shape==(len(inp),12)` at :38 would also catch a (12,n) load. Horizon/component of those cells (`wi%12`, `np.repeat(lab,12)`) *would* flip under F-order; they are C-order-correct.

4 shift: work/shift.py:60-67 - HIGH - AUC 1.0000 is not a row/file-index leak (feats.py:63-68 has no id/index/arange); wind KS~0.15 cannot do it. Most likely families: whole-disk 193/211 mean/std/p10/p90/dark-frac (feats.py:40-41, F0-17), 4×4 grid + west-strip dark/mean (43-52), last-8-slot disk series `key[:,-8:]` (62,67 → cols 695-838). Treat as real solar-cycle brightness split unless a column has 0 within-split variance. Fix: name the top-15 indices and drop split-constants before using val to spend a slot.
work/shift.py:73-74 - MED - `f{i:03d}` has no name map so the 1.0000 cannot be diagnosed from the log. Fix: print the 839 names.

5 phys: work/phys.py:109-115 - CLEAN - `model.fit(train, y_train-wind_19)` only; val/test = predict + wind_19.
work/phys.py:35-50 vs cnn6.py:62-68 - CLEAN - same `resized_cache/96px/{split}_images.npy` shape (n,2,96,96); ch0=193, ch1=211; `i193,i211=frame` matches `enumerate(("193","211"))`.

6 add_honest --base (diff vs HEAD): work/add_honest.py:44-56 - CLEAN - `--base TAG` only replaces `A` (`L(TAG)` vs E039 7-sum); 2-fold×20, w-on-fit / score-on-eval, GRID, ρ, GO bar unchanged. Meaning change is the baseline swap only.
work/add_honest.py:44-45 - LOW - `--base` must be argv[1]; a later `--base` is averaged into New. Fix: argparse.
