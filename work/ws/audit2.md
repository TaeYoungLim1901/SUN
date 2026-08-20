READ-ONLY AUDIT. Do NOT modify any file. Do NOT run anything on the remote server.
Write your findings to work/ws/auditB.md and nothing else.

Scope: today's measurement and judging code, which decides whether we spend one of our 8 remaining
competition submissions. Errors here are worse than a bad model.
Files: work/n50eval.py, work/anatomy.py, work/shift.py, work/tweight.py, work/phys.py,
and the edit made to work/add_honest.py (a new `--base TAG` flag; compare against git HEAD).
Reference conventions: work/paired.py, work/maxbag.py, work/noise_floor.py, CHECKLIST.md sections C/D/G.

Audit for:
1. Does n50eval.py rebuild the currently submitted card N50 correctly? It should be
   0.5*(0.075*img_ridge + 0.18*img_gbdt + 0.19*omni + 0.215*eswf)/0.66 + 0.5*BAGsmall32,
   where BAGsmall32 is the mean of the 32 flow-CNN seeds. Check the renormalisation by 0.66 and that
   val_BAGsmall32.npy is the same object work/maxbag.py wrote. Report any mismatch with NEXT.md's
   statement that N50's pack re-inference val = 62.633.
2. tweight.py: is the importance weight the right way round (test density over validation density)?
   Are the bins built on the pooled sample? Is the weighted RMSE formula sqrt(sum w*mse_row / sum w)?
   Does any card reconstruction there disagree with n50eval.py or maxbag.py?
3. anatomy.py: are the per-bin RMSEs computed over cells vs rows consistently with their labels? Is the
   "share of SSE" arithmetic right? Would the reported concentration (worst 1% of cells = 18.6% of SSE)
   change if the array were accidentally transposed?
4. shift.py: adversarial validation reported ROC AUC exactly 1.0000 between validation and test over the
   839 features. Judge whether that is plausible or whether it indicates a construction bug (for example
   a feature that encodes row order, file index, or a constant that differs per split). Name the most
   likely feature families involved, given work/feats.py builds the 839 columns.
5. phys.py: does it fit on train only? Does it add wind_19 back after predicting the residual? Are the
   coronal-hole thresholds applied to the 96px cache in the same channel order (0=193, 1=211) that
   work/cnn6.py assumes?
6. add_honest.py `--base` edit: does the base substitution change the meaning of the honest 2-fold
   protocol in any way beyond swapping the baseline blend?

Format: one line per finding, `file:line - severity(HIGH/MED/LOW) - what is wrong - the one-line fix`.
State explicitly if a section is clean. Max 30 lines. No prose.
