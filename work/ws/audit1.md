READ-ONLY AUDIT. Do NOT modify any file. Do NOT run training. Do NOT touch the remote server.
Write your findings to work/ws/auditA.md and nothing else.

Scope: the three new model scripts written today, each derived from work/cnn6.py:
  work/cnn7.py  = cnn6 + a 839-dim scalar block (feats.sample_feats) concatenated into Net.wind
  work/cnn7p.py = cnn7 + auxiliary head (12-way peak-step CE + peak magnitude L1), LAM=0.2
  work/cnn9.py  = cnn6 + Stonyhurst remap of the 4-channel (image,flow) tensor via grid_sample
Diff each against work/cnn6.py and audit ONLY for correctness:

1. LEAKAGE: any use of validation or test rows to fit, standardise, select or early-stop beyond what
   cnn6.py already did. Standardisation statistics must come from train only. Report any violation.
2. TRAIN/INFERENCE MISMATCH: does inference apply exactly the transforms training applied? Note the
   flip-TTA path: it flips the tensor north-south and negates the fy flow channel. In cnn9 the remap
   happens AFTER augmentation - check the TTA flip still corresponds to the same physical operation
   after remapping, and whether the fy negation is still correct in remapped coordinates.
3. cnn7p: is the auxiliary target built from train targets only? Is the peak-step label consistent with
   the residual convention (targets minus wind_19)? Does the aux head change the main output path at
   inference (model(...) now returns a tuple - check every call site)?
4. cnn9: verify the grid geometry. Disk radius in cnn6 is 0.47*PX pixels; grid_sample normalised
   coordinates span [-1,1] over the whole image with align_corners=False. Confirm 0.94 is the right
   scale factor and that lon/lat are not swapped, and that padding_mode="zeros" cannot inject a
   spurious bright edge given the med normalisation makes the background ~1.0 rather than 0.
5. Any silent dtype/scale issue (fp16 flow cache, FSTD division) that would make a saved model behave
   differently when re-run inside the submission pack. Context: a past bug shipped a pack without the
   flow std constant so inference fed flow 1.368x too small; check nothing analogous exists here.

Format: one line per finding, `file:line - severity(HIGH/MED/LOW) - what is wrong - the one-line fix`.
If a section is clean, say so in one line. No prose, no restating the code. Max 30 lines.
