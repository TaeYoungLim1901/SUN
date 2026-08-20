CONTEXT (do not re-propose anything below — all measured and closed on this task):
Task: SDO/AIA 193+211 solar disk image sequence (20 frames, 6h cadence) + 20 past wind speeds
      -> predict solar wind speed at 12 future horizons (6h..72h). Metric RMSE. val 1199 rows, test 3868.
Current: ensemble val RMSE 62.63, Public LB 55.83 (1st place). Components: image-ridge, image-GBDT,
      omni-GBDT, eswf-GBDT, and a 32-seed bag of a 2-stream CNN (image 2ch + Farneback optical flow 2ch,
      96px, GRU over 19 steps, target = residual vs last observed wind, flip-TTA over N-S axis).
CLOSED AXES (measured, do not suggest): optical-flow CNN variants (128px, shorter window, huber loss,
      full width, 3-stream, hss weighting) all residual-corr >=0.988 with the base seed bag;
      free/weighted blend fitting; horizon-ramped blend weights; base-subset search; seed-bag expansion;
      piecewise-linear post-hoc calibration; quantile mapping; horizon-wise affine calibration;
      output time shift; swapping components into the gbdt_v2 slot; last-wind persistence boosting.
HARD RULES: no external data, no pretrained weights, no pseudo-labels, validation must never be trained on,
      no east-west (left-right) image flip (breaks rotation direction). Inference must run on 2 CPU cores /
      8GB RAM / 1 A100, model artifact <= 700MB.
WHAT WE NEED: a genuinely NEW INPUT REPRESENTATION or NEW TARGET/LOSS formulation that would decorrelate
      from the existing components (residual corr < 0.95), worth ~-0.4 RMSE on the public LB.
