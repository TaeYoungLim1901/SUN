"""예측을 한두 스텝 당기면 도착 오차가 주는가. 로컬 번들만 본다."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_peak import hill_climb, load_bundle, peak_diag  # noqa: E402


def early(p, k):
    o = p.copy()
    if k <= 0:
        return o
    o[:, :-k] = p[:, k:]
    o[:, -k:] = p[:, -1:]
    return o


y, vals, _, _ = load_bundle(Path(__file__).resolve().parents[1] / "work/local_data/pred_bundle.npz")
keys = ["img_gbdt", "eswf", "img_ridge", "v2_gbdt", "cnn96b", "cnn96w3"]
w, s = hill_climb({k: vals[k] for k in keys}, y)
b = sum(w[k] * vals[k] for k in w)
print("base", round(s, 3), peak_diag(y, b))
for k in (1, 2, 3):
    d = peak_diag(y, early(b, k))
    print(f"shift-{k}", round(d["rmse"], 3),
          "pkH", round(d["peak_h_bias"], 1),
          "step", round(d["peak_step_mae"], 2),
          "HSS", round(d["hss_rmse"], 1), round(d["hss_bias"], 1))
print("--- per-model shift-1 delta ---")
for n, v in sorted(vals.items()):
    if v.shape != y.shape:
        continue
    d0, d1 = peak_diag(y, v), peak_diag(y, early(v, 1))
    print(f"  {n:12} {d0['rmse']:7.2f}->{d1['rmse']:7.2f} ({d1['rmse']-d0['rmse']:+.2f}) "
          f"step {d0['peak_step_mae']:.2f}->{d1['peak_step_mae']:.2f}")
