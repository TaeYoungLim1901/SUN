"""OOF 재료가 모이면 보정 계수를 뽑고, 게이트를 재고, 통과하면 팩까지 만든다.
노트북이 꺼져도 서버에서 끝까지 간다. /home/jovyan/submission/ 은 절대 건드리지 않는다.
산출: work/preds/calib_coef_G.npy, work/calib_verdict.json, (통과 시) work/pack_calib/
"""
import json, pathlib, subprocess, sys, time
import numpy as np, pandas as pd
ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
P = pathlib.Path("/home/jovyan/work/preds")
WORK = pathlib.Path("/home/jovyan/work")
T = [f"target_{i:02d}" for i in range(12)]
WG = dict(img_ridge=.125, img_gbdt=.25, omni=.125, eswf=.25)     # GBDT 부 (합 0.75)
WC = dict(cnn96b=.125, cnn96w3=.125)
KN = np.array([350., 400., 450., 500., 550., 600., 650., 720.])
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
d = lambda x: np.column_stack([np.ones_like(x), x] + [np.maximum(x-k, 0) for k in KN])

def fit(x, y, r=1e-3):
    A = d(np.ravel(x)); M = A.T@A; M += r*np.eye(A.shape[1])*np.trace(M)/A.shape[1]
    return np.linalg.solve(M, A.T@np.ravel(y))
ap = lambda x, c: (d(np.ravel(x))@c).reshape(np.shape(x))

need = ["oof_img_ridge.npy", "oof_img_gbdt.npy", "oof_eswf.npy", "oof_omni.npy"]
t0 = time.time()
while time.time()-t0 < 5400:
    if all((P/n).exists() for n in need):
        break
    print("waiting", [n for n in need if not (P/n).exists()], flush=True)
    time.sleep(60)
missing = [n for n in need if not (P/n).exists()]
if missing:
    print("FAIL 재료 없음", missing, flush=True); sys.exit(1)

ytr = pd.read_csv(ROOT/"train/targets.csv")[T].to_numpy(np.float64)
yva = pd.read_csv(ROOT/"validation/targets.csv")[T].to_numpy(np.float64)
O = {k: np.load(P/f"oof_{k}.npy").astype(np.float64) for k in WG}
G_oof = sum(w*O[k] for k, w in WG.items())/0.75
# oof_omni 는 horizon 부분집합만 채워져 있을 수 있다. 0 인 열은 뺀다.
ok = ~np.all(O["omni"] == 0, axis=0)
print("사용 horizon", np.where(ok)[0].tolist(), flush=True)
m = np.zeros_like(G_oof, bool); m[:, ok] = True
coef = fit(G_oof[m], ytr[m])
np.save(P/"calib_coef_G.npy", coef)

V = {k: np.load(P/f"val_{k}.npy").astype(np.float64) for k in list(WG)+list(WC)}
G_val = sum(w*V[k] for k, w in WG.items())/0.75
blend = 0.75*G_val + sum(w*V[k] for k, w in WC.items())
mix = 0.75*ap(G_val, coef) + sum(w*V[k] for k, w in WC.items())
c_val = fit(G_val, yva)                       # 대조용(채택 아님)
xs = np.array([380., 420., 450., 500., 550., 600., 650., 700.])
curve_oof = (d(xs)@coef - xs).round(1).tolist()
curve_val = (d(xs)@c_val - xs).round(1).tolist()
before, after = rmse(blend, yva), rmse(mix, yva)
sign_agree = sum(1 for a, b in zip(curve_oof, curve_val) if a*b > 0)
verdict = dict(before=round(before, 4), after=round(after, 4), delta=round(before-after, 4),
               curve_x=xs.tolist(), curve_oof=curve_oof, curve_val=curve_val,
               sign_agree=f"{sign_agree}/8", oof_G_rmse=round(rmse(G_oof[m], ytr[m]), 3),
               horizons=np.where(ok)[0].tolist(),
               pass_=bool(before-after >= 0.15 and sign_agree >= 6))
(WORK/"calib_verdict.json").write_text(json.dumps(verdict, indent=1))
print(json.dumps(verdict, indent=1), flush=True)
if not verdict["pass_"]:
    print("GATE FAIL — 팩 안 만든다", flush=True); sys.exit(0)

# ── 팩 생성 (제출 폴더는 손대지 않는다) ──
out = WORK/"pack_calib"; out.mkdir(exist_ok=True)
print(subprocess.run("cd /home/jovyan && python work/pack_omni.py",
                     shell=True, capture_output=True, text=True).stdout[-800:], flush=True)
import torch
blob = torch.load(WORK/"pack_out/model.pth", map_location="cpu", weights_only=False)
blob["calib_G"] = coef
torch.save(blob, out/"model.pth")
print("model.pth", (out/"model.pth").stat().st_size, flush=True)
r = subprocess.run(f"cd {out} && python /home/jovyan/work/pipeline.py",
                   shell=True, capture_output=True, text=True)
print(r.stdout[-800:], r.stderr[-400:], flush=True)
sub = pd.read_csv(out/"submission.csv")
print("submission", sub.shape, float(sub.iloc[:, 1:].to_numpy().min()), float(sub.iloc[:, 1:].to_numpy().max()), flush=True)
print("CHAIN DONE", flush=True)
