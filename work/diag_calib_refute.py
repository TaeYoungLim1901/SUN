"""보정 채택 반증. numpy 만. 학습·preds 덮기·submission 금지.

판정용 측정: val 타깃으로 최종 계수를 만들지 않는다.
산출: /home/jovyan/work/diag_calib_refute.json
"""
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
P = pathlib.Path("/home/jovyan/work/preds")
OUT = pathlib.Path("/home/jovyan/work/diag_calib_refute.json")
T = [f"target_{i:02d}" for i in range(12)]
WCOLS = [f"wind_{i:02d}" for i in range(20)]
WG = dict(img_ridge=0.125, img_gbdt=0.25, omni=0.125, eswf=0.25)
WC = dict(cnn96b=0.125, cnn96w3=0.125)
KN8 = np.array([350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 720.0])
KN5 = np.array([380.0, 450.0, 520.0, 600.0, 700.0])
BINS = [(-np.inf, 450.0), (450.0, 550.0), (550.0, 650.0), (650.0, np.inf)]
BINLAB = ["<450", "450-550", "550-650", ">650"]
QS = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
RIDGE = 1e-3
NREP = 20
SEEDS = list(range(NREP))


def rmse(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def load(tag, split):
    return np.load(P / f"{split}_{tag}.npy").astype(np.float64)


def design(x, knots):
    x = np.asarray(x, np.float64).ravel()
    cols = [np.ones_like(x), x] + [np.maximum(x - k, 0.0) for k in knots]
    return np.column_stack(cols)


def fit_hinge(x, y, knots, r=RIDGE):
    A = design(x, knots)
    yt = np.asarray(y, np.float64).ravel()
    M = A.T @ A
    p = A.shape[1]
    M = M + r * np.eye(p) * (np.trace(M) / p)
    try:
        return np.linalg.solve(M, A.T @ yt)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(M, A.T @ yt, rcond=None)[0]


def apply_hinge(x, coef, knots):
    return (design(x, knots) @ coef).reshape(np.shape(x))


def pava(y, w):
    y = np.asarray(y, np.float64)
    w = np.asarray(w, np.float64)
    s = []
    for i in range(len(y)):
        s.append([w[i], w[i] * y[i], 1])
        while len(s) >= 2 and s[-2][1] / s[-2][0] > s[-1][1] / s[-1][0] + 1e-15:
            ww, su, c = s.pop()
            s[-1][0] += ww
            s[-1][1] += su
            s[-1][2] += c
    out = np.empty(len(y), np.float64)
    i = 0
    for ww, su, c in s:
        out[i:i + c] = su / ww
        i += c
    return out


def iso_fit(x, y):
    x = np.asarray(x, np.float64).ravel()
    y = np.asarray(y, np.float64).ravel()
    if x.size == 0:
        return np.array([0.0]), np.array([0.0])
    o = np.argsort(x, kind="mergesort")
    xs, ys = x[o], y[o]
    ux, inv = np.unique(xs, return_inverse=True)
    sy = np.zeros(len(ux))
    sw = np.zeros(len(ux))
    np.add.at(sy, inv, ys)
    np.add.at(sw, inv, 1.0)
    fy = pava(sy / sw, sw)
    return ux, fy


def iso_apply(ux, fy, xnew):
    return np.interp(np.asarray(xnew, np.float64).ravel(), ux, fy).reshape(np.shape(xnew))


class UF:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra

    def labels(self):
        n = len(self.p)
        for i in range(n):
            self.p[i] = self.find(i)
        _, lab = np.unique(self.p, return_inverse=True)
        return lab


def shift_edges(w, ks):
    n = w.shape[0]
    edges = []
    for k in ks:
        head = {}
        width = 20 - k
        for j in range(n):
            key = w[j, :width].tobytes()
            head.setdefault(key, []).append(j)
        for i in range(n):
            key = w[i, k:].tobytes()
            for j in head.get(key, []):
                if i != j:
                    edges.append((i, j))
    return edges


def comp_stats(lab):
    sizes = np.bincount(lab)
    return dict(
        n_comp=int(len(sizes)),
        n_singleton=int((sizes == 1).sum()),
        size_min=int(sizes.min()),
        size_med=float(np.median(sizes)),
        size_max=int(sizes.max()),
        max_frac=round(float(sizes.max() / sizes.sum()), 4),
        top10=sorted(sizes.tolist(), reverse=True)[:10],
    )


def group_folds(lab, n_folds, rng):
    sizes = np.bincount(lab)
    order = rng.permutation(len(sizes))
    load = np.zeros(n_folds)
    assign = np.zeros(len(sizes), np.int32)
    for c in order:
        f = int(np.argmin(load))
        assign[c] = f
        load[f] += sizes[c]
    return assign[lab], load.astype(int).tolist()


def row_folds(n, n_folds, rng):
    ix = rng.permutation(n)
    f = np.empty(n, np.int32)
    for i, lo in enumerate(np.array_split(ix, n_folds)):
        f[lo] = i
    return f


def oof_apply(pred, y, folds, kind, knots=None):
    out = np.empty_like(pred, np.float64)
    nf = int(folds.max()) + 1
    for f in range(nf):
        tr, te = folds != f, folds == f
        if te.sum() == 0 or tr.sum() == 0:
            out[te] = pred[te]
            continue
        xtr, ytr = pred[tr], y[tr]
        xte = pred[te]
        if kind == "hinge":
            c = fit_hinge(xtr, ytr, knots)
            out[te] = apply_hinge(xte, c, knots)
        elif kind == "iso":
            if xtr.size < 2:
                out[te] = xte
            else:
                ux, fy = iso_fit(xtr, ytr)
                out[te] = iso_apply(ux, fy, xte)
        else:
            raise ValueError(kind)
    return out


def delta_of(base, cal, y):
    return round(rmse(base, y) - rmse(cal, y), 4)


def bin_table(pred, base, cal, y):
    rows = []
    for (lo, hi), lab in zip(BINS, BINLAB):
        m = (pred >= lo) & (pred < hi)
        n = int(m.sum())
        if n == 0:
            rows.append(dict(bin=lab, n=0, rmse0=None, rmse1=None, d=None, sse_share=None, trust=False))
            continue
        r0, r1 = rmse(base[m], y[m]), rmse(cal[m], y[m])
        sse = float(np.sum((base - y) ** 2))
        rows.append(dict(
            bin=lab, n=n,
            rmse0=round(r0, 4), rmse1=round(r1, 4), d=round(r0 - r1, 4),
            sse_share=round(float(np.sum((base[m] - y[m]) ** 2) / sse), 4),
            trust=n >= 200,
        ))
    return rows


def qtable(a):
    a = np.asarray(a, np.float64).ravel()
    return {f"q{q}": round(float(np.percentile(a, q)), 3) for q in QS}


def mix_G(G, coef, knots, cnn):
    return 0.75 * apply_hinge(G, coef, knots) + cnn


print("== load", flush=True)
y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)
xin = pd.read_csv(ROOT / "validation/inputs.csv")
w = xin[WCOLS].to_numpy(np.float64)
n = len(y)
V = {k: load(k, "val") for k in list(WG) + list(WC)}
Tst = {k: load(k, "test") for k in list(WG) + list(WC)}
G = sum(w_ * V[k] for k, w_ in WG.items()) / 0.75
cnn = sum(w_ * V[k] for k, w_ in WC.items())
blend = 0.75 * G + cnn
Gte = sum(w_ * Tst[k] for k, w_ in WG.items()) / 0.75
cnnte = sum(w_ * Tst[k] for k, w_ in WC.items())
blend_te = 0.75 * Gte + cnnte
base = rmse(blend, y)
print("blend", round(base, 4), "shape", blend.shape, "test", blend_te.shape, flush=True)
assert blend.shape == y.shape == (n, 12)

# ── 1. neighbor graph ──
print("== graph", flush=True)
w4 = np.round(w, 4)
graphs = {}
for name, arr, ks in (
    ("g6", w, (1,)),
    ("g6r4", w4, (1,)),
    ("gany", w, tuple(range(1, 20))),
    ("ganyr4", w4, tuple(range(1, 20))),
):
    e = shift_edges(arr, ks)
    uf = UF(n)
    for a, b in e:
        uf.union(a, b)
    lab = uf.labels()
    st = comp_stats(lab)
    st["n_edge"] = int(len(e))
    st["n_undirected"] = int(len({tuple(sorted(p)) for p in e}))
    graphs[name] = dict(lab=lab, **{k: v for k, v in st.items()})
    print(name, {k: v for k, v in st.items() if k != "top10"}, "top10", st["top10"], flush=True)

adj = int(np.sum(np.all(w[1:, 1:] == w[:-1, :-1], axis=1)))
print("adjacent_row_6h", adj, "rate", round(adj / max(n - 1, 1), 6), flush=True)


def repeat_delta(pred, y, fold_fn, kind, knots, n_folds, seeds):
    ds = []
    last = None
    for s in seeds:
        rng = np.random.RandomState(s)
        folds = fold_fn(rng, n_folds)
        cal = oof_apply(pred, y, folds, kind, knots)
        ds.append(rmse(pred, y) - rmse(cal, y))
        last = (folds, cal)
    a = np.asarray(ds, np.float64)
    return dict(
        mean=round(float(np.mean(a)), 4),
        median=round(float(np.median(a)), 4),
        std=round(float(np.std(a, ddof=1) if len(a) > 1 else 0.0), 4),
        min=round(float(np.min(a)), 4),
        max=round(float(np.max(a)), 4),
        n=len(a),
        n_ge_0=int((a >= 0).sum()),
        n_ge_01=int((a >= 0.1).sum()),
        last_d=round(float(a[-1]), 4),
    ), last


def gpart_oof(G, cnn, y, folds, knots):
    """GBDT부 보정 OOF: 계수 적합은 G→y, 적용은 0.75*cal(G)+cnn."""
    out = np.empty_like(G)
    nf = int(folds.max()) + 1
    for f in range(nf):
        tr, te = folds != f, folds == f
        if te.sum() == 0 or tr.sum() == 0:
            out[te] = 0.75 * G[te] + cnn[te]
            continue
        c = fit_hinge(G[tr], y[tr], knots)
        out[te] = mix_G(G[te], c, knots, cnn[te])
    return out


def repeat_gpart(G, cnn, y, fold_fn, knots, n_folds, seeds, blend):
    ds = []
    last = None
    for s in seeds:
        rng = np.random.RandomState(s)
        folds = fold_fn(rng, n_folds)
        mix = gpart_oof(G, cnn, y, folds, knots)
        ds.append(rmse(blend, y) - rmse(mix, y))
        last = (folds, mix)
    a = np.asarray(ds, np.float64)
    return dict(
        mean=round(float(np.mean(a)), 4),
        median=round(float(np.median(a)), 4),
        std=round(float(np.std(a, ddof=1) if len(a) > 1 else 0.0), 4),
        min=round(float(np.min(a)), 4),
        max=round(float(np.max(a)), 4),
        n=len(a),
        n_ge_0=int((a >= 0).sum()),
        n_ge_01=int((a >= 0.1).sum()),
    ), last


print("== 1 leak splits", flush=True)
leak = {}
# random row 2-fold (원 교차로 재현)
def fn_row(rng, nf, n=n):
    return row_folds(n, nf, rng)

row2_full, last_row = repeat_delta(blend, y, fn_row, "hinge", KN8, 2, SEEDS)
row2_g, last_rowg = repeat_gpart(G, cnn, y, fn_row, KN8, 2, SEEDS, blend)
row5_full, last_row5 = repeat_delta(blend, y, fn_row, "hinge", KN8, 5, [0])
print("row2 full", row2_full, "gpart", row2_g, "row5", row5_full, flush=True)
leak["row2_full_h8"] = row2_full
leak["row2_gpart_h8"] = row2_g
leak["row5_full_h8"] = row5_full

# component-grouped
for gname in ("g6", "g6r4", "gany", "ganyr4"):
    lab = graphs[gname]["lab"]

    def fn_g(rng, nf, lab=lab):
        f, load = group_folds(lab, nf, rng)
        return f

    d2, last2 = repeat_delta(blend, y, fn_g, "hinge", KN8, 2, SEEDS)
    dg, lastg = repeat_gpart(G, cnn, y, fn_g, KN8, 2, SEEDS, blend)
    d5, last5 = repeat_delta(blend, y, fn_g, "hinge", KN8, 5, [0])
    d5g, last5g = repeat_gpart(G, cnn, y, fn_g, KN8, 5, [0], blend)
    leak[f"{gname}_2_full_h8"] = d2
    leak[f"{gname}_2_gpart_h8"] = dg
    leak[f"{gname}_5_full_h8"] = d5
    leak[f"{gname}_5_gpart_h8"] = d5g
    print(gname, "2full", d2, "2g", dg, "5full", d5, "5g", d5g, flush=True)

# in-sample 상한 (채택 아님)
c_all = fit_hinge(blend, y, KN8)
ins_full = round(base - rmse(apply_hinge(blend, c_all, KN8), y), 4)
cG = fit_hinge(G, y, KN8)
ins_g = round(base - rmse(mix_G(G, cG, KN8, cnn), y), 4)
leak["insample_full_h8"] = ins_full
leak["insample_gpart_h8"] = ins_g
print("insample full", ins_full, "gpart", ins_g, flush=True)

# 대표 OOF: g6 5-fold seed 0 (6h 이웃 성분 유지)
lab_main = graphs["g6"]["lab"]
folds_main, load_main = group_folds(lab_main, 5, np.random.RandomState(0))
cal_full = oof_apply(blend, y, folds_main, "hinge", KN8)
cal_g = gpart_oof(G, cnn, y, folds_main, KN8)
print("main g6-5 load", load_main,
      "full", round(base - rmse(cal_full, y), 4),
      "gpart", round(base - rmse(cal_g, y), 4), flush=True)

# 누수 교차로 OOF (row 5-fold) — 가짜 이득이 어느 구간인지
folds_row5 = row_folds(n, 5, np.random.RandomState(0))
cal_row = oof_apply(blend, y, folds_row5, "hinge", KN8)
cal_rowg = gpart_oof(G, cnn, y, folds_row5, KN8)

# 성분 간 k>1 간선 (g6 성분을 잇는 창 중첩)
lab6 = graphs["g6"]["lab"]
e_any = shift_edges(w, tuple(range(1, 20)))
cross = sum(1 for a, b in e_any if lab6[a] != lab6[b])
print("cross_comp_edges", cross, "of", len(e_any), flush=True)

# 성분별 통계
comp_rows = []
for c in range(int(lab6.max()) + 1):
    m = lab6 == c
    comp_rows.append(dict(
        c=c, n=int(m.sum()),
        y_mean=round(float(y[m].mean()), 2),
        blend_mean=round(float(blend[m].mean()), 2),
        G_mean=round(float(G[m].mean()), 2),
        rmse=round(rmse(blend[m], y[m]), 3),
    ))
print("comps", comp_rows, flush=True)

# 체인 시간순 + gap 분할 (성분 내부, 6h 경로 정렬)
succ = {i: [] for i in range(n)}
indeg = np.zeros(n, np.int32)
for i, j in shift_edges(w, (1,)):
    # i.wind[1:]==j.wind[:-1] 이면 i 가 6h 앞
    # shift_edges(k=1): tail_i == head_j → i then j
    succ[i].append(j)
    indeg[j] += 1
chain_order = []
for c in range(int(lab6.max()) + 1):
    members = np.where(lab6 == c)[0]
    roots = [int(i) for i in members if indeg[i] == 0]
    if len(roots) != 1:
        # fallback: pick min indeg
        roots = [int(members[np.argmin(indeg[members])])]
    order = []
    seen = set()
    cur = roots[0]
    while cur is not None and cur not in seen:
        seen.add(cur)
        order.append(cur)
        nxts = [t for t in succ[cur] if lab6[t] == c and t not in seen]
        cur = nxts[0] if nxts else None
    chain_order.append(np.array(order, np.int32))
print("chain_lens", [len(o) for o in chain_order], flush=True)

GAP = 32  # groups.py GAP_MIN 192h
folds_gap = np.full(n, -1, np.int32)
for order in chain_order:
    L = len(order)
    a = max(L // 3, 1)
    # A | gap | B
    b0 = min(a + GAP, L)
    folds_gap[order[:a]] = 0
    folds_gap[order[b0:]] = 1
nA, nB, nG = int((folds_gap == 0).sum()), int((folds_gap == 1).sum()), int((folds_gap < 0).sum())
print("gap_split nA", nA, "nB", nB, "gap", nG, flush=True)
use = folds_gap >= 0
cal_gap = np.array(blend, copy=True)
if nA and nB:
    cal_gap[use] = oof_apply(blend[use], y[use], folds_gap[use], "hinge", KN8)
d_gap = round(rmse(blend[use], y[use]) - rmse(cal_gap[use], y[use]), 4) if nA and nB else None
mix_gap = np.array(blend, copy=True)
if nA and nB:
    mix_gap[use] = gpart_oof(G[use], cnn[use], y[use], folds_gap[use], KN8)
d_gap_g = round(rmse(blend[use], y[use]) - rmse(mix_gap[use], y[use]), 4) if nA and nB else None
print("gap hinge8 full", d_gap, "gpart", d_gap_g, "subset_rmse0", round(rmse(blend[use], y[use]), 4), flush=True)
leak["g6_chain_gap32_full_h8"] = d_gap
leak["g6_chain_gap32_gpart_h8"] = d_gap_g
leak["g6_chain_gap32_n"] = dict(A=nA, B=nB, gap=nG)
leak["cross_comp_edges"] = cross

# fold별 g6-5
fold_rows = []
for f in range(5):
    te = folds_main == f
    fold_rows.append(dict(
        f=f, n=int(te.sum()),
        rmse0=round(rmse(blend[te], y[te]), 4),
        rmse1=round(rmse(cal_full[te], y[te]), 4),
        d=round(rmse(blend[te], y[te]) - rmse(cal_full[te], y[te]), 4),
    ))
print("g6-5 per fold", fold_rows, flush=True)

# ── 2. bin stability (g6 5-fold OOF vs row 5-fold) ──
print("== 2 bins", flush=True)
bins_full = bin_table(blend, blend, cal_full, y)
bins_g = bin_table(blend, blend, cal_g, y)
bins_row = bin_table(blend, blend, cal_row, y)
bins_rowg = bin_table(blend, blend, cal_rowg, y)
print("g6-5 full", bins_full, flush=True)
print("g6-5 gpart", bins_g, flush=True)
print("row5 full", bins_row, flush=True)
print("row5 gpart", bins_rowg, flush=True)

# ── 3. test vs val distribution ──
print("== 3 dist", flush=True)
dist = dict(
    val=qtable(blend),
    test=qtable(blend_te),
    val_G=qtable(G),
    test_G=qtable(Gte),
    val_n=int(blend.size),
    test_n=int(blend_te.size),
    val_lt350=round(float((blend < 350).mean()), 4),
    val_gt720=round(float((blend > 720).mean()), 4),
    test_lt350=round(float((blend_te < 350).mean()), 4),
    test_gt720=round(float((blend_te > 720).mean()), 4),
    val_G_lt350=round(float((G < 350).mean()), 4),
    val_G_gt720=round(float((G > 720).mean()), 4),
    test_G_lt350=round(float((Gte < 350).mean()), 4),
    test_G_gt720=round(float((Gte > 720).mean()), 4),
    mean_val=round(float(blend.mean()), 3),
    mean_test=round(float(blend_te.mean()), 3),
    std_val=round(float(blend.std()), 3),
    std_test=round(float(blend_te.std()), 3),
)
# quantile diffs
dist["qdiff_test_minus_val"] = {k: round(dist["test"][k] - dist["val"][k], 3) for k in dist["val"]}
print("val q", dist["val"], flush=True)
print("test q", dist["test"], flush=True)
print("qdiff", dist["qdiff_test_minus_val"], flush=True)
print("out knot val", dist["val_lt350"], dist["val_gt720"],
      "test", dist["test_lt350"], dist["test_gt720"], flush=True)

# ── 4. form comparison, same splits ──
print("== 4 forms", flush=True)
forms = {}
for split_name, folds in (("g65", folds_main), ("row5", folds_row5)):
    for form, kind, kn in (
        ("hinge8", "hinge", KN8),
        ("hinge5", "hinge", KN5),
        ("iso", "iso", None),
    ):
        cal = oof_apply(blend, y, folds, kind, kn)
        d = delta_of(blend, cal, y)
        forms[f"{split_name}_{form}_full"] = d
        print(split_name, form, "full", d, "rmse", round(rmse(cal, y), 4), flush=True)
    for form, kn in (("hinge8", KN8), ("hinge5", KN5)):
        mix = gpart_oof(G, cnn, y, folds, kn)
        d = delta_of(blend, mix, y)
        forms[f"{split_name}_{form}_gpart"] = d
        print(split_name, form, "gpart", d, flush=True)
    mix_iso = np.empty_like(G)
    nf = int(folds.max()) + 1
    for f in range(nf):
        tr, te = folds != f, folds == f
        if te.sum() == 0:
            mix_iso[te] = 0.75 * G[te] + cnn[te]
            continue
        if tr.sum() == 0:
            mix_iso[te] = 0.75 * G[te] + cnn[te]
            continue
        ux, fy = iso_fit(G[tr], y[tr])
        mix_iso[te] = 0.75 * iso_apply(ux, fy, G[te]) + cnn[te]
    forms[f"{split_name}_iso_gpart"] = delta_of(blend, mix_iso, y)
    print(split_name, "iso gpart", forms[f"{split_name}_iso_gpart"], flush=True)

lab = graphs["g6"]["lab"]

def fn_g6(rng, nf, lab=lab):
    f, _ = group_folds(lab, nf, rng)
    return f

for form, kind, kn in (("hinge8", "hinge", KN8), ("hinge5", "hinge", KN5), ("iso", "iso", None)):
    d2, _ = repeat_delta(blend, y, fn_g6, kind, kn, 2, SEEDS)
    forms[f"g62_{form}_full_rep"] = d2
    print("g62", form, d2, flush=True)

for form, kind, kn in (("hinge8", "hinge", KN8), ("hinge5", "hinge", KN5), ("iso", "iso", None)):
    d2, _ = repeat_delta(blend, y, fn_row, kind, kn, 2, SEEDS)
    forms[f"row2_{form}_full_rep"] = d2
    print("row2", form, d2, flush=True)

# ── dump ──
report = dict(
    blend_rmse=round(base, 4),
    n_val=n,
    n_test=int(blend_te.shape[0]),
    graph={k: {kk: vv for kk, vv in v.items() if kk != "lab"} for k, v in graphs.items()},
    adjacent_row_6h=adj,
    leak=leak,
    main_g65_load=load_main,
    main_g65_full=round(base - rmse(cal_full, y), 4),
    main_g65_gpart=round(base - rmse(cal_g, y), 4),
    main_g65_folds=fold_rows,
    comps=comp_rows,
    bins_g65_full=bins_full,
    bins_g65_gpart=bins_g,
    bins_row5_full=bins_row,
    bins_row5_gpart=bins_rowg,
    dist=dist,
    forms=forms,
)
OUT.write_text(json.dumps(report, indent=1))
print("saved", OUT, flush=True)
print("DONE", flush=True)
