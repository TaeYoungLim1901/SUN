"""정직한 그룹 분할 — 공유 PNG 로 상대시각을 잇고, 시간 블록 + 퍼지로 fold 를 만든다.

누수 경로 둘 (NEXT.md §3.1, SUN_PLAN Day2-1):
  1. 입력 창 겹침 = 같은 PNG 를 공유 (≤120h)
  2. A 의 target 이 B 의 input 과 겹침 = 공유 PNG 없이도 |Δt0| < 192h

(2) 를 막으려면 상대시각이 필요하다. train 샘플끼리 PNG 프레임 인덱스로
t0 오프셋을 붙인다. test 시계열 복원이 아니다 (RULES.md §5-3 은 test 금지).

산출:
  work/preds/train_comp.npy    연결성분 id
  work/preds/train_tstep.npy   성분 안 상대시각 (6h 스텝, 성분별 최소=0)
  work/preds/train_fold.npy    시간순 n_folds 등분 (퍼지 전 배정)
  work/preds/groups.json       진단
"""
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
OUT = pathlib.Path("/home/jovyan/work/preds")
OUT.mkdir(parents=True, exist_ok=True)
IMG = [f"image_{i:02d}" for i in range(20)]
N_FOLDS = 5
# 27.2753일 = 109.1 스텝. kit.block_folds gap=1 캐링턴 블록과 같다.
GAP_STEPS = 109
# target↔input 겹침만 막는 하한 (192h / 6h)
GAP_MIN = 32


class OffsetUF:
    """t0[i] = t0[root] + d[i]  (단위: 6h 스텝). 반복 find — 사슬이면 재귀 한도 초과."""

    def __init__(self, n):
        self.p = np.arange(n)
        self.d = np.zeros(n, np.int64)
        self.rank = np.zeros(n, np.int8)
        self.conflicts = 0

    def find(self, i):
        path = []
        while self.p[i] != i:
            path.append(i)
            i = int(self.p[i])
        acc = 0
        for j in reversed(path):
            acc += int(self.d[j])
            self.d[j] = acc
            self.p[j] = i
        return i, (0 if not path else int(self.d[path[0]]))

    def union(self, a, b, t0a_minus_t0b):
        ra, da = self.find(a)
        rb, db = self.find(b)
        if ra == rb:
            if da - db != t0a_minus_t0b:
                self.conflicts += 1
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb, da, db, t0a_minus_t0b = rb, ra, db, da, -t0a_minus_t0b
        self.p[rb] = ra
        self.d[rb] = da - db - t0a_minus_t0b
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def build_timeline(inputs):
    """같은 PNG 가 frame f_a, f_b 에 있으면 t0_a - t0_b = f_b - f_a."""
    n = len(inputs)
    names = inputs[IMG].to_numpy()
    pos = {}
    for s in range(n):
        for f in range(20):
            pos.setdefault(names[s, f], []).append((s, f))
    uf = OffsetUF(n)
    for lst in pos.values():
        s0, f0 = lst[0]
        for s, f in lst[1:]:
            uf.union(s0, s, f - f0)
    comp = np.empty(n, np.int32)
    tstep = np.empty(n, np.int64)
    for i in range(n):
        r, off = uf.find(i)
        comp[i] = r
        tstep[i] = off
    remap = {r: k for k, r in enumerate(sorted(set(comp.tolist())))}
    comp = np.array([remap[c] for c in comp], np.int32)
    for c in range(comp.max() + 1):
        m = comp == c
        tstep[m] -= tstep[m].min()
    return comp, tstep, uf.conflicts, pos


def assign_folds(comp, tstep, n_folds=N_FOLDS):
    """성분마다 시간순으로 n_folds 등분. 작은 성분은 통째로 한 fold."""
    fold = np.full(len(comp), -1, np.int32)
    for c in range(int(comp.max()) + 1):
        idx = np.where(comp == c)[0]
        order = idx[np.argsort(tstep[idx], kind="stable")]
        if len(order) < n_folds:
            fold[order] = 0
            continue
        for k, ch in enumerate(np.array_split(order, n_folds)):
            fold[ch] = k
    return fold


def iter_folds(comp, tstep, fold, gap_steps=GAP_STEPS):
    """val = fold k. train 에서 같은 성분·|Δt|<gap 인 샘플은 뺀다."""
    n_folds = int(fold.max()) + 1
    for k in range(n_folds):
        val = fold == k
        train = ~val
        for c in np.unique(comp[val]):
            same = np.where(train & (comp == c))[0]
            if same.size == 0:
                continue
            vt = np.sort(tstep[val & (comp == c)])
            tt = tstep[same]
            # tt 가 vt 의 어떤 값과도 gap 안이면 탈락
            j = np.searchsorted(vt, tt)
            close = np.zeros(same.size, bool)
            for side in (j, j - 1):
                ok = (side >= 0) & (side < len(vt))
                close[ok] |= np.abs(tt[ok] - vt[side[ok]]) < gap_steps
            train[same[close]] = False
        yield train, val


def shared_png(inputs, a, b):
    sa = set(inputs.loc[a, IMG].to_numpy().ravel())
    sb = set(inputs.loc[b, IMG].to_numpy().ravel())
    return len(sa & sb)


def diagnose(inputs, comp, tstep, fold, conflicts):
    n = len(comp)
    sizes = np.bincount(comp)
    spans = [int(tstep[comp == c].max()) + 1 for c in range(len(sizes))]
    print(f"n={n} 성분={len(sizes)} 충돌={conflicts} "
          f"최대성분={sizes.max()} ({100 * sizes.max() / n:.1f}%) "
          f"단독={int((sizes == 1).sum())}", flush=True)
    print("상위 성분 크기/span(스텝):",
          [(int(s), sp) for s, sp in sorted(zip(sizes, spans), reverse=True)[:8]],
          flush=True)

    leak_png = []
    leak_dt = {GAP_MIN: [], GAP_STEPS: []}
    fold_rows = []
    for k, (tr, va) in enumerate(iter_folds(comp, tstep, fold, GAP_STEPS)):
        n_png = shared_png(inputs, tr, va)
        dts = []
        for c in np.unique(comp[va]):
            if not (tr & (comp == c)).any():
                continue
            vt = tstep[va & (comp == c)]
            tt = tstep[tr & (comp == c)]
            # 성분 안 최근접
            for t in (vt.min(), vt.max()):
                dts.append(int(np.min(np.abs(tt - t))))
        mindt = min(dts) if dts else None
        leak_png.append(n_png)
        fold_rows.append((k, int(tr.sum()), int(va.sum()), n_png, mindt))
        print(f"  fold{k} train={tr.sum():5d} val={va.sum():5d} "
              f"공유PNG={n_png} min|Δt|={mindt}", flush=True)
        if mindt is not None:
            leak_dt[GAP_MIN].append(mindt < GAP_MIN)
            leak_dt[GAP_STEPS].append(mindt < GAP_STEPS)

    # 파일명은 스플릿 디렉터리 안 로컬 번호(image_000694.png 가 train/val 에 둘 다 있다).
    # 디렉터리가 다르므로 교집합은 같은 파일이 아니다. 해시는 재지 않는다.
    va_x = pd.read_csv(ROOT / "validation/inputs.csv")
    te_x = pd.read_csv(ROOT / "test/inputs.csv")
    print(f"파일명 스킴 train={inputs[IMG[0]].iloc[0]} val={va_x[IMG[0]].iloc[0]} "
          f"(스플릿별 로컬 번호 — 교집합 해석 금지)", flush=True)
    assert not any(leak_png), f"fold 사이 공유 PNG: {leak_png}"
    assert not any(leak_dt[GAP_STEPS]), "퍼지 실패: 27일 갭이 안 생겼다"
    return {
        "n": n,
        "n_comp": int(len(sizes)),
        "conflicts": int(conflicts),
        "max_comp": int(sizes.max()),
        "singletons": int((sizes == 1).sum()),
        "sizes_top": [int(s) for s in sorted(sizes, reverse=True)[:12]],
        "spans_top": [sp for _, sp in sorted(zip(sizes, spans), reverse=True)[:12]],
        "folds": [{"k": k, "train": tr, "val": va, "shared_png": p, "min_dt": d}
                  for k, tr, va, p, d in fold_rows],
        "n_folds": N_FOLDS,
        "gap_steps": GAP_STEPS,
    }


def selfcheck():
    """창이 1스텝씩 밀리는 합성 시계열에서 공유 PNG / 퍼지를 검증."""
    n, w = 200, 20
    rows = []
    for i in range(n):
        rows.append({f"image_{f:02d}": f"t{i + f:04d}" for f in range(w)})
    inp = pd.DataFrame(rows)
    comp, tstep, conf, _ = build_timeline(inp)
    assert conf == 0 and len(set(comp)) == 1
    assert tstep[0] == 0 and tstep[-1] == n - 1
    fold = assign_folds(comp, tstep, 4)
    # 창 길이 20 + target 12 → 공유 PNG 차단은 20, target↔input 은 32.
    for tr, va in iter_folds(comp, tstep, fold, gap_steps=32):
        assert shared_png(inp, tr, va) == 0
        assert not (tr & va).any()
        mindt = np.min(np.abs(tstep[tr][:, None] - tstep[va][None, :]))
        assert mindt >= 32, mindt
    print("groups selfcheck OK", flush=True)


if __name__ == "__main__":
    selfcheck()
    tr_x = pd.read_csv(ROOT / "train/inputs.csv")
    comp, tstep, conflicts, _ = build_timeline(tr_x)
    fold = assign_folds(comp, tstep)
    info = diagnose(tr_x, comp, tstep, fold, conflicts)
    np.save(OUT / "train_comp.npy", comp)
    np.save(OUT / "train_tstep.npy", tstep)
    np.save(OUT / "train_fold.npy", fold)
    (OUT / "groups.json").write_text(json.dumps(info, indent=2))
    print("saved", OUT / "groups.json", flush=True)
