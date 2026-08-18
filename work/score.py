"""채점기 겸 제출 빌더.

하는 일 넷:
  1. `work/preds/val_*.npy` 를 전부 채점해 순위표로 찍는다 (`kit.score` 하나만 쓴다).
  2. `kit.hill_climb` 로 결합 가중치를 찾고 test 에 같은 가중치를 적용한다.
  3. 현재 리더보드 표와 대조해 예상 등수를 찍는다.
  4. `--write` 를 주면 `submission/` 에 제출 3종을 만든다 + 형식 버그 검사를 통과해야만 저장한다.

규정: validation 은 채점·모델선택 전용. 학습에 넣지 않는다 (RULES.md §1 7항).
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jovyan/work")
import kit  # noqa: E402

ROOT = pathlib.Path("/home/jovyan/public_dataset/competition_dataset_6h")
PRED = pathlib.Path("/home/jovyan/work/preds")
SUB = pathlib.Path("/home/jovyan/submission")
T = [f"target_{i:02d}" for i in range(12)]
W = [f"wind_{i:02d}" for i in range(20)]
# 2026-08-18 공개 리더보드 상위 10 (사람이 화면에서 옮겨 적은 값)
LB = [57.6114, 59.3444, 59.5471, 60.1962, 62.3393, 72.9812, 73.4683, 75.4502, 75.8028, 75.8100]


def rank_of(x):
    better = sum(1 for v in LB if v < x)
    return better + 1 if better < len(LB) else f">{len(LB)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="submission/ 에 제출 3종 생성")
    ap.add_argument("--only", nargs="*", help="이 이름들만 결합 후보로 쓴다")
    ap.add_argument("--clip", nargs=2, type=float, default=(200.0, 900.0),
                    help="물리 범위 clipping (RULES.md §6-4)")
    a = ap.parse_args()

    va_x = pd.read_csv(ROOT / "validation/inputs.csv")
    te_x = pd.read_csv(ROOT / "test/inputs.csv")
    y = pd.read_csv(ROOT / "validation/targets.csv")[T].to_numpy(np.float64)

    vals, tests = {}, {}
    for f in sorted(PRED.glob("val_*.npy")):
        name = f.stem[4:]
        tf = PRED / f"test_{name}.npy"
        if not tf.exists():
            print(f"  건너뜀 {name}: test 예측 없음")
            continue
        v, t = np.load(f), np.load(tf)
        # 버그 검사: 모양·NaN·행 수가 하나라도 어긋나면 후보에서 뺀다
        if v.shape != (len(va_x), 12) or t.shape != (len(te_x), 12):
            print(f"  건너뜀 {name}: shape {v.shape} {t.shape}")
            continue
        if not (np.isfinite(v).all() and np.isfinite(t).all()):
            print(f"  건너뜀 {name}: NaN/inf")
            continue
        vals[name], tests[name] = v, t
    if a.only:
        vals = {k: v for k, v in vals.items() if k in a.only}
        tests = {k: v for k, v in tests.items() if k in a.only}
    assert vals, "채점할 예측이 없다"

    persist = np.repeat(va_x[W[-1]].to_numpy(np.float64)[:, None], 12, axis=1)
    print("\n== 단일 모델 ==")
    print(f"{'이름':<14}{'RMSE':>9}{'MAE':>8}{'예상등수':>9}   horizon 6h..72h")
    rows = sorted(vals.items(), key=lambda kv: kit.score(y, kv[1]))
    for name, v in [("persistence", persist)] + rows:
        s = kit.score(y, v)
        per = kit.rmse_per_horizon(y, v)
        print(f"{name:<14}{s:9.3f}{np.mean(np.abs(v - y)):8.2f}{str(rank_of(s)):>9}   "
              + " ".join(f"{x:.0f}" for x in per))

    weights, blend_rmse = kit.hill_climb(vals, y, n_iter=80)
    print("\n== 결합 (kit.hill_climb) ==")
    for n, w in sorted(weights.items(), key=lambda kv: -kv[1]):
        print(f"  {n:<14} {w:.3f}")
    vb = sum(w * vals[n] for n, w in weights.items())
    tb = sum(w * tests[n] for n, w in weights.items())
    lo, hi = a.clip
    vc, tc = np.clip(vb, lo, hi), np.clip(tb, lo, hi)
    print(f"  결합 RMSE      {blend_rmse:.3f}   예상등수 {rank_of(blend_rmse)}")
    print(f"  clip[{lo:.0f},{hi:.0f}] 후 {kit.score(y, vc):.3f}")
    if len(vals) > 1:
        top = rows[0][0]
        for n, v in rows[1:4]:
            print(f"  잔차상관 {top}~{n}: {kit.resid_corr(vals[top], v, y):.3f}"
                  "  (0.84 이상이면 결합 이득 없음)")
    print("\n== 속도대별 분해 (결합 기준) ==")
    names = {0: "<400", 1: "400-500", 2: "500-600", 3: ">600"}
    print(f"  {'구간':<10}{'셀수':>8}{'RMSE':>9}{'편향':>9}")
    for band, n, r, bias in kit.breakdown(y, vc, kit.speed_band(y)):
        print(f"  {names.get(int(band), band):<10}{n:>8}{r:>9.2f}{bias:>9.2f}")

    (PRED / "score_report.json").write_text(json.dumps(
        {"single": {n: round(kit.score(y, v), 3) for n, v in vals.items()},
         "weights": weights, "blend": round(blend_rmse, 3),
         "blend_clipped": round(kit.score(y, vc), 3)}, indent=2, ensure_ascii=False))

    if not a.write:
        print("\n(--write 없이 채점만 했다. 제출 파일은 안 만들었다)")
        return

    SUB.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame(tc, columns=T)
    sub.insert(0, "sample_id", te_x.sample_id.tolist())
    # 제출 형식 버그 검사 — 하나라도 틀리면 저장하지 않는다 (RULES.md §0)
    ids = pd.read_csv(ROOT / "test/test_ids.csv").sample_id.tolist()
    assert sub.columns.tolist() == ["sample_id"] + T, sub.columns.tolist()
    assert len(sub) == 3868, len(sub)
    assert sub.sample_id.tolist() == ids, "test_ids.csv 와 순서가 다르다"
    assert sub.sample_id.is_unique
    assert np.isfinite(sub[T].to_numpy()).all()
    assert float(sub[T].to_numpy().min()) >= lo and float(sub[T].to_numpy().max()) <= hi
    sub.to_csv(SUB / "submission.csv", index=False)
    (PRED / "blend_weights.json").write_text(json.dumps(weights, indent=2))
    print(f"\n제출 저장: {SUB/'submission.csv'}  shape={sub.shape}")
    print(sub.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
