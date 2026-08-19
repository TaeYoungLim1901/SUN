"""pipeline.py 에 GBDT부 조각선형 보정을 넣는다. 게이트 통과 후에만 실행할 것.
계수는 model.pth blob['calib_G'] 에서 읽고, 없으면 보정 없이 기존 동작 그대로."""
import pathlib
p = pathlib.Path("/home/jovyan/work/pipeline.py")
s = p.read_text()
OLD = """    p = (WEIGHTS["ridge_v1"] * blob["ridge_v1"].predict(f1)
         + WEIGHTS["gbdt_v1"] * np.column_stack([m.predict(f1) for m in blob["gbdt_v1"]])
         + WEIGHTS["gbdt_v2"] * pj
         + WEIGHTS["eswf"] * pe)"""
NEW = """    wg = WEIGHTS["ridge_v1"] + WEIGHTS["gbdt_v1"] + WEIGHTS["gbdt_v2"] + WEIGHTS["eswf"]
    g = (WEIGHTS["ridge_v1"] * blob["ridge_v1"].predict(f1)
         + WEIGHTS["gbdt_v1"] * np.column_stack([m.predict(f1) for m in blob["gbdt_v1"]])
         + WEIGHTS["gbdt_v2"] * pj
         + WEIGHTS["eswf"] * pe) / wg
    p = wg * calib_apply(g)"""
assert OLD in s, "predict() 본문이 예상과 다르다"
s = s.replace(OLD, NEW, 1)
HELP = '''

# ── GBDT부 조각선형 보정 (계수는 train 정직 폴드 OOF 에서 적합, validation 미사용) ──
CALIB_KNOTS = np.array([350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 720.0])


def calib_design(x):
    return np.column_stack([np.ones_like(x), x] + [np.maximum(x - k, 0.0) for k in CALIB_KNOTS])


def calib_apply(g):
    """계수가 없으면 항등. 있으면 예측 수준별 조건부 편향을 편다."""
    c = globals().get("CALIB_G")
    if c is None:
        return g
    return (calib_design(np.asarray(g, np.float64).ravel()) @ c).reshape(np.shape(g))

'''
i = s.find("def predict(split, inputs):")
assert i > 0
s = s[:i] + HELP.lstrip("\n") + "\n" + s[i:]
j = s.find('CLIP = blob.get("clip", CLIP)')
assert j > 0
s = s[:j] + 'CALIB_G = blob.get("calib_G")\n' + s[j:]
p.write_text(s)
print("patched pipeline.py")
