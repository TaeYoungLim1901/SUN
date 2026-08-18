"""로컬 CPU 정적 검사. 데이터/서버 불필요."""
import ast
import json
import pathlib
import re
import sys

root = pathlib.Path(__file__).resolve().parents[1]
py = (root / "work/pipeline.py").read_text(encoding="utf-8")
ast.parse(py)
nb = json.loads((root / "work/code.ipynb").read_text(encoding="utf-8"))
src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
bad = re.findall(
    r"^\s*(?:from|import)\s+(kit|cnn|imgfeat\w*|windonly|score|bugcheck|eswf|feats|groups|pack)\b",
    src, re.M)
print("cells", len(nb["cells"]))
print("local_imports", bad or "none")
print("model.pth", "model.pth" in src)
print("submission.csv", "submission.csv" in src)
print("pretrained", bool(re.search(r"pretrained\s*=\s*True", src)))
m = re.search(r'MODE = os.environ.get\("SW_MODE", "(\w+)"\)', src)
print("MODE", m.group(1) if m else "?")
ok = (not bad) and ("model.pth" in src) and ("submission.csv" in src) and m and m.group(1) == "infer"
sys.exit(0 if ok else 1)
