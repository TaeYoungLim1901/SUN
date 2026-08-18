"""pack_out 3종이 준비되면 submission/ 으로 복사하고 bugcheck 한다.

제출 폴더를 학습 중간에 덮지 않는다. 이 스크립트가 복사하는 시점에만 바뀐다.
"""
import pathlib
import shutil
import subprocess
import sys

SRC = pathlib.Path("/home/jovyan/work/pack_out")
DST = pathlib.Path("/home/jovyan/submission")
NEED = ("code.ipynb", "model.pth", "submission.csv")


def main():
    missing = [n for n in NEED if not (SRC / n).exists()]
    if missing:
        sys.exit("없음: " + ", ".join(missing))
    DST.mkdir(parents=True, exist_ok=True)
    for n in NEED:
        shutil.copy2(SRC / n, DST / n)
        print("copy", n, (DST / n).stat().st_size)
    p = subprocess.run(["python", "/home/jovyan/work/bugcheck.py"], cwd="/home/jovyan")
    sys.exit(p.returncode)


if __name__ == "__main__":
    main()
