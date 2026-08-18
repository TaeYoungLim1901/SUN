"""제출 3종을 서버에서 로컬 `submit/` 로 내려받는다. 사람은 그 폴더만 업로드하면 된다.

내려받기 전에 서버에서 `work/bugcheck.py` 를 돌린다. **검사가 실패하면 아무것도 받지 않는다** —
깨진 파일을 손에 쥐고 업로드하는 게 제일 나쁜 결말이다.

    python tools/submit.py            # 검사 → 내려받기
    python tools/submit.py --force    # 검사 실패해도 받는다 (진단용)
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import remote  # noqa: E402

FILES = ("code.ipynb", "model.pth", "submission.csv")
LOCAL = pathlib.Path(__file__).resolve().parent.parent / "submit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="버그 검사 실패해도 내려받는다")
    a = ap.parse_args()

    r = remote.Remote()
    print("서버 버그 검사 (work/bugcheck.py)")
    out, _ = r.run(
        "import subprocess;"
        "p=subprocess.run('cd /home/jovyan && python work/bugcheck.py',"
        "shell=True,capture_output=True,text=True);"
        "print(p.stdout);print(p.stderr[-500:]);print('EXIT', p.returncode)",
        timeout=300, capture=True)
    print(out)
    if "EXIT 0" not in out and not a.force:
        sys.exit("검사 실패. 고치고 다시 돌려라. (--force 로 무시 가능)")

    LOCAL.mkdir(exist_ok=True)
    for f in FILES:
        r.get(f"submission/{f}", str(LOCAL / f))
    print(f"\n받았다: {LOCAL}")
    for f in FILES:
        p = LOCAL / f
        print(f"  {f:<16} {p.stat().st_size/1e6:8.2f} MB")
    print("\n이 폴더의 파일 3개를 대회 사이트에 그대로 올려라. 이름 바꾸지 마라.")


if __name__ == "__main__":
    main()
