"""본문 파일 + YOUTUBE.md 원본 구간(원문 보존)을 합쳐 하나의 치환용 파일을 만든다.

원문을 손으로 옮겨 적지 않고 파일에서 직접 잘라 붙이므로 한 글자도 안 틀린다.

사용: python tools/assemble.py <본문파일> <출력파일> <제목1> <시작1> <끝1> [<제목2> <시작2> <끝2> ...]
"""
import pathlib
import sys

TARGET = pathlib.Path("D:/hackathon/SUN/YOUTUBE.md")
lines = TARGET.read_text(encoding="utf-8").splitlines(keepends=True)

body = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
out = pathlib.Path(sys.argv[2])

parts = [body.rstrip("\n"), ""]
args = sys.argv[3:]
for i in range(0, len(args), 3):
    title, s, e = args[i], int(args[i + 1]), int(args[i + 2])
    parts.append(f"### {title}")
    parts.append("")
    parts.append("".join(lines[s - 1 : e]).rstrip("\n"))
    parts.append("")

out.write_text("\n".join(parts) + "\n", encoding="utf-8")
print(f"조립 완료: {out} ({len(out.read_text(encoding='utf-8').splitlines())}행)")
