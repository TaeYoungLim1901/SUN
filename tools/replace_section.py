"""YOUTUBE.md 의 [start,end] 행(1-based, 양끝 포함)을 새 파일 내용으로 치환한다.

치환 전에 **원문 손실 검사**를 돌린다: 치환 대상 원문의 각 줄이 새 내용 어딘가에
그대로 남아 있는지 확인하고, 빠진 줄이 있으면 아무것도 쓰지 않고 중단한다.
(2단 병기 원칙 ① — 원문은 한 글자도 삭제하지 않는다)

사용: python tools/replace_section.py <start> <end> <new_content_file> [--force]
"""
import pathlib
import sys

TARGET = pathlib.Path("D:/hackathon/SUN/YOUTUBE.md")

start, end, newfile = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
force = "--force" in sys.argv

lines = TARGET.read_text(encoding="utf-8").splitlines(keepends=True)
old = lines[start - 1 : end]
new = pathlib.Path(newfile).read_text(encoding="utf-8")

missing = [
    ln for ln in (l.strip() for l in old)
    if ln and not ln.startswith("#") and ln not in new
]
if missing:
    print(f"원문 손실 검사 실패 — 새 내용에서 사라진 줄 {len(missing)}개:")
    for m in missing[:40]:
        print("  -", m[:110])
    if not force:
        sys.exit(1)
    print("--force 지정됨. 그대로 진행한다.")

if not new.endswith("\n"):
    new += "\n"
TARGET.write_text("".join(lines[: start - 1]) + new + "".join(lines[end:]), encoding="utf-8")
print(f"치환 완료: {start}~{end} ({len(old)}행) -> {len(new.splitlines())}행")
