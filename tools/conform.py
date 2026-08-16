"""이미 한 번 고친 사실 오류가 되살아나는지 감시한다. 사실에 대한 회귀 테스트.

`factcheck.py` 와 역할이 다르다:
  factcheck — luna 를 불러 **새 주장**의 참·거짓을 가린다. 느리고 비싸다.
  conform   — **이미 판정 끝난 오류**가 다시 들어왔는지 grep 한다. 즉시·무료.

각 항목은 이 저장소에서 실제로 났던 오류다 (`POSTMORTEM.md` 의 ID 와 대응).
문서를 사람이 고치든 luna 가 고치든, 되살아나면 여기서 걸린다.

예외 — 아래 표시가 있는 줄은 건너뛴다. 이미 쓰던 관례를 그대로 쓴다:
  `~~취소선~~`  옛 값을 기록으로 남긴 것
  `[미검증]`     출처를 못 댄다고 표시한 것
  `[기록]`       실수 대장에 사례로 적은 것

사용:
  python tools/conform.py             # 저장소 전체
  python tools/conform.py --selftest
위반이 있으면 exit 1.
"""
import pathlib
import re
import sys

SKIP_DIRS = {".git", "factcheck", "__pycache__", "preds", "transcript", "audio"}
# conform.py 는 틀린 문자열을 정규식으로 들고 있으니 자기 자신은 볼 수 없다.
SKIP_FILES = {"POSTMORTEM.md", "notesC.md", "notesD.md", "conform.py"}

# 줄에 이게 있으면 건너뛴다. 옛 값을 **오류로 지목하는** 문장은 위반이 아니다.
# 앞의 셋은 문서가 이미 쓰던 표시, 나머지는 정정문에서 실제로 쓰는 말이다.
EXEMPT = ("~~", "[미검증]", "[기록]",
          "거짓", "불가", "금지", "아니다", "어긋", "정정", "틀렸", "하지 마라", "stale")

# (ID, 무엇이 틀렸나, 되살아남을 잡는 정규식, 올바른 값)
RULES = [
    ("M03", "출력 horizon 개수",
     r"(?:\b20\s*개?\s*horizon|horizon\s*\(20\)|N_HORIZON\s*=\s*20)",
     "12개 (6h × 12 = 72h, RULES.md §3)"),
    ("M03", "출력 창 길이",
     r"(?:5일\s*뒤까지\s*출력|미래\s*5일|출력\s*창\s*5일)",
     "미래 72h = 3일"),
    ("M05", "193Å 형성 온도",
     r"193\s*Å?[^\n]{0,30}?1\.2\s*MK",
     "약 1.6 MK (Collin p.4)"),
    ("M01", "AIA EUV 에 limb darkening 보정",
     r"limb\s*darkening[^\n]{0,40}(?:AIA|193|211|보정하)",
     "EUV 는 limb brightening. darkening 은 광구/HMI 현상"),
    ("M10", "27일 지속성을 제출 가능한 기저로 취급",
     r"persistence_27d[^\n]{0,40}(?:제출|기저\s*2장)",
     "제출 불가 — 입력 창이 120h 뿐 (DOMAIN.md §9.1)"),
    ("M06", "Box-Cox 를 첫날 무료 이득으로 서술",
     r"(?:Box-Cox|분포\s*변환)[^\n]{0,80}첫날\s*붙여라",
     "timeline RMSE 는 68.1→75.1 로 악화 (Collin p.13)"),
    ("M07", "Son 논문과 대회 설정이 동일하다는 서술",
     r"(?:논문과|Son)[^\n]{0,20}(?:거의\s*동일|사실상\s*동일)",
     "영상 cadence 가 다르다 (Son 12h vs 대회 6h)"),
    ("M09", "Moldwin 판본",
     r"Moldwin[^\n]{0,60}2판|Introduction to Space Weather[^\n]{0,40}2판",
     "보유 PDF 는 2008 초판"),
    ("M11", "Son et al. 2023 DOI",
     r"1538-4355",
     "1538-4365 (ApJS)"),
    ("M02", "1회차 주제·회차",
     r"2회차[^\n]{0,20}CME",
     "제1회 = 2024.07~08, 태양 이벤트 탐지"),
    ("M08", "출처 불명이던 §3.1 비교표 값",
     r"(?:87\.0|116\.3|130\.9)\s*(?:km/s)?",
     "27일 지속성 89.1 · WSA 86.5 · ESWF 91.8 (Collin p.17 Table 3)"),
    # 14.713 자체는 실재하는 값이다. 「적도」라는 라벨 없이 쓰였을 때만 혼동이다.
    ("M21", "적도 자전율을 적도라 안 밝히고 씀",
     r"14\.713(?![^\n]*적도)",
     "14.713 은 적도율(Snodgrass & Ulrich). 캐링턴 항성은 14.184°/일"),
    ("M04", "Surya/LoRA 를 계획 B 로 유지",
     r"Surya[^\n]{0,40}계획\s*B|계획\s*B[^\n]{0,20}Surya",
     "pretrained weight 사용이라 제외. 아키텍처 차용만 허용 (RULES.md §1 5항)"),
]


def scan(root):
    hits = []
    for path in sorted(pathlib.Path(root).rglob("*")):
        if path.suffix not in (".md", ".py") or not path.is_file():
            continue
        if path.name in SKIP_FILES or SKIP_DIRS & set(path.parts):
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if any(mark in line for mark in EXEMPT):
                continue
            for mid, what, pat, right in RULES:
                if re.search(pat, line):
                    hits.append((str(path), i, mid, what, right, line.strip()[:90]))
    return hits


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    hits = scan(".")
    for path, ln, mid, what, right, text in hits:
        print(f"{path}:{ln}  [{mid}] {what}")
        print(f"    발견: {text}")
        print(f"    정답: {right}")
    print(f"\n규칙 {len(RULES)}개 · 위반 {len(hits)}개")
    return 1 if hits else 0


def selftest():
    import tempfile

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    d = pathlib.Path(tempfile.mkdtemp())
    (d / "bad.md").write_text(
        "출력은 20개 horizon 이다\n"
        "N_HORIZON = 20\n"
        "193Å 은 1.2 MK 이다\n",
        encoding="utf-8")
    (d / "ok.md").write_text(
        "출력은 12개 horizon 이다\n"
        "~~20개 horizon~~ 이었다\n"           # 취소선은 면제
        "193Å 1.2 MK [미검증]\n"              # 미검증 표시는 면제
        "[기록] N_HORIZON = 20 이었다\n"       # 실수 대장 인용은 면제
        "`persistence_27d` 기저 제출은 불가하다\n"   # 오류로 지목하는 문장은 면제
        "193Å 1.2 MK 는 거짓이다\n",
        encoding="utf-8")
    (d / "POSTMORTEM.md").write_text("N_HORIZON = 20\n", encoding="utf-8")

    hits = scan(d)
    got = sorted((pathlib.Path(p).name, ln) for p, ln, *_ in hits)
    assert got == [("bad.md", 1), ("bad.md", 2), ("bad.md", 3)], got
    for mid, _, pat, right in RULES:
        assert re.fullmatch(r"M\d{2}", mid), f"POSTMORTEM ID 형식 아님: {mid}"
        assert right.strip(), f"{mid}: 정답을 안 적었다"
        re.compile(pat)                        # 정규식이 전부 컴파일되는지
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        sys.exit(main())
