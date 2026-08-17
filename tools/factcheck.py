"""도메인 문서의 주장이 **사실인지** 판정하는 팩트체커.

왜 있나: DOMAIN.md 의 수치가 딥리서치 출처에서 왔고 반복해서 틀렸다.
[기록] limb darkening/brightening 역전 · 대회 회차 · §3.1 비교표 전체 · 193Å 온도 1.2→1.6 MK ·
       Box-Cox 개선을 부분집합→전체로 확대 서술. 자세한 건 POSTMORTEM.md.

**출처 추적기가 아니다.** 묻는 것은 「이 명제가 참인가」지 「우리 PDF 에 적혀 있는가」가 아니다.
전파 지연 `1AU/400km/s = 4.3일` 은 인용이 아니라 **산수**로 가릴 문제고,
회합 자전 27.2753일은 우리 PDF 에 없어도 **참**이다. 그래서 근거 유형을 셋으로 나눈다:

  계산 — 산술·단위환산·유도. luna 가 직접 계산해서 판정한다
  출처 — 특정 논문의 결과·표값. --corpus 의 PDF 로 대조한다
  상식 — 분야에서 확립된 상수·정의·기전. PDF 에 없어도 참일 수 있다

판정: 참 / 거짓 / 과장 / 미확정.
`과장` 이 실전에서 제일 자주 나온다 — 한 사례값을 일반값처럼, 부분집합 결과를 전체처럼 쓰는 것.
참/거짓 이분법으로는 이게 안 잡힌다.

설계 원칙 — 게으르게:
  · 주장 파서를 만들지 않는다. 절(section) 을 통째로 넘기고 주장 추출은 luna 가 한다.
  · luna 산문을 파싱하지 않는다. TSV 판정 블록 하나만 강제한다.
  · PDF 텍스트 추출도 luna 가 한다. 이 스크립트는 stdlib 만 쓴다.
  · 절 내용 해시를 캐시해서 **안 바뀐 절은 재실행 시 건너뛴다.** 토큰이 여기서 산다.

사용:
  python tools/factcheck.py DOMAIN.md --corpus D:/hackathon/SUN
  python tools/factcheck.py DOMAIN.md --corpus D:/hackathon/SUN --only "3.1" --force
  python tools/factcheck.py --selftest        # codex 호출 없이 로직만 점검

산출: factcheck/report.md (거짓·과장이 위로 정렬) · factcheck/cache.json (판정 + 해시)
**거짓 또는 과장**이 하나라도 있으면 exit 1 — 게이트로 쓸 수 있다.
`미확정` 은 게이트를 막지 않는다. 못 가린 것으로 빌드를 세우면 아무도 안 쓴다.
"""
import argparse
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys

OUT = pathlib.Path("factcheck")
CACHE = OUT / "cache.json"
REPORT = OUT / "report.md"
VERDICTS = ("참", "거짓", "과장", "미확정")
MODEL = ["-m", "gpt-5.6-luna", "-c", 'model_reasoning_effort="xhigh"']

PROMPT = """아래 문서 절의 주장이 **사실인지** 판정해라. 한국어.

핵심: 묻는 것은 **「이 명제가 참인가」**다. 「우리 PDF 에 적혀 있는가」가 아니다.
출처 확인은 참·거짓을 가리는 수단 중 하나일 뿐이고, 유일한 수단이 아니다.
PDF 에 없다는 이유만으로 `미확정` 을 찍지 마라 — 그건 판정 회피다.

## 쓸 수 있는 근거 (주장 성격에 맞는 것을 골라라)

**계산** — 산술·단위환산·유도로 결정되는 주장. **직접 계산해서 확인해라.**
  python 을 쓸 수 있다. 예: `1 AU / 400 km/s` 가 정말 4.3일인가, 각도·주기·비율 환산이 맞는가.
  이런 주장에 인용을 찾지 마라. 계산이 답이다. `근거` 열에 계산식을 적어라.

**출처** — 특정 논문의 실험 결과·표값·그 논문만의 정의. 아래 PDF 로 대조해라.
{corpus}
  `pdftotext` 또는 PyMuPDF(`import fitz`) 로 추출한다. 통째로 들지 말고 나눠 읽어라.

**상식** — 분야에서 확립된 물리 상수·정의·기전. 우리 PDF 에 없어도 참일 수 있다.
  (예: 태양 회합 자전 주기, 이온의 형성 온도, 1 AU 의 값, EUV 선의 광학적 두께)
  네 지식으로 판정하되 **확립도가 낮으면 `미확정`** 으로 두고 왜인지 적어라.

## 검증 대상 — `{target}` 의 「{section}」 절

~~~원문
{body}
~~~

## 판정

이 절에서 **참·거짓을 물을 수 있는 주장**(수치·정의·인과·귀속)을 뽑아 하나씩 판정해라.
문체·구성·전략 의견·`[추론]` 로 이미 표시된 추측은 대상이 아니다. 주장이 없으면 0건으로 끝내라.

  참     — 사실이다
  거짓   — 사실이 아니다. **`정정` 열에 올바른 값을 반드시 적는다**
  과장   — 방향은 맞는데 강도·범위·일반성이 과하다. 예: 한 사례값을 일반값처럼 서술,
           부분집합 결과를 전체 결과처럼 서술, 「항상」·「훨씬」 같은 미검증 강조.
           `정정` 에 정확한 진술을 적는다
  미확정 — 계산으로도 출처로도 상식으로도 못 가린다. **왜 못 가리는지를 `근거` 에 적는다**

귀속(누가 처음 했는가, 몇 판인가, 몇 쪽인가)은 **주장 자체가 귀속일 때만** 판정해라.
숫자가 맞는데 인용이 안 붙었다는 이유로 `거짓` 을 찍지 마라.

## 출력 형식 — 반드시 이대로

먼저 근거를 자유롭게 서술해라. 마지막에 아래 블록을 **정확히 한 번** 출력한다.
탭 구분 5열. 각 열에 탭·줄바꿈을 넣지 마라.

```verdicts
주장<TAB>판정<TAB>정정<TAB>근거유형<TAB>근거
```

`근거유형` = `계산` / `출처` / `상식` 중 하나.
`근거` = 계산식, 또는 `Son p.3 Table 1` 같은 위치, 또는 상식 판정의 이유. 참이어도 반드시 적는다.
`정정` 은 거짓·과장일 때만 채우고 나머지는 `-`.
"""


def sections(text):
    """마크다운을 (제목, 본문) 목록으로 자른다. 첫 제목 앞은 '(머리말)'."""
    out, title, buf = [], "(머리말)", []
    for line in text.splitlines():
        if re.match(r"^#{1,3} ", line):
            if buf:
                out.append((title, "\n".join(buf)))
            title, buf = line.lstrip("# ").strip(), []
        else:
            buf.append(line)
    if buf:
        out.append((title, "\n".join(buf)))
    return [(t, b) for t, b in out if b.strip()]


def _rows(block):
    """정확히 5열인 행만 받는다. 잘린 행을 패딩해서 받으면 근거 없는 판정이 통과한다."""
    out = []
    for line in block.strip().splitlines():
        cells = [c.strip() for c in line.split("\t")]
        if len(cells) == 5 and cells[1] in VERDICTS:
            out.append(cells)
    return out


def parse_verdicts(stdout):
    """luna 출력에서 판정 블록을 꺼내 행 목록으로. 못 찾으면 None.

    태그를 믿지 않는다 — 모델이 ```verdicts 대신 ```text 로 찍는 일이 실제로 있었다.
    태그 붙은 블록을 먼저 보고, 없으면 **유효 판정행이 든 마지막 펜스**를 쓴다.
    행 검증(2번째 칸이 판정어)이 이미 엄격해서 엉뚱한 블록이 걸릴 여지가 거의 없다.
    """
    tagged = re.findall(r"```verdicts\s*\n(.*?)```", stdout, re.S)
    if tagged:
        return _rows(tagged[-1])
    for block in reversed(re.findall(r"```[^\n]*\n(.*?)```", stdout, re.S)):
        rows = _rows(block)
        if rows:
            return rows
    return None


def run_luna(prompt, cwd):
    # 윈도우의 codex 는 npm 셔임(codex.CMD)이라 이름만으론 subprocess 가 못 찾는다.
    exe = shutil.which("codex")
    if not exe:
        sys.exit("codex 를 PATH 에서 못 찾았다. `npm i -g @openai/codex` 확인.")
    # 프롬프트를 argv 로 넘기지 마라. .CMD 셔임은 cmd.exe 를 거치고, cmd 는 `<`·`>`·`|` 를
    # 리다이렉션으로 먹고 줄바꿈 인자를 못 받는다 — 프롬프트가 조용히 잘려서 전달된다.
    # codex exec 는 PROMPT 인자가 없으면 stdin 을 읽는다. 그 경로로 보낸다.
    p = subprocess.run(
        [exe, "exec", *MODEL, "-s", "workspace-write", "--skip-git-repo-check", "-"],
        input=prompt, cwd=cwd, capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )
    out = p.stdout or ""
    # 종료 코드를 버리면 호출 실패가 「판정 0건」으로 둔갑해 조용히 PASS 된다.
    if p.returncode != 0:
        out += f"\n\n[factcheck] codex exit={p.returncode}\nstderr:\n{p.stderr or ''}"
        return out, False
    return out, True


def main(argv=None):
    # 한국어 문서를 다루는데 윈도우 콘솔 기본이 cp949 라 제목의 —·§ 에서 죽는다.
    # 문자열마다 걸러내지 말고 스트림 한 번만 고친다.
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="검증할 마크다운")
    ap.add_argument("--corpus", required=True, help="1차 출처 PDF 가 있는 폴더")
    ap.add_argument("--only", default="", help="제목에 이 문자열이 든 절만")
    ap.add_argument("--force", action="store_true", help="캐시 무시하고 전부 다시")
    ap.add_argument("--reparse", action="store_true",
                    help="luna 를 다시 부르지 않고 factcheck/raw 의 저장분만 재파싱")
    a = ap.parse_args(argv)

    target = pathlib.Path(a.target)
    pdfs = sorted(pathlib.Path(a.corpus).glob("*.pdf"))
    if not pdfs:
        sys.exit(f"PDF 를 못 찾았다: {a.corpus}")
    corpus = "\n".join(f"  {p}" for p in pdfs)

    OUT.mkdir(exist_ok=True)
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}

    # 캐시 무효화는 **판정에 영향을 주는 모든 것**을 포함해야 한다.
    # 경로 문자열만 해시하면 같은 경로의 PDF 를 갈아끼워도, 프롬프트·모델을 고쳐도
    # 옛 판정이 그대로 살아남는다.
    corpus_digest = hashlib.sha256()
    for f in pdfs:
        corpus_digest.update(hashlib.sha256(f.read_bytes()).digest())
    env = corpus_digest.hexdigest() + PROMPT + "".join(MODEL) + str(target.resolve())

    rows_all, checked, skipped, failed = [], 0, 0, 0
    for order, (title, body) in enumerate(sections(target.read_text(encoding="utf-8"))):
        if a.only and a.only not in title:
            continue
        key = f"{target.name}::{order}::{title}"   # 같은 제목이 둘이면 충돌한다
        digest = hashlib.sha256((f"{order}\x00{title}\x00{body}\x00{env}").encode("utf-8")).hexdigest()[:16]
        hit = cache.get(key)
        if hit and hit["hash"] == digest and not a.force:
            rows_all += [[title] + r for r in hit["rows"]]
            skipped += 1
            continue

        # luna 응답은 항상 남긴다. 파싱이 깨졌을 때 이게 없으면 원인을 못 찾는다.
        raw = OUT / "raw" / (re.sub(r"[^\w.-]", "_", title)[:60] + ".txt")
        if a.reparse:
            if not raw.exists():
                print(f"  ~ 저장분 없음 — 건너뜀: {title}", flush=True)
                failed += 1
                continue
            print(f"[재파싱] {title}", flush=True)
            out, ok = raw.read_text(encoding="utf-8"), True
        else:
            print(f"[검증] {title}", flush=True)
            out, ok = run_luna(
                PROMPT.format(corpus=corpus, target=target.name, section=title, body=body),
                cwd=target.resolve().parent,
            )
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_text(out, encoding="utf-8")

        rows = parse_verdicts(out) if ok else None
        # 판정 블록이 없거나 호출이 실패한 것은 「깨끗함」이 아니라 **검증 실패**다.
        # 여기서 캐시하거나 조용히 넘기면 다음 실행에서 PASS 로 굳는다.
        # `[]` (블록은 있으나 성립한 판정행 0개) 도 실패다 — 캐시하면 다음 실행에서 PASS 로 굳는다.
        if not rows:
            print(f"  ! 검증 실패 — {raw} 확인", flush=True)
            failed += 1
            continue
        cache[key] = {"hash": digest, "rows": rows}
        rows_all += [[title] + r for r in rows]
        checked += 1

    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")

    n = {v: sum(1 for r in rows_all if r[2] == v) for v in VERDICTS}
    lines = [
        f"# 팩트체크 — {target.name}", "",
        f"참고 PDF {len(pdfs)}편 · 절 {checked}개 검증 · {skipped}개 캐시 재사용 · **{failed}개 검증 실패**",
        "**" + " · ".join(f"{v} {n[v]}" for v in VERDICTS) + "**", "",
        "| 절 | 주장 | 판정 | 정정 | 근거유형 | 근거 |", "|---|---|---|---|---|---|",
    ]
    order = {v: i for i, v in enumerate(VERDICTS[::-1])}   # 거짓·과장이 위로
    for r in sorted(rows_all, key=lambda r: order.get(r[2], 9), reverse=True):
        lines.append("| " + " | ".join(c.replace("|", "\\|") for c in r) + " |")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    wrong = n["거짓"] + n["과장"]
    print(f"\n{REPORT} — 거짓 {n['거짓']} · 과장 {n['과장']} · 미확정 {n['미확정']} · 검증실패 {failed}")
    # 검증 실패도 exit 1 이다. 「못 돌았다」가 「깨끗하다」로 보이면 게이트가 무의미하다.
    return 1 if (wrong or failed) else 0


def selftest():
    md = "머리\n# A\n본문a\n## B\n본문b\n### C\n\n#### 제목아님\n"
    s = sections(md)
    assert [t for t, _ in s] == ["(머리말)", "A", "B", "C"], s
    assert "제목아님" in dict(s)["C"], s          # h4 는 절로 안 자른다
    assert sections("# 빈절\n\n") == []            # 본문 없는 절은 버린다
    assert sections("# 빈절\n\n## 뒷절\n내용\n") == [("뒷절", "내용")]

    ok = "잡담\n```verdicts\n193Å 1.2MK\t거짓\t1.6 MK\t상식\tFe XII 형성온도\n```\n"
    assert parse_verdicts(ok) == [["193Å 1.2MK", "거짓", "1.6 MK", "상식", "Fe XII 형성온도"]]
    # 블록이 둘이면 마지막 것만, 판정어 아닌 행(헤더 포함)은 버린다
    two = "```verdicts\nx\t참\t-\t-\t-\n```\n```verdicts\n주장<TAB>판정\ny\t참\t-\t계산\t1AU/400=4.3d\n```"
    assert parse_verdicts(two) == [["y", "참", "-", "계산", "1AU/400=4.3d"]]
    assert parse_verdicts("판정 블록 없음") is None
    # 잘린 행은 패딩해서 받지 않는다 — 근거 없는 판정이 통과하면 안 된다
    assert parse_verdicts("```verdicts\nz\t미확정\n```") == []
    # 그 `[]` 는 main 에서 `if not rows` 로 failed 처리된다 (성공 아님, 캐시 안 함)
    # 태그를 틀리게 찍어도 받는다 (luna 가 실제로 ```text 로 찍었다)
    assert parse_verdicts("```text\nw\t과장\t부분집합\t출처\tCollin p.13\n```") == \
        [["w", "과장", "부분집합", "출처", "Collin p.13"]]
    # 판정행 없는 코드블록은 블록으로 안 친다
    assert parse_verdicts("```python\nprint(1)\n```") is None
    # 출처 유무는 판정이 아니다 — PDF 에 없어도 참일 수 있어야 한다
    assert "출처없음" not in VERDICTS
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        sys.exit(main())
