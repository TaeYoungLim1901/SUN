"""SUN 폴더 하나에서 개인 레포와 팀 레포 양쪽에 올린다. 커밋 메시지는 codex luna xhigh 가 쓴다.

두 레포는 포함 관계가 아니다. 교집합이 있는 두 집합이다 —
팀 레포에는 **내가 모르는 팀원 커밋**이 들어오고, 개인 레포에는 팀이 볼 필요 없는 내 문서가 있다.
파일 단위로 갈라야 하는데 브랜치는 스냅샷이라, 브랜치 둘이 유일하게 안 새는 구조다.

    main   내 작업분 전부 (개인 문서 + 내가 고친 팀 문서). 개인 private 레포에만 간다.
    team   팀과 주고받는 것. team/main 을 따라가고, 팀 레포 + 개인 레포 양쪽에 간다.

    python tools/push.py                          # main 커밋 → 개인 레포
    python tools/push.py --share DOMAIN.md RULES.md   # 그 파일들만 팀에 공개
    python tools/push.py --sync                   # 팀원 작업 받아서 main 에 합치기
    python tools/push.py --dry-run                # 메시지만 보고 멈춘다
    python tools/push.py -m "직접 쓴 메시지"
    python tools/push.py --selftest

`--share` 는 이름 댄 파일만 `main` 에서 `team` 으로 옮긴다. `add -A` 를 team 브랜치에서
돌리지 않는 것이 이 스크립트의 존재 이유다 — 그 한 번이 개인 파일을 팀 레포에 흘린다.
"""
import argparse
import re
import shutil
import subprocess
import sys

MODEL = ["-m", "gpt-5.6-luna", "-c", 'model_reasoning_effort="xhigh"']
PERSONAL, TEAM = "origin", "team"       # origin = 내 private 레포
MAIN, SHARED = "main", "team"           # 브랜치 이름
DIFF_CAP = 12000

PROMPT = """아래는 어느 깃 저장소의 **스테이징된 변경**이다. 커밋 메시지를 써라.

## 변경 요약
{stat}

## diff
~~~
{diff}
~~~

## 규칙

- Conventional Commits. `feat:` `fix:` `docs:` `refactor:` `chore:` `test:` 중 하나로 시작한다.
- 제목은 영어 한 줄, 72자 이하, 마침표 없음, 명령형.
- 변경이 여러 갈래면 본문에 `-` 목록으로 적는다. 한 갈래면 제목만 쓴다.
- **무엇을 왜** 바꿨는지 쓴다. 파일 목록을 나열하지 마라 (그건 diff 가 이미 말한다).
- 추측하지 마라. diff 에 없는 동기를 지어내지 마라.

## 출력 형식 — 반드시 이대로

설명을 먼저 자유롭게 써도 된다. 마지막에 아래 블록을 **정확히 한 번** 출력한다.

```commit
<제목 줄>

<본문 — 없으면 통째로 생략>
```
"""


def git(*args, check=True):
    p = subprocess.run(["git", *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if check and p.returncode != 0:
        sys.exit(f"git {' '.join(args)} 실패 (exit {p.returncode})\n{p.stderr.strip()}")
    return p.stdout.strip(), p.returncode


def run_luna(prompt):
    # 윈도우의 codex 는 npm 셔임이라 이름만으론 subprocess 가 못 찾는다.
    exe = shutil.which("codex")
    if not exe:
        sys.exit("codex 를 PATH 에서 못 찾았다. `npm i -g @openai/codex` 확인.")
    # 프롬프트를 argv 로 넘기지 마라 — .CMD 셔임이 cmd.exe 를 거치면서 줄바꿈과
    # `<`·`>`·`|` 를 먹는다. codex exec 는 PROMPT 인자가 없으면 stdin 을 읽는다.
    p = subprocess.run(
        [exe, "exec", *MODEL, "-s", "read-only", "--skip-git-repo-check", "-"],
        input=prompt, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if p.returncode != 0:
        sys.exit(f"codex 실패 (exit {p.returncode})\n{(p.stderr or '').strip()[:800]}")
    return p.stdout or ""


def parse_commit(out):
    """마지막 ```commit 블록만 받는다. 못 찾거나 비면 None — 조용히 넘기지 않는다."""
    blocks = re.findall(r"```(?:commit|text)?\s*\n(.*?)```", out, re.S)
    for block in reversed(blocks):
        msg = block.strip()
        # 제목이 Conventional Commits 가 아니면 코드블록을 잘못 집은 것이다.
        if msg and re.match(r"^(feat|fix|docs|refactor|chore|test|perf|style|build|ci)"
                            r"(\(.+?\))?!?: .+", msg):
            return msg
    return None


def remote_owner(url):
    """git@github.com:owner/repo.git · https://github.com/owner/repo(.git) → owner"""
    m = re.search(r"github\.com[:/]([^/]+)/", url)
    return m.group(1) if m else None


def message_for(stat, given):
    if given:
        return given
    diff, _ = git("diff", "--cached")
    if len(diff) > DIFF_CAP:
        diff = diff[:DIFF_CAP] + f"\n\n[... {len(diff) - DIFF_CAP}자 잘림. 요약만 보고 써라]"
    print(f"[luna] 커밋 메시지 생성 중 — {len(stat.splitlines())}개 파일", flush=True)
    msg = parse_commit(run_luna(PROMPT.format(stat=stat, diff=diff)))
    # 블록이 없으면 실패다. 빈 메시지로 커밋하면 이력이 못 쓰게 된다.
    if not msg:
        sys.exit("luna 가 ```commit 블록을 안 냈다. -m 으로 직접 써라.")
    return msg


def show(stat, msg):
    print(f"\n{stat}\n\n--- 커밋 메시지 ---\n{msg}\n-------------------")


def push_mine(a):
    """main 을 통째로 개인 레포에 올린다. 팀 레포는 건드리지 않는다."""
    branch, _ = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch == SHARED:
        sys.exit(f"`{SHARED}` 브랜치에서는 `add -A` 를 하지 않는다 — 개인 파일이 팀 레포로 샌다.\n"
                 f"    git checkout {MAIN}   후 다시. 팀 공개는 --share 로만.")
    git("add", "-A")
    stat, _ = git("diff", "--cached", "--stat")
    if not stat:
        print("올릴 게 없다 (스테이징된 변경 0).")
        return 0
    msg = message_for(stat, a.message)
    show(stat, msg)
    if a.dry_run:
        print("--dry-run — 커밋하지 않았다.")
        return 0
    git("commit", "-m", msg)
    url, _ = git("remote", "get-url", PERSONAL)
    git("push", "-u", PERSONAL, branch)
    print(f"올렸다: {branch} → {url}")
    return 0


def push_shared(a):
    """이름 댄 파일만 main → team 브랜치로 옮겨 팀 레포에 올린다."""
    dirty, _ = git("status", "--porcelain")
    if dirty:
        sys.exit("작업트리가 지저분하다. 먼저 `python tools/push.py` 로 main 에 커밋해라.\n" + dirty)

    back, _ = git("rev-parse", "--abbrev-ref", "HEAD")
    git("fetch", "-q", TEAM)
    git("checkout", "-q", SHARED)
    # 팀원이 먼저 올린 게 있으면 따라잡는다. 충돌나면 사람이 푼다.
    _, rc = git("merge", "--ff-only", f"{TEAM}/{MAIN}", check=False)
    if rc != 0:
        git("checkout", "-q", back)
        sys.exit(f"`{SHARED}` 가 {TEAM}/{MAIN} 보다 앞서 있거나 갈라졌다. 손으로 정리해라.")

    try:
        git("checkout", MAIN, "--", *a.share)
        stat, _ = git("diff", "--cached", "--stat")
        if not stat:
            print("팀에 새로 공개할 내용이 없다 (그 파일들은 이미 같다).")
            return 0
        msg = message_for(stat, a.message)
        show(stat, msg)
        if a.dry_run:
            git("reset", "-q", "--hard", "HEAD")
            print("--dry-run — 커밋하지 않았다.")
            return 0
        git("commit", "-m", msg)
        url, _ = git("remote", "get-url", TEAM)
        print(f"[push] {TEAM} = {remote_owner(url)} — {url}", flush=True)
        git("push", TEAM, f"{SHARED}:{MAIN}")
        git("push", "-u", PERSONAL, SHARED)     # 공개분도 개인 레포에 기록으로 남긴다
        print(f"팀에 공개했다: {' '.join(a.share)} → {url}")
    finally:
        git("checkout", "-q", back)
    return 0


def sync(a):
    """팀원이 올린 것을 team 브랜치로 받아 main 에 합친다. push 는 하지 않는다."""
    dirty, _ = git("status", "--porcelain")
    if dirty:
        sys.exit("작업트리가 지저분하다. 먼저 커밋해라.\n" + dirty)
    git("fetch", "-q", TEAM)
    behind, _ = git("rev-list", "--count", f"{SHARED}..{TEAM}/{MAIN}")
    if behind == "0":
        print("팀 레포에 새 커밋 없다.")
        return 0
    back, _ = git("rev-parse", "--abbrev-ref", "HEAD")
    git("checkout", "-q", SHARED)
    _, rc = git("merge", "--ff-only", f"{TEAM}/{MAIN}", check=False)
    git("checkout", "-q", back)
    if rc != 0:
        sys.exit(f"`{SHARED}` 가 갈라졌다. 손으로 정리해라.")
    print(f"팀 커밋 {behind}개 받았다. main 에 합친다.")
    # 충돌은 사람이 푼다. 여기서 자동으로 한쪽을 고르면 팀원 작업이 조용히 사라진다.
    _, rc = git("merge", "--no-edit", SHARED, check=False)
    if rc != 0:
        sys.exit("충돌났다. 풀고 `git commit` 한 뒤 `python tools/push.py`.")
    print("합쳤다. 확인하고 `python tools/push.py` 로 개인 레포에 올려라.")
    return 0


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description="개인 레포에 올리고, 지정한 것만 팀 레포에 공개한다")
    ap.add_argument("--share", nargs="+", metavar="PATH", help="팀에 공개할 파일 (이것만 간다)")
    ap.add_argument("--sync", action="store_true", help="팀원 작업을 받아 main 에 합친다")
    ap.add_argument("-m", "--message", help="커밋 메시지를 직접 준다 (luna 호출 안 함)")
    ap.add_argument("--dry-run", action="store_true", help="메시지만 만들고 멈춘다")
    a = ap.parse_args(argv)

    if git("rev-parse", "--is-inside-work-tree", check=False)[1] != 0:
        sys.exit("여긴 깃 저장소가 아니다.")
    if a.sync:
        return sync(a)
    return push_shared(a) if a.share else push_mine(a)


def selftest():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    assert parse_commit("설명\n```commit\nfix: guard empty verdicts\n```\n") == \
        "fix: guard empty verdicts"
    # 블록이 둘이면 마지막 것 (앞의 것은 luna 가 보여준 예시일 때가 많다)
    assert parse_commit("```commit\nfeat: a\n```\n```commit\nfeat: b\n\n- 본문\n```") == \
        "feat: b\n\n- 본문"
    # Conventional Commits 가 아닌 코드블록은 안 집는다 (luna 가 낸 예시 diff 등)
    assert parse_commit("```python\nprint(1)\n```") is None
    assert parse_commit("```commit\n\n```") is None
    assert parse_commit("블록 없음") is None
    assert remote_owner("https://github.com/andy6609/sungnam-to-kaist.git") == "andy6609"
    assert remote_owner("git@github.com:TaeYoungLim1901/SUN.git") == "TaeYoungLim1901"
    assert remote_owner("https://gitlab.com/x/y.git") is None
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        sys.exit(main())
