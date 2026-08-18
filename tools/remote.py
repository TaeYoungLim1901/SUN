"""대회 팀 노트북(Kubeflow JupyterLab)을 로컬 CLI 에서 조작한다.

서버는 KAIST Keycloak SSO 뒤에 있다. 사람이 브라우저로 한 번 로그인하고
「Copy as cURL」한 것만 넘겨주면, 그 뒤로는 이 스크립트가 전부 한다.
사람이 하는 일의 전체 절차는 `CONNECT.md`.

    python tools/remote.py auth              # 클립보드의 cURL 에서 쿠키 추출 → 저장 → 확인
    python tools/remote.py check             # GPU·디스크·주피터 버전
    python tools/remote.py ls [경로]
    python tools/remote.py put 로컬 원격
    python tools/remote.py get 원격 로컬
    python tools/remote.py run -c "코드" | -f 스크립트.py
    python tools/remote.py selftest          # 네트워크 없이 도는 자체점검

학습은 `run` 으로 붙잡지 마라. 백그라운드로 던지고 로그만 회수한다:
    run -c "import subprocess; subprocess.Popen('nohup python t.py > t.log 2>&1 &', shell=True)"
    get t.log .
"""
import argparse
import base64
import json
import os
import pathlib
import re
import subprocess
import sys
import uuid

DEFAULT_BASE = ("https://kasiai-kubeflow.kaist.ac.kr"
                "/notebook/astronomy-ai-2026-t87/nb-astronomy-ai-2026-t87")
COOKIE_FILE = pathlib.Path.home() / ".sun_cookie"
KERNEL_FILE = pathlib.Path.home() / ".sun_kernel"
TEXT_SUFFIXES = {".py", ".md", ".txt", ".csv", ".json", ".yaml", ".yml", ".sh"}

HOWTO = """쿠키가 없다. 사람이 30초만 쓰면 된다:
  1. 크롬에서 팀 노트북 lab 주소를 열고 SSO 로그인
  2. F12 → Network → Ctrl+R → 맨 위 `lab` 요청 우클릭 → Copy → Copy as cURL (bash)
  3. python tools/remote.py auth      (클립보드에서 알아서 뽑아 쓴다)"""


for _s in (sys.stdout, sys.stderr):   # 윈도우 콘솔 기본이 cp949 라 한글에서 터진다
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def join(base, *parts):
    """슬래시 중복 없이 URL 을 잇는다."""
    return "/".join([base.rstrip("/")] + [str(p).strip("/") for p in parts if str(p).strip("/")])


def parse_cookie(blob):
    """cURL 통짜든 `Cookie:` 헤더든 `name=value` 하나든 전부 쿠키 문자열로 만든다.

    oauth2-proxy 는 4KB 넘는 세션 쿠키를 `_0`, `_1` 로 쪼갠다. 그래서 개별 쿠키가 아니라
    Cookie 헤더 전체를 받는다.
    """
    blob = blob.strip()
    m = re.search(r"-b\s+'([^']*)'|-b\s+\"([^\"]*)\"|"
                  r"[-]H\s+['\"]cookie:\s*([^'\"]*)['\"]", blob, re.I)
    if m:
        blob = next(g for g in m.groups() if g)
    blob = re.sub(r"^\s*cookie:\s*", "", blob, flags=re.I).strip()
    pairs = [p.strip() for p in blob.split(";") if "=" in p]
    if not pairs:
        sys.exit("쿠키를 못 찾았다. Copy as cURL (bash) 로 다시 복사해라.")
    return "; ".join(pairs)


def load_cookie():
    raw = os.environ.get("SUN_COOKIE") or (
        COOKIE_FILE.read_text(encoding="utf-8") if COOKIE_FILE.exists() else "")
    if not raw.strip():
        sys.exit(HOWTO)
    return parse_cookie(raw)


def clipboard():
    p = subprocess.run(["powershell", "-NoProfile", "-Command", "Get-Clipboard -Raw"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.stdout or ""


class Remote:
    def __init__(self):
        import requests
        self.base = os.environ.get("SUN_BASE", DEFAULT_BASE).rstrip("/")
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "Mozilla/5.0"
        # 헤더로 박지 말고 쿠키 항아리에 넣는다. 그래야 서버가 갱신한 _xsrf 를 requests 가 따라간다.
        host = self.base.split("//", 1)[1].split("/", 1)[0]
        for pair in load_cookie().split("; "):
            k, _, v = pair.partition("=")
            self.s.cookies.set(k, v, domain=host)
        # Jupyter 는 비-GET 에 XSRF 를 요구한다. /lab 을 한 번 쳐서 _xsrf 를 최신으로 만든다.
        self.s.get(join(self.base, "lab"), timeout=30)

    def cookie_header(self):
        return "; ".join("%s=%s" % (c.name, c.value) for c in self.s.cookies)

    def req(self, method, path, **kw):
        xsrf = self.s.cookies.get("_xsrf")
        h = {"X-XSRFToken": xsrf} if (xsrf and method != "GET") else {}
        r = self.s.request(method, join(self.base, path), headers=h, timeout=120, **kw)
        ct = r.headers.get("content-type", "")
        if "text/html" in ct or r.url.startswith("https://kasiai-auth"):
            sys.exit("쿠키 만료. 브라우저에서 다시 로그인하고 `remote.py auth` 를 다시 돌려라. (exit 3)")
        if r.status_code >= 400:
            sys.exit("HTTP %d %s\n%s" % (r.status_code, path, r.text[:400]))
        return r

    # ---- contents ----
    def ls(self, path=""):
        j = self.req("GET", join("api/contents", path)).json()
        for c in sorted(j.get("content", []), key=lambda c: (c["type"] != "directory", c["name"])):
            print("%-40s %-10s %s" % (c["name"], c["type"], c.get("size") or ""))

    def put(self, local, remote):
        p = pathlib.Path(local)
        body = payload(p)
        self.req("PUT", join("api/contents", remote), json=body)
        print("put %s -> %s (%d bytes)" % (local, remote, p.stat().st_size))

    def get(self, remote, local):
        j = self.req("GET", join("api/contents", remote)).json()
        out = pathlib.Path(local)
        if out.is_dir():
            out = out / pathlib.PurePosixPath(remote).name
        if j["format"] == "base64":
            out.write_bytes(base64.b64decode(j["content"]))
        elif j["type"] == "notebook":
            out.write_text(json.dumps(j["content"], ensure_ascii=False, indent=1), encoding="utf-8")
        else:
            out.write_text(j["content"], encoding="utf-8")
        print("get %s -> %s (%d bytes)" % (remote, out, out.stat().st_size))

    # ---- kernel ----
    def kernel(self, new=False):
        kid = KERNEL_FILE.read_text().strip() if (KERNEL_FILE.exists() and not new) else ""
        if kid:
            r = self.s.get(join(self.base, "api/kernels", kid), timeout=30)
            if r.status_code == 200:
                return kid
        kid = self.req("POST", "api/kernels", json={"name": "python3"}).json()["id"]
        KERNEL_FILE.write_text(kid)
        return kid

    def run(self, code, new=False, timeout=900, capture=False):
        import websocket
        kid = self.kernel(new)
        url = join(self.base.replace("https://", "wss://", 1), "api/kernels", kid, "channels")
        ws = websocket.create_connection(
            url, header=["Cookie: " + self.cookie_header()],
            origin=self.base.split("/notebook")[0], timeout=timeout)
        msg_id = uuid.uuid4().hex
        ws.send(json.dumps({
            "header": {"msg_id": msg_id, "username": "sun", "session": uuid.uuid4().hex,
                       "msg_type": "execute_request", "version": "5.3"},
            "parent_header": {}, "metadata": {},
            "content": {"code": code, "silent": False, "store_history": True,
                        "user_expressions": {}, "allow_stdin": False, "stop_on_error": True},
            "channel": "shell"}))
        failed = False
        buf = []
        try:
            while True:
                m = json.loads(ws.recv())
                if m.get("parent_header", {}).get("msg_id") != msg_id:
                    continue
                t, c = m["header"]["msg_type"], m.get("content", {})
                if t == "stream":
                    if capture:
                        buf.append(c.get("text", ""))
                    else:
                        (sys.stderr if c.get("name") == "stderr"
                         else sys.stdout).write(c.get("text", ""))
                elif t in ("execute_result", "display_data"):
                    print(c.get("data", {}).get("text/plain", ""))
                elif t == "error":
                    failed = True
                    print("\n".join(c.get("traceback", [])), file=sys.stderr)
                elif t == "status" and c.get("execution_state") == "idle":
                    break
        except websocket.WebSocketTimeoutException:
            sys.exit("타임아웃 %ds. 학습은 nohup 백그라운드로 던지고 로그를 get 해라. (exit 4)" % timeout)
        finally:
            ws.close()
            sys.stdout.flush()
        if capture:
            return "".join(buf), failed
        if failed:
            sys.exit(1)
        return None, False

    def check(self):
        j = self.req("GET", "api/status").json()
        print("주피터 %s · 커널 %s개 · 마지막 활동 %s"
              % (j.get("version", "?"), j.get("kernels"), j.get("last_activity")))
        self.run(CHECK_CODE)


CHECK_CODE = r"""
import shutil, subprocess, sys
print("python", sys.version.split()[0])
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
except Exception as e:
    print("torch 없음:", e)
try:
    print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
except Exception as e:
    print("nvidia-smi 없음:", e)
u = shutil.disk_usage("/home/jovyan")
print("디스크 free %.1f GB / total %.1f GB" % (u.free/2**30, u.total/2**30))
"""


def payload(p):
    """확장자로 Jupyter contents API 페이로드를 만든다."""
    if p.suffix == ".ipynb":
        return {"type": "notebook", "content": json.loads(p.read_text(encoding="utf-8"))}
    if p.suffix in TEXT_SUFFIXES:
        return {"type": "file", "format": "text", "content": p.read_text(encoding="utf-8")}
    return {"type": "file", "format": "base64",
            "content": base64.b64encode(p.read_bytes()).decode()}


def selftest():
    assert join("http://a/b/", "/c/", "d") == "http://a/b/c/d"
    assert join("http://a", "") == "http://a"
    ck = "oauth2_proxy_kubeflow_0=aaa; oauth2_proxy_kubeflow_1=bbb"
    assert parse_cookie(ck) == ck
    assert parse_cookie("Cookie: " + ck) == ck
    assert parse_cookie("curl 'https://x/lab' -H 'cookie: %s' -H 'accept: */*'" % ck) == ck
    assert parse_cookie("curl 'https://x/lab' -b '%s' --compressed" % ck) == ck
    import tempfile
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "a.py").write_text("print(1)", encoding="utf-8")
    (d / "a.ipynb").write_text('{"cells": []}', encoding="utf-8")
    (d / "a.png").write_bytes(b"\x89PNG")
    assert payload(d / "a.py")["format"] == "text"
    assert payload(d / "a.ipynb")["type"] == "notebook"
    assert payload(d / "a.png")["format"] == "base64"
    print("selftest OK")


def cmd_auth():
    blob = clipboard()
    if "curl" not in blob.lower() and "=" not in blob:
        sys.exit(HOWTO)
    if "oauth2_proxy" not in blob:
        # SSO 리다이렉트 전 요청을 복사하면 여기 걸린다. 그대로 저장하면 401 만 본다.
        sys.exit("이 cURL 에 oauth2_proxy 쿠키가 없다 — 로그인 전 요청을 복사한 것이다. "
                 "Network 탭 Fetch/XHR 필터에서 api/... 요청을 복사해라 (CONNECT.md 3).")
    COOKIE_FILE.write_text(parse_cookie(blob), encoding="utf-8")
    print("쿠키 저장 →", COOKIE_FILE)
    Remote().check()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("auth")
    sub.add_parser("check")
    sub.add_parser("selftest")
    p = sub.add_parser("ls"); p.add_argument("path", nargs="?", default="")
    p = sub.add_parser("put"); p.add_argument("local"); p.add_argument("remote")
    p = sub.add_parser("get"); p.add_argument("remote"); p.add_argument("local", nargs="?", default=".")
    p = sub.add_parser("run")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("-c", "--code"); g.add_argument("-f", "--file")
    p.add_argument("--new", action="store_true"); p.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()

    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "auth":
        return cmd_auth()
    r = Remote()
    if a.cmd == "check":
        r.check()
    elif a.cmd == "ls":
        r.ls(a.path)
    elif a.cmd == "put":
        r.put(a.local, a.remote)
    elif a.cmd == "get":
        r.get(a.remote, a.local)
    elif a.cmd == "run":
        code = a.code if a.code else pathlib.Path(a.file).read_text(encoding="utf-8")
        r.run(code, new=a.new, timeout=a.timeout)


if __name__ == "__main__":
    main()
