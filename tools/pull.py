"""서버 파일을 Jupyter /files/ 로 스트리밍 다운로드. remote.py 의 인증 세션을 재사용."""
import sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))
sys.path.insert(0, r"D:\hackathon\sun\tools")
import remote

c = remote.Remote()

def pull(rpath, lpath):
    out = pathlib.Path(lpath); out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists(): return out.stat().st_size, 0.0, "skip"
    t0 = time.time(); n = 0
    with c.s.get(remote.join(c.base, "files", rpath), stream=True, timeout=1800) as r:
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk); n += len(chunk)
    return n, time.time() - t0, "ok"

if __name__ == "__main__":
    for rp, lp in (a.split("=", 1) for a in sys.argv[1:]):
        n, dt, how = pull(rp, lp)
        print(f"{how} {rp} -> {lp}  {n/1e6:.1f}MB  {dt:.1f}s  {n/1e6/max(dt,1e-9):.1f}MB/s", flush=True)
