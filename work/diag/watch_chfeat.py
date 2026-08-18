"""Start chfeat.py the instant align.py leaves /proc. No HistGBDT until then."""
import os
import subprocess
import time


def has_align():
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            cmd = open(f"/proc/{p}/cmdline", "rb").read().replace(b"\x00", b" ").decode("replace")
        except OSError:
            continue
        if "align.py" in cmd and "python" in cmd:
            return p
    return None


def has_chfeat():
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            cmd = open(f"/proc/{p}/cmdline", "rb").read().replace(b"\x00", b" ").decode("replace")
        except OSError:
            continue
        if "chfeat.py" in cmd and "python" in cmd:
            return p
    return None


def main():
    while True:
        pid = has_align()
        if pid is None:
            break
        print(f"align still pid={pid}", flush=True)
        time.sleep(20)
    if has_chfeat():
        print("chfeat already running", flush=True)
        return
    print("align gone — starting chfeat.py", flush=True)
    subprocess.Popen(
        "cd /home/jovyan/work && nohup python chfeat.py > chfeat.log 2>&1 &",
        shell=True,
    )
    time.sleep(2)
    print("chfeat pid", has_chfeat(), flush=True)


if __name__ == "__main__":
    main()
