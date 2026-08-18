"""sklearn_blob 이 생기면 merge → stage → bugcheck 까지 한 번에.

pack.py --sklearn-only 와 파일을 공유하지 않는다. 읽기만 하다가 blob 생성 후에 실행.
"""
import pathlib
import subprocess
import time

BLOB = pathlib.Path("/home/jovyan/work/pack_out/sklearn_blob.pt")
CNN = pathlib.Path("/home/jovyan/work/pack_out/cnn_blob.pt")
LOG = pathlib.Path("/home/jovyan/work/watch_pack.log")


def say(msg):
    line = msg + "\n"
    print(line, end="", flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def run(cmd):
    say("RUN " + " ".join(cmd))
    p = subprocess.run(cmd, cwd="/home/jovyan")
    say("EXIT %s %d" % (cmd[0], p.returncode))
    return p.returncode


def main():
    say("watch start")
    while not BLOB.exists():
        time.sleep(20)
    say("blob %d" % BLOB.stat().st_size)
    if not CNN.exists():
        say("cnn_blob missing — cnn-only")
        if run(["python", "work/pack.py", "--cnn-only"]) != 0:
            return
    if run(["python", "work/pack.py", "--merge"]) != 0:
        return
    if run(["python", "work/stage.py"]) != 0:
        return
    say("READY")


if __name__ == "__main__":
    main()
