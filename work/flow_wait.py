"""omniflow.json 이 생길 때까지 flow_runh 를 유지한다."""
import pathlib
import subprocess
import time

DONE = pathlib.Path("/home/jovyan/work/preds/omniflow.json")


def cmds():
    return subprocess.check_output(["ps", "-eo", "cmd"], text=True, errors="replace")


def main():
    while not DONE.exists():
        c = cmds()
        if "flow_runh.py" in c or "flow_h.py" in c:
            print("runner alive", flush=True)
            time.sleep(20)
            continue
        print("launch flow_runh", flush=True)
        subprocess.Popen(
            "PYTHONUNBUFFERED=1 nohup python work/flow_runh.py >> work/flow_runh.log 2>&1 &",
            shell=True)
        time.sleep(25)
    print("omniflow.json present", flush=True)


if __name__ == "__main__":
    main()
