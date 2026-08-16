"""audio/*.mp3 를 faster-whisper 로 재전사. 결과는 transcript/NN_<id>.txt.

사용: python tools/transcribe.py [model] [video_id ...]
  model 기본값 large-v3-turbo (CPU 8스레드 기준 large-v3 대비 수 배 빠름)
"""
import glob
import os
import pathlib
import sys
import time

from faster_whisper import WhisperModel

ROOT = pathlib.Path("D:/hackathon/SUN")
AUDIO = ROOT / "audio"
OUT = ROOT / "transcript"

# 영상 번호 ↔ 유튜브 id (작업지시서 3절 순번)
ID2NUM = {
    "wXxFJHA2tFE": 1,
    "oEf5HVk_A8g": 2,
    "fVnffH5pCRY": 3,
    "RGqZiu59Zzo": 4,
    "3--oD514lOU": 5,
    "ThQFDIgZmT0": 6,
    "fEmf1SFniHE": 7,
    "j0OryJ27XCg": 8,
    "2gSJuzmBuBI": 9,
}

# 도메인 용어를 넣어야 흑점/극대기/전리권 같은 전문어 인식률이 오른다
PROMPT = (
    "태양, 흑점, 코로나, 극대기, 극소기, 태양풍, 오로라, 전리권, 우주 날씨, "
    "코로나질량방출, 플레어, 광구, 채층, 자기장, 한국천문연구원, 전파망원경, "
    "천체망원경, 분광, 파장, 옹스트롬, 인공위성, 누리호"
)

model_name = sys.argv[1] if len(sys.argv) > 1 else "large-v3-turbo"
only = set(sys.argv[2:])

OUT.mkdir(exist_ok=True)
model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=os.cpu_count())

for f in sorted(glob.glob(str(AUDIO / "*.mp3"))):
    vid = pathlib.Path(f).stem
    if only and vid not in only:
        continue
    num = ID2NUM.get(vid, 0)
    dest = OUT / f"{num:02d}_{vid}.txt"
    if dest.exists():
        print("skip (exists):", dest.name, flush=True)
        continue
    t0 = time.time()
    segs, info = model.transcribe(
        f,
        language="ko",
        vad_filter=True,
        beam_size=5,
        condition_on_previous_text=False,  # ASR 환각 반복 방지
        initial_prompt=PROMPT,
    )
    tmp = dest.with_suffix(".part")
    with open(tmp, "w", encoding="utf-8") as fp:
        for s in segs:
            fp.write(f"[{int(s.start)//60}:{int(s.start)%60:02d}] {s.text.strip()}\n")
            fp.flush()
    tmp.replace(dest)
    print(f"done: {dest.name}  audio={info.duration:.0f}s  wall={time.time()-t0:.0f}s", flush=True)
