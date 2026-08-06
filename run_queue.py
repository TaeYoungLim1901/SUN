"""audio/ 안에서 아직 전사되지 않은 파일을 짧은 것부터 순차 처리한다.

모델은 한 번만 로드해서 재사용한다 (파일마다 재로딩하면 낭비).
이미 transcript/<id>.<model>.txt 가 있으면 건너뛴다.
"""
import sys, time, pathlib
from faster_whisper import WhisperModel

MODEL_SIZE = sys.argv[1] if len(sys.argv) > 1 else "medium"
AUDIO_DIR = pathlib.Path("audio")
OUT_DIR = pathlib.Path("transcript")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def ts(s):
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(sec):02d},{int((sec%1)*1000):03d}"


pending = [p for p in sorted(AUDIO_DIR.glob("*.m4a"))
           if not (OUT_DIR / f"{p.stem}.{MODEL_SIZE}.txt").exists()]
pending.sort(key=lambda p: p.stat().st_size)  # 짧은 것부터

if not pending:
    print("[queue] 처리할 파일 없음")
    sys.exit(0)

print(f"[queue] {len(pending)}개 대기: {[p.stem for p in pending]}", flush=True)

t0 = time.time()
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=8)
print(f"[load] {MODEL_SIZE} in {time.time()-t0:.0f}s", flush=True)

for n, audio in enumerate(pending, 1):
    stem = audio.stem
    print(f"\n===== [{n}/{len(pending)}] {stem} =====", flush=True)
    segments, info = model.transcribe(
        str(audio),
        language="ko",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        condition_on_previous_text=False,
    )
    dur = info.duration
    print(f"[info] {dur/60:.1f}min", flush=True)

    t1 = time.time()
    with open(OUT_DIR / f"{stem}.{MODEL_SIZE}.txt", "w", encoding="utf-8") as ft, \
         open(OUT_DIR / f"{stem}.{MODEL_SIZE}.srt", "w", encoding="utf-8") as fs, \
         open(OUT_DIR / f"{stem}.{MODEL_SIZE}.plain.txt", "w", encoding="utf-8") as fp:
        last = 0
        for i, seg in enumerate(segments, 1):
            line = seg.text.strip()
            ft.write(f"[{int(seg.start//60):02d}:{int(seg.start%60):02d}] {line}\n")
            fs.write(f"{i}\n{ts(seg.start)} --> {ts(seg.end)}\n{line}\n\n")
            fp.write(line + " ")
            if seg.end - last > 60:  # 1분마다만 진행률 출력 (로그 폭주 방지)
                el = time.time() - t1
                print(f"  {seg.end/dur*100:5.1f}%  {el/60:5.1f}m elapsed", flush=True)
                last = seg.end

    el = time.time() - t1
    print(f"[done] {stem}: {el/60:.1f}min / {dur/60:.1f}min audio = {el/dur:.2f}x", flush=True)

print(f"\n[queue] 전체 완료. 총 {(time.time()-t0)/60:.1f}분", flush=True)
