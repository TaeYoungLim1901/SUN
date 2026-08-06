"""유튜브 영상 전사. CPU int8 전용 (이 노트북에 전용 GPU 없음).

사용법: python transcribe.py <audio> [model_size]
출력:  transcript/<id>.<model>.txt  (타임스탬프 + 본문)
       transcript/<id>.<model>.srt  (자막 형식)
       transcript/<id>.<model>.plain.txt  (타임스탬프 없는 순수 본문)
"""
import sys, time, pathlib
from faster_whisper import WhisperModel


def main(audio: str, model_size: str = "medium", outdir: str = "transcript"):
    t0 = time.time()
    print(f"[load] {model_size} (int8, cpu)", flush=True)
    model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=8)
    print(f"[load] done in {time.time()-t0:.0f}s", flush=True)

    segments, info = model.transcribe(
        audio,
        language="ko",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        condition_on_previous_text=False,  # 반복 폭주 방지
    )
    dur = info.duration
    print(f"[info] audio {dur/60:.1f}min", flush=True)

    stem = pathlib.Path(audio).stem
    out = pathlib.Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    txt_path = out / f"{stem}.{model_size}.txt"
    srt_path = out / f"{stem}.{model_size}.srt"
    plain_path = out / f"{stem}.{model_size}.plain.txt"

    def ts(s):
        h, r = divmod(s, 3600)
        m, sec = divmod(r, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(sec):02d},{int((sec%1)*1000):03d}"

    t1 = time.time()
    with open(txt_path, "w", encoding="utf-8") as ft, \
         open(srt_path, "w", encoding="utf-8") as fs, \
         open(plain_path, "w", encoding="utf-8") as fp:
        for i, seg in enumerate(segments, 1):
            line = seg.text.strip()
            ft.write(f"[{int(seg.start//60):02d}:{int(seg.start%60):02d}] {line}\n")
            fs.write(f"{i}\n{ts(seg.start)} --> {ts(seg.end)}\n{line}\n\n")
            fp.write(line + " ")
            el = time.time() - t1
            eta = (el / max(seg.end, 1)) * (dur - seg.end)
            print(f"  {seg.end/dur*100:5.1f}%  elapsed {el/60:5.1f}m  eta {eta/60:5.1f}m  | {line[:40]}", flush=True)

    el = time.time() - t1
    print(f"[done] {el/60:.1f}min for {dur/60:.1f}min audio = {el/dur:.2f}x realtime", flush=True)
    print(f"[out] {txt_path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "medium")
