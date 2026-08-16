"""9개 영상 URL 검색. yt-dlp ytsearch로 제목 검색해 상위 후보를 출력한다."""
import json
from yt_dlp import YoutubeDL

QUERIES = {
    1: "[별소리 다 듣겠네] 해와 달과 지구가 만드는 우주쇼 일식과 코로나 YTN 사이언스",
    2: "다큐S프라임 강력한 태양 폭풍이 온다 YTN 사이언스",
    3: "2025년은 태양 활동이 가장 활발한 해 태양 활동 극대기가 지구에 미치는 영향 과학토크쇼 취미는 과학",
    5: "2021년 하계 교원천문연수 8 태양과 우주환경 김록순",
    6: "2021년 하계 교원천문연수 4 전파망원경의 이해 정태현",
    7: "2021년 하계 교원천문연수 1 천체망원경의 오늘과 내일 김상철",
    8: "우문과답 #2 우주에서 온 메시지는 한국천문연구원",
    9: "[우주알고주알] 우주에도 날씨가 있나요 한국천문연구원",
}

opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}
out = {}
with YoutubeDL(opts) as ydl:
    for n, q in QUERIES.items():
        info = ydl.extract_info(f"ytsearch5:{q}", download=False)
        cands = [
            {
                "title": e.get("title"),
                "url": f"https://www.youtube.com/watch?v={e.get('id')}",
                "channel": e.get("channel") or e.get("uploader"),
                "duration": e.get("duration"),
            }
            for e in info.get("entries", [])
        ]
        out[n] = cands
        print(f"\n=== {n}. {q[:40]} ===")
        for i, c in enumerate(cands):
            print(f"  [{i}] {c['title']}  | {c['channel']} | {c['duration']}s")
            print(f"      {c['url']}")

with open("D:/hackathon/SUN/tools/url_candidates.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
