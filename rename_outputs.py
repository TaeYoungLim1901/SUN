"""전사 결과물 파일명을 유튜브 ID -> 사람이 읽을 수 있는 제목으로 바꾼다.

전사가 전부 끝난 뒤에 실행할 것. (도중에 실행하면 큐가 중복 작업을 한다)
"""
import pathlib, shutil

# 유튜브 ID -> 파일명으로 쓸 제목 (윈도우에서 금지된 문자 \/:*?"<>| 는 제외)
TITLES = {
    "fEmf1SFniHE": "교원연수1_천체망원경의_오늘과_내일_김상철",
    "ThQFDIgZmT0": "교원연수4_전파망원경의_이해_정태현",
    "3--oD514lOU": "교원연수8_태양과_우주환경_김록순",
    "RGqZiu59Zzo": "ROKITS_누리호4차_우주에서_오로라_보기",
    "fVnffH5pCRY": "과학토크쇼_태양활동_극대기가_지구에_미치는_영향",
    "2gSJuzmBuBI": "우주알고주알_우주에도_날씨가_있나요",
    "wXxFJHA2tFE": "YTN_별소리다듣겠네_일식과_코로나",
    "oEf5HVk_A8g": "YTN_다큐S프라임_강력한_태양폭풍이_온다",
}

OUT = pathlib.Path("transcript")
renamed = skipped = 0

for f in sorted(OUT.iterdir()):
    if not f.is_file():
        continue
    vid = next((v for v in TITLES if f.name.startswith(v)), None)
    if vid is None:
        print(f"  [skip] 매핑 없음: {f.name}")
        skipped += 1
        continue
    new_name = f.name.replace(vid, TITLES[vid], 1)
    if new_name == f.name:
        continue
    target = f.with_name(new_name)
    if target.exists():
        print(f"  [skip] 이미 있음: {new_name}")
        skipped += 1
        continue
    shutil.move(str(f), str(target))
    print(f"  {f.name}  ->  {new_name}")
    renamed += 1

print(f"\n[done] {renamed}개 변경, {skipped}개 건너뜀")
