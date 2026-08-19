# 에이전트용 안내

**작업을 시작하면 [NEXT.md](NEXT.md) 부터 읽어라.** 지금 성적·서버 상태·다음 레버가 거기 있다.
그 다음이 [README.md](README.md) (저장소 지도·규칙·주의사항).

내용을 여기 복사하지 마라. 두 벌이 되면 갈라지고, 갈라진 문서가 이 저장소의
주된 실패 원인이었다 (`POSTMORTEM.md`).

바로 알아야 할 것 셋:

1. `RULES.md` 가 대회 규정의 **단일 기준**이다. 다른 문서와 어긋나면 다른 문서가 틀렸다.
2. `DOMAIN.md` §9 는 규정 때문에 **계산 불가능해진 피처 목록**이다. 피처를 제안하기 전에 읽어라.
3. 문서를 고쳤으면 `python tools/conform.py` 를 돌려라. 1초, 공짜다.

---

# 내 작업 방식 (main 브랜치 전용)

## 레포 둘, 브랜치 둘

두 레포는 포함 관계가 아니다. 팀 레포엔 내가 모르는 팀원 커밋이 들어오고,
개인 레포엔 팀이 볼 필요 없는 내 문서가 있다. 교집합만 공유한다.

| | 브랜치 | 원격 | 내용 |
|---|---|---|---|
| 개인 | `main` | `origin` = `TaeYoungLim1901/SUN` (private) | 내 작업분 전부 |
| 팀 | `shared` | `team` = `andy6609/sungnam-to-kaist` | 주고받는 것만 |

```
python tools/push.py                        # main 커밋 → 개인 레포
python tools/push.py --share DOMAIN.md ...  # 그 파일만 팀에 공개
python tools/push.py --sync                 # 팀원 작업 받아 main 에 합치기
```

**`shared` 브랜치에서 `git add -A` 금지.** 개인 파일이 팀 레포로 샌다. 공개는 `--share` 로만.
팀에 안 가는 것: 이 문단 아래 전부, `ARENA.md`, `GROK_REVIEW.md`,
`tools/push.py`, `tools/{data_probe,omni_fill,make_submission}.py`.

## 「작업 시작」

인사·재검토·계획 재작성 없이 즉시:

1. `git pull` (= `python tools/push.py --sync`).
2. `NEXT.md` 를 읽는다. 그게 지금 상태의 단일본이다.
   DOMAIN/PLAN/PDF/REVIEW/POSTMORTEM 전문 금지 — 지정된 절·줄만.
3. `NEXT.md` §3 의 레버를 그 순서로. 하나가 수락(숫자)을 통과하기 전에 다음으로 넘어가지 마라.
4. 새 실험 전에 `CLOSED_AXES.md` grep. 실험 하나 = `val_*.npy` + `test_*.npy` +
   `experiments/log.md` 한 줄. 글이 아니라 파일이 판정이다.
5. 단순 치환·검색은 위임한다:

```
codex exec -m gpt-5.6-luna -c model_reasoning_effort="xhigh" --skip-git-repo-check "<스펙>"
```

`-m luna` 금지 (400). 위임 스펙은 파일 1–2개 + 수락 명령 한 줄. 설계·판단은 위임하지 마라.

6. 스텝마다 `DONE: path:line — 한 줄`. 에세이 금지.
7. 문서를 고쳤으면 `python tools/conform.py`, 채점기를 건드렸으면 `python tools/kit.py`.

첫 메시지에 「작업 시작」만 있어도 위 순서를 바로 한다. 확인 질문 하지 마라.

8. 운영 규칙(위임·bug check·제출 슬롯·서버 한도)은 [`CHECKLIST.md`](CHECKLIST.md). 긴 작업·제출 전에 그 파일을 훑는다.

## 참고

옛 중복본과 1차 출처 PDF 6편은 `D:\hackathon\SUN_old_backup\` 에 있다.
**열지도 고치지도 마라.** `tools/factcheck.py --corpus` 가 PDF 를 쓸 때만 그 경로를 준다.
