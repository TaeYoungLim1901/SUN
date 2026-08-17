# 에이전트용 안내

**[README.md](README.md) 를 읽어라.** 이 저장소의 지도·규칙·주의사항이 전부 거기 있다.

내용을 여기 복사하지 마라. 두 벌이 되면 갈라지고, 갈라진 문서가 이 저장소의
주된 실패 원인이었다 (`POSTMORTEM.md`).

바로 알아야 할 것 셋:

1. `RULES.md` 가 대회 규정의 **단일 기준**이다. 다른 문서와 어긋나면 다른 문서가 틀렸다.
2. `DOMAIN.md` §9 는 규정 때문에 **계산 불가능해진 피처 목록**이다. 피처를 제안하기 전에 읽어라.
3. 문서를 고쳤으면 `python tools/conform.py` 를 돌려라. 1초, 공짜다.

---

<!-- 여기부터는 내 개인 작업 방식이다. `main` 브랜치에만 있고 팀 레포로는 안 간다.
     위쪽 팀 문단은 건드리지 마라 — 팀원이 고치면 충돌난다. -->

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
2. `GROK_REVIEW.md` **§0–§3만**, `ARENA.md` 만 읽는다. DOMAIN/PLAN/PDF/REVIEW/POSTMORTEM 전문 금지.
3. `GROK_REVIEW.md` §3 의 스텝을 순서대로. 수락 통과 전에 다음 스텝으로 넘어가지 마라.
4. 구현은 기본으로 직접 하거나, 단순 치환은:

```
codex exec -m gpt-5.6-luna -c model_reasoning_effort="xhigh" --skip-git-repo-check "<스펙>"
```

`-m luna` 금지 (400). sol / Grok 은 사람이 그 이름을 말하기 전에는 호출 금지.

5. 스텝마다 `DONE: path:line — 한 줄`. 에세이 금지.
6. 스텝이 끝나면 `python tools/conform.py` 와 `python tools/kit.py` 를 돌리고 멈춰라.
   학습 루프·새 모델 시작 금지.

첫 메시지에 「작업 시작」만 있어도 위 순서를 바로 한다. 확인 질문 하지 마라.

## 참고

옛 중복본과 1차 출처 PDF 6편은 `D:\hackathon\SUN_old_backup\` 에 있다.
**열지도 고치지도 마라.** `tools/factcheck.py --corpus` 가 PDF 를 쓸 때만 그 경로를 준다.
