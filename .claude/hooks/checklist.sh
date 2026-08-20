#!/usr/bin/env bash
# CHECKLIST.md 요약 + **시계를 숫자로** 컨텍스트에 밀어 넣는다.
# 시계를 넣는 이유: 2026-08-20 밤에 모델이 경과 시간을 기억으로 계산해 두 번 틀렸다.
# 「서버 시계를 써라」는 문구로는 안 막혔다 — 값을 직접 준다 (sol 검토 2026-08-21).
export PYTHONIOENCODING=utf-8
python - <<'PY'
import json, subprocess, datetime, os
DEADLINE = datetime.datetime(2026, 8, 21, 15, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=9)))
KST = datetime.timezone(datetime.timedelta(hours=9))
now = datetime.datetime.now(datetime.timezone.utc).astimezone(KST)
rem = int((DEADLINE - now).total_seconds())
h, m = divmod(max(rem, 0) // 60, 60)
clock = (f"[시계 — 이 값을 써라. 경과·마감을 기억으로 계산하지 마라]\n"
         f"now_kst={now:%Y-%m-%d %H:%M:%S} · deadline_kst=2026-08-21 15:00:00 · "
         f"remaining_seconds={rem} ({h}시간 {m}분)")
rules = """[SUN 체크리스트 — CHECKLIST.md 전문은 필요할 때 읽어라]
A 세션: remote.py check · 종료예정 KST 는 위 시계 값을 그대로 쓴다 · 새 실험 전 CLOSED_AXES.md grep
B 위임: 구현·검색은 codex(luna)/grok CLI. Claude 서브에이전트 금지. 판단은 위임 금지.
  위임 산출물은 ast.parse 로 끝내지 마라 — **작은 입력으로 실제 실행**해 형상·채널을 확인한다 (2026-08-21 채널 버그)
C 판정 동결: paired.py 와 add_honest.py 가 **같은 방향**이 아니면 채택도 기각도 하지 마라.
  **새 지표를 만들어 채택 근거로 쓰는 것은 자동 기각** (bulk 지표가 아티팩트였다)
D 제출: 하루 5장·미사용 소멸. 누르기 전 D-0(제출이 필수적인가) 4문항.
  보고 5항목: 버그체크 → 용량 → 재현 최대차 → 판정(예상값은 σ 와 함께) → 백업 경로
E 서버: CPU 2코어·RAM 8GB. CPU 잡 ≤2, GPU 잡 1개. setsid + PPID=1. 놀리지 마라
F 규정: validation 을 train 에 합치기 금지 · 외부데이터/pretrained/pseudo-label 금지 ·
  test 타깃 추정 금지 · 좌우(동서) 반전 금지 · 문서 고치면 tools/conform.py
G 검사기: 기준 파일이 없으면 **건너뛰지 말고 실패시킨다.** 산출물을 자기 자신과 비교하는 검사는 검사가 아니다"""
print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                         "additionalContext": clock + "\n" + rules}}))   # ensure_ascii=True: 순수 ASCII 로 나가야 cp949 환경에서도 안 깨진다
PY
