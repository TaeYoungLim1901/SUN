#!/usr/bin/env bash
# CHECKLIST.md 를 매 턴 컨텍스트에 밀어 넣는다. 사람이 매번 "체크리스트 지켜라" 라고
# 말해야 했던 것을 없애는 것이 목적이다 (2026-08-20).
# 출력은 JSON 한 덩어리. additionalContext 가 모델 컨텍스트로 들어간다.
cat <<'JSON'
{"hookSpecificOutput":{"hookEventName":"UserPromptSubmit","additionalContext":"[SUN 체크리스트 — CHECKLIST.md 전문은 필요할 때 읽어라]\nA 세션: remote.py check · 종료예정 KST(서버시계 기준) 먼저 제시 · 새 실험 전 CLOSED_AXES.md grep\nB 위임: 구현·검색·검증은 codex(luna)/grok CLI. Claude 서브에이전트 금지. 판단은 위임 금지\nC 판정: 전체 Δ 하나로 채택·기각 금지. work/paired.py (6h 연결요소 11개 짝지음 + 절반분할 + 부트스트랩 CI) 로 두 번 잰다. σ 는 그 쌍에서 직접 잰다 — 다른 쌍의 σ 를 빌리지 마라\nD 제출: 남은 슬롯은 팀 통틀어 10회(총량). 조건 전부 충족해야 한 장: 그룹 2-fold×20 중앙 Δ<0 · 이득 ≥16/20 · 전체 val Δ ≤ −0.3 · 빈 폴더 재현 CSV 최대차 ≤1e-6. 미달이면 안 낸다. 보고는 5항목(버그체크 → 용량 → 재추론 일치 → 예상 Public±σ → 백업 경로)\nE 서버: CPU 2코어·RAM 8GB. CPU 잡 ≤2, GPU 잡 1개씩. setsid + PPID=1. 놀리지 마라\nF 규정: validation 을 train 에 합치기 금지 · 외부데이터/pretrained/pseudo-label 금지 · 좌우(동서) 반전 금지 · 문서 고치면 tools/conform.py"}}
JSON
