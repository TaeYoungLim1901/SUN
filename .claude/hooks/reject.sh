#!/usr/bin/env bash
# 판정 스크립트(contrib/paired/add_honest/blend_group/noise_*)를 돌릴 때마다
# 기각 절차(CHECKLIST.md G)를 컨텍스트에 밀어 넣는다.
# 목적: 「숫자 하나 보고 기각」을 구조적으로 막는다 (2026-08-20 사람 지시).
cmd=$(cat)
case "$cmd" in
  *contrib.py*|*paired.py*|*add_honest.py*|*blend_group.py*|*noise_floor.py*|*noise_check.py*|*bagtest*)
    cat <<'JSON'
{"hookSpecificOutput":{"hookEventName":"PostToolUse","additionalContext":"[기각/채택 절차 — CHECKLIST.md G]\n1) 숫자 하나로 닫지 마라. 독립 측정 둘이 같은 방향이어야 한다: work/paired.py (성분별 부호+절반분할 σ+부트스트랩 CI) AND work/add_honest.py (w 를 반쪽에서 고르고 반대쪽 채점, 2-fold×20).\n2) contrib.py 의 Δ 는 val 전체에서 w 를 고른 **낙관치**다. 그것만으로 채택도 기각도 금지.\n3) σ 는 그 쌍에서 직접 잰다. 다른 쌍의 σ 를 빌리면 결론이 뒤집힌다 (k 축 3회 전복의 원인).\n4) 단독 성적이 나쁘다는 건 기각 사유가 아니다. 판정 대상은 ρ(잔차)와 결합 기여다.\n5) 기각 전에 설정 원인 하나는 제거해 본다 (증강·스케줄·격자 끝단 w 등).\n6) 기각하면 CLOSED_AXES.md 에 재개 조건과 함께 기록한다.\n7) 측정 끝나기 전에는 그 축에 대해 아무 말도 하지 않는다."}}
JSON
    ;;
esac
exit 0
