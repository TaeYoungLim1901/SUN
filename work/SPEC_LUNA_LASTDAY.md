목표: 2026-08-21 15:00 KST 전 Public 55.5576을 개선할 합법적 새 모델 축 조사.

읽기: NEXT.md, RULES.md, DOMAIN.md §9, CLOSED_AXES.md, experiments/log.md, work/cnn10.py, work/cnn16.py.

금지: 외부 데이터/가중치, validation 학습, test target 추정·재구성, 좌우 flip, 닫힌 축 반복.
닫힘: 단순 폭 확대, attention pooling, onset 보조손실만, pinball, 램프 가중, 시드 추가, 광류 CNN 변형, 사후보정, 재가중만.

작업:
1. 레포에서 아직 실행되지 않은 재개 조건과 구현 가능한 구조를 검색한다.
2. 후보 최대 3개를 파일:줄 근거와 함께 제시한다.
3. 최우선 하나는 work/cnn10.py 기준 최소 변경 지점을 의사 diff로 쓴다.
4. 출력 12 형상, train-only 라벨, 예상 GPU 시간, 실패 판정 명령을 적는다.
5. 파일 수정 금지. 설계 채택 판단 금지.

수락 형식: 후보별 `근거 | 새 정보 | 최소 변경 | 반증 시험`; 마지막에 최우선 1개.
