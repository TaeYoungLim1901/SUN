목표: KASI-KAIST 태양 EUV 20프레임+풍속20 → 미래 풍속12 RMSE에서 Public 55.5576 개선.

반드시 읽기: RULES.md, DOMAIN.md §9, CLOSED_AXES.md, experiments/log.md, work/cnn10.py, work/cnn16.py.

제약:
- train 9607만 학습. validation 1199는 선택/판정만.
- 외부 데이터·pretrained weight·pseudo-label·test 시계열 재구성·좌우 flip 금지.
- A100 40GB, CPU2, RAM8GB, GPU 잡 1개.
- 닫힌 축 반복 금지: 모델 폭만 확대, learned temporal pooling, pinball, onset auxiliary-only, ramp reweight, optical-flow 파생, seed 확대, 사후보정, blend 재탐색.

작업:
1. 유사 시계열/태양풍 대회·논문에서 큰 도약을 낸 원리를 이 입력에 옮길 후보 최대 3개.
2. 레포의 명시적 재개 조건을 우선한다.
3. 각 후보가 기존 실패와 본질적으로 다른 이유, 누수/규정 위험, 최소 실험(1시드≤30분)을 적는다.
4. 최우선 후보의 정확한 입력/출력/손실/재구성식과 작은 텐서 shape test를 적는다.
5. 파일 수정 금지. 최종 채택 판단 금지.

수락 형식: 숫자 중심 표 + 최우선 실험 사양 20줄 이내.
