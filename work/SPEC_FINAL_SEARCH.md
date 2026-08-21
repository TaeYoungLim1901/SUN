목표: 2026-08-21 15:00 KST 마감. 남은 제출 슬롯 **1장**. 현재 팀 최고 Public **55.5576**(flow55, 이미 제출·확보됨).
1위 VILAB 55.4648. 필요한 개선 = Public −0.093 이상.

규칙: 최종 점수는 **제출본 중 Public 최고가 자동 선택**되고, 그 카드의 **Public + Private 합산**이 최종 점수다.
즉 이미 낸 flow55 는 하방이 없다. 새 제출은 (a) Public 이 55.5576 보다 좋아야 선택되고,
(b) 선택되면 그 카드의 Private 도 최종에 들어간다. 따라서 **기대 실력이 같거나 나은 카드만** 의미가 있다.

문제: KASI-KAIST 태양풍 대회. 입력 = EUV 이미지 20프레임(6시간 간격 창) + 과거 풍속 20개.
출력 = 미래 풍속 12스텝. 지표 RMSE(km/s). train 9607 · validation 1199 · test 3868.

금지(실격): 외부 데이터 · pretrained weight · pseudo-label · test 타깃 추정/재구성 ·
validation 을 train 에 합치기 · 좌우(동서) flip 증강 · 대회 데이터 로컬 복사.

**반드시 먼저 읽어라: CLOSED_AXES.md 전문 · experiments/log.md · HANDOFF_CODEX.md.**
거기 닫힌 축을 다시 제안하면 그 제안은 자동 기각이다.

너의 작업 (검색 중심):
1. **GitHub·웹에서 이 대회 관련 공개 리포·노트북·후기를 다시 한 번 전수 탐색.**
   알려진 것: `torazang65/solar-wind` (같은 대회, best val 66.889 — 우리보다 나쁨).
   그 외에 KASI / KAIST / 성남 / "solar wind" / "competition_dataset_6h" / "image_00..19" /
   "target_00..11" / 태양풍 예측 대회 키워드로 **새 리포가 있는지** 확인하고 URL·마지막 커밋·val 수치를 적어라.
2. 유사 문제(태양풍 속도 예측, 시계열+영상 멀티모달 회귀)에서 **RMSE 를 크게 줄인 공개 기법** 중
   위 금지에 걸리지 않고 **9시간 안에 1시드 실험이 가능한** 것 최대 3개.
3. 각 후보마다: 왜 우리 닫힌 축과 본질적으로 다른가 · 최소 구현 diff 지점 · 1시드 GPU 시간 추정 ·
   실패 판정 명령(`work/add_honest.py --base FLOW55 <태그>` 와 `work/paired.py`).

출력: 표 중심, 40줄 이내. 파일 수정 금지. 채택 판단 금지. 근거 없는 추측은 "추측"이라고 표시.
