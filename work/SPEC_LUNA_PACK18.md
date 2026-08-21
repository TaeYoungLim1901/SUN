파일 2개만 수정: work/pipeline7.py, work/mkcard7.py. git sync/commit/push 금지. 다른 파일 금지.

목표: work/cnn18.py state_dict를 제출 추론에 arch `v18`로 지원.

pipeline7.py:
1. EncoderV6 입력 8채널·GRU·wind는 NetV9와 동일.
2. NetV18 state key/계산은 cnn18.py Net와 정확히 동일: trunk, amp, shape; softplus; softmax/max; `100*(up-down)`; 출력 (B,12).
3. FLOW_ARCHES에 v18 추가.
4. model 분기에 v18 추가.
5. v18도 Stonyhurst grid_sample을 적용(`arch in {v9,v18}`).
6. 기존 flow TTA(fy 부호) 사용.

mkcard7.py:
1. cnn18 접두 태그는 arch v18, norm med, wm 1.0.
2. 기존 cnn7/v9 동작 불변.

수락 명령:
`python -m py_compile work/pipeline7.py work/mkcard7.py`
`python -c "import ast; [ast.parse(open(p,encoding='utf-8').read()) for p in ('work/pipeline7.py','work/mkcard7.py')]; print('OK')"`

완료 출력: 수정 위치와 수락 명령 결과만. 설계 판단 금지.
