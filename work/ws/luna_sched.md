결론: **#1 LR peak 축소 → #2 EMA(β=.995) → #3 AdamW weight decay 증가**. BN은 batch=128이면 후순위다.

1) 원인  
- **LR(가장 유력):** `20×ceil(9607/128)=1520` step, `pct_start=.3`의 LR 정점이 약 ep6; 실제 유일한 대폭락 ep6=86.2와 일치한다.  
- **BN:** 동일 checkpoint를 val batch 크기/순서만 바꿔 RMSE가 변하면 BN, 변화가 없으면 기각.  
- **데이터 크기:** 1,199×12=14,388 cell, iid 근사 RMSE 표준오차는 약 `67/√(2N)=0.4`; 고정 val이 ep6에만 +18.5가 될 수 없다.  
- **잔차 스케일:** horizon별 gradient norm 변동이 residual σ와 동조하고 z-score 후 σ가 ≥30% 줄 때만 원인; 현재 bf16 overflow 증거는 없다.

2) 변경 순위  
① **LR:** [One-Cycle 근거](https://arxiv.org/abs/1708.07120)와 ep6 정점 일치. 한 줄: `sched = OneCycleLR(opt, 5e-4, total_steps=EPOCHS*steps, pct_start=.3)`. 고정 ep20 RMSE `≤66.1`(현 67.1)이면 GO.  
② **EMA:** [EMA 실증](https://arxiv.org/abs/2411.18704). 최소 설정: `AveragedModel(model, avg_fn=lambda a,p,n:.995*a+.005*p, use_buffers=True)`; step 뒤 update, ep20 EMA만 평가. `≤66.1`이면 GO.  
③ **정규화 대신 WD:** [AdamW 실증](https://arxiv.org/abs/1711.05101). 한 줄: `AdamW(..., lr=1e-3, weight_decay=.10)`. 고정 ep20 `≤66.1`이면 GO, 아니면 폐기.

3) EMA ≠ SWA  
- SWA last8은 epoch checkpoint 8개에 각각 12.5%; 실측 **+1.07 RMSE**, 진폭 `70→60`도 악화했다. [SWA](https://arxiv.org/abs/1803.05407)와 같은 “평균”이지만 표본 선택이 다르다.  
- EMA `.995`, 76 step/epoch이면 half-life `138 step=1.8 epoch`, 유효창 `200 step=2.6 epoch`, 8 epoch 이전 질량은 `.995^608≈4.7%`; 매 step 최근값을 더 크게 반영한다.  
- 따라서 SWA 실패가 EMA 실패를 뜻하지는 않지만, EMA도 반드시 **고정 ep20 RMSE**로 판정한다.

4) 남은 여유  
- seed curve상 `32→∞` 개선은 **약 0.025 RMSE**뿐이다.  
- 현재 best `64.9`→seed floor `63.81`의 이론적 여유는 약 **1.09 RMSE**; 평상 ep20 `67.1` 기준 상한은 **3.29**다. 실질적 여유는 seed가 아니라 **LR 진동·epoch 선택편향(2.77/1.34 RMSE)·과적합 궤적**에 있다.
