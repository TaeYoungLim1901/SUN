공식 대회 페이지와 GitHub·웹 검색 기준, **동일 데이터/val을 검증할 수 있는 추가 리포는 찾지 못했습니다**. 노트북·후기도 `torazang65`의 문서 외에는 확인되지 않았습니다. ([공식 대회](https://kasiai.kaist.ac.kr/))

| 공개 자료 | 마지막 커밋 | val / 비고 |
|---|---:|---|
| [`torazang65/solar-wind`](https://github.com/torazang65/solar-wind) | 2026-08-20 20:17 UTC | 동일 대회, best val **66.889**; [commit](https://github.com/torazang65/solar-wind/commit/2136992ed57bdb8cedd3c4dd13d3caa928008030) |
| [`Taegyu-Lee1117/Astrospace-AI-Competition`](https://github.com/Taegyu-Lee1117/Astrospace-AI-Competition) | 2026-03-10 05:13 UTC | 데이터 포맷·RMSE 기록 없음 |
| [`tmdekd/Astrospace-AI-Competition`](https://github.com/tmdekd/Astrospace-AI-Competition) | 2024-10-11 08:50 UTC | 포스터/결과 이미지만; 현 대회 val 없음 |
| [`maeng00/2024_KASI_KAIST_Space_AI_Competition`](https://github.com/maeng00/2024_KASI_KAIST_Space_AI_Competition) | 2025-12-07 17:58 UTC | 2024 탐지 대회, F1 0.884483; RMSE 아님 |
| [`LeeChangmin0310/2024_KASI_KAIST_Space_AI_Competition`](https://github.com/LeeChangmin0310/2024_KASI_KAIST_Space_AI_Competition) | 2025-03-10 04:41 UTC | 2024 탐지 대회, 현 대회와 무관 |

| 후보 기법 및 공개 성과 | 닫힌 축과 다른 점 / 최소 diff | 1시드 GPU |
|---|---|---:|
| MMP: EUV CNN + 풍속 시계열 CNN/BiLSTM + late fusion. 논문 RMSE 68 vs 90 ([논문](https://doi.org/10.34133/2022/9805707)) | 현재 `wind` 40차원 MLP를 `(20, 풍속·차분)` sequence encoder로 교체. 이미지 temporal attention·CH·remap 없음. pretrained GoogLeNet은 제거, random init. `cnn3.py`/`pipeline7.py`/`mkcard7.py` | **추측 10–25분** |
| TDAM: feature×lag attention + GRU. RMSE 62.882 vs 67.100 ([논문](https://doi.org/10.1029/2020SW002707)) | 허용된 풍속·차분 2×20에만 attention 적용; `cnn15`의 이미지 시간 attention과 다름. `cnn3.py` wind branch와 새 tag만 추가 | **추측 10–20분** |
| Spatial patch attention: per-frame CNN feature map을 4×4 tokens로 attention pooling. Solar Swin 72.21 vs 81.21 ([논문](https://doi.org/10.1029/2021SW002976)) | 시간축은 기존 GRU 유지, 공간 토큰만 변경. pretrained·좌우/남북 flip은 제거. `cnn3.py` encoder 최소 변경 | **추측 20–45분** |

공개 논문의 RMSE는 데이터·입력·금지조건이 달라 직접 비교할 수 없습니다. MMP/TDAM 논문에 포함된 외부·pretrained 입력은 사용하지 않는 전제입니다.

실패 판정은 각 tag에 대해:

```text
python work/add_honest.py --base FLOW55 <태그>
python work/paired.py preds/val_FLOW55.npy preds/val_<태그>.npy FLOW55 <태그>
```

동결 기준(저장소 기준): `add_honest` 중앙 개선 및 승률 조건, `paired` 전체 Δ ≤ −0.3·component gain ≥80%·bootstrap CI 상한 < 0을 함께 확인하며, 불일치하면 판정 보류. 파일 수정·채택 판단은 하지 않았습니다.
