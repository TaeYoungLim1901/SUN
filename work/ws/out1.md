# Q1 EUV input reps 2019–2026. Rank = E[ensemble ΔRMSE]/cost. EUV=193/211 PNG + past V only.

|#|rep|compute from 512px (no FITS/B0)|skill + URL|EUV-only?|h|ρ-risk|
|-|-|-|-|-|-|-|
|1|**polar/Stonyhurst waterfall tensor** (lon×lat×t pixels, not scalars)|limb-fit R,c; remap (r,θ) B0≈0; stack 20fr; ~13.3°/d x-shift|SpeedNet Stonyhurst persist maps > raw disk https://doi.org/10.1029/2025EA004523 ; Collin 4×3 grid **RMSE 68.1 CC.70** vs persist89.1 https://arxiv.org/abs/2410.05068 ; our polar-GBDT **67.68**|yes|3|**med** if pixels kept; high if GAP→eswf|
|2|**persist / binary CH maps as CNN ch** (SpeedNet-BM)|k=3 k-means 193+211 (not 0.35×med); min_t after x-shift; binary extra stream|BM **RMSE < EUV** all SC phases https://doi.org/10.1029/2025EA004523 ; WindNet 193/211 **76.4/.55** vs persist93 https://doi.org/10.1029/2020SW002478 ; Brown Swin **72.21/.634** https://doi.org/10.1029/2021SW002976|yes|2|**med-high** vs flow-CNN; 2-stream mask still open|
|3|**CH A+colat series → CNN-head** (Hofmeister/CATCH)|Otsu/k-means mask; A,φ_com,long/lat extent ×20|**v=478+(2.77e-9 A)(1−|φ|/61.4)** peak RMSE70 CC.70 https://arxiv.org/abs/1804.09579 ; Samara long-ext **cc.57** (.75 if A≲6e10 km²) https://arxiv.org/abs/2204.00368|yes (CATCH=EUV)|0.5|**high** vs eswf67.03/chfeat67.25; NEXT hybrid untested|
|4|CM ±7.5° strip A(t) (Rotter/ESWF)|frac dark in x∈[cx±0.13R]|Rotter peak **cc≈.60**, 80% ±0.5d https://arxiv.org/abs/1501.06697 ; Collin ESWF **91.8/.43** https://arxiv.org/abs/2410.05068 ; **our rotter96.20 eswf67.03**|yes|1|**dead** (eswf in N50; E013)|
|5|Carrington 360° synoptic|need 27d|WSA uses **mag** synoptic not EUV; window=**63°** only DOMAIN§9.1|no mag+t|—|**illegal**|
|6|CHIMERA 171/193/211|171 missing|CHIMERA=3λ https://www.swsc-journal.org/articles/swsc/abs/2018/01/swsc170041/swsc170041.html ; SPoCA-HEK low-lat A–V **cc.3** vs KMAS.9 https://arxiv.org/abs/2507.21641|**no 171**|—|seg family **closed E015** (k-means71.4 ρ.875)|
|7|open-flux proxy|need B·dA|EUV CH area **≠** in-situ open flux ×2–4 https://doi.org/10.3847/1538-4357/ac090a|**no mag**|—|**illegal** §9.1|
|8|disk-int. 193/211 irradiance|mean/median on-disk|no 4d HSS skill 2019–26; tracks cycle not streams|yes|0.5|**~0**; =cnn-med|

**Do #1 pixels.** Don't: 5–8, Rotter-linear, CHIMERA, 0.35×med. NYUAD 55–58 km/s uses **27/54d V + ImageNet** — illegal https://arxiv.org/abs/2507.17298
