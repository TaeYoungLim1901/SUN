# unused; not A_CM-GBDT / flow / west-limb / t-4d / tau. I[t=0..19,c=193/211,512,512]+wind[20]
1 ROBBINS_LAG_OCCUPANCY  d(θ)=3.69-0.07386θ[d]  V=330+930 I_s  |lat|<60  14° swath (CM→3.7d, E63°→8.3d)
https://arxiv.org/abs/astro-ph/0701275
r=0.47*512; dx,dy=(xx-255.5)/r,(yy-255.5)/r; disk=dx*dx+dy*dy<=1; lon,lat=deg(arcsin(clip(dx,-1,1))),deg(arcsin(clip(dy,-1,1)))
ch=(I[-1,0]<0.35*median(I[-1,0][disk]))&disk; th=arange(-77,41,7.0)
Is=[mean(ch[(abs(lon-t)<7)&(abs(lat)<60)&disk]) if any((abs(lon-t)<7)&disk) else 0 for t in th]
d=3.69-0.07386*th  # east θ<0 → larger d; west launches first
Vhat[h]=330+930*interp(0.25*(h+1), d[::-1], Is[::-1])  # Vršnak CM: V=350+900 A_±10, τ=4d https://arxiv.org/abs/1501.06697
2 HOFMEISTER_CO-LAT  v=478+(2.77e-9 A_km2)(1-|φ_co|/61.4); |φ|≳61.4° HSS misses Earth
https://arxiv.org/abs/1804.09579
r=0.47*512; dx,dy=(xx-255.5)/r,(yy-255.5)/r; disk=dx*dx+dy*dy<=1; mu=sqrt(clip(1-dx*dx-dy*dy,1e-3,1))
ch=(I[-1,0]<0.35*median(I[-1,0][disk]))&disk
A_km2=((ch/mu).sum())*(6.96e5/r)**2
phi_co=deg(arcsin(clip(mean(dy[ch]) if ch.any() else 0,-1,1)))
v_hof=478+(2.77e-9*A_km2)*(1-abs(phi_co)/61.4)
3 GARTON_DPHI  vmax=330.8+5.7Δθ (Δθ<67) else 710; Δt_SW=0.09Δθ[d] (HSS duration; models smear the ramp)
https://arxiv.org/abs/1811.11605
r=0.47*512; dx=(xx-255.5)/r; disk=dx*dx+((yy-255.5)/r)**2<=1; lon=deg(arcsin(clip(dx,-1,1)))
ch=(I[-1,0]<0.35*median(I[-1,0][disk]))&(I[-1,1]<0.45*median(I[-1,1][disk]))&disk  # 193∧211; no 171
dphi=(lon[ch].max()-lon[ch].min()) if ch.any() else 0
vmax=330.8+5.7*dphi if dphi<67 else 710.0
dt_sw=0.09*dphi; dVdt=(vmax-clip(wind[-1],300,500))/max(dt_sw,0.25)  # km/s/d
4 WEST_EDGE_SIR  t_SI=d(θ_W); Reiss ESWF: σ_t≈0.89d, σ_v≈118; high-v under 100–200; slow over
https://arxiv.org/abs/1607.07048
r=0.47*512; dx=(xx-255.5)/r; disk=dx*dx+((yy-255.5)/r)**2<=1; lon=deg(arcsin(clip(dx,-1,1)))
ch=(I[-1,0]<0.35*median(I[-1,0][disk]))&disk; thW=lon[ch].max() if ch.any() else 0  # west=+lon
t_SI=3.69-0.07386*thW; t_pk=3.69-0.07386*(mean(lon[ch]) if ch.any() else 0)
west=ch&(lon>=percentile(lon[ch],75)) if ch.any() else ch
edge_sharp=mean(I[-1,0][west])-mean(I[-1,0][ch]) if ch.any() else 0  # small → smeared SIR
5 RIGID_ROT_27D  CH live many CR + rigid rot ⇒ V(t)≈V(t−27d); no V[t−27]; proxy |ω−13.2°/d|
https://doi.org/10.1002/swe.20040
r=0.47*512; dx=(xx-255.5)/r; disk=dx*dx+((yy-255.5)/r)**2<=1; phi=[]
for t in range(20):
  c=(I[t,0]<0.35*median(I[t,0][disk]))&disk; phi+=[deg(arcsin(clip(mean(dx[c]) if c.any() else 0,-1,1)))]
omega=(phi[-1]-phi[0])/4.75; rec=abs(omega-13.2); Avar=std([(I[t,0][disk]<0.35*median(I[t,0][disk])).mean() for t in range(20)])
# rec≈0 & Avar small ⇒ long-lived CH (27d recurrence the 120h window cannot see) https://arxiv.org/abs/1501.06697
