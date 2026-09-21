#!/usr/bin/env python3
"""
Title: audit_item4_common_support.py

File type: Auxiliary analysis script (read-only re-analysis of pipeline artifacts)

Purpose:
    Layer 3 / item 4 (Tier A): NAE on the TEMPORAL SUPPORT COMMON to the 6 models.

    Intersection of the window centres: test instant in [max offset, min(offset+n-1)].
    Recomputes the active NAE per model on that common support and checks whether the ranking
    is preserved. Also repeats the OBSERVED-TARGET control on the common support. Read-only.

Repository: harmonica-nilm-reproducibility
Version: v2.0.0
Date: 2026-09-21

Developer:
    Prof. Wesley Pacheco Calixto, Dr.

Authors:
    1. Wesley Pacheco Calixto
       Federal University of Goias / University of Coimbra /
       Federal Institute of Goias, Brazil

    2. Jose A. Gobbes Cararo
       Federal University of Goias / Federal Institute Goiano, Brazil

    3. Guilherme A. Sousa Ribeiro
       Federal University of Goias / Federal Institute of Goias, Brazil

    4. Paulo Victor Santos
       Federal University of Goias / Federal Institute of Goias, Brazil

Corresponding author:
    Wesley Pacheco Calixto
    wesley.pacheco@isr.uc.pt

License:
    MIT

Citation:
    Calixto, W. P., Cararo, J. A. G., Ribeiro, G. A. S., & Santos, P. V.
    (2026). Reproducibility package for non-intrusive load monitoring
    in industrial three-phase multi-source systems (Version 2.0.0)
    [Computer software]. Zenodo.
    https://doi.org/10.5281/zenodo.22878924

Related publication:
    Scale-admissibility gate for non-intrusive load monitoring in
    three-phase industrial energy systems with an unmeasured support
    source. Applied Energy.
"""
from pathlib import Path
import os, numpy as np, pandas as pd
ROOT=str(Path(__file__).resolve().parents[2])  # repository root
WP,FR=ROOT,ROOT; OUT=ROOT+"/audits"
DT_H,E_MIN=1/60.0,1.0; SEEDS=[42,123,456,789,1024]
LABELS=[f"{g}_{p}" for g in ["G1","G2","G3","G4","G5","G6"] for p in "ABC"]
LOAD_METER={"G1":5,"G2":6,"G3":7,"G4":11,"G5":15,"G6":16}; PHASE_COL={"A":"p_a","B":"p_b","C":"p_c"}
PRED={"LSTM":"14_lstm_predictions","RCNN_att":"14_rcnn_att_predictions","Ensemble_G4":"15_ensemble_g4_predictions",
      "MoTE_v2":"15_mote_v2_predictions","PE_ES":"15_pe_es_predictions","SPEC":"15_spec_predictions"}
def nae(yt,yp): return float(np.sum(np.abs(yp-yt))/(np.abs(yt).sum()+1e-9)*100.0)

# offsets/centres per model
info={}
for m,pf in PRED.items():
    dd=pd.read_parquet(FR+f"/predictions/{pf}.parquet"); n=dd.window_idx.nunique(); off=(1001-n)//2
    info[m]=dict(dd=dd,n=n,off=off)
lo=max(v["off"] for v in info.values()); hi=min(v["off"]+v["n"]-1 for v in info.values())
print(f"common support: test instants [{lo},{hi}] = {hi-lo+1} windows (centres)")

# observed mask (conf==0) per output, test-local
conf=pd.read_parquet(WP+"/figdata/08_fill_confidence.parquet"); obs={}
for lab in LABELS:
    g,p=lab.split("_"); s=conf[conf.meter_id==LOAD_METER[g]].sort_values("time")[PHASE_COL[p]].to_numpy()
    obs[lab]=(s[-1000:]==0)
yr=np.load(WP+"/data/processed/13_Y_raw_test.npy")

rows=[]
for m,v in info.items():
    dd,n,off=v["dd"],v["n"],v["off"]
    ks=np.array([c-off for c in range(lo,hi+1)])          # window_idx of the model whose centres lie in [lo,hi]
    cen=np.arange(lo,hi+1)                                  # common central test instant
    na_full,na_obs=[],[]
    for j,lab in enumerate(LABELS):
        yt=yr[cen,j]; om=obs[lab][cen]; e_true=yt.sum()*DT_H/1000.0
        if abs(e_true)<E_MIN: continue
        f_s,o_s=[],[]
        for sd in SEEDS:
            yp_all=dd[dd.seed==sd].sort_values("window_idx")[lab].to_numpy(float)
            yp=yp_all[ks]
            f_s.append(nae(yt,yp))
            if om.sum()>=30: o_s.append(nae(yt[om],yp[om]))
        na_full.append(np.mean(f_s)); na_obs.append(np.mean(o_s) if o_s else np.nan)
    rows.append(dict(model=m, NAE_common=round(np.mean(na_full),3),
                     NAE_common_observed=round(np.nanmean(na_obs),3)))
df=pd.DataFrame(rows).sort_values("NAE_common")
df.to_csv(OUT+"/item4_common_support.csv",index=False)
print("\n=== Active NAE on the COMMON SUPPORT (965 windows), sorted ===")
print(df.to_string(index=False))
print("\n(compare with item 2 / own support: MoTE 4.78 < Ens 5.19 < LSTM 5.21 < RCNN 5.72 < PE-ES 6.19 < SPEC 6.56)")
