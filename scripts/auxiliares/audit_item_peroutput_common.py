#!/usr/bin/env python3
"""
Title: audit_item_peroutput_common.py

File type: Auxiliary analysis script (read-only re-analysis of pipeline artifacts)

Purpose:
    Per-output NAE (18) x [6 models + constant predictor] on the COMMON SUPPORT.
    Same logic as item 4 (seed-averaged, common support of 965 windows), reported per output.
    Read-only. Output: audits/item_peroutput_common.csv (18x8).

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
import numpy as np, pandas as pd
ROOT=str(Path(__file__).resolve().parents[2])  # repository root
WP,FR=ROOT,ROOT; OUT=ROOT+"/audits"
SEEDS=[42,123,456,789,1024]
LABELS=[f"{g}_{p}" for g in ["G1","G2","G3","G4","G5","G6"] for p in "ABC"]
PRED={"LSTM":"14_lstm_predictions","RCNN_att":"14_rcnn_att_predictions","Ensemble_G4":"15_ensemble_g4_predictions",
      "MoTE_v2":"15_mote_v2_predictions","PE_ES":"15_pe_es_predictions","SPEC":"15_spec_predictions"}
def nae(yt,yp): return float(np.sum(np.abs(yp-yt))/(np.abs(yt).sum()+1e-9)*100.0)
info={}
for m,pf in PRED.items():
    dd=pd.read_parquet(FR+f"/predictions/{pf}.parquet"); n=dd.window_idx.nunique(); off=(1001-n)//2
    info[m]=dict(dd=dd,n=n,off=off)
lo=max(v["off"] for v in info.values()); hi=min(v["off"]+v["n"]-1 for v in info.values())
cen=np.arange(lo,hi+1); print(f"common support [{lo},{hi}] = {len(cen)} windows")
yr=np.load(WP+"/data/processed/13_Y_raw_test.npy")
nullpo=pd.read_csv(OUT+"/item_null_predictor_peroutput.csv").set_index("output")
rows=[]
for j,lab in enumerate(LABELS):
    yt=yr[cen,j]; r={"output":lab}
    for m,v in info.items():
        ks=np.array([c-v["off"] for c in cen])
        s=[nae(yt, v["dd"][v["dd"].seed==sd].sort_values("window_idx")[lab].to_numpy(float)[ks]) for sd in SEEDS]
        r[m]=round(float(np.mean(s)),2)
    r["Constante"]=round(float(nullpo.loc[lab,"NAE_common"]),2)
    rows.append(r)
df=pd.DataFrame(rows)[["output","LSTM","RCNN_att","Ensemble_G4","MoTE_v2","PE_ES","SPEC","Constante"]]
df.to_csv(OUT+"/item_peroutput_common.csv",index=False)
print(df.to_string(index=False))
print("\nmeans:", {c:round(df[c].mean(),3) for c in df.columns if c!="output"})
