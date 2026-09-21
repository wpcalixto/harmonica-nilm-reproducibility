#!/usr/bin/env python3
"""
Title: audit_item3_pother_completeness.py

File type: Auxiliary analysis script (read-only re-analysis of pipeline artifacts)

Purpose:
    Layer 3 / item 3 (Tier A): completeness sensitivity of P_other.

    EXACT mask of P01 (05_select_meter_scope: operational = i_an > 0.5 A) applied on the
    1-minute GRID (mean per meter-minute cell, ingestion rule). C(t) = number of operational
    components {10,12,13,14} at minute t. Validated against the manuscript (phase A: C>=1=2196,
    C=1=1349, C=4=97; canonical artifact 06_alignment_coverage). Reports, per phase and
    completeness level (exact strata C=1..4 and nested supports C>=1,>=2,>=3,=4): n, % of grid,
    duration, E_MG and E_loads on the SAME support, energy of the partial aggregate, closure
    residual (kWh, kWh/h, %|E_MG|). Read-only; uncertainty is descriptive (n and duration
    reported).

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
ROOT = str(Path(__file__).resolve().parents[2])  # repository root
WP = ROOT; OUT = ROOT+"/audits"; os.makedirs(OUT, exist_ok=True)
DT_H, GRID = 1/60.0, 10000
COMP=[10,12,13,14]; LOADS=[5,6,7,11,15,16]; MG=1; OPCUR=0.5

d = pd.read_parquet(WP+"/data/interim/02_raw_standardized.parquet",
                    columns=["meter_id","time","i_an","p_a","p_b","p_c"])
d["gm"] = pd.to_datetime(d["time"], utc=True, errors="coerce").dt.floor("min")
# grid: mean per (meter, minute) cell, ingestion rule
cell = d.groupby(["meter_id","gm"])[["i_an","p_a","p_b","p_c"]].mean().reset_index()
cell["op"] = cell["i_an"] > OPCUR

cc = cell[cell.meter_id.isin(COMP) & cell.op]                 # cells of operational components
C  = cc.groupby("gm").meter_id.nunique()                      # C(t) per minute
pot= cc.groupby("gm")[["p_a","p_b","p_c"]].sum(min_count=1)   # partial P_other per phase
mg = cell[cell.meter_id==MG].set_index("gm")[["p_a","p_b","p_c"]]
ld = cell[cell.meter_id.isin(LOADS)].groupby("gm")[["p_a","p_b","p_c"]].sum(min_count=1)

tot=int(C.shape[0])
print("=== VALIDATION (P01 mask i_an>0.5, minute grid) ===")
print(f"  C>=1={tot} | C=1={int((C==1).sum())} C=2={int((C==2).sum())} C=3={int((C==3).sum())} C=4={int((C==4).sum())}  (canonical: 2196/1349/-/-/97)")

def E(df, idx, col): return float(np.nansum(df.reindex(idx)[col].to_numpy()))*DT_H/1000.0
PH=[("A","p_a"),("B","p_b"),("C","p_c")]
supports={"C=1":C==1,"C=2":C==2,"C=3":C==3,"C=4":C==4,"C>=1":C>=1,"C>=2":C>=2,"C>=3":C>=3}
rows=[]
for name,mask in supports.items():
    idx=C.index[mask.values]; n=len(idx); dur=n/60.0
    for ph,col in PH:
        e_mg=E(mg,idx,col); e_pot=E(pot,idx,col); e_ld=E(ld,idx,col); r=e_mg-e_ld-e_pot
        rows.append(dict(suporte=name,fase=ph,n=n,pct_grade=round(100*n/GRID,2),dur_h=round(dur,1),
            E_MG=round(e_mg,2),E_pother=round(e_pot,2),E_loads=round(e_ld,2),resid_kWh=round(r,2),
            pot_media_kWh_h=round(e_pot/dur,4) if dur else None,           # mean power of the aggregate (sign matters)
            resid_kWh_h=round(r/dur,4) if dur else None,
            resid_pct_EMG=round(100*r/abs(e_mg),1) if abs(e_mg)>1e-9 else None))
res=pd.DataFrame(rows); res.to_csv(OUT+"/item3_pother_completeness.csv",index=False)
print("\n=== P_other and RESIDUAL per support x phase (pot_media = kWh/h of the aggregate; sign/stability) ===")
print(res[["suporte","fase","n","dur_h","E_MG","E_pother","pot_media_kWh_h","resid_kWh","resid_pct_EMG"]].to_string(index=False))
print("\nsaved to", OUT+"/item3_pother_completeness.csv")
