#!/usr/bin/env python3
"""
Title: audit_item9_clustered_bootstrap.py

File type: Auxiliary statistical analysis script (read-only re-analysis of pipeline artifacts)

Purpose:
    Layer 3 / item 9 (Tier A): inference CLUSTERED BY LOAD (bootstrap).

    Delta_Gi = (1/3) sum_phi (NAE_competitor,i,phi - NAE_MoTE,i,phi);  Delta>0 favours MoTE.
    - COMMON temporal support (965 windows) + ORIGINALLY OBSERVED target (joins controls 2 and 4).
    - Paired differences per seed and output; mean of the 3 phases within each load.
    - Resamples the 6 LOADS with replacement (phase triplet kept together) AND the 5 seeds (paired).
    - 20,000 replicates, fixed seed; 95% percentile CI; shows the 6 per-load effects.
    - Secondary control: exact sign test on the 6 per-load effects.

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
import numpy as np, pandas as pd, os
ROOT=str(Path(__file__).resolve().parents[2])  # repository root
WP,FR=ROOT,ROOT; OUT=ROOT+"/audits"
SEEDS=[42,123,456,789,1024]
LOADS_G=["G1","G2","G3","G4","G5","G6"]; PH=["A","B","C"]
LOAD_METER={"G1":5,"G2":6,"G3":7,"G4":11,"G5":15,"G6":16}; PHASE_COL={"A":"p_a","B":"p_b","C":"p_c"}
PRED={"MoTE_v2":"15_mote_v2_predictions","LSTM":"14_lstm_predictions","Ensemble_G4":"15_ensemble_g4_predictions"}
def nae(yt,yp): return float(np.sum(np.abs(yp-yt))/(np.abs(yt).sum()+1e-9)*100.0)

info={m:pd.read_parquet(FR+f"/predictions/{pf}.parquet") for m,pf in PRED.items()}
offs={m:(1001-info[m].window_idx.nunique())//2 for m in PRED}
# support common to the 6 models of the study (uses the global offsets 6/15/18)
lo,hi=18,982; cen=np.arange(lo,hi+1)
conf=pd.read_parquet(WP+"/figdata/08_fill_confidence.parquet")
yr=np.load(WP+"/data/processed/13_Y_raw_test.npy")
LAB=[f"{g}_{p}" for g in LOADS_G for p in PH]
obs={lab:(conf[conf.meter_id==LOAD_METER[lab.split('_')[0]]].sort_values("time")[PHASE_COL[lab.split('_')[1]]].to_numpy()[-1000:]==0) for lab in LAB}

# NAE[model][seed][lab] on the common support + observed target
NAEm={m:{} for m in PRED}
for m in PRED:
    dd=info[m]; off=offs[m]; ks=cen-off
    for sd in SEEDS:
        g=dd[dd.seed==sd].sort_values("window_idx")
        for j,lab in enumerate(LAB):
            om=obs[lab][cen]; yt=yr[cen,j][om]; yp=g[lab].to_numpy(float)[ks][om]
            NAEm[m].setdefault(sd,{})[lab]=nae(yt,yp)

def delta_matrix(comp):
    """[5 seeds x 6 loads] of Delta = mean over phases (NAE_comp - NAE_MoTE)."""
    M=np.zeros((5,6))
    for si,sd in enumerate(SEEDS):
        for li,g in enumerate(LOADS_G):
            d=np.mean([NAEm[comp][sd][f"{g}_{p}"]-NAEm["MoTE_v2"][sd][f"{g}_{p}"] for p in PH])
            M[si,li]=d
    return M

rng=np.random.default_rng(20260719); B=20000
def boot(M):
    per_load=M.mean(0)                              # per-load effect (mean over seeds)
    pt=per_load.mean()                              # point estimate
    reps=np.empty(B)
    for b in range(B):
        li=rng.integers(0,6,6); si=rng.integers(0,5,5)
        reps[b]=M[np.ix_(si,li)].mean()
    ci=np.percentile(reps,[2.5,97.5])
    return pt,ci,per_load

print(f"common support [{lo},{hi}] = {len(cen)} windows; observed target; {B} replicates\n")
rows=[]
for comp in ["LSTM","Ensemble_G4"]:
    M=delta_matrix(comp); pt,ci,pl=boot(M)
    npos=int((pl>0).sum())
    # exact two-sided sign test on the 6 loads
    from math import comb
    k=max(npos,6-npos); psign=2*sum(comb(6,i) for i in range(k,7))/2**6
    print(f"=== MoTE_v2 vs {comp} (Delta>0 favours MoTE) ===")
    print(f"  mean Delta = {pt:+.3f} pp | 95% CI = [{ci[0]:+.3f}, {ci[1]:+.3f}] pp")
    print(f"  per-load effects (pp): "+", ".join(f"{g}={v:+.2f}" for g,v in zip(LOADS_G,pl)))
    print(f"  loads favouring MoTE: {npos}/6 | sign test: p_exact={psign:.3f}\n")
    rows.append(dict(contraste=f"MoTE_v2 vs {comp}",delta_pp=round(pt,3),
                     ci_low=round(ci[0],3),ci_high=round(ci[1],3),
                     cargas_pro_MoTE=f"{npos}/6",p_sinais=round(psign,3),
                     **{g:round(v,2) for g,v in zip(LOADS_G,pl)}))
# optional: LSTM vs Ensemble
Ml=delta_matrix("LSTM"); Me=delta_matrix("Ensemble_G4"); M=Ml-Me   # (LSTM-MoTE)-(Ens-MoTE)=LSTM-Ens
pt,ci,pl=boot(M)
print(f"=== LSTM vs Ensemble-G4 (Delta>0 favours Ensemble) ===")
print(f"  mean Delta = {pt:+.3f} pp | 95% CI = [{ci[0]:+.3f}, {ci[1]:+.3f}] pp")
rows.append(dict(contraste="LSTM vs Ensemble_G4",delta_pp=round(pt,3),ci_low=round(ci[0],3),ci_high=round(ci[1],3)))
pd.DataFrame(rows).to_csv(OUT+"/item9_clustered_bootstrap.csv",index=False)
print("\nsaved to",OUT+"/item9_clustered_bootstrap.csv")
