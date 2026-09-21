#!/usr/bin/env python3
"""
Title: audit_item_null_predictor.py

File type: Auxiliary analysis script (null-predictor control; read-only re-analysis of pipeline artifacts)

Purpose:
    Layer 3 / NULL predictor (baseline without M_G): control of the first hypothesis.

    y_null_j(t) = median{ y_j(t) : t in TRAIN, originally observed target },
    constant per output, estimated ONLY on the training split, ONLY on observed targets,
    per output, WITHOUT any M_G feature.

    Evaluation on the TEMPORAL SUPPORT COMMON to the 6 models and on OBSERVED TARGETS,
    same active NAE as item 4. Descriptive comparison with the 6 models. Read-only.

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
import numpy as np, pandas as pd, joblib
ROOT=str(Path(__file__).resolve().parents[2])  # repository root
WP,FR=ROOT,ROOT; OUT=ROOT+"/audits"
DT_H,E_MIN=1/60.0,1.0
LABELS=[f"{g}_{p}" for g in ["G1","G2","G3","G4","G5","G6"] for p in "ABC"]
LOAD_METER={"G1":5,"G2":6,"G3":7,"G4":11,"G5":15,"G6":16}; PHASE_COL={"A":"p_a","B":"p_b","C":"p_c"}
PRED={"LSTM":"14_lstm_predictions","RCNN_att":"14_rcnn_att_predictions","Ensemble_G4":"15_ensemble_g4_predictions",
      "MoTE_v2":"15_mote_v2_predictions","PE_ES":"15_pe_es_predictions","SPEC":"15_spec_predictions"}
def nae(yt,yp): return float(np.sum(np.abs(yp-yt))/(np.abs(yt).sum()+1e-9)*100.0)
def unwindow(a): return np.vstack([a[:,0,:], a[-1,1:,:]])

# ---- common support (same logic as item 4) ----
ns={}
for m,pf in {"LSTM":"14_lstm_predictions","MoTE_v2":"15_mote_v2_predictions","PE_ES":"15_pe_es_predictions"}.items():
    n=pd.read_parquet(FR+f"/predictions/{pf}.parquet").window_idx.nunique(); ns[m]=n
offs={m:(1001-n)//2 for m,n in ns.items()}
# uses all 6 for the actual intersection
alln={}
for m,pf in PRED.items():
    alln[m]=pd.read_parquet(FR+f"/predictions/{pf}.parquet").window_idx.nunique()
alloff={m:(1001-n)//2 for m,n in alln.items()}
lo=max(alloff.values()); hi=min(alloff[m]+alln[m]-1 for m in alln)
cen=np.arange(lo,hi+1)
print(f"common support: test instants [{lo},{hi}] = {len(cen)} windows")

# ---- observed mask per output (conf==0), test-local ----
conf=pd.read_parquet(WP+"/figdata/08_fill_confidence.parquet"); obs_te={}; obs_tr={}
for lab in LABELS:
    g,p=lab.split("_"); s=conf[conf.meter_id==LOAD_METER[g]].sort_values("time")[PHASE_COL[p]].to_numpy()
    obs_te[lab]=(s[-1000:]==0)      # test = last 1000
    obs_tr[lab]=(s[:8000]==0)       # train = first 8000

# ---- raw targets ----
yr=np.load(WP+"/data/processed/13_Y_raw_test.npy")                       # (1000,18) physical
scY=joblib.load(WP+"/data/processed/13_scaler_Y.pkl")
Ytr=scY.inverse_transform(unwindow(np.load(WP+"/data/processed/13_Y_train.npy")))  # (8000,18) physical
print("sanity: range yr[:,0] %.1f..%.1f | Ytr[:,0] %.1f..%.1f"%(yr[:,0].min(),yr[:,0].max(),Ytr[:,0].min(),Ytr[:,0].max()))

# ---- null constant per output (median of observed training targets) ----
null_c=np.array([np.median(Ytr[obs_tr[lab][:len(Ytr)],j]) for j,lab in enumerate(LABELS)])

# ---- evaluation on common support + observed ----
rows=[]; na_full=[]; na_obs=[]; Ep=Et=0.0; Ep_o=Et_o=0.0
for j,lab in enumerate(LABELS):
    yt=yr[cen,j]; om=obs_te[lab][cen]; e_true=yt.sum()*DT_H/1000.0
    active=abs(e_true)>=E_MIN
    yp=np.full_like(yt, null_c[j])
    nf=nae(yt,yp); no=nae(yt[om],yp[om]) if om.sum()>=30 else np.nan
    rows.append(dict(output=lab, null_const=round(null_c[j],2), active=bool(active),
                     NAE_common=round(nf,3), NAE_common_observed=round(no,3) if no==no else np.nan))
    if active:
        na_full.append(nf); na_obs.append(no)
        Ep+=yp.sum()*DT_H/1000.0; Et+=yt.sum()*DT_H/1000.0
        Ep_o+=yp[om].sum()*DT_H/1000.0; Et_o+=yt[om].sum()*DT_H/1000.0
perout=pd.DataFrame(rows)
perout.to_csv(OUT+"/item_null_predictor_peroutput.csv",index=False)
erg=abs(Ep-Et)/(abs(Et)+1e-9)*100; erg_o=abs(Ep_o-Et_o)/(abs(Et_o)+1e-9)*100

print("\n=== NULL PREDICTOR, per output (common support) ===")
print(perout.to_string(index=False))
print(f"\nmean active NAE (full)     = {np.mean(na_full):.3f}%  ({len(na_full)}/18 active)")
print(f"mean active NAE (observed) = {np.nanmean(na_obs):.3f}%")
print(f"ERG_net (secondary, full/obs) = {erg:.3f}% / {erg_o:.3f}%")

# ---- comparison with the 6 models (item 4) ----
m6=pd.read_csv(OUT+"/item4_common_support.csv")
comp=pd.concat([m6, pd.DataFrame([dict(model="NULL (mediana treino)",
       NAE_common=round(np.mean(na_full),3), NAE_common_observed=round(np.nanmean(na_obs),3))])],ignore_index=True).sort_values("NAE_common")
comp.to_csv(OUT+"/item_null_predictor_compare.csv",index=False)
print("\n=== Active NAE on the COMMON SUPPORT: models vs null predictor ===")
print(comp.to_string(index=False))
