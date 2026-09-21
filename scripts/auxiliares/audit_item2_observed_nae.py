#!/usr/bin/env python3
"""
Title: audit_item2_observed_nae.py

File type: Auxiliary analysis script (read-only re-analysis of pipeline artifacts)

Purpose:
    Layer 3 / item 2 (Tier A): active NAE restricted to ORIGINALLY OBSERVED targets, 6 models.

    Re-analysis of existing artifacts (no retraining). Compares the current active NAE with
    the NAE computed only on the windows whose CENTRAL target of each output was originally
    observed (confidence class 0 in 08_fill_confidence). Read-only; writes a summary CSV to
    audits/. Does not modify the pipeline.

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
WP, FR = ROOT, ROOT
OUT = ROOT+"/audits"; os.makedirs(OUT, exist_ok=True)
DT_H, E_MIN = 1/60.0, 1.0
SEEDS = [42,123,456,789,1024]
LABELS = [f"{g}_{p}" for g in ["G1","G2","G3","G4","G5","G6"] for p in "ABC"]
LOAD_METER = {"G1":5,"G2":6,"G3":7,"G4":11,"G5":15,"G6":16}   # Gamma1..Gamma6
PHASE_COL = {"A":"p_a","B":"p_b","C":"p_c"}
PRED = {"LSTM":"14_lstm_predictions","RCNN_att":"14_rcnn_att_predictions",
        "Ensemble_G4":"15_ensemble_g4_predictions","MoTE_v2":"15_mote_v2_predictions",
        "PE_ES":"15_pe_es_predictions","SPEC":"15_spec_predictions"}

def nae(yt,yp): return float(np.sum(np.abs(yp-yt))/(np.abs(yt).sum()+1e-9)*100.0)

# --- observed mask: conf==0, per (meter, phase), last 1000 instants (test) ---
conf = pd.read_parquet(WP+"/figdata/08_fill_confidence.parquet")
obs_mask = {}                                     # (label) -> bool[1000] in test-local index space
for lab in LABELS:
    g,p = lab.split("_"); m = LOAD_METER[g]; col = PHASE_COL[p]
    s = conf[conf.meter_id==m].sort_values("time")[col].to_numpy()
    obs_mask[lab] = (s[-1000:] == 0)              # True = originally observed

yr = np.load(WP+"/data/processed/13_Y_raw_test.npy")   # (1000,18) raw target per test-local instant

rows = []
for model, pf in PRED.items():
    dd = pd.read_parquet(FR+f"/predictions/{pf}.parquet")
    n = dd.window_idx.nunique(); off = (1001-n)//2      # W=1001-n ; offset=W//2
    W = 1001-n
    per_out_full, per_out_obs, cov = {}, {}, {}
    for j,lab in enumerate(LABELS):
        yt = yr[off:off+n, j]                            # central target per window
        omask = obs_mask[lab][off:off+n]                 # observed at the centre of each window
        e_true = yt.sum()*DT_H/1000.0                    # kWh (defines "active")
        naes_full, naes_obs = [], []
        for sd in SEEDS:
            yp = dd[dd.seed==sd].sort_values("window_idx")[lab].to_numpy(float)
            naes_full.append(nae(yt,yp))
            if omask.sum()>=30: naes_obs.append(nae(yt[omask],yp[omask]))
        per_out_full[lab] = (np.mean(naes_full), abs(e_true)>=E_MIN)
        per_out_obs[lab]  = (np.mean(naes_obs) if naes_obs else np.nan, abs(e_true)>=E_MIN)
        cov[lab] = omask.mean()
    act = [l for l in LABELS if per_out_full[l][1]]
    nae_full = np.mean([per_out_full[l][0] for l in act])
    obs_vals = [per_out_obs[l][0] for l in act if np.isfinite(per_out_obs[l][0])]
    nae_obs  = np.mean(obs_vals) if obs_vals else np.nan
    cov_med  = np.median([cov[l] for l in act])
    n_out_obs= sum(1 for l in act if np.isfinite(per_out_obs[l][0]))
    rows.append(dict(model=model, W=W, n_win=n, n_active=len(act),
                     NAE_active_full=round(nae_full,3),
                     NAE_active_observed=round(nae_obs,3) if np.isfinite(nae_obs) else None,
                     delta=round(nae_obs-nae_full,3) if np.isfinite(nae_obs) else None,
                     cov_obs_median=round(cov_med,3), n_outputs_avaliaveis=f"{n_out_obs}/{len(act)}"))
df = pd.DataFrame(rows)
df.to_csv(OUT+"/item2_observed_nae.csv", index=False)
print(df.to_string(index=False))
print("\nsaved to", OUT+"/item2_observed_nae.csv")
