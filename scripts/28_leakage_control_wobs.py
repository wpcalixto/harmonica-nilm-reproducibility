#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 28_leakage_control_wobs.py

File type: Diagnostic script (leakage control; does not alter the pipeline; stage 28)

Purpose:
    LATERAL DIAGNOSTIC (does not alter the pipeline).

    Strict control of the influence of reconstructed inputs: recomputes NAE/ERG only on the test
    windows in which ALL 48 M_G input features are originally observed over the W instants AND
    the 18 targets are observed at the evaluated instant (W_obs). The 12 constant-zero auxiliary
    columns are excluded from the check (they are never filled).

    Sources (read-only): 08_fill_confidence mask, predictions/reference targets of the current
    run, list of the 48 features (windows/reference_48/manifest.json). Outputs with prefix 28_.

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
from __future__ import annotations
import argparse, json, glob
from pathlib import Path
import numpy as np, pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent  # repository root (parent of scripts/)
FULL = ROOT               # predictions and processed data of the current run
LABELS = ['G1_A','G1_B','G1_C','G2_A','G2_B','G2_C','G3_A','G3_B','G3_C',
          'G4_A','G4_B','G4_C','G5_A','G5_B','G5_C','G6_A','G6_B','G6_C']
LOADS = {"G1":5,"G2":6,"G3":7,"G4":11,"G5":15,"G6":16}
MODELS = [("LSTM","14_lstm"),("RCNN_att","14_rcnn_att"),("Ensemble_G4","15_ensemble_g4"),
          ("PE_ES","15_pe_es"),("SPEC","15_spec"),("MoTE_v2","15_mote_v2")]
TEST0 = 9000


def _nae(a, b):
    return 100.0 * np.abs(a - b).sum() / (np.abs(a).sum() + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--full", default=str(FULL))
    args = ap.parse_args()
    root, full = Path(args.root), Path(args.full)

    mg48 = json.load(open(root / "data/processed/windows/reference_48/manifest.json"))["feature_names"]
    mask = pd.read_parquet(root / "data/processed/08_fill_confidence.parquet")
    tt = pd.to_datetime(mask["time"], utc=True); t0 = tt.min().floor("min")
    mask["gi"] = ((tt - t0).dt.total_seconds() // 60).astype(int)
    mg = mask[mask.meter_id == 1].set_index("gi")
    mg48 = [c for c in mg48 if c in mg.columns]          # 48 (excludes the dg_unit auxiliary columns)
    MG = np.ones(10000, bool)
    for c in mg48:
        MG &= (mg[c] == 0).reindex(range(10000), fill_value=False).values
    Lp = {}
    for g, m in LOADS.items():
        sub = mask[mask.meter_id == m].set_index("gi")
        for ph in "abc":
            Lp[f"{g}_{ph.upper()}"] = (sub[f"p_{ph}"] == 0).reindex(range(10000), fill_value=False).values
    TGT = np.ones(10000, bool)
    for l in LABELS:
        TGT &= Lp[l]

    y_raw = np.load(root / "data/processed/13_Y_raw_test.npy")
    rows = []
    for mn, pf in MODELS:
        f = glob.glob(str(full / f"predictions/{pf}_predictions.parquet"))
        if not f:
            continue
        d = pd.read_parquet(f[0])
        nwin = d[d.seed == d.seed.iloc[0]].shape[0]
        W = 1000 - nwin + 1; mid = W // 2
        wobs = np.array([MG[TEST0 + w:TEST0 + w + W].all() and TGT[TEST0 + mid + w] for w in range(nwin)])
        naA, naO, erA, erO = [], [], [], []
        for s in sorted(d.seed.unique()):
            sub = d[d.seed == s].sort_values("window_idx")
            yp = sub[LABELS].to_numpy(float); yt = y_raw[mid:mid + len(yp)]
            naA.append(np.mean([_nae(yt[:, j], yp[:, j]) for j in range(18)]))
            erA.append(abs(yt.sum() - yp.sum()) / (abs(yt.sum()) + 1e-9) * 100)
            if wobs.sum() > 5:
                naO.append(np.mean([_nae(yt[wobs, j], yp[wobs, j]) for j in range(18)]))
                erO.append(abs(yt[wobs].sum() - yp[wobs].sum()) / (abs(yt[wobs].sum()) + 1e-9) * 100)
        NAa, ERa = np.mean(naA), np.mean(erA)
        NAo = np.mean(naO) if naO else np.nan
        ERo = np.mean(erO) if erO else np.nan
        rows.append({"model": mn, "W": W, "NAE_all": round(NAa, 3),
                     "NAE_wobs": round(NAo, 3) if naO else None,
                     "dNAE": round(NAo - NAa, 3) if naO else None,
                     "ERG_all": round(ERa, 3), "ERG_wobs": round(ERo, 3) if erO else None,
                     "n_obs": int(wobs.sum()), "n_all": nwin,
                     "evaluable": bool(wobs.sum() > 5)})
    out = pd.DataFrame(rows)
    for p in (root / "audits" / "28_wobs_control.csv", root / "tabdata" / "28_wobs_control.csv"):
        out.to_csv(p, index=False)
    print(out.to_string(index=False))
    print("[28] W_obs (48 features + 18 targets) -> audits/tabdata/28_wobs_control.csv")


if __name__ == "__main__":
    main()
