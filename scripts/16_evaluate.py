#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 16_evaluate.py

File type: Evaluation script (pipeline stage 16: predictive and energy metrics from saved predictions)

Purpose:
    Predictive and energy evaluation, consuming saved PREDICTIONS.

    Evaluates the 6 models of Block B from the saved predictions (Watts), WITHOUT loading
    .keras models (independent of the Keras version):
        LSTM         predictions/14_lstm_predictions.parquet         (W=12)
        RCNN_att     predictions/14_rcnn_att_predictions.parquet     (W=12)
        PE_ES        predictions/15_pe_es_predictions.parquet        (W=W*)
        SPEC         predictions/15_spec_predictions.parquet         (W=W*)
        MoTE_v2      predictions/15_mote_v2_predictions.parquet      (W=36)
        Ensemble_G4  predictions/15_ensemble_g4_predictions.parquet  (W=12)

    Metrics per output (mean +/- sd over seeds), per three-phase load and systemic:
        NAE% . MAE . energy_abs_error_contrib% . Delta_reduction(kWh) . ERG% (energy).
        (energy_abs_error_contrib% = |E_true - E_pred| / sum|E_true| per output: contribution of
         the absolute energy error, NOT an "accuracy"; AAE = 100 - ERG is derived in script 18.)
    The evaluated models include PE-ES-Optuna (15_exp_*, HPO sensitivity, W from result.W_star).
    Reconstruction: every predicted series is aligned on the original time axis by the window
    mid-point (offset = W//2) before any metric. W* is read from the manifest of script 13.

    Inputs: data/processed/13_Y_raw_test.npy + manifest 11 (labels); manifest 13 (W*);
            predictions/{14,15}_*_predictions.parquet.
    Outputs: metrics/16_load_metrics_long.csv . 16_metrics_by_output.csv
             metrics/16_energy_by_load.csv . 16_system_erg.csv
             figdata/16_nae_by_model_long.csv . manifests/16_evaluate_params.json

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

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # repository root (parent of scripts/)
DATA, PREDS = PROJECT_ROOT / "data" / "processed", PROJECT_ROOT / "predictions"
METRICS, FIGDATA = PROJECT_ROOT / "metrics", PROJECT_ROOT / "figdata"
MANIFESTS, LOGS = PROJECT_ROOT / "manifests", PROJECT_ROOT / "logs"
for d in (METRICS, FIGDATA, MANIFESTS, LOGS):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(LOGS / "16_evaluate.log", mode="w", encoding="utf-8")])
log = logging.getLogger(__name__)

DELTA_T_HOURS = 1.0 / 60.0
E_MIN_KWH = 1.0   # outputs with |E_true| < 1 kWh -> inactive (excluded from the "active" means)
SEEDS = [42, 123, 456, 789, 1024]


def nae_pct(yt, yp):
    return float(np.sum(np.abs(yp - yt)) / (np.abs(yt).sum() + 1e-9) * 100.0)


def compute_metrics(y_true, y_pred, n_out):
    n = min(len(y_true), len(y_pred))
    yt, yp = y_true[:n], y_pred[:n]
    mae = np.mean(np.abs(yt - yp), axis=0)
    nae = np.array([nae_pct(yt[:, j], yp[:, j]) for j in range(n_out)])
    e_true = yt.sum(axis=0) * DELTA_T_HOURS / 1000.0
    e_pred = yp.sum(axis=0) * DELTA_T_HOURS / 1000.0
    # Percentage contribution of the ABSOLUTE energy error of each output relative to the
    # total system energy: |E_true_j - E_pred_j| / sum|E_true|. It is NOT an "accuracy".
    # (AAE = 1 - E_error/E_tot is derived as 100 - ERG in script 18.)
    energy_abs_err_contrib = np.abs(e_true - e_pred) / (np.abs(e_true).sum() + 1e-9) * 100.0
    delta = e_true - e_pred
    erg_sys = float(np.abs(e_true.sum() - e_pred.sum()) / (np.abs(e_true.sum()) + 1e-9) * 100.0)
    return dict(mae=mae, nae=nae, energy_abs_err_contrib=energy_abs_err_contrib,
                delta=delta, e_true=e_true, e_pred=e_pred, erg_sys=erg_sys)


def load_groups(labels):
    g = {}
    for i, lab in enumerate(labels):
        g.setdefault(str(lab).split("_")[0], []).append(i)
    return g


def main():
    log.info("=== 16_evaluate.py ===")
    y_raw_test = np.load(DATA / "13_Y_raw_test.npy")
    labels = json.loads((MANIFESTS / "13_create_windows_splits_params.json").read_text())["results"]["output_labels"]
    n_out = len(labels)
    groups = load_groups(labels)
    W_star = json.loads((MANIFESTS / "15_train_advanced_params.json").read_text())["methodology"]["W_star"]
    # W of PE-ES-Optuna (HPO sensitivity), from the manifest of that experiment
    exp_mf = MANIFESTS / "15_exp_pe_es_optuna_params.json"
    W_optuna = (json.loads(exp_mf.read_text())["resultado"]["W_star"]) if exp_mf.exists() else None
    log.info("output_dim=%d | W*=%d | W_optuna=%s | loads=%s", n_out, W_star, W_optuna, list(groups))

    MODELS = [("LSTM", "14_lstm", 12), ("RCNN_att", "14_rcnn_att", 12),
              ("PE_ES", "15_pe_es", W_star), ("SPEC", "15_spec", W_star),
              ("MoTE_v2", "15_mote_v2", 36), ("Ensemble_G4", "15_ensemble_g4", 12)]
    # PE-ES-Optuna enters the SAME evaluation (flagged as HPO sensitivity in text/tables)
    if W_optuna is not None:
        MODELS.append(("PE_ES_Optuna", "15_exp_pe_es_optuna", int(W_optuna)))

    long_rows, sys_rows, eload_rows = [], [], []
    e_true_ref = None
    for mname, prefix, W in MODELS:
        f = PREDS / f"{prefix}_predictions.parquet"
        if not f.exists():
            log.warning("  %s: predictions missing (%s); skipped.", mname, f.name)
            continue
        df = pd.read_parquet(f)
        mid = W // 2
        for seed in SEEDS:
            sub = df[df.seed == seed].sort_values("window_idx")
            if sub.empty:
                continue
            yp = sub[labels].to_numpy(dtype=float)
            yt = y_raw_test[mid:mid + len(yp)]
            m = compute_metrics(yt, yp, n_out)
            if e_true_ref is None:
                e_true_ref = m["e_true"]
            sys_rows.append({"model": mname, "seed": seed, "erg_sys_pct": round(m["erg_sys"], 4)})
            for j, lbl in enumerate(labels):
                long_rows.append({"model": mname, "seed": seed, "output": lbl,
                                  "nae_pct": round(float(m["nae"][j]), 4),
                                  "mae_W": round(float(m["mae"][j]), 4),
                                  "energy_abs_error_contrib_pct": round(float(m["energy_abs_err_contrib"][j]), 4),
                                  "delta_reduction_kwh": round(float(m["delta"][j]), 6),
                                  "e_true_kwh": round(float(m["e_true"][j]), 6),
                                  "e_pred_kwh": round(float(m["e_pred"][j]), 6),
                                  "active": bool(abs(m["e_true"][j]) >= E_MIN_KWH)})
            for gname, idxs in groups.items():
                et = float(np.sum(m["e_true"][idxs])); ep = float(np.sum(m["e_pred"][idxs]))
                eload_rows.append({"model": mname, "seed": seed, "load": gname,
                                   "e_true_kwh": round(et, 6), "e_pred_kwh": round(ep, 6),
                                   "erg_load_pct": round(abs(et - ep) / (abs(et) + 1e-9) * 100.0, 4)})

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(METRICS / "16_load_metrics_long.csv", index=False)
    # summary per model x output (mean +/- sd over seeds)
    by_out = (long_df.groupby(["model", "output", "active"])
              .agg(nae_mean=("nae_pct", "mean"), nae_std=("nae_pct", "std"),
                   energy_abs_error_contrib_mean=("energy_abs_error_contrib_pct", "mean")).reset_index().round(4))
    by_out.to_csv(METRICS / "16_metrics_by_output.csv", index=False)
    pd.DataFrame(eload_rows).to_csv(METRICS / "16_energy_by_load.csv", index=False)
    sys_df = (pd.DataFrame(sys_rows).groupby("model")
              .agg(erg_mean=("erg_sys_pct", "mean"), erg_std=("erg_sys_pct", "std")).reset_index().round(4))
    sys_df.to_csv(METRICS / "16_system_erg.csv", index=False)
    # figdata: mean active NAE per model (long format)
    fig = (long_df[long_df.active].groupby("model")["nae_pct"].agg(["mean", "std"]).reset_index().round(4))
    fig.to_csv(FIGDATA / "16_nae_by_model_long.csv", index=False)

    log.info("\n=== Summary (mean active NAE . systemic ERG) ===")
    summ = (long_df[long_df.active].groupby("model")["nae_pct"].mean().round(3)
            .to_frame("NAE_active_%").join(sys_df.set_index("model")["erg_mean"].rename("ERG_%")))
    for mdl, r in summ.iterrows():
        log.info("  %-12s NAE=%.3f%%  ERG=%.3f%%", mdl, r["NAE_active_%"], r["ERG_%"])

    manifest = {"script": "16_evaluate.py", "run_timestamp": datetime.now(timezone.utc).isoformat(),
                "consome": [m[1] for m in MODELS], "W_star": int(W_star), "n_outputs": n_out,
                "delta_t_hours": DELTA_T_HOURS, "e_min_kwh": E_MIN_KWH,
                "outputs": ["16_load_metrics_long.csv", "16_metrics_by_output.csv",
                            "16_energy_by_load.csv", "16_system_erg.csv", "16_nae_by_model_long.csv"]}
    (MANIFESTS / "16_evaluate_params.json").write_text(json.dumps(manifest, indent=2, default=str))
    log.info("Done. Metrics in metrics/16_*.")


if __name__ == "__main__":
    main()
