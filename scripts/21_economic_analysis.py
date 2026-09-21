#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 21_economic_analysis.py

File type: Pipeline script (stage 21: economic analysis)

Purpose:
    Economic analysis with the corrected balance.

      Op1   : energy per load (from 16_load_metrics_long)
      Op2-4 : C_real = E_tot*c_Gr ; C_hat = E_hat_tot*c_Gr ; C_error = E_error*c_Gr ;
              Delta_C = |C_real - C_hat|/C_real ; C_yearly = C_error*(T_yearly/T_eval)
      Op5   : split observed / predicted / computed_by_balance / inferred_P_other / M_G
      Op6   : propagate the D_G uncertainty into the cost (from 17_dg_uncertainty_interval)

    Data sources:
      metrics/16_load_metrics_long.csv (e_true_kwh, e_pred_kwh)
      dg/17_dg_energy_by_phase.csv (E_load_obs_kWh, E_load_pred_kWh, E_DG_*_kWh)
      dg/17_dg_uncertainty_interval.csv (u_total_kWh per phase)
      T_eval = n_test/60 (from manifest 11)
      Outputs in metrics/ (analysis CSVs); figdata/tabdata are populated downstream.

    Inputs:  metrics/16_load_metrics_long.csv . dg/17_dg_energy_by_phase.csv .
             dg/17_dg_uncertainty_interval.csv . manifests/11_*_params.json
    Outputs: metrics/21_economic_results.csv . 21_economic_sensitivity.csv .
             21_economic_sensitivity_long.csv . manifests/21_economic_analysis_params.json

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
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # repository root (parent of scripts/)
METRICS = PROJECT_ROOT / "metrics"
DG      = PROJECT_ROOT / "dg"
MANIFESTS = PROJECT_ROOT / "manifests"
LOGS    = PROJECT_ROOT / "logs"
for d in (METRICS, MANIFESTS, LOGS):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(LOGS / "21_economic_analysis.log", mode="w", encoding="utf-8")])
log = logging.getLogger(__name__)

# -- economic constants -----------------------------------------------------------------
C_GR     = 0.20      # USD/kWh: reference cost (generator/tariff); parameter of the article
T_YEARLY = 8760.0    # h/year
EPS      = 1e-9


def main():
    log.info("=== 21_economic_analysis.py ===")
    n_test = json.loads((MANIFESTS / "13_create_windows_splits_params.json").read_text()
                        )["results"]["n_test_ts"]
    T_EVAL_H = n_test / 60.0
    RATIO = T_YEARLY / T_EVAL_H
    log.info("c_Gr=%.2f USD/kWh | T_eval=%.4f h (n_test=%d) | T_yearly=%.0f h | ratio=%.3f",
             C_GR, T_EVAL_H, n_test, T_YEARLY, RATIO)

    lm = pd.read_csv(METRICS / "16_load_metrics_long.csv")
    en = pd.read_csv(DG / "17_dg_energy_by_phase.csv")
    iv = pd.read_csv(DG / "17_dg_uncertainty_interval.csv")

    # -- Op2-4: costs per (model, seed) --------------------------------------------------------
    log.info("\n===== Op2-4: C_real, C_hat, C_error, Delta_C, C_yearly =====")
    rows = []
    for (model, seed), g in lm.groupby(["model", "seed"]):
        E_true = g.e_true_kwh.sum(); E_pred = g.e_pred_kwh.sum()
        E_err  = (g.e_true_kwh - g.e_pred_kwh).abs().sum()
        C_real = E_true * C_GR; C_hat = E_pred * C_GR; C_err = E_err * C_GR
        rows.append({"model": model, "seed": seed,
                     "E_true_kWh": round(E_true, 4), "E_pred_kWh": round(E_pred, 4),
                     "E_error_kWh": round(E_err, 4), "C_real_USD": round(C_real, 4),
                     "C_hat_USD": round(C_hat, 4), "C_error_USD": round(C_err, 4),
                     "delta_C_pct": round(abs(C_real - C_hat) / (abs(C_real) + EPS) * 100.0, 4),
                     "C_yearly_USD": round(C_err * RATIO, 2)})
    econ = pd.DataFrame(rows)
    econ.to_csv(METRICS / "21_economic_results.csv", index=False)
    summ = (econ.groupby("model").agg(C_real_mean=("C_real_USD", "mean"),
                                      C_error_mean=("C_error_USD", "mean"),
                                      delta_C_mean=("delta_C_pct", "mean"),
                                      C_yearly_mean=("C_yearly_USD", "mean"))
            .reset_index().sort_values("C_error_mean"))
    for _, r in summ.iterrows():
        log.info("  %-12s C_real=%.3f C_error=%.4f Delta_C=%.3f%% C_yearly=%.2f USD/year",
                 r["model"], r.C_real_mean, r.C_error_mean, r.delta_C_mean, r.C_yearly_mean)

    # -- Op5: split by result type (R1, sum of phases A/B/C) -----------------------------------
    log.info("\n===== Op5: split by result type =====")
    r1 = en[(en.scenario == "R1") & (en.phase.isin(["A", "B", "C"]))]
    c  = lambda e: round(e * C_GR, 4)
    cy = lambda e: round(e * C_GR * RATIO, 2)
    # preserves the sensitivity flag coming from script 15 (analysis_group/is_primary)
    group_cols = ["model"] + [c_ for c_ in ("analysis_group", "is_primary") if c_ in r1.columns]
    srows = []
    for keys, m in r1.groupby(group_cols):
        meta = dict(zip(group_cols, keys)) if isinstance(keys, tuple) else {"model": keys}
        types = {"observado": m.E_load_obs_kWh.sum(),
                 "predito": m.E_load_pred_kWh.sum(),
                 "calculado_por_balanco": m.E_DG_obs_kWh.abs().sum(),
                 "inferido_P_other": m.E_other_kWh.sum(),
                 "medidor_geral": m.E_MG_kWh.sum()}
        for tp, e in types.items():
            srows.append({**meta, "tipo": tp, "E_kWh": round(e, 4),
                          "C_USD": c(e), "C_yearly_USD": cy(e)})
    sens = pd.DataFrame(srows)
    for _, r in sens[sens.model == summ.iloc[0]["model"]].iterrows():
        log.info("  [%s] %-24s E=%.3f kWh C=%.4f USD C_yearly=%.2f USD/year",
                 summ.iloc[0]["model"], r.tipo, r.E_kWh, r.C_USD, r.C_yearly_USD)

    # -- Op6: propagation of the D_G uncertainty (u_total_kWh per phase) ------------------------
    log.info("\n===== Op6: D_G uncertainty in the cost =====")
    u_ph = iv.drop_duplicates("phase")[["phase", "u_total_kWh"]].copy()  # model-independent
    u_ph["u_C_USD"] = (u_ph.u_total_kWh * C_GR).round(4)
    u_ph["u_C_yearly_USD"] = (u_ph.u_total_kWh * C_GR * RATIO).round(2)
    u_E_total = float(np.sqrt((u_ph.u_total_kWh ** 2).sum()))
    u_C_total = u_E_total * C_GR
    u_Cy_total = u_C_total * RATIO
    for _, r in u_ph.iterrows():
        log.info("  phase %s: u(E_DG)=%.4f kWh u(C)=%.4f USD u(C_yearly)=%.2f USD/year",
                 r.phase, r.u_total_kWh, r.u_C_USD, r.u_C_yearly_USD)
    log.info("  TOTAL: u(E_DG)=%.4f kWh u(C)=%.4f USD u(C_yearly)=%.2f USD/year",
             u_E_total, u_C_total, u_Cy_total)

    urows = [{"model": "UNCERTAINTY_DG", "tipo": f"u_DG_phase_{r.phase}",
              "E_kWh": round(r.u_total_kWh, 4), "C_USD": r.u_C_USD,
              "C_yearly_USD": r.u_C_yearly_USD} for _, r in u_ph.iterrows()]
    urows.append({"model": "UNCERTAINTY_DG", "tipo": "u_DG_total",
                  "E_kWh": round(u_E_total, 4), "C_USD": round(u_C_total, 4),
                  "C_yearly_USD": round(u_Cy_total, 2)})
    sens_all = pd.concat([sens, pd.DataFrame(urows)], ignore_index=True)
    sens_all.to_csv(METRICS / "21_economic_sensitivity.csv", index=False)

    # long format (for a later figure); kept in metrics/ for now
    fig = []
    for _, r in econ.iterrows():
        for mtr in ("C_real_USD", "C_hat_USD", "C_error_USD", "C_yearly_USD", "delta_C_pct"):
            fig.append({"model": r.model, "seed": r.seed, "tipo": "prediction_error",
                        "metric": mtr, "value": r[mtr]})
    for _, r in sens_all.iterrows():
        for mtr in ("C_yearly_USD", "E_kWh"):
            fig.append({"model": r.model, "seed": None, "tipo": r.tipo,
                        "metric": mtr, "value": r[mtr]})
    pd.DataFrame(fig).to_csv(METRICS / "21_economic_sensitivity_long.csv", index=False)
    log.info("Saved: metrics/21_economic_{results,sensitivity,sensitivity_long}.csv")

    manifest = {"script": "21_economic_analysis.py", "ex": "27_economic_analysis.py",
                "run_timestamp": pd.Timestamp.utcnow().isoformat(),
                "parameters": {"C_GR_USD_per_kWh": C_GR, "T_eval_h": round(T_EVAL_H, 4),
                               "T_yearly_h": T_YEARLY, "ratio_yearly": round(RATIO, 3),
                               "n_test": n_test},
                "model_sources": {
                    "prediction_costs": "models present in metrics/16_load_metrics_long.csv",
                    "energy_type_costs": "models present in dg/17_dg_energy_by_phase.csv",
                    "hpo_sensitivity": "PE_ES_Optuna appears only if included in upstream tables (14/15); marked via analysis_group/is_primary"},
                "results": {"model_summary": summ.to_dict("records"),
                            "u_C_DG_total_USD": round(u_C_total, 4),
                            "u_C_DG_yearly_total_USD": round(u_Cy_total, 2),
                            "u_E_DG_total_kWh": round(u_E_total, 4)}}
    (MANIFESTS / "21_economic_analysis_params.json").write_text(json.dumps(manifest, indent=2, default=str))
    log.info("Done. Manifest saved.")


if __name__ == "__main__":
    main()
