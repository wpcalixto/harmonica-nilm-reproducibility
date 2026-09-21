#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 24_export_article_results.py

File type: Pipeline script (stage 24: consolidation of values for the manuscript)

Purpose:
    Consolidation for the manuscript: exports values/decisions ready for the article.
    EVERYTHING is derived from the files of the pipeline (no hard-coded numbers). The
    PE-ES-Optuna model appears as SENSITIVITY (analysis_group), never in the main ranking.

    Data sources: 09_scenario_mapping (Gamma map per scenario), 11_kneedle_selection +
    11_candidate_sets (feature selection), 10_p_other_energy_by_phase (P_other), 17_dg_*
    (balance and uncertainty), table_05 (metrics); table_01 uses "parameter/value".

    Outputs: article/24_article_values.yaml . 24_article_results_summary.md .
             24_methods_values_checklist.md . manifests/24_*.json

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
import yaml

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # repository root (parent of scripts/)
TABDATA = PROJECT_ROOT / "tabdata"
AUDITS  = PROJECT_ROOT / "audits"
METRICS = PROJECT_ROOT / "metrics"
DG      = PROJECT_ROOT / "dg"
MANIFESTS = PROJECT_ROOT / "manifests"
ARTICLE = PROJECT_ROOT / "article"
LOGS    = PROJECT_ROOT / "logs"
for d in (ARTICLE, LOGS):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(LOGS / "24_export_article_results.log", mode="w", encoding="utf-8")])
log = logging.getLogger(__name__)

PRIMARY = ["LSTM", "RCNN_att", "PE_ES", "SPEC", "MoTE_v2", "Ensemble_G4"]
REF_DG = "MoTE_v2"      # reference model of the D_G balance (best main model)
missing = []


def rcsv(p: Path):
    if not p.exists():
        log.warning("  MISSING: %s", p.name); missing.append(p.name); return None
    return pd.read_csv(p)


def rnd(v, n=3):
    try:
        return round(float(v), n)
    except (TypeError, ValueError):
        return None


def main():
    log.info("=== 24_export_article_results.py ===")

    # -- 1. actual methodology (curated) ------------------------------------------------------
    methods = {
        "gap_filling": {"method": "analog/fractal self-similar (own-series fragments, overlap-add)",
                        "decision": "cross-meter imputation INVALIDATED (shared sentinel-floor artifact); "
                                    "preserves amplitude and frequency (held-out proof)", "status": "new (replaces Bulhões/MLP)"},
        "feature_selection": {"method": "RFECV-SVR (linear) + Kneedle elbow",
                              "decision": "per-output selection, union", "status": "kept (original)"},
        "models": {"method": "LSTM, RCNN-att, PE+ES, SPEC, MoTE_v2, Ensemble_G4",
                   "decision": "6 primary models; PE-ES-Optuna* as HPO-intensive sensitivity", "status": "kept"},
        "channel_audit": {"method": "eligibility / redundancy / energy-preservation audit",
                          "decision": "M2 excluded; M8/M9/M17/M18 dead; M10/M12/M13/M14 → P_other (P01)",
                          "status": "new"},
        "p_other": {"method": "P01 = M10+M12+M13+M14 (additive aggregate)",
                    "decision": "downstream closure term, not an individual load", "status": "new"},
        "dg_balance": {"method": "physical balance reconstruction (reconstructed M_G)",
                       "decision": "D_G as indirect balance reference; net consistent with zero within uncertainty",
                       "status": "new"},
    }

    # -- 2. dataset (table_01 parameter/value) -------------------------------------------------
    t1 = rcsv(TABDATA / "table_01_dataset_summary.csv")
    dataset = {}
    if t1 is not None:
        dataset = {str(r["parameter"]): r["value"] for _, r in t1.iterrows()}

    # -- 3. feature selection (11_kneedle_selection + candidate_sets) --------------------------
    # The CSV of script 11 carries the diagnostics (union/kneedle/n95) but records
    # n_adopted="PENDING_ABLATION": the final adoption (48) is decided by the ablation and
    # lives in data/processed/11_candidate_sets.json (authoritative). n_model = 48+12.
    kn = rcsv(AUDITS / "11_kneedle_selection.csv")
    cj = {}
    _cand = PROJECT_ROOT / "data" / "processed" / "11_candidate_sets.json"
    if _cand.exists():
        cj = json.loads(_cand.read_text())
    # global importance (glob_imp of the 177) -> importance share of the adopted features
    gi = rcsv(AUDITS / "11_global_importance.csv")
    adopted_importance_pct = None
    adopted_set = set(cj.get(cj.get("adopted", ""), [])) if cj else set()
    if gi is not None and adopted_set:
        _tot = float(gi["glob_imp"].sum())
        if _tot > 0:
            adopted_importance_pct = rnd(
                100.0 * gi.loc[gi["feature"].isin(adopted_set), "glob_imp"].sum() / _tot, 2)
    features = {}
    if kn is not None:
        k = kn.iloc[0]
        n_dg = int(k["n_dg"])
        n_adopted = int(cj["n_adopted"]) if "n_adopted" in cj else None
        n_universe = len(cj.get("all_177", [])) or None
        features = {"n_union": int(k["n_union"]), "knee_n": int(k["knee_n"]),
                    "knee_pct": rnd(k["knee_pct"], 2), "adopted_n": n_adopted,
                    # TWO distinct concepts, unambiguous names:
                    #  adopted_universe_pct   = fraction of the M_G universe (48/177)
                    #  adopted_importance_pct = sum of the global importance of the 48 / sum of the 177
                    "adopted_universe_pct": rnd(100.0 * n_adopted / n_universe, 2)
                                   if (n_adopted and n_universe) else None,
                    "adopted_importance_pct": adopted_importance_pct,
                    "n_dg": n_dg,
                    "n_model_input": (n_adopted + n_dg) if n_adopted else None,
                    "knee_feature": str(k.get("knee_feature", ""))}

    # -- 4. metrics (table_05: 6 main models + Optuna* tagged) --------------------------------
    t5 = rcsv(TABDATA / "table_05_model_metrics.csv")
    models, models_sensitivity = {}, {}
    best_nae = best_erg = None
    if t5 is not None:
        for _, r in t5.iterrows():
            name = str(r["model"])
            rec = {"NAE_pct": rnd(r["NAE_mean"]), "NAE_std": rnd(r["NAE_std"]),
                   "ERG_pct": rnd(r["ERG_mean"]), "ERG_std": rnd(r["ERG_std"]),
                   "sig_vs_LSTM": str(r.get("sig_vs_LSTM", "—")),
                   "cohen_d_vs_LSTM": rnd(r.get("cohen_d_vs_LSTM"), 3),
                   "effect_vs_LSTM": str(r.get("effect_vs_LSTM", "—"))}
            if str(r.get("analysis_group", "primary")) == "hpo_sensitivity" or name.endswith("*"):
                models_sensitivity[name] = rec
            else:
                models[name] = rec
        if models:
            best_nae = min(models, key=lambda m: models[m]["NAE_pct"])
            best_erg = min(models, key=lambda m: models[m]["ERG_pct"])
        log.info("  %d main + %d sensitivity | best NAE=%s best ERG=%s",
                 len(models), len(models_sensitivity), best_nae, best_erg)

    # -- 5. P_other (10_p_other_energy_by_phase) ----------------------------------------------
    po = rcsv(AUDITS / "10_p_other_energy_by_phase.csv")
    p_other = {}
    if po is not None:
        for _, r in po.iterrows():
            p_other[str(r["phase"]).upper()] = {"E_P01_filled_kWh": rnd(r["E_P01_filled_kWh"]),
                                                "E_components_sum_kWh": rnd(r["E_components_sum_kWh"])}

    # -- 6. D_G balance (17; REF=MoTE_v2, R2) + uncertainty ------------------------------------
    en = rcsv(DG / "17_dg_energy_by_phase.csv")
    iv = rcsv(DG / "17_dg_uncertainty_interval.csv")
    dg_balance = {}
    if en is not None:
        sub = en[(en.model == REF_DG) & (en.scenario == "R2")]
        for _, r in sub.iterrows():
            dg_balance[str(r["phase"])] = {"E_DG_pred_kWh": rnd(r["E_DG_pred_kWh"]),
                                           "E_MG_kWh": rnd(r["E_MG_kWh"]),
                                           "ERG_bal_pct": rnd(r["ERG_bal_pct"])}
    if iv is not None:
        for _, r in iv[iv.model == REF_DG].iterrows():
            ph = str(r["phase"])
            dg_balance.setdefault(ph, {})
            dg_balance[ph].update({"u_total_kWh": rnd(r["u_total_kWh"]),
                                   "CI95_low_kWh": rnd(r["CI95_low_kWh"]),
                                   "CI95_high_kWh": rnd(r["CI95_high_kWh"])})

    # -- 7. Gamma map per scenario (09_scenario_mapping) ---------------------------------------
    sm = rcsv(AUDITS / "09_scenario_mapping.csv")
    gamma = {}
    gamma_r1 = []
    if sm is not None:
        for scen in sorted(sm.scenario.unique()):
            s = sm[sm.scenario == scen]
            loads = s[s.role.astype(str).str.contains("gamma|load|Γ", case=False, na=False)] if "role" in s else s
            gamma[scen] = {"n_rows": int(len(s)),
                           "meters": sorted(s.meter_id.dropna().unique().astype(int).tolist()),
                           "n_gamma": int(s.gamma_id.dropna().nunique())}
        r1 = sm[sm.scenario == "R1"].sort_values(["gamma_id", "phase"])
        gamma_r1 = r1[["gamma_id", "meter_id", "phase", "channel"]].to_dict("records")

    # -- article_values.yaml -------------------------------------------------------------------
    av = {"generated_at": datetime.now(timezone.utc).isoformat(), "script": "24_export_article_results.py",
          "scope_note": "PE-ES-Optuna* = HPO-intensive sensitivity (larger search budget), not a primary model.",
          "dataset": dataset, "feature_selection": features,
          "models_primary": models, "models_sensitivity": models_sensitivity,
          "best_primary_nae_model": best_nae, "best_primary_erg_model": best_erg,
          "p_other": p_other, "dg_balance_ref_model": REF_DG,
          "dg_balance_ref_model_reason": "best primary pointwise NAE and primary-model reference "
                                         "for D_G balance (not necessarily best ERG — see sensitivity)",
          "dg_balance": dg_balance,
          "scenarios": gamma, "methods": methods}
    (ARTICLE / "24_article_values.yaml").write_text(
        yaml.dump(av, allow_unicode=True, sort_keys=False, default_flow_style=False), encoding="utf-8")
    log.info("  article/24_article_values.yaml")

    # -- summary.md ----------------------------------------------------------------------------
    L = [f"# Article Results Summary", f"> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · 24_export_article_results.py",
         "", "> PE-ES-Optuna* is an HPO-intensive **sensitivity** analysis (larger budget), not a primary model.", ""]
    L += ["## 1. Dataset", "", "| Parameter | Value |", "|---|---|"]
    for k, v in dataset.items():
        L.append(f"| {k} | {v} |")
    L += ["", "## 2. Feature selection (RFECV-SVR + Kneedle)", ""]
    for k, v in features.items():
        L.append(f"- **{k}**: {v}")
    L += ["", "## 3. Models — primary (6)", "",
          "| Model | NAE % | ±σ | ERG % | ±σ | vs LSTM | Cohen d | effect |", "|---|---|---|---|---|---|---|---|"]
    for m in PRIMARY:
        if m in models:
            r = models[m]
            L.append(f"| {m} | {r['NAE_pct']} | {r['NAE_std']} | {r['ERG_pct']} | {r['ERG_std']} | "
                     f"{r['sig_vs_LSTM']} | {r['cohen_d_vs_LSTM']} | {r['effect_vs_LSTM']} |")
    if models_sensitivity:
        L += ["", "### Sensitivity (HPO-intensive)", "", "| Model | NAE % | ERG % |", "|---|---|---|"]
        for m, r in models_sensitivity.items():
            L.append(f"| {m} | {r['NAE_pct']} | {r['ERG_pct']} |")
    L += ["", f"**Best primary NAE:** {best_nae} · **Best primary ERG:** {best_erg}", ""]
    L += ["## 4. P_other (P01) energy by phase", "", "| Phase | E_P01_filled kWh | E_components kWh |", "|---|---|---|"]
    for ph, v in p_other.items():
        L.append(f"| {ph} | {v['E_P01_filled_kWh']} | {v['E_components_sum_kWh']} |")
    L += ["", f"## 5. D_G balance ({REF_DG}, R2) + 95% uncertainty", "",
          "| Phase | E_DG_pred kWh | u_total kWh | CI95 low | CI95 high | ERG_bal % |", "|---|---|---|---|---|---|"]
    for ph, v in dg_balance.items():
        L.append(f"| {ph} | {v.get('E_DG_pred_kWh','—')} | {v.get('u_total_kWh','—')} | "
                 f"{v.get('CI95_low_kWh','—')} | {v.get('CI95_high_kWh','—')} | {v.get('ERG_bal_pct','—')} |")
    L += ["", "## 6. Scenarios (Γ map)", "", "| Scenario | n_gamma | meters |", "|---|---|---|"]
    for sc, v in gamma.items():
        L.append(f"| {sc} | {v['n_gamma']} | {v['meters']} |")
    (ARTICLE / "24_article_results_summary.md").write_text("\n".join(L), encoding="utf-8")
    log.info("  article/24_article_results_summary.md")

    # -- checklist.md (traceability) -----------------------------------------------------------
    C = ["# Methods & Values Checklist — traceability",
         f"> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", "",
         "| Value | Source file | Script |", "|---|---|---|",
         "| Dataset summary | tabdata/table_01_dataset_summary.csv | 18_build_tables.py |",
         "| Feature selection (knee) | audits/11_kneedle_selection.csv | 11_feature_selection_rfecv_svr_kneedle.py |",
         "| Model metrics | tabdata/table_05_model_metrics.csv | 14→18 |",
         "| Wilcoxon/PFI | metrics/18_statistical_tests.csv, 18_pfi_results.csv | 18_statistical_analysis.py |",
         "| P_other energy | audits/10_p_other_energy_by_phase.csv | 10_post_audit_and_features.py |",
         "| D_G balance | dg/17_dg_energy_by_phase.csv | 17_dg_balance.py |",
         "| D_G uncertainty | dg/17_dg_uncertainty_interval.csv | 17_dg_balance.py |",
         "| D_G sensitivity S0–S7 | dg/17_dg_sensitivity.csv | 17_dg_balance.py |",
         "| Economic | metrics/21_economic_results.csv | 21_economic_analysis.py |",
         "| Γ map by scenario | audits/09_scenario_mapping.csv | 09_build_model_datasets.py |",
         "| PE-ES-Optuna sensitivity | manifests/13_exp_pe_es_optuna_params.json; "
         "metrics/13_exp_pe_es_optuna_metrics.csv; tabdata/table_05_model_metrics.csv; "
         "metrics/18_statistical_tests_hpo_sensitivity.csv; dg/17_dg_energy_by_phase.csv; "
         "metrics/21_economic_results.csv | 13_exp_pe_es_optuna.py; 14→18 |",
         "", "## Γ map — scenario R1", "", "| Γ | meter | phase | channel |", "|---|---|---|---|"]
    for r in gamma_r1:
        gid = r["gamma_id"]; gid = int(gid) if pd.notna(gid) else "—"
        mid = r["meter_id"]; mid = int(mid) if pd.notna(mid) else "—"
        C.append(f"| G{gid} | M{mid} | {r['phase']} | {r['channel']} |")
    (ARTICLE / "24_methods_values_checklist.md").write_text("\n".join(C), encoding="utf-8")
    log.info("  article/24_methods_values_checklist.md")

    manifest = {"script": "24_export_article_results.py", "ex": "30_export_article_results.py",
                "run_timestamp": datetime.now(timezone.utc).isoformat(),
                "best_primary_nae_model": best_nae, "best_primary_erg_model": best_erg,
                "n_primary": len(models), "n_sensitivity": len(models_sensitivity),
                "missing_inputs": missing,
                "outputs": ["article/24_article_values.yaml", "article/24_article_results_summary.md",
                            "article/24_methods_values_checklist.md"]}
    (MANIFESTS / "24_export_article_results_params.json").write_text(json.dumps(manifest, indent=2, default=str))
    log.info("Done. %d missing.", len(missing))


if __name__ == "__main__":
    main()
