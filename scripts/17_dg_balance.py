#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 17_dg_balance.py

File type: Pipeline script (stage 17: load-consistency balance, sensitivity scenarios and sensitivity budget)

Purpose:
    Load-consistency balance (historically labelled D_G) + sensitivity + sensitivity budget.

    Design: 18 three-phase load outputs, P_other = P01 = M10+M12+M13+M14.

    Section A  BALANCE
        Reconstructed M_G = sum of observed loads(phase) + P_other(phase)   (closure, DG=0).
        For each model x scenario (R1/R2) x phase (A/B/C):
            R1: load side WITHOUT P_other -> D_G = P_L - M_G   (level ~ -P_other)
            R2: load side WITH P_other    -> D_G = (P_L+P_other) - M_G
        D_G_obs uses the observed loads; D_G_pred uses the predictions (mean of the 5 seeds).
        Energy by rectangular integration (dt = 1/60 h).
        ERG_bal = |E_DG_pred - E_DG_obs| / |E_MG| x 100 (energy closure error).

    Section B  SENSITIVITY: scenarios S0-S7, at the energy level:
        S0 baseline . S1-S3 losses 1/2/5% (M_G x (1-l)) . S4-S7 P_other without M12/M13/M10/M14.
        S1-S3 = sensitivity to LOSSES (appear in R2: D_G += mg x loss).
        S4-S7 = STRUCTURAL sensitivity of P01 (appear in R1: D_G changes by E_component).
        Component removal via the energy fraction of 10_p_other_components (sum to 100%/phase;
        M12 is negative -> removing it INCREASES P_other).

    Section C  SENSITIVITY BUDGET of R_cl, PER MODEL, directly in energy (three terms):
          u_met_L,m,phi = 0.5% x |E_L^obs| on the support of the model   (assumed perturbation of the loads)
          u_sync,m,phi  = sqrt((d_-30^2 + d_+30^2)/2)                    (actual shift of the observations, +/-30 s)
          u_seed,m,phi  = sd_s(E_R,m,s,phi)                              (directly in energy)
        Three-phase total: sums the phases WITHIN each seed before the sd, preserving the
        covariance between phases. Losses and P_other stay out (zero derivative in R_cl).
        Predictions FROZEN; +/-30 s by linear interpolation between minutes.
        The structural impact of P01 (S4-S7 vs S0) is reported SEPARATELY (scope decision,
        not random fluctuation; it does NOT enter the quadrature).

    Data sources:
        13_Y_raw_test.npy (18 loads, Watts) + timestamps of the split; P_other from 09_R2
        (P_other_A/B/C), aligned by timestamp; predictions 14_/15_*_predictions.parquet
        (per seed); the 18 output labels from the manifest (DYNAMIC); W* from the manifest
        of script 13; imputed fraction of P01 from 10_p_other_energy_by_phase.

    Outputs (dg/):
        17_dg_balance_timeseries.parquet . 17_dg_energy_by_phase.csv
        17_dg_sensitivity.csv . 17_dg_structural_impact.csv
        17_dg_uncertainty.csv . 17_dg_uncertainty_interval.csv
        manifests/17_dg_balance_params.json

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
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR   = Path(__file__).resolve().parent
# The optional environment variable HARMONIC_PROJECT_ROOT overrides the repository root
# (e.g. to evaluate the artifacts of another complete run).
PROJECT_ROOT = Path(os.environ["HARMONIC_PROJECT_ROOT"]).resolve() \
    if os.environ.get("HARMONIC_PROJECT_ROOT") else SCRIPT_DIR.parent
DATA   = PROJECT_ROOT / "data" / "processed"
PREDS  = PROJECT_ROOT / "predictions"
METRICS = PROJECT_ROOT / "metrics"
AUDITS = PROJECT_ROOT / "audits"
FIGDATA = PROJECT_ROOT / "figdata"
MANIFESTS = PROJECT_ROOT / "manifests"
LOGS   = PROJECT_ROOT / "logs"
DG     = PROJECT_ROOT / "dg"
for d in (DG, LOGS):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(LOGS / "17_dg_balance.log", mode="w", encoding="utf-8")])
log = logging.getLogger(__name__)

# -- constants ----------------------------------------------------------------------------
DELTA_T_HOURS  = 1.0 / 60.0
DT_KWH         = DELTA_T_HOURS / 1000.0     # W x step -> kWh
SEEDS          = [42, 123, 456, 789, 1024]
PHASES         = ["A", "B", "C"]
PH_IDX         = {"A": 0, "B": 1, "C": 2}
EPS            = 1e-9

# Section B scenarios
LOSS_SCEN      = {"S1": 0.01, "S2": 0.02, "S3": 0.05}      # losses
STRUCT_SCEN    = {"S4": 12, "S5": 13, "S6": 10, "S7": 14}  # component removed from P01

# Section C budget
METER_ACCURACY = 0.005   # Class 0.5S: +/-0.5%
FILL_ACCURACY  = 0.20    # +/-20% on the imputed value
REF_MODEL      = "MoTE_v2"   # reference for mean powers and inter-seed dispersion


# ══════════════════════════════════════════════════════════════════════════════
#  Loading / alignment
# ══════════════════════════════════════════════════════════════════════════════
def load_context() -> dict:
    labels = json.loads((MANIFESTS / "13_create_windows_splits_params.json").read_text()
                        )["results"]["output_labels"]
    y_obs = np.load(DATA / "13_Y_raw_test.npy").astype("float64")
    n_test = y_obs.shape[0]

    tl = pd.read_csv(FIGDATA / "13_split_timeline.csv")
    test_ts = pd.to_datetime(tl.loc[tl.split == "test", "timestamp"].to_numpy(), utc=True)
    assert len(test_ts) == n_test, f"timeline test={len(test_ts)} ≠ Y_raw_test={n_test}"

    r2 = pd.read_parquet(DATA / "09_R2.parquet")
    r2.index = pd.to_datetime(r2.index, utc=True)
    po = r2.reindex(test_ts)[["P_other_A", "P_other_B", "P_other_C"]]
    assert int(po.isna().sum().sum()) == 0, "P_other with NaN after reindex: misalignment"
    p_other = po.to_numpy(dtype="float64")
    mg_real = r2.reindex(test_ts)[["M_G_A", "M_G_B", "M_G_C"]].to_numpy(dtype="float64")

    phase_outputs = {ph: [i for i, l in enumerate(labels) if str(l).endswith(f"_{ph}")]
                     for ph in PHASES}
    W_star = int(json.loads((MANIFESTS / "15_train_advanced_params.json").read_text()
                            )["methodology"]["W_star"])

    # energy fractions per component of P01 (sum to 100%/phase) -> S4-S7
    comp = pd.read_csv(AUDITS / "10_p_other_components.csv")
    comp_frac = {ph: {} for ph in PHASES}
    for _, r in comp.iterrows():
        comp_frac[r["phase"].upper()][int(r["component_meter"])] = float(r["pct_of_components"]) / 100.0

    # imputed (fill) fraction of P01 per phase -> u_imp
    ener = pd.read_csv(AUDITS / "10_p_other_energy_by_phase.csv")
    fill_frac = {}
    for _, r in ener.iterrows():
        ph = r["phase"].upper()
        fill_frac[ph] = 1.0 - float(r["E_components_sum_kWh"]) / (float(r["E_P01_filled_kWh"]) + EPS)

    log.info("n_test=%d n_out=%d W*=%d | loads/phase=%s",
             n_test, len(labels), W_star, {k: len(v) for k, v in phase_outputs.items()})
    log.info("fill_frac(P01)=%s", {k: round(v, 3) for k, v in fill_frac.items()})
    return dict(labels=labels, n_out=len(labels), y_obs=y_obs, n_test=n_test,
                p_other=p_other, mg_real=mg_real, phase_outputs=phase_outputs,
                W_star=W_star, comp_frac=comp_frac, fill_frac=fill_frac)


def analysis_group(model: str) -> str:
    """Optuna is HPO-intensive sensitivity; the other models form the main group."""
    return "hpo_sensitivity" if model == "PE_ES_Optuna" else "primary"


def is_primary_model(model: str) -> bool:
    return analysis_group(model) == "primary"


# EXPLICIT map of canonical prediction files. Replaces the implicit prefix
# resolution, which resolved LSTM/RCNN_att to the old numbering ("12_*")
# and SILENTLY excluded them from the balance (the actual files are "14_*").
PREDICTION_FILES = {
    "LSTM":        "14_lstm_predictions.parquet",
    "RCNN_att":    "14_rcnn_att_predictions.parquet",
    "PE_ES":       "15_pe_es_predictions.parquet",
    "SPEC":        "15_spec_predictions.parquet",
    "MoTE_v2":     "15_mote_v2_predictions.parquet",
    "Ensemble_G4": "15_ensemble_g4_predictions.parquet",
}
# Sensitivity (outside the main ranking); enters only if the manifest exists.
PREDICTION_FILES_SENSITIVITY = {"PE_ES_Optuna": "15_exp_pe_es_optuna_predictions.parquet"}


def model_table(W_star: int) -> list:
    """(name, prediction_file, W). The first 6 are mandatory."""
    base = [("LSTM", PREDICTION_FILES["LSTM"], 12),
            ("RCNN_att", PREDICTION_FILES["RCNN_att"], 12),
            ("PE_ES", PREDICTION_FILES["PE_ES"], W_star),
            ("SPEC", PREDICTION_FILES["SPEC"], W_star),
            ("MoTE_v2", PREDICTION_FILES["MoTE_v2"], 36),
            ("Ensemble_G4", PREDICTION_FILES["Ensemble_G4"], 12)]
    exp_mf = MANIFESTS / "15_exp_pe_es_optuna_params.json"
    if exp_mf.exists():
        Wo = int(json.loads(exp_mf.read_text())["resultado"]["W_star"])
        base.append(("PE_ES_Optuna", PREDICTION_FILES_SENSITIVITY["PE_ES_Optuna"], Wo))
    return base


def seed_preds(filename: str, labels: list) -> dict:
    """Predictions PER SEED (frozen). {seed: array (n_win, n_out)}."""
    f = PREDS / filename
    if not f.is_file():
        raise FileNotFoundError(f"Prediction file missing: {f}")
    df = pd.read_parquet(f)
    return {int(sd): g.sort_values("window_idx")[labels].to_numpy(dtype="float64")
            for sd, g in df.groupby("seed")}


def mean_pred(filename: str, labels: list):
    """Reads the canonical parquet. HARD error if missing: never omit a model silently."""
    f = PREDS / filename
    if not f.is_file():
        raise FileNotFoundError(f"Prediction file missing: {f}")
    df = pd.read_parquet(f)
    return df.groupby("window_idx")[labels].mean().sort_index().to_numpy(dtype="float64")


def build_series(ctx) -> dict:
    """For each model: per phase, (obs_sum, pred_sum, po_ph, mid, n) already aligned."""
    series = {}
    missing = [(m, f) for m, f, _ in model_table(ctx["W_star"])
               if m in PREDICTION_FILES and not (PREDS / f).is_file()]
    if missing:
        raise FileNotFoundError(
            "Prediction files missing for main models:\n" +
            "\n".join(f"  {m}: {PREDS / f}" for m, f in missing))
    for mname, filename, W in model_table(ctx["W_star"]):
        pred = mean_pred(filename, ctx["labels"])
        mid = W // 2
        n = min(len(pred), len(ctx["y_obs"]) - mid)
        obs = ctx["y_obs"][mid:mid + n]
        po  = ctx["p_other"][mid:mid + n]
        prd = pred[:n]
        per_ph = {}
        for ph in PHASES:
            idx = ctx["phase_outputs"][ph]
            per_ph[ph] = dict(obs_sum=obs[:, idx].sum(axis=1),
                              pred_sum=prd[:, idx].sum(axis=1),
                              po_ph=po[:, PH_IDX[ph]],
                              mg_real=ctx["mg_real"][mid:mid + n, PH_IDX[ph]])
        series[mname] = dict(W=W, mid=mid, n=n, per_ph=per_ph)
    faltando = set(PREDICTION_FILES) - set(series)
    if faltando:
        raise RuntimeError(f"Incomplete balance; missing models: {sorted(faltando)}")
    log.info("build_series: %d models | %s", len(series),
             {m: (series[m]["W"], series[m]["n"]) for m in series})
    return series


# ══════════════════════════════════════════════════════════════════════════════
#  Core of the balance (R1/R2 energies for one phase)
# ══════════════════════════════════════════════════════════════════════════════
def balance_energy(obs_sum, pred_sum, po_ph, loss_frac=0.0):
    mg = (obs_sum + po_ph) * (1.0 - loss_frac)
    out = {}
    for scen in ("R1", "R2"):
        if scen == "R1":
            d_obs, d_pred = obs_sum - mg, pred_sum - mg
        else:
            d_obs, d_pred = (obs_sum + po_ph) - mg, (pred_sum + po_ph) - mg
        E_MG = float(mg.sum()) * DT_KWH
        E_DG_obs  = float(d_obs.sum())  * DT_KWH
        E_DG_pred = float(d_pred.sum()) * DT_KWH
        out[scen] = dict(E_MG_kWh=E_MG, E_DG_obs_kWh=E_DG_obs, E_DG_pred_kWh=E_DG_pred,
                         ERG_bal_pct=abs(E_DG_pred - E_DG_obs) / (abs(E_MG) + EPS) * 100.0,
                         E_other_kWh=float(po_ph.sum()) * DT_KWH,
                         E_load_obs_kWh=float(obs_sum.sum()) * DT_KWH,
                         E_load_pred_kWh=float(pred_sum.sum()) * DT_KWH,
                         d_obs=d_obs, d_pred=d_pred, mg=mg)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Section A: BALANCE
# ══════════════════════════════════════════════════════════════════════════════
def section_A(ctx, series) -> dict:
    log.info("\n===== Section A: D_G BALANCE (R1/R2 x phase x model) =====")
    ts_rows, en_rows = [], []
    for mname, S in series.items():
        agp = analysis_group(mname); isp = is_primary_model(mname)
        log.info("  %-12s W=%2d mid=%d n=%d [%s]", mname, S["W"], S["mid"], S["n"], agp)
        agg = {sc: {k: 0.0 for k in ("E_DG_obs_kWh", "E_DG_pred_kWh", "E_MG_kWh")} for sc in ("R1", "R2")}
        for ph in PHASES:
            d = S["per_ph"][ph]
            res = balance_energy(d["obs_sum"], d["pred_sum"], d["po_ph"], 0.0)
            E_MG_real = float(d["mg_real"].sum()) * DT_KWH
            for sc in ("R1", "R2"):
                r = res[sc]
                en_rows.append({"model": mname, "analysis_group": agp, "is_primary": isp,"scenario": sc, "phase": ph,
                                **{k: round(v, 4) for k, v in r.items()
                                   if k not in ("d_obs", "d_pred", "mg")},
                                "E_MG_real_kWh": round(E_MG_real, 4)})
                for k in agg[sc]:
                    agg[sc][k] += r[k]
                for t in range(S["n"]):
                    ts_rows.append({"model": mname, "analysis_group": agp, "is_primary": isp,"scenario": sc, "phase": ph, "t": t,
                                    "obs_sum_W": round(float(d["obs_sum"][t]), 3),
                                    "pred_sum_W": round(float(d["pred_sum"][t]), 3),
                                    "p_other_W": round(float(d["po_ph"][t]), 3),
                                    "M_G_W": round(float(r["mg"][t]), 3),
                                    "D_G_obs_W": round(float(r["d_obs"][t]), 3),
                                    "D_G_pred_W": round(float(r["d_pred"][t]), 3)})
        for sc in ("R1", "R2"):
            erg = abs(agg[sc]["E_DG_pred_kWh"] - agg[sc]["E_DG_obs_kWh"]) / (abs(agg[sc]["E_MG_kWh"]) + EPS) * 100.0
            en_rows.append({"model": mname, "analysis_group": agp, "is_primary": isp,"scenario": sc, "phase": "ABC",
                            "E_DG_obs_kWh": round(agg[sc]["E_DG_obs_kWh"], 4),
                            "E_DG_pred_kWh": round(agg[sc]["E_DG_pred_kWh"], 4),
                            "E_MG_kWh": round(agg[sc]["E_MG_kWh"], 4),
                            "ERG_bal_pct": round(erg, 4)})
            log.info("    %s %-2s ERG_bal=%.3f%%  E_DG_pred=%.3f  E_MG=%.3f kWh",
                     mname, sc, erg, agg[sc]["E_DG_pred_kWh"], agg[sc]["E_MG_kWh"])
    en_df = pd.DataFrame(en_rows)
    en_df.to_csv(DG / "17_dg_energy_by_phase.csv", index=False)
    en_df[en_df["is_primary"]].to_csv(DG / "17_dg_energy_by_phase_primary_models.csv", index=False)
    en_df[~en_df["is_primary"]].to_csv(DG / "17_dg_energy_by_phase_sensitivity.csv", index=False)
    ts_df = pd.DataFrame(ts_rows)
    ts_df.to_parquet(DG / "17_dg_balance_timeseries.parquet", index=False)
    ts_df[ts_df["is_primary"]].to_parquet(DG / "17_dg_balance_timeseries_primary_models.parquet", index=False)
    ts_df[~ts_df["is_primary"]].to_parquet(DG / "17_dg_balance_timeseries_sensitivity.parquet", index=False)
    log.info("Section A saved: 17_dg_energy_by_phase(.csv|_primary_models|_sensitivity) + timeseries(.parquet x3)")
    return {"energy_by_phase_rows": len(en_rows)}


# ══════════════════════════════════════════════════════════════════════════════
#  Section B: SENSITIVITY (S0-S7)
# ══════════════════════════════════════════════════════════════════════════════
def section_B(ctx, series) -> dict:
    log.info("\n===== Section B: SENSITIVITY (S0-S7) =====")
    rows, struct_rows = [], []
    for mname, S in series.items():
        agp = analysis_group(mname); isp = is_primary_model(mname)
        for ph in PHASES:
            d = S["per_ph"][ph]
            obs, prd, po = d["obs_sum"], d["pred_sum"], d["po_ph"]

            def emit(scen, kind, po_use, loss):
                res = balance_energy(obs, prd, po_use, loss)
                rows.append({"model": mname, "analysis_group": agp, "is_primary": isp,"phase": ph,
                             "scenario": scen, "kind": kind, "loss_frac": loss,
                             "E_MG_kWh": round(res["R2"]["E_MG_kWh"], 4),
                             "E_other_kWh": round(res["R2"]["E_other_kWh"], 4),
                             "E_DG_pred_R1_kWh": round(res["R1"]["E_DG_pred_kWh"], 4),
                             "E_DG_pred_R2_kWh": round(res["R2"]["E_DG_pred_kWh"], 4),
                             "ERG_bal_R2_pct": round(res["R2"]["ERG_bal_pct"], 4)})
                return res

            base = emit("S0", "baseline", po, 0.0)
            for scen, l in LOSS_SCEN.items():
                emit(scen, "loss", po, l)
            for scen, meter in STRUCT_SCEN.items():
                frac = ctx["comp_frac"][ph].get(meter, 0.0)
                po_mod = po * (1.0 - frac)         # removes the component (energy fraction)
                res = emit(scen, "structural", po_mod, 0.0)
                # structural impact = Delta E_DG_R1 vs S0 (= energy of the removed component)
                struct_rows.append({"model": mname, "analysis_group": agp, "is_primary": isp,"phase": ph, "scenario": scen,
                                    "removed_meter": meter, "comp_frac": round(frac, 4),
                                    "E_DG_R1_S0_kWh": round(base["R1"]["E_DG_pred_kWh"], 4),
                                    "E_DG_R1_kWh": round(res["R1"]["E_DG_pred_kWh"], 4),
                                    "delta_E_DG_kWh": round(res["R1"]["E_DG_pred_kWh"]
                                                            - base["R1"]["E_DG_pred_kWh"], 4),
                                    "delta_pct_of_E_MG": round(
                                        (res["R1"]["E_DG_pred_kWh"] - base["R1"]["E_DG_pred_kWh"])
                                        / (abs(base["R2"]["E_MG_kWh"]) + EPS) * 100.0, 4)})
    sens = pd.DataFrame(rows)
    sens.to_csv(DG / "17_dg_sensitivity.csv", index=False)
    sens[sens["is_primary"]].to_csv(DG / "17_dg_sensitivity_primary_models.csv", index=False)
    sens[~sens["is_primary"]].to_csv(DG / "17_dg_sensitivity_hpo_sensitivity.csv", index=False)
    struct_df = pd.DataFrame(struct_rows)
    struct_df.to_csv(DG / "17_dg_structural_impact.csv", index=False)
    struct_df[struct_df["is_primary"]].to_csv(DG / "17_dg_structural_impact_primary_models.csv", index=False)
    struct_df[~struct_df["is_primary"]].to_csv(DG / "17_dg_structural_impact_hpo_sensitivity.csv", index=False)

    # summary (REF model, ABC): losses and structural
    ref = sens[sens.model == REF_MODEL]
    for ph in PHASES:
        sub = ref[ref.phase == ph].set_index("scenario")
        log.info("  [%s %s] S0 ERG=%.3f%% | losses S1-S3 E_DG_R2=%.2f/%.2f/%.2f | "
                 "structural S4-S7 E_DG_R1=%.2f/%.2f/%.2f/%.2f", REF_MODEL, ph,
                 sub.loc["S0", "ERG_bal_R2_pct"],
                 sub.loc["S1", "E_DG_pred_R2_kWh"], sub.loc["S2", "E_DG_pred_R2_kWh"], sub.loc["S3", "E_DG_pred_R2_kWh"],
                 sub.loc["S4", "E_DG_pred_R1_kWh"], sub.loc["S5", "E_DG_pred_R1_kWh"],
                 sub.loc["S6", "E_DG_pred_R1_kWh"], sub.loc["S7", "E_DG_pred_R1_kWh"])
    log.info("Section B saved: 17_dg_sensitivity.csv . 17_dg_structural_impact.csv")
    return {"sensitivity_rows": len(sens), "structural_rows": len(struct_rows)}


# ══════════════════════════════════════════════════════════════════════════════
#  Section C: SENSITIVITY BUDGET
# ══════════════════════════════════════════════════════════════════════════════
def section_C(ctx, series) -> dict:
    """Sensitivity budget of R_cl, PER MODEL, directly in energy.

      u_met_L,m,phi = 0.5% * |E_L^obs| on the support of the model   (assumed perturbation)
      u_sync,m,phi  = sqrt((d_-30^2 + d_+30^2)/2)                    (actual shift of the observations)
      u_seed,m,phi  = sd_s(E_R,m,s,phi)                              (directly in energy)
    Three-phase total: sums the phases WITHIN each seed before the sd, preserving the
    covariance between phases. Losses and P_other stay out (zero derivative in R_cl).
    Predictions FROZEN; +/-30 s by linear interpolation between minutes.
    """
    log.info("\n===== Section C: SENSITIVITY BUDGET (per model) =====")
    comp_rows, u_by = [], {}
    for mname, filename, W in model_table(ctx["W_star"]):
        if mname not in series:
            continue
        S = series[mname]; mid, n = S["mid"], S["n"]
        preds = seed_preds(filename, ctx["labels"])
        seeds = sorted(preds)
        acc = {}                      # per phase: (E_met, E0[], dp[], dm[])
        for ph in PHASES:
            idx = ctx["phase_outputs"][ph]
            obs_full = ctx["y_obs"][:, idx].sum(axis=1)
            k0, k1 = mid + 1, mid + n - 1
            m = k1 - k0
            o0 = obs_full[k0:k1]
            op = 0.5 * (obs_full[k0:k1] + obs_full[k0 + 1:k1 + 1])
            om = 0.5 * (obs_full[k0 - 1:k1 - 1] + obs_full[k0:k1])
            E0, dp, dm = [], [], []
            for sd in seeds:
                pr = preds[sd][:n, idx].sum(axis=1)[1:1 + m]
                e0 = float((pr - o0).sum()) * DT_KWH
                E0.append(e0)
                dp.append(float((pr - op).sum()) * DT_KWH - e0)
                dm.append(float((pr - om).sum()) * DT_KWH - e0)
            acc[ph] = (METER_ACCURACY * abs(float(o0.sum()) * DT_KWH),
                       np.array(E0), np.array(dp), np.array(dm))
        # per phase + three-phase aggregate (sum within the seed)
        blocos = [(ph, *acc[ph]) for ph in PHASES]
        blocos.append(("ABC",
                       sum(acc[ph][0] for ph in PHASES),
                       sum(acc[ph][1] for ph in PHASES),
                       sum(acc[ph][2] for ph in PHASES),
                       sum(acc[ph][3] for ph in PHASES)))
        for ph, u_met, E0, dp, dm in blocos:
            u_seed = float(np.std(E0, ddof=1))
            u_sync = float(np.mean(np.sqrt((dm ** 2 + dp ** 2) / 2.0)))
            u_tot = float(np.sqrt(u_met ** 2 + u_seed ** 2 + u_sync ** 2))
            u_by[(mname, ph)] = (u_tot, float(np.mean(E0)))
            for nm, v, ut, ds in [
                ("u_met_load", u_met, "assumed", "0.5% de |E_L^obs| — perturbacao ASSUMIDA (comum aos modelos, no suporte de cada um)"),
                ("u_seed", u_seed, "computational", f"sd entre {len(seeds)} sementes de E_R — ESPECIFICO do modelo"),
                ("u_sync", u_sync, "measurement", "sqrt((d_-30^2+d_+30^2)/2); obs deslocada ±30 s (interpolacao linear)"),
            ]:
                comp_rows.append({"model": mname, "phase": ph, "component": nm, "u_kWh": round(v, 6),
                                  "in_quadrature": True, "u_type": ut, "description": ds})
            comp_rows.append({"model": mname, "phase": ph, "component": "u_total", "u_kWh": round(u_tot, 6),
                              "in_quadrature": None, "u_type": "combined",
                              "description": "sqrt(u_met^2+u_seed^2+u_sync^2) — covariancia nula como HIPOTESE"})
        log.info("  %-12s u_total/phase = %s | ABC = %.4f kWh", mname,
                 " ".join(f"{u_by[(mname,ph)][0]:.4f}" for ph in PHASES), u_by[(mname, "ABC")][0])
    T_HOURS = series[REF_MODEL]["n"] / 60.0
    u_total_kwh = {ph: u_by[(REF_MODEL, ph)][0] for ph in PHASES}

    pd.DataFrame(comp_rows).to_csv(DG / "17_dg_uncertainty.csv", index=False)

    # 95% intervals per model x phase (E_DG_pred R2 of S0 +/- 2*u_total)
    iv_rows = []
    for mname, Sm in series.items():
        for ph in PHASES:
            d = Sm["per_ph"][ph]
            e_dg = balance_energy(d["obs_sum"], d["pred_sum"], d["po_ph"], 0.0)["R2"]["E_DG_pred_kWh"]
            u = u_by[(mname, ph)][0]
            iv_rows.append({"model": mname, "analysis_group": analysis_group(mname),
                            "is_primary": is_primary_model(mname), "phase": ph,
                            "E_DG_pred_R2_kWh": round(e_dg, 4), "u_total_kWh": round(u, 4),
                            "CI95_low_kWh": round(e_dg - 2 * u, 4),
                            "CI95_high_kWh": round(e_dg + 2 * u, 4)})
    iv_df = pd.DataFrame(iv_rows)
    iv_df.to_csv(DG / "17_dg_uncertainty_interval.csv", index=False)
    iv_df[iv_df["is_primary"]].to_csv(DG / "17_dg_uncertainty_interval_primary_models.csv", index=False)
    iv_df[~iv_df["is_primary"]].to_csv(DG / "17_dg_uncertainty_interval_hpo_sensitivity.csv", index=False)
    log.info("Section C saved: 17_dg_uncertainty.csv . 17_dg_uncertainty_interval(.csv|_primary_models|_hpo_sensitivity)")
    return {"u_total_kWh_by_phase": {k: round(v, 4) for k, v in u_total_kwh.items()},
            "ref_model": REF_MODEL, "T_hours": round(T_HOURS, 3)}


def main():
    log.info("=== 17_dg_balance.py ===")
    ctx = load_context()
    series = build_series(ctx)
    results = {"A": section_A(ctx, series),
               "B": section_B(ctx, series),
               "C": section_C(ctx, series)}
    manifest = {
        "script": "17_dg_balance.py",
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "merge_de": ["23_reconstruct_dg_balance.py", "24_run_dg_sensitivity_analysis.py",
                     "25_quantify_uncertainty.py"],
        "metodologia": {
            "M_G": "reconstructed = sum loads_obs(phase) + P_other(phase) (closure, DG=0)",
            "predicao": "mean of the 5 seeds",
            "ERG_bal": "|E_DG_pred − E_DG_obs| / |E_MG| × 100",
            "cenarios_B": {"S0": "baseline", "S1-S3": "perdas 1/2/5%",
                           "S4-S7": "P_other sem M12/M13/M10/M14 (estrutural)"},
            "u_total": "sqrt(u2_met_load+sum u2_Pi+u2_sync): 3 terms with non-zero derivative in R_cl; u_loss and u_imp EXCLUDED (dR_cl/dP_loss=0, dR_cl/dP_other=0); structural scenarios reported as a diagnostic separate from the aggregate",
            "ref_model": REF_MODEL,
            "model_groups": {
                "primary": "6 modelos principais do Bloco B",
                "hpo_sensitivity": "PE_ES_Optuna; sensitivity analysis with intensive hyperparameter search",
            },
            "PE_ES_Optuna": "included in the D_G balance only as flagged sensitivity, not in the main ranking",
        },
        "results": {**results,
                    "models_evaluated": [
                        {"model": m, "prefix": p, "W": int(w),
                         "analysis_group": analysis_group(m), "is_primary": is_primary_model(m)}
                        for m, p, w in model_table(ctx["W_star"])]},
    }
    (MANIFESTS / "17_dg_balance_params.json").write_text(json.dumps(manifest, indent=2, default=str))
    log.info("Done. Manifest saved.")


if __name__ == "__main__":
    main()
