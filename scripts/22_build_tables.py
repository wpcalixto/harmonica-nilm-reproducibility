#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 22_build_tables.py

File type: Pipeline script (stage 22: consolidation of results into tabdata/ and generation of the LaTeX tables)

Purpose:
    Tables of the article.

    Part 1: consolidates the results of the pipeline (01-21) into tabdata/.
    Part 2: generates the .tex tables (booktabs) reading ONLY from tabdata/.

    Methodology rule: final tables read only tabdata/ (never audits/). Here Part 1 reads
    audits/metrics/dg/manifests and writes tabdata/; Part 2 reads only tabdata/ and writes
    tables/*.tex. "Report when there is no data": every table whose input is missing is
    logged and skipped (without aborting the script).

    Data sources: 13_window_metadata; 11_kneedle_selection / 11_selected_features_by_output;
    03_channel_eligibility; 09_scenario_mapping; 06_gap_profile_summary; 06_outlier_locations /
    08_fill_traceability / 10_energy_preservation_by_stage; 16_load_metrics_long /
    16_energy_by_load; 17_dg_energy_by_phase / 17_dg_sensitivity / 17_dg_structural_impact /
    17_dg_uncertainty / 17_dg_uncertainty_interval; 18_statistical_tests / 18_pfi_results;
    21_economic_results / 21_economic_sensitivity.

    Outputs: tabdata/table_*.csv + table_manifest.csv ; tables/table_*.tex ; manifest/log.

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
AUDITS   = PROJECT_ROOT / "audits"
METRICS  = PROJECT_ROOT / "metrics"
DG       = PROJECT_ROOT / "dg"
MANIFESTS = PROJECT_ROOT / "manifests"
TABDATA  = PROJECT_ROOT / "tabdata"
TABLES   = PROJECT_ROOT / "tables"
LOGS     = PROJECT_ROOT / "logs"
for d in (TABDATA, TABLES, LOGS):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(LOGS / "22_build_tables.log", mode="w", encoding="utf-8")])
log = logging.getLogger(__name__)

SEEDS = [42, 123, 456, 789, 1024]
generated = []


def rd(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        log.warning("  missing input: %s; the dependent table will be skipped", path.name)
        return None
    return pd.read_csv(path)


SENS_NOTE = ("PE-ES-Optuna* is reported only as an HPO-intensive sensitivity analysis "
             "(larger optimization budget); it is not part of the main model set.")


def mark_sensitivity(df: pd.DataFrame, model_col: str = "model") -> pd.DataFrame:
    """Flags PE_ES_Optuna as sensitivity: column analysis_group + display PE_ES_Optuna* +
    sorts the main models before the sensitivity ones. Idempotent."""
    if model_col not in df.columns:
        return df
    df = df.copy()
    if "analysis_group" not in df.columns:
        df["analysis_group"] = np.where(df[model_col] == "PE_ES_Optuna", "hpo_sensitivity", "primary")
    if "is_primary" not in df.columns:
        df["is_primary"] = df["analysis_group"] != "hpo_sensitivity"
    df[model_col] = df[model_col].mask(df[model_col] == "PE_ES_Optuna", "PE_ES_Optuna*")
    df["_g"] = (df["analysis_group"] == "hpo_sensitivity").astype(int)
    df = df.sort_values("_g", kind="stable").drop(columns="_g").reset_index(drop=True)
    return df


def save(df: pd.DataFrame, name: str, mark: bool = False):
    if mark:
        df = mark_sensitivity(df)
    df.to_csv(TABDATA / name, index=False)
    generated.append(name)
    log.info("  tabdata/%s (%dx%d)%s", name, df.shape[0], df.shape[1],
             "  [sensitivity tagged]" if mark else "")


def erg_by_model_seed(lm: pd.DataFrame) -> pd.DataFrame:
    """Systemic ERG per (model, seed) = |sum e_pred - sum e_true|/|sum e_true|*100."""
    g = lm.groupby(["model", "seed"]).apply(
        lambda x: abs(x.e_pred_kwh.sum() - x.e_true_kwh.sum()) / (abs(x.e_true_kwh.sum()) + 1e-9) * 100,
        include_groups=False).reset_index(name="erg_sys")
    return g


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: consolidation -> tabdata/
# ══════════════════════════════════════════════════════════════════════════════
def build_tabdata():
    log.info("=== PART 1: consolidation -> tabdata/ ===")

    # table_01: dataset summary
    try:
        wm = rd(AUDITS / "13_window_metadata.csv")
        gp = rd(AUDITS / "06_gap_profile_summary.csv")
        if wm is not None:
            w = wm.iloc[0]
            miss_mean = round(gp.pct_lacuna_media.mean(), 2) if gp is not None else None
            miss_max = round(gp.pct_lacuna_media.max(), 2) if gp is not None else None
            t01 = pd.DataFrame([
                ("Total meters", 16),
                ("Model outputs (loads×3φ)", int(w.n_outputs)),
                ("Features MG", int(w.n_mg_features)),
                ("Features DG (zeros)", int(w.n_dg_features)),
                ("Features total (X)", int(w.n_features)),
                ("Window W", int(w.W)),
                ("Train windows", int(w.n_train_windows)),
                ("Val windows", int(w.n_val_windows)),
                ("Test windows", int(w.n_test_windows)),
                ("Total timesteps", int(w.n_total_ts)),
                ("Period (days)", round(int(w.n_total_ts) / 1440, 2)),
                ("Resolution Δt (min)", 1),
                ("Mean missingness (%)", miss_mean),
                ("Max missingness (%)", miss_max),
                ("Seeds", ",".join(map(str, SEEDS))),
            ], columns=["parameter", "value"])
            save(t01, "table_01_dataset_summary.csv")
    except Exception as e:
        log.warning("  table_01 failed: %s", e)

    # table_02: channel eligibility (dedicated source of the channel audit: 03_channel_eligibility.csv)
    try:
        el = rd(AUDITS / "03_channel_eligibility.csv")
        if el is not None:
            keep = [c for c in ["section", "key", "n_channels", "pct"] if c in el.columns]
            t02 = el[keep].copy()
            if "pct" in t02:
                t02["pct"] = t02["pct"].round(2)
            save(t02, "table_02_channel_eligibility.csv")
            log.info("  table_02 <- 03_channel_eligibility.csv (%d classes+flags; explicit is_proxy)",
                     len(t02))
        else:
            log.warning("  table_02: 03_channel_eligibility.csv missing (run script 03).")
    except Exception as e:
        log.warning("  table_02 failed: %s", e)

    # table_03: preprocessing audit (outliers + fill + energy preservation)
    try:
        rows = []
        outl = rd(AUDITS / "06_outlier_locations.csv")
        if outl is not None:
            for ph, n in outl.groupby("fase").size().items():
                rows.append({"stage": "outliers", "phase": ph, "metric": "n_points_removed", "value": int(n)})
        ft = rd(AUDITS / "08_fill_traceability.csv")
        if ft is not None:
            agg = ft[["n_analog", "n_interp", "n_edge", "n_unfilled", "n_gap"]].sum()
            for k, v in agg.items():
                rows.append({"stage": "fill", "phase": "all", "metric": k, "value": int(v)})
        pres = rd(AUDITS / "10_energy_preservation_by_stage.csv")
        if pres is not None:
            for ph, g in pres.groupby("phase"):
                rows.append({"stage": "energy_preservation", "phase": ph,
                             "metric": "median_rel_filled_vs_raw_pct",
                             "value": round(g.rel_filled_vs_raw_pct.median(), 3)})
        if rows:
            save(pd.DataFrame(rows), "table_03_preprocessing_audit.csv")
    except Exception as e:
        log.warning("  table_03 failed: %s", e)

    # table_04: selected features (+ kneedle)
    try:
        sf = rd(AUDITS / "11_selected_features_by_output.csv")
        kn = rd(AUDITS / "11_kneedle_selection.csv")
        if sf is not None:
            ff = sf.groupby("feature").size().reset_index(name="n_outputs_selected")
            ff["type"] = ff.feature.apply(lambda f: "harmonic" if any(x in f for x in ("hrm", "thd"))
                                          else "DG" if "dg" in f.lower() else "fundamental")
            ff = ff.sort_values("n_outputs_selected", ascending=False).reset_index(drop=True)
            save(ff, "table_04_selected_features.csv")
        if kn is not None:
            save(kn.iloc[[0]].copy(), "table_04_kneedle.csv")
    except Exception as e:
        log.warning("  table_04 failed: %s", e)

    # table_05: model metrics (active NAE + ERG) + Wilcoxon vs LSTM
    try:
        lm = rd(METRICS / "16_load_metrics_long.csv")
        st = rd(METRICS / "18_statistical_tests.csv")
        if lm is not None:
            nae = lm[lm.active].groupby(["model", "seed"]).nae_pct.mean().reset_index()
            m_nae = nae.groupby("model").nae_pct.agg(NAE_mean="mean", NAE_std="std").round(3)
            erg = erg_by_model_seed(lm)
            m_erg = erg.groupby("model").erg_sys.agg(ERG_mean="mean", ERG_std="std").round(3)
            t05 = m_nae.join(m_erg).reset_index()
            if st is not None:
                def wil(m):
                    r = st[((st.model1 == "LSTM") & (st.model2 == m)) | ((st.model1 == m) & (st.model2 == "LSTM"))]
                    if m == "LSTM" or r.empty: return ("—", None, "—")
                    x = r.iloc[0]; return (x.sig, round(x.cohen_d, 3), x.effect)
                t05["sig_vs_LSTM"] = t05.model.map(lambda m: wil(m)[0])
                t05["cohen_d_vs_LSTM"] = t05.model.map(lambda m: wil(m)[1])
                t05["effect_vs_LSTM"] = t05.model.map(lambda m: wil(m)[2])
            t05 = t05.sort_values("NAE_mean").reset_index(drop=True)
            save(t05, "table_05_model_metrics.csv", mark=True)
    except Exception as e:
        log.warning("  table_05 failed: %s", e)

    # table_06: energy metrics (per load, mean over seeds) + DG per phase (R2)
    try:
        el = rd(METRICS / "16_energy_by_load.csv")
        if el is not None:
            t06a = el.groupby(["model", "load"]).agg(
                E_true_kWh=("e_true_kwh", "mean"), E_pred_kWh=("e_pred_kwh", "mean"),
                erg_load_pct=("erg_load_pct", "mean")).round(3).reset_index()
            save(t06a, "table_06_energy_by_load.csv", mark=True)
        dg = rd(DG / "17_dg_energy_by_phase.csv")
        if dg is not None:
            t06b = dg[dg.scenario == "R2"][["model", "phase", "E_MG_kWh", "E_other_kWh",
                                            "E_DG_obs_kWh", "E_DG_pred_kWh", "ERG_bal_pct"]].copy()
            save(t06b, "table_06_dg_energy_by_phase.csv", mark=True)
    except Exception as e:
        log.warning("  table_06 failed: %s", e)

    # table_07: dg sensitivity (S0-S7) + structural impact
    try:
        se = rd(DG / "17_dg_sensitivity.csv")
        si = rd(DG / "17_dg_structural_impact.csv")
        if se is not None: save(se.copy(), "table_07_dg_sensitivity.csv", mark=True)
        if si is not None: save(si.copy(), "table_07_dg_structural_impact.csv", mark=True)
    except Exception as e:
        log.warning("  table_07 failed: %s", e)

    # table_08: uncertainty (components + interval)
    try:
        uc = rd(DG / "17_dg_uncertainty.csv")
        ui = rd(DG / "17_dg_uncertainty_interval.csv")
        if uc is not None: save(uc.copy(), "table_08_uncertainty_components.csv")
        if ui is not None: save(ui.copy(), "table_08_uncertainty_interval.csv", mark=True)
    except Exception as e:
        log.warning("  table_08 failed: %s", e)

    # table_09: statistical tests + PFI top-15
    try:
        st = rd(METRICS / "18_statistical_tests.csv")
        pf = rd(METRICS / "18_pfi_results.csv")
        if st is not None:
            save(st[["label", "W", "p_value", "cohen_d", "abs_d", "effect", "sig", "n_pairs"]].copy(),
                 "table_09_statistical_tests.csv")
        if pf is not None:
            save(pf.head(15).copy(), "table_09_pfi_top15.csv")
    except Exception as e:
        log.warning("  table_09 failed: %s", e)

    # table_10: economic (summary per model + DG uncertainty)
    try:
        er = rd(METRICS / "21_economic_results.csv")
        es = rd(METRICS / "21_economic_sensitivity.csv")
        if er is not None:
            t10 = er.groupby("model").agg(
                C_real_USD=("C_real_USD", "mean"), C_error_USD=("C_error_USD", "mean"),
                delta_C_pct=("delta_C_pct", "mean"), C_yearly_USD=("C_yearly_USD", "mean")
            ).round(3).sort_values("delta_C_pct").reset_index()
            save(t10, "table_10_economic_analysis.csv", mark=True)
        if es is not None:
            u = es[es.model == "UNCERTAINTY_DG"].copy()
            if not u.empty: save(u, "table_10_economic_uncertainty.csv")
    except Exception as e:
        log.warning("  table_10 failed: %s", e)

    # table_11: extended metrics (NAE, MAE, AAE, ERG); MSE/RMSE not available from script 16
    try:
        lm = rd(METRICS / "16_load_metrics_long.csv")
        if lm is not None:
            # compatible with the new 16 (energy_abs_error_contrib_pct) and the old one (aae_pct)
            energy_col = ("energy_abs_error_contrib_pct"
                          if "energy_abs_error_contrib_pct" in lm.columns else "aae_pct")
            base = lm.groupby("model").agg(
                NAE_pct=("nae_pct", "mean"), MAE_W=("mae_W", "mean"),
                energy_abs_error_contrib_pct=(energy_col, "mean")).reset_index()
            erg = erg_by_model_seed(lm).groupby("model").erg_sys.mean().reset_index(name="ERG_pct")
            t11 = base.merge(erg, on="model")
            t11["AAE_pct"] = 100.0 - t11["ERG_pct"]   # AAE = 1 - E_error/E_tot
            t11 = t11[["model", "NAE_pct", "MAE_W", "AAE_pct", "ERG_pct",
                       "energy_abs_error_contrib_pct"]]
            t11 = t11.round({"NAE_pct": 3, "MAE_W": 1, "AAE_pct": 3, "ERG_pct": 3,
                             "energy_abs_error_contrib_pct": 3}).sort_values("NAE_pct").reset_index(drop=True)
            save(t11, "table_11_extended_metrics.csv", mark=True)
            log.info("  (AAE = 100-ERG; MSE/RMSE are not produced by script 16: omitted)")
    except Exception as e:
        log.warning("  table_11 failed: %s", e)

    # manifest
    rows = []
    for n in generated:
        p = TABDATA / n; df = pd.read_csv(p)
        rows.append({"filename": n, "n_rows": df.shape[0], "n_cols": df.shape[1],
                     "columns": ";".join(df.columns), "size_bytes": p.stat().st_size})
    pd.DataFrame(rows).to_csv(TABDATA / "table_manifest.csv", index=False)
    log.info("  tabdata/table_manifest.csv (%d entries)", len(rows))


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: tabdata/ -> tables/*.tex  (reads ONLY from tabdata/)
# ══════════════════════════════════════════════════════════════════════════════
_UNICODE_TEX = {"×": r"\ensuremath{\times}", "φ": r"\ensuremath{\phi}",
                "Δ": r"\ensuremath{\Delta}", "≥": r"\ensuremath{\ge}",
                "≤": r"\ensuremath{\le}", "−": "-", "μ": r"\ensuremath{\mu}",
                "σ": r"\ensuremath{\sigma}", "ρ": r"\ensuremath{\rho}",
                "α": r"\ensuremath{\alpha}"}


def _esc(s):
    s = (str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
         .replace("#", r"\#").replace("$", r"\$"))
    # unicode symbols -> LaTeX (\ensuremath works in text and math mode, no $ needed)
    for u, t in _UNICODE_TEX.items():
        s = s.replace(u, t)
    return s


def _fmt(v):
    if pd.isna(v): return "—"
    if isinstance(v, (int, np.integer)): return str(int(v))
    if isinstance(v, (float, np.floating)): return f"{v:.3f}".rstrip("0").rstrip(".")
    return _esc(v)


def to_tex(name_csv, caption, label, maxrows=60):
    p = TABDATA / name_csv
    if not p.exists():
        log.warning("  tex skipped (no tabdata): %s", name_csv); return None
    df = pd.read_csv(p)
    if df.empty:
        log.warning("  tex skipped (empty): %s", name_csv); return None
    # sensitivity note + hide the analysis_group column in the rendering
    has_sens = "analysis_group" in df.columns and (df["analysis_group"] == "hpo_sensitivity").any()
    df = df.drop(columns=[c for c in ("analysis_group", "is_primary") if c in df.columns])
    cap = caption + ("  " + SENS_NOTE if has_sens else "")
    trunc = len(df) > maxrows
    if trunc: df = df.head(maxrows)
    colfmt = "l" + "r" * (len(df.columns) - 1)
    L = [r"\begin{table}[htbp]", r"  \centering", r"  \small",
         f"  \\caption{{{cap}}}", f"  \\label{{{label}}}",
         f"  \\begin{{tabular}}{{{colfmt}}}", r"    \toprule",
         "    " + " & ".join(_esc(c) for c in df.columns) + r" \\", r"    \midrule"]
    for _, r_ in df.iterrows():
        L.append("    " + " & ".join(_fmt(v) for v in r_) + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}"]
    if trunc: L.append(f"  % truncated to {maxrows} rows")
    L.append(r"\end{table}")
    out = TABLES / name_csv.replace(".csv", ".tex")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    log.info("  tables/%s", out.name)
    return out.name


def build_tex():
    log.info("=== PART 2: tabdata/ -> tables/*.tex ===")
    CAP = {
        "table_01_dataset_summary.csv": ("Dataset and experiment summary.", "tab:dataset"),
        "table_02_channel_eligibility.csv": ("Channel eligibility from the 02B audit: primary-class counts and secondary flags (is\\_proxy shown explicitly).", "tab:eligibility"),
        "table_03_preprocessing_audit.csv": ("Preprocessing audit: outliers, gap filling and energy preservation.", "tab:preproc"),
        "table_04_selected_features.csv": ("Selected features by selection frequency across outputs.", "tab:features"),
        "table_04_kneedle.csv": ("RFECV-SVR + Kneedle selection summary.", "tab:kneedle"),
        "table_05_model_metrics.csv": ("Model performance (active NAE and system ERG) with Wilcoxon vs.\\ LSTM.", "tab:models"),
        "table_06_energy_by_load.csv": ("Energy per load: observed vs.\\ predicted.", "tab:eload"),
        "table_06_dg_energy_by_phase.csv": ("Diesel-generator balance energy by phase (scenario R2).", "tab:dgenergy"),
        "table_07_dg_sensitivity.csv": ("D\\_G sensitivity scenarios S0--S7.", "tab:dgsens"),
        "table_07_dg_structural_impact.csv": ("Structural impact of P\\_other components (S4--S7).", "tab:dgstruct"),
        "table_08_uncertainty_components.csv": ("Uncertainty components per phase (three-term budget).", "tab:unccomp"),
        "table_08_uncertainty_interval.csv": ("Load-consistency residual 95\\% uncertainty interval.", "tab:uncint"),
        "table_09_statistical_tests.csv": ("Pairwise Wilcoxon tests with Cohen's d.", "tab:wilcoxon"),
        "table_09_pfi_top15.csv": ("Permutation feature importance (top 15).", "tab:pfi"),
        "table_10_economic_analysis.csv": ("Economic analysis: cost error per model.", "tab:economic"),
        "table_10_economic_uncertainty.csv": ("Economic uncertainty of the load-consistency residual.", "tab:econunc"),
        "table_11_extended_metrics.csv": ("Extended metrics per model (NAE, MAE, AAE, ERG).", "tab:extended"),
    }
    made = []
    for csv, (cap, lab) in CAP.items():
        r = to_tex(csv, cap, lab)
        if r: made.append(r)
    pd.DataFrame({"tex_file": made}).to_csv(TABLES / "tables_manifest.csv", index=False)
    log.info("  %d .tex tables generated", len(made))
    return made


def main():
    log.info("=== 22_build_tables.py ===")
    build_tabdata()
    made = build_tex()
    manifest = {"script": "22_build_tables.py", "merge_de": ["28A_prepare_table_data.py", "28B_generate_tables_tex.py"],
                "run_timestamp": pd.Timestamp.utcnow().isoformat(),
                "n_tabdata": len(generated), "n_tex": len(made),
                "regra": "Part 1 reads audits/metrics/dg -> tabdata; Part 2 reads only tabdata -> tables/.tex"}
    (MANIFESTS / "22_build_tables_params.json").write_text(json.dumps(manifest, indent=2, default=str))
    log.info("Done: %d tabdata, %d tex.", len(generated), len(made))


if __name__ == "__main__":
    main()
