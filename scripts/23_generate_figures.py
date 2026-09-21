#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 23_generate_figures.py

File type: Figure-generation script (pipeline stage 23: result figures of the article)

Purpose:
    RESULT figures of the article.

    Rules (figure protocol):
      - Two phases: (A) prepare_figdata (reads dg/metrics/predictions -> writes figdata/23_*),
        (B) plot (reads ONLY figdata/ -> figures/*.pdf). No figure recomputes results.
      - Image protocol: usetex, serif/cm, EVERYTHING at 18 pt, constrained_layout, PDF 300 dpi,
        no title in the figure (title via the LaTeX caption), 1 file per figure, spines off.
      - Consistent palette. 6 MAIN models in the main figures; PE-ES-Optuna* enters as a
        SENSITIVITY SERIES (dashed style / distinct marker), not as a main model.
      - "Report when there is no data": a figure without input is logged and skipped.

    Methodological conclusion kept: MoTE_v2 = best main model (point-wise NAE);
    PE-ES-Optuna* = best energy sensitivity (ERG), with an additional HPO budget.

    Result figures generated:
      figure_10_pfi.pdf . figure_11_pred_obs_*.pdf . figure_16_nae_by_model.pdf .
      figure_13_p_other.pdf . figure_17_dg_balance.pdf . figure_17_dg_sensitivity.pdf .
      figure_16_uncertainty.pdf . figure_17A_cohen_d.pdf . figure_17B_nae_erg.pdf .
      figure_18A_c_error.pdf . figure_18B_c_yearly.pdf

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
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE PROTOCOL (12 rules; canonical for the whole article/SI)
#  1. PDF at 300 dpi with embedded fonts (pdf.fonttype/ps.fonttype = 42).
#  2. No title in the figure, since the title goes in the LaTeX \caption.
#  3. English, with the capitalisation used in the article.
#  4. Multi-panel: one image per panel, assembled in the .tex with \subfigure.
#  5. Multi-panel: legend only in the first panel (in-plot); the others without.
#  6. No explanatory text inside the figure; it goes in the body text and/or
#     in the LaTeX \caption.
#  7. A colour assigned to something is kept for that thing until the end of the article
#     (fixed palette; see CMAP/PHCOL).
#  8. Markers per model, from the beginning to the end of the article:
#     LSTM (o, circle), RCNN-att (s, square), PE-ES (^, triangle up),
#     SPEC (v, triangle down), MoTE_v2 (D, diamond), Ensemble-G4 (P, plus).
#  9. All fonts of the same size: letters, numbers and LaTeX symbols.
# 10. Font not smaller than 18 pt nor larger than 22 pt (current standard: 18 pt, uniform).
# 11. LaTeX symbols rendered by LaTeX itself (usetex=True, serif/cm),
#     identical between the figure PDF and the article PDF, with units in
#     brackets: [kWh], [\%], [W]; same tokens as the .tex (e.g. $u_{\mathrm{met},L}$).
# 12. Whenever possible, Okabe-Ito palette (colourblind-safe), fixed hex values.
# Operational: a figure reads only from figdata/ (no recomputation); no input -> log and skip.
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "text.usetex": True, "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 18, "axes.labelsize": 18, "xtick.labelsize": 18, "ytick.labelsize": 18,
    "legend.fontsize": 18, "legend.title_fontsize": 18, "axes.titlesize": 18,
    "figure.constrained_layout.use": True, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})

# Fallback: without LaTeX installed, disables usetex (for the final article, keep usetex=True).
try:
    import shutil
    if shutil.which("latex") is None:
        plt.rcParams["text.usetex"] = False
except Exception:
    pass

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # repository root (parent of scripts/)
DG       = PROJECT_ROOT / "dg"
METRICS  = PROJECT_ROOT / "metrics"
PREDS    = PROJECT_ROOT / "predictions"
DATA     = PROJECT_ROOT / "data" / "processed"
AUDITS   = PROJECT_ROOT / "audits"
MANIFESTS = PROJECT_ROOT / "manifests"
FIGDATA  = PROJECT_ROOT / "figdata"
FIGURES  = PROJECT_ROOT / "figures"
LOGS     = PROJECT_ROOT / "logs"
for d in (FIGDATA, FIGURES, LOGS):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(LOGS / "23_generate_figures.log", mode="w", encoding="utf-8")])
log = logging.getLogger(__name__)

# -- palette / style ------------------------------------------------------------------------
PRIMARY = ["LSTM", "RCNN_att", "PE_ES", "SPEC", "MoTE_v2", "Ensemble_G4"]
CMAP = {"LSTM": "#0072B2", "RCNN_att": "#E69F00", "PE_ES": "#009E73",
        "SPEC": "#D55E00", "MoTE_v2": "#CC79A7", "Ensemble_G4": "#56B4E9"}
OPTUNA = "PE_ES_Optuna"
OPTUNA_DISP = "PE-ES-Optuna*"
OPTUNA_COLOR = "#333333"
TYPE_COLOR = {"harmonic": "#0072B2", "harmónica": "#0072B2",
              "fundamental": "#009E73", "DG": "#D55E00"}
SEEDS = [42, 123, 456, 789, 1024]
DT_KWH = (1.0 / 60.0) / 1000.0

# -- standardised symbols (identical to the article text, rule 4) ---------------------------
MODEL_DISP = {
    "LSTM": "LSTM", "RCNN_att": "RCNN-att", "PE_ES": "PE-ES", "SPEC": "SPEC",
    "MoTE_v2": r"MoTE$_{v2}$", "Ensemble_G4": "Ensemble-G4",
    "PE_ES_Optuna": "PE-ES-Optuna*",
}
COMP_DISP = {"u_mg": r"$u_{M_G}$", "u_Pi": r"$u_{P_i}$", "u_sync": r"$u_{sync}$",
             "u_imp": r"$u_{imp}$", "u_loss": r"$u_{loss}$"}


def mdisp(m):
    """Model label consistent with the article text."""
    return MODEL_DISP.get(str(m), str(m).replace("_", r"\_"))


def meter_disp(m):
    """M1 = general meter M_G; the other meters keep their number."""
    return r"$M_G$" if int(m) == 1 else f"M{int(m)}"


def sf(fig, name: str):
    fig.savefig(FIGURES / name, format="pdf", dpi=300)
    plt.close(fig)
    log.info("  figures/%s", name)


def skip(name, reason="data unavailable"):
    log.warning("  SKIP figure %s: %s", name, reason)


def fdcsv(name):
    p = FIGDATA / name
    return pd.read_csv(p) if p.exists() else None


# Meters of the article scope: M_G (M1) + the six modelled loads Gamma1..Gamma6.
ANOM_METERS = [1, 5, 6, 7, 11, 15, 16, 100]   # M_G + 6 loads + P_other (P01)
ANOM_PHASES = ["p_a", "p_b", "p_c"]  # candidates; the best phase is chosen per meter
ANOM_VAR = "p_b"                      # legacy (compat.); the selection is now per meter


ANOM_PHASE_OVERRIDE = {6: "p_b", 7: "p_b"}   # manual phase choice (quality of the fill shape)


def _best_anom_phase(clean_df, meter):
    """Best power phase for DISPLAY in the article, prioritising legibility:
    (1) avoids a reverse phase (median < 0, which starts negative and looks wrong);
    (2) smallest largest-contiguous-gap (avoids a long/flat filling dominating);
    (3) highest observed coverage."""
    if meter in ANOM_PHASE_OVERRIDE:
        return ANOM_PHASE_OVERRIDE[meter]
    sub = clean_df[clean_df.meter_id == meter]
    cands = []
    for var in ANOM_PHASES:
        if var not in sub.columns:
            continue
        v = sub[var].to_numpy(dtype=float)
        fin = v[np.isfinite(v)]
        if len(fin) == 0:
            continue
        maxgap = cur = 0
        for b in ~np.isfinite(v):
            cur = cur + 1 if b else 0
            if cur > maxgap:
                maxgap = cur
        reverse = 1 if float(np.median(fin)) < 0 else 0
        cands.append((reverse, maxgap, -len(fin), var))
    if not cands:
        return ANOM_VAR
    cands.sort()
    return cands[0][3]
_SOLID_CLOSE_GAP = 15                 # merges marks separated by <= N samples (avoids a "barcode")


def _mask_runs(mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), (np.flatnonzero(d == -1) - 1).tolist()))


def _close_mask(mask, close_gap=_SOLID_CLOSE_GAP):
    """Solidifies the mask: merges holes <= close_gap between marks (the region becomes solid colour)."""
    mask = np.asarray(mask, dtype=bool)
    out = mask.copy()
    idx = np.flatnonzero(mask)
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a <= close_gap:
            out[a:b + 1] = True
    return out


def _prep_anomaly_fill():
    """Generates 23_anomaly_series.csv (clean observed + filled p_b per meter) and
    23_anomaly_regions.csv (gap/sentinel/outlier blocks), for fig_anomaly_fill."""
    clean_p = FIGDATA / "06_para_preencher_sanitizado.csv"
    if not clean_p.exists():                       # consistent v2: the fill runs directly on the validated data
        clean_p = FIGDATA / "06_para_preencher.csv"
    fill_p = FIGDATA / "08_preenchido.parquet"
    if not clean_p.exists() or not fill_p.exists():
        log.warning("  anomaly: source missing (%s / %s)", clean_p.name, fill_p.name); return

    _cols = ["meter_id", "time"] + ANOM_PHASES
    clean = pd.read_csv(clean_p, usecols=lambda c: c in _cols)
    fill = pd.read_parquet(fill_p, columns=[c for c in _cols])
    conf_p = FIGDATA / "08_fill_confidence.parquet"        # confidence class per point
    conf = pd.read_parquet(conf_p, columns=[c for c in _cols]) if conf_p.exists() else None
    srows = []
    for m in ANOM_METERS:
        c = clean[clean.meter_id == m].sort_values("time").reset_index(drop=True)
        f = fill[fill.meter_id == m].sort_values("time").reset_index(drop=True)
        k = (conf[conf.meter_id == m].sort_values("time").reset_index(drop=True)
             if conf is not None else None)
        for var in ANOM_PHASES:                        # ALL phases per meter
            n = min(len(c), len(f))
            if n == 0 or var not in c.columns or var not in f.columns:
                continue
            cv = c[var].to_numpy(dtype=float)[:n]
            fv = f[var].to_numpy(dtype=float)[:n]
            if k is not None and var in k.columns and len(k) >= n:
                kv = k[var].to_numpy(dtype=float)[:n]
            else:
                kv = np.full(n, -1)
            for x in range(n):
                srows.append({"meter_id": m, "x": x, "clean": cv[x], "filled": fv[x],
                              "conf": int(kv[x]) if np.isfinite(kv[x]) else -1, "phase": var})
    pd.DataFrame(srows).to_csv(FIGDATA / "23_anomaly_series.csv", index=False)
    log.info("  figdata/23_anomaly_series.csv (%d rows)", len(srows))

    # Sentinel/outlier are specific to the variable p_b; the gap is derived in the plot
    # from p_b itself (NaN that is neither sentinel nor outlier), ensuring
    # phase consistency (the gap locations are indexed by the reference current).
    def _loc(name, cols):
        p = FIGDATA / name
        return pd.read_csv(p, usecols=cols) if p.exists() else pd.DataFrame(columns=cols)

    sent = _loc("06_sentinel_locations.csv", ["variavel", "meter_id", "x_inicio", "x_fim"])
    out = _loc("06_outlier_locations.csv", ["variavel", "meter_id", "x"])
    rrows = []
    for m in ANOM_METERS:
        for var in ANOM_PHASES:                        # regions per (meter, phase)
            b = sent[(sent.meter_id == m) & (sent.variavel == var)]
            for _, r in b.iterrows():
                rrows.append({"meter_id": m, "phase": var, "kind": "sentinel",
                              "x0": int(r.x_inicio), "x1": int(r.x_fim)})
            o = out[(out.meter_id == m) & (out.variavel == var)]
            for _, r in o.iterrows():
                rrows.append({"meter_id": m, "phase": var, "kind": "outlier",
                              "x0": int(r.x), "x1": int(r.x)})
    pd.DataFrame(rrows, columns=["meter_id", "phase", "kind", "x0", "x1"]).to_csv(
        FIGDATA / "23_anomaly_regions.csv", index=False)
    log.info("  figdata/23_anomaly_regions.csv (%d blocks)", len(rrows))


def _prep_cov_roles():
    """Generates 23_cov_roles.csv: OPERATIONAL presence (non-sentinel) per meter/minute
    + final role code (1 general, 2 load, 3 P_other, 4 redundant, 5 inactive)."""
    raw_p = PROJECT_ROOT / "data" / "interim" / "02_raw_standardized.parquet"
    if not raw_p.exists():
        log.warning("  cov_roles: source missing (%s)", raw_p.name); return
    raw = pd.read_parquet(raw_p, columns=["meter_id", "time", "i_bn", "p_b"])
    raw["t"] = pd.to_datetime(raw["time"], utc=True, errors="coerce").dt.floor("min")
    ROLE = {1: 1}
    for m in (5, 6, 7, 11, 15, 16): ROLE[m] = 2          # Gamma loads (M15 = Gamma_5)
    for m in (10, 12, 13, 14): ROLE[m] = 3               # P_other (M14 is a distinct load -> P_other)
    ROLE[2] = 4
    for m in (8, 9, 17, 18): ROLE[m] = 5

    def operational(v):
        v = v.astype(float)
        if v.notna().sum() < 5:
            return pd.Series(False, index=v.index)
        mode = v.round(6).mode()
        mv = mode.iloc[0] if len(mode) else np.nan
        return v.notna() & (v.round(6) != mv)
    rows = []
    for m in sorted(raw.meter_id.unique()):
        d = raw[raw.meter_id == m]
        op = operational(d["i_bn"]) | operational(d["p_b"])
        g = pd.DataFrame({"t": d["t"].values, "op": op.values}).groupby("t")["op"].max()
        rc = ROLE.get(int(m), 5)
        for t, val in g.items():
            rows.append({"meter_id": int(m), "t": t, "present": int(bool(val)), "role": rc})
    pd.DataFrame(rows).to_csv(FIGDATA / "23_cov_roles.csv", index=False)
    log.info("  figdata/23_cov_roles.csv (%d rows)", len(rows))


# ══════════════════════════════════════════════════════════════════════════════
# PHASE A: prepares figdata/23_* from dg/metrics/predictions
# ══════════════════════════════════════════════════════════════════════════════
def prepare_figdata():
    log.info("=== PHASE A: prepare figdata/23_* ===")

    def cp(src: Path, dst: str, transform=None):
        if not src.exists():
            log.warning("  source missing: %s", src.name); return
        df = pd.read_csv(src)
        if transform is not None:
            df = transform(df)
        df.to_csv(FIGDATA / dst, index=False)
        log.info("  figdata/%s (%d rows)", dst, len(df))

    cp(METRICS / "16_pfi_top_features_long.csv", "23_pfi.csv")
    cp(METRICS / "18_model_comparison_long.csv", "23_model_comparison.csv")
    cp(METRICS / "18_statistical_tests.csv", "23_statistical_tests.csv")
    cp(METRICS / "21_economic_sensitivity_long.csv", "23_economic.csv")
    cp(DG / "17_dg_energy_by_phase.csv", "23_dg_energy.csv")
    cp(DG / "17_dg_sensitivity.csv", "23_dg_sensitivity.csv")
    cp(DG / "17_dg_uncertainty.csv", "23_uncertainty.csv")

    # NAE per model/output (active) -> boxplot
    lm = METRICS / "16_load_metrics_long.csv"
    if lm.exists():
        d = pd.read_csv(lm)
        d = d[d.active][["model", "seed", "output", "nae_pct"]]
        d.to_csv(FIGDATA / "23_nae.csv", index=False)
        log.info("  figdata/23_nae.csv (%d rows)", len(d))

    # P_other energy per phase (script 10) -> bars
    po = PROJECT_ROOT / "audits" / "10_p_other_energy_by_phase.csv"
    if po.exists():
        pd.read_csv(po).to_csv(FIGDATA / "23_p_other.csv", index=False)
        log.info("  figdata/23_p_other.csv")

    # -- figures of Block A/B (copy existing figdata of the pipeline) ---------------------------
    cp(FIGDATA / "02_timestamp_coverage.csv", "23_coverage.csv")
    cp(FIGDATA / "13_split_timeline.csv", "23_split.csv")

    # coverage per meter (current/power): raw (all 16 physical meters, excludes
    # P_other=M100) and usable signal after cleaning. Aligned per minute.
    def _cov_long(src_df, out_csv):
        df = src_df[["meter_id", "time", "i_bn", "p_b"]].dropna(subset=["meter_id", "time"]).copy()
        df = df[df["meter_id"].astype(int) != 100]
        df["t"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.floor("min")
        df["cur"] = df["i_bn"].notna().astype(int)
        df["pow"] = df["p_b"].notna().astype(int)
        agg = df.groupby(["meter_id", "t"], as_index=False)[["cur", "pow"]].max()
        agg.to_csv(FIGDATA / out_csv, index=False)
        log.info("  figdata/%s (%d rows, %d meters)", out_csv, len(agg), agg["meter_id"].nunique())

    interim_raw = PROJECT_ROOT / "data" / "interim" / "02_raw_standardized.parquet"
    if interim_raw.exists():
        _cov_long(pd.read_parquet(interim_raw, columns=["meter_id", "time", "i_bn", "p_b"]), "23_cov_raw.csv")
    else:
        log.warning("  source missing for raw coverage: %s", interim_raw)
    clean_base = FIGDATA / "06_para_preencher_sanitizado.csv"
    if not clean_base.exists():
        clean_base = FIGDATA / "06_para_preencher.csv"
    if clean_base.exists():
        _cov_long(pd.read_csv(clean_base, usecols=["meter_id", "time", "i_bn", "p_b"]), "23_cov_clean.csv")
    cp(FIGDATA / "10_raw_vs_processed_energy_long.csv", "23_energy_stage.csv")
    cp(FIGDATA / "11_feature_ranking_long.csv", "23_feature_ranking.csv")
    cp(AUDITS / "11_feature_ranking.csv", "23_feat_rank_full.csv")
    cp(AUDITS / "11_rfecv_scores.csv", "23_rfecv_scores.csv")
    cp(AUDITS / "11_kneedle_selection.csv", "23_kneedle.csv")
    cp(AUDITS / "11_selected_features_by_output.csv", "23_sel_by_output.csv")
    cp(METRICS / "16_energy_by_load.csv", "23_energy_by_load.csv")
    cp(AUDITS / "10_p_other_components.csv", "23_p_other_components.csv")
    cp(DG / "17_dg_uncertainty_interval.csv", "23_dg_interval.csv")
    cp(FIGDATA / "07_fill_validation_examples.csv", "23_fill_validation.csv")
    cp(AUDITS / "07_fill_validation.csv", "23_holdout.csv")   # held-out proof (v3/linear/MLP)
    cp(AUDITS / "04_inverse_residuals_by_phase.csv", "23_inverse_residuals.csv")

    # -- anomalies + filling per selected meter (M_G + 6 loads) ---------------------------------
    # clean observed (06) vs filled (08) of the p_b series, with gap regions
    # (06_gap_locations), sentinel and outlier (06_*_locations).
    _prep_anomaly_fill()
    _prep_cov_roles()   # operational coverage per role (reconstructed Fig. 5)

    hist = []
    for m, pfx in [("LSTM", "14_lstm"), ("RCNN_att", "14_rcnn_att"), ("PE_ES", "15_pe_es"),
                   ("SPEC", "15_spec"), ("MoTE_v2", "15_mote_v2")]:
        f = FIGDATA / f"{pfx}_training_history_long.csv"
        if f.exists():
            h = pd.read_csv(f); h["model"] = m; hist.append(h)
    if hist:
        pd.concat(hist).to_csv(FIGDATA / "23_training_history.csv", index=False)
        log.info("  figdata/23_training_history.csv")

    # predicted vs observed (MoTE, mean over seeds) for 4 representative loads
    labs_mf = MANIFESTS / "13_create_windows_splits_params.json"
    yraw = DATA / "13_Y_raw_test.npy"
    predf = PREDS / "15_mote_v2_predictions.parquet"
    if labs_mf.exists() and yraw.exists() and predf.exists():
        labels = json.loads(labs_mf.read_text())["results"]["output_labels"]
        y = np.load(yraw).astype(float)
        W = 36; mid = W // 2
        pr = pd.read_parquet(predf)
        yp = pr.groupby("window_idx")[labels].mean().sort_index().to_numpy()
        n = min(len(yp), len(y) - mid)
        yt = y[mid:mid + n]; yp = yp[:n]
        e = np.abs(yt.sum(0)) * DT_KWH
        # 9 panels of the 3x3 set (fig_resultado_12): best phases of the smaller loads,
        # M15 (3 phases) and M6 (2 phases). Covers the six loads.
        sel = ["G1_B", "G3_A", "G4_B", "G5_A", "G5_B", "G5_C", "G6_A", "G2_A", "G2_B"]
        sel = [s for s in sel if s in labels]
        rows = []
        for lb in sel:
            j = labels.index(lb)
            for t in range(n):
                rows.append({"output": lb, "t": t, "obs_W": round(float(yt[t, j]), 2),
                             "pred_W": round(float(yp[t, j]), 2)})
        pd.DataFrame(rows).to_csv(FIGDATA / "23_pred_obs.csv", index=False)
        log.info("  figdata/23_pred_obs.csv (loads=%s)", sel)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE B: figures (reads ONLY figdata/)
# ══════════════════════════════════════════════════════════════════════════════
def fig_10_pfi():
    d = fdcsv("23_pfi.csv")
    if d is None or d.empty: return skip("figure_10_pfi.pdf")
    d = d.sort_values("importance").tail(15)
    col(d)
    fig, ax = plt.subplots(figsize=(9, 7))
    cols = [TYPE_COLOR.get(t, "#777777") for t in d["feature_type"]]
    ax.barh(range(len(d)), d["importance"], color=cols)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([_tex(f) for f in d["feature"]])
    ax.set_xlabel(r"PFI importance $I_f=\overline{\Delta \mathrm{NAE}}_f/\mathrm{NAE}_0$")
    from matplotlib.patches import Patch
    seen = list(dict.fromkeys(d["feature_type"]))
    ax.legend(handles=[Patch(color=TYPE_COLOR.get(t, "#777"), label=t) for t in seen], loc="lower right")
    sf(fig, "figure_10_pfi.pdf")


def fig_16_nae():
    d = fdcsv("23_nae.csv")
    if d is None or d.empty: return skip("figure_16_nae_by_model.pdf")
    order = [m for m in PRIMARY if m in d.model.unique()]
    has_opt = OPTUNA in d.model.unique()
    groups = order + ([OPTUNA] if has_opt else [])
    data = [d[d.model == m]["nae_pct"].values for m in groups]
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    for i, m in enumerate(groups):
        c = OPTUNA_COLOR if m == OPTUNA else CMAP.get(m, "#777")
        bp["boxes"][i].set_facecolor(c); bp["boxes"][i].set_alpha(0.6)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels([mdisp(m) for m in groups], rotation=30, ha="right")
    ax.set_ylabel(r"NAE per output [\%]")
    if has_opt:
        ax.axvline(len(order) + 0.5, ls="--", color="gray", lw=1)
    sf(fig, "figure_16_nae_by_model.pdf")


def fig_13_p_other():
    d = fdcsv("23_p_other.csv")
    if d is None or d.empty: return skip("figure_13_p_other.pdf")
    fig, ax = plt.subplots(figsize=(8, 6))
    x = range(len(d))
    ax.bar(x, d["E_components_sum_kWh"], width=0.4, label="sum of components", color="#0072B2")
    ax.bar([i + 0.4 for i in x], d["E_P01_filled_kWh"], width=0.4, label=r"$P_{01}$ (filled)", color="#E69F00")
    ax.set_xticks([i + 0.2 for i in x]); ax.set_xticklabels([p.upper() for p in d["phase"]])
    ax.set_xlabel(r"Phase $\phi$"); ax.set_ylabel(r"Energy [kWh]"); ax.legend()
    sf(fig, "figure_13_p_other.pdf")


def fig_17_dg_balance():
    d = fdcsv("23_dg_energy.csv")
    if d is None or d.empty: return skip("figure_17_dg_balance.pdf")
    d = d[(d.scenario == "R2") & (d.phase.isin(["A", "B", "C"]))]
    prim = d[d.get("is_primary", True) == True] if "is_primary" in d else d[d.model.isin(PRIMARY)]
    piv = prim.pivot_table(index="model", columns="phase", values="E_DG_pred_kWh", aggfunc="mean")
    piv = piv.reindex([m for m in PRIMARY if m in piv.index])
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(piv)); w = 0.25
    for k, ph in enumerate(["A", "B", "C"]):
        ax.bar(x + (k - 1) * w, piv[ph], w, label=f"phase {ph}")
    ax.set_xticks(x); ax.set_xticklabels([mdisp(m) for m in piv.index], rotation=30, ha="right")
    ax.set_ylabel(r"$E_{D_G}^{pred}$ [kWh] (R2)"); ax.legend(title=r"Phase $\phi$")
    ax.axhline(0, color="k", lw=0.8)
    sf(fig, "figure_17_dg_balance.pdf")


def fig_15_sensitivity():
    d = fdcsv("23_dg_sensitivity.csv")
    if d is None or d.empty: return skip("figure_17_dg_sensitivity.pdf")
    ref = d[d.model == "MoTE_v2"]
    if ref.empty: return skip("figure_17_dg_sensitivity.pdf", "MoTE ausente")
    g = ref.groupby("scenario")[["E_DG_pred_R1_kWh", "E_DG_pred_R2_kWh"]].sum()
    order = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]
    g = g.reindex([s for s in order if s in g.index])
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(g))
    ax.plot(x, g["E_DG_pred_R2_kWh"], "o-", color="#0072B2", label="R2 (losses)")
    ax.plot(x, g["E_DG_pred_R1_kWh"], "s--", color="#D55E00", label="R1 (structural)")
    ax.set_xticks(x); ax.set_xticklabels(g.index)
    ax.set_xlabel("Scenario"); ax.set_ylabel(r"$E_{D_G}$ [kWh] (MoTE$_{v2}$, $\Sigma\phi$)"); ax.legend()
    sf(fig, "figure_17_dg_sensitivity.pdf")


def fig_16_uncertainty():
    d = fdcsv("23_uncertainty.csv")
    if d is None or d.empty: return skip("figure_16_uncertainty.pdf")
    comp = d[d.component != "u_total"]
    piv = comp.pivot_table(index="phase", columns="component", values="u_W", aggfunc="mean")
    order = [c for c in ["u_mg", "u_Pi", "u_sync", "u_imp", "u_loss"] if c in piv.columns]
    piv = piv[order]
    # GROUPED bars per component (the combination is in quadrature, not a linear sum)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(piv)); w = 0.8 / max(len(order), 1)
    for k, cpt in enumerate(order):
        ax.bar(x + (k - (len(order) - 1) / 2) * w, piv[cpt], w,
               label=COMP_DISP.get(cpt, _tex(cpt)))
    ax.set_xticks(x); ax.set_xticklabels([str(p).upper() for p in piv.index])
    ax.set_xlabel(r"Phase $\phi$"); ax.set_ylabel(r"Uncertainty component $u$ [W]")
    ax.legend(title="components (quadrature)", ncol=2)
    sf(fig, "figure_16_uncertainty.pdf")


def fig_17_stat():
    d = fdcsv("23_model_comparison.csv")
    if d is None or d.empty: return skip("figure_17A_cohen_d.pdf")
    # 17A: |cohen d| per comparison (bars)
    cd = d.dropna(subset=["cohen_d"]).drop_duplicates("comparison")
    cd = cd.assign(absd=cd.cohen_d.abs()).sort_values("absd")
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.barh(range(len(cd)), cd["absd"], color="#CC79A7")
    ax.set_yticks(range(len(cd)))
    ax.set_yticklabels([r" $\times$ ".join(mdisp(p.strip()) for p in str(c).split("×"))
                        for c in cd["comparison"]])
    ax.set_xlabel(r"$|\mathrm{Cohen}\ d|$")
    sf(fig, "figure_17A_cohen_d.pdf")
    # 17B: NAE x ERG plane per model (mean +/- deviation ACROSS the 5 seeds; bars in x and y).
    # NAE: mean per seed (over the outputs) and deviation across seeds (23_nae.csv);
    # ERG: mean and deviation across seeds (16_system_erg.csv). Inline labels, no legend.
    nae = fdcsv("23_nae.csv"); erg = fdcsv("16_system_erg.csv")
    if nae is not None and not nae.empty and erg is not None and not erg.empty:
        ps = nae.groupby(["model", "seed"])["nae_pct"].mean().reset_index()
        nst = ps.groupby("model")["nae_pct"].agg(nae_mean="mean", nae_std="std").reset_index()
        me = nst.merge(erg[["model", "erg_mean", "erg_std"]], on="model")
        order = [m for m in PRIMARY if m in set(me.model)]
        me = me.set_index("model").loc[order].reset_index()
        # label offset (pt) to avoid overlap, following the published layout
        LOFF = {"LSTM": (12, 10), "Ensemble_G4": (10, -22), "MoTE_v2": (10, 6),
                "RCNN_att": (4, 16), "PE_ES": (10, -14), "SPEC": (10, 6)}
        fig, ax = plt.subplots(figsize=(9, 7))
        for _, r in me.iterrows():
            c = CMAP.get(r.model, "#777")
            ax.errorbar(r.nae_mean, r.erg_mean, xerr=r.nae_std, yerr=r.erg_std,
                        fmt="o", ms=11, color=c, ecolor=c, elinewidth=1.3,
                        capsize=4, capthick=1.3, zorder=3)
            dx, dy = LOFF.get(r.model, (10, 6))
            ax.annotate(mdisp(r.model), (r.nae_mean, r.erg_mean), textcoords="offset points",
                        xytext=(dx, dy), color=c, va="center")
        ax.set_xlabel(r"Active NAE [\%]"); ax.set_ylabel(r"$\mathrm{ERG}_{\mathrm{net}}$ [\%]")
        sf(fig, "figure_17B_nae_erg.pdf")


def fig_18_economic():
    d = fdcsv("23_economic.csv")
    if d is None or d.empty: return skip("figure_18A_c_error.pdf")
    pe = d[(d.tipo == "prediction_error") & (d.metric == "C_error_USD")]
    if not pe.empty:
        g = pe.groupby("model")["value"].mean()
        g = g.reindex([m for m in PRIMARY if m in g.index] + ([OPTUNA] if OPTUNA in g.index else []))
        fig, ax = plt.subplots(figsize=(10, 6))
        cols = [OPTUNA_COLOR if m == OPTUNA else CMAP.get(m, "#777") for m in g.index]
        ax.bar(range(len(g)), g.values, color=cols)
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels([mdisp(m) for m in g.index], rotation=30, ha="right")
        ax.set_ylabel(r"$C_{\mathrm{error}}$ [USD]")
        sf(fig, "figure_18A_c_error.pdf")
    cy = d[(d.tipo == "prediction_error") & (d.metric == "C_yearly_USD")]
    if not cy.empty:
        g = cy.groupby("model")["value"].mean()
        g = g.reindex([m for m in PRIMARY if m in g.index] + ([OPTUNA] if OPTUNA in g.index else []))
        fig, ax = plt.subplots(figsize=(10, 6))
        cols = [OPTUNA_COLOR if m == OPTUNA else CMAP.get(m, "#777") for m in g.index]
        ax.bar(range(len(g)), g.values, color=cols)
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels([mdisp(m) for m in g.index], rotation=30, ha="right")
        ax.set_ylabel(r"$C_{\mathrm{yearly}}$ [USD/yr]")
        sf(fig, "figure_18B_c_yearly.pdf")


def fig_11_pred_obs():
    d = fdcsv("23_pred_obs.csv")
    if d is None or d.empty: return skip("figure_11_pred_obs.pdf")
    # font 21 ONLY in these panels (assembled 3x3 in the article, displayed at ~1/3 of the width);
    # image size unchanged (figsize 10x4.5). Does not alter the global protocol (18 pt).
    F21 = {"font.size": 18, "axes.labelsize": 18, "xtick.labelsize": 18,
           "ytick.labelsize": 18, "legend.fontsize": 18, "axes.titlesize": 18}
    with plt.rc_context(F21):
        for lb in d["output"].unique():
            g = d[d.output == lb]
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.plot(g["t"], g["obs_W"], color="#0072B2", lw=1.2, label="observed")
            ax.plot(g["t"], g["pred_W"], color="#D55E00", lw=1.0, ls="--", label=r"MoTE$_{v2}$")
            ax.set_xlabel("test time [min]"); ax.set_ylabel(r"Power [W]"); ax.legend()
            safe_lb = str(lb).replace("/", "-").replace("\\", "-").replace(" ", "_")
            sf(fig, f"figure_11_pred_obs_{safe_lb}.pdf")


# -- Block A/B (figdata available in the pipeline) ------------------------------------------
def fig_01_coverage():
    d = fdcsv("23_coverage.csv")
    if d is None or d.empty: return skip("figure_01_coverage.pdf")
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    BLUE = (0.122, 0.467, 0.706); GAP = (0.85, 0.85, 0.85)
    d = d.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    piv = d.pivot_table(index="meter_id", columns="timestamp",
                        values="row_present", aggfunc="max").sort_index()
    M = piv.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.imshow(M, aspect="auto", interpolation="nearest", vmin=0, vmax=1,
              cmap=ListedColormap([GAP, BLUE]))
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([meter_disp(m) for m in piv.index])
    ax.set_xticks(np.arange(0, M.shape[1], 1000))
    ax.set_xlabel("Sample index"); ax.set_ylabel("Meter")
    ax.legend(handles=[Patch(color=BLUE, label="sample present"),
                       Patch(color=GAP, label="absent")],
              ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    sf(fig, "figure_01_coverage.pdf")


# -- coverage per meter: raw sampling vs usable signal ---------------------------------------
_COV_CUR = "i_bn"   # reference current (phase B)
_COV_POW = "p_b"    # reference power (phase B)


def _cov_matrices(long_csv, meter_ref=None, grid_ref=None):
    """Reads figdata/<long_csv> (meter_id, t, cur, pow) and returns boolean current/power
    matrices (meters x time) on a regular 1-min grid. With meter_ref/grid_ref it
    reindexes to the same axes (raw vs clean comparison)."""
    d = fdcsv(long_csv)
    if d is None or d.empty:
        return None, None
    d["t"] = pd.to_datetime(d["t"], utc=True, errors="coerce")
    cur = d.pivot_table(index="meter_id", columns="t", values="cur", aggfunc="max")
    pw = d.pivot_table(index="meter_id", columns="t", values="pow", aggfunc="max")
    if grid_ref is None:
        grid_ref = pd.date_range(cur.columns.min(), cur.columns.max(), freq="1min")
    if meter_ref is None:
        meter_ref = sorted(cur.index.astype(int))
    cur = cur.reindex(index=meter_ref, columns=grid_ref, fill_value=0).fillna(0)
    pw = pw.reindex(index=meter_ref, columns=grid_ref, fill_value=0).fillna(0)
    return cur, pw


def _coverage_strip(cur, pw, out_name, lab_cur, lab_pow, lab_missing, split_lines=False):
    """Heatmap per meter: current band (blue) + power (red); white = absent;
    grey = separator between meters. Protocol (18 pt, no title).
    split_lines: marks the chronological train/val/test cuts (80/10/10)."""
    from matplotlib.patches import Patch
    if cur is None or cur.empty:
        return skip(out_name, "sem dados de cobertura")
    # Okabe-Ito blue/orange: separation by HUE and by LUMINANCE (L* ~45 vs ~70),
    # legible under colour-vision deficiency (Applied Energy accessibility).
    BLUE = np.array([0.000, 0.447, 0.698]); RED = np.array([0.902, 0.624, 0.000])
    WHITE = np.array([1.0, 1.0, 1.0]); GAP = np.array([0.85, 0.85, 0.85])
    meters = list(cur.index)
    img, ypos = [], []
    for mi, m in enumerate(meters):
        c = cur.loc[m].to_numpy().astype(bool); p = pw.loc[m].to_numpy().astype(bool)
        ypos.append(len(img) + 0.5)
        img.append(np.where(c[:, None], BLUE, WHITE))
        img.append(np.where(p[:, None], RED, WHITE))
        if mi < len(meters) - 1:
            img.append(np.tile(GAP, (len(c), 1)))
    n = cur.shape[1]
    fig, ax = plt.subplots(figsize=(13, 8.2))
    ax.imshow(np.stack(img, axis=0), aspect="auto", interpolation="nearest")
    ax.set_yticks(ypos); ax.set_yticklabels([meter_disp(m) for m in meters])
    ax.set_xticks(np.arange(0, n, 1000))
    ax.set_xlabel("Sample index"); ax.set_ylabel("Meter")
    if split_lines:
        for xf in (0.8, 0.9):
            ax.axvline(xf * n, color="black", lw=1.3, ls="--", zorder=6)
        trans = ax.get_xaxis_transform()
        for xc, lab in ((0.40, "train"), (0.85, "val"), (0.95, "test")):
            ax.text(xc * n, 1.01, lab, transform=trans, ha="center", va="bottom", fontsize=18)
    ax.legend(handles=[Patch(facecolor=BLUE, label=lab_cur),
                       Patch(facecolor=RED, label=lab_pow),
                       Patch(facecolor=WHITE, edgecolor="0.6", label=lab_missing)],
              ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    sf(fig, out_name)


def fig_coverage_raw_sampling():
    """Raw sampling coverage of all physical meters (excludes P_other)."""
    cur, pw = _cov_matrices("23_cov_raw.csv")
    _coverage_strip(cur, pw, "figure_coverage_raw_sampling.pdf",
                    "current reading", "power reading", "no reading", split_lines=True)


def fig_coverage_clean_signal():
    """Usable signal after cleaning, on the same axes (meters/grid) as the raw one;
    meters discarded in the audit appear without usable signal."""
    ref_cur, _ = _cov_matrices("23_cov_raw.csv")
    if ref_cur is None:
        return skip("figure_coverage_clean_signal.pdf", "cobertura bruta ausente")
    cur, pw = _cov_matrices("23_cov_clean.csv", meter_ref=list(ref_cur.index),
                            grid_ref=ref_cur.columns)
    _coverage_strip(cur, pw, "figure_coverage_clean_signal.pdf",
                    "usable current signal", "usable power signal", "missing after cleaning")


def fig_coverage_roles():
    """Operational (non-sentinel) coverage per meter, coloured by the final ROLE:
    general meter, supervised load, P_other component, redundant and inactive.
    Distinguishes the conditions that the binary version erased (P_other != discarded !=
    inactive != redundant). Light grey = no usable signal after sanitisation."""
    from matplotlib.patches import Patch
    d = fdcsv("23_cov_roles.csv")
    if d is None or d.empty:
        return skip("figure_coverage_roles.pdf", "sem 23_cov_roles")
    d["t"] = pd.to_datetime(d["t"], utc=True, errors="coerce")
    P = d.pivot_table(index="meter_id", columns="t", values="present", aggfunc="max").fillna(0)
    R = d.pivot_table(index="meter_id", columns="t", values="role", aggfunc="max").fillna(0)
    meters = sorted(P.index.astype(int))
    # Okabe-Ito palette, also scaled in LUMINANCE for reading under colour-vision
    # deficiency: black (L*~15) < blue (L*~45) < vermillion (L*~48, hue
    # opposite to blue) < grey (L*~61) < orange (L*~70) < light grey (L*~94).
    COL = {1: np.array([0.15, 0.15, 0.15]),    # general
           2: np.array([0.000, 0.447, 0.698]), # load        (Okabe-Ito blue)
           3: np.array([0.902, 0.624, 0.000]), # P_other     (Okabe-Ito orange)
           4: np.array([0.835, 0.369, 0.000]), # redundant   (Okabe-Ito vermillion)
           5: np.array([0.58, 0.58, 0.58])}    # inactive (medium grey, distinct)
    ABSENT = np.array([0.93, 0.93, 0.93])
    img, ypos = [], []
    for mi, m in enumerate(meters):
        pres = P.loc[m].to_numpy().astype(bool)
        role = int(R.loc[m].to_numpy().max()) or 5
        if role == 5:                          # inactive: whole row in the inactive grey
            band = np.tile(COL[5], (len(pres), 1))
        else:
            band = np.where(pres[:, None], COL.get(role, ABSENT), ABSENT)
        ypos.append(len(img) + 1.0)
        img.append(band); img.append(band)   # double band for legibility
        if mi < len(meters) - 1:
            img.append(np.tile(np.array([1.0, 1.0, 1.0]), (len(pres), 1)))
    n = P.shape[1]
    fig, ax = plt.subplots(figsize=(13, 8.2))
    ax.imshow(np.stack(img, axis=0), aspect="auto", interpolation="nearest")
    ax.set_yticks(ypos); ax.set_yticklabels([meter_disp(m) for m in meters])
    ax.set_xticks(np.arange(0, n, 1000)); ax.set_xticklabels(np.arange(0, n, 1000))
    for xf in (0.8, 0.9):
        ax.axvline(xf * n, color="black", lw=1.3, ls="--", zorder=6)
    _tr = ax.get_xaxis_transform()
    for xc, lab in ((0.40, "train"), (0.85, "val"), (0.95, "test")):
        ax.text(xc * n, 1.01, lab, transform=_tr, ha="center", va="bottom", fontsize=18)
    ax.set_xlabel("Sample index"); ax.set_ylabel("Meter")
    ax.legend(handles=[
        Patch(facecolor=COL[1], label="general meter"),
        Patch(facecolor=COL[2], label="supervised load"),
        Patch(facecolor=COL[3], label=r"$P_{\mathrm{other}}$ component"),
        Patch(facecolor=COL[4], label="redundant (excluded)"),
        Patch(facecolor=COL[5], label="inactive"),
        Patch(facecolor=ABSENT, edgecolor="0.6", label="no usable signal")],
        ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09))
    sf(fig, "figure_coverage_roles.pdf")


def fig_holdout_metrics():
    """Held-out proof: amplitude and roughness ratios (target 1) per window size
    and method (analog, linear, MLP). ONE IMAGE PER PDF (two PDFs; the panel
    is assembled in LaTeX). The analog method preserves amplitude/roughness; linear and
    MLP suppress the texture."""
    d = fdcsv("23_holdout.csv")
    if d is None or d.empty:
        return skip("figure_holdout_amplitude.pdf", "sem 23_holdout")
    NAME = {"fractal": "Analogue", "linear": "Linear", "mlp": "MLP"}
    COL = {"fractal": "#0072B2",
           "linear": "#D55E00",
           "mlp": "#009E73"}
    wins = sorted(d["janela_h"].unique())
    methods = ["fractal", "linear", "mlp"]
    n_win = int(d["n"].iloc[0]) if "n" in d.columns else 50
    for col, ttl, out in (
        ("razao_amplitude",  r"Amplitude ratio $R_A$", "figure_holdout_amplitude.pdf"),
        ("razao_frequencia", r"Roughness ratio $R_f$", "figure_holdout_frequency.pdf"),
    ):
        fig, ax = plt.subplots(figsize=(7, 5.2))
        x = np.arange(len(wins)); w = 0.26
        for k, meth in enumerate(methods):
            vals = [float(d[(d.janela_h == wh) & (d.metodo == meth)][col].iloc[0])
                    if len(d[(d.janela_h == wh) & (d.metodo == meth)]) else np.nan for wh in wins]
            ax.bar(x + (k - 1) * w, vals, w, color=COL[meth], label=NAME[meth])
        ax.axhline(1.0, color="0.4", lw=1, ls="--")
        ax.set_xticks(x); ax.set_xticklabels([f"{wh} h" for wh in wins])
        ax.set_xlabel(rf"Gap size ($n={n_win}$ per bar)")
        ax.set_ylabel(r"ratio (target $=1$)")
        ax.legend(frameon=False, loc="upper right")
        sf(fig, out)


def fig_split():
    d = fdcsv("23_split.csv")
    if d is None or d.empty: return skip("figure_split_timeline.pdf")
    coln = {"train": "#0072B2", "val": "#E69F00", "test": "#009E73"}
    fig, ax = plt.subplots(figsize=(11, 2.4))
    for s, c in coln.items():
        idx = d.index[d.split == s].to_numpy()
        if len(idx):
            ax.barh(0, len(idx), left=idx.min(), color=c, label=s, height=0.6)
    ax.set_yticks([]); ax.set_xlabel("timestep index"); ax.legend(ncol=3, loc="upper center")
    sf(fig, "figure_split_timeline.pdf")


def fig_07_energy_stage():
    d = fdcsv("23_energy_stage.csv")
    if d is None or d.empty: return skip("figure_07_energy_preservation.pdf")
    g = d.assign(absE=d.E_kWh.abs()).groupby(["stage", "phase"]).absE.sum().reset_index()
    order = [s for s in ["raw", "grid", "sanitized", "filled"] if s in g.stage.unique()]
    piv = g.pivot(index="stage", columns="phase", values="absE").reindex(order)
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(piv)); w = 0.25
    for k, ph in enumerate(piv.columns):
        ax.bar(x + (k - 1) * w, piv[ph], w, label=f"phase {ph}")
    ax.set_xticks(x); ax.set_xticklabels(piv.index)
    ax.set_xlabel("processing stage"); ax.set_ylabel(r"$\Sigma|E|$ [kWh]"); ax.legend(title=r"Phase $\phi$")
    sf(fig, "figure_07_energy_preservation.pdf")


def fig_09_rfecv():
    d = fdcsv("23_feature_ranking.csv")
    if d is None or d.empty or "feature" not in d.columns:
        return skip("figure_09_feature_ranking.pdf")
    sub = d.dropna(subset=["feature"])
    freq = (sub.groupby("feature")["frequency"].max() if "frequency" in sub
            else sub.groupby("feature").size()).sort_values().tail(20)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(len(freq)), freq.values, color="#009E73")
    ax.set_yticks(range(len(freq))); ax.set_yticklabels([_tex(f) for f in freq.index])
    ax.set_xlabel("selection frequency across outputs")
    sf(fig, "figure_09_feature_ranking.pdf")


# -- feature selection: RFECV curve, Kneedle elbow, family and heatmap -----------------------
def fig_selection_rfecv():
    d = fdcsv("23_rfecv_scores.csv")
    if d is None or d.empty: return skip("figure_selection_rfecv.pdf")
    fig, ax = plt.subplots(figsize=(9, 6))
    for _, g in d.groupby("output"):
        ax.plot(g["n_features"], g["mae_cv"], color="0.8", lw=0.6)
    m = d.groupby("n_features")["mae_cv"].mean()
    ax.plot(m.index, m.values, color="#0072B2", lw=2.2, label="mean over outputs")
    ax.set_xlabel("Number of features"); ax.set_ylabel("CV MAE")
    ax.legend()
    sf(fig, "figure_selection_rfecv.pdf")


def fig_selection_kneedle():
    d = fdcsv("23_feat_rank_full.csv"); k = fdcsv("23_kneedle.csv")
    if d is None or d.empty or "pct_cumul" not in d.columns:
        return skip("figure_selection_kneedle.pdf")
    d = d.sort_values("rank")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(d["rank"], d["pct_cumul"], color="#009E73", lw=2.2)
    if k is not None and not k.empty:
        kn = int(k["knee_n"].iloc[0]); kp = float(k["knee_pct"].iloc[0]); ad = int(k["adopted_n"].iloc[0])
        ax.scatter([kn], [kp], s=140, color="#D55E00", zorder=5, label=f"knee ($n={kn}$)")
        ax.axvline(ad, ls="--", color="gray", lw=1.3, label=f"adopted ($n={ad}$)")
    ax.set_xlabel("Feature rank"); ax.set_ylabel(r"Cumulative importance [\%]")
    ax.legend(loc="lower right")
    sf(fig, "figure_selection_kneedle.pdf")


_FAM_V = {"V harmonics", "Voltage", "THD (V)"}


def _feat_family(c):
    c = str(c)
    if c.startswith("hrm_v"): return "V harmonics"
    if c.startswith("hrm_i"): return "I harmonics"
    if c.startswith("thdv"):  return "THD (V)"
    if c.startswith("thdi"):  return "THD (I)"
    if c.startswith("v_"):    return "Voltage"
    if c.startswith("i_"):    return "Current"
    if c.startswith(("p_", "q_", "s_")): return "Power"
    if c.startswith("cos"):   return "Power factor"
    return "other"


def fig_feature_family():
    r = fdcsv("23_feat_rank_full.csv")
    if r is None or r.empty or "in_adopted" not in r.columns:
        return skip("figure_feature_family.pdf")
    from matplotlib.patches import Patch
    ad = r[r["in_adopted"].astype(str).str.lower().isin(["true", "1"])]
    counts = ad["feature"].map(_feat_family).value_counts().sort_values()
    BLUE = "#0072B2"; ORANGE = "#E69F00"
    cols = [BLUE if f in _FAM_V else ORANGE for f in counts.index]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(counts)), counts.values, color=cols)
    ax.set_yticks(range(len(counts))); ax.set_yticklabels(list(counts.index))
    ax.set_xlabel("Selected features (of 48)")
    ax.legend(handles=[Patch(color=BLUE, label="voltage-related"),
                       Patch(color=ORANGE, label="current / power")], loc="lower right")
    sf(fig, "figure_feature_family.pdf")


def fig_feature_heatmap():
    d = fdcsv("23_sel_by_output.csv"); r = fdcsv("23_feat_rank_full.csv")
    if d is None or d.empty: return skip("figure_feature_heatmap.pdf")
    d = d.copy(); d["load"] = d["output"].str.extract(r"(G\d+)")
    if r is not None and "in_adopted" in r.columns:
        adf = r[r["in_adopted"].astype(str).str.lower().isin(["true", "1"])].sort_values("rank")
        order = [f for f in adf["feature"] if f in set(d["feature"])]
    else:
        order = sorted(d["feature"].unique())
    loads = ["G1", "G2", "G3", "G4", "G5", "G6"]
    m = (d[d["feature"].isin(order)].groupby(["feature", "load"]).size()
         .unstack(fill_value=0).reindex(index=order, columns=loads, fill_value=0))
    M = m.T  # transposed: loads in rows, features in columns (wide figure)
    from matplotlib.colors import BoundaryNorm
    fig, ax = plt.subplots(figsize=(18, 4.6))
    # integer phase count (0..3) -> discrete scale of 4 blue tones, ticks at 0,1,2,3
    cmap = plt.get_cmap("Blues", 4)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    im = ax.imshow(M.values, aspect="auto", cmap=cmap, norm=norm)
    ax.set_yticks(range(len(loads)))
    ax.set_yticklabels([r"$\Gamma_%d$" % (i + 1) for i in range(6)])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([_tex(f) for f in order], rotation=90)
    ax.set_ylabel("Load"); ax.set_xlabel("Selected feature (by rank)")
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, ticks=[0, 1, 2, 3])
    cb.set_label("Phases selected")
    sf(fig, "figure_feature_heatmap.pdf")


# -- energy per load (obs vs pred) and composition of P_other -------------------------------
_LOAD_LABEL = {"G1": "M5", "G2": "M6", "G3": "M7", "G4": "M11", "G5": "M15", "G6": "M16"}


def fig_energy_by_load():
    d = fdcsv("23_energy_by_load.csv")
    if d is None or d.empty: return skip("figure_energy_by_load.pdf")
    ref = d[d.model == "MoTE_v2"]
    if ref.empty: return skip("figure_energy_by_load.pdf", "MoTE ausente")
    order = ["G1", "G2", "G3", "G4", "G5", "G6"]
    g = ref.groupby("load")[["e_true_kwh", "e_pred_kwh"]].mean().reindex([o for o in order if o in ref.load.unique()])
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(g)); w = 0.38
    ax.bar(x - w / 2, g["e_true_kwh"], w, label="observed", color="#0072B2")
    ax.bar(x + w / 2, g["e_pred_kwh"], w, label=r"MoTE$_{v2}$", color="#D55E00")
    ax.set_xticks(x); ax.set_xticklabels([_LOAD_LABEL.get(i, i) for i in g.index])
    ax.set_xlabel("Load"); ax.set_ylabel("Energy [kWh]"); ax.legend()
    sf(fig, "figure_energy_by_load.pdf")


def fig_pother_composition():
    d = fdcsv("23_p_other_components.csv")
    if d is None or d.empty: return skip("figure_pother_composition.pdf")
    piv = d.pivot_table(index="phase", columns="component_meter",
                        values="E_component_kWh", aggfunc="sum")
    phases = [p for p in ["a", "b", "c"] if p in piv.index]
    piv = piv.reindex(phases)
    comps = [c for c in [10, 12, 13, 14] if c in piv.columns]
    colors = {10: "#0072B2", 12: "#E69F00", 13: "#009E73", 14: "#CC79A7"}
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(phases)); posb = np.zeros(len(phases)); negb = np.zeros(len(phases))
    for c in comps:
        v = piv[c].to_numpy(dtype=float)
        base = np.where(v >= 0, posb, negb)
        ax.bar(x, v, 0.6, bottom=base, label=f"M{int(c)}", color=colors.get(int(c), "#777"))
        posb = posb + np.where(v >= 0, v, 0.0); negb = negb + np.where(v < 0, v, 0.0)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([p.upper() for p in phases])
    ax.set_xlabel(r"Phase $\phi$"); ax.set_ylabel("Energy [kWh]")
    ax.legend(title="component", ncol=2)
    sf(fig, "figure_pother_composition.pdf")


def fig_dg_interval():
    d = fdcsv("23_dg_interval.csv")
    if d is None or d.empty: return skip("figure_dg_interval.pdf")
    m = d[d.model == "MoTE_v2"].set_index("phase")
    phases = [p for p in ["A", "B", "C"] if p in m.index]
    e = [float(m.loc[p, "E_DG_pred_R2_kWh"]) for p in phases]
    u = [float(m.loc[p, "u_total_kWh"]) for p in phases]
    e.append(sum(e)); u.append(float(np.sqrt(sum(x ** 2 for x in u))))
    xlab = phases + ["ABC"]
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(xlab))
    ax.errorbar(x, e, yerr=[2 * ui for ui in u], fmt="o", color="#0072B2",
                capsize=6, ms=9, lw=1.6, label=r"$\hat{E}_{D_G}\pm 2u$")
    ax.axhline(0, color="#D55E00", ls="--", lw=1.3, label="zero")
    ax.set_xticks(x); ax.set_xticklabels(xlab)
    ax.set_xlabel(r"Phase $\phi$"); ax.set_ylabel(r"$E_{D_G}$ [kWh]")
    ax.legend(loc="upper right")
    sf(fig, "figure_dg_interval.pdf")


def fig_anomaly_fill():
    """One figure per (meter, phase) for M_G + 6 loads + P_other, ALL power phases:
    clean observed (blue), supported analog (red), contingency backbone (dashed
    orange), unsupported region (hatched) and uncertainty band; gap/sentinel/outlier
    shaded. Follows the image protocol (English, no title)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    s = fdcsv("23_anomaly_series.csv")
    r = fdcsv("23_anomaly_regions.csv")
    if s is None or s.empty:
        return skip("figure_anomaly_M*.pdf")
    if r is None:
        r = pd.DataFrame(columns=["meter_id", "phase", "kind", "x0", "x1"])
    if "phase" not in r.columns:
        r["phase"] = ""
    combos = [(m, ph) for m in ANOM_METERS for ph in ANOM_PHASES]
    for m, ph_var in combos:
        g = s[(s.meter_id == m) & (s.phase == ph_var)].sort_values("x")
        if g.empty:
            continue
        x = g.x.to_numpy(dtype=int)
        clean = g.clean.to_numpy(dtype=float)
        filled = g.filled.to_numpy(dtype=float)
        n = len(x)

        masks = {k: np.zeros(n, dtype=bool) for k in ("sentinel", "outlier")}
        for _, rr in r[(r.meter_id == m) & (r.phase == ph_var)].iterrows():
            if rr.kind not in masks:
                continue
            x0 = max(0, int(rr.x0)); x1 = min(n - 1, int(rr.x1))
            if x1 >= x0:
                masks[rr.kind][x0:x1 + 1] = True
        # gap = absence in p_b that is neither sentinel nor outlier (consistent in phase B)
        gap_mask = ~np.isfinite(clean) & ~masks["sentinel"] & ~masks["outlier"]
        g_out = _close_mask(masks["outlier"])
        g_sent = _close_mask(masks["sentinel"]) & ~g_out
        g_gap = _close_mask(gap_mask) & ~g_out & ~g_sent

        conf_arr = g.conf.to_numpy(dtype=float) if "conf" in g.columns else np.full(n, -1.0)
        observed = clean.copy()
        filled_gap = filled.copy()
        filled_gap[np.isfinite(clean)] = np.nan   # filled only where the observed is NaN
        # separate SUPPORTED filling (analog texture) from the CONTINGENCY backbone
        is_unsup = (conf_arr == 6) & ~np.isfinite(clean)          # UNSUPPORTED_BACKBONE
        filled_analog = filled_gap.copy(); filled_analog[is_unsup] = np.nan
        filled_unsup = filled_gap.copy(); filled_unsup[~is_unsup] = np.nan
        g_unsup = _close_mask(is_unsup)
        obs_fin = observed[np.isfinite(observed)]
        band = 1.4826 * float(np.median(np.abs(obs_fin - np.median(obs_fin)))) if len(obs_fin) else 0.0

        fig, ax = plt.subplots(figsize=(13, 2.1))
        trans = ax.get_xaxis_transform()
        ax.fill_between(x, 0, 1, where=g_gap & ~g_unsup, transform=trans, color="0.72", alpha=0.95, lw=0, step="mid", zorder=0)
        # region without textural support: hatched (not equivalent to analog filling)
        ax.fill_between(x, 0, 1, where=g_unsup, transform=trans, facecolor="none",
                        edgecolor="0.5", hatch="////", lw=0.0, step="mid", zorder=0)
        ax.fill_between(x, 0, 1, where=g_sent, transform=trans, color="red", alpha=0.5, lw=0, step="mid", zorder=1)
        ax.fill_between(x, 0, 1, where=g_out, transform=trans, color="#CC79A7", alpha=0.65, lw=0, step="mid", zorder=1)
        # uncertainty band around the contingency backbone
        if band > 0 and is_unsup.any():
            lo_b = np.where(is_unsup, filled_gap - band, np.nan)
            hi_b = np.where(is_unsup, filled_gap + band, np.nan)
            ax.fill_between(x, lo_b, hi_b, color="tab:orange", alpha=0.18, lw=0, zorder=1)
        ax.plot(x, filled_analog, color="tab:red", lw=0.8, zorder=2)                 # supported analog
        ax.plot(x, filled_unsup, color="tab:orange", lw=1.0, ls="--", zorder=2)      # contingency
        ax.plot(x, observed, color="tab:blue", lw=0.8, zorder=3)

        fin = np.concatenate([observed[np.isfinite(observed)], filled_gap[np.isfinite(filled_gap)]])
        if len(fin):
            lo, hi = float(np.percentile(fin, 0.5)), float(np.percentile(fin, 99.5))
            mg = max((hi - lo) * 0.05, 1e-9)
            ax.set_ylim(lo - mg, hi + mg)
        ph = (str(g["phase"].iloc[0]).split("_")[-1].upper()
              if "phase" in g.columns and len(g) else "B")
        ax.set_xlabel(r"sample", fontsize=18)
        ax.set_ylabel(rf"$P_{{{ph}}}$ [W]", fontsize=18)
        ax.set_xticks(np.arange(0, n, 1000))
        ax.tick_params(axis="both", labelsize=18)
        # the shared legend is generated ONCE (colour key); the per-meter figures
        # enter as subfigures under a single legend/caption in the article.
        sf(fig, f"figure_anomaly_M{m:02d}_{ph_var}.pdf")

    # shared colour key (one horizontal strip), included in the article above
    # the set of subfigures.
    from matplotlib.lines import Line2D as _L
    from matplotlib.patches import Patch as _P
    figk = plt.figure(figsize=(13, 0.5))
    figk.legend(handles=[
        _L([], [], color="tab:blue", lw=2, label=r"observed"),
        _L([], [], color="tab:red", lw=2, label=r"supported analog fill"),
        _L([], [], color="tab:orange", lw=2, ls="--", label=r"contingency backbone"),
        _P(facecolor="none", edgecolor="0.5", hatch="////", label=r"no textural support"),
        _P(facecolor="0.72", alpha=0.6, label=r"gap region"),
        _P(facecolor="red", alpha=0.35, label=r"sentinel region"),
        _P(facecolor="#CC79A7", alpha=0.4, label=r"outlier region"),
    ], ncol=7, frameon=False, loc="center")
    sf(figk, "figure_anomaly_legend.pdf")


def fig_28_training():
    d = fdcsv("23_training_history.csv")
    if d is None or d.empty: return skip("figure_28_training_history.pdf")
    val = d[d.split == "val"].groupby(["model", "epoch"]).loss.mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 6))
    for m in [x for x in PRIMARY if x in val.model.unique()]:
        g = val[val.model == m]
        ax.plot(g.epoch, g.loss, color=CMAP.get(m, "#777"), label=mdisp(m))
    ax.set_xlabel("epoch"); ax.set_ylabel("validation loss"); ax.set_yscale("log"); ax.legend()
    sf(fig, "figure_28_training_history.pdf")


def fig_03_fill_validation():
    # 1 figure per PDF (protocol): one held-out example per window size.
    d = fdcsv("23_fill_validation.csv")
    if d is None or d.empty: return skip("figure_03_fill_validation.pdf")
    combos = (d[["janela_h", "meter_id", "variavel"]].drop_duplicates()
              .sort_values("janela_h").to_records(index=False))
    for jh, m, v in combos:
        g = d[(d.janela_h == jh) & (d.meter_id == m) & (d.variavel == v)].sort_values("t")
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.plot(g.t, g.verdade, color="black", lw=2.2, label="ground truth")
        ax.plot(g.t, g.fractal, color="#0072B2", lw=1.8, label="fractal")
        ax.plot(g.t, g.linear, color="#E69F00", lw=1.6, ls="--", label="linear")
        if "mlp" in g.columns:
            ax.plot(g.t, g.mlp, color="#009E73", lw=1.6, ls=":", label="MLP")
        ax.set_xlabel("time in gap (min)"); ax.set_ylabel("value")
        ax.legend(loc="best", ncol=2, frameon=False)
        sf(fig, f"figure_03_fill_validation_{int(jh) * 60}min.pdf")


def fig_inverse_residuals():
    d = fdcsv("23_inverse_residuals.csv")
    if d is None or d.empty: return skip("figure_inverse_residuals.pdf")
    def val(m, ph, col):
        r = d[(d.meter_id == m) & (d.phase == ph)]
        return float(r[col].iloc[0]) if len(r) else float("nan")
    recon = [(16, "A"), (16, "B"), (16, "C")]
    persist = [(1, "A"), (2, "A"), (12, "A")]
    labels, before, after = [], [], []
    for m, ph in recon:
        labels.append(f"{meter_disp(m)}$\\cdot${ph}")
        before.append(val(m, ph, "resid_tri_before")); after.append(val(m, ph, "resid_tri_after"))
    for m, ph in persist:
        labels.append(f"{meter_disp(m)}$\\cdot${ph}")
        before.append(val(m, ph, "resid_pcos_before")); after.append(val(m, ph, "resid_pcos_after"))
    x = np.arange(len(labels)); wb = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.bar(x - wb / 2, before, wb, label="unadjusted", color="#b0b0b0")
    ax.bar(x + wb / 2, after, wb, label="adjusted (constrained fit)", color="#0072B2")
    for xi, a in zip(x, after):
        ax.text(xi + wb / 2, a + 0.05, f"{a:.2f}", ha="center", va="bottom")
    for xi, b in zip(x[:3], before[:3]):        # "before" (unadjusted) values of the M16 cases
        ax.text(xi - wb / 2, b + 0.05, f"{b:.2f}", ha="center", va="bottom", color="0.4")
    ax.axvline(2.5, color="0.75", lw=1, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("median relative residual"); ax.set_ylim(0, 2.75)
    ax.legend(loc="upper left", frameon=False)
    ax.text(1.0, 1.45, r"M16: reconcilable ($\varepsilon_{tri}$)", ha="center")
    ax.text(4.0, 2.55, r"reverse phase A: persistent ($\varepsilon_{pcos}$)", ha="center")
    sf(fig, "figure_inverse_residuals.pdf")


def fig_inverse_status():
    d = fdcsv("23_inverse_residuals.csv")
    if d is None or d.empty: return skip("figure_inverse_status.pdf")
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    STAT = {"COMPATIBLE": (0, "#009E73", "Compatible"),
            "DEAD_OR_INSUFFICIENT": (1, "#b0b0b0", "Dead or insufficient"),
            "ALERT": (2, "#f5c518", "Alert"),
            "INCOMPATIBLE_PERSISTENT": (3, "#D55E00", "Incompatible persistent")}
    order = ["COMPATIBLE", "DEAD_OR_INSUFFICIENT", "ALERT", "INCOMPATIBLE_PERSISTENT"]
    cmap = ListedColormap([STAT[k][1] for k in order])
    meters = sorted(d.meter_id.unique().tolist()); phases = ["A", "B", "C"]
    M = np.zeros((len(meters), 3), dtype=int)
    for i, m in enumerate(meters):
        for j, ph in enumerate(phases):
            r = d[(d.meter_id == m) & (d.phase == ph)]
            st = r["inverse_status"].iloc[0] if len(r) else "DEAD_OR_INSUFFICIENT"
            M[i, j] = STAT.get(st, (1, "", ""))[0]
    nm = len(meters)
    fig, ax = plt.subplots(figsize=(12.0, 4.4))
    ax.imshow(M.T, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")
    LETTER = {0: "C", 1: "I", 2: "A", 3: "P"}   # do not rely on colour alone (accessibility)
    for i in range(nm):
        for j in range(3):
            code = int(M[i, j])
            tc = "white" if code in (0, 3) else "black"
            ax.text(i, j, LETTER.get(code, "?"), ha="center", va="center",
                    color=tc, fontsize=18, fontweight="bold")
    ax.set_yticks(range(3)); ax.set_yticklabels(phases)
    ax.set_xticks(range(nm)); ax.set_xticklabels([meter_disp(m) for m in meters])
    ax.set_ylabel("phase"); ax.set_xlabel("meter")
    ax.set_xticks(np.arange(-0.5, nm, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    LETLEG = {"COMPATIBLE": "C", "DEAD_OR_INSUFFICIENT": "I",
              "ALERT": "A", "INCOMPATIBLE_PERSISTENT": "P"}
    leg = [Patch(facecolor=STAT[k][1], edgecolor="0.5", label=f"{LETLEG[k]} --- {STAT[k][2]}") for k in order]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.30),
              frameon=False, ncol=2, handlelength=1.2, columnspacing=2.0)
    sf(fig, "figure_inverse_status.pdf")


def report_missing():
    """Figure slots of the legacy list without an equivalent figdata in the pipeline (reported)."""
    for name, why in [
        ("figure_02_method_curation.pdf", "curation table: no figdata"),
        ("figure_04_m1m2_redundancy.pdf", "M1-M2 does not exist in the current scope"),
        ("figure_05_duplicate_group.pdf", "sem *_duplicate_groups_long em figdata"),
        ("figure_06_outliers_timeline.pdf", "no 07_outliers_timeline_long (locations only)"),
        ("figure_08_missingness.pdf", "sem 09_missingness_timeline_long"),
        ("figure_08_imputation_examples.pdf", "sem 10_imputation_examples_long"),
        ("figure_21_pqs_consistency.pdf", "sem 04_pqs_consistency_long"),
        ("figure_22_resampling.pdf", "sem 08_resampling_coverage_long"),
        ("figure_25_imputation_flags.pdf", "sem 10_imputation_method_flags"),
        ("figure_29_attention_weights.pdf", "gate weights not saved by script 13"),
        ("figure_31_j_by_scenario.pdf", "sem 05B_j_by_scenario_long"),
    ]:
        skip(name, why)


# -- normalisation of variable names to the article notation --------------------------------
# Converts the raw variable identifier (e.g. 'hrm_v_cn_25') into the standardised
# label (e.g. '$v_{CN,25}$'). Applied only via _tex(); any string
# outside this map keeps the previous behaviour (TeX escape).
_VARMAP = {
    'cos_a': '$FP_{A}$',
    'cos_b': '$FP_{B}$',
    'cos_c': '$FP_{C}$',
    'hrm_i_an_01': '$i_{AN,1}$',
    'hrm_i_an_03': '$i_{AN,3}$',
    'hrm_i_an_05': '$i_{AN,5}$',
    'hrm_i_an_07': '$i_{AN,7}$',
    'hrm_i_an_09': '$i_{AN,9}$',
    'hrm_i_an_11': '$i_{AN,11}$',
    'hrm_i_an_13': '$i_{AN,13}$',
    'hrm_i_an_15': '$i_{AN,15}$',
    'hrm_i_an_17': '$i_{AN,17}$',
    'hrm_i_an_19': '$i_{AN,19}$',
    'hrm_i_an_21': '$i_{AN,21}$',
    'hrm_i_an_23': '$i_{AN,23}$',
    'hrm_i_an_25': '$i_{AN,25}$',
    'hrm_i_an_27': '$i_{AN,27}$',
    'hrm_i_an_29': '$i_{AN,29}$',
    'hrm_i_an_31': '$i_{AN,31}$',
    'hrm_i_an_33': '$i_{AN,33}$',
    'hrm_i_an_35': '$i_{AN,35}$',
    'hrm_i_an_37': '$i_{AN,37}$',
    'hrm_i_an_39': '$i_{AN,39}$',
    'hrm_i_an_41': '$i_{AN,41}$',
    'hrm_i_an_43': '$i_{AN,43}$',
    'hrm_i_an_45': '$i_{AN,45}$',
    'hrm_i_an_47': '$i_{AN,47}$',
    'hrm_i_an_49': '$i_{AN,49}$',
    'hrm_i_bn_01': '$i_{BN,1}$',
    'hrm_i_bn_03': '$i_{BN,3}$',
    'hrm_i_bn_05': '$i_{BN,5}$',
    'hrm_i_bn_07': '$i_{BN,7}$',
    'hrm_i_bn_09': '$i_{BN,9}$',
    'hrm_i_bn_11': '$i_{BN,11}$',
    'hrm_i_bn_13': '$i_{BN,13}$',
    'hrm_i_bn_15': '$i_{BN,15}$',
    'hrm_i_bn_17': '$i_{BN,17}$',
    'hrm_i_bn_19': '$i_{BN,19}$',
    'hrm_i_bn_21': '$i_{BN,21}$',
    'hrm_i_bn_23': '$i_{BN,23}$',
    'hrm_i_bn_25': '$i_{BN,25}$',
    'hrm_i_bn_27': '$i_{BN,27}$',
    'hrm_i_bn_29': '$i_{BN,29}$',
    'hrm_i_bn_31': '$i_{BN,31}$',
    'hrm_i_bn_33': '$i_{BN,33}$',
    'hrm_i_bn_35': '$i_{BN,35}$',
    'hrm_i_bn_37': '$i_{BN,37}$',
    'hrm_i_bn_39': '$i_{BN,39}$',
    'hrm_i_bn_41': '$i_{BN,41}$',
    'hrm_i_bn_43': '$i_{BN,43}$',
    'hrm_i_bn_45': '$i_{BN,45}$',
    'hrm_i_bn_47': '$i_{BN,47}$',
    'hrm_i_bn_49': '$i_{BN,49}$',
    'hrm_i_cn_01': '$i_{CN,1}$',
    'hrm_i_cn_03': '$i_{CN,3}$',
    'hrm_i_cn_05': '$i_{CN,5}$',
    'hrm_i_cn_07': '$i_{CN,7}$',
    'hrm_i_cn_09': '$i_{CN,9}$',
    'hrm_i_cn_11': '$i_{CN,11}$',
    'hrm_i_cn_13': '$i_{CN,13}$',
    'hrm_i_cn_15': '$i_{CN,15}$',
    'hrm_i_cn_17': '$i_{CN,17}$',
    'hrm_i_cn_19': '$i_{CN,19}$',
    'hrm_i_cn_21': '$i_{CN,21}$',
    'hrm_i_cn_23': '$i_{CN,23}$',
    'hrm_i_cn_25': '$i_{CN,25}$',
    'hrm_i_cn_27': '$i_{CN,27}$',
    'hrm_i_cn_29': '$i_{CN,29}$',
    'hrm_i_cn_31': '$i_{CN,31}$',
    'hrm_i_cn_33': '$i_{CN,33}$',
    'hrm_i_cn_35': '$i_{CN,35}$',
    'hrm_i_cn_37': '$i_{CN,37}$',
    'hrm_i_cn_39': '$i_{CN,39}$',
    'hrm_i_cn_41': '$i_{CN,41}$',
    'hrm_i_cn_43': '$i_{CN,43}$',
    'hrm_i_cn_45': '$i_{CN,45}$',
    'hrm_i_cn_47': '$i_{CN,47}$',
    'hrm_i_cn_49': '$i_{CN,49}$',
    'hrm_v_an_01': '$v_{AN,1}$',
    'hrm_v_an_03': '$v_{AN,3}$',
    'hrm_v_an_05': '$v_{AN,5}$',
    'hrm_v_an_07': '$v_{AN,7}$',
    'hrm_v_an_09': '$v_{AN,9}$',
    'hrm_v_an_11': '$v_{AN,11}$',
    'hrm_v_an_13': '$v_{AN,13}$',
    'hrm_v_an_15': '$v_{AN,15}$',
    'hrm_v_an_17': '$v_{AN,17}$',
    'hrm_v_an_19': '$v_{AN,19}$',
    'hrm_v_an_21': '$v_{AN,21}$',
    'hrm_v_an_23': '$v_{AN,23}$',
    'hrm_v_an_25': '$v_{AN,25}$',
    'hrm_v_an_27': '$v_{AN,27}$',
    'hrm_v_an_29': '$v_{AN,29}$',
    'hrm_v_an_31': '$v_{AN,31}$',
    'hrm_v_an_33': '$v_{AN,33}$',
    'hrm_v_an_35': '$v_{AN,35}$',
    'hrm_v_an_37': '$v_{AN,37}$',
    'hrm_v_an_39': '$v_{AN,39}$',
    'hrm_v_an_41': '$v_{AN,41}$',
    'hrm_v_an_43': '$v_{AN,43}$',
    'hrm_v_an_45': '$v_{AN,45}$',
    'hrm_v_an_47': '$v_{AN,47}$',
    'hrm_v_an_49': '$v_{AN,49}$',
    'hrm_v_an_51': '$v_{AN,51}$',
    'hrm_v_bn_01': '$v_{BN,1}$',
    'hrm_v_bn_03': '$v_{BN,3}$',
    'hrm_v_bn_05': '$v_{BN,5}$',
    'hrm_v_bn_07': '$v_{BN,7}$',
    'hrm_v_bn_09': '$v_{BN,9}$',
    'hrm_v_bn_11': '$v_{BN,11}$',
    'hrm_v_bn_13': '$v_{BN,13}$',
    'hrm_v_bn_15': '$v_{BN,15}$',
    'hrm_v_bn_17': '$v_{BN,17}$',
    'hrm_v_bn_19': '$v_{BN,19}$',
    'hrm_v_bn_21': '$v_{BN,21}$',
    'hrm_v_bn_23': '$v_{BN,23}$',
    'hrm_v_bn_25': '$v_{BN,25}$',
    'hrm_v_bn_27': '$v_{BN,27}$',
    'hrm_v_bn_29': '$v_{BN,29}$',
    'hrm_v_bn_31': '$v_{BN,31}$',
    'hrm_v_bn_33': '$v_{BN,33}$',
    'hrm_v_bn_35': '$v_{BN,35}$',
    'hrm_v_bn_37': '$v_{BN,37}$',
    'hrm_v_bn_39': '$v_{BN,39}$',
    'hrm_v_bn_41': '$v_{BN,41}$',
    'hrm_v_bn_43': '$v_{BN,43}$',
    'hrm_v_bn_45': '$v_{BN,45}$',
    'hrm_v_bn_47': '$v_{BN,47}$',
    'hrm_v_bn_49': '$v_{BN,49}$',
    'hrm_v_bn_51': '$v_{BN,51}$',
    'hrm_v_cn_01': '$v_{CN,1}$',
    'hrm_v_cn_03': '$v_{CN,3}$',
    'hrm_v_cn_05': '$v_{CN,5}$',
    'hrm_v_cn_07': '$v_{CN,7}$',
    'hrm_v_cn_09': '$v_{CN,9}$',
    'hrm_v_cn_11': '$v_{CN,11}$',
    'hrm_v_cn_13': '$v_{CN,13}$',
    'hrm_v_cn_15': '$v_{CN,15}$',
    'hrm_v_cn_17': '$v_{CN,17}$',
    'hrm_v_cn_19': '$v_{CN,19}$',
    'hrm_v_cn_21': '$v_{CN,21}$',
    'hrm_v_cn_23': '$v_{CN,23}$',
    'hrm_v_cn_25': '$v_{CN,25}$',
    'hrm_v_cn_27': '$v_{CN,27}$',
    'hrm_v_cn_29': '$v_{CN,29}$',
    'hrm_v_cn_31': '$v_{CN,31}$',
    'hrm_v_cn_33': '$v_{CN,33}$',
    'hrm_v_cn_35': '$v_{CN,35}$',
    'hrm_v_cn_37': '$v_{CN,37}$',
    'hrm_v_cn_39': '$v_{CN,39}$',
    'hrm_v_cn_41': '$v_{CN,41}$',
    'hrm_v_cn_43': '$v_{CN,43}$',
    'hrm_v_cn_45': '$v_{CN,45}$',
    'hrm_v_cn_47': '$v_{CN,47}$',
    'hrm_v_cn_49': '$v_{CN,49}$',
    'hrm_v_cn_51': '$v_{CN,51}$',
    'i_an': '$i_{AN}$',
    'i_bn': '$i_{BN}$',
    'i_cn': '$i_{CN}$',
    'p_a': '$P_{A}$',
    'p_b': '$P_{B}$',
    'p_c': '$P_{C}$',
    'q_a': '$Q_{A}$',
    'q_b': '$Q_{B}$',
    'q_c': '$Q_{C}$',
    's_a': '$S_{A}$',
    's_b': '$S_{B}$',
    's_c': '$S_{C}$',
    'thdi_a': '$THD_{i_{A}}$',
    'thdi_b': '$THD_{i_{B}}$',
    'thdi_c': '$THD_{i_{C}}$',
    'thdv_a': '$THD_{v_{A}}$',
    'thdv_b': '$THD_{v_{B}}$',
    'thdv_c': '$THD_{v_{C}}$',
    'v_an': '$v_{AN}$',
    'v_bn': '$v_{BN}$',
    'v_cn': '$v_{CN}$',
}

_TEX_REPL = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "$": r"\$",
             "×": r"\ensuremath{\times}", "φ": r"\ensuremath{\phi}", "Δ": r"\ensuremath{\Delta}",
             "≥": r"\ensuremath{\ge}", "≤": r"\ensuremath{\le}", "−": "-",
             "μ": r"\ensuremath{\mu}", "σ": r"\ensuremath{\sigma}",
             "ρ": r"\ensuremath{\rho}", "α": r"\ensuremath{\alpha}"}


def _tex(s):
    s = str(s)
    if s in _VARMAP:                 # variable in the list -> standardised article notation
        return _VARMAP[s]
    for a, b in _TEX_REPL.items():   # any other string -> previous behaviour
        s = s.replace(a, b)
    return s


def col(d):  # placeholder to keep a stable signature
    return d


def main():
    log.info("=== 23_generate_figures.py ===")
    prepare_figdata()
    log.info("=== PHASE B: figures (reads only figdata/) ===")
    for fn in (fig_01_coverage, fig_coverage_raw_sampling, fig_coverage_clean_signal,
               fig_coverage_roles, fig_holdout_metrics,
               fig_inverse_residuals, fig_inverse_status,
               fig_split, fig_03_fill_validation, fig_07_energy_stage,
               fig_09_rfecv, fig_selection_rfecv, fig_selection_kneedle,
               fig_feature_family, fig_feature_heatmap,
               fig_energy_by_load, fig_pother_composition, fig_dg_interval,
               fig_10_pfi, fig_11_pred_obs, fig_16_nae, fig_13_p_other,
               fig_17_dg_balance, fig_15_sensitivity, fig_16_uncertainty, fig_17_stat,
               fig_18_economic, fig_28_training, fig_anomaly_fill):
        try:
            fn()
        except Exception as e:
            log.warning("  failure in %s: %s", fn.__name__, e)
    report_missing()
    n = len(list(FIGURES.glob("figure_*.pdf")))
    manifest = {"script": "23_generate_figures.py", "ex": "29_generate_figures.py",
                "run_timestamp": pd.Timestamp.utcnow().isoformat(),
                "escopo": "figuras de resultados (10–18); Bloco A/B via fig_bloco_* depois",
                "protocolo": "fig_bloco_A (usetex, 18pt, PDF); 6 principais + PE-ES-Optuna* sensibilidade",
                "n_figures": n}
    (MANIFESTS / "23_generate_figures_params.json").write_text(json.dumps(manifest, indent=2, default=str))
    log.info("Done: %d figures in figures/.", n)


if __name__ == "__main__":
    main()
