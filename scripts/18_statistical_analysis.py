#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 18_statistical_analysis.py

File type: Statistical analysis script (pipeline stage 18; Op3 requires the trained .keras models)

Purpose:
    Statistical analysis of the article (18 outputs, 6 models):
      Op1 : Wilcoxon signed-rank + Cohen's d for all pairs of models (NAE per output)
      Op2 : Pearson(NAE, ERG), test of H0: rho = 0
      Op3 : Permutation Feature Importance (PFI), 5 seeds x 5 repetitions x 48 features
             I_f = Delta NAE_f / NAE_baseline   (reference model = LSTM baseline, 14_lstm)

    Data sources:
      metrics/16_load_metrics_long.csv (nae_pct, e_true_kwh, e_pred_kwh per model x seed x output;
      systemic ERG per seed = |sum e_pred - sum e_true|/|sum e_true|*100); the 18 output labels
      come from the manifest via _nn_common; the windowed arrays via _nn_common.load_splits;
      LSTM models 14_lstm_seed_*; feature-selection manifest of script 11.

    Op3 (PFI) needs to load .keras models, so it runs on the GPU environment where the models
    were trained (TF 2.17). If the models cannot be loaded locally (Keras version skew) the PFI
    is SKIPPED with a warning; Op1/Op2 always run (they only depend on metrics).

    Inputs:  metrics/16_load_metrics_long.csv . 13_*.npy + scaler (via _nn_common) .
             manifests/11_feature_selection_rfecv_svr_kneedle_params.json . models/14_lstm_seed_*.keras
    Outputs: metrics/18_statistical_tests.csv . 18_model_comparison_summary.csv . 18_pfi_results.csv
             metrics/18_model_comparison_long.csv . 18_pfi_top_features_long.csv
             manifests/18_statistical_analysis_params.json  (figdata/ is populated downstream)

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
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon

import _nn_common as C

warnings.filterwarnings("ignore")

METRICS = C.METRICS_DIR
MANIFESTS = C.MANIFESTS_DIR
MODELS = C.MODELS_DIR
CKPT = C.PROJECT_ROOT / "checkpoints" / "18_pfi"
for d in (METRICS, MANIFESTS, CKPT, C.PROJECT_ROOT / "logs"):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(C.PROJECT_ROOT / "logs" / "18_statistical_analysis.log",
                                                  mode="w", encoding="utf-8")])
log = logging.getLogger(__name__)

N_REPS    = 5            # PFI repetitions per feature
E_MIN_KWH = 1.0          # active output
DELTA_T_HOURS = 1.0 / 60.0
PFI_REF   = ("LSTM", "14_lstm")   # reference model of the PFI
# Main statistical conclusion restricted to the 6 models of Block B; PE_ES_Optuna
# (intensive HPO search) is reported separately as sensitivity, without mixing.
PRIMARY_MODELS = ["LSTM", "RCNN_att", "PE_ES", "SPEC", "MoTE_v2", "Ensemble_G4"]


def effect_label(abs_d):
    # Effect-size labels (Cohen's d) in English, for the article.
    return "negligible" if abs_d < 0.2 else "small" if abs_d < 0.5 else "medium" if abs_d < 0.8 else "large"


def wilcoxon_with_d(a, b, label):
    a, b = np.array(a, float), np.array(b, float)
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    diffs = a - b
    if len(diffs[diffs != 0]) < 2:
        return {"label": label, "W": None, "p_value": None, "cohen_d": None,
                "abs_d": None, "effect": "ties", "sig": "—", "n_pairs": int(len(a)),
                "mean_diff": float(diffs.mean()) if len(diffs) else None}
    stat, p = wilcoxon(a, b, alternative="two-sided")
    d = diffs.mean() / (diffs.std() + 1e-12)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    log.info("  %-30s W=%6.0f p=%.4f %-3s d=%+.3f (%s)", label, stat, p, sig, d, effect_label(abs(d)))
    return {"label": label, "W": float(stat), "p_value": float(p), "cohen_d": float(d),
            "abs_d": float(abs(d)), "effect": effect_label(abs(d)), "sig": sig,
            "n_pairs": int(len(a)), "mean_diff": float(diffs.mean())}


def add_multiplicity_correction(rows, alpha=0.05):
    """Adds p_holm (Holm) and p_fdr_bh (Benjamini-Hochberg) to the paired tests,
    correcting for multiplicity over the family of evaluated pairs, plus sig_holm."""
    idx = [i for i, r in enumerate(rows) if r.get("p_value") is not None]
    if not idx:
        return rows
    p = np.asarray([rows[i]["p_value"] for i in idx], dtype=float)
    n = len(p)
    order = np.argsort(p, kind="mergesort")
    ps = p[order]
    holm = np.clip(np.maximum.accumulate((n - np.arange(n)) * ps), 0.0, 1.0)
    bh = np.clip(np.minimum.accumulate((n / np.arange(n, 0, -1)) * ps[::-1])[::-1], 0.0, 1.0)
    p_holm = np.empty(n); p_holm[order] = holm
    p_bh = np.empty(n); p_bh[order] = bh
    for k, i in enumerate(idx):
        rows[i]["p_holm"] = float(p_holm[k])
        rows[i]["p_fdr_bh"] = float(p_bh[k])
        rows[i]["sig_holm"] = bool(p_holm[k] < alpha)
    return rows


def feat_type(name):
    if any(x in name for x in ("hrm", "thd")):
        return "harmonic"
    if "dg" in name.lower():
        return "DG"
    return "fundamental"


# ══════════════════════════════════════════════════════════════════════════════
def op1_op2(labels):
    """Wilcoxon + Cohen d (Op1) and Pearson(NAE,ERG) (Op2) from 16_load_metrics_long."""
    lm_full = pd.read_csv(METRICS / "16_load_metrics_long.csv")
    # Tests (conclusion) restricted to the 6 main models; extras (Optuna) reported separately.
    lm = lm_full[lm_full.model.isin(PRIMARY_MODELS)].copy()
    extra = [m for m in sorted(lm_full.model.unique()) if m not in PRIMARY_MODELS]
    log.info("16_load_metrics_long: %s | main=%s | extra(sensitivity)=%s",
             lm_full.shape, sorted(lm.model.unique()), extra)

    def _sysrg(df):
        return (df.groupby(["model", "seed"]).apply(lambda g: pd.Series({
            "NAE_mean": g.loc[g.active, "nae_pct"].mean() if g.active.any() else g.nae_pct.mean(),
            "ERG_sys": abs(g.e_pred_kwh.sum() - g.e_true_kwh.sum()) / (abs(g.e_true_kwh.sum()) + 1e-9) * 100.0}),
            include_groups=False).reset_index())

    # -- Op1: paired Wilcoxon among the 6 main models (mean NAE per model x output)
    log.info("\n===== Op1: Wilcoxon + Cohen's d (6 main models) =====")
    nae_full = (lm_full.groupby(["model", "output"])["nae_pct"].mean().unstack("output").reindex(columns=labels))
    prim = [m for m in PRIMARY_MODELS if m in nae_full.index]
    wrows = []
    for m1, m2 in combinations(prim, 2):
        r = wilcoxon_with_d(nae_full.loc[m1].values, nae_full.loc[m2].values, f"{m1} × {m2}")
        r["model1"], r["model2"] = m1, m2
        wrows.append(r)
    add_multiplicity_correction(wrows)   # p_holm + p_fdr_bh + sig_holm over the 15 pairs
    pd.DataFrame(wrows).to_csv(METRICS / "18_statistical_tests.csv", index=False)
    log.info("Saved: metrics/18_statistical_tests.csv (%d main pairs)", len(wrows))

    # -- Op1 sensitivity: each extra model (Optuna) vs each main model, separately
    if extra:
        srows = []
        for mx in extra:
            for p in prim:
                r = wilcoxon_with_d(nae_full.loc[mx].values, nae_full.loc[p].values, f"{mx} × {p}")
                r["model1"], r["model2"], r["analysis_group"] = mx, p, "hpo_sensitivity"
                srows.append(r)
        add_multiplicity_correction(srows)   # correction within the sensitivity family (separately)
        pd.DataFrame(srows).to_csv(METRICS / "18_statistical_tests_hpo_sensitivity.csv", index=False)
        log.info("Saved: metrics/18_statistical_tests_hpo_sensitivity.csv (%d Optuna x main pairs)", len(srows))

    # -- Op2: Pearson(NAE, ERG), only the 6 main models (conclusion)
    log.info("\n===== Op2: Pearson(NAE, ERG) (6 main models) =====")
    sysrg = _sysrg(lm)
    r_val, p_val = pearsonr(sysrg.NAE_mean.values, sysrg.ERG_sys.values)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    log.info("  Pearson r(NAE,ERG)=%+.4f p=%.4f %s | H0:rho=0 -> %s (alpha=0.05) -> %s",
             r_val, p_val, sig, "rejected" if p_val < 0.05 else "not rejected",
             "correlated" if p_val < 0.05 else "independent")

    # Descriptive summary per model, ALL (with analysis_group) for figures/scatter
    sysrg_full = _sysrg(lm_full)
    summary = (sysrg_full.groupby("model").agg(NAE_mean=("NAE_mean", "mean"), NAE_std=("NAE_mean", "std"),
                                               ERG_mean=("ERG_sys", "mean"), ERG_std=("ERG_sys", "std"))
               .reset_index())
    summary["analysis_group"] = np.where(summary.model.isin(PRIMARY_MODELS), "primary", "hpo_sensitivity")
    summary = summary.sort_values(["analysis_group", "NAE_mean"])
    summary.to_csv(METRICS / "18_model_comparison_summary.csv", index=False)
    log.info("Saved: metrics/18_model_comparison_summary.csv")

    mne = summary.set_index("model")[["NAE_mean", "ERG_mean", "analysis_group"]].to_dict("index")
    comp = []
    for r in wrows:  # main pairs (with cohen_d)
        for m in (r["model1"], r["model2"]):
            comp.append({"comparison": r["label"], "cohen_d": r["cohen_d"], "model": m,
                         "NAE_mean": mne[m]["NAE_mean"], "ERG_mean": mne[m]["ERG_mean"],
                         "analysis_group": mne[m]["analysis_group"]})
    for m in extra:  # extra models (Optuna) without a pair, for the scatter as sensitivity
        comp.append({"comparison": None, "cohen_d": None, "model": m,
                     "NAE_mean": mne[m]["NAE_mean"], "ERG_mean": mne[m]["ERG_mean"],
                     "analysis_group": mne[m]["analysis_group"]})
    pd.DataFrame(comp).to_csv(METRICS / "18_model_comparison_long.csv", index=False)
    log.info("Saved: metrics/18_model_comparison_long.csv")
    return {"n_wilcoxon_pairs": len(wrows), "pearson_r": float(r_val), "pearson_p": float(p_val),
            "pearson_sig": sig, "pearson_interp": "correlated" if p_val < 0.05 else "independent"}


# ══════════════════════════════════════════════════════════════════════════════
def feat_names():
    # PFI is over the 48 SELECTED FEATURES (columns 0..47 of X, where
    # X = [X_mg(48 real) | X_dg(12 zero placeholders)]). The 12 DG columns are NOT
    # features (permuting a zero column changes nothing) -> excluded from the PFI.
    # list(candidate_sets[adopted]) = the 48, in the SAME order as the columns of X.
    cj = json.loads((C.DATA_DIR / "11_candidate_sets.json").read_text())
    return list(cj[cj["adopted"]])   # 48


def op3_pfi(data, labels):
    """PFI on the LSTM (14_lstm). Needs to load the models -> runs on the GPU environment; skipped locally if they cannot be loaded."""
    log.info("\n===== Op3: Permutation Feature Importance =====")
    FEAT = feat_names()
    mname, prefix = PFI_REF
    paths = [MODELS / f"{prefix}_seed_{s}.keras" for s in C.SEEDS]
    if not any(p.exists() for p in paths):
        log.warning("  Models %s missing; PFI SKIPPED (run where the models can be loaded).", prefix)
        return {"pfi": "skipped_no_models"}
    try:
        import tensorflow as tf  # noqa
        import keras
        C.make_masked_loss(C.phase_groups(labels))   # registers the custom loss
        models = []
        for p in paths:
            if p.exists():
                models.append(keras.models.load_model(str(p), safe_mode=False, compile=False))
        log.info("  LSTM models loaded: %d", len(models))
    except Exception as e:
        log.warning("  Failed to load models (%s); PFI SKIPPED (run where the models can be loaded).", e)
        return {"pfi": f"skipped_load_error: {e}"}

    X_te, scaler_Y, y_raw, mid = data["X_test"], data["scaler_Y"], data["Y_raw_test"], C.MID
    n_win = X_te.shape[0]
    e_test = np.abs(y_raw.sum(axis=0)) * DELTA_T_HOURS / 1000.0
    active = [j for j in range(len(labels)) if e_test[j] >= E_MIN_KWH]
    log.info("  X_test=%s active outputs=%d/%d", X_te.shape, len(active), len(labels))

    def nae_active(yp_W):
        yt = y_raw[mid:mid + n_win]
        vals = [abs(yp_W[:, j].sum() - yt[:, j].sum()) / (abs(yt[:, j].sum()) + 1e-9) * 100.0
                for j in active if abs(yt[:, j].sum()) > 1e-6]
        return float(np.mean(vals)) if vals else 0.0

    def nae_mean(X):
        return float(np.mean([nae_active(scaler_Y.inverse_transform(m.predict(X, verbose=0)[:, mid, :]))
                              for m in models]))

    baseline = nae_mean(X_te)
    log.info("  NAE baseline (LSTM, 5 seeds): %.4f%%", baseline)

    ck = CKPT / "18_pfi_checkpoint.json"
    imp = json.loads(ck.read_text()) if ck.exists() else {}
    if imp:
        log.info("  Checkpoint: %d/%d features already done", len(imp), len(FEAT))
    rng = np.random.default_rng(42)
    for fi, feat in enumerate(FEAT):
        if feat in imp:
            continue
        inc = []
        for _ in range(N_REPS):
            Xp = X_te.copy()
            Xp[:, :, fi] = X_te[rng.permutation(n_win), :, fi]
            inc.append(nae_mean(Xp) - baseline)
        imp[feat] = round(float(np.mean(inc)), 4)
        ck.write_text(json.dumps(imp))
        if fi % 10 == 0 or fi == len(FEAT) - 1:
            log.info("  [%2d/%d] %-28s DeltaNAE=%+.3f%%", fi + 1, len(FEAT), feat, imp[feat])

    rows = [{"feature": f, "delta_nae": d, "importance": round(d / baseline if abs(baseline) > 1e-6 else 0.0, 6),
             "feature_type": feat_type(f)} for f, d in imp.items()]
    pfi = pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
    pfi.to_csv(METRICS / "18_pfi_results.csv", index=False)
    pfi[["feature", "importance", "feature_type"]].head(20).to_csv(
        METRICS / "18_pfi_top_features_long.csv", index=False)
    log.info("Saved: metrics/18_pfi_results.csv + 18_pfi_top_features_long.csv (top 20)")
    log.info("  Top 5 PFI: %s", list(pfi.feature.head(5)))
    return {"baseline_nae_pct": baseline, "pfi_top_feature": pfi.iloc[0]["feature"],
            "pfi_top_I_f": float(pfi.iloc[0]["importance"]), "n_active": len(active)}


def main():
    log.info("=== 18_statistical_analysis.py ===")
    data = C.load_splits()
    labels = data["output_labels"]
    log.info("output_dim=%d", len(labels))
    res = {}
    res.update(op1_op2(labels))
    res.update(op3_pfi(data, labels))
    manifest = {"script": "18_statistical_analysis.py",
                "run_timestamp": pd.Timestamp.utcnow().isoformat(),
                "ex": "26_statistical_analysis.py",
                "parameters": {"seeds": C.SEEDS, "n_reps": N_REPS, "e_min_kwh": E_MIN_KWH,
                               "pfi_ref": PFI_REF[0], "pfi_formula": "I_f = ΔNAE_f / NAE_baseline"},
                "results": res}
    (MANIFESTS / "18_statistical_analysis_params.json").write_text(json.dumps(manifest, indent=2, default=str))
    log.info("Done. Manifest saved.")


if __name__ == "__main__":
    main()
