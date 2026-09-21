#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 11_feature_selection_rfecv_svr_kneedle.py

File type: Pipeline script (stage 11: feature selection RFECV-SVR + Kneedle)

Purpose:
    Feature selection RFECV-SVR + Kneedle.

    Runs linear RFECV-SVR per output on the training split and applies the Kneedle elbow on
    the normalised cumulative-importance curve to determine the diagnostic subsets of
    features.

    Methodology (MAIN variant):
      - X scaled on the training split (StandardScaler, no leakage);
      - Y scaled PER OUTPUT on the training split (StandardScaler) -> SVR coefficients
        comparable across loads of different magnitude;
      - TEMPORAL cross-validation (TimeSeriesSplit, n_splits=5) in the RFECV, with the folds
        defined on the ORIGINAL time grid and the target-validity mask applied inside each fold;
      - linear SVR with max_iter=20000 (convergence);
      - operational threshold: n* MG (>= IMPORTANCE_THRESHOLD cumulative) + 12 DG.
    The predominance of voltage / voltage-harmonic variables persisted in all variants
    (75-81%), confirming it as a finding rather than a scale artifact.

    Inputs:
      data/processed/10_original_features.parquet   X (M_G, 177 features)
      data/processed/09_R1.parquet                  Y (scenario R1; the M_G_* columns are
                                                    discarded: Y = only the loads Gamma_i)

    Outputs:
      audits/11_selected_features_by_output.csv
      audits/11_feature_ranking.csv
      audits/11_rfecv_scores.csv
      audits/11_kneedle_selection.csv
      audits/11_global_importance.csv
      data/processed/11_candidate_sets.json          (candidate sets for the ablation)
      figdata/11_feature_ranking_long.csv            consumed by script 23
      logs/11_feature_selection_rfecv_svr_kneedle.log
      manifests/11_feature_selection_rfecv_svr_kneedle_params.json

    Operations:
      1. Apply the selection on the training split only (chronological, no leakage).
      2. Run linear RFECV-SVR per output.
      3. Compute the performance curve per number of features.
      4. Apply Kneedle to determine n*.
      5. Rank by the absolute value of the SVR coefficients.
      6. Consolidate the candidate subsets (n* MG + 12 DG).
      7. Report n* and the total number of inputs.

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

import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # repository root (parent of scripts/)

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
AUDITS_DIR     = PROJECT_ROOT / "audits"
FIGDATA_DIR    = PROJECT_ROOT / "figdata"
LOGS_DIR       = PROJECT_ROOT / "logs"
MANIFESTS_DIR  = PROJECT_ROOT / "manifests"

IN_FEATURES = DATA_PROCESSED / "10_original_features.parquet"
IN_Y        = DATA_PROCESSED / "09_R1.parquet"
IN_CONF     = DATA_PROCESSED / "08_fill_confidence.parquet"   # target validity per point

# -- RFECV parameters -------------------------------------------------------------------
# MAIN SCOPE: OBSERVED_ONLY. The selection answers "which M_G variables explain the loads
# that were ACTUALLY MEASURED?", without selecting for the ease of predicting filled segments.
# The folds are defined on the ORIGINAL time grid of the training split; the validity mask is
# applied INSIDE each fold (the time line is not compressed). Scaling per fold (X and y).
SELECTION_SCOPE = os.environ.get("FS_SCOPE", "OBS")          # OBS | SUPPORTED | COMPLETE
CV_FOLDS     = 5
STEP         = 5
SCORING      = "neg_mean_absolute_error"
RANDOM_STATE = 42
N_MG         = 177   # features of the general meter (before selection)
MAX_ITER     = int(os.environ.get("FS_MAX_ITER", 200000))   # convergence of the linear SVR
MIN_TR_OBS, MIN_VAL_OBS = 50, 30                            # sufficient observed fold
IMPORTANCE_THRESHOLD = 0.95  # DIAGNOSTIC: smallest number of features whose normalised cumulative
                             # importance reaches >= 95% (n95). NOT the adopted subset.
N_DG         = 12    # auxiliary DG features (zeros)
TRAIN_RATIO  = 0.80  # chronological split, no leakage
N_REFERENCE  = 48    # previous reference (historical set), for the ablation
# output -> (meter, phase) mapping for the target-validity mask
G2METER = {"G1": 5, "G2": 6, "G3": 7, "G4": 11, "G5": 15, "G6": 16}
PH = {"A": "p_a", "B": "p_b", "C": "p_c"}

DG_FEATURE_NAMES = [f"dg_unit_{i:02d}" for i in range(1, N_DG + 1)]

for d in [DATA_PROCESSED, AUDITS_DIR, FIGDATA_DIR, LOGS_DIR, MANIFESTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(
            LOGS_DIR / "11_feature_selection_rfecv_svr_kneedle.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def reconstruct_n_features_steps(n_features_in: int, step: int,
                                  min_features: int, n_scores: int) -> np.ndarray:
    """Rebuilds the n_features axis for the RFECV score curve."""
    # RFECV stores the scores in increasing order of n_features
    # from min_features to n_features_in
    steps = list(range(min_features, n_features_in + 1, step))
    if not steps or steps[-1] != n_features_in:
        steps.append(n_features_in)
    # Adjust to the actual length
    if len(steps) != n_scores:
        steps = list(map(int, np.round(np.linspace(min_features, n_features_in, n_scores))))
    return np.array(steps[:n_scores], dtype=int)


def apply_kneedle(features_sorted: list, imp_values: np.ndarray,
                  n_adopted: int) -> dict:
    """
    Kneedle algorithm on the normalised cumulative-importance curve.
    Returns a dictionary with knee_idx, knee_n, pct_knee, pct_adopted.
    """
    n = len(imp_values)
    cumsum      = np.cumsum(imp_values)
    cumsum_norm = cumsum / cumsum[-1]
    x_norm      = np.linspace(0, 1, n)
    distances   = cumsum_norm - x_norm  # Kneedle: max distance to the diagonal
    knee_idx    = int(np.argmax(distances))
    n_knee      = knee_idx + 1
    pct_knee    = float(cumsum_norm[knee_idx] * 100)
    n_adopted = min(n_adopted, n)  # robustness: the union may be < n_adopted
    pct_adopted = float(cumsum_norm[n_adopted - 1] * 100)
    return {
        "knee_idx":    knee_idx,
        "knee_n":      n_knee,
        "pct_knee":    round(pct_knee, 2),
        "pct_adopted": round(pct_adopted, 2),
        "knee_feature": features_sorted[knee_idx],
        "cumsum_norm":  cumsum_norm,
        "x_norm":       x_norm,
        "distances":    distances,
    }


def main() -> None:
    log.info("=== 11_feature_selection_rfecv_svr_kneedle.py started ===")

    for f in [IN_FEATURES, IN_Y]:
        if not f.exists():
            log.error(f"Input not found: {f}")
            sys.exit(1)

    # -- 1. Load X and Y -------------------------------------------------------------------
    log.info("[1] Loading X and Y ...")
    X_df = pd.read_parquet(IN_FEATURES)
    Y_df = pd.read_parquet(IN_Y)

    # 09_R1 is consolidated (M_G + loads). For the selection, Y = only the loads Gamma_i;
    # the columns of the general meter (M_G_*) are the input X, not targets.
    mg_cols = [c for c in Y_df.columns if str(c).startswith("M_G")]
    if mg_cols:
        Y_df = Y_df.drop(columns=mg_cols)
        log.info(f"  Y: discarded {len(mg_cols)} M_G_* columns (input, not target): {mg_cols}")

    # Normalises the indices to UTC tz-aware: script 09 writes a tz-aware index and script 08
    # writes tz-naive (same UTC instants). Without this, the index intersection is empty.
    X_df.index = pd.to_datetime(X_df.index, utc=True)
    Y_df.index = pd.to_datetime(Y_df.index, utc=True)

    # Align the indices (both indexed by time)
    common_idx = X_df.index.intersection(Y_df.index)
    X_df = X_df.loc[common_idx]
    Y_df = Y_df.loc[common_idx]
    log.info(f"  X shape: {X_df.shape}  Y shape: {Y_df.shape}")

    feature_names = list(X_df.columns)
    output_names  = list(Y_df.columns)
    n_samples     = len(X_df)
    assert len(feature_names) == N_MG, (
        f"Esperadas {N_MG} features MG, encontradas {len(feature_names)}"
    )
    log.info(f"  Features MG: {N_MG}  |  Outputs: {len(output_names)}")

    # -- 2. Chronological split (train = first 80%) ----------------------------------------
    train_end = int(n_samples * TRAIN_RATIO)
    X_train = X_df.iloc[:train_end].values.astype(np.float64)
    Y_train = Y_df.iloc[:train_end].values.astype(np.float64)   # NOT globally scaled
    log.info(f"[2] Split: train={train_end}  test={n_samples - train_end}")

    # -- Target-validity mask per output + folds on the ORIGINAL grid --------------------
    conf = pd.read_parquet(IN_CONF)                             # 0=OBS,1=SHORT,2=MEDIUM,3=LONG,6=UNSUP
    def valid_mask(label):
        k, ph = label.split("_")
        cm = conf[conf.meter_id == G2METER[k]].sort_values("time")[PH[ph]].to_numpy()[:n_samples][:train_end]
        if SELECTION_SCOPE == "OBS":       return cm == 0
        if SELECTION_SCOPE == "SUPPORTED": return np.isin(cm, [0, 1, 2])
        return np.ones(train_end, dtype=bool)
    full_time = np.arange(train_end)
    full_folds = list(TimeSeriesSplit(n_splits=CV_FOLDS).split(full_time))
    def masked_cv(mask):                                       # folds of the original grid, masked per output
        ot = full_time[mask]; cv = []
        for a, b in full_folds:
            to = np.flatnonzero(np.isin(ot, a)); vo = np.flatnonzero(np.isin(ot, b))
            if len(to) >= MIN_TR_OBS and len(vo) >= MIN_VAL_OBS:
                cv.append((to, vo))
        return cv
    def make_est():                                            # scaling of X and y PER FOLD (no leakage)
        return TransformedTargetRegressor(
            regressor=Pipeline([("sc", StandardScaler()),
                                ("svr", SVR(kernel="linear", max_iter=MAX_ITER))]),
            transformer=StandardScaler())
    def imp_getter(est):
        return np.abs(est.regressor_.named_steps["svr"].coef_).ravel()
    log.info(f"  Scope={SELECTION_SCOPE} | folds on the original grid ({CV_FOLDS}) | "
             f"scaling per fold | SVR max_iter={MAX_ITER}")

    # -- 3. RFECV-SVR per output (valid targets, corrected CV) -----------------------------
    log.info("[3] RFECV-SVR per output (scope=%s) ...", SELECTION_SCOPE)
    n_outputs       = Y_train.shape[1]
    selected_flags  = np.zeros((n_outputs, N_MG), dtype=bool)
    importances     = np.zeros((n_outputs, N_MG), dtype=float)
    conv_flags      = {}
    scores_by_output = []

    t0 = time.time()
    for i, label in enumerate(output_names):
        t_i = time.time()
        mk = valid_mask(label); cv = masked_cv(mk)
        if len(cv) < 3:
            log.warning(f"  [{i+1:2d}/{n_outputs}] {label}: insufficient folds "
                        f"({len(cv)} valid, obs={int(mk.sum())}); ignored in the selection")
            conv_flags[label] = "sem_folds"; continue
        Xo = X_train[mk]; yo = Y_train[mk, i]
        rfecv = RFECV(make_est(), step=STEP, cv=cv, scoring=SCORING,
                      min_features_to_select=1, importance_getter=imp_getter, n_jobs=-1)
        rfecv.fit(Xo, yo)
        selected_flags[i] = rfecv.support_
        full_imp = np.zeros(N_MG, dtype=float)
        full_imp[rfecv.support_] = imp_getter(rfecv.estimator_)
        importances[i] = full_imp
        n_it = rfecv.estimator_.regressor_.named_steps["svr"].n_iter_
        conv_flags[label] = "sim" if np.all(np.ravel(n_it) < MAX_ITER) else "NAO"
        cv_scores = rfecv.cv_results_["mean_test_score"]
        nf_steps  = reconstruct_n_features_steps(
            n_features_in=rfecv.n_features_in_, step=STEP, min_features=1,
            n_scores=len(cv_scores))
        scores_by_output.append((label, nf_steps, cv_scores))
        n_sel   = int(rfecv.support_.sum()); elapsed = time.time() - t_i
        eta = (time.time() - t0) / (i + 1) * (n_outputs - i - 1) if i > 0 else 0
        log.info(f"  [{i+1:2d}/{n_outputs}] {label}: {n_sel} features  "
                 f"(obs={int(mk.sum())} folds={len(cv)} conv={conv_flags[label]} "
                 f"{elapsed:.0f}s | ETA {eta/60:.0f}min)")

    # -- 4. Union of the selected features -------------------------------------------------
    log.info("[4] Computing the union mask ...")
    union_mask = selected_flags.any(axis=0)
    n_union    = int(union_mask.sum())
    log.info(f"  Union: {n_union} features out of {N_MG}")

    selected_feature_names = [feature_names[j] for j in range(N_MG) if union_mask[j]]

    # Importance per output, restricted to the features of the union
    imp_union = importances[:, union_mask]   # (n_outputs, n_union)

    # Frequency: how many outputs selected each feature of the union
    freq_union = selected_flags[:, union_mask].sum(axis=0)  # (n_union,)

    # -- 5. Cumulative-importance curve and Kneedle ----------------------------------------
    log.info("[5] Applying Kneedle ...")
    # Sort the features of the union by decreasing mean importance
    mean_imp_union = imp_union.mean(axis=0)           # (n_union,)
    sort_idx       = np.argsort(mean_imp_union)[::-1]  # descending

    features_sorted = [selected_feature_names[j] for j in sort_idx]
    imp_values      = mean_imp_union[sort_idx]
    freq_sorted     = freq_union[sort_idx]

    # DIAGNOSTICS (not the adopted subset): n95 and Kneedle
    _cumnorm = np.cumsum(imp_values) / np.sum(imp_values)
    n95 = min(int(np.searchsorted(_cumnorm, IMPORTANCE_THRESHOLD) + 1), len(imp_values))
    kneedle = apply_kneedle(features_sorted, imp_values, n95)
    knee_n = int(kneedle["knee_n"])
    log.info(f"  DIAGNOSTICS: n_union={n_union}  Kneedle=#{knee_n} ({kneedle['pct_knee']:.1f}%)  "
             f"n95={n95} ({kneedle['pct_adopted']:.1f}%)  knee='{kneedle['knee_feature']}'")

    # -- 6. Candidate sets for the ABLATION (11/19/29/48/177) ------------------------------
    log.info("[6] Generating candidate sets for the ablation ...")
    from sklearn.svm import LinearSVR                          # fast, only to ORDER the reference-48
    glob_imp = np.zeros(N_MG)                                   # global ranking (all 177) for 48 and 177
    for i, label in enumerate(output_names):
        mk = valid_mask(label)
        if int(mk.sum()) < MIN_TR_OBS:
            continue
        Xo = StandardScaler().fit_transform(X_train[mk])
        yo = StandardScaler().fit_transform(Y_train[mk, i:i + 1]).ravel()
        lsvr = LinearSVR(max_iter=20000, random_state=RANDOM_STATE).fit(Xo, yo)
        glob_imp += np.abs(lsvr.coef_).ravel()
    glob_order = [feature_names[j] for j in np.argsort(glob_imp)[::-1]]
    # persists the GLOBAL importance of the 177 (basis of reference_48 and of adopted_importance_pct in script 24)
    _ord = np.argsort(glob_imp)[::-1]
    _gtot = float(glob_imp.sum()) or 1.0
    _gdf = pd.DataFrame({"glob_rank": np.arange(1, N_MG + 1),
                         "feature": [feature_names[j] for j in _ord],
                         "glob_imp": glob_imp[_ord],
                         "glob_imp_pct": 100.0 * glob_imp[_ord] / _gtot})
    _gdf["glob_cumpct"] = _gdf["glob_imp_pct"].cumsum()
    _gdf.to_csv(AUDITS_DIR / "11_global_importance.csv", index=False)
    log.info("  Saved: audits/11_global_importance.csv (global importance of the %d)", N_MG)
    candidate_sets = {                                         # keys with the ACTUAL count
        f"kneedle_{knee_n}":         features_sorted[:knee_n],
        f"n95_{n95}":                features_sorted[:n95],
        f"union_{n_union}":          list(features_sorted),
        f"reference_{N_REFERENCE}":  glob_order[:N_REFERENCE],
        f"all_{N_MG}":               list(feature_names),
    }
    json.dump(candidate_sets, open(DATA_PROCESSED / "11_candidate_sets.json", "w"),
              indent=1, ensure_ascii=False)
    log.info("  Saved: data/processed/11_candidate_sets.json (5 sets)")
    for _k, _v in candidate_sets.items():
        log.info(f"    {_k}: {len(_v)} features")

    # PROVISIONAL operational reference = UNION; FINAL adoption pending on the ablation
    provisional_reference = "UNION_OBS"
    n_adopted = n_union                        # provisional number (plumbing of audits/figdata)
    features_adopted = list(features_sorted)   # = ordered union (provisional, NOT final)
    all_model_features = features_adopted + DG_FEATURE_NAMES
    n_model = len(all_model_features)
    log.info(f"  provisional_reference=UNION_OBS ({n_union} MG + {N_DG} DG = {n_model}) | "
             f"n_adopted=PENDING_ABLATION")

    # -- 7. Verification -------------------------------------------------------------------
    log.info("[7] Verification:")
    log.info(f"  Input:    {N_MG} MG features")
    log.info(f"  Union:    {n_union} features (provisional reference; RFECV over {n_outputs} outputs)")
    log.info(f"  Diagnostics: Kneedle=#{knee_n}  n95={n95}")
    log.info(f"  Adoption: PENDING on the ablation (11/19/29/48/177) + {N_DG} auxiliary DG")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # AUDITS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # -- audit A: selected features per output ---------------------------------------------
    sel_rows = []
    for i, label in enumerate(output_names):
        sel_feat = [feature_names[j] for j in range(N_MG) if selected_flags[i, j]]
        for feat in sel_feat:
            sel_rows.append({"output": label, "feature": feat})
    sel_df = pd.DataFrame(sel_rows)
    sel_df.to_csv(AUDITS_DIR / "11_selected_features_by_output.csv", index=False)
    log.info(f"Saved: audits/11_selected_features_by_output.csv  ({len(sel_df)} rows)")

    # -- audit B: feature ranking (union, sorted by importance) ----------------------------
    rank_rows = []
    for rank, (feat, imp, freq) in enumerate(
        zip(features_sorted, imp_values, freq_sorted), start=1
    ):
        rank_rows.append({
            "rank":         rank,
            "feature":      feat,
            "mean_imp":     round(float(imp), 6),
            "frequency":    int(freq),
            "pct_cumul":    round(float(kneedle["cumsum_norm"][rank - 1] * 100), 2),
            "kneedle_dist": round(float(kneedle["distances"][rank - 1]), 6),
            "in_kneedle":   rank <= knee_n,
            "in_n95":       rank <= n95,
            "in_union":     True,   # all features of the ranking belong to the union
        })
    rank_df = pd.DataFrame(rank_rows)
    rank_df.to_csv(AUDITS_DIR / "11_feature_ranking.csv", index=False)
    log.info(f"Saved: audits/11_feature_ranking.csv  ({len(rank_df)} rows)")

    # -- audit C: RFECV scores per output --------------------------------------------------
    score_rows = []
    for label, nf_steps, cv_scores in scores_by_output:
        for nf, sc in zip(nf_steps, cv_scores):
            score_rows.append({
                "output":      label,
                "n_features":  int(nf),
                "mae_cv":      round(float(-sc), 6),  # positive
            })
    score_df = pd.DataFrame(score_rows)
    score_df.to_csv(AUDITS_DIR / "11_rfecv_scores.csv", index=False)
    log.info(f"Saved: audits/11_rfecv_scores.csv  ({len(score_df)} rows)")

    # -- audit D: selection summary (diagnostics + provisional status) ---------------------
    ka_df = pd.DataFrame([{
        "selection_scope":       SELECTION_SCOPE,
        "n_union":               n_union,
        "knee_n":                knee_n,
        "knee_pct":              kneedle["pct_knee"],
        "knee_feature":          kneedle["knee_feature"],
        "n95":                   n95,
        "n95_pct":               kneedle["pct_adopted"],
        "provisional_reference": provisional_reference,
        "n_adopted":             "PENDING_ABLATION",
        "n_convergiu":           sum(1 for v in conv_flags.values() if v == "sim"),
        "n_nao_convergiu":       sum(1 for v in conv_flags.values() if v == "NAO"),
        "n_dg":                  N_DG,
    }])
    ka_df.to_csv(AUDITS_DIR / "11_kneedle_selection.csv", index=False)
    conv_df = pd.DataFrame([{"output": k, "convergiu": v} for k, v in conv_flags.items()])
    conv_df.to_csv(AUDITS_DIR / "11_convergence_by_output.csv", index=False)
    log.info("Saved: audits/11_kneedle_selection.csv + 11_convergence_by_output.csv")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FIGDATA: long format for script 23
    # Columns: record_type, output, n_features, mae_cv, feature, frequency, coef_abs
    # fig09_rfecv_kneedle uses: n_features, mae_cv, output, feature, frequency
    # fig27_feature_heatmap uses: feature, output, coef_abs
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    log.info("Building figdata/11_feature_ranking_long.csv ...")

    fig_rows = []

    # Section A: MAE-RFECV curve per output
    for label, nf_steps, cv_scores in scores_by_output:
        for nf, sc in zip(nf_steps, cv_scores):
            fig_rows.append({
                "record_type": "mae_curve",
                "output":      label,
                "n_features":  int(nf),
                "mae_cv":      round(float(-sc), 6),
                "feature":     None,
                "frequency":   None,
                "coef_abs":    None,
            })

    # Section B: feature importance per output (union, sorted)
    # feature_to_freq: global lookup
    feature_to_freq = {
        features_sorted[k]: int(freq_sorted[k]) for k in range(len(features_sorted))
    }
    # For each output, for each feature of the union: coef_abs and frequency
    for i, label in enumerate(output_names):
        for k, feat in enumerate(features_sorted):
            # importances[i] indexed by the original feature_names
            feat_orig_idx = feature_names.index(feat)
            coef = float(importances[i, feat_orig_idx])
            fig_rows.append({
                "record_type": "feature_importance",
                "output":      label,
                "n_features":  None,
                "mae_cv":      None,
                "feature":     feat,
                "frequency":   int(feature_to_freq[feat]),
                "coef_abs":    round(coef, 8),
            })

    fig_df = pd.DataFrame(fig_rows)
    out_fig = FIGDATA_DIR / "11_feature_ranking_long.csv"
    fig_df.to_csv(out_fig, index=False)
    log.info(f"Saved: figdata/11_feature_ranking_long.csv  shape={fig_df.shape}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # MANIFEST
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    manifest = {
        "script":        "11_feature_selection_rfecv_svr_kneedle.py",
        "run_timestamp": datetime.now().isoformat(),
        "methodology": {
            "selection_scope":     SELECTION_SCOPE,
            "target_validity":     "selection restricted to the instants with ORIGINALLY observed target (OBS)",
            "X_scaling":           "StandardScaler POR FOLD (Pipeline; sem vazamento)",
            "Y_scaling":           "StandardScaler POR FOLD (TransformedTargetRegressor)",
            "cross_validation":    "folds on the ORIGINAL TIME GRID; mask applied inside each fold",
            "estimator":           f"SVR linear, max_iter={MAX_ITER}",
            "consolidation":       "union of the per-output subsets (provisional reference)",
            "kneedle_role":        "diagnostic; does NOT define the adopted subset",
            "n95_role":            "diagnostic (smallest n with >=95% cumulative importance); NOT adopted",
            "adoption":            "PENDING on the chronological ablation (11/19/29/48/177)",
        },
        "parameters": {
            "cv_folds":     CV_FOLDS,
            "cv_type":      "TimeSeriesSplit (original grid) + mask per output",
            "svr_max_iter": MAX_ITER,
            "step":         STEP,
            "scoring":      SCORING,
            "random_state": RANDOM_STATE,
            "n_mg":         N_MG,
            "n_dg":         N_DG,
            "train_ratio":  TRAIN_RATIO,
            "min_tr_obs":   MIN_TR_OBS,
            "min_val_obs":  MIN_VAL_OBS,
        },
        "results": {
            "n_samples":         n_samples,
            "train_end":         train_end,
            "n_outputs":         n_outputs,
            "n_union":           n_union,
            "n_kneedle":         knee_n,
            "knee_pct":          kneedle["pct_knee"],
            "n95":               n95,
            "n95_pct":           kneedle["pct_adopted"],
            "provisional_reference": provisional_reference,
            "n_adopted":         "PENDING_ABLATION",
            "n_convergiu":       sum(1 for v in conv_flags.values() if v == "sim"),
            "n_nao_convergiu":   sum(1 for v in conv_flags.values() if v == "NAO"),
            "candidate_set_sizes": {k: len(v) for k, v in candidate_sets.items()},
        },
        "inputs": {
            "features": str(IN_FEATURES),
            "Y":        str(IN_Y),
        },
        "inputs_extra": {"confidence": str(IN_CONF)},
        "outputs": {
            "candidate_sets": "data/processed/11_candidate_sets.json",
            "audits": [
                "audits/11_selected_features_by_output.csv",
                "audits/11_feature_ranking.csv",
                "audits/11_rfecv_scores.csv",
                "audits/11_kneedle_selection.csv",
                "audits/11_convergence_by_output.csv",
            ],
            "figdata": ["figdata/11_feature_ranking_long.csv"],
        },
    }
    mp = MANIFESTS_DIR / "11_feature_selection_rfecv_svr_kneedle_params.json"
    with open(mp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False, default=str)
    log.info(f"Manifest saved: {mp}")

    log.info("")
    log.info("=== SUMMARY ===")
    log.info(f"  Scope:    {SELECTION_SCOPE} (observed targets)")
    log.info(f"  Input:    {N_MG} MG features x {n_outputs} outputs")
    log.info(f"  Union:    {n_union} features (provisional reference)")
    log.info(f"  Kneedle:  #{knee_n} ({kneedle['pct_knee']:.1f}%)  [diagnostic]")
    log.info(f"  n95:      {n95} ({kneedle['pct_adopted']:.1f}%)  [diagnostic]")
    log.info(f"  Adopted:  PENDING (ablation 11/19/29/48/177)")
    log.info(f"  Convergence: {sum(1 for v in conv_flags.values() if v=='sim')}/{n_outputs}")
    log.info(f"  Total time: {(time.time() - t0) / 60:.1f} min")
    log.info("=== Done ===")


if __name__ == "__main__":
    main()
