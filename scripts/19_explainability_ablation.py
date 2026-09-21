#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 19_explainability_ablation.py

File type: Pipeline script (stage 19: ablation of electrical feature families + controls; surrogate model)

Purpose:
    Ablation of electrical feature families (two levels) + controls.

    EXPLAINABILITY runner. Consumes the input-block map produced by script 12
    (`audits/12_input_blocks_from_02b.csv`) and measures the effect of each electrical family G
    on the disaggregation error, with chronological validation.

    DESIGN (v2, without proxies). The channel audit of v2 did NOT classify any channel as a
    proxy (is_proxy=0). Therefore the proxy-based analysis (X1 = X0+proxies; X2 = X0+residuals)
    is NOT APPLICABLE and is reported as N/A, never as a numerical result. The families use ALL
    their features (without the `& is_proxy` filter, which produced empty sets).

    Two levels of analysis (12 auxiliary columns kept CONSTANT in every configuration):
      Main: restricted to the adopted model S48 (48 features), F_g^(48) = G_g intersect S48:
          . Addition    A_g   = X0 union F_g^(48)     (marginal value of the family over the base)
          . Removal     D_g   = S48 \\ F_g^(48)        (loss when the family is removed from the final model)
      Supplementary: informational capacity with the raw features of the 177:
          . A_g^(177) = X0 union F_g^(177)            (NOT an ablation of the final model; reintroduces
                                                      attributes rejected in the selection)
      X0 = electrical base (powers G_P) + 12 auxiliary columns.

    Controls (anti-artifact, proxy-independent):
      A0t   = X0 + normalised time index (spurious clock)
      Aperm = S48 with the non-base features PERMUTED in time (destroys the signal -> confirms
              genuine dependence; NAE must worsen)

    Criterion: addition/supplement, keep if Delta J = J(A_g) - J(X0) <= -delta_J; removal,
      family IMPORTANT if Delta J = J(D_g) - J(S48) > delta_J. delta_J = 0 (main); sensitivity 0.01.

    MANDATORY partition audit (aborts on failure): X0 union G_I union G_V union G_THD union
      G_hV union G_hI union G_sec = 177, no feature in two families, no orphan feature,
      q/s/cos (G_sec) present.

    SURROGATE: HistGradientBoostingRegressor (fast, CPU) as a consistent instrument across
    configurations. Confirmation with the deep models (Block B) is a separate GPU run.
    SMOKE mode (env SMOKE=1): subsamples and reduces iterations/seeds.

    Inputs:
        data/processed/10_original_features.parquet   (X: 177 features of M_G)
        data/processed/09_R1.parquet                  (Y: 18 loads; discards M_G_*)
        data/processed/11_candidate_sets.json         (adopted set S48)
        audits/12_input_blocks_from_02b.csv           (family map)
    Outputs:
        metrics/19_ablation_blocks.csv
        figdata/19_ablation_blocks_long.csv
        manifests/19_explainability_ablation_params.json
        logs/19_explainability_ablation.log

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
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

SCRIPT_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT  = SCRIPT_DIR.parent  # repository root (parent of scripts/)
PROCESSED     = PROJECT_ROOT / "data" / "processed"
AUDITS_DIR    = PROJECT_ROOT / "audits"
METRICS_DIR   = PROJECT_ROOT / "metrics"
FIGDATA_DIR   = PROJECT_ROOT / "figdata"
LOGS_DIR      = PROJECT_ROOT / "logs"
MANIFESTS_DIR = PROJECT_ROOT / "manifests"

X_FILE   = PROCESSED / "10_original_features.parquet"
Y_FILE   = PROCESSED / "09_R1.parquet"
S48_FILE = PROCESSED / "11_candidate_sets.json"
BLOCKS   = AUDITS_DIR / "12_input_blocks_from_02b.csv"

SEEDS      = [42, 123, 456, 789, 1024]
TRAIN_FRAC = 0.80
POWER_COLS = ["p_a", "p_b", "p_c"]     # electrical base G_P (X0)
N_AUX      = 12                         # auxiliary variables (zero placeholders), constant
DELTA_J    = 0.0
DELTA_J_SENS = 0.01
FAMILIES_ALL = ["G_P", "G_I", "G_V", "G_THD", "G_hV", "G_hI", "G_sec"]
ADD_FAM      = ["G_I", "G_V", "G_THD", "G_hV", "G_hI", "G_sec"]   # G_P is already X0
LABEL_A = {"G_I": "A1(+currents)", "G_V": "A2(+voltages)", "G_THD": "A5(+THD)",
           "G_hV": "A4(+harm.tens)", "G_hI": "A3(+harm.corr)", "G_sec": "A5b(+q/s/cos)"}

# Methodological lock: if an analysis ever REQUIRES proxies and there are none, abort.
ANALYSIS_REQUIRES_PROXY = False

SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    SEEDS = [42, 123]
    ROW_CAP, MAX_ITER = 3000, 40
else:
    ROW_CAP, MAX_ITER = None, 200


def setup_logger(p: Path) -> logging.Logger:
    p.parent.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("19_ablation"); lg.setLevel(logging.DEBUG); lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(p, mode="w", encoding="utf-8"); fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt)
    lg.addHandler(fh); lg.addHandler(ch); return lg


def nae_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean point-wise NAE over the outputs: mean_j sum|y_hat - y| / sum|y| (%)."""
    num = np.nansum(np.abs(y_pred - y_true), axis=0)
    den = np.nansum(np.abs(y_true), axis=0)
    per = np.where(den > 0, num / den, np.nan)
    return float(np.nanmean(per) * 100.0)


def erg_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Systemic ERG: |sum E_hat - sum E| / |sum E| (%), summing outputs and time."""
    e_true = np.nansum(y_true); e_pred = np.nansum(y_pred)
    return float(abs(e_pred - e_true) / abs(e_true) * 100.0) if e_true != 0 else np.nan


def fit_eval(Xtr, Ytr, Xte, Yte, seed) -> tuple:
    preds = np.full_like(Yte, np.nan, dtype=float)
    for j in range(Ytr.shape[1]):
        yj = Ytr[:, j]; ok = np.isfinite(yj)
        if ok.sum() < 50:
            continue
        m = HistGradientBoostingRegressor(max_iter=MAX_ITER, random_state=seed)
        m.fit(Xtr[ok], yj[ok])
        preds[:, j] = m.predict(Xte)
    return nae_percent(Yte, preds), erg_percent(Yte, preds)


def audit_partition(bl: pd.DataFrame, log) -> dict:
    """MANDATORY audit of the partition of the 177 features by family (aborts if invalid)."""
    fam = bl["family_G"]
    counts = {g: int((fam == g).sum()) for g in FAMILIES_ALL}
    n_sum = sum(counts.values()); n_total = len(bl)
    n_orphan = int((~fam.isin(FAMILIES_ALL)).sum())
    if n_sum != n_total or n_orphan != 0:
        raise RuntimeError(
            f"INVALID family partition: sum={n_sum} orphans={n_orphan} total={n_total} "
            f"(expected sum==total and 0 orphans). Counts={counts}")
    if counts["G_sec"] < 9:
        raise RuntimeError(f"Family G_sec (q/s/cos) incomplete: {counts['G_sec']}<9.")
    # no feature in two families: value_counts already guarantees a disjoint partition (one row/feature),
    # but we confirm that there is no duplicated feature in the map.
    if bl["feature"].duplicated().any():
        raise RuntimeError("Block map has duplicated features: non-disjoint partition.")
    log.info("Partition audit OK: %s | sum=%d==%d | orphans=0 | disjoint", counts, n_sum, n_total)
    return counts


def main() -> None:
    log = setup_logger(LOGS_DIR / "19_explainability_ablation.log")
    log.info("=== 19_explainability_ablation.py: family ablation (%s) ===",
             "SMOKE" if SMOKE else "full")
    for p in (X_FILE, Y_FILE, S48_FILE, BLOCKS):
        if not p.exists():
            log.error(f"Missing input: {p}"); sys.exit(1)

    X = pd.read_parquet(X_FILE).reset_index(drop=True)
    Ydf = pd.read_parquet(Y_FILE).reset_index(drop=True)
    yc = [c for c in Ydf.columns if not c.startswith("M_G")]
    n = min(len(X), len(Ydf))
    if ROW_CAP:
        n = min(n, ROW_CAP)
    X = X.iloc[:n].copy(); Y = Ydf[yc].iloc[:n].to_numpy(float)
    log.info(f"X={X.shape} Y={Y.shape} (outputs: {len(yc)})")

    bl = pd.read_csv(BLOCKS)
    feat = list(X.columns)
    idx = {f: i for i, f in enumerate(feat)}

    # -- MANDATORY partition audit ---------------------------------------------------------
    counts = audit_partition(bl, log)

    # -- proxy: NOT APPLICABLE in v2 (no channel classified as proxy) ------------------------
    n_proxy = int(bl["is_proxy"].sum()) if "is_proxy" in bl.columns else 0
    proxy_status = "NOT_APPLICABLE" if n_proxy == 0 else "APPLICABLE"
    if ANALYSIS_REQUIRES_PROXY and n_proxy == 0:
        raise RuntimeError(
            "Proxy-based analysis is not applicable: no channels were classified as proxies.")
    log.info("proxy_analysis_status=%s | n_proxy_channels=%d -> X1/X2 = N/A", proxy_status, n_proxy)

    # -- adopted set S48 and per-family subsets ----------------------------------------------
    cj = json.loads(S48_FILE.read_text())
    S48 = [f for f in cj[cj["adopted"]] if f in idx]           # 48 adopted features present
    s48_i = [idx[f] for f in S48]
    s48_set = set(s48_i)
    power_i = [idx[c] for c in POWER_COLS if c in idx]         # electrical base (G_P)

    def fam_idx(g, restrict=None):
        fs = bl.loc[bl["family_G"] == g, "feature"]
        if restrict is not None:
            fs = [f for f in fs if f in restrict]
        return [idx[f] for f in fs if f in idx]

    S48names = set(S48)
    fg48 = {g: fam_idx(g, S48names) for g in FAMILIES_ALL}     # F_g^(48)
    fg177 = {g: fam_idx(g) for g in FAMILIES_ALL}             # F_g^(177)
    if sum(len(v) for v in fg48.values()) != len(s48_i):
        raise RuntimeError("Sum|F_g intersect S48| != |S48|: per-family subset inconsistent with the adopted set.")
    log.info("S48=%d | F_g^(48)=%s", len(s48_i), {g: len(fg48[g]) for g in FAMILIES_ALL})

    Xraw = X.to_numpy(float)
    split = int(n * TRAIN_FRAC)
    tr, te = slice(0, split), slice(split, n)
    tnorm = (np.arange(n, dtype=float) / (n - 1)).reshape(-1, 1)
    AUX = np.zeros((n, N_AUX), dtype=float)   # 12 auxiliary columns CONSTANT in all configurations

    def mat(cols_idx):
        cols_idx = list(cols_idx)
        base = Xraw[:, cols_idx] if cols_idx else np.empty((n, 0))
        return np.column_stack([base, AUX])

    rng = np.random.default_rng(0)
    perm = rng.permutation(n)

    # -- set of configurations (name, family, analysis, cols_idx) ------------------------------
    configs = [("S48_ref", "-", "reference", s48_i),
               ("X0", "G_P", "base", power_i)]
    for g in ADD_FAM:                                          # addition (48)
        configs.append((f"add48:{g}", g, "addition_48", sorted(set(power_i) | set(fg48[g]))))
    for g in FAMILIES_ALL:                                     # removal (48)
        configs.append((f"rem48:{g}", g, "removal_48",
                        [i for i in s48_i if i not in set(fg48[g])]))
    for g in ADD_FAM:                                          # supplement (177)
        configs.append((f"add177:{g}", g, "supplement_177", sorted(set(power_i) | set(fg177[g]))))
    configs += [("A0t", "-", "control", None), ("Aperm", "-", "control", None)]

    def build(name, cols_idx):
        if name == "A0t":
            return np.column_stack([mat(power_i), tnorm])
        if name == "Aperm":
            non_base = [i for i in s48_i if i not in set(power_i)]
            Xp = Xraw.copy()
            Xp[:, non_base] = Xraw[np.ix_(perm, non_base)]
            return np.column_stack([Xp[:, s48_i], AUX])
        return mat(cols_idx)

    long_rows, summary, nae_by = [], [], {}
    for name, family, analysis, cols_idx in configs:
        M = build(name, cols_idx)
        naes, ergs = [], []
        for s in SEEDS:
            nae, erg = fit_eval(M[tr], Y[tr], M[te], Y[te], s)
            naes.append(nae); ergs.append(erg)
            long_rows.append({"block": name, "family": family, "analysis": analysis,
                              "seed": s, "NAE": nae, "ERG": erg, "n_features": M.shape[1]})
        nae_m = float(np.mean(naes)); nae_sd = float(np.std(naes)); erg_m = float(np.mean(ergs))
        nae_by[name] = nae_m
        summary.append({"block": name, "family": family, "analysis": analysis,
                        "rotulo": LABEL_A.get(family, name), "n_features": int(M.shape[1]),
                        "NAE_mean": round(nae_m, 4), "NAE_std": round(nae_sd, 4),
                        "ERG_mean": round(erg_m, 4)})
        log.info("  %-14s [%-14s] NAE=%.3f+/-%.3f ERG=%.3f (feat=%d)",
                 name, analysis, nae_m, nae_sd, erg_m, M.shape[1])

    # proxy-based X1/X2: explicit N/A (never numerical)
    for nm, desc in [("X1", "X0+proxies (N/A)"), ("X2", "X0+proxy residuals (N/A)")]:
        summary.append({"block": nm, "family": "-", "analysis": "proxy_NA", "rotulo": desc,
                        "n_features": 0, "NAE_mean": None, "NAE_std": None, "ERG_mean": None})

    # -- Delta J and decision per analysis type -----------------------------------------------
    base_x0 = nae_by["X0"]; base_s48 = nae_by["S48_ref"]

    def decide_row(r):
        a = r["analysis"]
        if a in ("addition_48", "supplement_177"):
            d = r["NAE_mean"] - base_x0                       # negative -> family HELPS
            dec = "mantem" if d <= -DELTA_J else ("remove" if d > DELTA_J else "inconclusivo")
            return pd.Series({"baseline": "X0", "delta_J": round(d, 4), "decisao_dJ0": dec})
        if a == "removal_48":
            d = r["NAE_mean"] - base_s48                      # positive -> removal WORSENS -> important
            dec = "importante" if d > DELTA_J else ("dispensavel" if d <= -DELTA_J else "inconclusivo")
            return pd.Series({"baseline": "S48_ref", "delta_J": round(d, 4), "decisao_dJ0": dec})
        if a == "control":
            d = r["NAE_mean"] - base_x0
            return pd.Series({"baseline": "X0", "delta_J": round(d, 4), "decisao_dJ0": "-"})
        return pd.Series({"baseline": "-", "delta_J": np.nan, "decisao_dJ0": "-"})

    sm = pd.DataFrame(summary)
    sm = pd.concat([sm, sm.apply(decide_row, axis=1)], axis=1)
    sm["decisao_dJ0.01"] = sm.apply(
        lambda r: (("mantem" if (r["NAE_mean"] - base_x0) <= -DELTA_J_SENS else
                    ("remove" if (r["NAE_mean"] - base_x0) > DELTA_J_SENS else "inconclusivo"))
                   if r["analysis"] in ("addition_48", "supplement_177")
                   and r["NAE_mean"] is not None else "-"), axis=1)

    METRICS_DIR.mkdir(parents=True, exist_ok=True); FIGDATA_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / "19_ablation_blocks.csv"
    sm.to_csv(out, index=False)
    pd.DataFrame(long_rows).to_csv(FIGDATA_DIR / "19_ablation_blocks_long.csv", index=False)
    log.info(f"Saved: {out.relative_to(PROJECT_ROOT)}")

    manifest = {
        "script": "19_explainability_ablation.py", "section": "§2.7/§14.10",
        "run_timestamp": datetime.now().isoformat(), "mode": "SMOKE" if SMOKE else "full",
        "surrogate": "HistGradientBoostingRegressor", "deep_model_confirmation": "pendente (Vast)",
        "design": "two-level family ablation (addition/removal on the 48 + supplement 177); 12 aux constant",
        "proxy_analysis_status": proxy_status, "n_proxy_channels": n_proxy,
        "x1_x2_status": "N/A (nenhum canal proxy na v2)",
        "family_partition_audit": {"counts": counts, "sum": int(sum(counts.values())),
                                   "n_features": int(len(bl)), "orphans": 0, "disjoint": True},
        "F_g_48": {g: len(fg48[g]) for g in FAMILIES_ALL},
        "seeds": SEEDS, "train_frac": TRAIN_FRAC, "delta_J": DELTA_J, "delta_J_sens": DELTA_J_SENS,
        "base_addition": "X0", "base_addition_NAE": round(base_x0, 4),
        "base_removal": "S48_ref", "base_removal_NAE": round(base_s48, 4),
        "blocks": sm.where(pd.notnull(sm), None).to_dict(orient="records"),
        "inputs": {"X": str(X_FILE.relative_to(PROJECT_ROOT)),
                   "Y": str(Y_FILE.relative_to(PROJECT_ROOT)),
                   "adopted_set": str(S48_FILE.relative_to(PROJECT_ROOT)),
                   "blocks_map": str(BLOCKS.relative_to(PROJECT_ROOT))},
        "outputs": {"summary": str(out.relative_to(PROJECT_ROOT))},
        "status": "success",
    }
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFESTS_DIR / "19_explainability_ablation_params.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log.info("=== 19 done ===")


if __name__ == "__main__":
    main()
