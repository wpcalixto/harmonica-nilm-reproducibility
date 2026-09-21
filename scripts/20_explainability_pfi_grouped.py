#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 20_explainability_pfi_grouped.py

File type: Pipeline script (stage 20: grouped permutation importance per feature family)

Purpose:
    GROUPED permutation importance (I_g) per feature family G.

    Complements the per-feature PFI of script 18 (which, under correlated features,
    underestimates redundant families) with the JOINT permutation of each family G on the
    ADOPTED MODEL:

        I_g = (NAE_perm(g) - NAE_base) / NAE_base

    permuting in a block all columns of family g present in the adopted set
    (F_g^(48) = G_g intersect S48). Preserves the intra-family correlation and breaks only the
    family-target relation. Reports mean and dispersion over seeds x repetitions.

    DESIGN (v2, without proxies). The channel audit of v2 did NOT classify any channel as a
    proxy (is_proxy=0); the proxy-based analysis is N/A. The families use ALL their features
    of the adopted set (without the `& is_proxy` filter, which produced empty groups). The
    base model is the surrogate trained on the adopted S48 (48 features + 12 auxiliary
    columns), mirroring the input of the deployed model. Confirmation with the deep model is a
    separate GPU run. SMOKE reduces the cost.

    MANDATORY audit of the partition of the 177 features by family (aborts if invalid).

    Inputs:
        data/processed/10_original_features.parquet . data/processed/09_R1.parquet
        data/processed/11_candidate_sets.json (adopted set S48) . audits/12_input_blocks_from_02b.csv
    Outputs:
        metrics/20_pfi_grouped.csv . figdata/20_pfi_grouped_long.csv
        manifests/20_explainability_pfi_grouped_params.json . logs/20_explainability_pfi_grouped.log

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

SEEDS = [42, 123, 456, 789, 1024]
N_REPS = 5
TRAIN_FRAC = 0.80
POWER_COLS = ["p_a", "p_b", "p_c"]
N_AUX = 12
FAMILIES = ["G_P", "G_I", "G_V", "G_THD", "G_hV", "G_hI", "G_sec"]
ANALYSIS_REQUIRES_PROXY = False

SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    SEEDS, N_REPS, ROW_CAP, MAX_ITER = [42, 123], 2, 3000, 40
else:
    ROW_CAP, MAX_ITER = None, 200


def setup_logger(p: Path) -> logging.Logger:
    p.parent.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("20_pfi_grouped"); lg.setLevel(logging.DEBUG); lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(p, mode="w", encoding="utf-8"); fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt)
    lg.addHandler(fh); lg.addHandler(ch); return lg


def nae_percent(y_true, y_pred) -> float:
    num = np.nansum(np.abs(y_pred - y_true), axis=0)
    den = np.nansum(np.abs(y_true), axis=0)
    per = np.where(den > 0, num / den, np.nan)
    return float(np.nanmean(per) * 100.0)


def audit_partition(bl: pd.DataFrame, log) -> dict:
    fam = bl["family_G"]
    counts = {g: int((fam == g).sum()) for g in FAMILIES}
    n_sum = sum(counts.values()); n_total = len(bl)
    n_orphan = int((~fam.isin(FAMILIES)).sum())
    if n_sum != n_total or n_orphan != 0 or bl["feature"].duplicated().any():
        raise RuntimeError(
            f"INVALID family partition: sum={n_sum} orphans={n_orphan} total={n_total} "
            f"(expected sum==total, 0 orphans, disjoint). Counts={counts}")
    if counts["G_sec"] < 9:
        raise RuntimeError(f"Family G_sec (q/s/cos) incomplete: {counts['G_sec']}<9.")
    log.info("Partition audit OK: %s | sum=%d==%d | orphans=0 | disjoint", counts, n_sum, n_total)
    return counts


def main() -> None:
    log = setup_logger(LOGS_DIR / "20_explainability_pfi_grouped.log")
    log.info("=== 20_explainability_pfi_grouped.py: grouped PFI I_g (%s) ===",
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
    Xraw = X.iloc[:n].to_numpy(float); Y = Ydf[yc].iloc[:n].to_numpy(float)
    feat = list(X.columns); idx = {f: i for i, f in enumerate(feat)}

    bl = pd.read_csv(BLOCKS)
    counts = audit_partition(bl, log)

    n_proxy = int(bl["is_proxy"].sum()) if "is_proxy" in bl.columns else 0
    proxy_status = "NOT_APPLICABLE" if n_proxy == 0 else "APPLICABLE"
    if ANALYSIS_REQUIRES_PROXY and n_proxy == 0:
        raise RuntimeError(
            "Proxy-based analysis is not applicable: no channels were classified as proxies.")
    log.info("proxy_analysis_status=%s | n_proxy_channels=%d -> groups by COMPLETE family", proxy_status, n_proxy)

    # -- adopted set S48 (+12 aux) = model input; groups = F_g^(48) --------------------------
    cj = json.loads(S48_FILE.read_text())
    S48 = [f for f in cj[cj["adopted"]] if f in idx]
    s48_i = [idx[f] for f in S48]
    S48names = set(S48)
    # groups per family WITHIN the S48 block (positions 0..47); aux columns are 48..59 (inert)
    grp = {}
    for g in FAMILIES:
        gi = [idx[f] for f in bl.loc[bl["family_G"] == g, "feature"] if f in idx and f in S48names]
        grp[g] = [s48_i.index(i) for i in gi]                 # positions within X_model
    if sum(len(v) for v in grp.values()) != len(s48_i):
        raise RuntimeError("Sum|F_g intersect S48| != |S48|: family groups inconsistent with the adopted set.")
    log.info("S48=%d | groups F_g^(48)=%s", len(s48_i), {g: len(grp[g]) for g in FAMILIES})

    AUX = np.zeros((n, N_AUX), dtype=float)
    Xmodel = np.column_stack([Xraw[:, s48_i], AUX])           # 48 + 12 aux = 60 (mirrors the deployed model)
    split = int(n * TRAIN_FRAC)
    tr, te = slice(0, split), slice(split, n)

    long_rows = []
    for seed in SEEDS:
        models = []
        for j in range(Y.shape[1]):
            yj = Y[tr, j]; ok = np.isfinite(yj)
            if ok.sum() < 50:
                models.append(None); continue
            m = HistGradientBoostingRegressor(max_iter=MAX_ITER, random_state=seed)
            m.fit(Xmodel[tr][ok], yj[ok]); models.append(m)

        def predict(mat):
            P = np.full((mat.shape[0], Y.shape[1]), np.nan)
            for j, m in enumerate(models):
                if m is not None:
                    P[:, j] = m.predict(mat)
            return P

        base = nae_percent(Y[te], predict(Xmodel[te]))
        rng = np.random.default_rng(seed)
        for g in FAMILIES:
            cols = grp[g]
            if not cols:
                continue
            for r in range(N_REPS):
                Xp = Xmodel[te].copy()
                perm = rng.permutation(Xp.shape[0])
                Xp[:, cols] = Xp[perm][:, cols]               # permutes the family as a BLOCK
                nae_p = nae_percent(Y[te], predict(Xp))
                long_rows.append({"family": g, "seed": seed, "rep": r,
                                  "NAE_base": base, "NAE_perm": nae_p,
                                  "I_g": (nae_p - base) / base})
        log.info(f"  seed {seed}: NAE_base={base:.3f}")

    lg = pd.DataFrame(long_rows)
    summ = (lg.groupby("family")["I_g"]
            .agg(I_g_mean="mean", I_g_std="std",
                 I_g_q1=lambda s: s.quantile(0.25), I_g_q3=lambda s: s.quantile(0.75),
                 n="size").reset_index()
            .sort_values("I_g_mean", ascending=False).round(4))
    METRICS_DIR.mkdir(parents=True, exist_ok=True); FIGDATA_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / "20_pfi_grouped.csv"
    summ.to_csv(out, index=False)
    lg.to_csv(FIGDATA_DIR / "20_pfi_grouped_long.csv", index=False)
    log.info("I_g per family:\n%s", summ.to_string(index=False))
    log.info(f"Saved: {out.relative_to(PROJECT_ROOT)}")

    manifest = {
        "script": "20_explainability_pfi_grouped.py", "section": "§2.7/§14.12",
        "run_timestamp": datetime.now().isoformat(), "mode": "SMOKE" if SMOKE else "full",
        "surrogate": "HistGradientBoostingRegressor", "deep_model_confirmation": "pendente (Vast)",
        "model_input": "adopted S48 (48 features + 12 auxiliary)",
        "proxy_analysis_status": proxy_status, "n_proxy_channels": n_proxy,
        "family_partition_audit": {"counts": counts, "sum": int(sum(counts.values())),
                                   "n_features": int(len(bl)), "orphans": 0, "disjoint": True},
        "F_g_48": {g: len(grp[g]) for g in FAMILIES},
        "seeds": SEEDS, "n_reps": N_REPS, "train_frac": TRAIN_FRAC,
        "families": FAMILIES, "grouped_permutation": True,
        "result": summ.to_dict(orient="records"),
        "inputs": {"X": str(X_FILE.relative_to(PROJECT_ROOT)),
                   "Y": str(Y_FILE.relative_to(PROJECT_ROOT)),
                   "adopted_set": str(S48_FILE.relative_to(PROJECT_ROOT)),
                   "blocks_map": str(BLOCKS.relative_to(PROJECT_ROOT))},
        "outputs": {"summary": str(out.relative_to(PROJECT_ROOT))},
        "status": "success",
    }
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFESTS_DIR / "20_explainability_pfi_grouped_params.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log.info("=== 20 done ===")


if __name__ == "__main__":
    main()
