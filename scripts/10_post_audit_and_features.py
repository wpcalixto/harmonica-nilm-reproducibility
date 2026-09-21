#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 10_post_audit_and_features.py

File type: Pipeline script (stage 10: post-filling audits + M_G feature matrix)

Purpose:
    Post-filling audits + feature matrix X.

    Consolidates three pieces of the pipeline design:

      SECTION A: energy audit ACROSS STAGES.
        Measures the electrical energy (kWh) per meter x phase along the actual stages:
          raw       -> data/interim/02_raw_standardized.parquet   (IRREGULAR timestamps)
          gridded   -> data/interim/06_para_preencher.csv          (1-min grid; sentinels/outliers/gaps = NaN)
          sanitized -> data/interim/06_para_preencher_sanitizado.csv
          filled    -> data/processed/08_preenchido.parquet
        On the raw data it uses trapezoidal integration over the actual dt (energy_nonuniform);
        on the others, the sum over the 1-min grid (energy_uniform). Captures distortion
        introduced by the gridding itself, which script 08 (pre-fill vs fill, both on the grid)
        does not detect.

      SECTION B: energy composition of P_other / P01.
        Breaks down the energy of P01 (meter 100 = M10+M12+M13+M14) per component and per
        phase, from the raw data (operational rows i_an>0.5, secondary scale of v2), and compares
        it with the final energy of P01 after filling.

      SECTION C: feature matrix X.
        Extracts the 177 original features of M_G (meter 1) from the filled data (script 08),
        without inserting new attributes.

    Outputs:
      audits/10_energy_preservation_by_stage.csv
      audits/10_energy_preservation_alerts.csv
      figdata/10_raw_vs_processed_energy_long.csv
      audits/10_p_other_components.csv
      audits/10_p_other_energy_by_phase.csv
      data/processed/10_original_features.parquet
      audits/10_feature_dictionary.csv
      logs/10_post_audit_and_features.log
      manifests/10_post_audit_and_features_params.json

    Each section is independent: if an input is missing, the section is skipped with a warning.

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
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_project_root() -> Path:
    """Repository root: the parent of the directory that contains this script."""
    return SCRIPT_DIR.parent


ROOT = resolve_project_root()
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
AUDITS = ROOT / "audits"
FIGDATA = ROOT / "figdata"
LOGS = ROOT / "logs"
MANIFESTS = ROOT / "manifests"

RAW_FILE = INTERIM / "02_raw_standardized.parquet"
GRID_FILE = INTERIM / "06_para_preencher.csv"
SAN_FILE = INTERIM / "06_para_preencher_sanitizado.csv"
FILL_FILE = PROCESSED / "08_preenchido.parquet"

TIME_COL, METER_COL = "time", "meter_id"
PHASES = ["a", "b", "c"]
PHASE_TO_PCOL = {"a": "p_a", "b": "p_b", "c": "p_c"}

METER_MG = 1
POTHER_METER_ID = 100
POTHER_COMPONENTS = [10, 12, 13, 14]   # = P01 (see script 05)
OPERATIONAL_I_AN = 0.5                  # v2: current on the secondary scale (~1-8 A); operational floor 0.5 A

ALERT_PCT = 5.0    # |relative change between stages| above this -> alert

# Feature families (universe of 177 of the article)
META_COLS = {"Unnamed: 0", TIME_COL, "datetime_read", METER_COL, "tag_meter_id"}


def setup_logger() -> logging.Logger:
    LOGS.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("10_post_audit_and_features")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOGS / "10_post_audit_and_features.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    lg.addHandler(fh)
    lg.addHandler(ch)
    return lg


# ─────────────────────────────────────────────────────────────────────────────
# Energy (with the us/ns fix)
# ─────────────────────────────────────────────────────────────────────────────
def energy_nonuniform(df: pd.DataFrame, meter: int, pcol: str, mask=None) -> float:
    """Energy [kWh] with IRREGULAR timestamps (trapezoidal over the actual dt, cap 5 min)."""
    sub = df[df[METER_COL] == meter]
    if mask is not None:
        sub = sub[mask.reindex(sub.index, fill_value=False)] if hasattr(mask, "reindex") else sub[mask]
    sub = sub.sort_values(TIME_COL)[[TIME_COL, pcol]].dropna()
    if len(sub) < 2:
        return 0.0
    vals = pd.to_numeric(sub[pcol], errors="coerce").to_numpy(dtype=float)
    t = pd.to_datetime(sub[TIME_COL])
    ts = (t - t.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    dt = np.clip(np.diff(ts), 0, 300)
    # TRAPEZOIDAL integration over the actual dt: 0.5*(P_i + P_{i+1})*dt (more defensible
    # for irregular timestamps than the left rectangular sum).
    return float(np.nansum(0.5 * (vals[:-1] + vals[1:]) * dt) / 3_600_000.0)


def energy_uniform(df: pd.DataFrame, meter: int, pcol: str) -> float:
    """Energy [kWh] on a uniform 1-min grid: sum(W)/60000."""
    sub = df[df[METER_COL] == meter]
    vals = pd.to_numeric(sub[pcol], errors="coerce").to_numpy(dtype=float)
    return float(np.nansum(vals) / 60_000.0)


def rel_pct(ref: float, cur: float) -> float:
    return float((cur - ref) / max(abs(ref), 1e-6) * 100.0)


def read_any(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION A: energy across stages
# ─────────────────────────────────────────────────────────────────────────────
def section_energy_by_stage(logger: logging.Logger) -> None:
    if not RAW_FILE.exists() or not FILL_FILE.exists():
        logger.warning("SECTION A skipped: raw and/or filled missing.")
        return
    raw = read_any(RAW_FILE)
    grid = read_any(GRID_FILE) if GRID_FILE.exists() else None
    san = read_any(SAN_FILE) if SAN_FILE.exists() else None
    fill = read_any(FILL_FILE)
    # real meters common to raw and filled (P01=100 is derived -> outside this table)
    meters = sorted(set(raw[METER_COL].dropna().astype(int)) & set(fill[METER_COL].dropna().astype(int)))
    meters = [m for m in meters if m != POTHER_METER_ID]
    rows, longrows = [], []
    for m in meters:
        for ph in PHASES:
            pc = PHASE_TO_PCOL[ph]
            e_raw = energy_nonuniform(raw, m, pc)
            e_grid = energy_uniform(grid, m, pc) if grid is not None else np.nan
            e_san = energy_uniform(san, m, pc) if san is not None else np.nan
            e_fill = energy_uniform(fill, m, pc)
            rows.append({
                "meter_id": m, "phase": ph,
                "E_raw_kWh": round(e_raw, 6), "E_grid_kWh": round(e_grid, 6),
                "E_sanitized_kWh": round(e_san, 6), "E_filled_kWh": round(e_fill, 6),
                "rel_grid_vs_raw_pct": round(rel_pct(e_raw, e_grid), 4) if grid is not None else np.nan,
                "rel_sanitized_vs_grid_pct": round(rel_pct(e_grid, e_san), 4) if (grid is not None and san is not None) else np.nan,
                "rel_filled_vs_sanitized_pct": round(rel_pct(e_san, e_fill), 4) if san is not None else np.nan,
                "rel_filled_vs_raw_pct": round(rel_pct(e_raw, e_fill), 4),
            })
            for stage, val in [("raw", e_raw), ("gridded", e_grid), ("sanitized", e_san), ("filled", e_fill)]:
                longrows.append({"meter_id": m, "phase": ph, "stage": stage, "E_kWh": round(float(val), 6)})
    df = pd.DataFrame(rows)
    AUDITS.mkdir(parents=True, exist_ok=True)
    FIGDATA.mkdir(parents=True, exist_ok=True)
    df.to_csv(AUDITS / "10_energy_preservation_by_stage.csv", index=False)
    pd.DataFrame(longrows).to_csv(FIGDATA / "10_raw_vs_processed_energy_long.csv", index=False)
    # Alert ONLY at the SANITISATION step (grid -> envelope/defrag): there the energy should
    # change little. Large changes indicate reconstruction (defrag of a fragmented series,
    # e.g. M15) or envelope distortion; worth reviewing.
    # NO alert on raw->grid (gridding removes sentinels/outliers/gaps, changes by design)
    # nor on sanitized->filled (the filling ADDS energy in the gaps, proportional to the
    # gap fraction; expected, not an anomaly).
    al = df[df["rel_sanitized_vs_grid_pct"].abs() > ALERT_PCT].copy()
    al.to_csv(AUDITS / "10_energy_preservation_alerts.csv", index=False)
    logger.info(f"SECTION A: {len(df)} rows (meters {meters}) -> 10_energy_preservation_by_stage.csv | "
                f"{len(al)} alerts at sanitisation (|delta grid->envelope| > {ALERT_PCT}%): "
                f"{sorted(set(zip(al.meter_id.tolist(), al.phase.tolist())))}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION B: energy composition of P_other / P01
# ─────────────────────────────────────────────────────────────────────────────
def section_pother_breakdown(logger: logging.Logger) -> None:
    if not RAW_FILE.exists():
        logger.warning("SECTION B skipped: raw missing.")
        return
    raw = read_any(RAW_FILE)
    comp_rows, energy_rows = [], []
    # energy per component, on the operational rows (i_an>0.5, secondary scale of v2)
    for ph in PHASES:
        pc = PHASE_TO_PCOL[ph]
        e_p01 = (energy_uniform(read_any(FILL_FILE), POTHER_METER_ID, pc)
                 if FILL_FILE.exists() and POTHER_METER_ID in read_any(FILL_FILE)[METER_COL].values else np.nan)
        comp_e = {}
        for mid in POTHER_COMPONENTS:
            sub = raw[raw[METER_COL] == mid]
            op = pd.to_numeric(sub.get("i_an"), errors="coerce") > OPERATIONAL_I_AN
            e = energy_nonuniform(sub[op], mid, pc) if op.any() else 0.0
            comp_e[mid] = e
        soma = sum(comp_e.values())
        for mid in POTHER_COMPONENTS:
            comp_rows.append({
                "p_other_meter": POTHER_METER_ID, "component_meter": mid, "phase": ph,
                "E_component_kWh": round(comp_e[mid], 6),
                "pct_of_components": round(100 * comp_e[mid] / max(soma, 1e-9), 2),
            })
        energy_rows.append({
            "phase": ph, "E_components_sum_kWh": round(soma, 6),
            "E_P01_filled_kWh": round(e_p01, 6) if np.isfinite(e_p01) else np.nan,
            "n_components": len(POTHER_COMPONENTS),
        })
    AUDITS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comp_rows).to_csv(AUDITS / "10_p_other_components.csv", index=False)
    pd.DataFrame(energy_rows).to_csv(AUDITS / "10_p_other_energy_by_phase.csv", index=False)
    logger.info(f"SECTION B: P_other (P01={POTHER_METER_ID}) decomposed into {POTHER_COMPONENTS} "
                f"-> 10_p_other_components.csv . 10_p_other_energy_by_phase.csv")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION C: feature matrix X (M_G)
# ─────────────────────────────────────────────────────────────────────────────
def family_of(col: str) -> str:
    if col.startswith("hrm_i_"):
        return "harmonica_corrente"
    if col.startswith("hrm_v_"):
        return "harmonica_tensao"
    if col.startswith(("thdi", "thdv")):
        return "thd"
    if col.startswith(("p_", "q_", "s_", "cos_")):
        return "secundaria"
    if col.startswith(("i_", "v_")):
        return "primaria"
    return "outra"


def section_features_X(logger: logging.Logger) -> None:
    if not FILL_FILE.exists():
        logger.warning("SECTION C skipped: filled missing.")
        return
    fill = read_any(FILL_FILE)
    if METER_MG not in fill[METER_COL].values:
        logger.warning(f"SECTION C skipped: M_G (meter {METER_MG}) missing in the filled data.")
        return
    mg = fill[fill[METER_COL] == METER_MG].sort_values(TIME_COL).copy()
    feat_cols = [c for c in mg.columns if c not in META_COLS]
    X = mg.set_index(TIME_COL)[feat_cols]
    PROCESSED.mkdir(parents=True, exist_ok=True)
    X.to_parquet(PROCESSED / "10_original_features.parquet")
    dic = pd.DataFrame({
        "feature": feat_cols,
        "familia": [family_of(c) for c in feat_cols],
        "n_nan": [int(pd.to_numeric(mg[c], errors="coerce").isna().sum()) for c in feat_cols],
    })
    dic.to_csv(AUDITS / "10_feature_dictionary.csv", index=False)
    logger.info(f"SECTION C: X matrix of M_G = {X.shape[0]} rows x {X.shape[1]} features "
                f"-> 10_original_features.parquet | families: {dic['familia'].value_counts().to_dict()}")
    return X.shape


def main() -> None:
    logger = setup_logger()
    logger.info("=== 10_post_audit_and_features.py started ===")
    logger.info(f"Project root: {ROOT}")
    section_energy_by_stage(logger)
    section_pother_breakdown(logger)
    xshape = section_features_X(logger)

    MANIFESTS.mkdir(parents=True, exist_ok=True)
    manifest = {
        "script": "10_post_audit_and_features.py",
        "run_timestamp": datetime.now().isoformat(),
        "inputs": {"raw": str(RAW_FILE), "gridded": str(GRID_FILE),
                   "sanitized": str(SAN_FILE), "filled": str(FILL_FILE)},
        "p_other": {"meter_id": POTHER_METER_ID, "components": POTHER_COMPONENTS},
        "features_X_meter": METER_MG,
        "features_X_shape": list(xshape) if xshape else None,
        "alert_pct": ALERT_PCT,
    }
    (MANIFESTS / "10_post_audit_and_features_params.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info(f"Manifest saved: {MANIFESTS / '10_post_audit_and_features_params.json'}")
    logger.info("=== 10_post_audit_and_features.py done ===")


if __name__ == "__main__":
    main()
