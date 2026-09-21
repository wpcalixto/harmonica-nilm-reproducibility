#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 05_select_meter_scope_candidate.py

File type: Pipeline script (stage 05: generic pre-selection of the meter scope + P_other pseudo-meter)

Purpose:
    Generic pre-selection of the scope per meter.

    Objective
    ---------
    Read the four CSVs produced by the ingestion (02) and reduce the scope per meter,
    preserving the positional alignment between the files.

    This version does NOT fix roles such as Gamma, P_other, sensitivity or redundancy.
    It only knows:

        METER_MG  = general meter, preserved by design;
        CANDIDATE = meters that are candidates to remain in the flow;

    The final decision on the role of each meter is taken in later stages.

    Default inputs in data/interim/ (outputs of 02)
    -----------------------------------------------
    02_primarias.csv
    02_secundarias.csv
    02_distorcao_meta.csv
    02_harmonicas.csv

    Outputs in data/interim/ (does NOT overwrite the 02_* files)
    -----------------------------------------------------------
    05_primarias.csv
    05_secundarias.csv
    05_distorcao_meta.csv
    05_harmonicas.csv

    The script writes the 05_*.csv files only with the kept meters and does NOT overwrite
    the 02_* files. The decision table per meter goes to audits/05_meter_scope_preselection.csv.

    P_other / P01 (pseudo-meter)
    ----------------------------
    At the end, appends to the 05_*.csv a pseudo-meter P01 (meter_id=100) = aggregate of the
    rejected real meters M10+M12+M13+M14 (additive quantities summed: i/p/q/s/hrm_i; the others
    averaged). It is the closure term used later as P_other (scenario R2 of script 09).

    Preliminary quality criterion
    -----------------------------
    For each candidate meter, the quality of the energy signal is computed using current as
    anchor:

      - fraction of finite cells classified as preliminary sentinel;
      - fraction of rows with at least one preliminarily useful current;
      - temporal coverage relative to the meter with most rows.

    By default, a candidate is dropped if:

      - pct_prelim_sentinel_energy >= 60%; or
      - pct_useful_energy < 40%; or
      - coverage_frac < 70%.

    These limits can be changed by command-line arguments.

    Common usage
    ------------
    Main flow:
        python scripts/05_select_meter_scope_candidate.py

    Audit only, without activating the scope:
        python scripts/05_select_meter_scope_candidate.py --dry-run

    Generate the scope-filtered files without replacing the originals:
        python scripts/05_select_meter_scope_candidate.py --no-activate

    Restore the full original files:
        python scripts/05_select_meter_scope_candidate.py --restore-full

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

import argparse
import json
import logging
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_project_root() -> Path:
    """Repository root: the parent of the directory that contains this script."""
    return SCRIPT_DIR.parent


PROJECT_ROOT = resolve_project_root()
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
AUDITS_DIR = PROJECT_ROOT / "audits"
LOGS_DIR = PROJECT_ROOT / "logs"
MANIFESTS_DIR = PROJECT_ROOT / "manifests"

TIME_COL = "time"
METER_COL = "meter_id"

CSV_BASE = {
    "primarias": "02_primarias.csv",
    "secundarias": "02_secundarias.csv",
    "distorcao_meta": "02_distorcao_meta.csv",
    "harmonicas": "02_harmonicas.csv",
}

CSV_SCOPE = {
    "primarias": "05_primarias.csv",
    "secundarias": "05_secundarias.csv",
    "distorcao_meta": "05_distorcao_meta.csv",
    "harmonicas": "05_harmonicas.csv",
}

CSV_FULL = {kind: fname.replace("02_", "02_full_") for kind, fname in CSV_BASE.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Minimal prescribed design
# ─────────────────────────────────────────────────────────────────────────────

# General meter preserved by design.
METER_MG = {1}

# Candidate meters. The script does not assign Gamma/P_other/sensitivity roles.
CANDIDATE = {2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}

# Columns used in the preliminary energy criterion. By default uses current as anchor.
CORE_CURRENT_COLS = ["i_an", "i_bn", "i_cn"]
CORE_POWER_COLS: list[str] = []
ENERGY_COLS = CORE_CURRENT_COLS + CORE_POWER_COLS


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("05_select_meter_scope_candidate_only")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def parse_meter_set(text: str | None) -> Set[int]:
    """Accepts '1,2,3' or '1 2 3'."""
    if text is None or str(text).strip() == "":
        return set()
    vals = re.split(r"[,;\s]+", str(text).strip())
    out: Set[int] = set()
    for v in vals:
        if not v:
            continue
        try:
            out.add(int(v))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid meter in list: {v!r}") from exc
    return out


def as_sorted_list(values: Iterable[int]) -> List[int]:
    return sorted(int(v) for v in values)


def base_path(kind: str) -> Path:
    return INTERIM_DIR / CSV_BASE[kind]


def scope_path(kind: str) -> Path:
    return INTERIM_DIR / CSV_SCOPE[kind]


def full_path(kind: str) -> Path:
    return INTERIM_DIR / CSV_FULL[kind]


def choose_input_path(kind: str) -> Path:
    """Reads 01_full_*.csv if it exists; otherwise reads 01_*.csv."""
    fp = full_path(kind)
    bp = base_path(kind)
    return fp if fp.exists() else bp


def require_files(paths: Iterable[Path], logger: logging.Logger) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        for p in missing:
            logger.error(f"Mandatory file missing: {p}")
        sys.exit(1)


def count_csv_rows(path: Path) -> int:
    """Counts the data rows of a CSV, excluding the header."""
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        n = sum(1 for _ in fh)
    return max(n - 1, 0)


def _numeric_series(s: pd.Series) -> pd.Series:
    """Converts a column to numeric, accepting a decimal comma if present."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


# ─────────────────────────────────────────────────────────────────────────────
# Preliminary quality per meter
# ─────────────────────────────────────────────────────────────────────────────


def preliminary_sentinel_mask_for_column(
    values: pd.Series,
    *,
    round_decimals: int,
    sentinel_dominance_frac: float,
    min_unique_for_real: int,
    robust_amp_eps: float,
    min_finite: int,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Detects a preliminary sentinel in one column of one meter.

    The rule is conservative and serves only for the scope pre-selection:
      1. horizontal dominant value >= sentinel_dominance_frac -> sentinel;
      2. constant/almost constant column -> not informative;
      3. column with few distinct values and low robust amplitude -> not informative.
    """
    x = _numeric_series(values).to_numpy(dtype=float)
    finite = np.isfinite(x)
    sentinel = np.zeros(len(x), dtype=bool)

    meta = {
        "n_finite": int(finite.sum()),
        "n_unique_round": 0,
        "dominant_value": None,
        "dominant_count": 0,
        "dominant_frac": 0.0,
        "robust_amp_q99_q01": 0.0,
        "rule": "none",
    }

    if finite.sum() < min_finite:
        meta["rule"] = "insufficient_finite"
        return sentinel, finite, meta

    xf = x[finite]
    rounded = np.round(xf, round_decimals)
    values_unique, counts = np.unique(rounded, return_counts=True)
    if len(values_unique) == 0:
        return sentinel, finite, meta

    order = np.argsort(counts)[::-1]
    dominant_value = float(values_unique[order[0]])
    dominant_count = int(counts[order[0]])
    dominant_frac = float(dominant_count / max(len(xf), 1))

    q01, q99 = np.nanpercentile(xf, [1, 99])
    robust_amp = float(abs(q99 - q01))
    n_unique = int(len(values_unique))

    meta.update({
        "n_unique_round": n_unique,
        "dominant_value": dominant_value,
        "dominant_count": dominant_count,
        "dominant_frac": round(dominant_frac, 6),
        "robust_amp_q99_q01": round(robust_amp, 12),
    })

    rounded_all = np.round(x, round_decimals)
    is_dominant = finite & np.isclose(
        rounded_all,
        dominant_value,
        atol=10 ** (-round_decimals),
        rtol=0,
    )

    if dominant_frac >= sentinel_dominance_frac:
        sentinel[is_dominant] = True
        meta["rule"] = "dominant_horizontal_value"
        return sentinel, finite, meta

    if n_unique < min_unique_for_real and robust_amp <= robust_amp_eps:
        sentinel[finite] = True
        meta["rule"] = "constant_or_quasi_constant_column"
        return sentinel, finite, meta

    if robust_amp <= robust_amp_eps:
        sentinel[finite] = True
        meta["rule"] = "zero_robust_amplitude"
        return sentinel, finite, meta

    meta["rule"] = "real_variation"
    return sentinel, finite, meta


def compute_preliminary_energy_metrics(
    input_paths: Dict[str, Path],
    meter_series: pd.Series,
    meters_present: List[int],
    logger: logging.Logger,
    *,
    round_decimals: int,
    sentinel_dominance_frac: float,
    min_unique_for_real: int,
    robust_amp_eps: float,
    min_finite_per_column: int,
) -> pd.DataFrame:
    """Computes preliminary energy-quality metrics per meter."""
    prim = pd.read_csv(input_paths["primarias"], low_memory=False)
    sec = pd.read_csv(input_paths["secundarias"], low_memory=False)

    if len(prim) != len(meter_series) or len(sec) != len(meter_series):
        logger.error(
            "Alignment failure while computing preliminary metrics: "
            f"primarias={len(prim)}, secundarias={len(sec)}, meta={len(meter_series)}"
        )
        sys.exit(1)

    source_df: Dict[str, pd.DataFrame] = {}
    for c in CORE_CURRENT_COLS:
        if c in prim.columns:
            source_df[c] = prim
    for c in CORE_POWER_COLS:
        if c in sec.columns:
            source_df[c] = sec

    if not source_df:
        logger.warning("No central energy column found for the pre-selection.")

    meter_values = pd.to_numeric(meter_series, errors="coerce").astype("Int64")
    rows = []
    detail_rows = []

    for m in meters_present:
        m = int(m)
        idx = (meter_values == m).to_numpy(dtype=bool)
        n_rows = int(idx.sum())

        if n_rows == 0:
            rows.append({
                METER_COL: m,
                "n_rows_meter": 0,
                "n_finite_energy_cells": 0,
                "n_prelim_sentinel_energy_cells": 0,
                "pct_prelim_sentinel_energy": 0.0,
                "n_useful_energy_rows": 0,
                "pct_useful_energy": 0.0,
                "energy_cols_used": "",
                "sentinel_cols": "",
            })
            continue

        finite_any = np.zeros(n_rows, dtype=bool)
        useful_any = np.zeros(n_rows, dtype=bool)
        used_cols = []
        sentinel_cols = []
        n_finite_cells = 0
        n_sentinel_cells = 0

        for c, df_src in source_df.items():
            vals = df_src.loc[idx, c].reset_index(drop=True)
            smask, fmask, meta = preliminary_sentinel_mask_for_column(
                vals,
                round_decimals=round_decimals,
                sentinel_dominance_frac=sentinel_dominance_frac,
                min_unique_for_real=min_unique_for_real,
                robust_amp_eps=robust_amp_eps,
                min_finite=min_finite_per_column,
            )
            if fmask.sum() > 0:
                used_cols.append(c)
            if smask.sum() > 0:
                sentinel_cols.append(c)
            finite_any |= fmask
            useful_any |= (fmask & ~smask)
            n_finite_cells += int(fmask.sum())
            n_sentinel_cells += int(smask.sum())

            detail_rows.append({
                METER_COL: m,
                "variavel": c,
                "n_rows_meter": n_rows,
                "n_finite": int(fmask.sum()),
                "n_prelim_sentinel": int(smask.sum()),
                "pct_prelim_sentinel_in_col": round(100 * int(smask.sum()) / max(int(fmask.sum()), 1), 4),
                **meta,
            })

        pct_sentinel = 100.0 * n_sentinel_cells / max(n_finite_cells, 1)
        n_useful_rows = int(useful_any.sum())
        pct_useful = 100.0 * n_useful_rows / max(n_rows, 1)

        rows.append({
            METER_COL: m,
            "n_rows_meter": n_rows,
            "n_finite_energy_cells": int(n_finite_cells),
            "n_prelim_sentinel_energy_cells": int(n_sentinel_cells),
            "pct_prelim_sentinel_energy": round(pct_sentinel, 4),
            "n_useful_energy_rows": n_useful_rows,
            "pct_useful_energy": round(pct_useful, 4),
            "energy_cols_used": ",".join(used_cols),
            "sentinel_cols": ",".join(sentinel_cols),
        })

    metrics = pd.DataFrame(rows)
    detail = pd.DataFrame(detail_rows)
    metrics.to_csv(AUDITS_DIR / "05_preliminary_energy_quality_by_meter.csv", index=False)
    detail.to_csv(AUDITS_DIR / "05_preliminary_sentinel_by_meter_variable.csv", index=False)
    metrics.to_csv(AUDITS_DIR / "05_useful_energy_points_by_meter.csv", index=False)
    logger.info("Saved: audits/05_preliminary_energy_quality_by_meter.csv")
    logger.info("Saved: audits/05_preliminary_sentinel_by_meter_variable.csv")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Scope decision
# ─────────────────────────────────────────────────────────────────────────────


INTERLEAVE_PARTIAL_COV = 0.90   # each meter of the pair needs coverage < this (it is "partial")


def detect_duplicate_meters(
    keep_meters: Set[int],
    input_paths: Dict[str, Path],
    meter_series: pd.Series,
    time_series,
    *,
    meter_mg: Set[int],
    corr_threshold: float,
    magn_tol: float,
    min_overlap: int,
    interleave_max_overlap_frac: float,
    interleave_min_union: float,
    logger: logging.Logger,
) -> Tuple[Set[int], List[dict], List[dict]]:
    """Detects DUPLICATED meters among the kept ones and leaves only 1 per group.

    TWO types of duplicate:
      (A) OVERLAPPING: same time window: Pearson correlation of the 3 currents on the
          common minutes >= corr_threshold AND close magnitude (<= magn_tol).
      (B) INTERLEAVED: almost disjoint windows (overlap_frac <= interleave_max_overlap_frac),
          both with PARTIAL coverage (< INTERLEAVE_PARTIAL_COV) and union >= interleave_min_union
          of the period. This is the M14/M15 case: the same point split into two IDs that
          alternate in time (the direct correlation misses them, as there are no common minutes).

    The GENERAL METER (M_G) never enters the removal. Per group, the meter MOST SIMILAR
    TO M_G is KEPT (highest correlation with M1); tie -> highest coverage -> lowest id.

    Returns (set_to_remove, similarity_rows, group_rows).
    """
    keep = sorted(int(m) for m in keep_meters)
    if len(keep) < 2:
        return set(), [], []

    prim = pd.read_csv(input_paths["primarias"], low_memory=False)
    cols = [c for c in CORE_CURRENT_COLS if c in prim.columns]
    if not cols:
        return set(), [], []
    mv = pd.to_numeric(meter_series, errors="coerce").astype("Int64").to_numpy()
    tmin = pd.to_datetime(pd.Series(time_series).reset_index(drop=True),
                          errors="coerce", utc=True).dt.floor("1min")

    # Minute x (3 currents) matrix per meter (mean per minute).
    permeter: Dict[int, pd.DataFrame] = {}
    for m in keep:
        sel = (mv == m)
        if not sel.any():
            continue
        sub = pd.DataFrame({c: _numeric_series(prim.loc[sel, c]).to_numpy(dtype=float) for c in cols})
        sub["_t"] = tmin[sel].to_numpy()
        g = sub.dropna(subset=["_t"]).groupby("_t")[cols].mean()
        permeter[m] = g.dropna(how="all")
    if not permeter:
        return set(), [], []

    n_ref = max(len(df) for df in permeter.values())               # reference coverage
    cov = {m: (len(permeter[m]) / n_ref if n_ref else 0.0) for m in permeter}
    idx_set = {m: set(permeter[m].index) for m in permeter}

    # Correlation of each meter with M_G (common minutes), to choose the representative.
    mg_list = sorted(meter_mg)
    corr_to_mg: Dict[int, float] = {}
    gkey = mg_list[0] if (mg_list and mg_list[0] in permeter) else None
    for m in permeter:
        if m in meter_mg:
            corr_to_mg[m] = 1.0
            continue
        if gkey is None:
            corr_to_mg[m] = float("nan")
            continue
        jo = permeter[m].join(permeter[gkey], lsuffix="_a", rsuffix="_g", how="inner").dropna()
        rs = []
        if len(jo) >= min_overlap:
            for c in cols:
                xa = jo[f"{c}_a"].to_numpy(dtype=float); xg = jo[f"{c}_g"].to_numpy(dtype=float)
                if xa.std() > 1e-12 and xg.std() > 1e-12:
                    rs.append(float(np.corrcoef(xa, xg)[0, 1]))
        corr_to_mg[m] = float(np.nanmean(rs)) if rs else float("nan")

    sim_rows: List[dict] = []
    edges: List[Tuple[int, int]] = []
    for i in range(len(keep)):
        for j in range(i + 1, len(keep)):
            a, b = keep[i], keep[j]
            if a not in permeter or b not in permeter:
                continue
            overlap = len(idx_set[a] & idx_set[b])
            la, lb = len(permeter[a]), len(permeter[b])
            ovl_frac = overlap / max(1, min(la, lb))
            union_frac = (la + lb - overlap) / n_ref if n_ref else 0.0
            involves_mg = bool((a in meter_mg) or (b in meter_mg))

            # (A) OVERLAPPING duplicate (needs common minutes)
            r = float("nan"); mag = float("nan"); dup_over = False
            if overlap >= min_overlap:
                jo = permeter[a].join(permeter[b], lsuffix="_a", rsuffix="_b", how="inner").dropna()
                rs, mags = [], []
                for c in cols:
                    xa = jo[f"{c}_a"].to_numpy(dtype=float); xb = jo[f"{c}_b"].to_numpy(dtype=float)
                    if xa.std() > 1e-12 and xb.std() > 1e-12:
                        rs.append(float(np.corrcoef(xa, xb)[0, 1]))
                    ma, mb = float(np.median(np.abs(xa))), float(np.median(np.abs(xb)))
                    if max(ma, mb) > 0:
                        mags.append(abs(ma - mb) / max(ma, mb))
                r = float(np.nanmean(rs)) if rs else 0.0
                mag = float(np.nanmean(mags)) if mags else 1.0
                dup_over = bool(r >= corr_threshold and mag <= magn_tol)

            # (B) INTERLEAVED duplicate (disjoint, partial windows, high union)
            dup_inter = bool(
                ovl_frac <= interleave_max_overlap_frac
                and cov[a] < INTERLEAVE_PARTIAL_COV and cov[b] < INTERLEAVE_PARTIAL_COV
                and union_frac >= interleave_min_union
            )

            sim_rows.append({
                "meter_a": a, "meter_b": b, "n_minutos_comuns": overlap,
                "overlap_frac": round(ovl_frac, 6), "uniao_frac": round(union_frac, 6),
                "cov_a": round(cov[a], 4), "cov_b": round(cov[b], 4),
                "correlacao": (round(r, 6) if r == r else None),
                "magn_rel_diff": (round(mag, 6) if mag == mag else None),
                "corr_a_mg": (round(corr_to_mg.get(a, float('nan')), 6) if corr_to_mg.get(a) == corr_to_mg.get(a) else None),
                "corr_b_mg": (round(corr_to_mg.get(b, float('nan')), 6) if corr_to_mg.get(b) == corr_to_mg.get(b) else None),
                "duplicata_sobreposta": dup_over, "duplicata_intercalada": dup_inter,
                "envolve_medidor_geral": involves_mg,
            })
            # A candidate duplicating the general meter also forms an edge: M_G will be the
            # representative (corr_to_mg=1.0) and the candidate is the one dropped.
            if dup_over or dup_inter:
                edges.append((a, b))

    # Groups the duplicates (union-find) and chooses 1 representative per group.
    parent = {m: m for m in keep}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)

    groups: Dict[int, List[int]] = {}
    for m in keep:
        groups.setdefault(find(m), []).append(m)

    def _rep_key(m):
        c = corr_to_mg.get(m, float("nan"))
        return (c if c == c else float("-inf"), cov.get(m, 0.0), -m)

    drop: Set[int] = set()
    group_rows: List[dict] = []
    for members in groups.values():
        if len(members) > 1:
            rep = max(members, key=_rep_key)               # most similar to M_G
            removed = [m for m in sorted(members) if m != rep]
            drop.update(removed)
            group_rows.append({"grupo": "+".join(f"M{m}" for m in sorted(members)),
                               "mantido": f"M{rep}",
                               "removidos": "+".join(f"M{m}" for m in removed),
                               "corr_mantido_mg": round(corr_to_mg.get(rep, float('nan')), 6) if corr_to_mg.get(rep) == corr_to_mg.get(rep) else None})
    drop -= {int(m) for m in meter_mg}        # safety: the general meter is NEVER removed
    if drop:
        logger.info(f"Deduplication: groups {group_rows} -> removes {as_sorted_list(drop)}")
    else:
        logger.info("Deduplication: no duplicated pair (neither overlapping nor interleaved).")
    return drop, sim_rows, group_rows


def decide_meter_scope(
    meters_present: List[int],
    quality_metrics: pd.DataFrame,
    meter_mg: Set[int],
    candidates: Set[int],
    force_keep: Set[int],
    force_drop: Set[int],
    min_useful_energy_frac: float,
    max_sentinel_energy_frac: float,
    min_coverage_frac: float,
    apply_quality_to_main_meter: bool,
) -> Tuple[pd.DataFrame, Set[int]]:
    """Defines the pre-selection per meter with only two roles: M_G and candidate."""
    rows = []
    keep: Set[int] = set()
    metric_idx = quality_metrics.set_index(METER_COL).to_dict(orient="index") if len(quality_metrics) else {}
    min_useful_pct = 100.0 * float(min_useful_energy_frac)
    max_sentinel_pct = 100.0 * float(max_sentinel_energy_frac)
    min_coverage_pct = 100.0 * float(min_coverage_frac)
    max_rows_ref = max((int(v.get("n_rows_meter", 0)) for v in metric_idx.values()), default=0)

    for m in meters_present:
        m = int(m)
        process = False
        pre_role = "not_candidate"
        reason = "fora_da_lista_candidate"
        forced = False

        if m in force_drop:
            process = False
            pre_role = "excluded_force_drop"
            reason = "force_drop"
            forced = True
        elif m in force_keep:
            process = True
            pre_role = "force_keep"
            reason = "force_keep"
            forced = True
        elif m in meter_mg:
            process = True
            pre_role = "M_G"
            reason = "medidor_geral"
        elif m in candidates:
            process = True
            pre_role = "candidate"
            reason = "candidato_prescrito"

        metric = metric_idx.get(m, {})
        n_rows_meter = int(metric.get("n_rows_meter", 0))
        n_finite_cells = int(metric.get("n_finite_energy_cells", 0))
        n_sent_cells = int(metric.get("n_prelim_sentinel_energy_cells", 0))
        pct_sentinel = float(metric.get("pct_prelim_sentinel_energy", 0.0))
        n_useful = int(metric.get("n_useful_energy_rows", 0))
        pct_useful = float(metric.get("pct_useful_energy", 0.0))
        sentinel_cols = str(metric.get("sentinel_cols", ""))
        coverage_frac = (n_rows_meter / max_rows_ref) if max_rows_ref else 0.0

        quality_gate_applied = bool(
            process
            and not forced
            and (pre_role == "candidate" or (pre_role == "M_G" and apply_quality_to_main_meter))
        )
        sentinel_gate_passed = bool(pct_sentinel < max_sentinel_pct)
        useful_gate_passed = bool(pct_useful >= min_useful_pct)
        coverage_gate_passed = bool(coverage_frac >= min_coverage_frac)

        if quality_gate_applied and not sentinel_gate_passed:
            process = False
            pre_role = "excluded_pre05_prelim_sentinel"
            reason = f"pct_prelim_sentinel_energy={pct_sentinel:.2f}%>={max_sentinel_pct:.1f}%"
        elif quality_gate_applied and not useful_gate_passed:
            process = False
            pre_role = "excluded_pre05_min_useful_energy"
            reason = f"pct_useful_energy={pct_useful:.2f}%<{min_useful_pct:.1f}%"
        elif quality_gate_applied and not coverage_gate_passed:
            process = False
            pre_role = "excluded_pre05_min_coverage"
            reason = f"cobertura={coverage_frac * 100:.1f}%<{min_coverage_pct:.1f}%"

        if process:
            keep.add(m)

        rows.append({
            METER_COL: m,
            "process_in_next_script": bool(process),
            "pre_role": pre_role,
            "is_main_meter": bool(m in meter_mg),
            "is_candidate": bool(m in candidates),
            "n_rows_meter": n_rows_meter,
            "n_finite_energy_cells": n_finite_cells,
            "n_prelim_sentinel_energy_cells": n_sent_cells,
            "pct_prelim_sentinel_energy": round(pct_sentinel, 4),
            "n_useful_energy_rows": n_useful,
            "pct_useful_energy": round(pct_useful, 4),
            "sentinel_cols": sentinel_cols,
            "coverage_frac": round(coverage_frac, 4),
            "quality_gate_applied": quality_gate_applied,
            "sentinel_gate_passed": sentinel_gate_passed,
            "useful_gate_passed": useful_gate_passed,
            "coverage_gate_passed": coverage_gate_passed,
            "max_sentinel_energy_pct": round(max_sentinel_pct, 4),
            "min_useful_energy_pct": round(min_useful_pct, 4),
            "min_coverage_pct": round(min_coverage_pct, 4),
            "pre_exclusion_reason": reason,
        })

    return pd.DataFrame(rows), keep


# ─────────────────────────────────────────────────────────────────────────────
# Backup, activation and restoration
# ─────────────────────────────────────────────────────────────────────────────


def activate_scope_files(logger: logging.Logger) -> List[dict]:
    """Activates the scope by replacing 01_*.csv with the 01_scope_*.csv versions."""
    activated = []

    for kind in CSV_BASE:
        original = base_path(kind)
        backup = full_path(kind)
        scoped = scope_path(kind)

        if not original.exists():
            raise FileNotFoundError(f"Original file missing: {original}")
        if not scoped.exists():
            raise FileNotFoundError(f"Filtered file missing: {scoped}")

        if not backup.exists():
            shutil.copyfile(original, backup)
            logger.info(f"Backup created: {backup.name}")
        else:
            logger.info(f"Backup already existed: {backup.name}")

        shutil.copyfile(scoped, original)
        logger.info(f"Scope activated: {scoped.name} -> {original.name}")

        activated.append({
            "kind": kind,
            "backup": backup.name,
            "scoped": scoped.name,
            "activated_as": original.name,
        })

    return activated


def restore_full_files(logger: logging.Logger) -> List[dict]:
    """Restores 01_*.csv from 01_full_*.csv."""
    restored = []
    for kind in CSV_BASE:
        original = base_path(kind)
        backup = full_path(kind)
        if not backup.exists():
            raise FileNotFoundError(f"Backup missing, cannot restore: {backup}")
        shutil.copyfile(backup, original)
        logger.info(f"Restored: {backup.name} -> {original.name}")
        restored.append({"kind": kind, "from": backup.name, "to": original.name})
    return restored


# ─────────────────────────────────────────────────────────────────────────────
# P_other (P01) + Gamma_5 merge: read from meter_map.yaml
# ─────────────────────────────────────────────────────────────────────────────

def _meter_map_roles() -> dict:
    """roles from meter_map.yaml (p_other_components, gamma5_merge_meters)."""
    try:
        import yaml
        cfg = yaml.safe_load((PROJECT_ROOT / "config" / "meter_map.yaml").read_text(encoding="utf-8"))
        return (cfg or {}).get("roles", {}) or {}
    except Exception:
        return {}


_ROLES = _meter_map_roles()
POTHER_ID = 100
POTHER_LABEL = "P01"
# v2: P_other = M10+M12+M13+M14 (M14 is a load distinct from M15 by the electrical audit).
POTHER_METERS = list(_ROLES.get("p_other_components", [10, 12, 13, 14]))
# Gamma_5 merge RETIRED (electrical audit: M14 != M15). No dedup protection.
GAMMA5_MERGE_METERS = set(_ROLES.get("gamma5_merge_meters", []))
# Additive quantities (summed across meters): currents, powers and current harmonics.
RE_ADDITIVE_POTHER = re.compile(r"^(i_[abc]n$|p_[abc]$|q_[abc]$|s_[abc]$|hrm_i_[abc]n_)")


def append_pother_meter(input_paths: dict, output_paths: dict, logger: logging.Logger) -> int:
    """Builds P01 = sum/mean of M10,M12,M13,M14 (operational rows) and appends it
    as meter_id=100 to the 05_*.csv files. Additive quantities summed; the others (v, cos,
    thd, hrm_v) averaged. Rows in the same temporal order in the 4 files (keeps alignment)."""
    meta = pd.read_csv(input_paths["distorcao_meta"], low_memory=False)
    mid = pd.to_numeric(meta[METER_COL], errors="coerce")
    prim = pd.read_csv(input_paths["primarias"], low_memory=False)
    i_an = pd.to_numeric(prim["i_an"], errors="coerce").to_numpy()
    # v2: current on the secondary scale (~1-8 A); sentinel floor = 0. Operational > 0.5 A.
    op = i_an > 0.5
    sel = mid.isin(POTHER_METERS).to_numpy() & op
    if int(sel.sum()) == 0:
        logger.warning(f"P_other: no operational rows in M{POTHER_METERS}; nothing appended.")
        return 0
    times_all = meta[TIME_COL].astype(str).to_numpy()
    key = times_all[sel]
    uniq = np.array(sorted(pd.unique(key)))            # ISO -> lexicographic = chronological
    n = len(uniq)
    sel_idx = np.flatnonzero(sel)

    for kind in CSV_BASE:
        full = pd.read_csv(input_paths[kind], low_memory=False)
        sub = full.iloc[sel_idx].copy()
        tkey = pd.Series(key)
        cols = {}
        for c in full.columns:
            if c == METER_COL:
                cols[c] = np.full(n, POTHER_ID)
            elif c == TIME_COL:
                cols[c] = uniq
            elif c in ("Unnamed: 0", "datetime_read", "tag_meter_id"):
                # tag_meter_id is a float per meter (not a name); P01 is identified
                # by meter_id=100; keeps the numeric column to preserve the dtype.
                cols[c] = np.full(n, np.nan)
            else:
                v = pd.to_numeric(sub[c], errors="coerce")
                g = v.groupby(tkey.values)
                agg = g.sum(min_count=1) if RE_ADDITIVE_POTHER.match(c) else g.mean()
                cols[c] = pd.Series(uniq).map(agg).to_numpy()
        rows = pd.DataFrame(cols, columns=list(full.columns))
        cur = pd.read_csv(output_paths[kind], low_memory=False)
        pd.concat([cur, rows], ignore_index=True).to_csv(output_paths[kind], index=False)

    logger.info(f"P_other (meter_id={POTHER_ID}='{POTHER_LABEL}') appended: {n} rows in each 05_*.csv "
                f"(aggregate of M{POTHER_METERS}).")
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pre-selects meters with minimal roles: M_G and CANDIDATE."
    )
    p.add_argument(
        "--main-meter",
        type=parse_meter_set,
        default=METER_MG,
        help="General meter(s), e.g. --main-meter 1. Default: 1.",
    )
    p.add_argument(
        "--candidate-meters",
        type=parse_meter_set,
        default=CANDIDATE,
        help="List of candidates, e.g. --candidate-meters 2,5,6,7,10,11,15,16.",
    )
    p.add_argument(
        "--apply-quality-to-main-meter",
        action="store_true",
        help="Also applies the quality criteria to the general meter. By default, M_G is preserved.",
    )
    p.add_argument(
        "--min-useful-energy-frac",
        type=float,
        default=0.40,
        help="Minimum fraction of rows with a useful energy signal for candidates. Default: 0.40.",
    )
    p.add_argument(
        "--max-sentinel-energy-frac",
        type=float,
        default=0.60,
        help="Maximum allowed fraction of finite energy cells classified as preliminary sentinel. Default: 0.60.",
    )
    p.add_argument(
        "--min-coverage-frac",
        type=float,
        default=0.50,
        help="Minimum temporal coverage relative to the meter with most rows. Default: 0.50 "
             "(admits the partial meters M14/M15 ~51%; the deduplication keeps M15 and removes M14).",
    )
    p.add_argument(
        "--sentinel-dominance-frac",
        type=float,
        default=0.80,
        help="Horizontal value dominance for a preliminary sentinel. Default: 0.80.",
    )
    p.add_argument(
        "--min-unique-for-real",
        type=int,
        default=4,
        help="Minimum number of rounded unique values to consider real variation. Default: 4.",
    )
    p.add_argument(
        "--placeholder-round-decimals",
        type=int,
        default=6,
        help="Decimal places used to detect the horizontal dominant value. Default: 6.",
    )
    p.add_argument(
        "--robust-amp-eps",
        type=float,
        default=1e-9,
        help="Minimum robust amplitude q99-q01 to not consider a column constant. Default: 1e-9.",
    )
    p.add_argument(
        "--min-finite-per-column",
        type=int,
        default=10,
        help="Minimum number of finite points to evaluate an energy column. Default: 10.",
    )
    p.add_argument(
        "--force-keep",
        type=parse_meter_set,
        default=set(),
        help="List of meters to keep, e.g. --force-keep 3,4,8",
    )
    p.add_argument(
        "--force-drop",
        type=parse_meter_set,
        default=set(),
        help="List of meters to drop, e.g. --force-drop 9,17,18",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Computes the selection and writes audits/manifest, but neither generates 01_scope_*.csv nor activates the scope.",
    )
    p.add_argument(
        "--no-activate",
        action="store_true",
        help="Generates 01_scope_*.csv, but does not replace the 01_*.csv files.",
    )
    p.add_argument(
        "--restore-full",
        action="store_true",
        help="Restores the complete 01_*.csv files from the 01_full_*.csv backups and exits.",
    )
    p.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disables the similarity-based deduplication among kept meters.",
    )
    p.add_argument(
        "--dup-corr-threshold",
        type=float,
        default=0.999,
        help="Minimum correlation (mean of the 3 currents, common minutes) to consider a duplicate. Default: 0.999.",
    )
    p.add_argument(
        "--dup-magn-tol",
        type=float,
        default=0.05,
        help="Maximum relative magnitude difference to consider a duplicate. Default: 0.05 (5%).",
    )
    p.add_argument(
        "--dup-min-overlap",
        type=int,
        default=200,
        help="Minimum number of common minutes between two meters to assess OVERLAPPING duplication. Default: 200.",
    )
    p.add_argument(
        "--dup-interleave-max-overlap-frac",
        type=float,
        default=0.05,
        help="Maximum overlap fraction to consider an INTERLEAVED duplicate (disjoint windows). Default: 0.05.",
    )
    p.add_argument(
        "--dup-interleave-min-union",
        type=float,
        default=0.80,
        help="Minimum coverage union of the pair to consider an INTERLEAVED duplicate. Default: 0.80.",
    )
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    args = build_parser().parse_args()

    for d in [INTERIM_DIR, AUDITS_DIR, LOGS_DIR, MANIFESTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(LOGS_DIR / "05_select_meter_scope_candidate_only.log")
    logger.info("=== 05_select_meter_scope_candidate.py started ===")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Run timestamp: {datetime.now().isoformat()}")

    if args.restore_full:
        restored = restore_full_files(logger)
        manifest = {
            "script": "05_select_meter_scope_candidate_only.py",
            "section": "§02A",
            "run_timestamp": datetime.now().isoformat(),
            "project_root": str(PROJECT_ROOT),
            "operation": "restore_full",
            "restored": restored,
            "status": "success",
        }
        with open(MANIFESTS_DIR / "05_select_meter_scope_candidate_only_params.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
        logger.info("=== restoration done ===")
        return

    # Parameter validation.
    for name in ["min_useful_energy_frac", "max_sentinel_energy_frac", "min_coverage_frac"]:
        val = float(getattr(args, name))
        if not (0.0 <= val <= 1.0):
            logger.error(f"--{name.replace('_', '-')} must be between 0 and 1.")
            sys.exit(1)
    if not (0.0 < float(args.sentinel_dominance_frac) <= 1.0):
        logger.error("--sentinel-dominance-frac must be in (0, 1].")
        sys.exit(1)
    if int(args.placeholder_round_decimals) < 0:
        logger.error("--placeholder-round-decimals must be >= 0.")
        sys.exit(1)

    input_paths = {kind: choose_input_path(kind) for kind in CSV_BASE}
    require_files(input_paths.values(), logger)

    logger.info("Effective input files:")
    for kind, path in input_paths.items():
        source_type = "full_backup" if path.name.startswith("01_full_") else "base"
        logger.info(f"  {kind:14s}: {path.name} ({source_type})")

    meta_path = input_paths["distorcao_meta"]
    meta = pd.read_csv(meta_path, low_memory=False)
    if METER_COL not in meta.columns:
        logger.error(f"Column {METER_COL!r} missing in {meta_path}")
        sys.exit(1)

    meter_series = pd.to_numeric(meta[METER_COL], errors="coerce")
    if meter_series.isna().all():
        logger.error(f"Column {METER_COL!r} could not be interpreted as numeric in {meta_path}")
        sys.exit(1)

    meters_present = as_sorted_list(meter_series.dropna().astype(int).unique())
    n_total = int(len(meta))

    meter_mg = set(args.main_meter or set())
    candidates = set(args.candidate_meters or set())
    force_keep = set(args.force_keep or set())
    force_drop = set(args.force_drop or set())

    if force_keep & force_drop:
        logger.error(f"Meters simultaneously in force_keep and force_drop: {as_sorted_list(force_keep & force_drop)}")
        sys.exit(1)

    logger.info(f"METER_MG: {as_sorted_list(meter_mg)}")
    logger.info(f"CANDIDATE: {as_sorted_list(candidates)}")

    quality_metrics = compute_preliminary_energy_metrics(
        input_paths=input_paths,
        meter_series=meter_series,
        meters_present=meters_present,
        logger=logger,
        round_decimals=int(args.placeholder_round_decimals),
        sentinel_dominance_frac=float(args.sentinel_dominance_frac),
        min_unique_for_real=int(args.min_unique_for_real),
        robust_amp_eps=float(args.robust_amp_eps),
        min_finite_per_column=int(args.min_finite_per_column),
    )

    decision_df, keep_meters = decide_meter_scope(
        meters_present=meters_present,
        quality_metrics=quality_metrics,
        meter_mg=meter_mg,
        candidates=candidates,
        force_keep=force_keep,
        force_drop=force_drop,
        min_useful_energy_frac=float(args.min_useful_energy_frac),
        max_sentinel_energy_frac=float(args.max_sentinel_energy_frac),
        min_coverage_frac=float(args.min_coverage_frac),
        apply_quality_to_main_meter=bool(args.apply_quality_to_main_meter),
    )

    # Components of P_other (incl. M14 in v2) do NOT enter the dedup of modelled loads:
    # they are aggregated into P01. Avoids labelling them as "duplicate": M14 != M15 by the
    # electrical audit (I/S ~5.7x, cos(phi) 0.36 vs 0.71, opposite Q); no union/merge.
    pother_in_keep = keep_meters & set(POTHER_METERS)
    if pother_in_keep:
        keep_meters = keep_meters - set(POTHER_METERS)
        sel_po = decision_df[METER_COL].isin(pother_in_keep)
        decision_df.loc[sel_po, "process_in_next_script"] = False
        decision_df.loc[sel_po, "pre_role"] = "p_other_component"
        decision_df.loc[sel_po, "pre_exclusion_reason"] = "agregado_em_P01"
        logger.info(f"P_other: outside the dedup, aggregated into P01: {as_sorted_list(pother_in_keep)}")

    # Deduplication by similarity: among the kept meters, merges near-duplicates and leaves 1.
    dup_sim, dup_groups = [], []
    if not args.no_dedup:
        dup_drop, dup_sim, dup_groups = detect_duplicate_meters(
            keep_meters, input_paths, meter_series, meta[TIME_COL],
            meter_mg=meter_mg,
            corr_threshold=float(args.dup_corr_threshold),
            magn_tol=float(args.dup_magn_tol),
            min_overlap=int(args.dup_min_overlap),
            interleave_max_overlap_frac=float(args.dup_interleave_max_overlap_frac),
            interleave_min_union=float(args.dup_interleave_min_union),
            logger=logger,
        )
        # (Gamma_5 merge RETIRED in v2: M14 != M15 by the electrical audit; M14 was already
        # routed to P_other above, outside the dedup.)
        if dup_drop:
            keep_meters = keep_meters - dup_drop
            sel = decision_df[METER_COL].isin(dup_drop)
            decision_df.loc[sel, "process_in_next_script"] = False
            decision_df.loc[sel, "pre_role"] = "excluded_pre05_duplicate"
            decision_df.loc[sel, "pre_exclusion_reason"] = "duplicata_por_similaridade"
    pd.DataFrame(dup_sim).to_csv(AUDITS_DIR / "05_duplicate_similarity.csv", index=False)
    pd.DataFrame(dup_groups).to_csv(AUDITS_DIR / "05_duplicate_groups.csv", index=False)

    keep_mask = meter_series.astype("Int64").isin(keep_meters).to_numpy()
    n_keep = int(keep_mask.sum())

    logger.info("Scope summary:")
    logger.info(f"  meters present: {meters_present}")
    logger.info(f"  meters kept:    {as_sorted_list(keep_meters)}")
    logger.info(f"  meters dropped: {as_sorted_list(set(meters_present) - keep_meters)}")
    logger.info(f"  rows: {n_keep}/{n_total} ({100 * n_keep / max(n_total, 1):.2f}%)")

    decision_df.to_csv(AUDITS_DIR / "05_meter_scope_preselection.csv", index=False)

    output_paths = {kind: scope_path(kind) for kind in CSV_SCOPE}
    row_summary = []

    if not args.dry_run:
        for kind in CSV_BASE:
            in_path = input_paths[kind]
            out_path = output_paths[kind]
            logger.info(f"Filtering {kind}: {in_path.name} -> {out_path.name}")

            df = pd.read_csv(in_path, low_memory=False)
            if len(df) != n_total:
                logger.error(
                    f"Alignment failure: {in_path.name} has {len(df)} rows, "
                    f"but {meta_path.name} has {n_total}."
                )
                sys.exit(1)

            df_scope = df.loc[keep_mask].reset_index(drop=True)
            df_scope.to_csv(out_path, index=False)

            row_summary.append({
                "kind": kind,
                "input_file": in_path.name,
                "scope_file": out_path.name,
                "n_rows_input": int(len(df)),
                "n_rows_output": int(len(df_scope)),
                "pct_rows_kept": round(100 * len(df_scope) / max(len(df), 1), 4),
            })
    else:
        logger.info("dry-run active: the 01_scope_*.csv files were not generated.")
        for kind in CSV_BASE:
            row_summary.append({
                "kind": kind,
                "input_file": input_paths[kind].name,
                "scope_file": CSV_SCOPE[kind],
                "n_rows_input": int(n_total),
                "n_rows_output": int(n_keep),
                "pct_rows_kept": round(100 * n_keep / max(n_total, 1), 4),
            })

    pd.DataFrame(row_summary).to_csv(AUDITS_DIR / "05_row_filter_summary.csv", index=False)

    activation: List[dict] = []
    if not args.dry_run:
        lens = [count_csv_rows(out_path) for out_path in output_paths.values()]
        if len(set(lens)) != 1:
            logger.error(f"Failure: the 01_scope_*.csv files do not have the same number of rows: {lens}")
            sys.exit(1)
        logger.info(f"Validation OK: all 01_scope_*.csv files have {lens[0]} rows.")

        # P_other (P01): aggregates M10,M12,M13,M14 and appends it as meter_id=100 to the 05_* files.
        append_pother_meter(input_paths, output_paths, logger)

        logger.info("Writes 05_*.csv and does NOT overwrite 02_*.")
    else:
        logger.info("dry-run active: the scope was not activated.")

    manifest = {
        "script": "05_select_meter_scope_candidate_only.py",
        "section": "§02A",
        "run_timestamp": datetime.now().isoformat(),
        "project_root": str(PROJECT_ROOT),
        "inputs": {k: str(v) for k, v in input_paths.items()},
        "outputs": {k: str(v) for k, v in output_paths.items()} if not args.dry_run else {},
        "parameters": {
            "meter_mg": as_sorted_list(meter_mg),
            "candidate": as_sorted_list(candidates),
            "min_useful_energy_frac": float(args.min_useful_energy_frac),
            "max_sentinel_energy_frac": float(args.max_sentinel_energy_frac),
            "min_coverage_frac": float(args.min_coverage_frac),
            "sentinel_dominance_frac": float(args.sentinel_dominance_frac),
            "min_unique_for_real": int(args.min_unique_for_real),
            "placeholder_round_decimals": int(args.placeholder_round_decimals),
            "robust_amp_eps": float(args.robust_amp_eps),
            "min_finite_per_column": int(args.min_finite_per_column),
            "apply_quality_to_main_meter": bool(args.apply_quality_to_main_meter),
            "force_keep": as_sorted_list(force_keep),
            "force_drop": as_sorted_list(force_drop),
            "dry_run": bool(args.dry_run),
            "activate_scope": bool((not args.dry_run) and (not args.no_activate)),
        },
        "results": {
            "meters_present": meters_present,
            "meters_kept": as_sorted_list(keep_meters),
            "meters_excluded_pre03": as_sorted_list(set(meters_present) - keep_meters),
            "n_rows_input": n_total,
            "n_rows_output": n_keep,
            "pct_rows_kept": round(100 * n_keep / max(n_total, 1), 4),
            "scope_activation": {"activated": bool(activation), "files": activation},
            "audit_files": [
                "audits/05_meter_scope_preselection.csv",
                "audits/05_row_filter_summary.csv",
                "audits/05_preliminary_energy_quality_by_meter.csv",
                "audits/05_preliminary_sentinel_by_meter_variable.csv",
            ],
        },
        "status": "success",
    }

    with open(MANIFESTS_DIR / "05_select_meter_scope_candidate_only_params.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Saved: audits/05_meter_scope_preselection.csv")
    logger.info("Saved: audits/05_row_filter_summary.csv")
    logger.info("Saved: audits/05_preliminary_energy_quality_by_meter.csv")
    logger.info("Saved: audits/05_preliminary_sentinel_by_meter_variable.csv")
    logger.info("Saved: manifests/05_select_meter_scope_candidate_only_params.json")
    logger.info("=== 05_select_meter_scope_candidate.py completed ===")


if __name__ == "__main__":
    main()
