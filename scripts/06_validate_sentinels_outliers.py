#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 06_validate_sentinels_outliers.py

File type: Data preprocessing script (pipeline stage 06: validation, 1-min alignment, real gaps, sentinels and outliers)

Purpose:
    Validation, alignment, real gaps, sentinels and outliers.

    Reads the 4 per-type CSVs produced by the scope pre-selection (script 05) and:
      1. Counts the variables (columns) per CSV.
      2. Lists the columns (and the type of each).
      3. Checks the decimal separator (comma vs dot).
      4. Converts NaN -> 0 in the numeric columns (upstream of the alignment).
      5. ALIGNS everything to a COMMON 1-min GRID, with configurable or automatic origin
         (by default, the latest start among meters; derived pseudo-meters, e.g. P01=100,
         are excluded from the origin computation; see --pseudo-meter-ids).
         Mean per minute; minutes without reading remain GAPS.
      The categories are located and saved IN ORDER, in 3 SEPARATE files:
      6. REAL ALIGNMENT GAPS: minutes without reading after the common 1-min grid, located in
         the aligned raw data before any conversion of sentinel/outlier to NaN.
         -> 06_gap_locations.csv.
      7. SENTINELS: (a) any point equal to the exact global floor of the variable, even
         isolated, and (b) approximate/almost horizontal lower plateau, confirmed by a run
         >=3 and then applied to all points of the same local lower group. For confirmed
         blocks, one finite sample at each edge is also removed, so that the first/last
         point of the sentinel does not remain as an anchor for the filling. Only LOCATES;
         does not convert to NaN at this step.
         -> 06_sentinel_locations.csv.
      8. OUTLIERS: short points outside the robust moving median, located after masking
         sentinels and gap edges only in a local working copy. Only LOCATES.
         -> 06_outlier_locations.csv.
      9. AT THE END, after saving the three locations: sentinels + outliers BECOME NaN in
         the consolidated file for filling. Real gaps were already NaN by absence of reading.

    Rules: figures read from figdata/ -> this script COPIES the source files there; tables
    read from tabdata/ -> this script writes the DATA there (the .tex is produced by script 22).
    Never audits/.

    Inputs (from script 05):
        data/interim/05_{primarias,secundarias,distorcao_meta,harmonicas}.csv

    Outputs:
        data/interim/06_{primarias,secundarias,distorcao_meta,harmonicas}.csv   (aligned)
        data/interim/06_*_sentinelas.csv . 06_*_limpo.csv
        data/interim/06_para_preencher.csv                       (consolidated, 177 vars)
        audits/06_sentinel_locations.csv . 06_outlier_locations.csv . 06_gap_locations.csv   (3 locations)
        audits/06_variable_inventory.csv . 06_columns_long.csv . 06_alignment_coverage.csv
        tabdata/06_sentinels_by_variable.csv . 06_sentinels_currents.csv . 06_outliers.csv
        figdata/06_{primarias,secundarias,para_preencher,sentinel_locations,outlier_locations,gap_locations}
        logs/06_validate_and_zero.log . manifests/06_validate_and_zero_params.json

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

import argparse
import json
import logging
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR    = Path(__file__).resolve().parent


def resolve_project_root() -> Path:
    """Repository root: the parent of the directory that contains this script."""
    return SCRIPT_DIR.parent


PROJECT_ROOT  = resolve_project_root()

INTERIM_DIR   = PROJECT_ROOT / "data" / "interim"
AUDITS_DIR    = PROJECT_ROOT / "audits"
TABDATA_DIR   = PROJECT_ROOT / "tabdata"
FIGDATA_DIR   = PROJECT_ROOT / "figdata"
LOGS_DIR      = PROJECT_ROOT / "logs"
MANIFESTS_DIR = PROJECT_ROOT / "manifests"

CSV_INPUTS = {
    "primarias":     "05_primarias.csv",
    "secundarias":   "05_secundarias.csv",
    "distorcao_meta": "05_distorcao_meta.csv",
    "harmonicas":    "05_harmonicas.csv",
}

TIME_COL, METER_COL = "time", "meter_id"
META_NONNUMERIC = {"time", "tag_meter_id", "datetime_read"}
# per-reading meta that loses meaning after gridding (dropped in the alignment)
META_DROP_ON_ALIGN = {"Unnamed: 0", "datetime_read", "tag_meter_id"}

# Common time grid
GRID_FREQ   = "1min"
# Grid origin.
# - "auto_latest_start": starts at the latest start among meters, rounded up.
# - "auto_first_start": starts at the first global timestamp, rounded down.
# - "manual": uses GRID_ORIGIN or --grid-origin.
GRID_ORIGIN_MODE = "auto_latest_start"
GRID_ORIGIN = None

# General meter for the physical lock.
# Physical lock by the general meter (M1 = M_G): ON by default, since it is the correct
# result (without it, M11/M15/M16 show data where the general meter has no reading). To
# disable: --main-meter-id none.
MAIN_METER_ID = 1

# DERIVED pseudo-meters (e.g. P_other/P01 = meter_id 100, aggregated in script 05).
# They are not physical measurements, so they must NOT dictate the grid origin (otherwise
# their later start shifts everyone's grid). They are aligned normally; they are only
# left out of the auto_latest_start computation. Comma-separated list; 'none' = none.
PSEUDO_METER_IDS = "100"

RE_COMMA_DECIMAL = re.compile(r"^-?\d+,\d+$")


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("06_validate_and_zero")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8"); fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


def save_table_data(df, name):
    """Rule (ii): the table DATA go to tabdata/, NEVER audits/.
    The .tex is generated by script 22 (which reads only from tabdata/)."""
    TABDATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABDATA_DIR / f"{name}.csv", index=False)


SENT_MIN_RUN = 3      # minimum run to confirm an approximate lower plateau
CURRENT_TRIO = ["i_an", "i_bn", "i_cn"]   # current: the sentinel is detected on the 3 phases together
SENT_EDGE_PAD = 1      # removes 1 finite sample before/after confirmed blocks
SENT_EDGE_PAD_MIN_RUN = 3  # padding only in blocks with >=3 points, avoiding enlarging singletons




def contiguous_runs(mask: np.ndarray, min_run: int = 1):
    """Returns contiguous True blocks as (start, end_inclusive) pairs."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not mask.any():
        return []
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends) if (e - s + 1) >= min_run]


def expand_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Expands a boolean mask to protect gap/sentinel edges."""
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0 or mask.size == 0 or not mask.any():
        return mask.copy()
    out = mask.copy()
    idx = np.flatnonzero(mask)
    n = len(mask)
    for k in range(1, radius + 1):
        left = idx - k
        right = idx + k
        out[left[left >= 0]] = True
        out[right[right < n]] = True
    return out



def pad_sentinel_edges(core_mask: np.ndarray,
                       values: np.ndarray,
                       radius: int = SENT_EDGE_PAD,
                       min_run_for_pad: int = SENT_EDGE_PAD_MIN_RUN) -> tuple[np.ndarray, np.ndarray]:
    """Includes the first and last samples adjacent to sentinel blocks.

    The floor/plateau detector identifies the core of the sentinel. In real series,
    the entry and exit of the plateau may appear as a transition sample, slightly
    above the floor threshold. If such samples are preserved, they become anchors
    of the filling and reappear as bars/valleys in the figures.

    Rule: for each confirmed block of length >= min_run_for_pad, up to `radius`
    finite point(s) immediately before and after the block are marked. Points
    already missing due to a real gap are not marked as sentinel.
    """
    core_mask = np.asarray(core_mask, dtype=bool)
    values = np.asarray(values, dtype=float)
    padded = core_mask.copy()
    edge_mask = np.zeros(len(core_mask), dtype=bool)

    if radius <= 0 or core_mask.size == 0 or not core_mask.any():
        return padded, edge_mask

    finite = np.isfinite(values)
    n = len(core_mask)
    for rs, re_ in contiguous_runs(core_mask, min_run_for_pad):
        left0 = max(0, rs - radius)
        for xi in range(left0, rs):
            if finite[xi] and not padded[xi]:
                padded[xi] = True
                edge_mask[xi] = True
        right1 = min(n - 1, re_ + radius)
        for xi in range(re_ + 1, right1 + 1):
            if finite[xi] and not padded[xi]:
                padded[xi] = True
                edge_mask[xi] = True

    return padded, edge_mask

def discover_global_placeholder_values(aligned: pd.DataFrame, elec_cols: list[str], logger: logging.Logger) -> dict:
    """Discovers exact floor values per variable using dominant global repetition.

    The rule avoids depending on per-meter confirmation. The value is discovered on the
    whole variable because placeholders appear as highly repeated global constants,
    whereas physical values rarely repeat with 6 decimals over thousands of samples.
    The application remains local: only real runs are marked in the meter x variable pair.
    """
    out = {}
    rows = []
    for c in elec_cols:
        vals = aligned[c].to_numpy(dtype=float)
        fin = vals[np.isfinite(vals)]
        if len(fin) < 50:
            out[c] = []
            continue
        rounded = np.round(fin, 6)
        counts = pd.Series(rounded).value_counts()
        if len(counts) < 2:
            out[c] = []
            continue
        second = int(counts.iloc[1]) if len(counts) > 1 else 0
        cand = []
        for value, count in counts.head(5).items():
            count = int(count)
            # Strong dominance: captures constant placeholders and avoids occasional physical values.
            if count >= max(50, int(0.005 * len(fin))) and count >= max(20 * max(second, 1), 50):
                cand.append(float(value))
                rows.append({
                    "variavel": c,
                    "valor_piso_global": float(value),
                    "n_ocorrencias": count,
                    "n_finitos": int(len(fin)),
                    "pct_ocorrencias": round(100 * count / max(len(fin), 1), 4),
                    "segundo_maior_count": second,
                    "criterio": "repeticao_global_dominante_6casas"
                })
        out[c] = cand
    cand_df = pd.DataFrame(rows, columns=[
        "variavel", "valor_piso_global", "n_ocorrencias", "n_finitos",
        "pct_ocorrencias", "segundo_maior_count", "criterio"
    ])
    cand_df.to_csv(AUDITS_DIR / "06_sentinel_global_candidates.csv", index=False)
    logger.info("Global sentinel candidates -> 06_sentinel_global_candidates.csv (%d variables)" %
                (cand_df["variavel"].nunique() if len(cand_df) else 0))
    return out


def allow_low_horizontal_detector(var: str, values_without_exact: np.ndarray) -> bool:
    """Controls when the lower-plateau detector may complement the exact floor."""
    if var.startswith(("v_", "thdv", "hrm_v", "cos")):
        return False
    if re.match(r"^[pq]_", var):
        fin = values_without_exact[np.isfinite(values_without_exact)]
        if len(fin) < 30:
            return False
        # For active/reactive power with a negative operating band, the floor value is usually
        # an exact placeholder near zero; the numeric lower detector would mark the negative
        # operation itself as sentinel. In such cases only the exact floor is used.
        if float(np.nanmedian(fin)) < 0:
            return False
    return True


def _near_physical_zero(var: str, value: float, scale: float) -> bool:
    """Protects physical zero/low values when the channel may be without consumption.

    For an approximate plateau, values near zero in p/q/s/cos are not classified
    as sentinel just for being low.
    """
    if not re.match(r"^(p|q|s|cos)_", var):
        return False
    ztol = max(1e-9, 0.002 * max(float(scale), 1.0))
    return abs(float(value)) <= ztol


def detect_low_horizontal_mask(v, var: str, existing_mask=None, min_run=SENT_MIN_RUN):
    """Detects an almost horizontal lower plateau of the series itself.

    Updated rule:
      1. the lower group must be vertically separated from the main band;
      2. the group must contain at least one horizontal run with >=3 points;
      3. once the artificial lower group is confirmed, all points of that same
         local lower group are marked, including isolated points.

    This third condition removes the problem observed in the figures: isolated points
    at the same floor as the sentinel remained as valid observations and became
    anchors of the filling.
    """
    v = np.asarray(v, dtype=float)
    if existing_mask is None:
        existing_mask = np.zeros(len(v), dtype=bool)
    existing_mask = np.asarray(existing_mask, dtype=bool)
    out = np.zeros(len(v), dtype=bool)

    fin = v[np.isfinite(v) & ~existing_mask]
    if len(fin) < 30:
        return out, None
    if not allow_low_horizontal_detector(var, fin):
        return out, None

    uniq = np.unique(np.round(fin, 3))
    if len(uniq) < 2:
        return out, None

    q05, q25, q75, q95 = np.nanpercentile(fin, [5, 25, 75, 95])
    iqr = float(q75 - q25)
    span = float(q95 - q05)
    scale = max(iqr, span, 1e-9)

    best = None
    for i, gap in enumerate(np.diff(uniq)):
        if gap <= 0:
            continue
        lower = float(uniq[i])
        upper = float(uniq[i + 1])
        if _near_physical_zero(var, lower, scale):
            continue
        lower_count = int((fin <= lower + 1e-9).sum())
        upper_count = int((fin >= upper - 1e-9).sum())
        ratio = lower_count / max(len(fin), 1)
        if lower_count < min_run or upper_count < min_run:
            continue
        if ratio < 0.005 or ratio > 0.95:
            continue
        sep_min = max(0.03 * scale, 1e-6)
        if float(gap) < sep_min:
            continue
        score = float(gap) * min(ratio, 1.0 - ratio)
        if best is None or score > best["score"]:
            best = {
                "lower": lower,
                "upper": upper,
                "gap_y": float(gap),
                "lower_count": lower_count,
                "upper_count": upper_count,
                "ratio_lower": float(ratio),
                "iqr": iqr,
                "span_5_95": span,
                "score": score,
            }

    if best is None:
        return out, None

    candidate = np.isfinite(v) & ~existing_mask & (v <= best["lower"] + 1e-9)
    confirmed = np.zeros(len(v), dtype=bool)
    for rs, re_ in contiguous_runs(candidate, min_run):
        confirmed[rs:re_ + 1] = True
    if not confirmed.any():
        return np.zeros(len(v), dtype=bool), None

    # Marks the complete lower group, including isolated points.
    out[candidate] = True
    best["confirmed_run_points"] = int(confirmed.sum())
    best["isolated_points_added"] = int(candidate.sum() - confirmed.sum())
    return out, best

def detect_sentinel_mask(v, var: str, global_values: list[float], min_run=SENT_MIN_RUN):
    """Joins two pieces of sentinel evidence.

    Updated rule:
      1. dominant exact global floor: any occurrence is a sentinel, including an
         isolated point or a run shorter than 3;
      2. approximate/local lower plateau: first requires confirmation by a run
         >=3; then marks all points of the same local lower group.
    """
    v = np.asarray(v, dtype=float)
    exact_mask = np.zeros(len(v), dtype=bool)
    used_values = []
    for cand in global_values:
        hit = np.isfinite(v) & np.isclose(np.round(v, 6), float(cand), atol=5e-6, rtol=0)
        if hit.any():
            exact_mask |= hit
            used_values.append(float(cand))

    low_mask, low_meta = detect_low_horizontal_mask(v, var, existing_mask=exact_mask, min_run=min_run)
    final_mask = exact_mask | low_mask

    criteria = []
    if exact_mask.any():
        criteria.append("piso_global_exato_dominante_inclusive_isolado")
    if low_mask.any():
        criteria.append("patamar_inferior_horizontal_local_grupo_completo")
    return final_mask, exact_mask, low_mask, {
        "criterio": "+".join(criteria) if criteria else "",
        "valores_globais_usados": sorted(set(used_values)),
        "low_meta": low_meta,
    }



# ─────────────────────────────────────────────────────────────────────────────
# Synchronised current sentinel: conservative segmental version
# ─────────────────────────────────────────────────────────────────────────────
# v3 was permissive because it voted per point (>=2 phases) in a learned zone.
# That produced false positives in meters with a real low/stable regime and still
# left residues in multi-level bands. The rule below is per SEGMENT:
#   1. uses exact global-floor anchors only as seeds;
#   2. joins nearby anchors to form a candidate window;
#   3. accepts the window only if the plateau has robust contrast against the
#      operational context in at least two currents;
#   4. once accepted, removes the whole window, covering multi-level floors.
# Without a global anchor, the fallback remains conservative: plateau in the 3 currents.

SYNC_ANCHOR_BRIDGE_GAP = 35      # joins floor seeds separated by up to 35 min
SYNC_SEGMENT_CONTEXT = 360       # points before/after to estimate the operational regime
SYNC_SEGMENT_MIN_RUN = 3
SYNC_SEGMENT_MIN_ANCHORS = 1
SYNC_SEGMENT_MIN_PHASES = 2
SYNC_SEGMENT_Z = 5.0             # minimum separation on the local robust scale
SYNC_SEGMENT_AMP_FRAC = 0.025    # minimum separation as a fraction of the operational amplitude
SYNC_SEGMENT_FLAT_RATIO = 0.85   # the plateau must be narrower than the context
SYNC_FALLBACK_MIN_PHASES = 3

OUTLIER_MAX_RUN = 5
OUTLIER_GLOBAL_K = 5.0           # extreme global envelope = OUTLIER_GLOBAL_K x IQR (short runs only)
OUTLIER_FORBIDDEN_RADIUS = 3
HAMPEL_WINDOW = 61
HAMPEL_MIN_PERIODS = 21
HAMPEL_MIN_JUMP_FRAC = 0.03


def _mad_scale(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)))
    return float(1.4826 * mad)


def _safe_iqr(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return 0.0
    q1, q3 = np.nanpercentile(x, [25, 75])
    return float(max(q3 - q1, 0.0))


def close_short_false_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Closes small False intervals between True seeds."""
    mask = np.asarray(mask, dtype=bool)
    out = mask.copy()
    if max_gap <= 0 or mask.size == 0 or not mask.any():
        return out
    runs = contiguous_runs(mask, 1)
    if len(runs) <= 1:
        return out
    for (_, e1), (s2, _) in zip(runs[:-1], runs[1:]):
        gap = s2 - e1 - 1
        if 0 < gap <= max_gap:
            out[e1 + 1:s2] = True
    return out


def _context_values(v: np.ndarray, rs: int, re_: int, exclude: np.ndarray) -> np.ndarray:
    """Operational context around the segment, with global fallback."""
    n = len(v)
    left0 = max(0, rs - SYNC_SEGMENT_CONTEXT)
    right1 = min(n, re_ + 1 + SYNC_SEGMENT_CONTEXT)
    ctx_mask = np.zeros(n, dtype=bool)
    ctx_mask[left0:rs] = True
    ctx_mask[re_ + 1:right1] = True
    ctx_mask &= ~exclude
    ctx = v[ctx_mask & np.isfinite(v)]
    if len(ctx) >= 30:
        return ctx
    # global fallback outside the candidate segment
    fallback = np.ones(n, dtype=bool)
    fallback[rs:re_ + 1] = False
    fallback &= ~exclude
    return v[fallback & np.isfinite(v)]


def _segment_phase_is_placeholder(v: np.ndarray, rs: int, re_: int, exclude: np.ndarray) -> tuple[bool, dict]:
    """Tests whether the segment is an anomalous plateau for one current phase."""
    v = np.asarray(v, dtype=float)
    seg = v[rs:re_ + 1]
    seg = seg[np.isfinite(seg)]
    ctx = _context_values(v, rs, re_, exclude)
    meta = {
        "n_seg": int(len(seg)),
        "n_ctx": int(len(ctx)),
        "med_seg": np.nan,
        "med_ctx": np.nan,
        "diff": np.nan,
        "scale": np.nan,
        "amp_ctx": np.nan,
        "iqr_seg": np.nan,
        "iqr_ctx": np.nan,
        "passed": False,
    }
    if len(seg) < SYNC_SEGMENT_MIN_RUN or len(ctx) < 30:
        return False, meta

    med_seg = float(np.nanmedian(seg))
    med_ctx = float(np.nanmedian(ctx))
    iqr_seg = _safe_iqr(seg)
    iqr_ctx = _safe_iqr(ctx)
    q05, q95 = np.nanpercentile(ctx, [5, 95])
    amp_ctx = float(max(q95 - q05, iqr_ctx, 1e-9))
    scale = max(_mad_scale(ctx), 0.15 * iqr_ctx, 1e-6)
    diff = abs(med_seg - med_ctx)

    contrast_ok = diff >= max(SYNC_SEGMENT_Z * scale, SYNC_SEGMENT_AMP_FRAC * amp_ctx)
    flat_ok = iqr_seg <= max(SYNC_SEGMENT_FLAT_RATIO * max(iqr_ctx, 1e-9), 3.0 * scale)

    passed = bool(contrast_ok and flat_ok)
    meta.update({
        "med_seg": round(med_seg, 9),
        "med_ctx": round(med_ctx, 9),
        "diff": round(diff, 9),
        "scale": round(scale, 9),
        "amp_ctx": round(amp_ctx, 9),
        "iqr_seg": round(iqr_seg, 9),
        "iqr_ctx": round(iqr_ctx, 9),
        "passed": passed,
    })
    return passed, meta


def _fallback_synchronized_plateau(vmap: dict) -> np.ndarray:
    """Conservative fallback when there is no exact global floor."""
    phases = list(vmap)
    n = len(vmap[phases[0]])
    masks = []
    for c in phases:
        lm, _ = detect_low_horizontal_mask(vmap[c], c, existing_mask=None, min_run=SENT_MIN_RUN)
        masks.append(np.asarray(lm, dtype=bool))
    if len(masks) < SYNC_FALLBACK_MIN_PHASES:
        return np.zeros(n, dtype=bool)
    stacked = np.vstack(masks)
    sync = stacked.sum(axis=0) >= SYNC_FALLBACK_MIN_PHASES
    out = np.zeros(n, dtype=bool)
    for rs, re_ in contiguous_runs(sync, SENT_MIN_RUN):
        out[rs:re_ + 1] = True
    return out


def detect_current_sentinel_synchronized(vmap: dict, floors_by_var: dict) -> np.ndarray:
    """Detects synchronised 'no reading' in the currents, per meter.

    The decision unit is the temporal segment, not the isolated point. This avoids
    two errors observed in v3: (i) false positive in a meter with a real low/stable
    regime; (ii) residues in multi-level bands, since the whole accepted segment
    is removed.
    """
    phases = list(vmap)
    n = len(vmap[phases[0]])
    finite_all = np.ones(n, dtype=bool)
    for c in phases:
        finite_all &= np.isfinite(vmap[c])
    if not finite_all.any():
        return np.zeros(n, dtype=bool)

    # Exact seeds: any phase at the global floor, with the three currents finite.
    anchor = np.zeros(n, dtype=bool)
    for c in phases:
        v = np.asarray(vmap[c], dtype=float)
        for f in floors_by_var.get(c, []):
            anchor |= np.isfinite(v) & np.isclose(np.round(v, 6), float(f), atol=5e-6, rtol=0)
    anchor &= finite_all

    if not anchor.any():
        return _fallback_synchronized_plateau(vmap)

    # Joins nearby seeds, but acceptance requires robust contrast in the segment.
    seeds = close_short_false_gaps(anchor, max_gap=SYNC_ANCHOR_BRIDGE_GAP)
    # An EXACT floor is always 'no reading', even without contrast (a 100% placeholder meter,
    # e.g. M8, has no operational context to contrast). The segments below only
    # EXTEND the anchors to cover multi-level bands with jitter (e.g. M16 at 196/197).
    out = anchor.copy()

    # Avoids using the segment/candidates themselves as context.
    for rs, re_ in contiguous_runs(seeds, SYNC_SEGMENT_MIN_RUN):
        if int(anchor[rs:re_ + 1].sum()) < SYNC_SEGMENT_MIN_ANCHORS:
            continue
        seg_mask = np.zeros(n, dtype=bool)
        seg_mask[rs:re_ + 1] = True
        # Excludes only the current segment from the context. Does not exclude all seeds,
        # so as not to remove too much context in meters with several short failures.
        passed_phases = 0
        for c in phases:
            ok, _ = _segment_phase_is_placeholder(np.asarray(vmap[c], dtype=float), rs, re_, seg_mask)
            passed_phases += int(ok)
        if passed_phases >= SYNC_SEGMENT_MIN_PHASES:
            out[rs:re_ + 1] = True

    return out

def phase_group_key(var: str):
    """(group, phase) for ENERGY variables groupable by phase triplet; None otherwise.
    Voltage (v, thdv, hrm_v) -> None (does not receive a sentinel). Each quantity/harmonic order
    becomes a triplet (a/b/c) whose floor is detected on the 3 phases TOGETHER."""
    mm = re.match(r"^(i)_([abc])n$", var)
    if mm:
        return (f"{mm.group(1)}_n", mm.group(2))
    mm = re.match(r"^(p|q|s|cos|thdi)_([abc])$", var)
    if mm:
        return (mm.group(1), mm.group(2))
    mm = re.match(r"^(hrm_i)_([abc])n_(\d+)$", var)
    if mm:
        return (f"{mm.group(1)}_{mm.group(3)}", mm.group(2))
    return None



def detect_outlier_mask(vals, sentinel_mask=None, gap_mask=None, k_iqr: float = 8.0):
    """Detects point outliers by local Hampel + extreme global envelope.

    Correction with respect to v3: the reference gap does NOT block the marking of the
    candidate point. In some active/apparent power channels, absurd values are finite
    exactly when the current-based gap mask is active. Therefore `gap_mask` protects
    the reference window but does not prevent a finite cell of `vals` from being
    marked as an outlier. Points already sentinel remain excluded, since they will be
    removed by the sentinel step itself.
    """
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    if sentinel_mask is None:
        sentinel_mask = np.zeros(n, dtype=bool)
    if gap_mask is None:
        gap_mask = np.zeros(n, dtype=bool)

    sentinel_mask = np.asarray(sentinel_mask, dtype=bool)
    gap_mask = np.asarray(gap_mask, dtype=bool)

    # Reference: removes sentinel/gap and their edges.
    reference_forbidden = expand_mask(sentinel_mask | gap_mask, radius=OUTLIER_FORBIDDEN_RADIUS)

    # Candidate: finite value that is not yet a sentinel. Do not use gap_mask here.
    candidate = np.isfinite(vals) & ~sentinel_mask
    reference = np.isfinite(vals) & ~reference_forbidden
    fin = vals[reference]
    if len(fin) < max(30, HAMPEL_MIN_PERIODS):
        return np.zeros(n, dtype=bool)

    global_iqr = _safe_iqr(fin)
    global_mad = _mad_scale(fin)
    global_scale = max(global_iqr, global_mad, 1e-9)
    min_jump = max(HAMPEL_MIN_JUMP_FRAC * global_scale, 1e-9)

    # Extreme global envelope. The previous 12*IQR was loose and let through isolated
    # peaks ~7-8*IQR above the regime (M05/M06/M07). OUTLIER_GLOBAL_K*IQR is used,
    # protected by the short-run filter (<=OUTLIER_MAX_RUN), which prevents removing
    # a real regime change. Wide-range meters (M01/M02, general/large loads)
    # are not affected, since their peaks stay inside the envelope.
    q1, q3 = np.nanpercentile(fin, [25, 75])
    iqr = float(max(q3 - q1, 0.0))
    if iqr <= 0:
        medg = float(np.nanmedian(fin))
        madg = max(_mad_scale(fin), 1e-9)
        glo, ghi = medg - OUTLIER_GLOBAL_K * madg, medg + OUTLIER_GLOBAL_K * madg
    else:
        glo, ghi = float(q1 - OUTLIER_GLOBAL_K * iqr), float(q3 + OUTLIER_GLOBAL_K * iqr)
    global_extreme = candidate & ((vals < glo) | (vals > ghi))

    y = pd.Series(vals.astype(float))
    y_for_roll = y.mask(reference_forbidden)
    win = int(HAMPEL_WINDOW)
    if win % 2 == 0:
        win += 1
    minp = min(int(HAMPEL_MIN_PERIODS), win)

    med = y_for_roll.rolling(window=win, center=True, min_periods=minp).median()
    absdev = (y_for_roll - med).abs()
    mad = absdev.rolling(window=win, center=True, min_periods=minp).median()
    scale = 1.4826 * mad.to_numpy(dtype=float)
    medv = med.to_numpy(dtype=float)

    residual = np.abs(vals - medv)
    scale_floor = max(0.05 * global_scale, 1e-9)
    z = residual / np.maximum(scale, scale_floor)

    local_extreme = (
        candidate
        & np.isfinite(medv)
        & (z >= float(k_iqr))
        & (residual >= min_jump)
    )

    raw_mask = local_extreme | global_extreme

    # Short-run rule: removes only a SHORT EVENT (isolated spike). Sustained bands
    # (e.g. noisy start-up of M06/M07) are NOT treated as outliers; they are real data.
    out = np.zeros(n, dtype=bool)
    for rs, re_ in contiguous_runs(raw_mask, 1):
        if (re_ - rs + 1) <= OUTLIER_MAX_RUN:
            out[rs:re_ + 1] = True
    return out

def col_phase(c):
    m = (re.match(r"^(?:i|v)_([abc])n$", c) or re.match(r"^(?:p|q|s|cos|thdi|thdv)_([abc])$", c)
         or re.match(r"^hrm_[iv]_([abc])n_", c))
    return m.group(1) if m else None


def is_energy_side(c):
    """ENERGY variables (current side)."""
    return bool(re.match(r"^(i_[abc]n|p_[abc]|q_[abc]|s_[abc]|cos_[abc]|thdi_[abc]|hrm_i_[abc]n_)", c))


def is_voltage(c):
    """VOLTAGE variables (v/thdv/hrm_v). They receive the 'no reading' removal
    (synchronised from the current + M1 lock) only in the acquisition-blackout minutes;
    otherwise they are preserved (the grid may stay energised without load)."""
    return bool(re.match(r"^(v_[abc]n|thdv_[abc]|hrm_v_[abc]n_)", c))


def comma_decimal_cols(df):
    out = []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            sample = df[c].dropna().astype(str).head(500)
            if len(sample) and sample.str.match(RE_COMMA_DECIMAL).mean() > 0.5:
                out.append(c)
    return out



def parse_optional_int(text_value):
    """Converts an optional argument to int; accepts None/none/null to disable."""
    if text_value is None:
        return None
    s = str(text_value).strip().lower()
    if s in {"", "none", "null", "na", "nan", "false", "off", "disable", "disabled"}:
        return None
    return int(s)


def parse_id_set(text_value) -> set:
    """Converts 'a,b,c' into {int}. Accepts None/none/'' for an empty set."""
    if text_value is None:
        return set()
    s = str(text_value).strip().lower()
    if s in {"", "none", "null", "na", "nan"}:
        return set()
    return {int(x) for x in s.replace(";", ",").split(",") if x.strip()}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Validation, alignment to a common grid, primary location of gaps, "
            "sentinels and outliers."
        )
    )
    p.add_argument(
        "--grid-origin-mode",
        choices=["auto_latest_start", "auto_first_start", "manual"],
        default=GRID_ORIGIN_MODE,
        help=(
            "Grid-origin mode. auto_latest_start uses the latest start among "
            "meters; auto_first_start uses the first global timestamp; manual uses --grid-origin. "
            "Default: auto_latest_start."
        ),
    )
    p.add_argument(
        "--grid-origin",
        default=GRID_ORIGIN,
        help=(
            "Manual grid origin, e.g. '2020-10-22 15:21:00+00:00'. "
            "Used when --grid-origin-mode manual."
        ),
    )
    p.add_argument(
        "--main-meter-id",
        default=MAIN_METER_ID,
        help=(
            "ID of the general meter for the physical lock. Default: 1 (lock enabled). "
            "Use 'none' to disable."
        ),
    )
    p.add_argument(
        "--pseudo-meter-ids",
        default=PSEUDO_METER_IDS,
        help=(
            "IDs of derived pseudo-meters (e.g. P01=100), comma-separated, "
            "excluded from the grid-origin computation (auto_latest_start). They are aligned "
            "normally. 'none' for none. Default: 100."
        ),
    )
    return p


def resolve_grid_origin(full: pd.DataFrame,
                        *,
                        mode: str,
                        explicit_origin,
                        grid_freq: str,
                        logger: logging.Logger,
                        pseudo_meter_ids=None) -> pd.Timestamp:
    """Resolves the grid origin without tying the script to an experiment date.

    pseudo_meter_ids: derived meters (e.g. P01) that must NOT dictate the origin;
    they are excluded from the auto_latest_start computation (but are still aligned)."""
    if mode == "auto_latest_start":
        starts = (
            full.dropna(subset=[TIME_COL])
                .groupby(METER_COL)[TIME_COL]
                .min()
        )
        if pseudo_meter_ids:
            drop = [m for m in starts.index if m in pseudo_meter_ids]
            if drop:
                starts = starts.drop(labels=drop)
                logger.info(f"Grid origin: pseudo-meters {sorted(drop)} excluded from auto_latest_start.")
        if starts.empty:
            logger.error("There are no valid timestamps to compute the automatic grid origin.")
            sys.exit(1)
        return starts.max().ceil(grid_freq)

    if mode == "auto_first_start":
        t0 = full[TIME_COL].dropna().min()
        if pd.isna(t0):
            logger.error("There are no valid timestamps to compute the automatic grid origin.")
            sys.exit(1)
        return t0.floor(grid_freq)

    if mode == "manual":
        if explicit_origin is None or str(explicit_origin).strip().lower() in {"", "none", "null"}:
            logger.error("--grid-origin-mode manual requires --grid-origin.")
            sys.exit(1)
        t0 = pd.Timestamp(explicit_origin)
        if t0.tzinfo is None:
            t0 = t0.tz_localize("UTC")
        else:
            t0 = t0.tz_convert("UTC")
        return t0

    logger.error(f"Invalid grid-origin mode: {mode!r}")
    sys.exit(1)

def main() -> None:
    args = build_parser().parse_args()
    grid_origin_mode = str(args.grid_origin_mode)
    explicit_grid_origin = args.grid_origin
    main_meter_id = parse_optional_int(args.main_meter_id)
    pseudo_meter_ids = parse_id_set(args.pseudo_meter_ids)

    logger = setup_logger(LOGS_DIR / "06_validate_and_zero.log")
    logger.info("=== 06_validate_sentinels_outliers.py: multi-source industrial NILM ===")
    logger.info(f"Run timestamp: {datetime.now().isoformat()}")
    for d in (INTERIM_DIR, AUDITS_DIR, MANIFESTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ---------- 1-4. Validation + NaN->0, and reading of the 4 CSVs ----------------
    dfs, inventory_rows, columns_rows, type_vars = {}, [], [], {}
    for tipo, infile in CSV_INPUTS.items():
        path = INTERIM_DIR / infile
        if not path.exists():
            logger.error(f"Input not found: {path}; run scripts 02 and 05 first")
            sys.exit(1)
        df = pd.read_csv(path, low_memory=False)
        for c in df.columns:
            columns_rows.append({"csv": tipo, "column": c, "dtype": str(df[c].dtype),
                                 "n_nan": int(df[c].isna().sum())})
        text_cols  = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        comma_cols = comma_decimal_cols(df)
        num_cols   = [c for c in df.columns
                      if c not in META_NONNUMERIC and pd.api.types.is_numeric_dtype(df[c])]
        n_nan_before = int(df[num_cols].isna().sum().sum())
        df[num_cols] = df[num_cols].fillna(0)
        inventory_rows.append({
            "csv": tipo, "n_variables": df.shape[1], "n_rows": df.shape[0],
            "n_numeric_cols": len(num_cols), "n_text_cols": len(text_cols),
            "n_comma_decimal_cols": len(comma_cols), "decimal_ok": len(comma_cols) == 0,
            "n_nan_before": n_nan_before,
        })
        logger.info(f"  {tipo:14s}: {df.shape[1]:3d} variables | text={len(text_cols)} | "
                    f"decimal-comma={len(comma_cols)} | NaN->0={n_nan_before}")
        dfs[tipo] = df
        # ELECTRICAL variables of this type (excludes meta)
        type_vars[tipo] = [c for c in df.columns
                           if c not in META_NONNUMERIC and c not in META_DROP_ON_ALIGN
                           and c != METER_COL]

    # ---------- 5. Alignment to the common 1-minute grid -------------------
    # Rebuilds the full table (CSVs aligned by row order; meta in the distortion file)
    full = pd.concat([dfs[t].reset_index(drop=True) for t in CSV_INPUTS], axis=1)
    full = full.loc[:, ~full.columns.duplicated()]
    full[TIME_COL] = pd.to_datetime(full[TIME_COL], utc=True)
    meters = sorted(full[METER_COL].dropna().unique().tolist())

    grid_end = full[TIME_COL].max().ceil(GRID_FREQ)
    grid_origin = resolve_grid_origin(
        full,
        mode=grid_origin_mode,
        explicit_origin=explicit_grid_origin,
        grid_freq=GRID_FREQ,
        logger=logger,
        pseudo_meter_ids=pseudo_meter_ids,
    )
    if grid_origin > grid_end:
        logger.error(f"Grid origin ({grid_origin}) is after the end of the data ({grid_end}).")
        sys.exit(1)
    grid = pd.date_range(grid_origin, grid_end, freq=GRID_FREQ, tz="UTC")
    logger.info(f"Common grid: {grid_origin} -> {grid_end}  ({len(grid)} minutes) | "
                f"{len(meters)} meters | mode={grid_origin_mode}")

    elec_cols = sorted({c for vs in type_vars.values() for c in vs})
    full["_t1m"] = full[TIME_COL].dt.floor(GRID_FREQ)
    grp = (full.groupby([METER_COL, "_t1m"])[elec_cols].mean().reset_index())

    frames = []
    for m in meters:
        sub = grp[grp[METER_COL] == m].set_index("_t1m").reindex(grid)
        sub.index.name = TIME_COL
        sub[METER_COL] = m
        frames.append(sub.reset_index())
    aligned = pd.concat(frames, ignore_index=True)

    # ---------- Outputs: 4 aligned CSVs (meter_id + time + variables of the type) -----
    cov_rows = []
    for m in meters:
        sm = aligned[aligned[METER_COL] == m]
        n_data = int(sm["i_an"].notna().sum()) if "i_an" in sm else int(sm[elec_cols[0]].notna().sum())
        cov_rows.append({"meter_id": m, "n_grid_minutes": len(sm),
                         "n_with_data": n_data, "n_gaps": len(sm) - n_data,
                         "pct_with_data": round(100 * n_data / max(len(sm), 1), 2)})

    out_files = {}
    for tipo, vs in type_vars.items():
        cols = [METER_COL, TIME_COL] + vs
        out = aligned[cols]
        path = INTERIM_DIR / f"06_{tipo}.csv"
        out.to_csv(path, index=False)
        out_files[tipo] = {"file": path.name, "n_cols": len(cols), "n_rows": len(out)}
        logger.info(f"  aligned -> 06_{tipo}.csv  ({len(vs)} variables + meter_id,time | {len(out)} rows)")

    # ---------- 6. REAL ALIGNMENT GAPS -----------------------------
    # A gap, at this step, means EXCLUSIVELY a minute without reading after
    # the alignment to the 1-minute grid. Sentinels and outliers are still
    # observed values in the aligned raw data and do NOT enter gap_loc.
    csv_of = {v: tipo for tipo, vs in type_vars.items() for v in vs}   # variable -> CSV
    total_pts = len(aligned)
    meter_idx = {m: (aligned[METER_COL] == m).values for m in meters}

    gloc = []
    for m in meters:
        idx = meter_idx[m]
        tcol = aligned.loc[idx, TIME_COL].values
        if "i_an" in aligned.columns:
            gapm = np.isnan(aligned.loc[idx, "i_an"].values.astype(float))
            ref_col = "i_an"
        else:
            gapm = np.isnan(aligned.loc[idx, elec_cols[0]].values.astype(float))
            ref_col = elec_cols[0]
        dd = np.diff(np.r_[0, gapm.astype(np.int8), 0])
        for gs, ge in zip(np.flatnonzero(dd == 1), np.flatnonzero(dd == -1) - 1):
            gloc.append({METER_COL: m, "variavel_referencia": ref_col,
                         "x_inicio": int(gs), "x_fim": int(ge),
                         "n_pontos": int(ge - gs + 1),
                         "t_inicio": pd.Timestamp(tcol[gs]), "t_fim": pd.Timestamp(tcol[ge]),
                         "tipo_lacuna": "alinhamento_sem_leitura"})
    gap_loc = pd.DataFrame(gloc, columns=[METER_COL, "variavel_referencia", "x_inicio", "x_fim",
                                          "n_pontos", "t_inicio", "t_fim", "tipo_lacuna"])
    gap_loc.to_csv(AUDITS_DIR / "06_gap_locations.csv", index=False)
    logger.info("1) Real alignment gaps -> 06_gap_locations.csv (%d blocks)" % len(gap_loc))

    # ---------- 7. SENTINELS: exact global floor + local lower plateau ----
    # The discovery of the floor value is global per variable, but the marking is always local.
    # Exact global floor: marks any point, even isolated.
    # Approximate plateau: confirms by a run >=3 and then includes isolated points of the
    # same local lower group.
    # Edges: for confirmed blocks (>=3 points), also removes 1 finite point before
    # and 1 finite point after, since those transition points were used as anchors
    # by the filling and reappeared as artificial bars/valleys.
    placeholder_values = discover_global_placeholder_values(aligned, elec_cols, logger)
    sent_count = {c: 0 for c in elec_cols}
    sentinel_masks = {}
    sloc = []

    for m in meters:
        idx = meter_idx[m]
        tcol = aligned.loc[idx, TIME_COL].values

        # SENTINEL: 'no reading' detected in the CURRENT (anchor) and PROPAGATED to ALL
        # ENERGY variables (p, q, s, cos, thdi, harmonics). The current is the anchor:
        # when it freezes (no reading), everything freezes together. Detecting each quantity
        # independently over-marked the harmonics (small/noisy values).
        cur_present = [c for c in CURRENT_TRIO if c in aligned.columns]
        sync_cur = None
        if len(cur_present) == 3:
            vmap = {c: aligned.loc[idx, c].values.astype(float) for c in cur_present}
            sync_cur = detect_current_sentinel_synchronized(
                vmap, {c: placeholder_values.get(c, []) for c in cur_present}
            )

        for c in elec_cols:
            vals = aligned.loc[idx, c].values.astype(float)
            if sync_cur is not None and (is_energy_side(c) or is_voltage(c)):
                # synchronised 'no reading' from the current -> propagates to ENERGY and VOLTAGE
                # (during an acquisition blackout the voltage is also a placeholder). Only where finite.
                core_mask = sync_cur & np.isfinite(vals)
                exact_mask = core_mask.copy()
                low_mask = np.zeros(len(vals), dtype=bool)
                meta = {"criterio": "sem_leitura_sincronizado_corrente_adaptativo"}
            elif is_energy_side(c):
                core_mask, exact_mask, low_mask, meta = detect_sentinel_mask(
                    vals, c, placeholder_values.get(c, []), min_run=SENT_MIN_RUN
                )
            else:
                # Voltage without an available current triplet: subject only to gap/outlier.
                continue
            if not core_mask.any():
                continue

            # Also removes the finite edges of confirmed sentinel blocks.
            # This fixes the case observed in the figures: the first/last point of the
            # plateau was preserved, used as an anchor in the filling and
            # reappeared as a bar/valley in the filled signal.
            smask, edge_mask = pad_sentinel_edges(
                core_mask, vals, radius=SENT_EDGE_PAD, min_run_for_pad=SENT_EDGE_PAD_MIN_RUN
            )

            sentinel_masks[(m, c)] = smask
            sent_count[c] += int(smask.sum())

            for rs, re_ in contiguous_runs(smask, 1):
                block_vals = vals[rs:re_ + 1]
                valor_y = float(np.nanmedian(block_vals))
                exact_overlap = bool(exact_mask[rs:re_ + 1].any())
                low_overlap = bool(low_mask[rs:re_ + 1].any())
                edge_overlap = bool(edge_mask[rs:re_ + 1].any())
                criterio_bloco = []
                if exact_overlap:
                    criterio_bloco.append("piso_global_exato_dominante")
                if low_overlap:
                    criterio_bloco.append("patamar_inferior_horizontal_local")
                if edge_overlap:
                    criterio_bloco.append("borda_expandida_1ponto")
                if not criterio_bloco:
                    criterio_bloco.append(str(meta.get("criterio", "sentinela_mascara_expandida")))
                sloc.append({
                    "csv": csv_of.get(c, ""),
                    "variavel": c,
                    "fase": col_phase(c),
                    METER_COL: m,
                    "valor_y": round(valor_y, 6),
                    "x_inicio": int(rs),
                    "x_fim": int(re_),
                    "n_pontos": int(re_ - rs + 1),
                    "t_inicio": pd.Timestamp(tcol[rs]),
                    "t_fim": pd.Timestamp(tcol[re_]),
                    "criterio": "+".join(criterio_bloco),
                })

    # ---------- 7b. PHYSICAL LOCK BY THE GENERAL METER (M1 = M_G) ----------------
    # System physics: M1 is the general meter. If the general meter has no useful energy
    # signal in a minute, NO sub-meter can have a useful energy signal in that minute.
    # Decision by STRUCTURE (M1), not by local value; fixes M11/M15/M16, which
    # showed data where M1 has no reading.
    #   mg_invalid_time_mask = (all currents of M1 NaN/sentinel)
    #                          OR (all powers of M1 NaN/sentinel)
    # Propagates to ALL ENERGY and VOLTAGE variables of the other meters in the
    # acquisition-blackout minutes (i/p/q/s/cos/thdi/hrm_i + v/thdv/hrm_v): when the
    # general meter does not read, the voltage is also a placeholder. Criterion: sem_leitura_por_medidor_geral_M1.
    MG_METER = main_meter_id
    n_samp = int(meter_idx[meters[0]].sum()) if meters else 0
    if MG_METER is not None and MG_METER in meters and n_samp > 0:
        idx_mg = meter_idx[MG_METER]

        def _mg_all_invalid(cols):
            cols = [c for c in cols if c in aligned.columns]
            if not cols:
                return np.zeros(n_samp, dtype=bool)
            inv = np.ones(n_samp, dtype=bool)
            for c in cols:
                v = aligned.loc[idx_mg, c].values.astype(float)
                sm = sentinel_masks.get((MG_METER, c), np.zeros(n_samp, dtype=bool))
                inv &= (~np.isfinite(v)) | sm   # NaN (gap) or sentinel
            return inv

        mg_invalid = _mg_all_invalid(["i_an", "i_bn", "i_cn"]) | _mg_all_invalid(["p_a", "p_b", "p_c"])

        mg_blocks = mg_pts = 0
        if mg_invalid.any():
            for m in meters:
                if m == MG_METER:
                    continue
                idx = meter_idx[m]
                tcol = aligned.loc[idx, TIME_COL].values
                for c in elec_cols:
                    if not (is_energy_side(c) or is_voltage(c)):
                        continue
                    vals = aligned.loc[idx, c].values.astype(float)
                    add = mg_invalid & np.isfinite(vals)
                    if not add.any():
                        continue
                    prev = sentinel_masks.get((m, c), np.zeros(n_samp, dtype=bool))
                    new_only = add & ~prev
                    sentinel_masks[(m, c)] = prev | add
                    if not new_only.any():
                        continue
                    sent_count[c] += int(new_only.sum())
                    for rs, re_ in contiguous_runs(new_only, 1):
                        sloc.append({
                            "csv": csv_of.get(c, ""),
                            "variavel": c,
                            "fase": col_phase(c),
                            METER_COL: m,
                            "valor_y": round(float(np.nanmedian(vals[rs:re_ + 1])), 6),
                            "x_inicio": int(rs),
                            "x_fim": int(re_),
                            "n_pontos": int(re_ - rs + 1),
                            "t_inicio": pd.Timestamp(tcol[rs]),
                            "t_fim": pd.Timestamp(tcol[re_]),
                            "criterio": "sem_leitura_por_medidor_geral_M1",
                        })
                        mg_blocks += 1
                        mg_pts += int(re_ - rs + 1)
        logger.info("2b) Physical lock M1 (general meter): %d invalid minutes -> "
                    "%d blocks . %d pts on the energy side of the other meters"
                    % (int(mg_invalid.sum()), mg_blocks, mg_pts))
    else:
        if MG_METER is None:
            logger.info("2b) Physical lock by the general meter: disabled (--main-meter-id none).")
        else:
            logger.info(f"2b) Physical lock by the general meter: meter {MG_METER} absent or without samples.")

    sent_loc = pd.DataFrame(sloc, columns=[
        "csv", "variavel", "fase", METER_COL, "valor_y",
        "x_inicio", "x_fim", "n_pontos", "t_inicio", "t_fim", "criterio"
    ])
    sent_loc.to_csv(AUDITS_DIR / "06_sentinel_locations.csv", index=False)
    npts = int(sent_loc["n_pontos"].sum()) if len(sent_loc) else 0
    n_singletons = int((sent_loc["n_pontos"] == 1).sum()) if len(sent_loc) else 0
    logger.info("2) Sentinels -> 06_sentinel_locations.csv (%d blocks . %d pts . %d vars . %d isolated blocks)"
                % (len(sent_loc), npts, sent_loc["variavel"].nunique() if len(sent_loc) else 0, n_singletons))

    # ---------- Sentinel tables (data -> tabdata/) --------------------
    t1 = pd.DataFrame([{"variavel": c, "lado": "energia" if is_energy_side(c) else "tensao",
                        "fase": col_phase(c) or "-", "n_sentinelas": sent_count[c],
                        "pct_sentinelas": round(100 * sent_count[c] / max(total_pts, 1), 3)}
                       for c in elec_cols])
    save_table_data(t1, "06_sentinels_by_variable")
    t2 = t1[t1["variavel"].isin(["i_an", "i_bn", "i_cn"])][
        ["variavel", "fase", "n_sentinelas", "pct_sentinelas"]]
    save_table_data(t2, "06_sentinels_currents")
    logger.info("Tables (data -> tabdata/): 06_sentinels_by_variable . 06_sentinels_currents")

    # ---------- 8. OUTLIERS: robust envelope, away from edges -----------
    # An outlier is a short/anomalous point. Sentinel/gap edges do not enter as outliers.
    OUT_VARS = list(elec_cols)
    K_IQR = 8.0  # local Hampel/MAD threshold; name kept for compatibility
    out_count = {v: 0 for v in OUT_VARS}
    oloc = []

    # Gap mask per meter, to protect edges in the outlier detection.
    gap_masks = {m: np.zeros(len(aligned.loc[meter_idx[m]]), dtype=bool) for m in meters}
    if len(gap_loc):
        for m, grp in gap_loc.groupby(METER_COL):
            if m not in gap_masks:
                continue
            gm = gap_masks[m]
            for _, r in grp.iterrows():
                xi = max(0, int(r["x_inicio"]))
                xf = min(len(gm) - 1, int(r["x_fim"]))
                if xf >= xi:
                    gm[xi:xf + 1] = True
            gap_masks[m] = gm

    for m in meters:
        idx = meter_idx[m]
        t = aligned.loc[idx, TIME_COL].values
        gmask = gap_masks.get(m, np.zeros(len(t), dtype=bool))
        for v in OUT_VARS:
            vals_raw = aligned.loc[idx, v].values.astype(float)
            smask = sentinel_masks.get((m, v), np.zeros(len(vals_raw), dtype=bool))
            omask = detect_outlier_mask(vals_raw, sentinel_mask=smask, gap_mask=gmask, k_iqr=K_IQR)
            n = int(omask.sum())
            if n == 0:
                continue
            out_count[v] += n
            oloc.append(pd.DataFrame({
                "csv": csv_of.get(v),
                "variavel": v,
                "fase": col_phase(v),
                METER_COL: m,
                "x": np.flatnonzero(omask),
                TIME_COL: t[omask],
                "valor": vals_raw[omask],
                "criterio": "iqr_robusto_fora_bordas_sentinela_lacuna"
            }))

    out_loc = pd.concat(oloc, ignore_index=True) if oloc else pd.DataFrame(
        columns=["csv", "variavel", "fase", METER_COL, "x", TIME_COL, "valor", "criterio"])
    out_loc.to_csv(AUDITS_DIR / "06_outlier_locations.csv", index=False)
    logger.info(f"3) Outliers -> 06_outlier_locations.csv ({len(out_loc)} points | robust IQR envelope, k={K_IQR:.1f})")

    # ---------- 9. ONLY NOW: sentinels + outliers BECOME NaN -----------
    # The three independent locations have already been saved:
    #   06_gap_locations.csv, 06_sentinel_locations.csv, 06_outlier_locations.csv.
    # From here on, the filling base is created.
    aligned_sentinel = aligned.copy()
    for (mm, vv), smask in sentinel_masks.items():
        sel = (aligned_sentinel[METER_COL] == mm).values
        col = aligned_sentinel.loc[sel, vv].to_numpy(copy=True)
        col[smask] = np.nan
        aligned_sentinel.loc[sel, vv] = col

    for tipo, vs in type_vars.items():
        aligned_sentinel[[METER_COL, TIME_COL] + vs].to_csv(
            INTERIM_DIR / f"06_{tipo}_sentinelas.csv", index=False)
    logger.info("  sentinels -> 06_*_sentinelas.csv (generated after saving gaps/sentinels/outliers)")

    aligned_clean = aligned.copy()
    for (mm, vv), smask in sentinel_masks.items():
        sel = (aligned_clean[METER_COL] == mm).values
        col = aligned_clean.loc[sel, vv].to_numpy(copy=True)
        col[smask] = np.nan
        aligned_clean.loc[sel, vv] = col

    if len(out_loc):
        for (mm, vv), grp in out_loc.groupby([METER_COL, "variavel"]):
            sel = (aligned_clean[METER_COL] == mm).values
            col = aligned_clean.loc[sel, vv].to_numpy(copy=True)
            col[grp["x"].to_numpy()] = np.nan
            aligned_clean.loc[sel, vv] = col
    logger.info("End: real gaps preserved; sentinels + outliers -> NaN only in the consolidated file.")

    # Clean DATA (real gaps + sentinels + outliers as NaN)
    for tipo, vs in type_vars.items():
        aligned_clean[[METER_COL, TIME_COL] + vs].to_csv(
            INTERIM_DIR / f"06_{tipo}_limpo.csv", index=False)
    logger.info("  clean data -> 06_*_limpo.csv")

    # CONSOLIDATED CSV for the next script: all variables, without sentinels or outliers.
    ordered_cols = [METER_COL, TIME_COL]
    for tipo in ["primarias", "secundarias", "distorcao_meta", "harmonicas"]:
        ordered_cols += type_vars.get(tipo, [])
    fill_df = aligned_clean[ordered_cols]
    fill_df.to_csv(INTERIM_DIR / "06_para_preencher.csv", index=False)
    n_gap = int(fill_df.drop(columns=[METER_COL, TIME_COL]).isna().sum().sum())
    n_cell = fill_df.drop(columns=[METER_COL, TIME_COL]).size
    logger.info(f"CSV for the next script -> 06_para_preencher.csv ({len(ordered_cols)-2} variables . "
                f"{n_gap}/{n_cell} cells to fill = {100*n_gap/n_cell:.1f}%)")

    # Table: outliers per variable (csv + latex)
    total_pts = len(aligned)
    t_out = pd.DataFrame([{"variavel": v, "fase": col_phase(v), "n_outliers": out_count[v],
                           "pct_outliers": round(100 * out_count[v] / max(total_pts, 1), 3)}
                          for v in OUT_VARS])
    save_table_data(t_out, "06_outliers")
    logger.info(f"Table (data -> tabdata/): 06_outliers | total outliers={int(sum(out_count.values()))}")

    pd.DataFrame(inventory_rows).to_csv(AUDITS_DIR / "06_variable_inventory.csv", index=False)
    pd.DataFrame(columns_rows).to_csv(AUDITS_DIR / "06_columns_long.csv", index=False)
    pd.DataFrame(cov_rows).to_csv(AUDITS_DIR / "06_alignment_coverage.csv", index=False)
    logger.info("Saved: 06_variable_inventory.csv · 06_columns_long.csv · 06_alignment_coverage.csv")

    # ---------- figdata: copy of the source files of the FIGURES (rule: figures read only figdata/) ----
    # The generators read the SAME files, but from figdata/ (not from data/interim nor audits/).
    FIGDATA_DIR.mkdir(parents=True, exist_ok=True)
    fig_sources = [INTERIM_DIR / "06_primarias.csv", INTERIM_DIR / "06_secundarias.csv",
                   INTERIM_DIR / "06_para_preencher.csv", AUDITS_DIR / "06_sentinel_locations.csv",
                   AUDITS_DIR / "06_outlier_locations.csv", AUDITS_DIR / "06_gap_locations.csv"]
    for src in fig_sources:
        shutil.copyfile(src, FIGDATA_DIR / src.name)
    logger.info(f"figdata <- copy of {len(fig_sources)} source files of the figures (same data)")

    manifest = {
        "script": "06_validate_and_zero.py", "section": "§03",
        "run_timestamp": datetime.now().isoformat(),
        "inputs": dict(CSV_INPUTS),
        "operations": ["count variables", "listar colunas", "verificar decimal",
                       "NaN->0 (numeric)", "align to the common 1-min grid (configurable/automatic origin)",
                       "sentinela por piso global exato inclusive isolado",
                       "synchronised current sentinel with adaptive tolerance per meter",
                       "sentinela por patamar inferior local com grupo completo",
                       "expansao de 1 ponto nas bordas de blocos confirmados de sentinela",
                       "outlier pontual por Hampel local fora de bordas"],
        "grid": {"freq": GRID_FREQ, "origin_mode": grid_origin_mode, "origin": str(grid_origin), "end": str(grid_end),
                 "n_minutes": len(grid), "aggregation": "mean por minuto"},
        "main_meter_lock": {"enabled": main_meter_id is not None, "main_meter_id": main_meter_id},
        "results": {
            "total_variables_input": int(sum(r["n_variables"] for r in inventory_rows)),
            "n_meters": len(meters),
            "comma_decimal_total": int(sum(r["n_comma_decimal_cols"] for r in inventory_rows)),
            "aligned_outputs": out_files,
            "mean_pct_with_data": round(float(np.mean([r["pct_with_data"] for r in cov_rows])), 2),
        },
        "note": "Aligned outputs: all meters on the SAME grid, with configurable/automatic origin. "
                "Gaps (minutes without reading) remain NaN. Confirmed sentinel blocks also remove 1 finite edge point. Per-reading meta (tag/datetime_read/"
                "Unnamed:0) is discarded in the gridding; identification becomes meter_id+time.",
        "status": "success",
    }
    with open(MANIFESTS_DIR / "06_validate_and_zero_params.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Saved: manifests/06_validate_and_zero_params.json")
    logger.info("=== 06_validate_sentinels_outliers.py completed ===")
    logger.info(f"Summary: {len(meters)} meters aligned | grid {len(grid)} min from {grid_origin} | "
                f"mean coverage {manifest['results']['mean_pct_with_data']:.1f}%")


if __name__ == "__main__":
    main()
