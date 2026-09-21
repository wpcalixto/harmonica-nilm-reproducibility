#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 03_audit_physical_informational.py

File type: Diagnostic script (pipeline stage 03: physical and informational channel audit; read-only)

Purpose:
    Physical and informational audit (read-only).

    DIAGNOSTIC stage between ingestion (02) and scope selection (05). Does NOT correct, does
    NOT remove, does NOT impute: reads the standardised parquet (faithful to the raw data),
    measures and classifies every channel (meter x variable x phase). The per-channel verdict
    governs the downstream policy.

    TWO-LEVEL DESIGN:
      * PHASE LEVEL: `phase_physical_consistency` measures whether the phase CLOSES as an
        electrical system (triangle P^2+Q^2<=S^2, cos in [-1,1], THD>=0, harmonics 0-100 %,
        voltage). It is a MARKER; it is NOT automatically inherited by the channels.
      * CHANNEL LEVEL: `channel_class`; each QUANTITY is evaluated by its own criterion.
        A phase can be NON_PHYSICAL and still have `p_a` as ACTIVE_POWER_VALID.

    ACTIVE POWER has its own rule (not downgraded by Q/S/cos/V/I): P_phi is a physical
    candidate if it is not a dominant sentinel, not stuck, has a temporal dynamics compatible
    with a load and a finite accumulated energy. A triangle violation is attributed to Q/S
    (own check S>=|P|, |Q|<=S), never to P.

    METHODOLOGICAL DECISION: the magnitude test against an external reference (P_lim) is
    WITHDRAWN from the main study: there is no source independent of the CSV, so any limit
    extracted from the data itself would be circular. With external_power_limit_kw null, one
    always gets energy_plausibility=WARN and scale_status=UNKNOWN_NO_REFERENCE, and
    FAIL_EXTERNAL_REF is never applied. The per-meter/phase mechanism remains in the code,
    inert, only activatable if a reliable external reference appears. Admissible conclusion:
    "P_phi was the most reliable quantity after the internal audit", NOT "the absolute scale
    of P_phi is confirmed".

    REDUNDANCY tested against a HIERARCHICAL REFERENCE of AUDITED ACTIVE powers:
      P_ref = { P_phi ACTIVE_POWER_VALID } union { P_tot = sum of valid P_phi }.
      R2_{x|P_ref} = R2( x(t) ~ f(P_ref(t)) ); redundant if >= redund_r2 (strong >= redund_strong).
    Thus a current/harmonic is not declared redundant because of a weak or broken phase power.

    CHANNEL TAXONOMY (real physical data):
      PHYSICAL_VALID        quantity passes its own physical checks (valid measurement)
      ACTIVE_POWER_VALID    active power passes its own audit (physical basis / balance)
      PHYSICAL_ALERT        live quantity with a physical inconsistency (outside the instrument
                            range, cos>1, THD<0, S<|P|): alert, not exclusion
      REDUNDANT_WITH_POWER  explained by P_ref (R2>=0.95)
      DEAD_OR_STUCK         sentinel, constant or without dynamics (exclude)
      EXCLUDE               no physical nor informational value (exclude)

    The voltage band is the INSTRUMENT RANGE (Parametros.jpeg): L-N 1-300 Vrms.

    Method guarantees:
      * SENTINEL-AWARE: physics measured only on real points (NaN/zero/dominant floor excluded).
      * PRE-FILL: informational statistics before the filling; MAD robust to offset.

    Input:
        data/interim/02_raw_standardized.parquet   (produced by 02, without repair)

    Outputs (prefix 03_):
        audits/03_phys_audit.csv                 (meter x phase; phase consistency)
        audits/03_info_audit.csv                 (meter x variable x phase; metrics)
        audits/03_channel_classification.csv     (consolidated verdict per channel)
        audits/03_channel_eligibility.csv        (aggregated eligibility; source of table_02)
        audits/03_energy_coherence_alert.csv
        audits/03_audit_summary.md
        logs/03_audit_physical_informational.log
        manifests/03_audit_physical_informational_params.json

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
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT  = SCRIPT_DIR.parent  # repository root (parent of scripts/)

CONFIG_DIR    = PROJECT_ROOT / "config"
INTERIM_DIR   = PROJECT_ROOT / "data" / "interim"
AUDITS_DIR    = PROJECT_ROOT / "audits"
LOGS_DIR      = PROJECT_ROOT / "logs"
MANIFESTS_DIR = PROJECT_ROOT / "manifests"

PARQUET_IN    = INTERIM_DIR / "02_raw_standardized.parquet"

TIMESTAMP_COL = "time"
METER_COL     = "meter_id"
PHASES        = ["a", "b", "c"]

DT_HOURS        = 1.0 / 60.0           # 1-min step -> energy in kWh (P in kW)
MIN_REAL_POINTS = 10                   # minimum number of real points for regressions (fixed)
DYN_MIN_UNIQUE  = 10                   # minimum number of distinct values = has dynamics

# ---------------------------------------------------------------------------
# Thresholds: read from the audit_physical block of preprocessing_config.yaml (canonical
# source, validated by 01_config). The literals are the self-contained FALLBACK.
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "V_NOMINAL_PN": 380.0 / np.sqrt(3.0),  # nominal phase-neutral voltage (~219.4 V); operational marker
    "V_TOL": 0.20,                          # +/-20 %: nominal deviation (secondary marker)
    "V_LN_MIN": 1.0,                        # instrument range L-N (Parametros.jpeg): 1-300 Vrms
    "V_LN_MAX": 300.0,                      # physical GATE of the voltage (replaces the +/-20 % band)
    "H_MAX_ORDER": 51.0,                    # maximum harmonic order supported by the instrument
    "TRI_TOL": 0.10,                        # |P^2+Q^2-S^2|/S^2 above this = triangle violation
    "PHYS_FRAC": 0.02,                      # fraction of violating points -> non-physical quantity
    "REVERSE_FRAC": 0.50,                   # >50 % of P<0 -> predominantly reverse channel
    "STUCK_FRAC": 0.50,                     # longest constant run / n_real above this = stuck
    "LIVE_MIN": 0.05,                       # below this many live real points = dead
    "REDUND_R2": 0.95,                      # R2 vs P_ref >= this -> redundant with power
    "REDUND_STRONG": 0.99,                  # strong redundancy threshold
    "CLOCK_R2": 0.50,                       # R2 vs time index >= this and non-redundant -> clock
    "SENT_FLOOR_FRAC": 0.02,                # most frequent finite value with freq >= this = floor
    "REV_NORMAL": 0.05,                     # reverse_frac < this -> NORMAL convention
    "REV_REVERSED": 0.80,                   # reverse_frac > this -> REVERSED_OR_BIDIRECTIONAL
    "ENERGY_ALPHA": 1.5,                    # tolerance alpha of the magnitude test with external reference
    "COHERENCE_WARN_RATIO": 5.0,            # |E_MG|/|E_L| outside [1/r, r] -> coherence alert
}
_YAML_KEYS = {
    "v_nominal_pn": "V_NOMINAL_PN", "v_tol": "V_TOL",
    "v_ln_min": "V_LN_MIN", "v_ln_max": "V_LN_MAX", "h_max_order": "H_MAX_ORDER",
    "tri_tol": "TRI_TOL",
    "phys_frac": "PHYS_FRAC", "reverse_frac": "REVERSE_FRAC", "stuck_frac": "STUCK_FRAC",
    "live_min": "LIVE_MIN", "redund_r2": "REDUND_R2", "redund_strong": "REDUND_STRONG",
    "clock_r2": "CLOCK_R2", "sent_floor_frac": "SENT_FLOOR_FRAC",
    "rev_normal": "REV_NORMAL", "rev_reversed": "REV_REVERSED",
    "energy_alpha": "ENERGY_ALPHA", "coherence_warn_ratio": "COHERENCE_WARN_RATIO",
}


def build_channel_eligibility(chan_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates the per-channel verdict into a tidy eligibility table:
    counts per PRIMARY CLASS (channel_class) + secondary FLAGS (count of True).
    is_proxy is recorded EXPLICITLY as 0 (v2 has no proxy channels)."""
    total = int(len(chan_df))

    def _count_true(col: str) -> int:
        if col not in chan_df.columns:
            return 0
        s = chan_df[col]
        if s.dtype == bool:
            return int(s.sum())
        return int(s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"]).sum())

    rows = []
    for cls, nch in chan_df["channel_class"].value_counts().items():
        rows.append({"section": "primary_class", "key": str(cls), "n_channels": int(nch),
                     "pct": round(100.0 * int(nch) / total, 2) if total else 0.0})
    # secondary flags (display name -> actual column of the audit; is_proxy is explicitly 0)
    flag_map = [("is_proxy", None), ("reverse_flow", "reverse_flow_flag"),
                ("redundant", "redund_strong"), ("scale_suspect", "scale_suspect_flag"),
                ("requires_review", "requires_review"),
                ("requires_sensitivity", "requires_sensitivity")]
    for name, col in flag_map:
        nch = 0 if col is None else _count_true(col)
        rows.append({"section": "flag", "key": name, "n_channels": int(nch),
                     "pct": round(100.0 * nch / total, 2) if total else 0.0})
    out = pd.DataFrame(rows)
    out.attrs["total_channels"] = total
    return out


def _load_thresholds() -> tuple:
    """Reads audit_physical from preprocessing_config.yaml; falls back if absent.
    Returns (dict of numeric thresholds, config of external_power_limit_kw)."""
    vals = dict(_DEFAULTS)
    ext = {}
    try:
        import yaml
        cfg = yaml.safe_load((CONFIG_DIR / "preprocessing_config.yaml").read_text(encoding="utf-8"))
        blk = (cfg or {}).get("audit_physical", {}) or {}
        for yk, ck in _YAML_KEYS.items():
            if blk.get(yk) is not None:
                vals[ck] = float(blk[yk])
        ext = blk.get("external_power_limit_kw", {}) or {}
        if not isinstance(ext, dict):           # backward compat.: old global scalar
            ext = {"default": float(ext)}
        src = blk.get("external_power_limit_source", {}) or {}
    except Exception:
        src = {}
    return vals, ext, src


def _load_meter_map() -> tuple:
    """(roles, labels) from meter_map.yaml. labels: {meter_id:int -> 'M_G'/'M5'/...}."""
    try:
        import yaml
        cfg = yaml.safe_load((CONFIG_DIR / "meter_map.yaml").read_text(encoding="utf-8"))
        roles = (cfg or {}).get("roles", {}) or {}
        labels = {int(k): str(v) for k, v in ((cfg or {}).get("meters", {}) or {})
                  .get("labels", {}).items()}
        return roles, labels
    except Exception:
        return {}, {}


_T, EXTERNAL_P_LIM_CFG, EXTERNAL_P_LIM_SOURCE = _load_thresholds()
V_NOMINAL_PN = _T["V_NOMINAL_PN"]; V_TOL = _T["V_TOL"]; TRI_TOL = _T["TRI_TOL"]
V_LN_MIN = _T["V_LN_MIN"]; V_LN_MAX = _T["V_LN_MAX"]; H_MAX_ORDER = _T["H_MAX_ORDER"]
PHYS_FRAC = _T["PHYS_FRAC"]; REVERSE_FRAC = _T["REVERSE_FRAC"]; STUCK_FRAC = _T["STUCK_FRAC"]
LIVE_MIN = _T["LIVE_MIN"]; REDUND_R2 = _T["REDUND_R2"]; REDUND_STRONG = _T["REDUND_STRONG"]
CLOCK_R2 = _T["CLOCK_R2"]; SENT_FLOOR_FRAC = _T["SENT_FLOOR_FRAC"]
REV_NORMAL = _T["REV_NORMAL"]; REV_REVERSED = _T["REV_REVERSED"]
ENERGY_ALPHA = _T["ENERGY_ALPHA"]; COHERENCE_WARN_RATIO = _T["COHERENCE_WARN_RATIO"]
ROLES, METER_LABELS = _load_meter_map()


def get_power_limit(meter_label: str, phase: str):
    """External limit P_lim per meter/phase with hierarchical fallback:
       1) by_meter[label][phase] -> 2) by_meter[label] (scalar) -> 3) default -> 4) None."""
    limits = EXTERNAL_P_LIM_CFG or {}
    by_meter = limits.get("by_meter", {}) or {}
    default = limits.get("default", None)
    ml = by_meter.get(meter_label, None)
    if isinstance(ml, dict):
        v = ml.get(phase, None)
        return v if v is not None else default
    if ml is not None:
        return ml
    return default

# ---------------------------------------------------------------------------
# Variable types (same patterns as script 02)
# ---------------------------------------------------------------------------

RE_PRIMARIA   = re.compile(r"^[iv]_[abc]n$")
RE_SECUNDARIA = re.compile(r"^(p|q|s|cos)_[abc]$")
RE_DISTORCAO  = re.compile(r"^thd[iv]_[abc]$")
RE_HARMONICA  = re.compile(r"^hrm_[iv]_")
RE_PHASE      = re.compile(r"_([abc])n?(?:_|$)")


def phase_of(col: str):
    m = RE_PHASE.search(col)
    return m.group(1) if m else None


def var_family(col: str) -> str:
    if RE_PRIMARIA.match(col):
        return "current" if col.startswith("i_") else "voltage"
    if RE_SECUNDARIA.match(col):
        return col.split("_")[0]            # p / q / s / cos
    if RE_DISTORCAO.match(col):
        return col[:4]                      # thdi / thdv
    if RE_HARMONICA.match(col):
        return "hrm_i" if col.startswith("hrm_i") else "hrm_v"
    return "other"


# ---------------------------------------------------------------------------
# Conservative mask of REAL points (sentinel-aware, pre-fill)
# ---------------------------------------------------------------------------

def real_mask(v: np.ndarray):
    finite = np.isfinite(v)
    mask = finite & (v != 0.0)
    floor_value = np.nan
    if finite.sum() >= MIN_REAL_POINTS:
        vals, counts = np.unique(v[finite], return_counts=True)
        i = int(np.argmax(counts))
        if counts[i] >= SENT_FLOOR_FRAC * finite.sum() and counts[i] >= 20:
            floor_value = float(vals[i])
            mask &= (v != floor_value)
    return mask, floor_value


def longest_const_run(v: np.ndarray) -> int:
    if v.size == 0:
        return 0
    changes = np.concatenate(([True], v[1:] != v[:-1]))
    idx = np.flatnonzero(changes)
    run_lengths = np.diff(np.concatenate((idx, [v.size])))
    return int(run_lengths.max())


def r2_multi(y: np.ndarray, X: np.ndarray):
    """R2 of y ~ X (with intercept), OLS via lstsq. NaN if the sample is insufficient."""
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if ok.sum() < MIN_REAL_POINTS:
        return np.nan
    yy, XX = y[ok], X[ok]
    if np.var(yy) == 0:
        return np.nan
    A = np.column_stack([XX, np.ones(XX.shape[0])])
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
        if not np.all(np.isfinite(coef)):
            return np.nan
        resid = yy - A @ coef
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    if not (np.isfinite(ss_res) and np.isfinite(ss_tot)) or ss_tot <= 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def signed_corr(y: np.ndarray, x: np.ndarray):
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < MIN_REAL_POINTS or np.var(y[ok]) == 0 or np.var(x[ok]) == 0:
        return 0
    r = float(np.corrcoef(y[ok], x[ok])[0, 1])
    return int(np.sign(r))


def frac_viol(cond: np.ndarray, valid: np.ndarray) -> float:
    n = int(valid.sum())
    return float((cond & valid).sum() / n) if n else np.nan


def is_dead_or_stuck(xr: np.ndarray, live_frac: float) -> bool:
    if xr.size < MIN_REAL_POINTS or live_frac < LIVE_MIN:
        return True
    if np.unique(xr).size <= 2:
        return True
    if longest_const_run(xr) / xr.size > STUCK_FRAC:
        return True
    return False


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("03_audit")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8"); fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# PHASE LEVEL: physical consistency of the phase (marker, not inherited)
# ---------------------------------------------------------------------------

def audit_phase_consistency(df_m: pd.DataFrame, meter, cols: set):
    rows = []
    for ph in PHASES:
        p = df_m[f"p_{ph}"].to_numpy(float) if f"p_{ph}" in cols else None
        q = df_m[f"q_{ph}"].to_numpy(float) if f"q_{ph}" in cols else None
        s = df_m[f"s_{ph}"].to_numpy(float) if f"s_{ph}" in cols else None
        cph = df_m[f"cos_{ph}"].to_numpy(float) if f"cos_{ph}" in cols else None
        vph = df_m[f"v_{ph}n"].to_numpy(float) if f"v_{ph}n" in cols else None
        row = {"meter_id": meter, "phase": ph.upper()}

        if p is not None and q is not None and s is not None:
            valid = np.isfinite(p) & np.isfinite(q) & np.isfinite(s) & (s != 0)
            r_tri = np.where(valid, (p ** 2 + q ** 2 - s ** 2) / np.where(s == 0, np.nan, s ** 2), np.nan)
            row["tri_resid_median"] = float(np.nanmedian(r_tri)) if valid.any() else np.nan
            row["tri_viol_frac"] = frac_viol(np.abs(r_tri) > TRI_TOL, valid)
            row["reverse_frac"] = frac_viol(p < 0, np.isfinite(p) & (p != 0))
        else:
            row["tri_resid_median"] = row["tri_viol_frac"] = row["reverse_frac"] = np.nan

        cos = cph if cph is not None else (p / np.where(s == 0, np.nan, s)
                                           if (p is not None and s is not None) else None)
        row["cos_oob_frac"] = frac_viol(np.abs(cos) > 1.0, np.isfinite(cos)) if cos is not None else np.nan

        thd_cols = [c for c in (f"thdi_{ph}", f"thdv_{ph}") if c in cols]
        if thd_cols:
            neg = np.zeros(len(df_m), bool); valid = np.zeros(len(df_m), bool)
            for c in thd_cols:
                x = df_m[c].to_numpy(float); vld = np.isfinite(x)
                neg |= (x < 0) & vld; valid |= vld
            row["thd_neg_frac"] = frac_viol(neg, valid)
        else:
            row["thd_neg_frac"] = np.nan

        hcols = [c for c in cols if RE_HARMONICA.match(c) and phase_of(c) == ph]
        if hcols:
            oob = np.zeros(len(df_m), bool); valid = np.zeros(len(df_m), bool)
            for c in hcols:
                x = df_m[c].to_numpy(float); vld = np.isfinite(x)
                oob |= ((x < 0) | (x > 100)) & vld; valid |= vld
            row["harm_oob_frac"] = frac_viol(oob, valid)
        else:
            row["harm_oob_frac"] = np.nan

        if vph is not None:
            m, _ = real_mask(vph)
            row["v_median"] = float(np.median(vph[m])) if m.any() else np.nan
            # instrument range L-N (1-300 Vrms) as physical gate
            row["v_implausible_frac"] = frac_viol((vph < V_LN_MIN) | (vph > V_LN_MAX), m)
        else:
            row["v_median"] = row["v_implausible_frac"] = np.nan

        viol = [row.get(k) for k in ("tri_viol_frac", "cos_oob_frac", "thd_neg_frac",
                                     "harm_oob_frac", "v_implausible_frac")]
        nonphys = any((f is not None and np.isfinite(f) and f > PHYS_FRAC) for f in viol)
        row["phase_physical_consistency"] = "NON_PHYSICAL" if nonphys else "PHYSICAL"
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Own rule of the ACTIVE POWER
# ---------------------------------------------------------------------------

def audit_active_power(x: np.ndarray, meter_label: str = "", phase_up: str = ""):
    """Classifies P_phi by its OWN criterion (independent of Q/S/cos/V/I) and attaches
    energy-plausibility MARKERS (they never exclude P). Returns (status, metrics).

    Tests: (1) reverse flow by count and by energy; (2) magnitude with external
    reference when available (otherwise scale UNKNOWN); (3) relative coherence
    E_MG vs E_L is systemic -> computed in main (alert). ACTIVE_POWER_VALID means
    "best physical quantity available after its own audit", not "problem-free"."""
    m, floor = real_mask(x)
    xr = x[m]
    n_real = int(m.sum())
    live_frac = float(n_real / len(x)) if len(x) else 0.0
    absx = np.abs(xr)
    sum_abs = float(absx.sum())
    signed_e = float(np.nansum(xr) * DT_HOURS) if n_real else np.nan
    abs_e = float(sum_abs * DT_HOURS) if n_real else np.nan
    rev_frac = float((xr < 0).mean()) if n_real else np.nan
    rev_e_frac = float(absx[xr < 0].sum() / sum_abs) if sum_abs > 0 else np.nan
    p95 = float(np.percentile(absx, 95)) if n_real else np.nan

    # Test 1: sign convention / reverse flow
    if not np.isfinite(rev_frac):
        sign_conv, rev_flag = "UNKNOWN", "NONE"
    elif rev_frac < REV_NORMAL:
        sign_conv, rev_flag = "NORMAL", "LOW"
    elif rev_frac <= REV_REVERSED:
        sign_conv, rev_flag = "MIXED", "MEDIUM"
    else:
        sign_conv, rev_flag = "REVERSED_OR_BIDIRECTIONAL", "HIGH"

    # Test 2: magnitude with external reference PER METER/PHASE.
    # METHODOLOGICAL DECISION: WITHDRAWN from the main study for lack of a source
    # independent of the CSV (a P_lim extracted from the data itself would be circular). With
    # external_power_limit_kw null, get_power_limit() returns None -> WARN/UNKNOWN and
    # FAIL_EXTERNAL_REF is NEVER applied. The mechanism stays inert, only activatable if
    # a reliable external reference appears in the config.
    p_lim = get_power_limit(meter_label, phase_up)
    requires_review = False
    if p_lim is not None and n_real:
        t_hours = n_real * DT_HOURS
        ok = (p95 <= ENERGY_ALPHA * p_lim) and (abs_e <= ENERGY_ALPHA * p_lim * t_hours)
        energy_plaus = "PASS" if ok else "FAIL_EXTERNAL_REF"
        scale_status = "OK" if ok else "SCALE_INCONSISTENT"
        requires_review = not ok
    else:
        energy_plaus = "WARN"                 # not verifiable without an external reference
        scale_status = "UNKNOWN_NO_REFERENCE"

    metrics = {"n_real": n_real, "live_frac": live_frac, "floor_value": floor,
               "e_kwh": signed_e, "signed_energy_kwh": signed_e, "absolute_energy_kwh": abs_e,
               "p95_abs_kw": p95, "reverse_frac": rev_frac, "reverse_energy_frac": rev_e_frac,
               "sign_convention": sign_conv, "reverse_flow_flag": rev_flag,
               "p_lim_kw": p_lim, "energy_plausibility": energy_plaus, "scale_status": scale_status,
               "scale_suspect_flag": "NONE",  # may be raised by the coherence test (main)
               "requires_review": requires_review,
               "requires_sensitivity": (rev_flag == "HIGH")}

    if is_dead_or_stuck(xr, live_frac):
        return "DEAD_OR_STUCK", metrics
    has_dynamics = np.unique(xr).size > DYN_MIN_UNIQUE
    if has_dynamics and np.isfinite(signed_e):
        return "ACTIVE_POWER_VALID", metrics
    return "PHYSICAL_ALERT", metrics


def build_p_ref(df_m: pd.DataFrame, cols: set, p_status: dict):
    """P_ref = AUDITED active powers (ACTIVE_POWER_VALID) + P_tot (their sum)."""
    valid = [f"p_{ph}" for ph in PHASES
             if f"p_{ph}" in cols and p_status.get(ph) == "ACTIVE_POWER_VALID"]
    if not valid:
        return None, valid
    Pmat = df_m[valid].to_numpy(float)
    Ptot = np.nansum(Pmat, axis=1)
    return np.column_stack([Pmat, Ptot]), valid


# ---------------------------------------------------------------------------
# CHANNEL LEVEL: classification per quantity
# ---------------------------------------------------------------------------

def own_phys_ok(fam: str, x: np.ndarray, m: np.ndarray,
                p_ph: np.ndarray, s_ph: np.ndarray) -> bool:
    """OWN physical check of the quantity. True = physically valid quantity."""
    if fam == "voltage":                        # INSTRUMENT RANGE L-N (1-300 Vrms)
        return frac_viol((x < V_LN_MIN) | (x > V_LN_MAX), m) <= PHYS_FRAC
    if fam == "current":                        # RMS: I >= 0
        return frac_viol(x < 0, m) <= PHYS_FRAC
    if fam == "cos":
        return frac_viol(np.abs(x) > 1.0, m) <= PHYS_FRAC
    if fam in ("thdi", "thdv"):
        return frac_viol(x < 0, m) <= PHYS_FRAC
    if fam == "s":                              # apparent: S>=0 and S>=|P|
        if p_ph is not None:
            bad = (x < 0) | (x < np.abs(p_ph) * (1 - PHYS_FRAC))
        else:
            bad = x < 0
        return frac_viol(bad, m & np.isfinite(x)) <= PHYS_FRAC
    if fam == "q":                              # reactive: |Q|<=S
        if s_ph is not None:
            return frac_viol(np.abs(x) > s_ph * (1 + PHYS_FRAC), m & np.isfinite(s_ph)) <= PHYS_FRAC
        return True
    if fam in ("hrm_i", "hrm_v"):               # measured harmonic: 0 <= h <= 100 %
        return frac_viol((x < 0) | (x > 100), m) <= PHYS_FRAC
    return True


def classify_channel(fam, dead, own_ok, r2_ref, n_unique) -> str:
    """Taxonomy (real data): live + non-redundant -> PHYSICAL_VALID (physics OK) or
    PHYSICAL_ALERT (physics fails). No proxy classes."""
    if dead:
        return "DEAD_OR_STUCK"
    if n_unique <= DYN_MIN_UNIQUE:
        return "EXCLUDE"                        # live but without useful variation
    if np.isfinite(r2_ref) and r2_ref >= REDUND_R2:
        return "REDUNDANT_WITH_POWER"
    return "PHYSICAL_VALID" if own_ok else "PHYSICAL_ALERT"


def audit_channels(df_m: pd.DataFrame, meter, cols: set, tnorm: np.ndarray,
                   p_status: dict, p_metrics: dict, P_ref):
    rows = []
    elec = [c for c in df_m.columns
            if RE_PRIMARIA.match(c) or RE_SECUNDARIA.match(c)
            or RE_DISTORCAO.match(c) or RE_HARMONICA.match(c)]
    for c in elec:
        ph = phase_of(c)
        fam = var_family(c)
        x = df_m[c].to_numpy(float)
        row = {"meter_id": meter, "variable": c, "family": fam,
               "phase": ph.upper() if ph else ""}

        # Active power: uses the own verdict already computed
        if fam == "p":
            met = p_metrics[ph]
            row.update(met)
            xr = x[real_mask(x)[0]]
            med = float(np.median(xr)) if xr.size else np.nan
            mad = float(1.4826 * np.median(np.abs(xr - med))) if xr.size else np.nan
            row.update({"median": med, "mad": mad,
                        "robust_cv": (mad / abs(med)) if med not in (0, np.nan) and np.isfinite(med) and med != 0 else np.nan,
                        "n_unique": int(np.unique(xr).size) if xr.size else 0,
                        "R2_vs_Pref": np.nan, "redund_strong": False,
                        "rho_sign_vs_S": 0, "time_R2_clock": np.nan,
                        "own_phys_ok": np.nan, "channel_class": p_status[ph]})
            rows.append(row); continue

        m, floor = real_mask(x)
        xr = x[m]; n_real = int(m.sum())
        live_frac = float(n_real / len(x)) if len(x) else 0.0
        row.update({"n_real": n_real, "live_frac": live_frac, "floor_value": floor})
        dead = is_dead_or_stuck(xr, live_frac)
        if n_real < MIN_REAL_POINTS:
            row.update(dict(median=np.nan, mad=np.nan, robust_cv=np.nan, n_unique=0,
                            R2_vs_Pref=np.nan, redund_strong=False, rho_sign_vs_S=0,
                            time_R2_clock=np.nan, own_phys_ok=np.nan,
                            channel_class="DEAD_OR_STUCK"))
            rows.append(row); continue

        med = float(np.median(xr))
        mad = float(1.4826 * np.median(np.abs(xr - med)))
        nuni = int(np.unique(xr).size)
        row.update(dict(median=med, mad=mad,
                        robust_cv=(mad / abs(med)) if med != 0 else np.nan, n_unique=nuni))

        # Redundancy against P_ref (audited active powers + P_tot)
        yy = np.where(m, x, np.nan)
        r2ref = r2_multi(yy, P_ref) if P_ref is not None else np.nan
        row["R2_vs_Pref"] = r2ref
        row["redund_strong"] = bool(np.isfinite(r2ref) and r2ref >= REDUND_STRONG)
        s_ph_arr = df_m[f"s_{ph}"].to_numpy(float) if (ph and f"s_{ph}" in cols) else None
        row["rho_sign_vs_S"] = signed_corr(yy, s_ph_arr) if s_ph_arr is not None else 0
        row["time_R2_clock"] = r2_multi(yy, tnorm.reshape(-1, 1))

        p_ph_arr = df_m[f"p_{ph}"].to_numpy(float) if (ph and f"p_{ph}" in cols) else None
        ok = own_phys_ok(fam, x, m, p_ph_arr, s_ph_arr)
        row["own_phys_ok"] = bool(ok)
        row["channel_class"] = classify_channel(fam, dead, ok, r2ref, nuni)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Test 3: relative energy coherence E_MG vs E_L (SYSTEMIC; alert, not exclusion)
# ---------------------------------------------------------------------------

def energy_coherence_alert(chan_df: pd.DataFrame):
    """Compares the energy of M_G against the sum of the audited loads per phase.
    Returns (alert_rows, phase_status). Produces an ALERT, never an exclusion (D_G not measured)."""
    mg_ids = ROLES.get("m_g_input", [1])
    load_ids = list(ROLES.get("modeled_loads", [])) + list(ROLES.get("p_other_components", []))
    eps = 1e-9
    p = chan_df[chan_df["family"] == "p"]
    rows, status = [], {}
    for ph in [q.upper() for q in PHASES]:
        e_mg = float(np.nansum(p[(p.meter_id.isin(mg_ids)) & (p.phase == ph)]["signed_energy_kwh"]))
        e_l = float(np.nansum(p[(p.meter_id.isin(load_ids)) & (p.phase == ph)]["signed_energy_kwh"]))
        r_e = abs(e_mg) / (abs(e_l) + eps)
        coherent = (1.0 / COHERENCE_WARN_RATIO) <= r_e <= COHERENCE_WARN_RATIO
        st = "OK" if coherent else "WARN"
        d_e = e_l - e_mg                         # E_loss=0 (not measured) -> closure residual
        rows.append({"phase": ph, "E_MG_kwh": e_mg, "E_L_kwh": e_l,
                     "R_E": r_e, "delta_E_kwh": d_e, "coherence_status": st,
                     "note": "D_G not measured: alert, not exclusion"})
        status[ph] = {"R_E": r_e, "status": st}
    return rows, status


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log_path = LOGS_DIR / "03_audit_physical_informational.log"
    logger = setup_logger(log_path)
    logger.info("=== 03_audit_physical_informational.py: physical + informational audit ===")
    logger.info(f"Run timestamp: {datetime.now().isoformat()}")

    if not PARQUET_IN.exists():
        logger.error(f"Parquet not found (run script 02 first): {PARQUET_IN}")
        sys.exit(1)

    df = pd.read_parquet(PARQUET_IN)
    logger.info(f"Read: {PARQUET_IN.name}: {df.shape[0]} rows x {df.shape[1]} columns")
    if METER_COL not in df.columns:
        logger.error(f"Meter column '{METER_COL}' missing."); sys.exit(1)

    AUDITS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    meters = sorted(df[METER_COL].dropna().unique().tolist())
    cols = set(df.columns)
    logger.info(f"Meters: {meters}")

    phase_rows, chan_rows = [], []
    for meter in meters:
        df_m = df[df[METER_COL] == meter].reset_index(drop=True)
        n = len(df_m)
        tnorm = (np.arange(n, dtype=float) / (n - 1)) if n > 1 else np.zeros(n)

        # 1) active power per phase (own rule) -> P_ref
        mlabel = METER_LABELS.get(int(meter), f"M{meter}")
        p_status, p_metrics = {}, {}
        for ph in PHASES:
            if f"p_{ph}" in cols:
                st, met = audit_active_power(df_m[f"p_{ph}"].to_numpy(float), mlabel, ph.upper())
            else:
                st, met = "DEAD_OR_STUCK", {"n_real": 0, "live_frac": 0.0,
                                           "floor_value": np.nan, "e_kwh": np.nan, "reverse_frac": np.nan}
            p_status[ph] = st; p_metrics[ph] = met
        P_ref, p_valid = build_p_ref(df_m, cols, p_status)

        # 2) phase consistency (marker)
        phase_rows.extend(audit_phase_consistency(df_m, meter, cols))
        # 3) classification per channel
        chan_rows.extend(audit_channels(df_m, meter, cols, tnorm, p_status, p_metrics, P_ref))
        logger.info(f"  meter {meter}: n={n} | valid P={p_valid or '-'} | "
                    f"P_status={ {k: v for k, v in p_status.items()} }")

    phase_df = pd.DataFrame(phase_rows)
    chan_df = pd.DataFrame(chan_rows)

    # consolidation (inherits the phase MARKER, without overriding the channel class)
    pv = phase_df[["meter_id", "phase", "phase_physical_consistency"]]
    chan_df = chan_df.merge(pv, on=["meter_id", "phase"], how="left")

    # Test 3: systemic energy coherence (alert) + raising of markers on M_G
    for col, default in [("coherence_R_E", np.nan), ("coherence_status", "")]:
        if col not in chan_df.columns:
            chan_df[col] = default
    alert_rows, phase_status = energy_coherence_alert(chan_df)
    mg_ids = ROLES.get("m_g_input", [1])
    for ph, st in phase_status.items():
        mg_p = (chan_df["family"] == "p") & (chan_df["meter_id"].isin(mg_ids)) & (chan_df["phase"] == ph)
        chan_df.loc[mg_p, "coherence_R_E"] = st["R_E"]
        chan_df.loc[mg_p, "coherence_status"] = st["status"]
        if st["status"] == "WARN":
            chan_df.loc[mg_p, "scale_suspect_flag"] = "HIGH"
            chan_df.loc[mg_p, "requires_sensitivity"] = True
    alert_df = pd.DataFrame(alert_rows)

    marker_cols = ["signed_energy_kwh", "absolute_energy_kwh", "p95_abs_kw",
                   "reverse_frac", "reverse_energy_frac", "sign_convention",
                   "reverse_flow_flag", "p_lim_kw", "energy_plausibility", "scale_status",
                   "scale_suspect_flag", "requires_review", "requires_sensitivity",
                   "coherence_R_E", "coherence_status"]
    cls_cols = (["meter_id", "variable", "family", "phase", "n_real", "e_kwh",
                 "R2_vs_Pref", "redund_strong", "time_R2_clock", "own_phys_ok",
                 "phase_physical_consistency", "channel_class"] + marker_cols)
    cls_df = chan_df[[c for c in cls_cols if c in chan_df.columns]].copy()

    p_phys = AUDITS_DIR / "03_phys_audit.csv"
    p_info = AUDITS_DIR / "03_info_audit.csv"
    p_cls  = AUDITS_DIR / "03_channel_classification.csv"
    p_alert = AUDITS_DIR / "03_energy_coherence_alert.csv"
    phase_df.to_csv(p_phys, index=False)
    chan_df.to_csv(p_info, index=False)
    cls_df.to_csv(p_cls, index=False)
    alert_df.to_csv(p_alert, index=False)
    logger.info(f"Saved: {p_phys.relative_to(PROJECT_ROOT)} ({len(phase_df)} rows)")
    logger.info(f"Saved: {p_info.relative_to(PROJECT_ROOT)} ({len(chan_df)} rows)")
    logger.info(f"Saved: {p_cls.relative_to(PROJECT_ROOT)} ({len(cls_df)} channels)")
    logger.info(f"Saved: {p_alert.relative_to(PROJECT_ROOT)} (energy coherence, {len(alert_df)} phases)")

    # -- dedicated producer of the channel eligibility (source of table_02) ------------------
    # Aggregates the per-channel verdict into counts per primary class + secondary
    # flags. is_proxy is recorded EXPLICITLY (0 in v2), not hidden.
    p_elig = AUDITS_DIR / "03_channel_eligibility.csv"
    build_channel_eligibility(chan_df).to_csv(p_elig, index=False)
    logger.info(f"Saved: {p_elig.relative_to(PROJECT_ROOT)} (aggregated eligibility -> table_02)")

    cls_counts = cls_df["channel_class"].value_counts().to_dict()
    phase_counts = phase_df["phase_physical_consistency"].value_counts().to_dict()
    n_apv = int((cls_df["channel_class"] == "ACTIVE_POWER_VALID").sum())
    n_plim = int(chan_df.loc[chan_df["family"] == "p", "p_lim_kw"].notna().sum()) \
        if "p_lim_kw" in chan_df.columns else 0
    n_review = int((chan_df.get("requires_review") == True).sum())
    req_sens = chan_df[(chan_df["family"] == "p") & (chan_df.get("requires_sensitivity") == True)]
    for _, r in req_sens.iterrows():
        logger.warning(f"P requires sensitivity: M{r['meter_id']} {r['variable']} "
                       f"(reverse={r.get('reverse_flow_flag')}, scale={r.get('scale_suspect_flag')}, "
                       f"E={r.get('signed_energy_kwh'):.3g} kWh)")

    alert_md = [f"- fase {a['phase']}: E_MG={a['E_MG_kwh']:.4g} · E_L={a['E_L_kwh']:.4g} · "
                f"R_E={a['R_E']:.3g} · {a['coherence_status']}" for a in alert_rows]
    lines = [
        "# Physical and informational audit (03): two levels + energy markers", "",
        f"- Parquet: `{PARQUET_IN.name}`: {df.shape[0]} rows x {df.shape[1]} columns",
        f"- Medidores: {meters}", "",
        "## Phase consistency (marker, not inherited)", "",
        *[f"- **{k}**: {v}" for k, v in phase_counts.items()], "",
        "## Classification per quantity (channel)", "",
        *[f"- **{k}**: {v}" for k, v in cls_counts.items()], "",
        f"**Valid active power (ACTIVE_POWER_VALID): {n_apv} channels**: physical axis "
        "of the modelling and of the balance, preserved even when the phase is flagged NON_PHYSICAL.", "",
        "## Energy-plausibility markers of the active power (they do not exclude P)", "",
        f"- Canais de P que exigem sensibilidade (`requires_sensitivity`): {len(req_sens)}", "",
        "### Energy coherence E_MG vs E_L (test 3; alert, not exclusion)", "",
        *alert_md, "",
        "## Limiares", "",
        f"- Faixa instrumento L-N=[{V_LN_MIN:.0f},{V_LN_MAX:.0f}] V; TRI_TOL={TRI_TOL}; PHYS_FRAC={PHYS_FRAC}; "
        f"REDUND_R2={REDUND_R2} (forte {REDUND_STRONG}); STUCK_FRAC={STUCK_FRAC}; LIVE_MIN={LIVE_MIN}; "
        f"rev∈[{REV_NORMAL},{REV_REVERSED}]; α={ENERGY_ALPHA}; "
        f"external reference P_lim per meter/phase ({n_plim} configured; the others UNKNOWN).", "",
        "Notes: physical verdict PER QUANTITY + phase marker; active P by its own rule "
        "(triangle attributed to Q/S); redundancy against P_ref (audited active powers + P_tot). "
        "Energy plausibility only produces a marker/sensitivity flag; it would block only with an external reference.",
    ]
    p_md = AUDITS_DIR / "03_audit_summary.md"
    p_md.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved: {p_md.relative_to(PROJECT_ROOT)}")

    manifest = {
        "script": "03_audit_physical_informational.py", "section": "§3 / §10 (v8)",
        "run_timestamp": datetime.now().isoformat(),
        "input": str(PARQUET_IN.relative_to(PROJECT_ROOT)),
        "n_rows": int(df.shape[0]), "n_cols": int(df.shape[1]),
        "meters": meters, "n_channels": int(len(cls_df)),
        "design": "two_level: phase_physical_consistency (marker) + channel_class (by grandeza)",
        "active_power_rule": "own criteria; triangle attributed to Q/S; P never demoted by Q/S/cos/V/I",
        "active_power_energy_markers": "reverse/sign_convention + external-ref magnitude (optional) + "
                                       "systemic E_MG vs E_L coherence; markers/sensitivity only, no exclusion",
        "redundancy_reference": "P_ref = audited ACTIVE_POWER_VALID phases + P_tot",
        "external_power_limit_kw": EXTERNAL_P_LIM_CFG,
        "external_power_limit_source": EXTERNAL_P_LIM_SOURCE,
        "external_power_limits_active": n_plim,
        "active_power_requires_review": n_review,
        "thresholds": {
            "V_NOMINAL_PN": V_NOMINAL_PN, "V_TOL": V_TOL, "TRI_TOL": TRI_TOL,
            "PHYS_FRAC": PHYS_FRAC, "REVERSE_FRAC": REVERSE_FRAC, "STUCK_FRAC": STUCK_FRAC,
            "LIVE_MIN": LIVE_MIN, "REDUND_R2": REDUND_R2, "REDUND_STRONG": REDUND_STRONG,
            "CLOCK_R2": CLOCK_R2, "SENT_FLOOR_FRAC": SENT_FLOOR_FRAC,
            "REV_NORMAL": REV_NORMAL, "REV_REVERSED": REV_REVERSED,
            "ENERGY_ALPHA": ENERGY_ALPHA, "COHERENCE_WARN_RATIO": COHERENCE_WARN_RATIO,
        },
        "phase_consistency_counts": phase_counts,
        "channel_class_counts": cls_counts,
        "active_power_valid_channels": n_apv,
        "active_power_requires_sensitivity": int(len(req_sens)),
        "energy_coherence_alert": alert_rows,
        "read_only": "does not correct/remove/impute; diagnostic only",
        "outputs": {
            "phys_audit": str(p_phys.relative_to(PROJECT_ROOT)),
            "info_audit": str(p_info.relative_to(PROJECT_ROOT)),
            "channel_classification": str(p_cls.relative_to(PROJECT_ROOT)),
            "energy_coherence_alert": str(p_alert.relative_to(PROJECT_ROOT)),
            "summary_md": str(p_md.relative_to(PROJECT_ROOT)),
            "log": str(log_path.relative_to(PROJECT_ROOT)),
        },
        "status": "success",
    }
    p_manifest = MANIFESTS_DIR / "03_audit_physical_informational_params.json"
    with open(p_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved: {p_manifest.relative_to(PROJECT_ROOT)}")

    logger.info("=== 03 done ===")
    logger.info(f"Phase: {phase_counts} | Channels: {cls_counts} | ACTIVE_POWER_VALID={n_apv}")


if __name__ == "__main__":
    main()
