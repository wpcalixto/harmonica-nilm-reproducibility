#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 08_fill_gaps_no_clipping.py

File type: Data preprocessing script (pipeline stage 08: analog/fractal gap filling by wave portions)

Purpose:
    Analog/fractal gap filling by wave portions.

    Fills the gaps AFTER the sanitisation of stage 06/06b. The main input is the sanitised base
    with NaN at the points to fill. The category rule / number of wave portions is NOT fixed
    in this script: it is read from the file produced upstream (`audits/06_gap_rule.csv`).

    WITHOUT CLIPPING (this version): the amplitude guard NO LONGER uses the hard cut
    `np.clip(filled, obs.min, obs.max)`, which flattened ~214k filled points at the observed
    extremes. Instead, the texture of the gap is smoothly rescaled around the base line /
    anchor; only a negligible residual clip remains. Result: filling with natural amplitude
    (not capped), with no points left unfilled.

    Default inputs:
        data/interim/06_para_preencher_sanitizado.csv
        audits/06_gap_rule.csv
        audits/06_dead_channels.csv
        audits/06_long_gaps.csv        (optional, used for traceability)

    Default outputs:
        data/processed/08_preenchido.parquet
        data/processed/08_fill_confidence.parquet     (per-point confidence mask)
        data/processed/08_preenchido_referencia.parquet (scientific reference matrix)
        audits/08_fill_traceability.csv
        audits/08_gap_rule_used.csv
        figdata/08_preenchido.parquet
        logs/08_fill_gaps.log
        manifests/08_fill_gaps_params.json

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
import math
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # repository root (parent of scripts/)

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
AUDITS_DIR = PROJECT_ROOT / "audits"
LOGS_DIR = PROJECT_ROOT / "logs"
MANIFESTS_DIR = PROJECT_ROOT / "manifests"
FIGDATA_DIR = PROJECT_ROOT / "figdata"

DEFAULT_INPUT_CSV = INTERIM_DIR / "06_para_preencher_sanitizado.csv"
DEFAULT_OUTPUT_PARQUET = PROCESSED_DIR / "08_preenchido.parquet"
DEFAULT_GAP_RULE_CSV = AUDITS_DIR / "06_gap_rule.csv"
DEFAULT_LONG_GAPS_CSV = AUDITS_DIR / "06_long_gaps.csv"
DEFAULT_DEAD_CHANNELS_CSV = AUDITS_DIR / "06_dead_channels.csv"

TIME_COL, METER_COL = "time", "meter_id"

# channel side (energy has a sentinel; voltage does not); labelling only
RE_ENERGY = re.compile(
    r"^(i_[abc]n|p_[abc]|q_[abc]|s_[abc]|cos_[abc]|thdi_[abc]|hrm_i_[abc]n_)"
)

# ════════════════════════════════════════════════════════════════════════════
#  PARAMETERS OF THE ANALOG/FRACTAL FILLING
# ════════════════════════════════════════════════════════════════════════════
K_CALIB_L = [60, 180, 360, 720]
K_CALIB_K = [1.16, 1.57, 1.61, 1.84]
W = 6
N_CAND = 8
MIN_CTX = 4
MIN_FRAG = 3
MIN_OBS_SERIES = 12
SERIES_SAMPLE = 64
AMP_TINY = 1e-9
SEED = 42
MAX_PORTION = 120
MICRO_GAP = 2

# Range guard: avoids the hard cut that flattened filled segments at the observed
# minimum/maximum. Instead of applying np.clip point by point, the texture of the
# gap is smoothly rescaled around the base line / anchor. A final clip remains only
# as residual protection, with a margin beyond the observed range.
RANGE_GUARD_MARGIN_FRAC = 0.05
RANGE_GUARD_EPS = 1e-12

_LOG_KL = np.log(K_CALIB_L)
GAP_RULE = []

# ════════════════════════════════════════════════════════════════════════════
#  v3 (PRODUCTION METHOD): adaptive external trend + multi-criterion selection
#  + envelope guard. Parameters FROZEN from the Optuna HPO (fill HPO study
#  _v3, 200 trials, tuning seed 7, revalidated on seed 101).
#  The previous v1 (line + frequency-only selection) is preserved as
#  analog_fill_v1_legacy.
# ════════════════════════════════════════════════════════════════════════════
W_FREQ, W_AMP, W_SLOPE, W_LEVEL = 1.833039, 0.636921, 1.018538, 1.777577
CTX_TREND = 123        # lateral window (min) of the external trend (weighted Theil-Sen)
ALPHA_EDGE = 0.196578  # edge-closure zone (only at the ends)
TREND_TAU = 0.269364   # decay of the weight with proximity to the edge (x CTX_TREND)
SLOPE_CAP_K = 1.589493 # |beta_ext| <= K*|delta/T|
ENV_K = 4.553440       # winsorisation of the texture at +/-K*Ac
BLEND_LO, BLEND_HI = 46.0, 342.0   # adaptive blend by length (lambda_L)
TAU_Z, S_Z = 3.894467, 1.579300    # modulation by trend confidence (lambda_C)
EDGE_SLOPE_K = 3.0     # marker of edge-slope risk
REGIME_TAU = 4.0       # marker of possible regime change
REGIME_MIN_LEN = 180
LONGGAP_LOWCONF = 720  # T > 720 min -> low confidence (mask for sensitivity)

# v3.1: backbone by ROBUST ANCHORS. Fixes the single-point anchoring
# (immediate neighbour, if a dip/outlier or lateral transient, raised the backbone -> the
# envelope guard flattened the texture -> almost linear filling in long gaps: M6/M7).
W_LAT = 600            # lateral window (min) of the robust median/MAD (wide: ignores transients)
EDGE_Z = 4.0           # anomalous edge (|v_edge - lateral_median|/MAD > EDGE_Z) -> absorbed
K_DRIFT = 2.0          # maximum total drift of the slope = K_DRIFT * MAD_pooled
DRIFT_L0, DRIFT_P = 342.0, 2.0   # g_L = 1/(1+(Lg/L0)^p): damps the slope in long gaps

# Codes of the filling confidence mask (parallel parquet 08_fill_confidence).
# Taxonomy with a support x contingency distinction: a gap receives analog texture only when
# there is an admissible library of fragments (predominance of observations, limited reuse);
# otherwise, only the central contingency backbone is returned (UNSUPPORTED_BACKBONE),
# marked with the lowest confidence and NOT equated to supported analog filling.
CONF_OBSERVED, CONF_SHORT, CONF_MEDIUM = 0, 1, 2
CONF_LONG_LOWCONF, CONF_CAT4_LOWCONF, CONF_DEAD = 3, 4, 5
CONF_UNSUPPORTED = 6
CONF_LABELS = {0: "OBSERVED", 1: "FILLED_SHORT", 2: "FILLED_MEDIUM",
               3: "FILLED_LONG_LOWCONF", 4: "FILLED_CAT4_LOWCONF", 5: "DEAD_NOT_FILLED",
               6: "UNSUPPORTED_BACKBONE"}

# Per-gap support test (third way): analog texture only when the admissible library
# (windows of the portion size, with observed fraction >= SUP_TAU_OBS and no long filling)
# offers >= SUP_MIN_UNIQUE unique blocks and reuse <= SUP_R_MAX; otherwise, contingency backbone.
SUP_TAU_OBS = 0.85
SUP_MIN_UNIQUE = 3
SUP_R_MAX = 5


def parse_args():
    p = argparse.ArgumentParser(
        description="Analog/fractal filling after the sanitisation of stage 06."
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT_CSV,
                   help="Sanitised CSV to fill. Default: data/interim/06_para_preencher_sanitizado.csv")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PARQUET,
                   help="Filled parquet. Default: data/processed/08_preenchido.parquet")
    p.add_argument("--gap-rule", type=Path, default=DEFAULT_GAP_RULE_CSV,
                   help="Dynamic gap rule produced by stage 06. Default: audits/06_gap_rule.csv")
    p.add_argument("--dead-channels", type=Path, default=DEFAULT_DEAD_CHANNELS_CSV,
                   help="Dead channels produced by stage 06. Default: audits/06_dead_channels.csv")
    p.add_argument("--long-gaps", type=Path, default=DEFAULT_LONG_GAPS_CSV,
                   help="Long gaps produced by stage 06, for traceability. Default: audits/06_long_gaps.csv")
    p.add_argument("--seed", type=int, default=SEED, help="Seed of the random generator.")
    return p.parse_args()


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("08_fill_gaps")
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


def lado(col: str) -> str:
    return "energia" if RE_ENERGY.match(col) else "tensao"


def _to_int_or_none(x):
    if pd.isna(x):
        return None
    try:
        return int(float(x))
    except Exception:
        return None


def load_gap_rule(path: Path) -> list[tuple[int, int, int]]:
    """Reads the dynamic rule produced upstream (stage 06).

    Accepts CSVs with columns:
        categoria, min_min, max_min, n_porcoes_onda
    or, for compatibility:
        categoria, min_min, max_min, n_porcoes
    """
    if not path.exists():
        raise FileNotFoundError(f"Rule file not found: {path}")
    rule_df = pd.read_csv(path)
    required = {"min_min", "max_min"}
    if not required.issubset(rule_df.columns):
        raise ValueError(
            f"{path.name} must contain at least the columns min_min and max_min. "
            f"Columns found: {list(rule_df.columns)}"
        )
    ncol = "n_porcoes_onda" if "n_porcoes_onda" in rule_df.columns else "n_porcoes"
    if ncol not in rule_df.columns:
        raise ValueError(
            f"{path.name} must contain n_porcoes_onda or n_porcoes. "
            f"Columns found: {list(rule_df.columns)}"
        )

    out = []
    for _, r in rule_df.sort_values("min_min").iterrows():
        lo = _to_int_or_none(r["min_min"])
        hi = _to_int_or_none(r["max_min"])
        npor = _to_int_or_none(r[ncol])
        if lo is None or npor is None:
            continue
        if hi is None:
            hi = 10**12
        out.append((lo, hi, npor))
    if not out:
        raise ValueError(f"No valid rule was read from {path}")
    return out


def load_dead_pairs(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    dead = pd.read_csv(path)
    if METER_COL not in dead.columns or "variavel" not in dead.columns:
        return set()
    return {(str(r[METER_COL]), str(r["variavel"])) for _, r in dead.iterrows()}


def load_long_gaps_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(len(pd.read_csv(path)))
    except Exception:
        return None


def n_porcoes(length: int) -> int:
    for lo, hi, npor in GAP_RULE:
        if lo <= length <= hi:
            return npor
    return GAP_RULE[-1][2]


def k_of_length(Lg: int) -> float:
    """K(L): amplitude gain by gap size, interpolated in log(L)."""
    return float(np.interp(np.log(max(Lg, 1)), _LOG_KL, K_CALIB_K))


def runs(mask: np.ndarray):
    """List of (start, end_inclusive) where mask is True."""
    out = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def detrend(y: np.ndarray) -> np.ndarray:
    """Removes the straight line between the first and the last point."""
    L = len(y)
    if L < 2:
        return np.zeros(L)
    x = np.arange(L)
    return y - (y[0] + (y[-1] - y[0]) * x / (L - 1))


def freq(y: np.ndarray) -> float:
    """Frequency = number of direction changes per point."""
    if len(y) < 3:
        return 0.0
    return float(np.sum(np.diff(np.sign(np.diff(y))) != 0) / (len(y) - 2))


def context_dev(v: np.ndarray, gs: int, ge: int):
    before = v[max(0, gs - W):gs]
    after = v[ge + 1:ge + 1 + W]
    before = before[np.isfinite(before)]
    after = after[np.isfinite(after)]
    parts = []
    if len(before) >= 2:
        parts.append(detrend(before))
    if len(after) >= 2:
        parts.append(detrend(after))
    return np.concatenate(parts) if parts else np.array([])


def pick_fragment(v, lib, Lg, Fc, rng):
    usable = [(s, e) for (s, e) in lib if (e - s + 1) >= Lg]
    if not usable:
        return None
    weights = np.array([e - s + 1 - Lg + 1 for (s, e) in usable], dtype=float)
    weights /= weights.sum()
    best = None
    best_df = np.inf
    for _ in range(N_CAND):
        ri = rng.choice(len(usable), p=weights)
        s, e = usable[ri]
        off = rng.integers(0, (e - s + 1) - Lg + 1)
        st = s + int(off)
        frag = v[st:st + Lg]
        if not np.all(np.isfinite(frag)):
            continue
        df = abs(freq(detrend(frag)) - Fc)
        if df < best_df:
            best_df = df
            best = frag
    return None if best is None else best.copy()


def series_texture(v, lib, rng, wlen=6):
    usable = [(s, e) for (s, e) in lib if (e - s + 1) >= wlen]
    if not usable:
        return 0.0, 0.0
    weights = np.array([e - s + 1 - wlen + 1 for (s, e) in usable], dtype=float)
    weights /= weights.sum()
    amps, frqs = [], []
    for _ in range(SERIES_SAMPLE):
        ri = rng.choice(len(usable), p=weights)
        s, e = usable[ri]
        off = rng.integers(0, (e - s + 1) - wlen + 1)
        d = detrend(v[s + int(off):s + int(off) + wlen])
        amps.append(np.std(d))
        frqs.append(freq(d))
    return float(np.median(amps)), float(np.median(frqs))


def interp_hold(v):
    s = pd.Series(v).interpolate("linear", limit_direction="both")
    return s.ffill().bfill().values


def observed_bounds_with_margin(v: np.ndarray):
    """Observed physical range with a small margin.

    The previous version applied np.clip(filled, obs.min(), obs.max()) at the end.
    That produced artificial plateaus at the top and bottom of long gaps when the
    reconstructed texture exceeded the observed range by a few points.

    The new rule uses the observed range only as a guard, with a relative margin.
    The main correction is done by a global rescaling of the gap texture, not by
    point-by-point cutting.
    """
    obs = np.asarray(v, dtype=float)
    obs = obs[np.isfinite(obs)]
    if len(obs) < MIN_OBS_SERIES:
        return None
    lo = float(np.nanmin(obs))
    hi = float(np.nanmax(obs))
    span = max(float(hi - lo), float(np.nanstd(obs)), AMP_TINY)
    margin = float(RANGE_GUARD_MARGIN_FRAC) * span
    return lo - margin, hi + margin


def rescale_segment_to_bounds(candidate: np.ndarray,
                              center: np.ndarray,
                              lower: float,
                              upper: float):
    """Rescales the gap texture to fit the envelope without flattening the wave.

    candidate = proposed signal for the gap.
    center    = base line or anchor around which the texture was added.

    If any point exceeds the envelope, the amplitude of the deviation
    (candidate - center) is reduced once. Thus the wave shape is preserved and no
    horizontal cuts appear at the upper/lower bound.
    """
    y = np.asarray(candidate, dtype=float).copy()
    c = np.asarray(center, dtype=float).copy()
    if y.size == 0:
        return y, 1.0, 0
    c = np.clip(c, lower, upper)
    d = y - c
    scale = 1.0

    pos = d > RANGE_GUARD_EPS
    if np.any(pos):
        allowed = (upper - c[pos]) / d[pos]
        scale = min(scale, float(np.nanmin(allowed)))

    neg = d < -RANGE_GUARD_EPS
    if np.any(neg):
        allowed = (lower - c[neg]) / d[neg]  # d < 0; positive ratio
        scale = min(scale, float(np.nanmin(allowed)))

    if not np.isfinite(scale):
        scale = 1.0
    scale = max(0.0, min(1.0, scale))
    y2 = c + scale * d

    residual = (y2 < lower) | (y2 > upper)
    n_residual = int(np.sum(residual))
    if n_residual:
        # Residual protection only against numerical error or a base line already outside the guard.
        # It is not the main amplitude-control mechanism.
        y2 = np.clip(y2, lower, upper)
    return y2, scale, n_residual


def assign_gap_values(filled: np.ndarray,
                      original: np.ndarray,
                      gs: int,
                      ge: int,
                      candidate: np.ndarray,
                      center: np.ndarray,
                      st: dict) -> None:
    """Assigns filled values with a smooth amplitude guard."""
    bounds = observed_bounds_with_margin(original)
    if bounds is None:
        filled[gs:ge + 1] = candidate
        return
    lower, upper = bounds
    guarded, scale, n_residual = rescale_segment_to_bounds(candidate, center, lower, upper)
    filled[gs:ge + 1] = guarded
    if scale < 0.999999:
        st["n_guard_rescaled"] += int(ge - gs + 1)
    st["n_guard_clipped"] += int(n_residual)


def n_portions_eff(Lg):
    """Effective number of portions: dynamic rule of stage 06 + maximum limit per portion."""
    base = n_porcoes(Lg)
    return max(base, int(np.ceil(Lg / MAX_PORTION))) if Lg > 360 else base


def build_texture(v, lib, Lg, N, Fc, Ac, rng):
    plen = max(MIN_FRAG, int(np.ceil(Lg / N)))
    hop = max(1, plen // 2)
    out = np.zeros(Lg)
    wsum = np.zeros(Lg)
    any_frag = False
    pos = 0
    while pos < Lg:
        Lk = min(plen, Lg - pos)
        if Lk < MIN_FRAG:
            break
        frag = pick_fragment(v, lib, Lk, Fc, rng)
        if frag is not None:
            d = detrend(np.asarray(frag, dtype=float))
            w = np.hanning(Lk) if Lk > 2 else np.ones(Lk)
            out[pos:pos + Lk] += d * w
            wsum[pos:pos + Lk] += w
            any_frag = True
        pos += hop
    if not any_frag:
        return None
    wsum[wsum < 1e-9] = 1.0
    tex = detrend(out / wsum)
    s = float(np.std(tex))
    if s < AMP_TINY:
        return None
    return tex / s * Ac


# ════════════════════════════════════════════════════════════════════════════
#  Blocks of v3 (multi-criterion selection + adaptive external trend)
# ════════════════════════════════════════════════════════════════════════════
def _smoothstep(z):
    z = np.clip(z, 0.0, 1.0)
    return z * z * (3.0 - 2.0 * z)


def _robust_slope(y):
    """Robust slope (Theil-Sen): median of the slopes of all pairs (i<j).
    Vectorised (np.triu_indices): same set of pairs as the O(n^2) loop, identical median."""
    y = np.asarray(y, float); y = y[np.isfinite(y)]
    n = len(y)
    if n < 2:
        return 0.0
    i, j = np.triu_indices(n, k=1)
    return float(np.median((y[j] - y[i]) / (j - i)))


def _weighted_robust_line(t, y, w):
    """WEIGHTED least-squares line (proximity) on smoothed data."""
    t = np.asarray(t, float); y = np.asarray(y, float); w = np.asarray(w, float)
    sw = float(w.sum())
    if sw <= 0:
        return 0.0, float(np.median(y)) if len(y) else 0.0
    tm = float((w * t).sum() / sw); ym = float((w * y).sum() / sw)
    var = float((w * (t - tm) ** 2).sum())
    if var <= 0:
        return 0.0, ym
    beta = float((w * (t - tm) * (y - ym)).sum() / var)
    return beta, ym - beta * tm


_PICK_CACHE = {}   # Lg -> (usable, weights); valid only inside one analog_fill (fixed lib/channel)


def pick_fragment_multi(v, lib, Lg, Fc, sig_ctx, m_gap, rng):
    """Multi-criterion analog selection: matches frequency, amplitude, slope and level.
    usable/weights (deterministic per Lg, with a fixed lib in the channel) are memoised;
    identical result (same weights -> same RNG choices), only faster."""
    cached = _PICK_CACHE.get(Lg)
    if cached is None:
        usable = [(s, e) for (s, e) in lib if (e - s + 1) >= Lg]
        weights = None
        if usable:
            weights = np.array([e - s + 1 - Lg + 1 for (s, e) in usable], dtype=float)
            weights /= weights.sum()
        _PICK_CACHE[Lg] = (usable, weights)
    else:
        usable, weights = cached
    if not usable:
        return None
    eps = sig_ctx + AMP_TINY
    exp_level = m_gap * Lg
    best, best_d = None, np.inf
    for _ in range(N_CAND):
        ri = rng.choice(len(usable), p=weights)
        s, e = usable[ri]
        off = int(rng.integers(0, (e - s + 1) - Lg + 1))
        pos = s + off
        frag = v[pos:pos + Lg]
        if not np.all(np.isfinite(frag)):
            continue
        fd = detrend(frag)
        d_freq = abs(freq(fd) - Fc)
        d_amp = abs(float(np.std(fd)) - sig_ctx) / eps
        d_slope = abs(_robust_slope(frag) - m_gap) * Lg / eps
        d_level = abs((frag[-1] - frag[0]) - exp_level) / eps
        d = W_FREQ * d_freq + W_AMP * d_amp + W_SLOPE * d_slope + W_LEVEL * d_level
        if d < best_d:
            best_d, best = d, frag
    return None if best is None else best.copy()


def build_texture_multi(v, lib, Lg, N, Fc, Ac, sig_ctx, m_gap, rng):
    """Same as build_texture, but with pick_fragment_multi + envelope guard (ENV_K)."""
    plen = max(MIN_FRAG, int(np.ceil(Lg / N)))
    hop = max(1, plen // 2)
    out = np.zeros(Lg); wsum = np.zeros(Lg); any_frag = False; pos = 0
    while pos < Lg:
        Lk = min(plen, Lg - pos)
        if Lk < MIN_FRAG:
            break
        frag = pick_fragment_multi(v, lib, Lk, Fc, sig_ctx, m_gap, rng)
        if frag is not None:
            d = detrend(np.asarray(frag, dtype=float))
            w = np.hanning(Lk) if Lk > 2 else np.ones(Lk)
            out[pos:pos + Lk] += d * w; wsum[pos:pos + Lk] += w; any_frag = True
        pos += hop
    if not any_frag:
        return None
    wsum[wsum < 1e-9] = 1.0
    tex = detrend(out / wsum)
    s = float(np.std(tex))
    if s < AMP_TINY:
        return None
    tex = tex / s * Ac
    return np.clip(tex, -ENV_K * Ac, ENV_K * Ac)


def external_trend_backbone(filled, gs, ge, vb, va):
    """Backbone v3.1: LEVEL by robust lateral anchors (median/MAD of a W_LAT window),
    with absorption of anomalous edges, smoothstep transition between anchors, damped
    lateral slope (length/agreement) and limited drift. Fixes the single-point anchoring
    that raised the backbone and made the envelope guard flatten the texture (M6/M7).
    The texture is added by the caller. Returns the Lg inner points."""
    n = len(filled); Lg = ge - gs + 1
    left  = filled[max(0, gs - W_LAT):gs];         left  = left[np.isfinite(left)]
    right = filled[ge + 1:min(n, ge + 1 + W_LAT)]; right = right[np.isfinite(right)]
    if len(left) < 3 or len(right) < 3:            # insufficient lateral context: simple line
        u = np.linspace(0.0, 1.0, Lg + 2)
        return (vb + (va - vb) * u)[1:-1]
    m_L, m_R = float(np.median(left)), float(np.median(right))
    s_L = 1.4826 * float(np.median(np.abs(left - m_L))) + AMP_TINY
    s_R = 1.4826 * float(np.median(np.abs(right - m_R))) + AMP_TINY
    close_b = vb if abs(vb - m_L) / s_L <= EDGE_Z else m_L
    close_a = va if abs(va - m_R) / s_R <= EDGE_Z else m_R
    u = np.linspace(0.0, 1.0, Lg); h = 3.0 * u ** 2 - 2.0 * u ** 3
    T0 = m_L + (m_R - m_L) * h
    bL, bR = _robust_slope(left), _robust_slope(right)
    c_beta = float(np.exp(-abs(bL - bR) / (abs(bL) + abs(bR) + AMP_TINY)))
    g_L = 1.0 / (1.0 + (Lg / DRIFT_L0) ** DRIFT_P)
    beta = g_L * c_beta * 0.5 * (bL + bR)
    cap = K_DRIFT * float(np.median([s_L, s_R])) / max(Lg, 1)
    beta = float(np.clip(beta, -cap, cap))
    T0 = T0 + beta * (np.arange(Lg) - Lg / 2.0)
    q = min(0.10, 60.0 / max(Lg, 1))
    wL = np.where(u < q, 1.0 - _smoothstep(u / q), 0.0)
    wR = np.where(u > 1.0 - q, _smoothstep((u - (1.0 - q)) / q), 0.0)
    return T0 + wL * (close_b - T0[0]) + wR * (close_a - T0[-1])


def _backbone_v3_frozen(filled, gs, ge, vb, va):
    """Backbone of v3 (PRESERVED, not called; v3.1 above): external trend
    (weighted Theil-Sen on the real lateral points) + local closure at the edges
    + adaptive blend lambda_L*lambda_C. Returns the Lg inner points (closes exactly at vb/va)."""
    n = len(filled); Lg = ge - gs + 1
    la, ra = gs - 1, ge + 1
    L0, L1 = max(0, la - CTX_TREND), la + 1
    R0, R1 = ra, min(n, ra + CTX_TREND + 1)
    Lidx, Ridx = np.arange(L0, L1), np.arange(R0, R1)
    Lvals = filled[Lidx]; Lvals = Lvals[np.isfinite(Lvals)]
    Rvals = filled[Ridx]; Rvals = Rvals[np.isfinite(Rvals)]
    idx = np.r_[Lidx, Ridx].astype(float)
    vals = filled[np.r_[Lidx, Ridx]]
    m = np.isfinite(vals); idx, vals = idx[m], vals[m]
    gap_idx = np.arange(la, ra + 1, dtype=float)
    T = float(gap_idx[-1] - gap_idx[0])
    if len(vals) < 4 or float(np.ptp(idx)) < 2.0:
        u = (gap_idx - gap_idx[0]) / (gap_idx[-1] - gap_idx[0])
        return (vb + (va - vb) * u)[1:-1]
    vals_s = pd.Series(vals).rolling(5, center=True, min_periods=1).median().values
    dist = np.minimum(np.abs(idx - la), np.abs(idx - ra))
    w = np.exp(-dist / (TREND_TAU * CTX_TREND))
    beta, alpha0 = _weighted_robust_line(idx, vals_s, w)
    beta_cap = SLOPE_CAP_K * abs(va - vb) / max(T, 1.0)
    beta = float(np.clip(beta, -beta_cap, beta_cap))
    sw = float(w.sum())
    tm = float((w * idx).sum() / sw); ym = float((w * vals_s).sum() / sw)
    alpha0 = ym - beta * tm
    T_raw = alpha0 + beta * gap_idx
    db, da = vb - T_raw[0], va - T_raw[-1]
    u = np.linspace(0.0, 1.0, len(gap_idx))
    wb, wa = np.zeros_like(u), np.zeros_like(u)
    lz, rz = u < ALPHA_EDGE, u > (1.0 - ALPHA_EDGE)
    wb[lz] = 1.0 - _smoothstep(u[lz] / ALPHA_EDGE)
    wa[rz] = _smoothstep((u[rz] - (1.0 - ALPHA_EDGE)) / ALPHA_EDGE)
    B = T_raw + db * wb + da * wa
    line_full = vb + (va - vb) * u
    lam_L = _smoothstep((Lg - BLEND_LO) / max(BLEND_HI - BLEND_LO, 1.0))
    mad_L = 1.4826 * float(np.median(np.abs(Lvals - np.median(Lvals)))) if Lvals.size >= 2 else 0.0
    mad_R = 1.4826 * float(np.median(np.abs(Rvals - np.median(Rvals)))) if Rvals.size >= 2 else 0.0
    z_b = abs(db) / (mad_L + AMP_TINY)
    z_a = abs(da) / (mad_R + AMP_TINY)
    lam_C = 1.0 - _smoothstep((max(z_b, z_a) - TAU_Z) / max(S_Z, 1e-6))
    lam = lam_L * lam_C
    B = (1.0 - lam) * line_full + lam * B
    return B[1:-1]


def gap_flags(filled, gs, ge, vb, va, mad_series):
    """Risk markers (do not change the shape): edge_slope_risk, regime, low_conf."""
    Lg = ge - gs + 1
    ctx = context_dev(filled, gs, ge)
    mad_ctx = 1.4826 * float(np.median(np.abs(ctx - np.median(ctx)))) if ctx.size else 0.0
    mad = max(mad_ctx, mad_series, AMP_TINY)
    m_L = _robust_slope(filled[max(0, gs - W):gs])
    m_R = _robust_slope(filled[ge + 1:ge + 1 + W])
    edge_slope_risk = (abs(m_L) * Lg > EDGE_SLOPE_K * mad) or (abs(m_R) * Lg > EDGE_SLOPE_K * mad)
    possible_regime_change = (abs(va - vb) / mad > REGIME_TAU and Lg >= REGIME_MIN_LEN)
    low_conf = (Lg > LONGGAP_LOWCONF) or possible_regime_change
    return edge_slope_risk, possible_regime_change, low_conf


def analog_fill_v1_legacy(v, rng):
    """Fills a series in two passes: micro-gaps and remaining gaps."""
    v = v.astype(float).copy()
    n = len(v)
    st = {
        "n_obs": int(np.isfinite(v).sum()),
        "n_gap": int((~np.isfinite(v)).sum()),
        "n_analog": 0,
        "n_interp": 0,
        "n_edge": 0,
        "n_unfilled": 0,
        "n_guard_rescaled": 0,
        "n_guard_clipped": 0,
        "skip_reason": "",
    }
    if st["n_obs"] < MIN_OBS_SERIES:
        st["n_unfilled"] = st["n_gap"]
        st["skip_reason"] = "min_obs_series"
        return v, st

    filled = v.copy()

    # STEP 0: micro-gaps: linear, to consolidate observed fragments.
    for gs, ge in runs(~np.isfinite(filled)):
        Lg = ge - gs + 1
        if gs == 0 or ge == n - 1 or Lg > MICRO_GAP:
            continue
        vb, va = filled[gs - 1], filled[ge + 1]
        candidate = vb + (va - vb) * np.arange(1, Lg + 1) / (Lg + 1)
        # An inner micro-gap has its own line as centre; the guard acts only
        # if the anchors are already extreme.
        assign_gap_values(filled, v, gs, ge, candidate, candidate, st)
        st["n_interp"] += Lg

    lib = [(s, e) for (s, e) in runs(np.isfinite(filled)) if (e - s + 1) >= MIN_FRAG]
    s_amp, s_freq = series_texture(filled, lib, rng)

    # STEP 1: remaining gaps: base line + texture by portions.
    for gs, ge in runs(~np.isfinite(filled)):
        Lg = ge - gs + 1

        if gs == 0 or ge == n - 1:
            Le = Lg
            if gs == 0:
                anchor = filled[ge + 1]
                src = filled[ge + 1:ge + 1 + Le]
            else:
                anchor = filled[gs - 1]
                src = filled[max(0, gs - Le):gs]
            src = src[np.isfinite(src)]
            center = np.full(Le, anchor, dtype=float)
            if len(src) >= MIN_FRAG and np.std(src) > AMP_TINY:
                tiled = np.tile(src, int(np.ceil(Le / len(src))))[:Le]
                candidate = anchor + detrend(tiled)
                assign_gap_values(filled, v, gs, ge, candidate, center, st)
                st["n_analog"] += Le
            else:
                assign_gap_values(filled, v, gs, ge, center, center, st)
                st["n_edge"] += Le
            continue

        vb, va = filled[gs - 1], filled[ge + 1]
        line = vb + (va - vb) * np.arange(1, Lg + 1) / (Lg + 1)
        ctx = context_dev(filled, gs, ge)
        if len(ctx) >= MIN_CTX and np.std(ctx) > AMP_TINY:
            Ac, Fc = float(np.std(ctx)) * k_of_length(Lg), freq(ctx)
        else:
            Ac, Fc = s_amp, s_freq

        if Ac <= AMP_TINY:
            assign_gap_values(filled, v, gs, ge, line, line, st)
            st["n_interp"] += Lg
            continue

        tex = build_texture(filled, lib, Lg, n_portions_eff(Lg), Fc, Ac, rng)
        if tex is None:
            assign_gap_values(filled, v, gs, ge, line, line, st)
            st["n_interp"] += Lg
            continue

        candidate = line + tex
        # Central correction: the texture is reduced as a whole if it exceeds the observed
        # range with margin. The hard point-by-point clip is no longer used, since it produced
        # flattened waves at the top/bottom in the figures.
        assign_gap_values(filled, v, gs, ge, candidate, line, st)
        st["n_analog"] += Lg

    rem = int((~np.isfinite(filled)).sum())
    if rem:
        before = int((~np.isfinite(filled)).sum())
        held = interp_hold(filled)
        # If a final residual remains, applies the same guard only to the points still missing.
        missing = ~np.isfinite(filled)
        filled2 = filled.copy()
        filled2[missing] = held[missing]
        filled = filled2
        after = int((~np.isfinite(filled)).sum())
        st["n_interp"] += before - after
        st["n_unfilled"] += after

    # The previous version did: filled = np.clip(filled, obs.min(), obs.max()).
    # This was removed because it created artificial plateaus at the bounds of the series.
    return filled, st


def _conf_code(cat, low_conf):
    """Confidence code per gap category (+ low-confidence flag)."""
    if cat <= 1:
        return CONF_SHORT
    if cat >= 4:
        return CONF_CAT4_LOWCONF
    return CONF_LONG_LOWCONF if low_conf else CONF_MEDIUM


def _adm_capacity(is_obs, lib, plen):
    """Number of admissible NON-overlapping windows of size plen (observed fraction >= SUP_TAU_OBS).
    The lib is frozen before the long gaps -> no LONG point enters a window."""
    cap = 0
    for s, e in lib:
        p = s
        while p + plen <= e + 1:
            if is_obs[p:p + plen].mean() >= SUP_TAU_OBS:
                cap += 1; p += plen
            else:
                p += max(1, plen // 2)
    return cap


def gap_supported(is_obs, lib, Lg):
    """Is analog texture supported? Requires >= SUP_MIN_UNIQUE admissible blocks of the
    portion size and reuse <= SUP_R_MAX. plen replicates build_texture_multi (effective portion)."""
    N = n_portions_eff(Lg); plen = max(MIN_FRAG, int(np.ceil(Lg / N)))
    cap = _adm_capacity(is_obs, lib, plen)
    reuse = int(np.ceil(Lg / plen)) / max(1, cap)
    return cap >= SUP_MIN_UNIQUE and reuse <= SUP_R_MAX


def analog_fill(v, rng):
    """PRODUCTION METHOD (v3): backbone = adaptive external trend (weighted Theil-Sen
    on the lateral points + local closure + blend lambda_L*lambda_C), texture by
    multi-criterion selection with envelope guard. Micro-gaps, edges and amplitude
    guard identical to v1. Tracks risk/low-confidence markers in st and the per-point
    confidence mask in st['conf']."""
    v = v.astype(float).copy()
    n = len(v)
    st = {"n_obs": int(np.isfinite(v).sum()), "n_gap": int((~np.isfinite(v)).sum()),
          "n_analog": 0, "n_interp": 0, "n_edge": 0, "n_unfilled": 0,
          "n_guard_rescaled": 0, "n_guard_clipped": 0,
          "n_edge_risk": 0, "n_regime": 0, "n_low_conf": 0, "n_unsupported": 0,
          "skip_reason": ""}
    conf = np.where(np.isfinite(v), CONF_OBSERVED, CONF_DEAD).astype(np.int8)
    if st["n_obs"] < MIN_OBS_SERIES:
        st["n_unfilled"] = st["n_gap"]; st["skip_reason"] = "min_obs_series"
        st["conf"] = conf
        return v, st

    filled = v.copy()
    obs = v[np.isfinite(v)]
    mad_series = 1.4826 * float(np.median(np.abs(obs - np.median(obs)))) if obs.size else 0.0

    # STEP 0: micro-gaps: linear (identical to v1).
    for gs, ge in runs(~np.isfinite(filled)):
        Lg = ge - gs + 1
        if gs == 0 or ge == n - 1 or Lg > MICRO_GAP:
            continue
        vb, va = filled[gs - 1], filled[ge + 1]
        candidate = vb + (va - vb) * np.arange(1, Lg + 1) / (Lg + 1)
        assign_gap_values(filled, v, gs, ge, candidate, candidate, st)
        conf[gs:ge + 1] = CONF_SHORT
        st["n_interp"] += Lg

    lib = [(s, e) for (s, e) in runs(np.isfinite(filled)) if (e - s + 1) >= MIN_FRAG]
    _PICK_CACHE.clear()                                  # donor cache per Lg is per channel
    is_obs = (conf == CONF_OBSERVED)                     # provenance for the support test
    s_amp, s_freq = series_texture(filled, lib, rng)

    # STEP 1: remaining gaps: external trend (v3) + multi-criterion texture.
    for gs, ge in runs(~np.isfinite(filled)):
        Lg = ge - gs + 1
        if gs == 0 or ge == n - 1:  # edges: identical to v1
            Le = Lg
            if gs == 0:
                anchor = filled[ge + 1]; src = filled[ge + 1:ge + 1 + Le]
            else:
                anchor = filled[gs - 1]; src = filled[max(0, gs - Le):gs]
            src = src[np.isfinite(src)]
            center = np.full(Le, anchor, dtype=float)
            if len(src) >= MIN_FRAG and np.std(src) > AMP_TINY:
                tiled = np.tile(src, int(np.ceil(Le / len(src))))[:Le]
                candidate = anchor + detrend(tiled)
                assign_gap_values(filled, v, gs, ge, candidate, center, st)
                st["n_analog"] += Le
            else:
                assign_gap_values(filled, v, gs, ge, center, center, st)
                st["n_edge"] += Le
            conf[gs:ge + 1] = _conf_code(n_porcoes(Lg), False)   # edge: no regime
            continue

        vb, va = filled[gs - 1], filled[ge + 1]
        base = external_trend_backbone(filled, gs, ge, vb, va)   # backbone v3
        er, rc, lc = gap_flags(filled, gs, ge, vb, va, mad_series)
        if er: st["n_edge_risk"] += Lg
        if rc: st["n_regime"] += Lg
        if lc: st["n_low_conf"] += Lg

        # third way: only gaps of cat>=2 (Lg > short-gap limit) may lack support;
        # short ones stay analog (a small-portion library always exists). Without an
        # admissible library -> contingency backbone, WITHOUT texture, lowest confidence.
        if Lg > GAP_RULE[0][1] and not gap_supported(is_obs, lib, Lg):
            assign_gap_values(filled, v, gs, ge, base, base, st)
            conf[gs:ge + 1] = CONF_UNSUPPORTED
            st["n_unsupported"] += Lg
            continue
        conf[gs:ge + 1] = _conf_code(n_porcoes(Lg), lc)   # supported: by category + confidence

        ctx = context_dev(filled, gs, ge)
        if len(ctx) >= MIN_CTX and np.std(ctx) > AMP_TINY:
            sig_ctx = float(np.std(ctx))
            Ac, Fc = sig_ctx * k_of_length(Lg), freq(ctx)
        else:
            sig_ctx, Ac, Fc = s_amp, s_amp, s_freq
        m_gap = (va - vb) / (Lg + 1)

        if Ac <= AMP_TINY:
            assign_gap_values(filled, v, gs, ge, base, base, st)
            st["n_interp"] += Lg
            continue

        tex = build_texture_multi(filled, lib, Lg, n_portions_eff(Lg),
                                  Fc, Ac, sig_ctx, m_gap, rng)
        if tex is None:
            assign_gap_values(filled, v, gs, ge, base, base, st)
            st["n_interp"] += Lg
            continue

        candidate = base + tex
        assign_gap_values(filled, v, gs, ge, candidate, base, st)
        st["n_analog"] += Lg

    rem = int((~np.isfinite(filled)).sum())
    if rem:
        before = int((~np.isfinite(filled)).sum())
        held = interp_hold(filled)
        missing = ~np.isfinite(filled)
        filled2 = filled.copy(); filled2[missing] = held[missing]; filled = filled2
        after = int((~np.isfinite(filled)).sum())
        st["n_interp"] += before - after
        st["n_unfilled"] += after
    # points filled in the tail (were DEAD/NaN and are now finite) -> MEDIUM
    tail_filled = (conf == CONF_DEAD) & np.isfinite(filled)
    conf[tail_filled] = CONF_MEDIUM
    st["conf"] = conf
    return filled, st


def main():
    args = parse_args()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(LOGS_DIR / "08_fill_gaps.log")
    logger.info("=== 08_fill_gaps_no_clipping.py: analog/fractal filling after sanitisation ===")
    logger.info(f"Run timestamp: {datetime.now().isoformat()}")

    global GAP_RULE
    GAP_RULE = load_gap_rule(args.gap_rule)
    dead_pairs = load_dead_pairs(args.dead_channels)
    n_long_gaps = load_long_gaps_count(args.long_gaps)

    logger.info(f"Input: {args.input}")
    logger.info(f"Gap rule: {args.gap_rule} -> {GAP_RULE}")
    logger.info(f"Dead channels reported by stage 06: {len(dead_pairs)}")
    if n_long_gaps is not None:
        logger.info(f"Long gaps reported by stage 06: {n_long_gaps}")
    else:
        logger.info("File 06_long_gaps.csv not found or unreadable; the filling will follow the NaN mask of the input.")

    df = pd.read_csv(args.input, parse_dates=[TIME_COL])
    cols = [c for c in df.columns if c not in (METER_COL, TIME_COL)]
    logger.info(
        f"Read: {args.input.name} | {len(df)} rows x {len(cols)} variables | "
        f"{df[METER_COL].nunique()} meters"
    )

    rng = np.random.default_rng(args.seed)
    dfx = df.sort_values([METER_COL, TIME_COL]).reset_index(drop=True)
    out = dfx.copy()
    conf_out = dfx.copy()               # parallel mask parquet (confidence codes)
    trace = []

    logger.info("Filling gaps with fragments of the series itself and the dynamic rule of stage 06...")
    for m in sorted(dfx[METER_COL].unique()):
        mask = (dfx[METER_COL] == m).values
        for c in cols:
            v = dfx.loc[mask, c].values.astype(float)
            key = (str(m), c)
            if key in dead_pairs:
                stt = {
                    "n_obs": int(np.isfinite(v).sum()),
                    "n_gap": int((~np.isfinite(v)).sum()),
                    "n_analog": 0,
                    "n_interp": 0,
                    "n_edge": 0,
                    "n_unfilled": int((~np.isfinite(v)).sum()),
                    "n_guard_rescaled": 0,
                    "n_guard_clipped": 0,
                    "skip_reason": "dead_channel_06",
                }
                filled = v
                cf = np.where(np.isfinite(v), CONF_OBSERVED, CONF_DEAD).astype(np.int8)
            else:
                filled, stt = analog_fill(v, rng)
                cf = stt.pop("conf")
            out.loc[mask, c] = filled
            conf_out.loc[mask, c] = cf
            stt.update({METER_COL: m, "variavel": c, "lado": lado(c)})
            trace.append(stt)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)

    # Per-point confidence mask (same shape as the value): enables sensitivity
    # analyses and the exclusion of cat4/low-confidence windows in training.
    for c in cols:
        conf_out[c] = conf_out[c].astype(np.int8)
    conf_path = args.output.with_name("08_fill_confidence.parquet")
    conf_out.to_parquet(conf_path, index=False)
    counts = {CONF_LABELS[k]: int((conf_out[cols].values == k).sum()) for k in CONF_LABELS}
    logger.info(f"Confidence mask -> {conf_path.name} | count per class: {counts}")

    # SCIENTIFIC REFERENCE matrix: identical to the complete one, but with the points without
    # textural support (UNSUPPORTED_BACKBONE) masked (NaN): for metrics, model selection and
    # evaluation only on valid targets. The COMPLETE matrix (args.output) keeps the contingency
    # backbone only out of computational necessity (avoid NaN in the scripts), always accompanied
    # by the mandatory confidence mask.
    ref = out.copy()
    for c in cols:
        ref.loc[conf_out[c].values == CONF_UNSUPPORTED, c] = np.nan
    ref_path = args.output.with_name("08_supported_reference.parquet")
    ref.to_parquet(ref_path, index=False)
    n_masked = int((conf_out[cols].values == CONF_UNSUPPORTED).sum())
    logger.info(f"Scientific reference matrix -> {ref_path.name} | {n_masked} "
                f"UNSUPPORTED_BACKBONE points masked (not equated to supported filling)")

    FIGDATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.output, FIGDATA_DIR / "08_preenchido.parquet")
    shutil.copyfile(conf_path, FIGDATA_DIR / "08_fill_confidence.parquet")
    shutil.copyfile(ref_path, FIGDATA_DIR / "08_supported_reference.parquet")

    AUDITS_DIR.mkdir(parents=True, exist_ok=True)
    tdf = pd.DataFrame(trace)
    tdf.to_csv(AUDITS_DIR / "08_fill_traceability.csv", index=False)
    shutil.copyfile(args.gap_rule, AUDITS_DIR / "08_gap_rule_used.csv")

    for c in ["n_guard_rescaled", "n_guard_clipped"]:
        if c not in tdf.columns:
            tdf[c] = 0
        tdf[c] = tdf[c].fillna(0)
    tot_keys = ["n_gap", "n_analog", "n_interp", "n_edge", "n_unfilled", "n_guard_rescaled", "n_guard_clipped"]
    tot = {k: int(tdf[k].sum()) for k in tot_keys}
    logger.info(
        f"FILLING of {tot['n_gap']} gap points: "
        f"analog={tot['n_analog']} . line/interp={tot['n_interp']} . "
        f"edge={tot['n_edge']} . unfilled={tot['n_unfilled']} . "
        f"guard_rescaled={tot['n_guard_rescaled']} . guard_residual_clip={tot['n_guard_clipped']}"
    )
    logger.info(f"Output: {args.output}")
    logger.info("Audit: audits/08_fill_traceability.csv . audits/08_gap_rule_used.csv")

    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFESTS_DIR / "08_fill_gaps_params.json").write_text(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "gap_rule_csv": str(args.gap_rule),
                "gap_rule_loaded": GAP_RULE,
                "dead_channels_csv": str(args.dead_channels),
                "n_dead_pairs_loaded": len(dead_pairs),
                "long_gaps_csv": str(args.long_gaps),
                "n_long_gaps_loaded": n_long_gaps,
                "fill_totals": tot,
                "fill_params": {
                    "K_calib_L": K_CALIB_L,
                    "K_calib_K": K_CALIB_K,
                    "W": W,
                    "N_CAND": N_CAND,
                    "MIN_FRAG": MIN_FRAG,
                    "MIN_OBS_SERIES": MIN_OBS_SERIES,
                    "MICRO_GAP": MICRO_GAP,
                    "MAX_PORTION": MAX_PORTION,
                    "RANGE_GUARD_MARGIN_FRAC": RANGE_GUARD_MARGIN_FRAC,
                    "range_guard_policy": "reescala_suave_da_textura_sem_clip_global_minmax",
                    "SEED": args.seed,
                },
                "timestamp": datetime.now().isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("=== 08_fill_gaps_no_clipping.py completed ===")


if __name__ == "__main__":
    main()
