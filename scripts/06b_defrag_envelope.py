#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 06b_defrag_envelope.py

File type: Data preprocessing script (pipeline stage 06b: defragmentation + envelope sanitisation)

Purpose:
    Defragmentation + envelope sanitisation (single self-contained script; runs between 06 and 07).

    STEP 1: Defragmentation. For each fragmented series (meter x variable), reconstructs by
        the most correlated DONOR (a*donor+b) + overlap-add of real islands (Hanning window).
        No linear interpolation. Fills only where M1 has a reading (global-gap lock).
        Intermediate output: data/interim/06A_desfragmentado.csv.

    STEP 2: Envelope (approved method). Per series, uses a clean REFERENCE SEGMENT as the
        amplitude pattern. The window is anchored by TIMESTAMP (--ref-start-time/--ref-end-time,
        default 2020-10-28 15:59 -> 2020-10-29 05:18 UTC), resolved by temporal search in each
        series; generic, independent of the grid origin.
        Envelope = [min_ref - k*std, max_ref + k*std]. Points outside:
           - 1 isolated point -> clip at the bound;
           - block (2+) -> replacement by a real fragment of the reference segment;
           - initial block (< --startup-scan-max-points) -> a single synthetic segment.
        Iterative passes + final envelope guard. Runs with --min-ref-obs-frac 0.45.
        Recomputes the gap files (gap_rule/long_gaps/dead_channels) for script 07.

    Default input (output of script 06):
        data/interim/06_para_preencher.csv

    Main outputs:
        data/interim/06A_desfragmentado.csv             (intermediate, after STEP 1)
        data/interim/06_para_preencher_sanitizado.csv   (final, input of script 07)
        audits/06A_defrag_map.csv
        audits/06_reference_amplitude_thresholds.csv
        audits/06_reference_amplitude_repair_map.csv
        audits/06_reference_amplitude_action_summary.csv
        audits/06_gap_rule.csv . 06_long_gaps.csv . 06_dead_channels.csv
        figdata/06_para_preencher_sanitizado.csv
        logs/06_sanitize_waveforms_reference_segment_repair.log
        manifests/06_sanitize_waveforms_reference_segment_repair_params.json

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
from typing import List, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_project_root() -> Path:
    """Repository root: the parent of the directory that contains this script."""
    return SCRIPT_DIR.parent


PROJECT_ROOT = resolve_project_root()
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
AUDITS_DIR = PROJECT_ROOT / "audits"
TABLES_DIR = PROJECT_ROOT / "tables"
FIGDATA_DIR = PROJECT_ROOT / "figdata"
LOGS_DIR = PROJECT_ROOT / "logs"
MANIFESTS_DIR = PROJECT_ROOT / "manifests"

TIME_COL = "time"
METER_COL = "meter_id"
DEFAULT_INPUT = INTERIM_DIR / "06_para_preencher.csv"
DEFAULT_OUTPUT = INTERIM_DIR / "06_para_preencher_sanitizado.csv"

RE_ENERGY = re.compile(r"^(i_[abc]n|p_[abc]|q_[abc]|s_[abc]|cos_[abc]|thdi_[abc]|hrm_i_[abc]n_)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: DEFRAGMENTATION: donor + overlap-add of islands
# ─────────────────────────────────────────────────────────────────────────────
TIME, MET = "time", "meter_id"
MAIN_METER = 1

MED_ISLAND_MAX = 5
MAX_ISLAND_SELF = 60
NAN_LO, NAN_HI = 0.10, 0.95
MIN_ISLANDS = 30
MIN_DONOR_COV = 0.55
MIN_CORR = 0.5
MIN_ISLAND_LIB = 8
CTX = 60
RNG = np.random.default_rng(42)


def finite_runs(v):
    fin = np.isfinite(v); o = []; i = 0; n = len(fin)
    while i < n:
        if fin[i]:
            j = i
            while j < n and fin[j]:
                j += 1
            o.append((i, j - 1)); i = j
        else:
            i += 1
    return o


def robust_fit(y_tgt, x_src):
    a, b = np.polyfit(x_src, y_tgt, 1)
    return float(a), float(b)


def overlap_add_fill(L, islands, lvl, rng, overlap_frac=0.4):
    if not islands:
        return None
    acc = np.zeros(L); wsum = np.zeros(L)
    pos = 0; guard = 0
    while pos < L and guard < 100000:
        guard += 1
        isl = islands[int(rng.integers(0, len(islands)))]
        k = len(isl)
        if k < 2:
            continue
        w = np.hanning(k)
        end = min(pos + k, L)
        acc[pos:end] += isl[:end - pos] * w[:end - pos]
        wsum[pos:end] += w[:end - pos]
        pos += max(1, int(k * (1.0 - overlap_frac)))
    wsum[wsum < 1e-9] = 1.0
    return acc / wsum + lvl


def run_defrag(input_path, output_path, prefix="06A", logger=None):
    df = pd.read_csv(input_path, parse_dates=[TIME], low_memory=False)
    varc = [c for c in df.columns if c not in (MET, TIME)]
    for c in varc:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    meters = sorted(df[MET].unique())
    df = df.sort_values([MET, TIME]).reset_index(drop=True)
    base = df[df[MET] == meters[0]].sort_values(TIME)
    n = len(base)
    S = {(m, c): df[df[MET] == m].sort_values(TIME)[c].to_numpy(float) for m in meters for c in varc}
    # global order per meter (to write back into the df)
    order = {m: df.index[df[MET] == m].to_numpy() for m in meters}

    def coverage(m, c):
        return float(np.isfinite(S[(m, c)]).mean())

    def ranked_donors(m, c):
        tv = S[(m, c)]; cand = []
        for d in meters:
            if d == m:
                continue
            dv = S[(d, c)]
            mask = np.isfinite(tv) & np.isfinite(dv)
            if mask.sum() < 100 or coverage(d, c) < MIN_DONOR_COV:
                continue
            if np.std(tv[mask]) < 1e-9 or np.std(dv[mask]) < 1e-9:
                continue
            r = float(np.corrcoef(tv[mask], dv[mask])[0, 1])
            if np.isfinite(r) and abs(r) >= MIN_CORR:
                cand.append((abs(r), d))
        cand.sort(reverse=True)
        return [(d, r) for r, d in cand]

    out = df.copy()
    rows = []
    n_series = 0; n_pts = 0

    for m in meters:
        for c in varc:
            v = S[(m, c)]
            nfin = int(np.isfinite(v).sum())
            if nfin < 50:
                continue
            runs = finite_runs(v)
            isl = [e - s + 1 for s, e in runs]
            med_isl = float(np.median(isl)); max_isl = max(isl)
            pct_nan = 1.0 - nfin / n
            fragmented = (med_isl <= MED_ISLAND_MAX and len(runs) >= MIN_ISLANDS
                          and NAN_LO <= pct_nan <= NAN_HI and max_isl < MAX_ISLAND_SELF)
            if not fragmented:
                continue
            donors = ranked_donors(m, c)
            if not donors:
                rows.append(dict(meter_id=m, variavel=c, status="sem_doador", pct_nan=round(pct_nan, 3)))
                continue

            # global gap = M1 (same variable) without reading -> do NOT fill there
            allow = np.isfinite(S[(MAIN_METER, c)]) if (MAIN_METER, c) in S else np.ones(n, bool)
            gap = ~np.isfinite(v)
            cur = v.copy()

            # 1) cascade of donors
            recon = np.full(n, np.nan); usados = []
            for d, r in donors:
                dv = S[(d, c)]
                mask = np.isfinite(v) & np.isfinite(dv)
                if mask.sum() < 100:
                    continue
                a, b = robust_fit(v[mask], dv[mask])
                pred = a * dv + b
                take = gap & allow & ~np.isfinite(recon) & np.isfinite(pred)
                if take.any():
                    recon[take] = pred[take]; usados.append(f"M{d}({r:.2f}):{int(take.sum())}")
            u1 = gap & allow & np.isfinite(recon)
            cur[u1] = recon[u1]

            # 2) residual (within allow) by overlap-add of real islands
            islands = [cur[s:e + 1] - np.nanmedian(cur[s:e + 1])
                       for s, e in finite_runs(cur) if (e - s + 1) >= MIN_ISLAND_LIB]
            res = (~np.isfinite(cur)) & allow
            if res.any() and islands:
                for gs, ge in finite_runs(np.where(res, 0.0, np.nan)):
                    L = ge - gs + 1
                    left = cur[max(0, gs - CTX):gs]; right = cur[ge + 1:ge + 1 + CTX]
                    nb = np.concatenate([left[np.isfinite(left)], right[np.isfinite(right)]])
                    lvl = float(np.median(nb)) if len(nb) else 0.0
                    fill = overlap_add_fill(L, islands, lvl, RNG)
                    if fill is not None:
                        cur[gs:ge + 1] = fill

            # writes back into the df (in the temporal order of the meter)
            out.loc[order[m], c] = cur
            n_series += 1
            n_rec = int(np.isfinite(cur).sum() - nfin)
            n_pts += n_rec
            rows.append(dict(meter_id=m, variavel=c, status="reconstruido", doadores="|".join(usados),
                             pct_nan=round(pct_nan, 3), n_reconstruido=n_rec,
                             cobertura_final=round(np.isfinite(cur).mean(), 3),
                             lacuna_global_preservada=int((gap & ~allow).sum())))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    FIGDATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output_path, FIGDATA_DIR / Path(output_path).name)
    AUDITS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(AUDITS_DIR / f"{prefix}_defrag_map.csv", index=False)
    print(f"Defragmentation: {n_series} fragmented series | {n_pts} reconstructed points")
    print(f"Output: {output_path}")
    if 15 in meters:
        for c in ["i_an", "i_bn", "i_cn"]:
            o = S[(15, c)]; r = out[out[MET] == 15].sort_values(TIME)[c].to_numpy(float)
            print(f"  M15 {c}: {100*np.isfinite(o).mean():.0f}% -> {100*np.isfinite(r).mean():.0f}%")





def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("06_reference_segment_repair")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def contiguous_runs(mask: np.ndarray, min_run: int = 1) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not mask.any():
        return []
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends) if (e - s + 1) >= min_run]


def close_short_false_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if max_gap <= 0 or mask.size == 0 or not mask.any():
        return mask.copy()
    out = mask.copy()
    runs = contiguous_runs(mask, 1)
    for (_, e1), (s2, _) in zip(runs[:-1], runs[1:]):
        gap = s2 - e1 - 1
        if 0 < gap <= max_gap:
            out[e1 + 1:s2] = True
    return out


def phase_of(col: str) -> str | None:
    m = (
        re.match(r"^(?:i|v)_([abc])n$", col)
        or re.match(r"^(?:p|q|s|cos|thdi|thdv)_([abc])$", col)
        or re.match(r"^hrm_[iv]_([abc])n_", col)
    )
    return m.group(1) if m else None


def family_of(col: str) -> str:
    if col.startswith("hrm_i_"):
        return "hrm_i"
    if col.startswith("hrm_v_"):
        return "hrm_v"
    return col.split("_")[0]


def lado(col: str) -> str:
    return "energia" if RE_ENERGY.match(col) else "tensao"


def infer_grid_step_min(df: pd.DataFrame) -> int:
    vals = []
    if TIME_COL not in df.columns:
        return 1
    for _, g in df[[METER_COL, TIME_COL]].dropna().groupby(METER_COL):
        t = pd.to_datetime(g[TIME_COL], utc=True, errors="coerce").dropna().sort_values()
        if len(t) >= 2:
            dt = t.diff().dropna().dt.total_seconds().to_numpy() / 60.0
            dt = dt[np.isfinite(dt) & (dt > 0)]
            vals.extend(dt.tolist())
    return max(1, int(round(float(np.nanmedian(vals))))) if vals else 1


def resample_reference_segment(ref_values: np.ndarray, length: int) -> np.ndarray | None:
    """Resamples the reference segment of the series itself to the length of the block."""
    ref = np.asarray(ref_values, dtype=float)
    if length <= 0:
        return np.array([], dtype=float)
    if np.isfinite(ref).sum() < 3:
        return None
    s = pd.Series(ref).interpolate("linear", limit_direction="both").ffill().bfill()
    arr = s.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        return None
    if len(arr) == length:
        return arr.copy()
    src_x = np.linspace(0.0, 1.0, len(arr))
    dst_x = np.linspace(0.0, 1.0, length)
    return np.interp(dst_x, src_x, arr)


def build_threshold(v: np.ndarray, ref_start: int, ref_end: int, sd_k: float, min_ref_obs_frac: float):
    ref = v[ref_start:ref_end + 1]
    finite = ref[np.isfinite(ref)]
    n_ref = len(ref)
    n_fin = len(finite)
    if n_ref <= 0 or n_fin < max(3, int(np.ceil(min_ref_obs_frac * n_ref))):
        return None
    ref_min = float(np.nanmin(finite))
    ref_max = float(np.nanmax(finite))
    ref_mean = float(np.nanmean(finite))
    ref_std = float(np.nanstd(finite, ddof=1)) if n_fin >= 2 else 0.0
    lower = ref_min - sd_k * ref_std
    upper = ref_max + sd_k * ref_std
    return {
        "ref_values": ref,
        "ref_n_total": int(n_ref),
        "ref_n_finite": int(n_fin),
        "ref_min": ref_min,
        "ref_max": ref_max,
        "ref_mean": ref_mean,
        "ref_std": ref_std,
        "lower": float(lower),
        "upper": float(upper),
        "ref_amplitude": float(ref_max - ref_min),
        "limit_amplitude_ref_plus_sd": float((ref_max - ref_min) + sd_k * ref_std),
    }


def build_repair_segments(
    violation: np.ndarray,
    *,
    startup_scan_max_points: int,
    startup_min_bad_points: int,
    startup_min_span_points: int,
    merge_block_gap_points: int,
    clip_max_run_points: int,
) -> List[dict]:
    """Turns points outside the envelope into actions: clip or synthetic replacement."""
    violation = np.asarray(violation, dtype=bool)
    n = len(violation)
    remaining = violation.copy()
    segments: List[dict] = []

    # Initial block: merges violations close to the start into a single segment, when supported.
    scan_end = min(n - 1, int(startup_scan_max_points))
    if scan_end >= 0:
        early_idx = np.flatnonzero(violation[:scan_end + 1])
        if len(early_idx) >= int(startup_min_bad_points):
            span_s = int(early_idx[0])
            span_e = int(early_idx[-1])
            if (span_e - span_s + 1) >= int(startup_min_span_points):
                segments.append({
                    "x_inicio": span_s,
                    "x_fim": span_e,
                    "acao_sintetica": "substituir_por_trecho_referencia_da_propria_serie",
                    "motivo": "bloco_inicial_fora_do_envelope",
                })
                remaining[span_s:span_e + 1] = False

    # Other blocks: closes small violation-free intervals and classifies by length.
    closed = close_short_false_gaps(remaining, int(merge_block_gap_points))
    for s, e in contiguous_runs(closed, 1):
        span_len = e - s + 1
        # Ensures that the closed block contains at least one original violation.
        bad_count = int(remaining[s:e + 1].sum())
        if bad_count == 0:
            continue
        if span_len <= int(clip_max_run_points) and bad_count <= int(clip_max_run_points):
            action = "clipar_no_limite_do_envelope"
            reason = "ponto_isolado_fora_do_envelope"
        else:
            action = "substituir_por_trecho_referencia_da_propria_serie"
            reason = "bloco_fora_do_envelope"
        segments.append({"x_inicio": int(s), "x_fim": int(e), "acao_sintetica": action, "motivo": reason})
    segments.sort(key=lambda r: (r["x_inicio"], r["x_fim"]))
    return segments



def expand_substitution_segment(seg: dict, n: int, padding: int) -> dict:
    """Expands only replacement blocks to include residual edges.

    Isolated points keep the clip action and are not expanded, following the
    methodological rule: a single point outside the envelope must be clipped at the
    bound; a longer segment must be replaced by a pattern of the series itself.
    """
    out = dict(seg)
    if int(padding) <= 0:
        return out
    if str(out.get("acao_sintetica", "")) != "substituir_por_trecho_referencia_da_propria_serie":
        return out
    out["x_inicio"] = max(0, int(out["x_inicio"]) - int(padding))
    out["x_fim"] = min(int(n) - 1, int(out["x_fim"]) + int(padding))
    out["motivo"] = str(out.get("motivo", "")) + "_com_margem"
    return out

def apply_segment_action(v: np.ndarray, seg: dict, thr: dict) -> tuple[np.ndarray, str]:
    out = v.copy()
    s, e = int(seg["x_inicio"]), int(seg["x_fim"])
    L = e - s + 1
    lower, upper = float(thr["lower"]), float(thr["upper"])
    if seg["acao_sintetica"] == "clipar_no_limite_do_envelope":
        part = out[s:e + 1]
        finite = np.isfinite(part)
        part2 = part.copy()
        part2[finite] = np.clip(part2[finite], lower, upper)
        out[s:e + 1] = part2
        return out, "aplicado_clip"

    repl = resample_reference_segment(np.asarray(thr["ref_values"], dtype=float), L)
    if repl is None:
        return out, "falha_sem_referencia_suficiente"
    repl = np.clip(repl, lower, upper)
    out[s:e + 1] = repl
    return out, "aplicado_substituicao_referencia"


def find_gaps(isna: np.ndarray) -> List[Tuple[int, int]]:
    return [(s, e - s + 1) for s, e in contiguous_runs(isna, 1)]


def build_gap_rule(max_fillable_gap_min: int, short_gap_max_min: int, dead_pct: float, grid_step_min: int) -> pd.DataFrame:
    max_fillable = int(max(max_fillable_gap_min, short_gap_max_min + 1))
    b1 = int(max(1, short_gap_max_min))
    b2 = int(max(b1 + 1, np.floor(max_fillable / 3)))
    b3 = int(max(b2 + 1, np.floor(2 * max_fillable / 3)))
    rows = [
        {"categoria": 1, "min_min": 1, "max_min": b1, "faixa_min": f"1-{b1}", "n_porcoes_onda": 1, "criterio": "lacuna_curta_parametrica"},
        {"categoria": 2, "min_min": b1 + 1, "max_min": b2, "faixa_min": f"{b1 + 1}-{b2}", "n_porcoes_onda": 2, "criterio": "terco_1_da_maior_lacuna_preenchivel"},
        {"categoria": 3, "min_min": b2 + 1, "max_min": b3, "faixa_min": f"{b2 + 1}-{b3}", "n_porcoes_onda": 3, "criterio": "terco_2_da_maior_lacuna_preenchivel"},
        {"categoria": 4, "min_min": b3 + 1, "max_min": "", "faixa_min": f">={b3 + 1}", "n_porcoes_onda": 4, "criterio": "terco_3_ou_maior"},
    ]
    df = pd.DataFrame(rows)
    df["max_fillable_gap_min"] = int(max_fillable_gap_min)
    df["short_gap_max_min"] = int(short_gap_max_min)
    df["dead_pct"] = float(dead_pct)
    df["grid_step_min"] = int(grid_step_min)
    return df


def n_porcoes_from_rule(length_min: int, rule_df: pd.DataFrame) -> int:
    for _, r in rule_df.iterrows():
        lo = int(r["min_min"])
        hi_raw = r["max_min"]
        hi = int(hi_raw) if str(hi_raw).strip() != "" else 10**12
        if lo <= int(length_min) <= hi:
            return int(r["n_porcoes_onda"])
    return int(rule_df.iloc[-1]["n_porcoes_onda"])


def characterize_gaps(df: pd.DataFrame, *, prefix: str, dead_pct: float, short_gap_max_min: int, grid_step_min: int) -> dict:
    cols = [c for c in df.columns if c not in (METER_COL, TIME_COL)]
    profile_rows, all_gap_rows, dead_rows = [], [], []
    for m, g in df.groupby(METER_COL):
        g = g.sort_values(TIME_COL).reset_index(drop=True)
        times = pd.to_datetime(g[TIME_COL], utc=True, errors="coerce").to_numpy()
        for c in cols:
            isna = g[c].isna().to_numpy(dtype=bool)
            n_na = int(isna.sum())
            pct = 100.0 * n_na / max(len(isna), 1)
            dead = bool(pct >= dead_pct)
            gaps = find_gaps(isna)
            if dead:
                dead_rows.append({METER_COL: m, "variavel": c, "lado": lado(c), "pct_lacuna": round(pct, 4), "n_min_lacuna": int(n_na * grid_step_min)})
            for st, ln_points in gaps:
                all_gap_rows.append({
                    METER_COL: m,
                    "variavel": c,
                    "lado": lado(c),
                    "inicio": pd.Timestamp(times[st]) if st < len(times) else pd.NaT,
                    "fim": pd.Timestamp(times[min(st + ln_points - 1, len(times) - 1)]) if len(times) else pd.NaT,
                    "duracao_min": int(ln_points * grid_step_min),
                    "dead_channel": dead,
                })
            profile_rows.append({METER_COL: m, "variavel": c, "lado": lado(c), "pct_lacuna": round(pct, 4), "n_lacunas": len(gaps), "morto": dead})

    all_gaps = pd.DataFrame(all_gap_rows)
    if len(all_gaps):
        fillable = all_gaps[~all_gaps["dead_channel"]]
        max_fillable_gap_min = int(fillable["duracao_min"].max()) if len(fillable) else 0
    else:
        max_fillable_gap_min = 0
    rule_df = build_gap_rule(max_fillable_gap_min, short_gap_max_min, dead_pct, grid_step_min)
    rule_df.to_csv(AUDITS_DIR / f"{prefix}_gap_rule.csv", index=False)

    cat_n = {1: 0, 2: 0, 3: 0, 4: 0}
    cat_min = {1: 0, 2: 0, 3: 0, 4: 0}
    long_rows = []
    if len(all_gaps):
        for _, r in all_gaps.iterrows():
            if bool(r["dead_channel"]):
                continue
            npor = n_porcoes_from_rule(int(r["duracao_min"]), rule_df)
            cat_n[npor] += 1
            cat_min[npor] += int(r["duracao_min"])
            if npor >= 2:
                long_rows.append({
                    METER_COL: r[METER_COL],
                    "variavel": r["variavel"],
                    "lado": r["lado"],
                    "inicio": r["inicio"],
                    "fim": r["fim"],
                    "duracao_min": int(r["duracao_min"]),
                    "categoria": int(npor),
                    "n_porcoes": int(npor),
                })

    pd.DataFrame(profile_rows).to_csv(AUDITS_DIR / f"{prefix}_gap_profile_by_channel.csv", index=False)
    pd.DataFrame(long_rows, columns=[METER_COL, "variavel", "lado", "inicio", "fim", "duracao_min", "categoria", "n_porcoes"]).to_csv(AUDITS_DIR / f"{prefix}_long_gaps.csv", index=False)
    pd.DataFrame(dead_rows, columns=[METER_COL, "variavel", "lado", "pct_lacuna", "n_min_lacuna"]).to_csv(AUDITS_DIR / f"{prefix}_dead_channels.csv", index=False)

    dist = pd.DataFrame([
        {"categoria": k, "n_lacunas": int(cat_n[k]), "total_min": int(cat_min[k])}
        for k in (1, 2, 3, 4)
    ])
    dist.to_csv(AUDITS_DIR / f"{prefix}_gap_size_distribution.csv", index=False)
    return {"max_fillable_gap_min": max_fillable_gap_min, "cat_counts": cat_n, "cat_minutes": cat_min}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sanitises series by an envelope rule and replacement with a reference segment of the series itself.")
    p.add_argument("--input", default=str(DEFAULT_INPUT), help="Input CSV with gaps as NaN.")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Sanitised output CSV.")
    p.add_argument("--prefix", default="06", help="Prefix of the audits.")
    # Reference window anchored by TIMESTAMP ("where it is measured"): generic,
    # independent of the grid origin. The fixed indices remain as an optional override.
    p.add_argument("--ref-start-time", default="2020-10-28 15:59:00",
                   help="Start of the reference window as a UTC TIMESTAMP (where it is measured).")
    p.add_argument("--ref-end-time", default="2020-10-29 05:18:00",
                   help="End of the reference window as a UTC TIMESTAMP.")
    p.add_argument("--ref-start", type=int, default=None,
                   help="(Optional override) fixed start index; if omitted, uses --ref-start-time.")
    p.add_argument("--ref-end", type=int, default=None,
                   help="(Optional override) fixed end index; if omitted, uses --ref-end-time.")
    p.add_argument("--sd-k", type=float, default=1.0, help="Multiplier of the standard deviation of the reference segment.")
    p.add_argument("--min-ref-obs-frac", type=float, default=0.45, help="Minimum fraction of finite values in the reference segment of the series.")
    p.add_argument("--startup-scan-max-points", type=int, default=700, help="Initial region for grouping the start-up block.")
    p.add_argument("--startup-min-bad-points", type=int, default=30, help="Minimum number of points outside the envelope to form the initial block.")
    p.add_argument("--startup-min-span-points", type=int, default=60, help="Minimum extent of the initial block.")
    p.add_argument("--merge-block-gap-points", type=int, default=10, help="Closes small good intervals between violations.")
    p.add_argument("--clip-max-run-points", type=int, default=1, help="Runs up to this length are clipped, not replaced.")
    p.add_argument("--repair-padding-points", type=int, default=3, help="Margin added before/after replaced blocks. Does not affect clipped single points. Default: 3.")
    p.add_argument("--max-repair-passes", type=int, default=2, help="Maximum number of repair passes per series. Default: 2.")
    p.add_argument("--final-envelope-guard", action="store_true", default=True, help="Applies the final guard: any finite value still outside the envelope is clipped.")
    p.add_argument("--dead-pct", type=float, default=99.0, help="Gap percentage for a dead channel.")
    p.add_argument("--short-gap-max-min", type=int, default=60, help="Short-gap limit in the dynamic rule.")
    p.add_argument("--dry-run", action="store_true", help="Generates the maps, but does not apply the repairs to the output CSV.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    for d in (INTERIM_DIR, AUDITS_DIR, TABLES_DIR, FIGDATA_DIR, LOGS_DIR, MANIFESTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(LOGS_DIR / "06_sanitize_waveforms_reference_segment_repair.log")
    logger.info("=== 06b_defrag_envelope.py: defragmentation + envelope sanitisation ===")
    logger.info(f"Run timestamp: {datetime.now().isoformat()}")

    prefix = str(args.prefix)
    raw_input = Path(args.input)
    if not raw_input.exists():
        logger.error(f"Input missing: {raw_input}")
        sys.exit(1)
    # STEP 1: defragmentation (06_para_preencher -> 06A_desfragmentado)
    defrag_output = INTERIM_DIR / "06A_desfragmentado.csv"
    run_defrag(raw_input, defrag_output, prefix="06A", logger=logger)
    # STEP 2: envelope: reads the defragmented output and writes the final sanitised file
    input_path = defrag_output
    output_path = Path(args.output)

    df = pd.read_csv(input_path, parse_dates=[TIME_COL], low_memory=False)
    if METER_COL not in df.columns or TIME_COL not in df.columns:
        logger.error(f"Input must contain {METER_COL!r} and {TIME_COL!r}.")
        sys.exit(1)
    cols = [c for c in df.columns if c not in (METER_COL, TIME_COL)]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out = df.sort_values([METER_COL, TIME_COL]).reset_index(drop=True).copy()
    thresholds = []
    repairs = []
    before_na = int(out[cols].isna().sum().sum())

    # Reference window anchored by TIMESTAMP (generic: independent of the grid origin).
    # Per series, the index is resolved by temporal search. The fixed indices
    # (--ref-start/--ref-end) remain available as an optional override.
    def _to_utc(ts):
        t = pd.Timestamp(ts)
        return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    ref_start_time = _to_utc(args.ref_start_time)
    ref_end_time = _to_utc(args.ref_end_time)
    use_index_override = (args.ref_start is not None and args.ref_end is not None)
    logger.info(
        f"Reference window by timestamp: {ref_start_time} -> {ref_end_time}"
        + (f" | index override {args.ref_start}..{args.ref_end}" if use_index_override else "")
    )

    for m, idx_global in out.groupby(METER_COL).groups.items():
        idx_global = np.asarray(list(idx_global), dtype=int)
        g = out.loc[idx_global].sort_values(TIME_COL).reset_index(drop=True)
        global_order = out.loc[idx_global].sort_values(TIME_COL).index.to_numpy()
        times = pd.to_datetime(g[TIME_COL], utc=True, errors="coerce").to_numpy()

        # Resolves the window of this series (same real segment on any grid).
        if use_index_override:
            ref_start_idx, ref_end_idx = int(args.ref_start), int(args.ref_end)
        else:
            tidx = pd.DatetimeIndex(pd.to_datetime(g[TIME_COL], utc=True, errors="coerce"))
            ref_start_idx = int(tidx.searchsorted(ref_start_time, side="left"))
            ref_end_idx = int(tidx.searchsorted(ref_end_time, side="right")) - 1

        for c in cols:
            v = g[c].to_numpy(dtype=float)
            thr = build_threshold(v, ref_start_idx, ref_end_idx, float(args.sd_k), float(args.min_ref_obs_frac))
            if thr is None:
                thresholds.append({METER_COL: m, "variavel": c, "fase": phase_of(c), "familia": family_of(c), "status": "sem_referencia_suficiente"})
                continue
            thresholds.append({
                METER_COL: m,
                "variavel": c,
                "fase": phase_of(c),
                "familia": family_of(c),
                "status": "ok",
                "ref_start": ref_start_idx,
                "ref_end": ref_end_idx,
                "ref_n_total": thr["ref_n_total"],
                "ref_n_finite": thr["ref_n_finite"],
                "ref_min": thr["ref_min"],
                "ref_max": thr["ref_max"],
                "ref_mean": thr["ref_mean"],
                "ref_std": thr["ref_std"],
                "limite_inferior": thr["lower"],
                "limite_superior": thr["upper"],
                "ref_amplitude": thr["ref_amplitude"],
                "limit_amplitude_ref_plus_sd": thr["limit_amplitude_ref_plus_sd"],
            })

            new_v = v.copy()
            initial_violation = np.isfinite(new_v) & ((new_v < thr["lower"]) | (new_v > thr["upper"]))
            if not initial_violation.any():
                continue

            any_repair = False
            for repair_pass in range(1, int(args.max_repair_passes) + 1):
                violation = np.isfinite(new_v) & ((new_v < thr["lower"]) | (new_v > thr["upper"]))
                segs = build_repair_segments(
                    violation,
                    startup_scan_max_points=int(args.startup_scan_max_points),
                    startup_min_bad_points=int(args.startup_min_bad_points),
                    startup_min_span_points=int(args.startup_min_span_points),
                    merge_block_gap_points=int(args.merge_block_gap_points),
                    clip_max_run_points=int(args.clip_max_run_points),
                )
                if not segs:
                    break

                # Expands only replacement blocks, to capture edges that are still bad
                # (e.g. drops visually marked during the review). Single points are still
                # only clipped at the bound.
                segs = [
                    expand_substitution_segment(seg, len(new_v), int(args.repair_padding_points))
                    for seg in segs
                ]

                for seg in segs:
                    s, e = int(seg["x_inicio"]), int(seg["x_fim"])
                    original = new_v[s:e + 1]
                    n_bad = int(violation[s:e + 1].sum())
                    n_finite = int(np.isfinite(original).sum())
                    if not args.dry_run:
                        tmp, status_apply = apply_segment_action(new_v, seg, thr)
                        new_v = tmp
                    else:
                        status_apply = "dry_run"
                    any_repair = True
                    repairs.append({
                        METER_COL: m,
                        "variavel": c,
                        "fase": phase_of(c),
                        "familia": family_of(c),
                        "x_inicio": s,
                        "x_fim": e,
                        "n_pontos": e - s + 1,
                        "n_pontos_finitos": n_finite,
                        "n_pontos_fora_envelope": n_bad,
                        "t_inicio": pd.Timestamp(times[s]) if s < len(times) else pd.NaT,
                        "t_fim": pd.Timestamp(times[e]) if e < len(times) else pd.NaT,
                        "valor_min_original": float(np.nanmin(original)) if np.isfinite(original).any() else np.nan,
                        "valor_max_original": float(np.nanmax(original)) if np.isfinite(original).any() else np.nan,
                        "limite_inferior": thr["lower"],
                        "limite_superior": thr["upper"],
                        "ref_start": ref_start_idx,
                        "ref_end": ref_end_idx,
                        "repair_pass": int(repair_pass),
                        "acao_sintetica": seg["acao_sintetica"],
                        "motivo": seg["motivo"],
                        "status_aplicacao": status_apply,
                    })

                # If the current pass already removed all violations, stop.
                residual_violation = np.isfinite(new_v) & ((new_v < thr["lower"]) | (new_v > thr["upper"]))
                if not residual_violation.any():
                    break

            # Final guard: do not leave any value outside the envelope.
            # This step is logged separately and affects only finite points still in violation.
            final_violation = np.isfinite(new_v) & ((new_v < thr["lower"]) | (new_v > thr["upper"]))
            if final_violation.any() and bool(args.final_envelope_guard):
                for s, e in contiguous_runs(final_violation, 1):
                    original = new_v[s:e + 1]
                    if not args.dry_run:
                        part = original.copy()
                        fin = np.isfinite(part)
                        part[fin] = np.clip(part[fin], float(thr["lower"]), float(thr["upper"]))
                        new_v[s:e + 1] = part
                        status_apply = "aplicado_clip_guarda_final"
                    else:
                        status_apply = "dry_run"
                    any_repair = True
                    repairs.append({
                        METER_COL: m,
                        "variavel": c,
                        "fase": phase_of(c),
                        "familia": family_of(c),
                        "x_inicio": int(s),
                        "x_fim": int(e),
                        "n_pontos": int(e - s + 1),
                        "n_pontos_finitos": int(np.isfinite(original).sum()),
                        "n_pontos_fora_envelope": int(final_violation[s:e + 1].sum()),
                        "t_inicio": pd.Timestamp(times[s]) if s < len(times) else pd.NaT,
                        "t_fim": pd.Timestamp(times[e]) if e < len(times) else pd.NaT,
                        "valor_min_original": float(np.nanmin(original)) if np.isfinite(original).any() else np.nan,
                        "valor_max_original": float(np.nanmax(original)) if np.isfinite(original).any() else np.nan,
                        "limite_inferior": thr["lower"],
                        "limite_superior": thr["upper"],
                        "ref_start": ref_start_idx,
                        "ref_end": ref_end_idx,
                        "repair_pass": int(args.max_repair_passes) + 1,
                        "acao_sintetica": "clipar_no_limite_do_envelope",
                        "motivo": "guarda_final_envelope",
                        "status_aplicacao": status_apply,
                    })

            if any_repair and not args.dry_run:
                out.loc[global_order, c] = new_v

    thresholds_df = pd.DataFrame(thresholds)
    repairs_df = pd.DataFrame(repairs)
    thresholds_df.to_csv(AUDITS_DIR / f"{prefix}_reference_amplitude_thresholds.csv", index=False)
    repairs_df.to_csv(AUDITS_DIR / f"{prefix}_reference_amplitude_repair_map.csv", index=False)
    if len(repairs_df):
        summary = (
            repairs_df.groupby(["acao_sintetica", "motivo", "familia", "fase"], dropna=False)
            .agg(n_blocos=("n_pontos", "count"), n_pontos=("n_pontos", "sum"), n_fora_envelope=("n_pontos_fora_envelope", "sum"), n_series=("variavel", "nunique"))
            .reset_index()
        )
    else:
        summary = pd.DataFrame(columns=["acao_sintetica", "motivo", "familia", "fase", "n_blocos", "n_pontos", "n_fora_envelope", "n_series"])
    summary.to_csv(AUDITS_DIR / f"{prefix}_reference_amplitude_action_summary.csv", index=False)

    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)
        shutil.copyfile(output_path, FIGDATA_DIR / output_path.name)
    after_na = int(out[cols].isna().sum().sum())
    grid_step_min = infer_grid_step_min(out)
    gap_info = characterize_gaps(out, prefix=prefix, dead_pct=float(args.dead_pct), short_gap_max_min=int(args.short_gap_max_min), grid_step_min=grid_step_min)

    manifest = {
        "script": "06_sanitize_waveforms_reference_segment_repair_v2.py",
        "run_timestamp": datetime.now().isoformat(),
        "input": str(input_path),
        "output": str(output_path),
        "dry_run": bool(args.dry_run),
        "reference_window": {
            "ref_start_time": str(ref_start_time),
            "ref_end_time": str(ref_end_time),
            "index_override": ({"ref_start": int(args.ref_start), "ref_end": int(args.ref_end)}
                               if use_index_override else None),
        },
        "parameters": {
            "sd_k": float(args.sd_k),
            "min_ref_obs_frac": float(args.min_ref_obs_frac),
            "startup_scan_max_points": int(args.startup_scan_max_points),
            "startup_min_bad_points": int(args.startup_min_bad_points),
            "startup_min_span_points": int(args.startup_min_span_points),
            "merge_block_gap_points": int(args.merge_block_gap_points),
            "clip_max_run_points": int(args.clip_max_run_points),
            "repair_padding_points": int(args.repair_padding_points),
            "max_repair_passes": int(args.max_repair_passes),
            "final_envelope_guard": bool(args.final_envelope_guard),
            "dead_pct": float(args.dead_pct),
            "short_gap_max_min": int(args.short_gap_max_min),
        },
        "results": {
            "n_threshold_rows": int(len(thresholds_df)),
            "n_repair_blocks": int(len(repairs_df)),
            "n_cells_na_before": before_na,
            "n_cells_na_after": after_na,
            "max_fillable_gap_min_after": int(gap_info["max_fillable_gap_min"]),
            "grid_step_min": int(grid_step_min),
        },
        "outputs": {
            "thresholds": str(AUDITS_DIR / f"{prefix}_reference_amplitude_thresholds.csv"),
            "repair_map": str(AUDITS_DIR / f"{prefix}_reference_amplitude_repair_map.csv"),
            "action_summary": str(AUDITS_DIR / f"{prefix}_reference_amplitude_action_summary.csv"),
            "gap_rule": str(AUDITS_DIR / f"{prefix}_gap_rule.csv"),
            "long_gaps": str(AUDITS_DIR / f"{prefix}_long_gaps.csv"),
            "dead_channels": str(AUDITS_DIR / f"{prefix}_dead_channels.csv"),
        },
        "status": "success",
    }
    with open(MANIFESTS_DIR / f"{prefix}_sanitize_waveforms_reference_segment_repair_params.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Thresholds: {len(thresholds_df)} series")
    logger.info(f"Repairs mapped: {len(repairs_df)} blocks")
    logger.info(f"Output: {output_path if not args.dry_run else 'dry-run without writing the output'}")
    logger.info("=== completed ===")


if __name__ == "__main__":
    main()
