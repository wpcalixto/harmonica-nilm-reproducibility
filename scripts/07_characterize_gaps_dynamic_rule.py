#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 07_characterize_gaps_dynamic_rule.py

File type: Pipeline script (stage 07: gap characterisation and dynamic filling rule)

Purpose:
    Gap characterisation and dynamic computation of the filling rule.

    Objective
    ---------
    Read the consolidated base prepared for filling, find contiguous NaN blocks per
    (meter, variable), identify dead channels and compute dynamically the GAP_RULE from the
    largest fillable gap observed in the non-dead channels.

    The script does NOT fill gaps. It sizes the gaps and produces the inputs required by the
    subsequent filling stage (script 08).

    Default input (output of script 06)
    -----------------------------------
        data/interim/06_para_preencher.csv

    Default outputs, prefix configurable with --output-prefix
    ---------------------------------------------------------
        audits/07_gap_rule.csv
        audits/07_gap_size_distribution.csv
        audits/07_gap_profile_by_channel.csv
        audits/07_gap_profile_summary.csv
        audits/07_long_gaps.csv
        audits/07_dead_channels.csv
        logs/07_characterize_gaps.log
        manifests/07_characterize_gaps_params.json
        (with --run-heldout: audits/07_fill_validation, figdata/07_fill_validation_examples.csv)

    Dynamic rule
    ------------
    Category 1 keeps the operational interpretation of a short gap:
        1 .. short_gap_max_min  -> 1 wave portion

    Categories 2, 3 and 4 are computed from the largest fillable gap, i.e. the largest gap
    found in non-dead channels. By default:
        base = floor(max_fillable_gap_min / 3)
        cat 2: short_gap_max_min + 1 .. base
        cat 3: base + 1 .. 2*base
        cat 4: 2*base + 1 .. open

    With max_fillable_gap_min = 5282 and short_gap_max_min = 60, the rule yields:
        1..60, 61..1760, 1761..3520, >=3521.

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
import importlib.util
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

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
LOGS_DIR = PROJECT_ROOT / "logs"
MANIFESTS_DIR = PROJECT_ROOT / "manifests"

DEFAULT_INPUT_CSV = INTERIM_DIR / "06_para_preencher.csv"
TIME_COL, METER_COL = "time", "meter_id"

# Default parameters. The filling rule is NO LONGER fixed; it is computed in main().
DEAD_PCT = 99.0
SHORT_GAP_MAX_MIN = 60
OUTPUT_PREFIX = "07"

# channel side (energy has a sentinel; voltage does not); labelling only
RE_ENERGY = re.compile(
    r"^(i_[abc]n|p_[abc]|q_[abc]|s_[abc]|cos_[abc]|thdi_[abc]|hrm_i_[abc]n_)"
)


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("07_characterize_gaps")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def save_table(df, name, caption, label):
    """Project convention: every 'table' = one .csv (audits/) + one .tex (tables/)."""
    AUDITS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(AUDITS_DIR / f"{name}.csv", index=False)
    cols = list(df.columns)
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%")
    lines = [
        r"\begin{table}[!h]",
        r"\centering",
        r"\small",
        r"\caption{%s}" % caption,
        r"\label{%s}" % label,
        r"\begin{tabular}{l" + "r" * (len(cols) - 1) + "}",
        r"\hline",
        " & ".join(r"\textbf{%s}" % esc(c) for c in cols) + r" \\",
        r"\hline",
    ]
    for _, r in df.iterrows():
        lines.append(" & ".join(esc(r[c]) for c in cols) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    (TABLES_DIR / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")


def find_gaps(isna: np.ndarray):
    """Contiguous NaN blocks. Returns a list of (start_idx, length_min)."""
    if not isna.any():
        return []
    d = np.diff(np.r_[0, isna.astype(np.int8), 0])
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), (ends - starts).tolist()))


def n_porcoes(duration_min: int, gap_rule: list[tuple[int, int | None, int]]) -> int:
    """Classifies the gap duration using the rule computed by this script."""
    duration_min = int(duration_min)
    for lo, hi, npor in gap_rule:
        if duration_min >= int(lo) and (hi is None or duration_min <= int(hi)):
            return int(npor)
    return int(gap_rule[-1][2])


def lado(col: str) -> str:
    return "energia" if RE_ENERGY.match(col) else "tensao"


# ════════════════════════════════════════════════════════════════════════════
#  HELD-OUT PROOF of the filling method (PART of the characterisation; does NOT alter
#  the outputs consumed by script 08). Uses the SAME series as script 08 (06_para_preencher):
#  hides a REAL window (the truth), fills it with 3 methods and compares only on the
#  true points. The 4 dead channels are excluded (they have no truth).
# ════════════════════════════════════════════════════════════════════════════
VAL_WSIZES = [60, 180, 360, 720]  # 1h, 3h, 6h, 12h
VAL_N = 50  # windows per size
VAL_SEED = 7
VAL_MIN_OBS_FRAC = 0.5  # minimum number of real points in the window (truth)
MLP_HIDDEN = (32, 16)
MLP_MAX_ITER = 300


def _load_fill04():
    cands = sorted(SCRIPT_DIR.glob("0*_fill_gaps*.py"))
    if not cands:
        raise FileNotFoundError(f"No *_fill_gaps*.py found in {SCRIPT_DIR}")
    spec = importlib.util.spec_from_file_location("fill04", cands[-1])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fill_linear(v, a, W):
    return pd.Series(v).interpolate("linear", limit_direction="both").values[a : a + W]


def _fill_mlp(v, a, W):
    from sklearn.neural_network import MLPRegressor

    n = len(v)
    lo, hi = max(0, a - W), min(n, a + 2 * W)
    pos = [
        i for i in (list(range(lo, a)) + list(range(a + W, hi))) if np.isfinite(v[i])
    ]
    if len(pos) < 10:
        return _fill_linear(v, a, W)
    X = np.array(pos, float)
    mu, sd = X.mean(), (X.std() or 1.0)
    try:
        mlp = MLPRegressor(
            hidden_layer_sizes=MLP_HIDDEN,
            max_iter=MLP_MAX_ITER,
            random_state=VAL_SEED,
            early_stopping=True,
            n_iter_no_change=15,
        )
        mlp.fit(((X - mu) / sd).reshape(-1, 1), v[pos].astype(float))
        return mlp.predict(
            ((np.arange(a, a + W, dtype=float) - mu) / sd).reshape(-1, 1)
        )
    except Exception:
        return _fill_linear(v, a, W)


def _val_stats(filled, truth, freqfn):
    m = np.isfinite(truth)
    f, t = np.asarray(filled)[m], truth[m]
    if len(t) < 5:
        return None
    return {
        "amp": (np.std(f) + 1e-12) / (np.std(t) + 1e-12),
        "freq": (freqfn(f) + 1e-12) / (freqfn(t) + 1e-12),
        "energy": (np.sum(np.abs(f)) + 1e-12) / (np.sum(np.abs(t)) + 1e-12),
        "mae": float(np.mean(np.abs(f - t))),
    }


def run_validation(df, prof, logger):
    """Held-out proof: hides real windows, fills them (fractal/linear/MLP) and measures."""
    try:
        from scipy.stats import ks_2samp, wilcoxon
    except Exception:
        logger.info("scipy missing; held-out proof skipped.")
        return
    try:
        fill04 = _load_fill04()
    except Exception as exc:
        logger.info(f"Filling script unavailable; held-out proof skipped: {exc}")
        return
    # Script 08 populates GAP_RULE only in its main(); when reusing analog_fill as a module,
    # we load the rule here (06_gap_rule.csv; falls back to 07_gap_rule.csv if absent).
    try:
        rc = fill04.DEFAULT_GAP_RULE_CSV
        if not rc.exists():
            rc = rc.with_name("07_gap_rule.csv")
        fill04.GAP_RULE = fill04.load_gap_rule(rc)
    except Exception as exc:
        logger.info(f"Gap rule unavailable for the held-out proof; skipped: {exc}")
        return
    rng = np.random.default_rng(VAL_SEED)
    alive = prof[~prof["morto"]][[METER_COL, "variavel"]].values.tolist()
    cache = {}

    def get_series(m, c):
        key = (m, c)
        if key not in cache:
            g = df[df[METER_COL] == m].sort_values(TIME_COL)
            cache[key] = g[c].values.astype(float)
        return cache[key]

    methods = ["fractal", "linear", "mlp"]
    rows, examples = [], {}
    logger.info(
        f"Held-out proof: {VAL_N} windows x {VAL_WSIZES} min x {len(methods)} methods..."
    )
    for W in VAL_WSIZES:
        per_win, tries = [], 0
        while len(per_win) < VAL_N and tries < VAL_N * 80:
            tries += 1
            m, c = alive[int(rng.integers(len(alive)))]
            v = get_series(m, c)
            n = len(v)
            if n < 3 * W + 2:
                continue
            a = int(rng.integers(W, n - 2 * W))
            win = v[a : a + W]
            if np.isfinite(win).sum() < VAL_MIN_OBS_FRAC * W:
                continue
            if not (np.isfinite(v[a - 1]) and np.isfinite(v[a + W])):
                continue
            truth = win.copy()
            vt = v.copy()
            vt[a : a + W] = np.nan
            fills = {
                "fractal": fill04.analog_fill(vt, rng)[0][a : a + W],
                "linear": _fill_linear(vt, a, W),
                "mlp": _fill_mlp(vt, a, W),
            }
            sts = {
                meth: _val_stats(fills[meth], truth, fill04.freq) for meth in methods
            }
            if any(sts[meth] is None for meth in methods):
                continue
            fin = np.isfinite(truth)
            rec = {}
            for meth in methods:
                for k in ("amp", "freq", "energy", "mae"):
                    rec[f"{k}_{meth}"] = sts[meth][k]
                rec[f"ks_{meth}"] = ks_2samp(
                    np.asarray(fills[meth])[fin], truth[fin]
                ).pvalue
            per_win.append(rec)
            if W not in examples:
                examples[W] = (m, c, truth, fills)
        pw = pd.DataFrame(per_win)
        for meth in methods:
            row = {"janela_h": W // 60, "janela_min": W, "metodo": meth, "n": len(pw)}
            for k, lbl in (
                ("amp", "razao_amplitude"),
                ("freq", "razao_frequencia"),
                ("energy", "razao_energia"),
                ("ks", "ks_pvalor"),
                ("mae", "mae"),
            ):
                row[lbl] = round(float(pw[f"{k}_{meth}"].median()), 4)
            rows.append(row)
        for base in ("linear", "mlp"):
            try:
                p = wilcoxon(
                    (pw["amp_fractal"] - 1).abs(), (pw[f"amp_{base}"] - 1).abs()
                ).pvalue
                better = (pw["amp_fractal"] - 1).abs().median() < (
                    pw[f"amp_{base}"] - 1
                ).abs().median()
                logger.info(
                    f"  W={W}min . Wilcoxon |amp-1| fractal vs {base}: p={p:.3g} "
                    f"({'fractal better' if better else 'baseline better'})"
                )
            except Exception:
                pass

    val = pd.DataFrame(rows)
    save_table(
        val,
        "07_fill_validation",
        "Held-out proof of the filling by window size (1h/3h/6h/12h) and method "
        "(fractal/linear/MLP): razao de amplitude, frequencia e energia (alvo=1), KS "
        "(p-valor; alto=indistinguivel da verdade) e MAE ponto-a-ponto (referencia).",
        "tab:fill_val",
    )
    logger.info(f"Held-out proof -> 07_fill_validation (.csv/.tex): {len(val)} rows")
    # Figures: NOT generated here. By project rule, only the figure scripts save
    # images. Here only the DATA of the held-out examples (per window) are stored, which
    # the figure generator (script 23) plots as PDF.
    figdata_dir = PROJECT_ROOT / "figdata"
    figdata_dir.mkdir(parents=True, exist_ok=True)
    ex_rows = []
    for W, (m, c, truth, fills) in examples.items():
        for t in range(len(truth)):
            ex_rows.append({
                "janela_h": W // 60, "meter_id": m, "variavel": c, "t": t,
                "verdade": truth[t], "fractal": fills["fractal"][t],
                "linear": fills["linear"][t], "mlp": fills["mlp"][t],
            })
    pd.DataFrame(ex_rows, columns=["janela_h", "meter_id", "variavel", "t",
                                   "verdade", "fractal", "linear", "mlp"]).to_csv(
        figdata_dir / "07_fill_validation_examples.csv", index=False)
    logger.info("Held-out examples (data for the figure) -> figdata/07_fill_validation_examples.csv")



def parse_bool(text_value: str | None) -> bool:
    if text_value is None:
        return False
    return str(text_value).strip().lower() in {"1", "true", "yes", "sim", "y", "s"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Characterises gaps and computes dynamically the GAP_RULE for the filling."
    )
    p.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_CSV),
        help="Consolidated CSV with gaps as NaN. Default: data/interim/06_para_preencher.csv.",
    )
    p.add_argument(
        "--output-prefix",
        default=OUTPUT_PREFIX,
        help="Prefix of the outputs in audits/tables/logs/manifests. Default: 07.",
    )
    p.add_argument(
        "--dead-pct",
        type=float,
        default=DEAD_PCT,
        help="Gap percentage to classify a dead channel. Default: 99.0.",
    )
    p.add_argument(
        "--short-gap-max-min",
        type=int,
        default=SHORT_GAP_MAX_MIN,
        help="Upper limit of the short category, in minutes. Default: 60.",
    )
    p.add_argument(
        "--run-heldout",
        action="store_true",
        help="Runs the held-out proof of the filling. Not executed by default.",
    )
    return p


def infer_grid_step_minutes(df: pd.DataFrame) -> float:
    """Infers the mean duration of the time step per meter, in minutes."""
    diffs = []
    for _, g in df[[METER_COL, TIME_COL]].dropna().groupby(METER_COL):
        t = pd.to_datetime(g[TIME_COL], errors="coerce", utc=True).sort_values()
        if len(t) < 2:
            continue
        dt = t.diff().dropna().dt.total_seconds().to_numpy(dtype=float)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if len(dt):
            diffs.extend(dt.tolist())
    if not diffs:
        return 1.0
    return float(np.nanmedian(diffs) / 60.0)


def points_to_minutes(n_points: int, grid_step_min: float) -> int:
    """Converts a number of consecutive points into an approximate integer duration, in minutes."""
    return int(round(float(n_points) * float(grid_step_min)))


def calculate_gap_rule(max_fillable_gap_min: int,
                       short_gap_max_min: int = SHORT_GAP_MAX_MIN) -> list[tuple[int, int | None, int]]:
    """Computes the category rule from the largest fillable gap.

    The rule preserves the short category up to `short_gap_max_min` minutes. The other
    boundaries are derived from `floor(max_fillable_gap_min / 3)`, reproducing the
    reference rule when the largest fillable gap is 5282 min:
    1..60, 61..1760, 1761..3520, >=3521.
    """
    short_gap_max_min = int(max(1, short_gap_max_min))
    max_fillable_gap_min = int(max(1, max_fillable_gap_min))

    base = int(max_fillable_gap_min // 3)
    base = max(base, short_gap_max_min)
    b1 = short_gap_max_min
    b2 = max(base, b1)
    b3 = max(2 * base, b2)

    return [
        (1, b1, 1),
        (b1 + 1, b2, 2),
        (b2 + 1, b3, 3),
        (b3 + 1, None, 4),
    ]


def gap_rule_to_dataframe(gap_rule: list[tuple[int, int | None, int]],
                          *,
                          max_fillable_gap_min: int,
                          short_gap_max_min: int,
                          dead_pct: float,
                          grid_step_min: float) -> pd.DataFrame:
    rows = []
    for categoria, (lo, hi, npor) in enumerate(gap_rule, start=1):
        rows.append({
            "categoria": categoria,
            "min_min": int(lo),
            "max_min": ("" if hi is None else int(hi)),
            "faixa_min": f">={lo}" if hi is None else f"{lo}-{hi}",
            "n_porcoes_onda": int(npor),
            "criterio": (
                "lacuna_curta_parametrica" if categoria == 1
                else "tercos_da_maior_lacuna_preenchivel"
            ),
            "max_fillable_gap_min": int(max_fillable_gap_min),
            "short_gap_max_min": int(short_gap_max_min),
            "dead_pct": float(dead_pct),
            "grid_step_min": float(grid_step_min),
        })
    return pd.DataFrame(rows)


def collect_gap_inventory(df: pd.DataFrame,
                          cols: list[str],
                          *,
                          dead_pct: float,
                          grid_step_min: float):
    """First pass: collects the gaps and identifies dead channels without using GAP_RULE."""
    channel_rows = []
    gap_rows = []
    dead_rows = []
    fillable_lengths = []

    for m, g in df.groupby(METER_COL):
        g = g.sort_values(TIME_COL).reset_index(drop=True)
        times = g[TIME_COL].values
        for c in cols:
            isna = g[c].isna().values
            n_na = int(isna.sum())
            pct = 100.0 * n_na / max(len(isna), 1)
            dead = bool(pct >= dead_pct)
            gaps = find_gaps(isna)

            channel_rows.append({
                METER_COL: m,
                "variavel": c,
                "lado": lado(c),
                "pct_lacuna": round(pct, 2),
                "n_lacunas": int(len(gaps)),
                "morto": dead,
                "n_pontos_lacuna": n_na,
                "duracao_lacuna_min": points_to_minutes(n_na, grid_step_min),
            })

            if dead:
                dead_rows.append({
                    METER_COL: m,
                    "variavel": c,
                    "lado": lado(c),
                    "pct_lacuna": round(pct, 2),
                    "n_pontos_lacuna": n_na,
                    "n_min_lacuna": points_to_minutes(n_na, grid_step_min),
                })

            for st, ln_points in gaps:
                duration_min = points_to_minutes(ln_points, grid_step_min)
                row = {
                    METER_COL: m,
                    "variavel": c,
                    "lado": lado(c),
                    "inicio": pd.Timestamp(times[st]),
                    "fim": pd.Timestamp(times[min(st + ln_points - 1, len(times) - 1)]),
                    "x_inicio": int(st),
                    "x_fim": int(st + ln_points - 1),
                    "n_pontos": int(ln_points),
                    "duracao_min": int(duration_min),
                    "morto": dead,
                }
                gap_rows.append(row)
                if not dead:
                    fillable_lengths.append(int(duration_min))

    return channel_rows, gap_rows, dead_rows, fillable_lengths


def main():
    args = build_parser().parse_args()
    input_csv = Path(args.input)
    output_prefix = str(args.output_prefix).strip() or OUTPUT_PREFIX
    dead_pct = float(args.dead_pct)
    short_gap_max_min = int(args.short_gap_max_min)

    for d in (AUDITS_DIR, TABLES_DIR, LOGS_DIR, MANIFESTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(LOGS_DIR / f"{output_prefix}_characterize_gaps.log")
    logger.info(f"=== {output_prefix}_characterize_gaps.py: gap characterisation ===")
    logger.info(f"Run timestamp: {datetime.now().isoformat()}")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Input: {input_csv}")

    if not input_csv.exists():
        logger.error(f"Input not found: {input_csv}")
        sys.exit(1)
    if not (0.0 < dead_pct <= 100.0):
        logger.error("--dead-pct must be in (0, 100].")
        sys.exit(1)
    if short_gap_max_min < 1:
        logger.error("--short-gap-max-min must be >= 1.")
        sys.exit(1)

    df = pd.read_csv(input_csv, parse_dates=[TIME_COL])
    if TIME_COL not in df.columns or METER_COL not in df.columns:
        logger.error(f"The file must contain the columns {TIME_COL!r} and {METER_COL!r}.")
        sys.exit(1)

    cols = [c for c in df.columns if c not in (METER_COL, TIME_COL)]
    grid_step_min = infer_grid_step_minutes(df)
    logger.info(
        f"Read: {input_csv.name} | {len(df)} rows x {len(cols)} variables | "
        f"{df[METER_COL].nunique()} meters | step~{grid_step_min:.6g} min"
    )

    # First pass: locates the gaps and identifies dead channels without using a fixed rule.
    channel_rows, gap_rows, dead_rows, fillable_lengths = collect_gap_inventory(
        df, cols, dead_pct=dead_pct, grid_step_min=grid_step_min
    )
    max_fillable_gap_min = int(max(fillable_lengths)) if fillable_lengths else short_gap_max_min
    gap_rule = calculate_gap_rule(
        max_fillable_gap_min=max_fillable_gap_min,
        short_gap_max_min=short_gap_max_min,
    )
    gap_rule_df = gap_rule_to_dataframe(
        gap_rule,
        max_fillable_gap_min=max_fillable_gap_min,
        short_gap_max_min=short_gap_max_min,
        dead_pct=dead_pct,
        grid_step_min=grid_step_min,
    )
    gap_rule_df.to_csv(AUDITS_DIR / f"{output_prefix}_gap_rule.csv", index=False)
    logger.info(
        f"Dynamic gap rule -> {output_prefix}_gap_rule.csv | "
        f"max_fillable_gap_min={max_fillable_gap_min}"
    )

    # Second logical pass: applies the computed rule to the collected gaps.
    cat_n = {1: 0, 2: 0, 3: 0, 4: 0}
    cat_min = {1: 0, 2: 0, 3: 0, 4: 0}
    per_channel_cat = {}
    long_rows = []
    for gr in gap_rows:
        if gr["morto"]:
            continue
        npor = n_porcoes(int(gr["duracao_min"]), gap_rule)
        gr["categoria"] = int(npor)
        gr["n_porcoes"] = int(npor)
        cat_n[npor] += 1
        cat_min[npor] += int(gr["duracao_min"])
        key = (gr[METER_COL], gr["variavel"])
        if key not in per_channel_cat:
            per_channel_cat[key] = {1: 0, 2: 0, 3: 0, 4: 0}
        per_channel_cat[key][npor] += 1
        if npor >= 2:
            long_rows.append({
                METER_COL: gr[METER_COL],
                "variavel": gr["variavel"],
                "lado": gr["lado"],
                "inicio": gr["inicio"],
                "fim": gr["fim"],
                "x_inicio": gr["x_inicio"],
                "x_fim": gr["x_fim"],
                "n_pontos": gr["n_pontos"],
                "duracao_min": gr["duracao_min"],
                "categoria": int(npor),
                "n_porcoes": int(npor),
            })

    profile_rows = []
    for r in channel_rows:
        key = (r[METER_COL], r["variavel"])
        cc = per_channel_cat.get(key, {1: 0, 2: 0, 3: 0, 4: 0})
        rr = dict(r)
        rr.update({"cat1": cc[1], "cat2": cc[2], "cat3": cc[3], "cat4": cc[4]})
        profile_rows.append(rr)

    total_n = sum(cat_n.values())
    dist_rows = []
    for _, row in gap_rule_df.iterrows():
        k = int(row["categoria"])
        dist_rows.append({
            "categoria": k,
            "faixa_min": row["faixa_min"],
            "n_porcoes_onda": int(row["n_porcoes_onda"]),
            "n_lacunas": int(cat_n[k]),
            "pct_lacunas": round(100.0 * cat_n[k] / max(total_n, 1), 2),
            "total_min": int(cat_min[k]),
        })
    dist = pd.DataFrame(dist_rows)
    save_table(
        dist,
        f"{output_prefix}_gap_size_distribution",
        "Distribution of the gaps by size category and number of wave portions "
        "of the filling (non-dead channels). Rule computed dynamically by script 07.",
        "tab:gap_dist",
    )

    prof = pd.DataFrame(profile_rows)
    prof.to_csv(AUDITS_DIR / f"{output_prefix}_gap_profile_by_channel.csv", index=False)

    summ = (
        prof.groupby(METER_COL)
        .agg(
            pct_lacuna_media=("pct_lacuna", "mean"),
            n_canais=("variavel", "count"),
            n_mortos=("morto", "sum"),
            lacunas_cat2=("cat2", "sum"),
            lacunas_cat3=("cat3", "sum"),
            lacunas_cat4=("cat4", "sum"),
        )
        .reset_index()
    )
    summ["pct_lacuna_media"] = summ["pct_lacuna_media"].round(1)
    summ["lacunas_longas"] = summ["lacunas_cat2"] + summ["lacunas_cat3"] + summ["lacunas_cat4"]
    summ = summ.drop(columns=["lacunas_cat2", "lacunas_cat3", "lacunas_cat4"])
    save_table(
        summ,
        f"{output_prefix}_gap_profile_summary",
        "Gap profile per meter: mean gap percentage, dead channels "
        "e lacunas longas (categoria >=2).",
        "tab:gap_summary",
    )

    long_df = pd.DataFrame(long_rows) if long_rows else pd.DataFrame(
        columns=[
            METER_COL, "variavel", "lado", "inicio", "fim", "x_inicio", "x_fim",
            "n_pontos", "duracao_min", "categoria", "n_porcoes",
        ]
    )
    long_df.to_csv(AUDITS_DIR / f"{output_prefix}_long_gaps.csv", index=False)

    dead_df = pd.DataFrame(dead_rows) if dead_rows else pd.DataFrame(
        columns=[METER_COL, "variavel", "lado", "pct_lacuna", "n_pontos_lacuna", "n_min_lacuna"]
    )
    save_table(
        dead_df,
        f"{output_prefix}_dead_channels",
        "Dead channels: not filled because there are not enough real points from which to extract the wave.",
        "tab:dead_channels",
    )

    logger.info("Dynamic rule computed:")
    for _, r in gap_rule_df.iterrows():
        logger.info(
            f"  cat{int(r['categoria'])}: {r['faixa_min']} min . "
            f"{int(r['n_porcoes_onda'])} wave portion(s)"
        )
    logger.info("Distribution (non-dead channels):")
    for k in (1, 2, 3, 4):
        faixa = dist.loc[dist["categoria"] == k, "faixa_min"].iloc[0]
        logger.info(f"  cat{k} ({faixa} min): {cat_n[k]} gaps . {cat_min[k]} min")

    n_dead = len(dead_df)
    logger.info(
        f"Dead channels: {n_dead} "
        f"(meters: {sorted(dead_df[METER_COL].unique().tolist()) if n_dead else '-'})"
    )
    logger.info(f"Long gaps (cat>=2) to fill: {len(long_df)}")
    logger.info(
        f"Outputs: {output_prefix}_gap_rule.csv . {output_prefix}_gap_size_distribution (.csv/.tex) . "
        f"{output_prefix}_gap_profile_by_channel.csv . {output_prefix}_gap_profile_summary (.csv/.tex) . "
        f"{output_prefix}_long_gaps.csv . {output_prefix}_dead_channels (.csv/.tex)"
    )

    if bool(args.run_heldout):
        run_validation(df, prof, logger)
    else:
        logger.info("Held-out proof not executed. Use --run-heldout to enable it.")

    manifest = {
        "script": f"{output_prefix}_characterize_gaps.py",
        "input": str(input_csv),
        "output_prefix": output_prefix,
        "gap_rule_source": "calculada_no_07_a_partir_da_maior_lacuna_preenchivel_em_canais_nao_mortos",
        "gap_rule": [
            {"categoria": int(r[2]), "min_min": int(r[0]), "max_min": (None if r[1] is None else int(r[1])), "n_porcoes": int(r[2])}
            for r in gap_rule
        ],
        "dead_pct": dead_pct,
        "short_gap_max_min": short_gap_max_min,
        "grid_step_min": grid_step_min,
        "max_fillable_gap_min": max_fillable_gap_min,
        "cat_counts": {str(k): int(v) for k, v in cat_n.items()},
        "cat_minutes": {str(k): int(v) for k, v in cat_min.items()},
        "n_dead_channels": int(n_dead),
        "n_long_gaps": int(len(long_df)),
        "timestamp": datetime.now().isoformat(),
    }
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFESTS_DIR / f"{output_prefix}_characterize_gaps_params.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Saved: manifests/{output_prefix}_characterize_gaps_params.json")
    logger.info(f"=== {output_prefix}_characterize_gaps.py completed ===")


if __name__ == "__main__":
    main()
