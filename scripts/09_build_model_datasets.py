#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 09_build_model_datasets.py

File type: Pipeline script (stage 09: final construction of the modelling datasets R0/R1/R2)

Purpose:
    Final construction of the modelling datasets.

    Definitions
    -----------
    D_raw   = raw data of the file data/raw/dados_maior_v2_transformed.csv.
    D_final = final dataset after alignment, cleaning, sentinel removal, outlier treatment
              and filling.
    R0      = initial reference scenario, RESTRICTED to the meters that reach this stage
              (loads of the article {M2,M5,M6,M7,M10,M11,M14,M16} intersect available data =
              M5,M6,M7,M11,M16). It is not the full reproduction of the initial configuration:
              M2, M10 and M14 do not reach this stage (M2 ~ duplicate of M_G; M14 is an
              ELECTRICALLY DISTINCT load from M15 -> part of P_other; M10 sparse -> P_other).
    R1      = corrected scenario: uses only the audited/selected independent loads
              (candidates of script 05): M5, M6, M7, M11, M15, M16.
    R2      = corrected scenario with energy closure: R1 + P_other, where P_other = P01
              (meter 100 = M10+M12+M13+M14 aggregated in script 05).

    Objective
    ---------
    Execute only the tasks that still belong to the final post-processing:
      1. audit the energy preservation of the filling;
      2. mask dead/blocked channels in the model-ready base;
      3. build the 3 consolidated datasets R0, R1 and R2 (X = M_G; Y = loads of the scenario);
      4. write the scenario mapping and the manifest.

    What this script does NOT do
    ----------------------------
    This script ONLY audits the previous scope; it does NOT exclude meters anew. The scope
    selection (including the representative of M8/M9/M17/M18 and the inclusion of M12/M13)
    is decided by scripts 03-05. This stage uses the meters that reach it.

    Expected official inputs
    ------------------------
      data/interim/06_para_preencher.csv       (pre-fill; or --pre-fill-file)
      data/processed/08_preenchido.parquet     (filled; or --filled-file)
      audits/05_meter_scope_preselection.csv   (scope; audit only)
      audits/06_dead_channels.csv              (or the equivalent produced upstream)

    Main outputs
    ------------
      data/processed/09_R0.parquet             R0 (M_G + loads {5,6,7,11,16})
      data/processed/09_R1.parquet             R1 (M_G + loads {5,6,7,11,15,16})
      data/processed/09_R2.parquet             R2 (R1 + P_other = P01/meter 100)
      data/processed/09_model_ready.parquet    filled base (masked for dead channels)

      audits/09_scenario_mapping.csv           authoritative roles per column/scenario
      audits/09_energy_preservation_audit.csv  energy audit of the filling
      audits/09_gamma5_electrical.csv          M14 vs M15 electrical diagnostic

      manifests/09_build_model_datasets_params.json

    Common usage
    ------------
      python scripts/09_build_model_datasets.py

    With a specific filled file:
      python scripts/09_build_model_datasets.py --filled-file data/processed/08_preenchido.parquet

    Automatically filter meters outside the approved scope:
      python scripts/09_build_model_datasets.py --filter-to-scope

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
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

TIME_COL = "time"
METER_COL = "meter_id"
PHASES = ["a", "b", "c"]
PHASE_TO_PCOL = {"a": "p_a", "b": "p_b", "c": "p_c"}
PCOL_TO_PHASE = {v: k for k, v in PHASE_TO_PCOL.items()}

# Audit parameters.
DEAD_PCT_DEFAULT = 99.0
PRESERVE_ALERT_PCT_DEFAULT = 1.0
PRESERVE_BLOCK_PCT_DEFAULT = 5.0
NEAR_ZERO_KWH_DEFAULT = 1.0

# Final methodological scope.
METER_MG = 1
P_OTHER_METERS = {100}          # pseudo-meter P01 = P_other (M10+M12+M13+M14 aggregated in script 05)
SENSITIVITY_METERS = set()      # no separate sensitivity meter (components aggregated in P01)
REDUNDANT_GROUP = {8, 9, 17, 18}

# -- Scenarios: use ONLY the meters that reach this stage. ---------------------------------
# Dependent (Y) as in the header; independent (X) = M_G. P_other = P01 (meter 100).
R0_LOADS = [5, 6, 7, 11, 16]            # article {2,5,6,7,10,11,14,16} intersect arriving data
R1_LOADS = [5, 6, 7, 11, 15, 16]        # audited/selected independent loads (candidates of script 05)
POTHER_METER_ID = 100                   # P01 = P_other (M10+M12+M13+M14 aggregated in script 05)
GAMMA_ID = {5: 1, 6: 2, 7: 3, 11: 4, 15: 5, 16: 6}  # stable gamma_id per meter (same structure across scenarios)

# Scenario R0: reference (5 loads x 3 phases = 15 outputs). Stable gamma_id (M15=Gamma5 only in R1).
R0_GAMMA = {
    1: (5, ["a", "b", "c"]),
    2: (6, ["a", "b", "c"]),
    3: (7, ["a", "b", "c"]),
    4: (11, ["a", "b", "c"]),
    6: (16, ["a", "b", "c"]),
}

# Scenario R1 (main): 6 loads x 3 phases = 18 outputs. Stable gamma_id (M15=Gamma5).
R1_DESIGN = {
    1: [(5, "a"), (5, "b"), (5, "c")],
    2: [(6, "a"), (6, "b"), (6, "c")],
    3: [(7, "a"), (7, "b"), (7, "c")],
    4: [(11, "a"), (11, "b"), (11, "c")],
    5: [(15, "a"), (15, "b"), (15, "c")],
    6: [(16, "a"), (16, "b"), (16, "c")],
}
CHAN_TO_GAMMA_R1 = {ch: gid for gid, chans in R1_DESIGN.items() for ch in chans}
R0_SOURCE_METERS = {mid for mid, _ in R0_GAMMA.values()}


@dataclass
class Paths:
    project_root: Path
    interim: Path
    processed: Path
    audits: Path
    figdata: Path
    logs: Path
    manifests: Path


# ─────────────────────────────────────────────────────────────────────────────
# Paths, log and reading
# ─────────────────────────────────────────────────────────────────────────────


def resolve_project_root() -> Path:
    """Repository root: the parent of the directory that contains this script.
    The optional environment variable HARMONIC_PROJECT_ROOT overrides it."""
    script_dir = Path(__file__).resolve().parent
    env = os.environ.get("HARMONIC_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return script_dir.parent


def make_paths(project_root: Path) -> Paths:
    p = Paths(
        project_root=project_root,
        interim=project_root / "data" / "interim",
        processed=project_root / "data" / "processed",
        audits=project_root / "audits",
        figdata=project_root / "figdata",
        logs=project_root / "logs",
        manifests=project_root / "manifests",
    )
    for d in [p.processed, p.audits, p.figdata, p.logs, p.manifests]:
        d.mkdir(parents=True, exist_ok=True)
    return p


def setup_logger(paths: Paths) -> logging.Logger:
    logger = logging.getLogger("09_build_model_datasets")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(paths.logs / "09_build_model_datasets.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported extension: {path}")


def normalize_time_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if TIME_COL not in df.columns and df.index.name == TIME_COL:
        df = df.reset_index()
    if TIME_COL not in df.columns:
        raise ValueError(f"Mandatory column missing: {TIME_COL}")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True).dt.tz_convert(None)
    if df[TIME_COL].isna().any():
        raise ValueError("There are invalid timestamps in the time column.")
    return df


def normalize_meter_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if METER_COL not in df.columns:
        raise ValueError(f"Mandatory column missing: {METER_COL}")
    df[METER_COL] = pd.to_numeric(df[METER_COL], errors="raise").astype(int)
    return df


def require_power_cols(df: pd.DataFrame) -> None:
    missing = [c for c in PHASE_TO_PCOL.values() if c not in df.columns]
    if missing:
        raise ValueError(f"Power columns missing: {missing}")


def numeric_array(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def energy_1min_kwh(vals: np.ndarray) -> float:
    """Energy in kWh assuming power in W and a 1-min grid."""
    return float(np.nansum(vals.astype(float)) / 60_000.0)


def safe_rel_pct(reference: float, candidate: float) -> float:
    return float(abs(candidate - reference) / max(abs(reference), 1e-6) * 100.0)


def channel_name(mid: int, ph: str) -> str:
    return f"M{int(mid)}-{ph.upper()}"


def gamma_col_name(gid: int, ph: str) -> str:
    return f"G{int(gid)}_{ph.upper()}"


def pcol_from_phase(ph: str) -> str:
    return PHASE_TO_PCOL[str(ph).lower()]


def parse_path_arg(value: str | None, root: Path) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    p = Path(value).expanduser()
    return p if p.is_absolute() else root / p


# ─────────────────────────────────────────────────────────────────────────────
# Decision files of the previous stages
# ─────────────────────────────────────────────────────────────────────────────


def load_scope_decision(scope_path: Path, logger: logging.Logger) -> tuple[pd.DataFrame, set[int]]:
    if not scope_path.exists():
        raise FileNotFoundError(f"Scope file missing: {scope_path}")
    scope = pd.read_csv(scope_path)
    if METER_COL not in scope.columns or "process_in_next_script" not in scope.columns:
        raise ValueError(
            f"{scope_path.name} must contain the columns {METER_COL!r} and 'process_in_next_script'."
        )
    scope[METER_COL] = pd.to_numeric(scope[METER_COL], errors="raise").astype(int)
    proc = scope["process_in_next_script"]
    if proc.dtype != bool:
        proc = proc.astype(str).str.lower().isin(["true", "1", "yes", "sim"])
    scope["process_in_next_script"] = proc
    allowed = set(scope.loc[scope["process_in_next_script"], METER_COL].astype(int).tolist())
    logger.info(f"Scope loaded from {scope_path.name}: {len(allowed)} approved meters.")
    return scope, allowed


def channel_key_from_dead_row(row: pd.Series) -> tuple[int, str] | None:
    if METER_COL not in row.index:
        return None
    mid = int(row[METER_COL])

    if "phase" in row.index and pd.notna(row["phase"]):
        ph = str(row["phase"]).strip().lower()
        if ph in PHASES:
            return mid, ph

    for var_col in ["variavel", "variable", "var", "column"]:
        if var_col in row.index and pd.notna(row[var_col]):
            var = str(row[var_col]).strip().lower()
            if var in PCOL_TO_PHASE:
                return mid, PCOL_TO_PHASE[var]
            # Accepts names such as M05-A, M5_A, etc.
            for ph in PHASES:
                if var.endswith(f"-{ph}") or var.endswith(f"_{ph}") or var.endswith(f"{ph.upper()}"):
                    return mid, ph

    if "channel" in row.index and pd.notna(row["channel"]):
        txt = str(row["channel"]).strip().lower()
        for ph in PHASES:
            if txt.endswith(f"-{ph}") or txt.endswith(f"_{ph}"):
                return mid, ph
    return None


def load_dead_power_channels(paths: Paths, dead_file: Path | None, logger: logging.Logger) -> tuple[Path | None, set[tuple[int, str]]]:
    candidates = []
    if dead_file is not None:
        candidates.append(dead_file)
    candidates.extend([
        paths.audits / "06_dead_channels.csv",
        paths.audits / "06A_dead_channels.csv",
        paths.audits / "07_dead_channels.csv",
        paths.audits / "04_dead_channels.csv",
        paths.audits / "03_dead_channels.csv",
    ])
    path = first_existing(candidates)
    dead: set[tuple[int, str]] = set()
    if path is None:
        logger.warning("Dead-channel file not found; dead channels will be inferred from pct_nan_pre.")
        return None, dead

    df = pd.read_csv(path)
    if METER_COL not in df.columns:
        logger.warning(f"{path.name} does not contain meter_id; dead-channel list ignored.")
        return path, dead
    df[METER_COL] = pd.to_numeric(df[METER_COL], errors="coerce").astype("Int64")
    df = df[df[METER_COL].notna()].copy()

    for _, row in df.iterrows():
        key = channel_key_from_dead_row(row)
        if key is not None:
            dead.add(key)
    logger.info(f"Dead channels loaded from {path.name}: {len(dead)} power channels.")
    return path, dead


def enforce_scope(
    df_pre: pd.DataFrame,
    df_fill: pd.DataFrame,
    allowed_meters: set[int],
    filter_to_scope: bool,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    meters_pre = set(df_pre[METER_COL].unique().tolist())
    meters_fill = set(df_fill[METER_COL].unique().tolist())
    extra_pre = sorted(meters_pre - allowed_meters)
    extra_fill = sorted(meters_fill - allowed_meters)

    if extra_pre or extra_fill:
        msg = (
            "Meters outside the scope approved by script 05 were found: "
            f"pre={extra_pre}, fill={extra_fill}."
        )
        if not filter_to_scope:
            raise RuntimeError(msg + " Re-run script 05 with scope activation or use --filter-to-scope.")
        logger.warning(msg + " Applying the automatic filter to the approved scope.")
        df_pre = df_pre[df_pre[METER_COL].isin(allowed_meters)].copy()
        df_fill = df_fill[df_fill[METER_COL].isin(allowed_meters)].copy()

    common = sorted(set(df_pre[METER_COL].unique()).intersection(set(df_fill[METER_COL].unique())))
    if not common:
        raise RuntimeError("No meters in common between the pre-fill base and the filled base.")
    df_pre = df_pre[df_pre[METER_COL].isin(common)].copy()
    df_fill = df_fill[df_fill[METER_COL].isin(common)].copy()
    logger.info(f"Effective meters after filling: {common}")
    return df_pre, df_fill


# ─────────────────────────────────────────────────────────────────────────────
# Post-filling audit
# ─────────────────────────────────────────────────────────────────────────────


def build_aligned_channel_vectors(
    df_pre: pd.DataFrame,
    df_fill: pd.DataFrame,
    mid: int,
    ph: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pcol = pcol_from_phase(ph)
    a = df_pre[df_pre[METER_COL] == mid][[TIME_COL, pcol]].sort_values(TIME_COL)
    b = df_fill[df_fill[METER_COL] == mid][[TIME_COL, pcol]].sort_values(TIME_COL)
    merged = pd.merge(a, b, on=TIME_COL, how="outer", suffixes=("_pre", "_fill")).sort_values(TIME_COL)
    return (
        merged[TIME_COL].to_numpy(),
        numeric_array(merged[f"{pcol}_pre"]),
        numeric_array(merged[f"{pcol}_fill"]),
    )


def compute_energy_audit(
    df_pre: pd.DataFrame,
    df_fill: pd.DataFrame,
    dead_declared: set[tuple[int, str]],
    *,
    dead_pct: float,
    preserve_alert_pct: float,
    preserve_block_pct: float,
    near_zero_kwh: float,
    logger: logging.Logger,
) -> pd.DataFrame:
    rows = []
    meters = sorted(set(df_pre[METER_COL].unique()).union(set(df_fill[METER_COL].unique())))
    for mid in meters:
        for ph in PHASES:
            _, pre, fill = build_aligned_channel_vectors(df_pre, df_fill, int(mid), ph)
            n_total = int(len(pre))
            pre_obs = np.isfinite(pre)
            fill_obs = np.isfinite(fill)
            n_obs_pre = int(pre_obs.sum())
            n_nan_pre = int(n_total - n_obs_pre)
            pct_nan_pre = 100.0 * n_nan_pre / max(n_total, 1)
            n_nan_after = int((~fill_obs).sum())
            n_new_filled_cells = int((~pre_obs & fill_obs).sum())

            E_pre_obs = energy_1min_kwh(pre[pre_obs]) if n_obs_pre else 0.0
            E_fill_on_obs = energy_1min_kwh(fill[pre_obs]) if n_obs_pre else 0.0
            E_fill_total = energy_1min_kwh(fill)
            drel_preserve = safe_rel_pct(E_pre_obs, E_fill_on_obs)
            drel_total_gain = safe_rel_pct(E_pre_obs, E_fill_total)

            near_zero = bool(abs(E_pre_obs) < near_zero_kwh)
            dead_by_pct = bool(pct_nan_pre >= dead_pct)
            dead_by_file = bool((int(mid), ph) in dead_declared)
            dead = bool(dead_by_pct or dead_by_file)
            filled_dead_violation = bool(dead and n_new_filled_cells > 0)
            preserve_alert = bool((not near_zero) and drel_preserve > preserve_alert_pct)
            preserve_block = bool((not near_zero) and drel_preserve > preserve_block_pct)

            rows.append({
                "meter_id": int(mid),
                "phase": ph,
                "channel": channel_name(int(mid), ph),
                "n_total": n_total,
                "n_obs_pre": n_obs_pre,
                "n_nan_pre": n_nan_pre,
                "pct_nan_pre": round(pct_nan_pre, 4),
                "n_nan_after_fill": n_nan_after,
                "n_new_filled_cells": n_new_filled_cells,
                "E_pre_observed_kWh": round(E_pre_obs, 8),
                "E_fill_on_observed_kWh": round(E_fill_on_obs, 8),
                "E_fill_total_kWh": round(E_fill_total, 8),
                "drel_preserve_observed_pct": round(drel_preserve, 8),
                "drel_total_gain_pct": round(drel_total_gain, 8),
                "near_zero_energy": near_zero,
                "dead_by_pct": dead_by_pct,
                "dead_declared_by_file": dead_by_file,
                "dead": dead,
                "filled_dead_violation": filled_dead_violation,
                "preserve_alert": preserve_alert,
                "preserve_block": preserve_block,
            })

    out = pd.DataFrame(rows)
    logger.info(
        "Energy audit: %d channels; dead=%d; blocked=%d; filled-dead violations=%d.",
        len(out),
        int(out["dead"].sum()) if len(out) else 0,
        int(out["preserve_block"].sum()) if len(out) else 0,
        int(out["filled_dead_violation"].sum()) if len(out) else 0,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Final decision per channel, without deciding the meter scope
# ─────────────────────────────────────────────────────────────────────────────


def decide_channels_from_scope(energy: pd.DataFrame, allowed_meters: set[int], logger: logging.Logger) -> pd.DataFrame:
    rows = []
    for _, r in energy.iterrows():
        mid = int(r[METER_COL])
        ph = str(r["phase"]).lower()
        gamma_id = CHAN_TO_GAMMA_R1.get((mid, ph))
        in_allowed_scope = bool(mid in allowed_meters)
        in_r0_design = bool(mid in R0_SOURCE_METERS)
        in_r1_design = bool(gamma_id is not None)

        role_final = "excluded"
        reason_parts = []
        use_in_y_r1 = False
        use_in_p_other = False

        if not in_allowed_scope:
            role_final = "excluded_outside_03_scope"
            reason_parts.append("medidor_fora_do_escopo_aprovado_no_03")
        elif bool(r["dead"]):
            role_final = "dead"
            reason_parts.append("canal_morto_pct_ou_lista")
            if bool(r["filled_dead_violation"]):
                reason_parts.append("alerta_canal_morto_recebeu_preenchimento")
        elif bool(r["preserve_block"]):
            role_final = "blocked"
            reason_parts.append("preservacao_energia_observada_falhou")
        elif mid == METER_MG:
            role_final = "M_G_input"
            reason_parts.append("medidor_geral_entrada_X")
        elif in_r1_design:
            role_final = "Gamma_i"
            use_in_y_r1 = True
            reason_parts.append("canal_do_desenho_R1")
        elif mid in P_OTHER_METERS:
            role_final = "P_other"
            use_in_p_other = True
            reason_parts.append("pseudo_medidor_P01_P_other")
        elif mid in SENSITIVITY_METERS:
            role_final = "sensitivity_excluded"
            reason_parts.append("medidor_de_sensibilidade_nao_usado_no_cenario_principal")
        elif mid in REDUNDANT_GROUP:
            role_final = "redundant_excluded_by_03"
            reason_parts.append("grupo_redundante_nao_decidido_no_07")
        elif in_r0_design:
            role_final = "excluded_from_R1_design"
            reason_parts.append("canal_presente_no_R0_mas_fora_do_R1")
        else:
            role_final = "excluded_unassigned"
            reason_parts.append("sem_papel_no_cenario_principal")

        rows.append({
            **r.to_dict(),
            "in_03_scope": in_allowed_scope,
            "role_final": role_final,
            "reason": ";".join(reason_parts),
            "gamma_id_R1": gamma_id,
            "in_R0_design": in_r0_design,
            "in_R1_design": in_r1_design,
            "use_in_Y_R1": bool(use_in_y_r1),
            "use_in_P_other": bool(use_in_p_other),
        })

    dec = pd.DataFrame(rows)
    logger.info("Final decision per channel:")
    for role, n in dec["role_final"].value_counts().sort_index().items():
        logger.info(f"  {role:28s}: {int(n)} channels")
    return dec


def apply_dead_and_blocked_mask(df_fill: pd.DataFrame, decisions: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    df = df_fill.copy()
    bad = decisions[decisions["role_final"].isin(["dead", "blocked"])]
    n_cells = 0
    for _, r in bad.iterrows():
        mid = int(r[METER_COL])
        ph = str(r["phase"]).lower()
        pcol = pcol_from_phase(ph)
        mask = df[METER_COL] == mid
        n_cells += int(mask.sum())
        df.loc[mask, pcol] = np.nan
    logger.info(f"Model-ready: {n_cells} cells masked in dead/blocked channels.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Construction of X, Y and P_other
# ─────────────────────────────────────────────────────────────────────────────


def all_times(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(np.sort(df[TIME_COL].unique()), name=TIME_COL)


def series_for_channel(df: pd.DataFrame, times: pd.DatetimeIndex, mid: int, ph: str) -> pd.Series:
    pcol = pcol_from_phase(ph)
    sub = df[df[METER_COL] == int(mid)].sort_values(TIME_COL)
    if sub.empty:
        return pd.Series(np.nan, index=times, dtype=float)
    s = sub.set_index(TIME_COL)[pcol]
    return pd.to_numeric(s.reindex(times), errors="coerce")


def build_scenario_dataset(df, scenario, load_meters, with_p_other=False, logger=None):
    """Consolidated dataset of a scenario, SAME STRUCTURE as the others:
    index = time; columns = M_G_A/B/C (independent) + G{gid}_{PHASE} of the loads
    (dependent, power per phase) [+ P_other_A/B/C in R2].
    Uses ONLY the meters present in the arriving data (no exclusion of its own)."""
    times = all_times(df)
    present = set(int(m) for m in df[METER_COL].unique())
    D = pd.DataFrame(index=times)
    rows = []
    # Independent (X): general meter.
    for ph in PHASES:
        col = f"M_G_{ph.upper()}"
        D[col] = series_for_channel(df, times, METER_MG, ph)
        rows.append({"scenario": scenario, "role": "M_G_input", "gamma_id": None,
                     "meter_id": METER_MG, "phase": ph, "column_name": col,
                     "channel": channel_name(METER_MG, ph)})
    # Dependent (Y): loads of the scenario (only those that arrive).
    skipped = []
    for mid in load_meters:
        if mid not in present:
            skipped.append(mid)
            continue
        gid = GAMMA_ID.get(mid, mid)
        for ph in PHASES:
            col = gamma_col_name(gid, ph)
            D[col] = series_for_channel(df, times, mid, ph)
            rows.append({"scenario": scenario, "role": "Gamma_i", "gamma_id": gid,
                         "meter_id": mid, "phase": ph, "column_name": col,
                         "channel": channel_name(mid, ph),
                         "E_kWh": round(energy_1min_kwh(D[col].to_numpy(dtype=float)), 8)})
    # Closure (R2): P_other = P01 (meter 100).
    if with_p_other:
        for ph in PHASES:
            col = f"P_other_{ph.upper()}"
            D[col] = series_for_channel(df, times, POTHER_METER_ID, ph)
            rows.append({"scenario": scenario, "role": "P_other", "gamma_id": None,
                         "meter_id": POTHER_METER_ID, "phase": ph, "column_name": col,
                         "channel": channel_name(POTHER_METER_ID, ph),
                         "E_kWh": round(energy_1min_kwh(D[col].to_numpy(dtype=float)), 8)})
    if logger is not None:
        n_loads = sum(1 for r in rows if r["role"] == "Gamma_i") // 3
        msg = f"{scenario}: {D.shape[0]} rows x {D.shape[1]} columns | {n_loads} loads present"
        if skipped:
            msg += f" | medidores ausentes ignorados: {skipped}"
        logger.info(msg)
    return D, pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Report and manifest
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Arguments and main
# ─────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Builds X_MG, Y_R0, Y_R1 and P_other following the scope defined in script 05."
    )
    p.add_argument("--pre-fill-file", default=None, help="Pre-filling file. Default: data/interim/06_para_preencher.csv")
    p.add_argument("--filled-file", default=None, help="Filled file. Default: 08_preenchido.parquet; then 05_preenchido.parquet")
    p.add_argument("--scope-file", default=None, help="Scope file. Default: audits/05_meter_scope_preselection.csv")
    p.add_argument("--dead-file", default=None, help="Dead-channel file. Default: searches 05/04/03_dead_channels.csv")
    p.add_argument("--filter-to-scope", action="store_true", help="Automatically filters meters outside the scope instead of aborting.")
    p.add_argument("--dead-pct", type=float, default=DEAD_PCT_DEFAULT, help="Pre-fill NaN percentage to declare a channel dead by inference. Default: 99.")
    p.add_argument("--preserve-alert-pct", type=float, default=PRESERVE_ALERT_PCT_DEFAULT, help="Alert for energy change at observed points. Default: 1%%.")
    p.add_argument("--preserve-block-pct", type=float, default=PRESERVE_BLOCK_PCT_DEFAULT, help="Block for energy change at observed points. Default: 5%%.")
    p.add_argument("--near-zero-kwh", type=float, default=NEAR_ZERO_KWH_DEFAULT, help="Energy below which the relative error does not block. Default: 1 kWh.")
    return p


def analyze_gamma5_electrical(df_pre, paths, logger):
    """Electrical analysis M14 vs M15 (documents the evidence of DISTINCT LOADS that
    motivated M15=Gamma5 and M14->P_other). Reads the standardised RAW (02), where M14 and M15
    exist as separate meters; M14 no longer appears in the pre-fill (aggregated into P01).
    Uses operational readings (removes the sentinel floor = modal value). Key diagnostic:
    equal V (same busbar) and consistent I/S gain indicate scale/CT; distinct cos(phi) and
    sign of Q indicate different regimes/loads."""
    PHASES = ["a", "b", "c"]
    QTMPL = {"v": "v_{p}n", "i": "i_{p}n", "p": "p_{p}", "q": "q_{p}", "s": "s_{p}", "cos": "cos_{p}"}
    raw_p = paths.interim / "02_raw_standardized.parquet"
    if not raw_p.exists():
        logger.warning("Gamma5 analysis: %s missing; using the pre-fill (M14 may not exist).", raw_p.name)
        src = df_pre
    else:
        src = pd.read_parquet(raw_p)

    def opstats(m, col):
        if col not in src.columns:
            return np.nan, np.nan, 0
        v = src[src[METER_COL] == m][col].dropna()
        if len(v) < 5:
            return np.nan, np.nan, 0
        mode = v.round(6).mode()
        if len(mode):                       # removes the sentinel floor (modal value)
            v = v[v.round(6) != mode.iloc[0]]
        if len(v) < 5:
            return np.nan, np.nan, 0
        return float(v.median()), float(v.quantile(.75) - v.quantile(.25)), int(len(v))

    rows = []
    for ph in PHASES:
        for q, tmpl in QTMPL.items():
            col = tmpl.format(p=ph)
            if col not in df_pre.columns:
                continue
            m14, iqr14, n14 = opstats(14, col)
            m15, iqr15, n15 = opstats(15, col)
            ratio = (m15 / m14) if (m14 and not np.isnan(m14) and m14 != 0) else np.nan
            rows.append({"phase": ph, "quantity": q, "col": col,
                         "M14_median": m14, "M14_iqr": iqr14, "M14_n": n14,
                         "M15_median": m15, "M15_iqr": iqr15, "M15_n": n15,
                         "ratio_M15_M14": ratio})
    out = pd.DataFrame(rows)
    out.to_csv(paths.audits / "09_gamma5_electrical.csv", index=False)

    def med(q, field="ratio_M15_M14"):
        s = out[out.quantity == q][field].dropna()
        return float(s.median()) if len(s) else np.nan
    v_r, i_r, s_r, p_r = med("v"), med("i"), med("s"), med("p")
    cos14, cos15 = med("cos", "M14_median"), med("cos", "M15_median")
    q14, q15 = med("q", "M14_median"), med("q", "M15_median")

    logger.info("-- Gamma5 ELECTRICAL ANALYSIS (M14 vs M15; operational pre-fill data) --")
    logger.info(f"  median ratios M15/M14 : V={v_r:.2f}  I={i_r:.2f}  S={s_r:.2f}  P={p_r:.2f}")
    logger.info(f"  power factor cos(phi)  : M14={cos14:.2f}  M15={cos15:.2f}  (|delta|={abs(cos15-cos14):.2f})")
    logger.info(f"  median Q               : M14={q14:.3g}  M15={q15:.3g}  (sign "
                f"{'EQUAL' if np.sign(q14)==np.sign(q15) else 'OPPOSITE'})")
    same_V = abs(v_r - 1) < 0.05
    consist_IS = abs(i_r - s_r) / max(abs(s_r), 1e-9) < 0.10
    same_cos = abs(cos15 - cos14) < 0.10
    same_Qsign = np.sign(q14) == np.sign(q15)
    logger.info(f"  checks: V_equal={same_V}  gain_I~S={consist_IS}  cos_equal={same_cos}  Q_same_sign={same_Qsign}")
    if same_V and consist_IS and same_cos and same_Qsign:
        verdict = "MESMA_CARGA (V igual, ganho I/S consistente, cosφ e sinal de Q iguais) -> unir com alinhamento de escala"
    elif same_V and (not same_cos or not same_Qsign):
        verdict = "CARGAS_DISTINTAS (same busbar, but cos(phi) and/or sign of Q differ; the gain is not only CT) -> promote to 2 loads"
    else:
        verdict = "INDETERMINADO -> inspecionar por fase"
    logger.info(f"  >>> VERDICT: {verdict}")
    logger.info(f"  Detail per phase/quantity -> audits/09_gamma5_electrical.csv")
    return out, verdict


def main() -> None:
    args = build_parser().parse_args()
    project_root = resolve_project_root()
    paths = make_paths(project_root)
    logger = setup_logger(paths)

    logger.info("=== 09_build_model_datasets.py started ===")
    logger.info(f"Project root: {project_root}")

    input_pre = parse_path_arg(args.pre_fill_file, project_root) or (paths.interim / "06_para_preencher.csv")
    input_fill = parse_path_arg(args.filled_file, project_root) or first_existing([
        paths.processed / "08_preenchido.parquet",
        paths.processed / "05_preenchido.parquet",
    ])
    scope_file = parse_path_arg(args.scope_file, project_root) or (paths.audits / "05_meter_scope_preselection.csv")
    dead_file = parse_path_arg(args.dead_file, project_root)

    if not input_pre.exists():
        logger.error(f"Pre-fill input missing: {input_pre}")
        sys.exit(1)
    if input_fill is None or not input_fill.exists():
        logger.error("Filled base missing. Expected: data/processed/08_preenchido.parquet, or pass --filled-file.")
        sys.exit(1)
    if not scope_file.exists():
        logger.error(f"Scope file missing: {scope_file}")
        sys.exit(1)

    logger.info(f"Pre-fill input: {input_pre}")
    logger.info(f"Filled input: {input_fill}")
    logger.info(f"Scope (script 05): {scope_file}")

    scope_df, allowed_meters = load_scope_decision(scope_file, logger)
    dead_file_used, dead_channels = load_dead_power_channels(paths, dead_file, logger)

    df_pre = read_table(input_pre)
    df_fill = read_table(input_fill)
    df_pre = normalize_meter_col(normalize_time_col(df_pre))
    df_fill = normalize_meter_col(normalize_time_col(df_fill))
    require_power_cols(df_pre)
    require_power_cols(df_fill)

    # Electrical diagnostic M14 vs M15 (records the evidence of distinct loads that
    # motivated M15=Gamma5 and M14->P_other; NO merging). Kept as an audit in the article.
    analyze_gamma5_electrical(df_pre, paths, logger)

    # This script ONLY audits the previous scope; it does NOT exclude meters anew.
    # Scope selection/exclusion is the responsibility of scripts 03-05. Here all arriving
    # meters are used (including the pseudo-meter P01=100), reduced only to the set common
    # to the pre-fill and the filled bases. (scope_df is used for audit only.)
    allowed_meters = set(df_pre[METER_COL].unique()) | set(df_fill[METER_COL].unique())
    df_pre, df_fill = enforce_scope(df_pre, df_fill, allowed_meters, args.filter_to_scope, logger)

    energy = compute_energy_audit(
        df_pre,
        df_fill,
        dead_channels,
        dead_pct=float(args.dead_pct),
        preserve_alert_pct=float(args.preserve_alert_pct),
        preserve_block_pct=float(args.preserve_block_pct),
        near_zero_kwh=float(args.near_zero_kwh),
        logger=logger,
    )
    energy_path = paths.audits / "09_energy_preservation_audit.csv"
    energy.to_csv(energy_path, index=False)
    logger.info(f"Saved: {energy_path}")

    # 'decisions' is used ONLY internally for the masking of dead/blocked channels
    # (right below); the masking uses only role_final in {dead, blocked}, derived from
    # the energy audit (not from the role labels). The role labelling was reconciled with
    # the canonical scope (R1 with M15=Gamma5; P_other=P01/meter 100). Authoritative roles
    # are in audits/09_scenario_mapping.csv.
    decisions = decide_channels_from_scope(energy, allowed_meters, logger)

    df_model = apply_dead_and_blocked_mask(df_fill, decisions, logger)
    model_ready_path = paths.processed / "09_model_ready.parquet"
    df_model.to_parquet(model_ready_path, index=False)
    logger.info(f"Saved: {model_ready_path}")

    # -- 3 consolidated datasets (R0, R1, R2), same structure, no exclusion. ------------------
    # Built from df_fill (data arriving at this stage). Independent = M_G;
    # dependent = loads of the scenario (power per phase); P_other(R2) = P01 (meter 100).
    ds_R0, map_R0 = build_scenario_dataset(df_fill, "R0", R0_LOADS, with_p_other=False, logger=logger)
    ds_R1, map_R1 = build_scenario_dataset(df_fill, "R1", R1_LOADS, with_p_other=False, logger=logger)
    ds_R2, map_R2 = build_scenario_dataset(df_fill, "R2", R1_LOADS, with_p_other=True, logger=logger)

    r0_path = paths.processed / "09_R0.parquet"
    r1_path = paths.processed / "09_R1.parquet"
    r2_path = paths.processed / "09_R2.parquet"
    ds_R0.to_parquet(r0_path)
    ds_R1.to_parquet(r1_path)
    ds_R2.to_parquet(r2_path)
    logger.info(f"Saved: {r0_path} shape={ds_R0.shape}")
    logger.info(f"Saved: {r1_path} shape={ds_R1.shape}")
    logger.info(f"Saved: {r2_path} shape={ds_R2.shape}")

    mapping = pd.concat([map_R0, map_R1, map_R2], ignore_index=True, sort=False)
    mapping_path = paths.audits / "09_scenario_mapping.csv"
    mapping.to_csv(mapping_path, index=False)
    logger.info(f"Saved: {mapping_path}")

    manifest = {
        "script": "09_build_model_datasets.py",
        "run_timestamp": datetime.now().isoformat(),
        "inputs": {"pre_fill": str(input_pre), "filled": str(input_fill), "scope": str(scope_file)},
        "meters_arriving": sorted(int(m) for m in df_fill[METER_COL].unique()),
        "independent_X": f"M_G (meter {METER_MG})",
        "scenarios": {
            "R0": {"loads": R0_LOADS, "shape": list(ds_R0.shape), "file": str(r0_path)},
            "R1": {"loads": R1_LOADS, "shape": list(ds_R1.shape), "file": str(r1_path)},
            "R2": {"loads": R1_LOADS, "p_other_meter": POTHER_METER_ID,
                   "shape": list(ds_R2.shape), "file": str(r2_path)},
        },
        "note": "No exclusion at this stage (responsibility of scripts 03-05); uses only the meters that arrive.",
    }
    (paths.manifests / "09_build_model_datasets_params.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info(f"Manifest saved: {paths.manifests / '09_build_model_datasets_params.json'}")
    logger.info("=== 09_build_model_datasets.py done ===")


if __name__ == "__main__":
    main()
