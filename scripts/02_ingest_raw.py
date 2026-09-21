#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 02_ingest_raw.py

File type: Data ingestion script (pipeline stage 02)

Purpose:
    Reading of the raw file + split by variable TYPE (parquet + 4 CSVs).

    Reads ALL variables of the raw file and builds the outputs WITHOUT any correction (no
    sentinel decoding, no outlier removal, no imputation): read-only, structural
    standardisation and split by type. Corrections start in script 06 (sentinels and
    outliers), 06b (envelope sanitisation) and 08 (gap filling).

    Operations:
      1. Read the raw CSV (data/raw/dados_maior_v2_transformed.csv): all columns.
      2. Convert the timestamp to datetime; sort by time (stable).
      3. Remove only EXACT duplicates (no electrical aggregation).
      4. Write the standardised parquet.
      5. Split the columns into 4 TYPES and write 1 CSV per type.

    Types (182 columns in total):
      - primary     (6):  i_an,i_bn,i_cn, v_an,v_bn,v_cn               (direct measurements)
      - secondary   (12): p_*,q_*,s_*,cos_*                            (derived)
      - distortion+meta (11): thdi_*,thdv_* (6) + meta (5: Unnamed:0,time,
                              datetime_read,meter_id,tag_meter_id)
      - harmonics   (153): hrm_i_* (75) + hrm_v_* (78)                 (spectrum)

    Alignment NOTE: the meta columns (time, meter_id) go ONLY in the distortion+meta CSV.
    The other 3 CSVs are aligned by ROW ORDER (all come from the same sorted table); to
    rebuild a record, read the 4 files and concatenate by position.

    Outputs (all with prefix 02_):
        data/interim/02_raw_standardized.parquet
        data/interim/02_primarias.csv
        data/interim/02_secundarias.csv
        data/interim/02_distorcao_meta.csv
        data/interim/02_harmonicas.csv
        audits/02_raw_overview.csv
        figdata/02_timestamp_coverage.csv
        logs/02_ingest_raw.log
        manifests/02_ingest_raw_params.json

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

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT  = SCRIPT_DIR.parent  # repository root (parent of scripts/)


def _raw_csv_from_config() -> Path:
    """Reads files.raw_csv from project_config.yaml (single source of truth).
    Falls back to data/raw/dados_maior_v2_transformed.csv if the config cannot be read."""
    try:
        import yaml
        cfg = yaml.safe_load((PROJECT_ROOT / "config" / "project_config.yaml").read_text(encoding="utf-8"))
        return PROJECT_ROOT / cfg["files"]["raw_csv"]
    except Exception:
        return PROJECT_ROOT / "data" / "raw" / "dados_maior_v2_transformed.csv"


RAW_CSV       = _raw_csv_from_config()
INTERIM_DIR   = PROJECT_ROOT / "data" / "interim"
AUDITS_DIR    = PROJECT_ROOT / "audits"
FIGDATA_DIR   = PROJECT_ROOT / "figdata"
LOGS_DIR      = PROJECT_ROOT / "logs"
MANIFESTS_DIR = PROJECT_ROOT / "manifests"

TIMESTAMP_COL = "time"
METER_COL     = "meter_id"

# ---------------------------------------------------------------------------
# Variable types (column-name patterns)
# ---------------------------------------------------------------------------

RE_PRIMARIA   = re.compile(r"^[iv]_[abc]n$")          # i_an..i_cn, v_an..v_cn
RE_SECUNDARIA = re.compile(r"^(p|q|s|cos)_[abc]$")    # p_*, q_*, s_*, cos_*
RE_DISTORCAO  = re.compile(r"^thd[iv]_")              # thdi_*, thdv_*
RE_HARMONICA  = re.compile(r"^hrm_[iv]_")             # hrm_i_*, hrm_v_*


def split_types(cols):
    """Splits the columns into (primary, secondary, distortion, harmonics, meta),
    preserving the original order within each group."""
    prim = [c for c in cols if RE_PRIMARIA.match(c)]
    sec  = [c for c in cols if RE_SECUNDARIA.match(c)]
    dist = [c for c in cols if RE_DISTORCAO.match(c)]
    harm = [c for c in cols if RE_HARMONICA.match(c)]
    classified = set(prim + sec + dist + harm)
    meta = [c for c in cols if c not in classified]   # everything else = meta
    return prim, sec, dist, harm, meta


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("02_ingest_raw")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8"); fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


def main() -> None:
    log_path = LOGS_DIR / "02_ingest_raw.log"
    logger = setup_logger(log_path)
    logger.info("=== 02_ingest_raw.py: multi-source industrial NILM ===")
    logger.info(f"Run timestamp: {datetime.now().isoformat()}")

    # 1. Read raw CSV --------------------------------------------------------
    if not RAW_CSV.exists():
        logger.error(f"Raw CSV not found: {RAW_CSV}")
        sys.exit(1)
    logger.info(f"Reading: {RAW_CSV.name} ({RAW_CSV.stat().st_size / 1e6:.1f} MB)")
    df = pd.read_csv(RAW_CSV, low_memory=False)
    n_rows_raw, n_cols_raw = df.shape
    logger.info(f"Loaded: {n_rows_raw} rows × {n_cols_raw} columns")

    # 2. Timestamp + sorting -------------------------------------------------
    if TIMESTAMP_COL not in df.columns:
        logger.error(f"Timestamp column '{TIMESTAMP_COL}' not found.")
        sys.exit(1)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=False, errors="coerce")
    n_invalid_ts = int(df[TIMESTAMP_COL].isna().sum())
    if n_invalid_ts:
        logger.warning(f"{n_invalid_ts} rows with unparseable timestamp; kept as NaT")
    df = df.sort_values(TIMESTAMP_COL, kind="stable").reset_index(drop=True)
    logger.info("Timestamp converted; sorted by time (stable)")

    # 3. Exact duplicates ----------------------------------------------------
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_exact_dups = n_before - len(df)
    logger.info(f"Exact duplicates removed: {n_exact_dups} (no electrical aggregation)")

    n_rows_final, n_cols_final = df.shape
    meters = sorted(df[METER_COL].dropna().unique().tolist()) if METER_COL in df.columns else []
    t_start, t_end = df[TIMESTAMP_COL].min(), df[TIMESTAMP_COL].max()
    duration_days = (t_end - t_start).total_seconds() / 86400 if pd.notna(t_start) and pd.notna(t_end) else None
    logger.info(f"Final shape: {n_rows_final} x {n_cols_final} | meters: {meters} | "
                f"period {t_start} -> {t_end} ({duration_days:.2f} days)")

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    AUDITS_DIR.mkdir(parents=True, exist_ok=True)
    FIGDATA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    # 4. Standardised parquet ------------------------------------------------
    parquet_path = INTERIM_DIR / "02_raw_standardized.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info(f"Saved parquet: {parquet_path.relative_to(PROJECT_ROOT)}")

    # 5. Split by TYPE -> 4 CSVs --------------------------------------------
    prim, sec, dist, harm, meta = split_types(list(df.columns))
    logger.info(f"Types: primary={len(prim)} secondary={len(sec)} "
                f"distortion={len(dist)} meta={len(meta)} harmonics={len(harm)}")

    csv_specs = {
        "02_primarias.csv":     prim,
        "02_secundarias.csv":   sec,
        "02_distorcao_meta.csv": meta + dist,   # meta goes together with distortion
        "02_harmonicas.csv":    harm,
    }
    csv_outputs = {}
    for fname, cols in csv_specs.items():
        path = INTERIM_DIR / fname
        df[cols].to_csv(path, index=False)
        csv_outputs[fname] = {"n_cols": len(cols), "path": str(path.relative_to(PROJECT_ROOT))}
        logger.info(f"Saved CSV: {fname}  ({len(cols)} columns)")

    # 6. Audit per meter -----------------------------------------------------
    overview_rows = []
    for meter in meters:
        m_df = df[df[METER_COL] == meter]
        m_start, m_end = m_df[TIMESTAMP_COL].min(), m_df[TIMESTAMP_COL].max()
        overview_rows.append({
            "meter_id": meter, "n_rows": len(m_df),
            "t_start": m_start, "t_end": m_end,
            "duration_days": (m_end - m_start).total_seconds() / 86400
            if pd.notna(m_start) and pd.notna(m_end) else None,
            "n_nulls_timestamp": int(m_df[TIMESTAMP_COL].isna().sum()),
        })
    overview_path = AUDITS_DIR / "02_raw_overview.csv"
    pd.DataFrame(overview_rows).to_csv(overview_path, index=False)
    logger.info(f"Saved: {overview_path.relative_to(PROJECT_ROOT)}")

    # 7. Sampling coverage (ROW presence, not actual data) -------------------
    if METER_COL in df.columns and pd.notna(t_start) and pd.notna(t_end):
        full_index = pd.date_range(start=t_start.floor("min"), end=t_end.ceil("min"), freq="1min")
        coverage_rows = []
        for meter in meters:
            present_ts = set(df.loc[df[METER_COL] == meter, TIMESTAMP_COL].dt.floor("min").dropna())
            for ts in full_index:
                coverage_rows.append({"timestamp": ts, "meter_id": meter,
                                      "row_present": int(ts in present_ts)})
        coverage_df = pd.DataFrame(coverage_rows)
    else:
        coverage_df = pd.DataFrame(columns=["timestamp", "meter_id", "row_present"])
    coverage_path = FIGDATA_DIR / "02_timestamp_coverage.csv"
    coverage_df.to_csv(coverage_path, index=False)
    logger.info(f"Saved: {coverage_path.relative_to(PROJECT_ROOT)}")

    # 8. Manifest ------------------------------------------------------------
    manifest = {
        "script": "02_ingest_raw.py", "section": "§01",
        "run_timestamp": datetime.now().isoformat(),
        "input_file": str(RAW_CSV), "input_size_mb": round(RAW_CSV.stat().st_size / 1e6, 1),
        "n_rows_raw": n_rows_raw, "n_cols_raw": n_cols_raw,
        "n_exact_duplicates_removed": n_exact_dups,
        "n_rows_final": n_rows_final, "n_cols_final": n_cols_final,
        "n_invalid_timestamps": n_invalid_ts,
        "meters": meters, "n_meters": len(meters),
        "t_start": str(t_start), "t_end": str(t_end),
        "duration_days": round(duration_days, 4) if duration_days else None,
        "no_repairs": "leitura fiel — sentinelas/outliers preservados (consertos no 02 e 04)",
        "variable_types": {
            "primarias": prim, "secundarias": sec,
            "distorcao": dist, "meta": meta, "harmonicas_count": len(harm),
        },
        "outputs": {
            "parquet": str(parquet_path.relative_to(PROJECT_ROOT)),
            "csv_by_type": csv_outputs,
            "overview": str(overview_path.relative_to(PROJECT_ROOT)),
            "coverage": str(coverage_path.relative_to(PROJECT_ROOT)),
            "log": str(log_path.relative_to(PROJECT_ROOT)),
        },
        "status": "success",
    }
    manifest_path = MANIFESTS_DIR / "02_ingest_raw_params.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved: {manifest_path.relative_to(PROJECT_ROOT)}")

    logger.info("=== 02_ingest_raw.py completed successfully ===")
    logger.info(f"Summary: {n_rows_final} rows | parquet + 4 CSVs "
                f"(prim {len(prim)}, sec {len(sec)}, dist+meta {len(meta)+len(dist)}, harm {len(harm)})")


if __name__ == "__main__":
    main()
