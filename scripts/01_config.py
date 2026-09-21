#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 01_config.py

File type: Configuration loader and validator (pipeline stage 01; imported by no other script)

Purpose:
    Central project configuration loader and validator.

    Loads all 6 YAML configuration files (config/), validates that all locked methods have
    status 'locked_main_method', creates the output directories, checks that the raw data
    file exists, and writes the resolved manifest + log.

    Outputs:
        manifests/01_config_resolved.json
        logs/01_config.log

    Success criterion:  all locked methods appear in the manifest with status locked_main_method
    Failure criterion:  the script raises SystemExit if any method is generic/equivalent/to_be_decided

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
import sys
from datetime import datetime
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # repository root (parent of scripts/)
CONFIG_DIR = PROJECT_ROOT / "config"

# ---------------------------------------------------------------------------
# Output directories to create
# ---------------------------------------------------------------------------

OUTPUT_SUBDIRS = [
    "audits", "figdata", "tabdata", "tables", "models",
    "predictions", "dg", "metrics", "figures",
    "logs", "manifests", "article",
]

# Statuses that indicate a method has NOT been properly locked
FORBIDDEN_STATUSES = {"generic", "equivalent", "to_be_decided"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("01_config")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def load_yaml(path: Path, logger: logging.Logger) -> dict:
    if not path.exists():
        logger.error(f"Config file not found: {path}")
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    logger.info(f"Loaded: {path.name}")
    return data


def extract_locked_methods(configs: dict) -> list[dict]:
    """Walk all config dicts and collect entries with a 'status' field."""
    methods = []

    def _walk(obj, path=""):
        if isinstance(obj, dict):
            if "status" in obj:
                methods.append({
                    "path": path,
                    "status": obj["status"],
                    "name": obj.get("name", path.split(".")[-1]),
                    "description": obj.get("description", obj.get("method", "")),
                })
            for k, v in obj.items():
                _walk(v, f"{path}.{k}" if path else k)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                _walk(item, f"{path}[{i}]")

    for cfg_name, cfg_data in configs.items():
        _walk(cfg_data, cfg_name)

    return methods


def validate_locked_methods(methods: list[dict], logger: logging.Logger) -> bool:
    """
    Fail if any method's status is in FORBIDDEN_STATUSES.
    Return True if all locked methods are properly marked.
    """
    ok = True
    locked = [m for m in methods if m["status"] == "locked_main_method"]
    forbidden = [m for m in methods if m["status"] in FORBIDDEN_STATUSES]

    logger.info(f"Found {len(locked)} method(s) with status 'locked_main_method'")

    if forbidden:
        for m in forbidden:
            logger.error(
                f"FORBIDDEN STATUS '{m['status']}' at path '{m['path']}': "
                "fix config before proceeding"
            )
        ok = False
    else:
        logger.info("Validation passed: no forbidden statuses found")

    return ok


def create_directories(logger: logging.Logger) -> None:
    for subdir in OUTPUT_SUBDIRS:
        p = PROJECT_ROOT / subdir
        p.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory ready: {p.relative_to(PROJECT_ROOT)}")

    for data_subdir in ["data/raw", "data/interim", "data/processed", "src"]:
        p = PROJECT_ROOT / data_subdir
        p.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory ready: {p.relative_to(PROJECT_ROOT)}")

    logger.info("All directories created/verified")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    log_path = PROJECT_ROOT / "logs" / "01_config.log"
    # Ensure log dir exists before logger starts
    (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_path)
    logger.info("=== 01_config.py: multi-source industrial NILM ===")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Run timestamp: {datetime.now().isoformat()}")

    # ------------------------------------------------------------------
    # 1. Load all 6 YAML config files
    # ------------------------------------------------------------------
    yaml_files = {
        "project": CONFIG_DIR / "project_config.yaml",
        "meter_map": CONFIG_DIR / "meter_map.yaml",
        "preprocessing": CONFIG_DIR / "preprocessing_config.yaml",
        "model": CONFIG_DIR / "model_config.yaml",
        "dg": CONFIG_DIR / "dg_config.yaml",
        "scenario": CONFIG_DIR / "scenario_config.yaml",
    }

    configs = {}
    for name, path in yaml_files.items():
        configs[name] = load_yaml(path, logger)

    # ------------------------------------------------------------------
    # 2. Assert key locked parameters
    # ------------------------------------------------------------------
    W = configs["preprocessing"]["windows"]["W"]
    seeds = configs["model"]["seeds"]
    assert W == 12, f"W must be 12 — got {W}"
    assert len(seeds) == 5, f"Must have 5 seeds — got {len(seeds)}"
    logger.info(f"W = {W} (locked)")
    logger.info(f"Seeds = {seeds} (locked, {len(seeds)} seeds)")

    # ------------------------------------------------------------------
    # 3. Validate locked method statuses
    # ------------------------------------------------------------------
    methods = extract_locked_methods(configs)
    valid = validate_locked_methods(methods, logger)
    if not valid:
        logger.error("FAILURE: forbidden method status detected. Fix config files and re-run.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Create directories
    # ------------------------------------------------------------------
    create_directories(logger)

    # ------------------------------------------------------------------
    # 5. Check raw data file exists
    # ------------------------------------------------------------------
    raw_csv = PROJECT_ROOT / configs["project"]["files"]["raw_csv"]
    if raw_csv.exists():
        size_mb = raw_csv.stat().st_size / 1e6
        logger.info(f"Raw data found: {raw_csv.name} ({size_mb:.1f} MB)")
    else:
        logger.warning(f"Raw data NOT found at: {raw_csv}")

    # ------------------------------------------------------------------
    # 6. Build and write resolved manifest
    # ------------------------------------------------------------------
    scenario_cfg = configs["scenario"]
    s_scenarios = list(scenario_cfg.get("sensitivity_scenarios", {}).keys())

    manifest = {
        "script": "01_config.py",
        "section": "§00",
        "run_timestamp": datetime.now().isoformat(),
        "project_root": str(PROJECT_ROOT),
        "W": W,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "scenarios_main": list(scenario_cfg["scenarios"].keys()),
        "scenarios_sensitivity": s_scenarios,
        "default_scenario": scenario_cfg["default_scenario"],
        "meters_all": configs["meter_map"]["meters"]["all_ids"],
        "meter_main_grid": configs["meter_map"]["meters"]["main_grid"],
        "phases": configs["meter_map"]["phases"],
        "locked_methods": [m for m in methods if m["status"] == "locked_main_method"],
        "validation_passed": valid,
        "raw_data_exists": raw_csv.exists(),
        "raw_data_path": str(raw_csv),
        "configs_loaded": list(yaml_files.keys()),
        "status": "success",
    }

    manifest_path = PROJECT_ROOT / "manifests" / "01_config_resolved.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(f"Manifest written: {manifest_path.relative_to(PROJECT_ROOT)}")
    logger.info(f"Log written: {log_path.relative_to(PROJECT_ROOT)}")
    logger.info("=== 01_config.py completed successfully ===")
    logger.info(
        f"Summary: W={W}, seeds={seeds}, "
        f"main scenarios={manifest['scenarios_main']}, "
        f"sensitivity scenarios={len(manifest['scenarios_sensitivity'])}, "
        f"locked methods={len(manifest['locked_methods'])}"
    )


if __name__ == "__main__":
    main()
