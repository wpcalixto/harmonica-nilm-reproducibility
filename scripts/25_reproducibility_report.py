#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 25_reproducibility_report.py

File type: Reproducibility validation script (pipeline stage 25)

Purpose:
    Reproducibility report: library versions, parameters, seeds, scripts, checksums,
    scenarios and decisions, collected from the manifests and artifacts of the pipeline.

    Anchors checked: 09_R1/R2, 11_*, 10_kneedle, 14_load_metrics, 17_dg_energy,
    17_dg_uncertainty_interval, table_05, 20_article_values; scenarios from
    09_scenario_mapping / 17_dg_energy / 17_dg_sensitivity; training stages
    14_train_baselines / 15_train_advanced (including the PE-ES-Optuna sensitivity run).

    Outputs: manifests/25_reproducibility_manifest.json, article/25_pipeline_report.md, log.

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

import csv
import hashlib
import importlib.metadata
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # repository root (parent of scripts/)
MANIFESTS = PROJECT_ROOT / "manifests"
ARTICLE   = PROJECT_ROOT / "article"
LOGS      = PROJECT_ROOT / "logs"
for d in (MANIFESTS, ARTICLE, LOGS):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(LOGS / "25_reproducibility_report.log", mode="w", encoding="utf-8")])
log = logging.getLogger(__name__)

DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
TRAINING_SCRIPTS = ["14_train_baselines_params", "15_train_advanced_params",
                    "15_exp_pe_es_optuna_params"]
PACKAGES = ["tensorflow", "keras", "numpy", "pandas", "scikit-learn", "scipy",
            "PyYAML", "kneed", "optuna", "pyarrow", "matplotlib"]
ANCHOR_FILES = [
    "data/processed/09_R1.parquet", "data/processed/09_R2.parquet",
    "data/processed/13_X_train.npy", "data/processed/13_X_test.npy",
    "data/processed/13_Y_raw_test.npy",
    "audits/11_kneedle_selection.csv", "metrics/16_load_metrics_long.csv",
    "dg/17_dg_energy_by_phase.csv", "dg/17_dg_uncertainty_interval.csv",
    "tabdata/table_05_model_metrics.csv", "article/24_article_values.yaml",
]
KEY_DELIVERABLES = {  # glob -> description
    "tables/table_*.tex": "LaTeX tables (18)",
    "figures/figure_*.pdf": "PDF figures (19)",
    "dg/17_dg_*.csv": "D_G outputs (15)",
    "metrics/18_*.csv": "statistics/PFI (16)",
    "metrics/21_economic_*.csv": "economic (17)",
    "article/2*_*.yaml": "article values (20)",
    "article/2*_*.md": "article reports/checklists (20–21)",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()[:16]


def pkg_version(name):
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "n/a"


def main():
    log.info("=== 25_reproducibility_report.py ===")

    # [1] manifests
    manifests = {}
    for mf in sorted(MANIFESTS.glob("*.json")):
        try:
            manifests[mf.stem] = json.loads(mf.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("  error in %s: %s", mf.name, e)
    def skey(s):
        p = s.split("_")[0]
        try:
            return (int("".join(ch for ch in p if ch.isdigit()) or 999), s)
        except ValueError:
            return (999, s)
    order = sorted(manifests, key=skey)
    log.info("[1] %d manifests", len(manifests))

    # [2] seeds
    seeds_by = {}
    for ts in TRAINING_SCRIPTS:
        if ts in manifests:
            d = manifests[ts]
            s = d.get("seeds") or d.get("parameters", {}).get("seeds")
            if s:
                seeds_by[ts.replace("_params", "")] = s
    all_seeds = sorted({x for v in seeds_by.values() for x in v}) or DEFAULT_SEEDS
    log.info("[2] seeds: %s", all_seeds)

    # [3] versions
    libs = {"python": platform.python_version(), "platform": platform.platform()}
    for p in PACKAGES:
        libs[p] = pkg_version(p)
    log.info("[3] TF=%s numpy=%s pandas=%s", libs.get("tensorflow"), libs.get("numpy"), libs.get("pandas"))

    # [4] checksums
    checks = {}
    for rp in ANCHOR_FILES:
        f = PROJECT_ROOT / rp
        checks[rp] = ({"sha256_16": sha256(f), "size_bytes": f.stat().st_size}
                      if f.exists() else {"sha256_16": None, "size_bytes": None})
    n_anchor_ok = sum(1 for v in checks.values() if v["sha256_16"])
    log.info("[4] anchors present: %d/%d", n_anchor_ok, len(ANCHOR_FILES))

    # [5] key deliverables (globs)
    deliverables = {}
    for pat, desc in KEY_DELIVERABLES.items():
        n = len(list(PROJECT_ROOT.glob(pat)))
        deliverables[pat] = {"n": n, "desc": desc}
        log.info("[5] %-28s %d (%s)", pat, n, desc)

    # [6] scenarios
    scen = {}
    sm = PROJECT_ROOT / "audits" / "09_scenario_mapping.csv"
    if sm.exists():
        rows = list(csv.DictReader(sm.open(encoding="utf-8")))
        for s in sorted({r["scenario"] for r in rows}):
            sub = [r for r in rows if r["scenario"] == s]
            scen[s] = {"n_gamma": len({r["gamma_id"] for r in sub if r["gamma_id"]}),
                       "n_rows": len(sub)}
    de = PROJECT_ROOT / "dg" / "17_dg_energy_by_phase.csv"
    if de.exists():
        rows = list(csv.DictReader(de.open(encoding="utf-8")))
        scen["dg_balance"] = {"scenarios": sorted({r["scenario"] for r in rows})}
    ds = PROJECT_ROOT / "dg" / "17_dg_sensitivity.csv"
    if ds.exists():
        rows = list(csv.DictReader(ds.open(encoding="utf-8")))
        scen["dg_sensitivity"] = {"n_scenarios": len({r["scenario"] for r in rows})}
    log.info("[6] scenarios: %s", list(scen))

    status = "OK" if (n_anchor_ok == len(ANCHOR_FILES)) else f"INCOMPLETE ({len(ANCHOR_FILES)-n_anchor_ok} anchors missing)"

    repro = {"script": "25_reproducibility_report.py", "ex": "31_reproducibility_report.py",
             "generated_at": datetime.now(timezone.utc).isoformat(),
             "pipeline_manifests": order, "n_manifests": len(manifests),
             "seeds": {"global": all_seeds, "by_script": seeds_by},
             "library_versions": libs, "checksums": checks,
             "key_deliverables": deliverables, "scenarios": scen,
             "anchors_present": f"{n_anchor_ok}/{len(ANCHOR_FILES)}",
             "reproducibility_status": status}
    (MANIFESTS / "25_reproducibility_manifest.json").write_text(json.dumps(repro, indent=2, ensure_ascii=False, default=str))
    log.info("  manifests/25_reproducibility_manifest.json")

    # markdown report
    ic = "✓" if status == "OK" else "⚠"
    L = ["# Pipeline Reproducibility Report",
         f"> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · 25_reproducibility_report.py",
         f"> Status: **{status}**", "",
         "## 1. Pipeline manifests", "", "| # | manifest | timestamp |", "|---|---|---|"]
    for s in order:
        ts = str(manifests[s].get("run_timestamp") or manifests[s].get("generated_at") or "—")[:19]
        L.append(f"| {s.split('_')[0]} | {s} | {ts} |")
    L += ["", "## 2. Seeds", "", f"Global seeds: **{all_seeds}**", "",
          "| Script | Seeds |", "|---|---|"]
    for k, v in seeds_by.items():
        L.append(f"| {k} | {v} |")
    L += ["", "## 3. Library versions", "", "| Library | Version |", "|---|---|",
          f"| Python | {libs['python']} |", f"| Platform | {libs['platform']} |"]
    for p in PACKAGES:
        L.append(f"| {p} | {libs[p]} |")
    L += ["", "## 4. Key deliverables", "", "| Pattern | Count | Description |", "|---|---|---|"]
    for pat, v in deliverables.items():
        L.append(f"| `{pat}` | {v['n']} | {v['desc']} |")
    L += ["", "## 5. Checksums (SHA-256, first 16 hex)", "", "| File | SHA-256(16) | Bytes |", "|---|---|---|"]
    for rp, v in checks.items():
        cs = v["sha256_16"] or "MISSING"
        sz = f"{v['size_bytes']:,}" if v["size_bytes"] else "—"
        L.append(f"| {rp} | `{cs}` {'✓' if v['sha256_16'] else '⚠'} | {sz} |")
    L += ["", "## 6. Scenarios", "", "| Key | Detail |", "|---|---|"]
    for k, v in scen.items():
        L.append(f"| {k} | {v} |")
    L += ["", "---", "*Auto-generated by `25_reproducibility_report.py`.*"]
    (ARTICLE / "25_pipeline_report.md").write_text("\n".join(L), encoding="utf-8")
    log.info("  article/25_pipeline_report.md")
    log.info("=== Done | %s %s ===", ic, status)


if __name__ == "__main__":
    main()
