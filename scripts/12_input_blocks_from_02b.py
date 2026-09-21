#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 12_input_blocks_from_02b.py

File type: Pipeline script (stage 12: input-block map for explainability; read-only annotation)

Purpose:
    Link between the channel audit (script 03) and the explainability runners: map of the
    input blocks X0/X1/X2.

    Materialises the INPUT BLOCKS of the explainability analysis from the channel audit
    (script 03), WITHOUT touching the locked selection logic (script 11) or the training. It
    is a read-only stage that only ANNOTATES each M_G feature with its channel class, its
    family G, and its membership in the blocks X0/X1/X2 and in the ablation families A0-A7.

    Block definition (residual version, canonical):
        X0 = audited active powers                    (channel_class = ACTIVE_POWER_VALID)
        X1 = X0 + normalised instrumental proxies     (approved proxies, robust z-score)
        X2 = X0 + proxy residuals                     (r_x = z_x - f(P,Q,S); only the NON-redundant ones)

    Redundancy rule (consistent with the channel audit): a proxy with R2_vs_Pref >= redund_r2
    (0.95) is redundant -> leaves X2 (what is already explained by the power is not
    residualised); the others enter as residuals. R2 >= 0.99 is only a marker of strong
    redundancy.

    Families G (= ablation families A0-A7):
        G_P=A0 (powers) . G_I=A1 (currents) . G_V=A2 (voltages) .
        G_hI=A3 (current harmonics) . G_hV=A4 (voltage harmonics) . G_THD=A5 (THD) .
        G_sec (q/s/cos, secondary proxy) . X1=A6 (all proxies) . X2=A7 (residuals).

    This script does NOT decide the exclusion of any feature from the main flow; that is a
    separate, approved decision. It only delivers the advisory map consumed by the
    explainability runner.

    Inputs:
        audits/03_channel_classification.csv            (classes per channel; filters M_G)
        audits/10_feature_dictionary.csv                (universe of M_G features)
        audits/11_selected_features_by_output.csv       (selected features; flag)
        config/meter_map.yaml (roles.m_g_input) . config/preprocessing_config.yaml (redund_r2)

    Outputs:
        audits/12_input_blocks_from_02b.csv
        manifests/12_input_blocks_from_02b_params.json
        logs/12_input_blocks_from_02b.log

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

SCRIPT_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT  = SCRIPT_DIR.parent  # repository root (parent of scripts/)
CONFIG_DIR    = PROJECT_ROOT / "config"
AUDITS_DIR    = PROJECT_ROOT / "audits"
LOGS_DIR      = PROJECT_ROOT / "logs"
MANIFESTS_DIR = PROJECT_ROOT / "manifests"

CLASS_CSV  = AUDITS_DIR / "03_channel_classification.csv"
FEATDICT   = AUDITS_DIR / "10_feature_dictionary.csv"
SELECTED   = AUDITS_DIR / "11_selected_features_by_output.csv"
OUT_CSV    = AUDITS_DIR / "12_input_blocks_from_02b.csv"

# classes that enter X1 as normalised proxies
PROXY_CLASSES = {"PROXY_AUDITABLE", "SPECTRAL_PROXY", "PHYSICAL_INVALID"}

RE_PHASE = re.compile(r"_([abc])n?(?:_|$)")


def _cfg():
    """(m_g_id, redund_r2) from the config; fallback (1, 0.95)."""
    mg, r2 = 1, 0.95
    try:
        import yaml
        mm = yaml.safe_load((CONFIG_DIR / "meter_map.yaml").read_text(encoding="utf-8")) or {}
        ids = (mm.get("roles", {}) or {}).get("m_g_input", [1])
        mg = int(ids[0]) if ids else 1
        pp = yaml.safe_load((CONFIG_DIR / "preprocessing_config.yaml").read_text(encoding="utf-8")) or {}
        blk = (pp.get("audit_physical", {}) or {})
        if blk.get("redund_r2") is not None:
            r2 = float(blk["redund_r2"])
    except Exception:
        pass
    return mg, r2


def family_G(col: str) -> str:
    if col.startswith("hrm_v_"):
        return "G_hV"
    if col.startswith("hrm_i_"):
        return "G_hI"
    if col.startswith(("thdi", "thdv")):
        return "G_THD"
    if re.match(r"^p_[abc]$", col):
        return "G_P"
    if re.match(r"^(q|s|cos)_[abc]$", col):
        return "G_sec"
    if col.startswith("v_"):
        return "G_V"
    if col.startswith("i_"):
        return "G_I"
    return "G_outra"


ABLATION_OF = {"G_P": "A0", "G_I": "A1", "G_V": "A2", "G_hI": "A3",
               "G_hV": "A4", "G_THD": "A5", "G_sec": "A5b", "G_outra": "-"}


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("12_input_blocks")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8"); fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


def main() -> None:
    log_path = LOGS_DIR / "12_input_blocks_from_02b.log"
    logger = setup_logger(log_path)
    logger.info("=== 12_input_blocks_from_02b.py: link 03 (audit) -> 12 (blocks) ===")
    logger.info(f"Run timestamp: {datetime.now().isoformat()}")

    for p in (CLASS_CSV, FEATDICT):
        if not p.exists():
            logger.error(f"Missing input: {p}"); sys.exit(1)

    m_g_id, redund_r2 = _cfg()
    logger.info(f"M_G = {m_g_id} | redund_r2 = {redund_r2}")

    cls = pd.read_csv(CLASS_CSV)
    cls = cls[cls["meter_id"] == m_g_id][["variable", "phase", "channel_class",
                                          "R2_vs_Pref", "redund_strong"]].copy()
    feat = pd.read_csv(FEATDICT)[["feature"]].copy()

    # 'selected' = set ADOPTED by the ablation (if recorded in 11_candidate_sets.json);
    # otherwise falls back to the RFECV union (11_selected_features_by_output).
    sel = set(); sel_src = "vazio"
    CAND = PROJECT_ROOT / "data" / "processed" / "11_candidate_sets.json"
    if CAND.exists():
        cj = json.loads(CAND.read_text())
        ad = cj.get("adopted")
        if ad and ad in cj:
            sel = set(map(str, cj[ad])); sel_src = f"adotado='{ad}' ({len(sel)} feats)"
    if not sel and SELECTED.exists():
        sdf = pd.read_csv(SELECTED)
        col = "feature" if "feature" in sdf.columns else sdf.columns[-1]
        sel = set(sdf[col].astype(str)); sel_src = "RFECV union (adoption not recorded)"
    if not sel:
        logger.warning("neither adopted set nor union available; column 'selected' will be empty.")
    logger.info(f"'selected' <- {sel_src}")

    df = feat.merge(cls, left_on="feature", right_on="variable", how="left")
    n_unmatched = int(df["channel_class"].isna().sum())
    if n_unmatched:
        logger.warning(f"{n_unmatched} features of script 10 without a class in the channel audit (flagged UNCLASSIFIED).")
    df["channel_class"] = df["channel_class"].fillna("UNCLASSIFIED")

    df["family_G"] = df["feature"].map(family_G)
    df["ablation_family"] = df["family_G"].map(ABLATION_OF).fillna("-")
    df["is_proxy"] = df["channel_class"].isin(PROXY_CLASSES)
    df["redundant"] = df["R2_vs_Pref"].fillna(-1) >= redund_r2
    df["in_X0"] = df["channel_class"] == "ACTIVE_POWER_VALID"
    df["in_X1"] = df["in_X0"] | df["is_proxy"]
    df["residualize"] = df["is_proxy"] & (~df["redundant"])
    df["in_X2"] = df["in_X0"] | df["residualize"]
    df["selected"] = df["feature"].astype(str).isin(sel)

    cols = ["feature", "phase", "family_G", "channel_class", "R2_vs_Pref", "redundant",
            "is_proxy", "residualize", "in_X0", "in_X1", "in_X2", "ablation_family", "selected"]
    df = df[cols]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    logger.info(f"Saved: {OUT_CSV.relative_to(PROJECT_ROOT)} ({len(df)} features)")

    counts = {
        "n_features": int(len(df)),
        "X0": int(df["in_X0"].sum()), "X1": int(df["in_X1"].sum()), "X2": int(df["in_X2"].sum()),
        "residualize": int(df["residualize"].sum()), "redundant": int(df["redundant"].sum()),
        "selected": int(df["selected"].sum()),
        "by_family": df["family_G"].value_counts().to_dict(),
        "by_class": df["channel_class"].value_counts().to_dict(),
    }
    logger.info(f"Blocks: X0={counts['X0']} X1={counts['X1']} X2={counts['X2']} "
                f"(residualize={counts['residualize']}, redundant={counts['redundant']}, "
                f"selected={counts['selected']})")
    logger.info(f"Per family: {counts['by_family']}")

    manifest = {
        "script": "12_input_blocks_from_02b.py", "section": "§14.10.1/§14.11/§14.12",
        "run_timestamp": datetime.now().isoformat(),
        "role": "advisory: map of blocks X0/X1/X2 from the channel audit; does not alter selection nor training",
        "m_g_id": m_g_id, "redund_r2": redund_r2, "proxy_classes": sorted(PROXY_CLASSES),
        "block_definition": {
            "X0": "channel_class == ACTIVE_POWER_VALID",
            "X1": "X0 + proxies (PROXY_AUDITABLE/SPECTRAL_PROXY/PHYSICAL_INVALID)",
            "X2": "X0 + residuals of the non-redundant proxies (R2_vs_Pref < redund_r2)",
        },
        "counts": counts,
        "inputs": {"classes": str(CLASS_CSV.relative_to(PROJECT_ROOT)),
                   "feature_dict": str(FEATDICT.relative_to(PROJECT_ROOT)),
                   "selected": str(SELECTED.relative_to(PROJECT_ROOT))},
        "outputs": {"blocks": str(OUT_CSV.relative_to(PROJECT_ROOT)),
                    "log": str(log_path.relative_to(PROJECT_ROOT))},
        "status": "success",
    }
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    p_manifest = MANIFESTS_DIR / "12_input_blocks_from_02b_params.json"
    with open(p_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved: {p_manifest.relative_to(PROJECT_ROOT)}")
    logger.info("=== 12 done ===")


if __name__ == "__main__":
    main()
