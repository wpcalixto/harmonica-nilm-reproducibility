#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 27_generate_si_gap_duration_histogram.py

File type: Figure-generation script (Supplementary Information; pipeline stage 27)

Purpose:
    Supplementary figure (drawing only).

    Reads exclusively figdata/26_gap_duration_histogram.csv and
    figdata/26_gap_duration_thresholds.csv (produced by script 26) and generates
    figures/figure_S_gap_duration_histogram.pdf. Does NOT recompute any result (project rule:
    figure scripts never recompute data).

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
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shutil

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # repository root (parent of scripts/)

plt.rcParams.update({
    "text.usetex": True, "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 18, "axes.labelsize": 18, "xtick.labelsize": 15, "ytick.labelsize": 15,
    "legend.fontsize": 14, "savefig.bbox": "tight", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
if shutil.which("latex") is None:
    plt.rcParams["text.usetex"] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(PROJECT_ROOT))
    args = ap.parse_args()
    root = Path(args.project_root)

    hist = pd.read_csv(root / "figdata" / "26_gap_duration_histogram.csv")
    thr = pd.read_csv(root / "figdata" / "26_gap_duration_thresholds.csv")
    centers = np.sqrt(hist["bin_min"] * hist["bin_max"].clip(lower=1))
    widths = hist["bin_max"] - hist["bin_min"]

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.bar(hist["bin_min"], hist["n_blocos"], width=widths, align="edge",
           color="#2c6fbb", edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Gap duration (min)")
    ax.set_ylabel("Number of gap blocks")

    # vertical lines: finite (dynamic) category bounds + 12 h held-out
    cat = sorted(set(int(t) for t in thr.loc[thr["tipo"] == "categoria", "threshold_min"]))
    # keep only the distinct upper bounds (avoids +1 pairs)
    cat_up = [b for b in cat if (b + 1) not in cat]
    for b in cat_up:
        ax.axvline(b, color="0.45", lw=1.0, ls=":")
    ho = int(thr.loc[thr["tipo"] == "heldout_12h", "threshold_min"].iloc[0])
    ax.axvline(ho, color="#d62728", lw=1.6, ls="--", label=f"held-out limit ({ho} min)")
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(root / "figures" / "figure_S_gap_duration_histogram.pdf", format="pdf", dpi=300)
    plt.close(fig)
    print(f"[27] figure generated | bins={len(hist)} | categories={cat_up} | held-out={ho}")


if __name__ == "__main__":
    main()
