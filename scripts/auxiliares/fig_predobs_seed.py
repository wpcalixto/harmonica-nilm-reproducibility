#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: fig_predobs_seed.py

File type: Figure-generation script (auxiliary; reads pipeline artifacts only)

Purpose:
    Predicted-vs-observed panels of the LSTM showing the INDIVIDUAL prediction of each seed
    (not the mean over seeds).

    Generates:
      - seed 42  -> fig_resultado_predobs_{1..8}.pdf   (ARTICLE, main text)
      - seeds 123/456/789/1024 -> fig_si_predobs_s{seed}_{1..8}.pdf  (Supplementary Information)
      - GENERIC legend (reusable in all panels): fig_resultado_predobs_legend.pdf
        and fig_si_predobs_legend.pdf

    In each panel: orange = reference target (observed + reconstructed); blue = individual
    prediction of the seed; grey band = min-max envelope of the 5 seeds.

    Inputs:
        figdata/rev_predobs.csv                    (obs_{o} = reference target)
        predictions/14_lstm_predictions.parquet    (prediction per seed x window)
    Output directory: figures/

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
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import shutil

plt.rcParams.update({
    "text.usetex": True, "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 18, "axes.labelsize": 18, "xtick.labelsize": 18, "ytick.labelsize": 18,
    "legend.fontsize": 18, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
if shutil.which("latex") is None:
    plt.rcParams["text.usetex"] = False

ROOT = Path(__file__).resolve().parents[2]  # repository root
FR = ROOT
_rev1 = ROOT / "figdata" / "rev_predobs.csv"
_rev2 = FR / "figdata" / "rev_predobs.csv"
REVCSV = _rev1 if _rev1.exists() else _rev2
PARQUET = FR / "predictions" / "14_lstm_predictions.parquet"
ART = ROOT / "figures"; ART.mkdir(parents=True, exist_ok=True)

OI = {"blue": "#0072B2", "verm": "#D55E00", "grey": "#999999"}
SEL8 = ["G5_C", "G5_A", "G4_A", "G2_B", "G1_A", "G3_C", "G3_B", "G6_C"]
# seed -> output file prefix
JOBS = {42: "fig_resultado_predobs", 123: "fig_si_predobs_s123",
        456: "fig_si_predobs_s456", 789: "fig_si_predobs_s789",
        1024: "fig_si_predobs_s1024"}


def _save(fig, name):
    fig.savefig(ART / name, format="pdf", dpi=300); plt.close(fig); print("  fig", name)


def main():
    p = pd.read_csv(REVCSV)
    dd = pd.read_parquet(PARQUET)
    seeds = sorted(dd.seed.unique())
    # min-max envelope (common to all figures) and per-seed prediction
    env = {}
    perseed = {s: {} for s in seeds}
    for o in SEL8:
        allP = np.array([dd[dd.seed == s].sort_values("window_idx")[o].to_numpy(float) for s in seeds])
        env[o] = (allP.min(0), allP.max(0))
        for k, s in enumerate(seeds):
            perseed[s][o] = allP[k]
    x = p["t"]
    for seed, prefix in JOBS.items():
        for i, o in enumerate(SEL8, 1):
            lo, hi = env[o]
            fig, ax = plt.subplots(figsize=(13, 2.1))
            ax.fill_between(x, lo, hi, color=OI["grey"], alpha=0.5, lw=0, zorder=1)
            ax.plot(x, p[f"obs_{o}"], color=OI["verm"], lw=1.6, zorder=2)
            ax.plot(x, perseed[seed][o], color=OI["blue"], lw=1.6, zorder=3)
            ax.set_xlabel("Test window"); ax.set_ylabel("Power [W]")
            _save(fig, f"{prefix}_{i}.pdf")
    # GENERIC legend (same for article and SI; the seed goes in the LaTeX \caption)
    handles = [
        Line2D([], [], color=OI["verm"], lw=2.4, label="reference target (obs.\\ + rec.)"),
        Line2D([], [], color=OI["blue"], lw=2.4, label="individual LSTM prediction"),
        Patch(facecolor=OI["grey"], alpha=0.5, label="5-seed range"),
    ]
    for legname in ("fig_resultado_predobs_legend.pdf", "fig_si_predobs_legend.pdf"):
        figk = plt.figure(figsize=(12, 0.6))
        figk.legend(handles=handles, ncol=3, frameon=False, loc="center",
                    handlelength=2.2, columnspacing=2.5)
        _save(figk, legname)
    print(f"[fig_predobs_seed] seeds={list(JOBS)}, {len(SEL8)} panels each.")


if __name__ == "__main__":
    main()
