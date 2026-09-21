#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: fig_pfi_grouped.py

File type: Figure-generation script (auxiliary; reads pipeline artifacts only)

Purpose:
    Reproduces fig_resultado_11.pdf: GROUPED permutation importance (I_g) per electrical
    family, for the LSTM.

    Closes the provenance gap of the figure: script 20 writes only the CSV
    (metrics/20_pfi_grouped.csv) and no generator in the repository drew this figure. Here the
    drawing is traceable to the data.

    Axis label written in full ("Grouped permutation importance"), without the abbreviation
    PFI, which is not defined in the body of the article.

    Input:  figdata/20_pfi_grouped_long.csv (and metrics/20_pfi_grouped.csv)
    Output: figures/fig_resultado_11.pdf

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
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True, "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 18, "axes.labelsize": 18, "xtick.labelsize": 18, "ytick.labelsize": 18,
    "legend.fontsize": 18, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
if shutil.which("latex") is None:
    plt.rcParams["text.usetex"] = False

ROOT = Path(__file__).resolve().parents[2]  # repository root
MET = ROOT / "metrics"
FIGD = ROOT / "figdata"
ART = ROOT / "figures"; ART.mkdir(parents=True, exist_ok=True)

BAR = "#0072B2"   # Okabe-Ito blue

# family -> calligraphic label used in the article
LAB = {"G_V": r"$\mathcal{G}_{V}$", "G_hV": r"$\mathcal{G}_{hV}$",
       "G_I": r"$\mathcal{G}_{I}$", "G_THD": r"$\mathcal{G}_{THD}$",
       "G_sec": r"$\mathcal{G}_{sec}$", "G_hI": r"$\mathcal{G}_{hI}$",
       "G_P": r"$\mathcal{G}_{P}$"}


def main():
    # Error bar = standard deviation of the FIVE PER-SEED MEANS, not of the 25
    # observations (5 seeds x 5 repetitions). The seed is the unit of repetition;
    # the 5 internal permutations of each seed are not independent. The figure is
    # therefore consistent with the caption ("+/- standard deviation across five seeds")
    # and with the per-family comparison table.
    lg = pd.read_csv(FIGD / "20_pfi_grouped_long.csv")
    per_seed = lg.groupby(["family", "seed"]).I_g.mean()
    d = (per_seed.groupby("family").agg(I_g_mean="mean", I_g_sd_seed=lambda s: s.std(ddof=1))
         .reset_index().sort_values("I_g_mean", ascending=False))
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, d.I_g_mean, width=0.68, color=BAR,
           yerr=d.I_g_sd_seed, capsize=4, error_kw={"ecolor": "black", "elinewidth": 1.2})
    ax.axhline(0, color="0.4", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([LAB.get(f, f) for f in d.family])
    ax.set_ylabel(r"Grouped permutation importance, $I_g$")
    fig.savefig(ART / "fig_resultado_11.pdf", format="pdf", dpi=300)
    plt.close(fig)
    print(f"  fig fig_resultado_11.pdf ({len(d)} families; bar = sd of the 5 per-seed means)")


if __name__ == "__main__":
    main()
