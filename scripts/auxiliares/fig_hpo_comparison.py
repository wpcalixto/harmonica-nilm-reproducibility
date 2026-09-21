#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: fig_hpo_comparison.py

File type: Figure-generation script (auxiliary; reads pipeline artifacts only)

Purpose:
    Reproduces fig_resultado_12.pdf: test ERG_net in the default and tuned (HPO) configurations,
    for PE-ES and LSTM.

    Closes the provenance gap of the hyperparameter-sensitivity figure: reads
    audits/hpo2_comparison.csv (produced by hpo_two_models.py) and draws the bars with error
    bars (+/- 1 standard deviation across the 5 seeds). The comparison is DESCRIPTIVE: the default
    and HPO configurations also differ in window length W (PE-ES 30->12, LSTM 12->36) and are
    evaluated on distinct temporal supports, so the difference does not isolate the effect of
    the optimisation.

    Input:  audits/hpo2_comparison.csv (and audits/hpo2_final_per_seed.csv for the per-seed points)
    Output: figures/fig_resultado_12.pdf

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
AUD = ROOT / "audits"
ART = ROOT / "figures"; ART.mkdir(parents=True, exist_ok=True)

COL = {"standard": "#2ca02c", "HPO": "#9467bd"}   # green = default, purple = HPO
MODELS = ["PE_ES", "LSTM"]
LAB = {"PE_ES": "PE-ES", "LSTM": "LSTM"}
XPOS = {("PE_ES", "standard"): 0.0, ("PE_ES", "HPO"): 0.7,
        ("LSTM", "standard"): 2.0, ("LSTM", "HPO"): 2.7}


def main():
    # points per seed (avoids a bar extending below zero for a non-negative metric
    # and shows the distribution); lines connect the same seed default->HPO.
    d = pd.read_csv(AUD / "hpo2_final_per_seed.csv")
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in MODELS:
        # lines paired by seed
        for s in sorted(d[d.kind == m].seed.unique()):
            ys = []
            for cfg in ("standard", "HPO"):
                row = d[(d.kind == m) & (d.config == cfg) & (d.seed == s)]
                if not row.empty:
                    ys.append((XPOS[(m, cfg)], float(row.erg_test.iloc[0])))
            if len(ys) == 2:
                ax.plot([ys[0][0], ys[1][0]], [ys[0][1], ys[1][1]],
                        color="0.75", lw=0.9, zorder=1)
        for cfg in ("standard", "HPO"):
            sub = d[(d.kind == m) & (d.config == cfg)]
            xp = XPOS[(m, cfg)]
            ax.scatter(np.full(len(sub), xp), sub.erg_test, s=55, color=COL[cfg],
                       edgecolor="white", linewidth=0.6, zorder=3,
                       label=cfg if m == "PE_ES" else None)
            ax.plot([xp - 0.18, xp + 0.18], [sub.erg_test.mean()] * 2,
                    color="black", lw=2.4, zorder=4)
    ax.axhline(0, color="0.6", lw=0.9, ls="--", zorder=0)
    ax.set_xticks([0.35, 2.35])
    ax.set_xticklabels([LAB[m] for m in MODELS])
    ax.set_ylabel(r"$\mathrm{ERG}_{\mathrm{net}}$ [\%] (test)")
    ax.set_xlim(-0.5, 3.2)
    ax.set_ylim(-0.03, 1.5)
    ax.legend(frameon=True, loc="upper center", ncol=2, title=None)
    fig.savefig(ART / "fig_resultado_12.pdf", format="pdf", dpi=300)
    plt.close(fig)
    print("  fig fig_resultado_12.pdf (points per seed, from hpo2_final_per_seed.csv)")


if __name__ == "__main__":
    main()
