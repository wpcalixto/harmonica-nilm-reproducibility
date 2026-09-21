#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: gen_si_dg_figures.py

File type: Figure-generation script (Supplementary Information; reads pipeline artifacts only)

Purpose:
    Supplementary-Information figures for the balance, sensitivity and sensitivity-budget block
    of the load-consistency residual R_cl.

    Same protocol as gen_review_figures.py: ONE PDF per panel, single 18 pt font, English, no
    title (goes in the LaTeX caption), Okabe-Ito palette, symbols identical to the manuscript.
    Reads ONLY the artifacts of 17_dg_balance.py (dg/), never fixed values.

    Outputs (figures/):
        fig_si_dg_loss.pdf         sensitivity to the loss hypothesis (S0-S3)
        fig_si_dg_structural.pdf   structural impact of the P_other composition (S4-S7)
        fig_si_dg_ubudget.pdf      components of the sensitivity budget per phase
        fig_si_dg_intervals.pdf    expanded intervals (k_u=2) per model x phase

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
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import shutil

plt.rcParams.update({
    "text.usetex": True, "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 18, "axes.labelsize": 18, "xtick.labelsize": 18, "ytick.labelsize": 18,
    "legend.fontsize": 18, "axes.titlesize": 18, "figure.titlesize": 18,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
if shutil.which("latex") is None:
    plt.rcParams["text.usetex"] = False

ROOT = Path(__file__).resolve().parents[2]  # repository root
DG   = ROOT / "dg"
ART  = ROOT / "figures"; ART.mkdir(parents=True, exist_ok=True)

OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "verm": "#D55E00",
      "sky": "#56B4E9", "yellow": "#F0E442", "purple": "#CC79A7", "grey": "#999999"}

MODELS = ["LSTM", "RCNN_att", "PE_ES", "SPEC", "MoTE_v2", "Ensemble_G4"]
LAB = {"LSTM": "LSTM", "RCNN_att": "RCNN-att", "PE_ES": "PE-ES", "SPEC": "SPEC",
       "MoTE_v2": r"MoTE$_{v2}$", "Ensemble_G4": "Ensemble-G4"}
MCOL = {"LSTM": OI["blue"], "RCNN_att": OI["orange"], "PE_ES": OI["green"],
        "SPEC": OI["verm"], "MoTE_v2": OI["purple"], "Ensemble_G4": OI["sky"]}
# distinct markers: the loss curves cluster and may coincide
MMRK = {"LSTM": "o", "RCNN_att": "s", "PE_ES": "^", "SPEC": "v", "MoTE_v2": "D",
        "Ensemble_G4": "P"}
PHCOL = {"A": OI["blue"], "B": OI["orange"], "C": OI["green"]}
UCOMP = ["u_met_load", "u_seed", "u_sync"]   # only terms with non-zero derivative in R_cl
ULAB = {"u_met_load": r"$u_{\mathrm{met},L}$", "u_seed": r"$u_{\mathrm{seed}}$",
        "u_sync": r"$u_{\mathrm{sync}}$", "u_imp": r"$u_{\mathrm{imp}}$",
        "u_loss": r"$u_{\mathrm{loss}}$"}
T_HOURS = 16.0833   # test support (n/60), used for W -> kWh


def _save(fig, name):
    fig.savefig(ART / name, format="pdf", dpi=300); plt.close(fig); print("  fig", name)


def _check(df, col, esperado, ctx):
    achado = set(df[col].unique())
    if not esperado <= achado:
        raise RuntimeError(f"{ctx}: missing {sorted(esperado - achado)} in '{col}'")


# -- 1. sensitivity to the loss hypothesis (S0-S3) ----------------------------------------
def fig_loss():
    s = pd.read_csv(DG / "17_dg_sensitivity.csv")
    s = s[s.is_primary & s.kind.isin(["baseline", "loss"])]
    _check(s, "model", set(MODELS), "sensibilidade de perdas")
    iv = pd.read_csv(DG / "17_dg_uncertainty_interval.csv")
    U = 2.0 * np.sqrt((iv[iv.model == "MoTE_v2"].u_total_kWh ** 2).sum())   # k_u=2, 3 phases

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.axhspan(-U, U, color=OI["grey"], alpha=0.30, zorder=0,
               label=r"$\pm k_u u$ ($k_u{=}2$, 3 phases; $u_{\mathrm{loss}}$, $u_{\mathrm{imp}}$ excluded)")
    for m in MODELS:
        d = s[s.model == m].groupby("loss_frac", as_index=False).E_DG_pred_R2_kWh.sum()
        d = d.sort_values("loss_frac")
        ax.plot(100 * d.loss_frac, d.E_DG_pred_R2_kWh, marker=MMRK[m], ms=9, lw=2.0,
                mfc="white", mew=2.0, color=MCOL[m], label=LAB[m], zorder=3)
    ax.axhline(0, color="0.3", lw=1.3, zorder=1)
    ax.set_xlabel(r"Assumed loss fraction $\ell$ [\%] \quad (S0, S1, S2, S3)")
    ax.set_ylabel(r"$E_{R_{\mathrm{cl}}}$ (3 phases) [kWh]")
    ax.set_xticks([0, 1, 2, 5])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False)
    _save(fig, "fig_si_dg_loss.pdf")


# -- 2. energy composition of P_other (S4-S7) ---------------------------------------------
def fig_structural():
    # This is NOT a sensitivity of R_cl (which excludes P_other by definition). When a
    # component of P_other is removed, the change in the closure is EXACTLY the energy of
    # the component (the predictive term of the model cancels in the difference), so the
    # quantity is independent of the architecture. We verify this and aggregate the models.
    # We report the energy of each component as a fraction of E_MG (a property of the
    # composition of P_other, not of R_cl).
    d = pd.read_csv(DG / "17_dg_structural_impact.csv")
    d = d[d.is_primary]
    _check(d, "model", set(MODELS), "impacto estrutural")
    spread = (d.pivot_table(index=["phase", "removed_meter"], columns="model",
                            values="delta_pct_of_E_MG").agg(lambda r: r.max() - r.min(), axis=1))
    assert spread.max() < 0.05, f"esperado model-independent, spread={spread.max():.3f} pp"
    print(f"    (model-independent: spread across architectures = {spread.max():.3f} pp)")
    order = [12, 13, 10, 14]        # scenarios S4, S5, S6, S7
    piv = (d.groupby(["removed_meter", "phase"])["delta_pct_of_E_MG"].mean()
           .unstack().reindex(order))
    x = np.arange(len(order)); w = 0.26
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for k, ph in enumerate("ABC"):
        ax.bar(x + (k - 1) * w, piv[ph], w, color=PHCOL[ph], label=f"Phase {ph}")
    ax.axhline(0, color="0.3", lw=1.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{4+i}\nno M{m}" for i, m in enumerate(order)])
    ax.set_xlabel("$P_{\\mathrm{other}}$ component removed (S4--S7)")
    ax.set_ylabel(r"Component energy [\% of $E_{\mathrm{MG}}$]")
    ax.legend(loc="upper left", frameon=False)
    _save(fig, "fig_si_dg_structural.pdf")


# -- 3. sensitivity budget per phase --------------------------------------------------------
def fig_ubudget():
    """Budget PER MODEL (three-phase aggregate): stacked bars of the 3 terms."""
    u = pd.read_csv(DG / "17_dg_uncertainty.csv")
    fora = {"u_loss", "u_imp"} & set(u.component)
    if fora:
        raise RuntimeError(f"{sorted(fora)} still present; budget not updated")
    _check(u, "model", set(MODELS), "orcamento de sensibilidade")
    a = u[u.phase == "ABC"].pivot(index="model", columns="component", values="u_kWh").reindex(MODELS)
    x = np.arange(len(MODELS)); w = 0.26
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for k, c in enumerate(UCOMP):
        ax.bar(x + (k - 1) * w, a[c], w, color=[OI["blue"], OI["orange"], OI["green"]][k],
               label=ULAB[c])
    ax.plot(x, a["u_total"], "k_", ms=26, mew=2.4, label=r"$u_{\mathrm{comb}}$")
    ax.set_xticks(x); ax.set_xticklabels([LAB[m] for m in MODELS])
    ax.set_xlabel("Model")
    ax.set_ylabel(r"$u$, three phases combined [kWh]")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, frameon=False)
    _save(fig, "fig_si_dg_ubudget.pdf")


# -- 4. expanded intervals per model x phase ------------------------------------------------
def fig_intervals():
    iv = pd.read_csv(DG / "17_dg_uncertainty_interval.csv")
    iv = iv[iv.is_primary]
    _check(iv, "model", set(MODELS), "intervalos expandidos")
    rows = [(m, ph) for m in MODELS for ph in "ABC"]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(11, 9))
    for (m, ph), yy in zip(rows, y):
        r = iv[(iv.model == m) & (iv.phase == ph)].iloc[0]
        e = r.E_DG_pred_R2_kWh; lo, hi = r.CI95_low_kWh, r.CI95_high_kWh
        ax.plot([lo, hi], [yy, yy], color=MCOL[m], lw=2.4, solid_capstyle="butt", zorder=2)
        ax.plot([lo, lo], [yy - .22, yy + .22], color=MCOL[m], lw=2.0, zorder=2)
        ax.plot([hi, hi], [yy - .22, yy + .22], color=MCOL[m], lw=2.0, zorder=2)
        ax.plot(e, yy, "o", ms=8, color=MCOL[m], zorder=3)
    ax.axvline(0, color="0.25", lw=1.6, ls="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{LAB[m]}-{ph}" for m, ph in rows])
    ax.set_xlabel(r"$E_{R_{\mathrm{cl}}}$ with expanded interval, $k_u{=}2$ [kWh]")
    ax.set_ylim(y.min() - .8, y.max() + .8)
    _save(fig, "fig_si_dg_intervals.pdf")


if __name__ == "__main__":
    for f in (fig_loss, fig_structural, fig_ubudget, fig_intervals):
        f()
    print("[gen_si_dg_figures] done.")
