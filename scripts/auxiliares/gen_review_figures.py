#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: gen_review_figures.py

File type: Figure-generation script (auxiliary; also prepares figdata/rev_*.csv from pipeline artifacts)

Purpose:
    SINGLE GENERATOR of the review figures, following the figure protocol:
      - ONE PDF per panel (no internal subplots/subfigures);
      - reads ONLY from figdata/ (rev_*.csv), prepared by --prep;
      - single 18 pt font in every figure (never smaller); English; no title (goes in the LaTeX caption);
      - symbols identical to the manuscript; discrete colourblind-safe palette (Okabe-Ito) + cividis for maps.

    Usage:
      python scripts/auxiliares/gen_review_figures.py --prep   # (re)populates figdata/rev_*.csv from the artifacts
      python scripts/auxiliares/gen_review_figures.py          # draws all PDFs into figures/

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
import argparse, json, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import shutil

# -- Protocol (identical to the figure protocol of script 23: EVERYTHING at 18 pt) ---------------
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
WP = ROOT
FR = ROOT
FIGDATA = WP / "figdata"
ART = ROOT / "figures"; ART.mkdir(parents=True, exist_ok=True)

# Okabe-Ito (colourblind-safe)
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "verm": "#D55E00",
      "sky": "#56B4E9", "yellow": "#F0E442", "purple": "#CC79A7", "black": "#000000", "grey": "#999999"}
SEQ = "cividis"  # sequential colourblind-safe colormap (maps)


def _save(fig, name):
    fig.savefig(ART / name, format="pdf", dpi=300); plt.close(fig); print("  fig", name)


# ========================= PREP: populate figdata/rev_*.csv =========================
def prep():
    FIGDATA.mkdir(exist_ok=True)
    N = 10000
    # coverage (presence)
    r = pd.read_csv(WP/"data/raw/dados_maior_v2_transformed.csv", usecols=["time", "meter_id"])
    t = pd.to_datetime(r["time"], utc=True); idx = ((t - t.min().floor("min")).dt.total_seconds()//60).astype(int)
    order = [1, 5, 6, 7, 11, 15, 16, 10, 12, 13, 14, 2, 8, 9, 17, 18]
    pres = np.zeros((len(order), N), np.uint8)
    for row, m in enumerate(order):
        ii = idx[r.meter_id == m].values; ii = ii[(ii >= 0) & (ii < N)]; pres[row, ii] = 1
    np.save(FIGDATA/"rev_cov_presence.npy", pres)
    json.dump(order, open(FIGDATA/"rev_cov_order.json", "w"))
    # M2 vs M_G (cumulative energy + cross-correlation)
    rr = pd.read_csv(WP/"data/raw/dados_maior_v2_transformed.csv", usecols=["time", "meter_id", "p_a", "p_b", "p_c"])
    rr["mm"] = pd.to_datetime(rr["time"], utc=True).dt.floor("min"); DT = 1/60000.0
    ce, xc = {}, {}; lags = np.arange(-30, 31)
    for ph, c in {"A": "p_a", "B": "p_b", "C": "p_c"}.items():
        sg = rr[rr.meter_id == 1].dropna(subset=[c]).groupby("mm")[c].mean()
        s2 = rr[rr.meter_id == 2].dropna(subset=[c]).groupby("mm")[c].mean()
        i = sg.index.intersection(s2.index); g = sg.loc[i].values; m = s2.loc[i].values; n = len(i)
        ce[f"MG_{ph}"] = np.cumsum(g)*DT; ce[f"M2_{ph}"] = np.cumsum(m)*DT
        xc[ph] = [np.corrcoef((g[max(L, 0):n+min(L, 0)]-g.mean())/(g.std()+1e-9), (m[max(-L, 0):n+min(-L, 0)]-m.mean())/(m.std()+1e-9))[0, 1] if n-abs(L) > 10 else 0 for L in lags]
    pd.DataFrame({k: pd.Series(v) for k, v in ce.items()}).to_csv(FIGDATA/"rev_m2_cumenergy.csv", index=False)
    pd.DataFrame({"lag": lags, **xc}).to_csv(FIGDATA/"rev_m2_xcorr.csv", index=False)
    # audit (coherence + eligibility)
    ph = pd.read_csv(WP/"audits/03_phys_audit.csv")
    inv = pd.read_csv(WP/"audits/04_inverse_residuals_by_phase.csv")[["meter_id", "phase", "resid_vi_before", "resid_pcos_before", "resid_tri_before"]]
    cls = pd.read_csv(WP/"audits/03_channel_classification.csv"); pw = cls[cls.family == "p"][["meter_id", "phase", "n_real", "requires_sensitivity", "scale_suspect_flag", "channel_class"]]
    d = ph.merge(inv, on=["meter_id", "phase"], how="left").merge(pw, on=["meter_id", "phase"], how="left")
    om = [1, 2, 5, 6, 7, 11, 15, 16, 10, 12, 13, 14, 8, 9, 17, 18]; d["mk"] = d.meter_id.map({m: i for i, m in enumerate(om)}); d = d.sort_values(["mk", "phase"]).reset_index(drop=True)
    d["dead"] = d.meter_id.isin([8, 9, 17, 18]); d["star"] = ((d.meter_id == 16) | ((d.meter_id.isin([1, 2, 12])) & (d.phase.str.lower() == "a")))
    d["label"] = d.apply(lambda r: (r"$M_G$" if r.meter_id == 1 else f"M{int(r.meter_id)}")+f"-{r.phase.upper()}", axis=1)
    d.to_csv(FIGDATA/"rev_audit.csv", index=False)
    # inverse residuals (long format) + status
    pd.read_csv(WP/"figdata/04_inverse_residuals_long.csv").to_csv(FIGDATA/"rev_inverse_residuals.csv", index=False)
    pd.read_csv(WP/"audits/04_inverse_residuals_by_phase.csv")[["meter_id", "phase", "inverse_status"]].to_csv(FIGDATA/"rev_inverse_status.csv", index=False)
    # held-out validation per window
    pd.read_csv(WP/"audits/26_heldout_window_level.csv").to_csv(FIGDATA/"rev_heldout.csv", index=False)
    # feature selection
    pd.read_csv(FR/"audits/11_feature_ranking.csv")[["rank", "pct_cumul"]].to_csv(FIGDATA/"rev_sel_union.csv", index=False)
    pd.read_csv(FR/"audits/11_global_importance.csv")[["glob_rank", "glob_cumpct"]].to_csv(FIGDATA/"rev_sel_global.csv", index=False)
    pd.DataFrame({"family": ["G_P", "G_sec", "G_I", "G_V", "G_THD", "G_hI", "G_hV"], "universe": [3, 9, 3, 3, 6, 75, 78], "adopted": [3, 9, 3, 3, 6, 7, 17]}).to_csv(FIGDATA/"rev_sel_family.csv", index=False)
    # per-output NAE
    pd.read_csv(FR/"metrics/16_metrics_by_output.csv")[["model", "output", "nae_mean"]].to_csv(FIGDATA/"rev_perout.csv", index=False)
    # predicted x observed (8 outputs with lowest NAE, reference + LSTM 5 seeds)
    yr = np.load(WP/"data/processed/13_Y_raw_test.npy"); dd = pd.read_parquet(FR/"predictions/14_lstm_predictions.parquet")
    labels = [f"{g}_{p}" for g in ["G1", "G2", "G3", "G4", "G5", "G6"] for p in "ABC"]; sel = ["G5_C", "G5_A", "G4_A", "G2_B", "G1_A", "G3_C", "G3_B", "G6_C"]
    rows = {"t": np.arange(989)}
    for o in sel:
        j = labels.index(o); rows[f"obs_{o}"] = yr[6:6+989, j]
        P = np.array([dd[dd.seed == s].sort_values("window_idx")[o].to_numpy(float) for s in sorted(dd.seed.unique())])
        rows[f"mu_{o}"] = P.mean(0); rows[f"sd_{o}"] = P.std(0)
    pd.DataFrame(rows).to_csv(FIGDATA/"rev_predobs.csv", index=False)
    # load-consistency residual per model: signed centre E_R,ABC and
    # the OWN budget of each architecture (u_comb of the ABC block), k_u=2.
    # P_other is NOT part of the residual; R1/R2 do not apply to the current formulation.
    u = pd.read_csv(FR/"dg/17_dg_uncertainty.csv")
    eb = pd.read_csv(FR/"dg/17_dg_energy_by_phase_primary_models.csv")
    eb = eb[(eb.scenario == "R2") & (eb.phase != "ABC")]   # signed centre, 3 phases summed
    mods = ["LSTM", "RCNN_att", "PE_ES", "SPEC", "MoTE_v2", "Ensemble_G4"]
    falta = [m for m in mods if m not in set(eb.model)]
    if falta:
        raise RuntimeError(f"rev_balanco: models missing in the balance: {falta}")
    uc = u[(u.phase == "ABC") & (u.component == "u_total")].set_index("model").u_kWh
    ER = eb.groupby("model").E_DG_pred_kWh.sum()
    pd.DataFrame({"model": mods,
                  "E_R": [float(ER[m]) for m in mods],
                  "U":   [2.0*float(uc[m]) for m in mods]}).to_csv(FIGDATA/"rev_balanco.csv", index=False)
    # gap histogram is already in figdata (26_*)
    print("figdata rev_* prepared.")


# ========================= DRAWING: one PDF per panel =========================
def _lab(m): return r"$M_G$" if m == 1 else f"M{m}"

def draw():
    # 1. coverage (a) presence  (b) split
    pres = np.load(FIGDATA/"rev_cov_presence.npy"); order = json.load(open(FIGDATA/"rev_cov_order.json")); N = pres.shape[1]
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.imshow(pres, aspect="auto", cmap=ListedColormap(["white", OI["blue"]]), interpolation="nearest", extent=[0, N, len(order), 0])
    ax.set_yticks(np.arange(len(order))+0.5); ax.set_yticklabels([_lab(m) for m in order]); ax.set_ylabel("Meter"); ax.set_xlim(0, N); ax.set_xlabel(r"Grid instant")
    for xb, tx in [(8000, "val."), (9000, "test")]:
        ax.axvline(xb, color="k", lw=1.6); ax.text(xb+80, len(order)-0.4, tx, rotation=90, va="bottom", ha="left")
    ax.axvline(6124, color=OI["verm"], lw=1.4, ls=":"); _save(fig, "fig_resultado_cov_a.pdf")
    fig, ax = plt.subplots(figsize=(16, 2.4))
    for s, e, c, l in [(0, 8000, OI["green"], "train"), (8000, 9000, OI["orange"], "val"), (9000, 10000, OI["verm"], "test")]:
        ax.axvspan(s, e, color=c, alpha=0.85); ax.text((s+e)/2, 0.5, l, ha="center", va="center", color="white")
    for b in [8000, 9000]: ax.axvline(b, color="k", lw=1.4)
    ax.set_yticks([]); ax.set_xlim(0, N); ax.set_xlabel(r"Grid instant ($N=10\,000$, $W=12$)"); _save(fig, "fig_resultado_cov_b.pdf")

    # 2. audit (a) coherence (b) eligibility, TRANSPOSED (48 on the x axis)
    d = pd.read_csv(FIGDATA/"rev_audit.csv")
    rows = [("* "+l if s else l) for l, s in zip(d.label, d.star)]
    def tf(x): return x.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"]).astype(float)
    def cc(series):
        v = series.astype(float).clip(0, 1).values.copy(); v[d.dead.values] = np.nan; return v
    A = np.column_stack([cc(d.resid_vi_before.fillna(0)), cc(d.resid_pcos_before.fillna(0)), cc(np.abs(d.resid_tri_before.fillna(0))), cc(d.tri_viol_frac.fillna(0)), cc(d.reverse_frac.fillna(0)), cc(d.cos_oob_frac.fillna(0))])
    colsA = [r"$\mathrm{med}(\varepsilon_{VI})$", r"$\mathrm{med}(\varepsilon_{P})$", r"$\mathrm{med}(\varepsilon_{\mathrm{tri}})$", r"$f(\varepsilon_{\mathrm{tri}}{>}0.10)$", r"$r_{\mathrm{rev}}$", r"$\cos\varphi$ oob"]
    B = np.column_stack([(d.n_real.fillna(0)/10000).clip(0, 1), tf(d.scale_suspect_flag), tf(d.requires_sensitivity), (d.channel_class == "DEAD_OR_STUCK").astype(float)])
    colsB = ["oper.\\ coverage", "scale susp.", "req.\\ sens.", "inactive"]
    for M, cols, nm in [(A.T, colsA, "fig_resultado_audit_a.pdf"), (B.T, colsB, "fig_resultado_audit_b.pdf")]:
        fig, ax = plt.subplots(figsize=(18, 4.6)); cm = plt.get_cmap(SEQ).copy(); cm.set_bad("0.85")
        im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cm, vmin=0, vmax=1)
        ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols); ax.set_xticks(range(len(rows))); ax.set_xticklabels(rows, rotation=90)
        fig.colorbar(im, ax=ax, fraction=0.012, pad=0.01); _save(fig, nm)

    # 3. M2 (a) energy (b) cross-correlation
    ce = pd.read_csv(FIGDATA/"rev_m2_cumenergy.csv"); xc = pd.read_csv(FIGDATA/"rev_m2_xcorr.csv")
    colr = {"A": OI["blue"], "B": OI["orange"], "C": OI["green"]}
    fig, ax = plt.subplots(figsize=(9, 6))
    for ph in "ABC":
        ax.plot(ce[f"MG_{ph}"].dropna(), color=colr[ph], lw=2.2, label=f"$M_G$-{ph}"); ax.plot(ce[f"M2_{ph}"].dropna(), color=colr[ph], lw=2.2, ls="--")
    ax.set_xlabel("Instant (min)"); ax.set_ylabel("Cumulative energy [kWh]"); ax.legend(); _save(fig, "fig_resultado_m2_a.pdf")
    fig, ax = plt.subplots(figsize=(9, 6))
    for ph in "ABC": ax.plot(xc["lag"], xc[ph], color=colr[ph], lw=2.2, label=f"phase {ph}")
    ax.axhline(0, color="0.6", lw=1.0); ax.set_xlabel("Lag [min]"); ax.set_ylabel("Cross-correlation"); ax.set_ylim(-1, 1); ax.legend(); _save(fig, "fig_resultado_m2_b.pdf")

    # 4. inverse residuals: 3 PDFs
    dl = pd.read_csv(FIGDATA/"rev_inverse_residuals.csv")
    exc = [(16, "A"), (16, "B"), (16, "C"), (1, "A"), (2, "A"), (12, "A")]; labels = [_lab(m)+f"-{p}" for m, p in exc]
    for t, tl, nm in [("vi", r"$\varepsilon_{VI}$", "a"), ("pcos", r"$\varepsilon_{P}$", "b"), ("tri", r"$\varepsilon_{\mathrm{tri}}$", "c")]:
        bef = [max(dl[(dl.meter_id == m) & (dl.phase == p) & (dl.term == t) & (dl.stage == "before")].residual.iloc[0], 1e-4) for m, p in exc]
        aft = [max(dl[(dl.meter_id == m) & (dl.phase == p) & (dl.term == t) & (dl.stage == "after")].residual.iloc[0], 1e-4) for m, p in exc]
        x = np.arange(len(exc)); w = 0.38; fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(x-w/2, bef, w, color=OI["grey"], label="identity", edgecolor="k", lw=0.5)
        ax.bar(x+w/2, aft, w, color=OI["blue"], label="post-fit", edgecolor="k", lw=0.5)
        ax.axhline(0.05, color=OI["verm"], lw=1.5, ls="--"); ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_ylim(8e-5, 4)
        ax.set_ylabel(r"Relative residual (median)"); ax.set_xlabel(tl)
        if nm == "a": ax.legend(loc="upper left")
        _save(fig, f"fig_resultado_02_{nm}.pdf")

    # 5. inverse status (1 PDF)
    ds = pd.read_csv(FIGDATA/"rev_inverse_status.csv")
    smap = {"COMPATIBLE": (0, "C", OI["green"]), "ALERT": (1, "A", OI["orange"]), "INCOMPATIBLE_PERSISTENT": (2, "P", OI["verm"]), "DEAD_OR_INSUFFICIENT": (3, "I", OI["grey"])}
    meters = [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]; M = np.full((3, 16), np.nan); L = np.empty_like(M, object); cnt = {k: 0 for k in smap}
    for j, m in enumerate(meters):
        for i, p in enumerate("ABC"):
            rr = ds[(ds.meter_id == m) & (ds.phase == p)]
            if len(rr): st = rr.inverse_status.iloc[0]; M[i, j] = smap[st][0]; L[i, j] = smap[st][1]; cnt[st] += 1
    fig, ax = plt.subplots(figsize=(16, 3.8))
    ax.imshow(M, aspect="auto", cmap=ListedColormap([smap[k][2] for k in ["COMPATIBLE", "ALERT", "INCOMPATIBLE_PERSISTENT", "DEAD_OR_INSUFFICIENT"]]), vmin=0, vmax=3)
    for i in range(3):
        for j in range(16):
            if isinstance(L[i, j], str): ax.text(j, i, L[i, j], ha="center", va="center", color="white")
    ax.set_xticks(range(16)); ax.set_xticklabels([_lab(m) for m in meters], rotation=45, ha="right"); ax.set_yticks(range(3)); ax.set_yticklabels([f"Phase {p}" for p in "ABC"])
    leg = [Patch(facecolor=smap[k][2], edgecolor="k", label=f"{smap[k][1]}: {lb} ({cnt[k]})") for k, lb in [("COMPATIBLE", "compatible"), ("ALERT", "alert"), ("INCOMPATIBLE_PERSISTENT", "persistent"), ("DEAD_OR_INSUFFICIENT", "inactive/insuf.")]]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.5), ncol=4); _save(fig, "fig_resultado_03.pdf")

    # 6. held-out (a) R_A (b) R_f
    w = pd.read_csv(FIGDATA/"rev_heldout.csv"); sizes = [60, 180, 360, 720]; slab = ["1~h", "3~h", "6~h", "12~h"]
    methods = [("fractal", "Analog", OI["blue"]), ("linear", "Linear", OI["orange"]), ("mlp", "MLP", OI["green"])]
    for metric, ylim, nm in [("RA", (0, 2.2), "a"), ("Rf", (-0.05, 1.35), "b")]:
        fig, ax = plt.subplots(figsize=(9, 6)); pos = 0; xt = []; xtl = []
        for W in sizes:
            for mk, ml, mc in methods:
                bp = ax.boxplot(w[w.janela_min == W][f"{metric}_{mk}"].to_numpy(float), positions=[pos], widths=0.7, patch_artist=True, showfliers=False, medianprops=dict(color="k", lw=1.6)); bp["boxes"][0].set_facecolor(mc); bp["boxes"][0].set_alpha(0.85); pos += 1
            xt.append(pos-2); xtl.append(slab[sizes.index(W)]); pos += 1
        ax.axhline(1.0, color="0.4", lw=1.3, ls="--"); ax.set_xticks(xt); ax.set_xticklabels(xtl); ax.set_ylim(*ylim); ax.set_xlabel("Window size"); ax.set_ylabel(r"$R_A$" if metric == "RA" else r"$R_f$")
        if nm == "a": ax.legend(handles=[Patch(facecolor=mc, alpha=0.85, label=ml) for _, ml, mc in methods], loc="upper right")
        _save(fig, f"fig_resultado_07_{nm}.pdf")

    # 7. selection (a) union (b) global (c) family
    u = pd.read_csv(FIGDATA/"rev_sel_union.csv"); g = pd.read_csv(FIGDATA/"rev_sel_global.csv"); fm = pd.read_csv(FIGDATA/"rev_sel_family.csv")
    fig, ax = plt.subplots(figsize=(8, 6)); ax.plot(u["rank"], u["pct_cumul"], color=OI["blue"], lw=2.4)
    for nn, c in [(8, OI["verm"]), (19, OI["orange"]), (29, "0.4")]: ax.axvline(nn, color=c, ls="--", lw=1.5)
    ax.plot(8, u["pct_cumul"].iloc[7], "o", color=OI["verm"], ms=10); ax.set_xlabel("Number of features"); ax.set_ylabel(r"Cumulative importance [\%]"); ax.set_ylim(0, 102); _save(fig, "fig_resultado_selfeat_a.pdf")
    fig, ax = plt.subplots(figsize=(8, 6)); ax.plot(g["glob_rank"], g["glob_cumpct"], color=OI["green"], lw=2.4); ax.axvline(48, color=OI["verm"], ls="--", lw=1.5); ax.plot(48, g["glob_cumpct"].iloc[47], "o", color=OI["verm"], ms=10)
    ax.set_xlabel("Number of features"); ax.set_ylabel(r"Cumulative importance [\%]"); ax.set_ylim(0, 102); _save(fig, "fig_resultado_selfeat_b.pdf")
    ftex = {"G_P": r"$\mathcal{G}_P$", "G_sec": r"$\mathcal{G}_{sec}$", "G_I": r"$\mathcal{G}_I$", "G_V": r"$\mathcal{G}_V$", "G_THD": r"$\mathcal{G}_{THD}$", "G_hI": r"$\mathcal{G}_{hI}$", "G_hV": r"$\mathcal{G}_{hV}$"}
    fig, ax = plt.subplots(figsize=(8, 6)); yy = np.arange(len(fm))
    ax.barh(yy+0.2, fm["universe"], 0.4, color=OI["sky"], label="universe $177$"); ax.barh(yy-0.2, fm["adopted"], 0.4, color=OI["green"], label="adopted $48$")
    ax.set_yticks(yy); ax.set_yticklabels([ftex[f] for f in fm.family]); ax.set_xscale("log"); ax.set_xlabel("Number of features (log)"); ax.legend(loc="lower right"); _save(fig, "fig_resultado_selfeat_c.pdf")

    # 8. per-output heatmap (1 PDF, cividis)
    bo = pd.read_csv(FIGDATA/"rev_perout.csv"); labels = [f"{gg}_{p}" for gg in ["G1", "G2", "G3", "G4", "G5", "G6"] for p in "ABC"]
    labtex = [rf"$\Gamma_{{{gg[1]}}}$-{p}" for gg in ["G1", "G2", "G3", "G4", "G5", "G6"] for p in "ABC"]
    om = ["MoTE_v2", "Ensemble_G4", "LSTM", "RCNN_att", "PE_ES", "SPEC"]; onm = {"MoTE_v2": r"MoTE$_{v2}$", "Ensemble_G4": "Ensemble-G4", "LSTM": "LSTM", "RCNN_att": "RCNN-att", "PE_ES": "PE-ES", "SPEC": "SPEC"}
    M = np.array([[bo[(bo.model == m) & (bo.output == o)].nae_mean.mean() for o in labels] for m in om])
    fig, ax = plt.subplots(figsize=(18, 5.2)); im = ax.imshow(M, aspect="auto", cmap=SEQ, vmin=0, vmax=min(15, np.nanpercentile(M, 98)))
    ax.set_xticks(range(18)); ax.set_xticklabels(labtex, rotation=60, ha="right"); ax.set_yticks(range(6)); ax.set_yticklabels([onm[m] for m in om])
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label=r"NAE [\%]"); _save(fig, "fig_resultado_perout.pdf")

    # 9. predicted x observed: ALL predobs panels (article: seed 42,
    # fig_resultado_predobs_*; SI: seeds 123/456/789/1024, fig_si_predobs_s*_*)
    # are generated by fig_predobs_seed.py (INDIVIDUAL prediction per seed, not the
    # mean). rev_predobs.csv (reference target obs_{o}) is still produced in the
    # preparation block above and serves as input to that generator.

    # 10. load-consistency residual per model (1 PDF)
    b = pd.read_csv(FIGDATA/"rev_balanco.csv"); lab = {"MoTE_v2": r"MoTE$_{v2}$", "Ensemble_G4": "Ensemble-G4", "PE_ES": "PE-ES", "SPEC": "SPEC",
           "LSTM": "LSTM", "RCNN_att": "RCNN-att"}
    x = np.arange(len(b)); fig, ax = plt.subplots(figsize=(13, 6))
    ax.axhline(0, color="0.35", lw=1.4, ls="--", zorder=1)
    ax.errorbar(x, b["E_R"], yerr=b["U"], fmt="o", ms=9, color=OI["blue"], ecolor=OI["blue"],
                capsize=7, lw=1.8, mfc="white", mew=2.0, zorder=3,
                label=r"$\overline{E}_{R,ABC}\pm k_u u_{\mathrm{comb}}$ ($k_u{=}2$)")
    ax.set_xticks(x); ax.set_xticklabels([lab[m] for m in b.model])
    ax.set_ylabel(r"Load consistency residual $\overline{E}_{R,ABC}$ [kWh]")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=1, frameon=False)
    _save(fig, "fig_resultado_balanco.pdf")

    # 11. gap-duration histogram (SI, 1 PDF)
    h = pd.read_csv(WP/"figdata/26_gap_duration_histogram.csv"); thr = pd.read_csv(WP/"figdata/26_gap_duration_thresholds.csv")
    fig, ax = plt.subplots(figsize=(11, 6)); ax.bar(h["bin_min"], h["n_blocos"], width=h["bin_max"]-h["bin_min"], align="edge", color=OI["blue"], edgecolor="white", lw=0.3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Gap duration (min)"); ax.set_ylabel("Number of gap blocks")
    cat = sorted(set(int(t) for t in thr.loc[thr.tipo == "categoria", "threshold_min"])); cat = [c for c in cat if (c+1) not in cat]
    for c in cat: ax.axvline(c, color="0.45", lw=1.1, ls=":")
    ho = int(thr.loc[thr.tipo == "heldout_12h", "threshold_min"].iloc[0]); ax.axvline(ho, color=OI["verm"], lw=1.8, ls="--", label=f"held-out limit ({ho} min)")
    ax.legend(loc="upper right"); _save(fig, "figure_S_gap_duration_histogram.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--prep", action="store_true"); a = ap.parse_args()
    if a.prep: prep()
    else: draw()
    print("[gen_review_figures] done.")
