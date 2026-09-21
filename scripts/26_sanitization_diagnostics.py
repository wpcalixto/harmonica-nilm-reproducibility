#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 26_sanitization_diagnostics.py

File type: Diagnostic script (stage 26: sanitisation diagnostics; does not alter the pipeline 01-25)

Purpose:
    LATERAL DIAGNOSTIC (does not alter the pipeline 01-25).

    Reads existing artifacts, reuses the actual filling functions (analog_fill/freq of script
    08) and the validation helpers (script 07) and produces, with prefix 26_ and WITHOUT
    overwriting anything:

      Task 1 (gap-duration histogram):
        audits/26_gap_blocks.csv
        tabdata/26_gap_duration_summary.csv
        figdata/26_gap_duration_histogram.csv
        figdata/26_gap_duration_thresholds.csv

      Task 3 (held-out with 50 individual windows + Wilcoxon/Holm + effect size):
        audits/26_heldout_window_level.csv
        tabdata/26_heldout_summary.csv
        tabdata/26_heldout_wilcoxon_holm.csv
        figdata/26_heldout_window_level.csv

    The 50 windows per duration reproduce the protocol of script 07 (same seed=7, same
    inputs), so that the MEDIANS coincide with 07_fill_validation; this script only preserves
    the PER-WINDOW values (previously discarded) and extends the inference.

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
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # repository root (parent of scripts/)


def _load(prefix_glob):
    cands = sorted(SCRIPT_DIR.glob(prefix_glob))
    if not cands:
        raise FileNotFoundError(f"No {prefix_glob} in {SCRIPT_DIR}")
    spec = importlib.util.spec_from_file_location(cands[-1].stem, cands[-1])
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# Task 1: gap blocks and logarithmic histogram
# ─────────────────────────────────────────────────────────────────────────────
def task_gap_histogram(root: Path, mod07):
    interim = root / "data" / "interim" / "06_para_preencher.csv"
    df = pd.read_csv(interim, parse_dates=[mod07.TIME_COL])
    prof = pd.read_csv(root / "audits" / "07_gap_profile_by_channel.csv")
    alive = set(map(tuple, prof.loc[~prof["morto"], [mod07.METER_COL, "variavel"]].values.tolist()))
    elec = [c for c in df.columns if c not in (mod07.METER_COL, mod07.TIME_COL)]
    blocks = []
    for m, g in df.groupby(mod07.METER_COL):
        g = g.sort_values(mod07.TIME_COL)
        for c in elec:
            if (m, c) not in alive:
                continue
            v = g[c].to_numpy(float)
            isnan = ~np.isfinite(v)
            if not isnan.any():
                continue
            # contiguous NaN runs
            idx = np.flatnonzero(np.diff(np.r_[0, isnan.view(np.int8), 0]))
            for s, e in zip(idx[0::2], idx[1::2]):
                blocks.append((int(m), c, int(s), int(e - s)))  # duration in min (1-min grid)
    gb = pd.DataFrame(blocks, columns=["meter_id", "variavel", "inicio_idx", "duracao_min"])
    gb.to_csv(root / "audits" / "26_gap_blocks.csv", index=False)

    d = gb["duracao_min"].to_numpy()
    summ = pd.DataFrame([{
        "n_blocos": len(d), "n_pontos": int(d.sum()),
        "min": int(d.min()), "p25": float(np.percentile(d, 25)),
        "mediana": float(np.median(d)), "p75": float(np.percentile(d, 75)),
        "p95": float(np.percentile(d, 95)), "max": int(d.max()),
        "media": float(d.mean()),
    }])
    summ.to_csv(root / "tabdata" / "26_gap_duration_summary.csv", index=False)

    # log histogram (log10 bins from 1 min to the maximum)
    dmax = int(d.max())
    edges = np.unique(np.round(np.logspace(0, np.log10(dmax + 1), 40)).astype(int))
    counts, _ = np.histogram(d, bins=edges)
    hist = pd.DataFrame({"bin_min": edges[:-1], "bin_max": edges[1:], "n_blocos": counts})
    hist.to_csv(root / "figdata" / "26_gap_duration_histogram.csv", index=False)

    # dynamic thresholds: finite category bounds + held-out limit (720)
    gd = pd.read_csv(root / "audits" / "07_gap_size_distribution.csv")
    cat_bounds = []
    for fx in gd["faixa_min"].astype(str):
        for tok in fx.replace("≥", "").replace(">=", "").split("-"):
            tok = tok.strip()
            if tok.isdigit():
                cat_bounds.append(int(tok))
    cat_bounds = sorted(set(b for b in cat_bounds if b > 1))
    thr = pd.DataFrame({"threshold_min": cat_bounds + [720],
                        "tipo": ["categoria"] * len(cat_bounds) + ["heldout_12h"]})
    thr.drop_duplicates("threshold_min").to_csv(root / "figdata" / "26_gap_duration_thresholds.csv", index=False)
    print(f"[T1] blocks={len(gb):,} points={int(d.sum()):,} max={dmax} min | thresholds={cat_bounds}+[720]")


# ─────────────────────────────────────────────────────────────────────────────
# Task 3: per-window held-out + Wilcoxon/Holm + effect size
# ─────────────────────────────────────────────────────────────────────────────
def _rank_biserial(x, y):
    """Rank-biserial correlation (paired Wilcoxon): (R+ - R-)/(R+ + R-)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = pd.Series(np.abs(d)).rank().to_numpy()
    rp = r[d > 0].sum(); rn = r[d < 0].sum()
    tot = rp + rn
    return float((rp - rn) / tot) if tot else 0.0


def _holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p)
    m = len(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        run = max(run, val); adj[i] = min(run, 1.0)
    return adj


def task_heldout(root: Path, mod07, mod08):
    from scipy.stats import ks_2samp, wilcoxon
    interim = root / "data" / "interim" / "06_para_preencher.csv"
    df = pd.read_csv(interim, parse_dates=[mod07.TIME_COL])
    prof = pd.read_csv(root / "audits" / "07_gap_profile_by_channel.csv")
    # gap rule for analog_fill (identical to 07.run_validation)
    rc = mod08.DEFAULT_GAP_RULE_CSV
    if not rc.exists():
        rc = rc.with_name("07_gap_rule.csv")
    mod08.GAP_RULE = mod08.load_gap_rule(rc)

    rng = np.random.default_rng(mod07.VAL_SEED)
    alive = prof[~prof["morto"]][[mod07.METER_COL, "variavel"]].values.tolist()
    cache = {}

    def get_series(m, c):
        key = (m, c)
        if key not in cache:
            g = df[df[mod07.METER_COL] == m].sort_values(mod07.TIME_COL)
            cache[key] = g[c].values.astype(float)
        return cache[key]

    methods = ["fractal", "linear", "mlp"]
    win_rows = []
    for W in mod07.VAL_WSIZES:
        got, tries = 0, 0
        while got < mod07.VAL_N and tries < mod07.VAL_N * 80:
            tries += 1
            m, c = alive[int(rng.integers(len(alive)))]
            v = get_series(m, c); n = len(v)
            if n < 3 * W + 2:
                continue
            a = int(rng.integers(W, n - 2 * W))
            win = v[a : a + W]
            if np.isfinite(win).sum() < mod07.VAL_MIN_OBS_FRAC * W:
                continue
            if not (np.isfinite(v[a - 1]) and np.isfinite(v[a + W])):
                continue
            truth = win.copy(); vt = v.copy(); vt[a : a + W] = np.nan
            fills = {"fractal": mod08.analog_fill(vt, rng)[0][a : a + W],
                     "linear": mod07._fill_linear(vt, a, W),
                     "mlp": mod07._fill_mlp(vt, a, W)}
            sts = {meth: mod07._val_stats(fills[meth], truth, mod08.freq) for meth in methods}
            if any(sts[meth] is None for meth in methods):
                continue
            fin = np.isfinite(truth)
            rec = {"janela_h": W // 60, "janela_min": W, "meter_id": int(m),
                   "variavel": c, "inicio": int(a)}
            for meth in methods:
                s = sts[meth]
                rec[f"RA_{meth}"] = s["amp"]; rec[f"Rf_{meth}"] = s["freq"]
                rec[f"RM_{meth}"] = s["energy"]; rec[f"MAE_{meth}"] = s["mae"]
                rec[f"ks_{meth}"] = ks_2samp(np.asarray(fills[meth])[fin], truth[fin]).pvalue
            win_rows.append(rec); got += 1
    W_df = pd.DataFrame(win_rows)
    W_df.to_csv(root / "audits" / "26_heldout_window_level.csv", index=False)
    W_df.to_csv(root / "figdata" / "26_heldout_window_level.csv", index=False)

    # summary per window x method (median/IQR/percentiles/%ks<0.05)
    srows = []
    for W in sorted(W_df["janela_min"].unique()):
        sub = W_df[W_df["janela_min"] == W]
        for meth in methods:
            for metric in ("RA", "Rf", "RM", "MAE"):
                x = sub[f"{metric}_{meth}"].to_numpy(float)
                srows.append({"janela_min": int(W), "metodo": meth, "metrica": metric,
                              "n": len(x), "mediana": np.median(x),
                              "q1": np.percentile(x, 25), "q3": np.percentile(x, 75),
                              "iqr": np.percentile(x, 75) - np.percentile(x, 25),
                              "p5": np.percentile(x, 5), "p95": np.percentile(x, 95)})
            ks = sub[f"ks_{meth}"].to_numpy(float)
            srows.append({"janela_min": int(W), "metodo": meth, "metrica": "ks_p",
                          "n": len(ks), "mediana": np.median(ks), "q1": np.percentile(ks, 25),
                          "q3": np.percentile(ks, 75), "iqr": np.percentile(ks, 75) - np.percentile(ks, 25),
                          "p5": np.percentile(ks, 5), "p95": np.percentile(ks, 95),
                          "frac_p_lt_0_05": float(np.mean(ks < 0.05))})
    pd.DataFrame(srows).round(4).to_csv(root / "tabdata" / "26_heldout_summary.csv", index=False)

    # paired Wilcoxon fractal vs {linear, mlp} on |R-1| and MAE, Holm + effect size
    wrows = []
    for W in sorted(W_df["janela_min"].unique()):
        sub = W_df[W_df["janela_min"] == W]
        for base in ("linear", "mlp"):
            block = []
            for metric, dev in (("RA", True), ("Rf", True), ("RM", True), ("MAE", False)):
                fa = sub[f"{metric}_fractal"].to_numpy(float)
                bl = sub[f"{metric}_{base}"].to_numpy(float)
                xf = np.abs(fa - 1) if dev else fa
                xb = np.abs(bl - 1) if dev else bl
                try:
                    p = wilcoxon(xf, xb, zero_method="wilcox").pvalue
                except Exception:
                    p = np.nan
                block.append({"janela_min": int(W), "comparacao": f"fractal_vs_{base}",
                              "metrica": (f"|{metric}-1|" if dev else metric),
                              "mediana_fractal": float(np.median(xf)),
                              "mediana_base": float(np.median(xb)),
                              "delta_mediano": float(np.median(xf) - np.median(xb)),
                              "p_wilcoxon": p, "rank_biserial": _rank_biserial(xf, xb)})
            padj = _holm([b["p_wilcoxon"] for b in block])
            for b, pa in zip(block, padj):
                b["p_holm"] = float(pa); wrows.append(b)
    pd.DataFrame(wrows).round(5).to_csv(root / "tabdata" / "26_heldout_wilcoxon_holm.csv", index=False)

    # check: do the medians reproduce 07_fill_validation?
    ref = pd.read_csv(root / "audits" / "07_fill_validation.csv")
    chk = W_df.groupby("janela_min").apply(
        lambda s: pd.Series({"RA_fractal_med": s["RA_fractal"].median()})).reset_index()
    print("[T3] windows:", dict(W_df.groupby("janela_min").size()),
          "| RA_fractal medians:", [round(x, 4) for x in chk["RA_fractal_med"]])
    print("     (compare with 07_fill_validation fractal razao_amplitude:",
          ref[ref.metodo == "fractal"]["razao_amplitude"].tolist(), ")")


def task_confidence(root: Path):
    """Reconciles the confidence mask (already existing in 08_fill_confidence) per chronological
    partition. Does NOT recompute the mask: only reads the artifact and aggregates."""
    d = pd.read_parquet(root / "data" / "processed" / "08_fill_confidence.parquet")
    ec = [c for c in d.columns if c not in ("meter_id", "time")]
    t = pd.to_datetime(d["time"], utc=True); t0 = t.min().floor("min")
    gi = ((t - t0).dt.total_seconds() // 60).astype(int).to_numpy()
    M = d[ec].to_numpy()
    # 0 OBSERVED,1 SHORT,2 MEDIUM,3 LONG_LOWCONF,4 CAT4_LOWCONF,5 DEAD,6 UNSUPPORTED
    cls = {"OBSERVED": [0], "SHORT": [1], "MEDIUM_SUPPORTED": [2],
           "LONG_LOWCONF": [3, 4], "UNSUPPORTED_BACKBONE": [6]}
    splits = {"global": (0, 10000), "train": (0, 8000), "validation": (8000, 9000), "test": (9000, 10000)}
    rows = []
    for name, (a, b) in splits.items():
        sub = M[(gi >= a) & (gi < b)]
        vals = sub[np.isfinite(sub)].astype(int)
        u, c = np.unique(vals, return_counts=True); cc = dict(zip(u.tolist(), c.tolist()))
        tot = sum(cc.get(k, 0) for grp in cls.values() for k in grp)
        for cname, codes in cls.items():
            n = sum(cc.get(k, 0) for k in codes)
            rows.append({"split": name, "classe": cname, "n": n,
                         "pct": round(100 * n / tot, 3) if tot else 0.0, "n_total": tot})
    by_split = pd.DataFrame(rows)
    by_split.to_csv(root / "tabdata" / "26_confidence_by_split.csv", index=False)
    by_split.to_csv(root / "figdata" / "26_confidence_by_split.csv", index=False)
    # reconciliation: sum of the partitions == global, per class
    rec = []
    for cname in cls:
        g = by_split[(by_split.split == "global") & (by_split.classe == cname)]["n"].iloc[0]
        s = by_split[(by_split.split.isin(["train", "validation", "test"])) & (by_split.classe == cname)]["n"].sum()
        rec.append({"classe": cname, "global": int(g), "soma_particoes": int(s), "fecha": bool(g == s)})
    pd.DataFrame(rec).to_csv(root / "audits" / "26_confidence_reconciliation.csv", index=False)
    ok = all(r["fecha"] for r in rec)
    print(f"[T2] mask reconciled per partition | all classes close: {ok}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(PROJECT_ROOT))
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--run-gap-histogram", action="store_true")
    ap.add_argument("--run-heldout", action="store_true")
    ap.add_argument("--run-confidence", action="store_true")
    args = ap.parse_args()
    root = Path(args.project_root)
    do_all = args.run_all or not (args.run_gap_histogram or args.run_heldout or args.run_confidence)
    mod07 = _load("07_*.py"); mod08 = _load("08_*.py")
    if do_all or args.run_gap_histogram:
        task_gap_histogram(root, mod07)
    if do_all or args.run_confidence:
        task_confidence(root)
    if do_all or args.run_heldout:
        task_heldout(root, mod07, mod08)
    print("[26] diagnostics done (prefix 26_, no overwrite).")


if __name__ == "__main__":
    main()
