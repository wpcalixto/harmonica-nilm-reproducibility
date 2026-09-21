#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: collect_ablation.py

File type: Auxiliary analysis script (consolidation of the feature-set ablation results)

Purpose:
    Consolidates the ablation (VALIDATION split): mean +/- SE of the global NAE per feature
    set, one-standard-error rule, PAIRED comparison against reference_48, and a per-output GUARD
    (degradation >5% in >=4/5 seeds). Does NOT decide n_adopted automatically: it delivers the
    evidence for the decision. The test split is locked.

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
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "results" / "ablation"  # repository root / results / ablation
SETS = ["kneedle_8", "n95_19", "union_29", "reference_48", "all_177"]
GUARD_OUT = ["G6_C", "G5_A", "G5_B", "G5_C", "G6_B"]   # special attention (M15, M16-B, worst of the pilot)

summ, glob, po = {}, {}, {}
for s in SETS:
    d = ROOT / s
    if not (d / "summary.json").exists():
        continue
    summ[s] = json.load(open(d / "summary.json"))
    glob[s] = pd.read_csv(d / "val_global.csv").set_index("seed")["val_global_nae"]
    po[s] = pd.read_csv(d / "val_per_output.csv").pivot(index="seed", columns="output", values="val_nae_pct")
present = [s for s in SETS if s in summ]
if not present:
    print("No results."); raise SystemExit

rows = [{"conjunto": s, "n_mg": summ[s]["n_mg"],
         "NAE_glob": round(summ[s]["mean_global_nae"], 3), "SE": round(summ[s]["se_global_nae"], 3),
         "pior_saida": summ[s]["worst_output"], "pior": round(summ[s]["worst_val_nae"], 3),
         "G6_C": round(summ[s]["G6_C"], 3), "M16B": round(summ[s]["M16B_G6_B"], 3)} for s in present]
df = pd.DataFrame(rows)
df.to_csv(ROOT / "confirmation_comparison.csv", index=False)
print("=== GLOBAL validation NAE (mean over seeds +/- SE=SD/sqrt(n)) ===")
print(df.to_string(index=False))

# one-standard-error rule on the mean global NAE
best = df.loc[df["NAE_glob"].idxmin()]; lim = float(best["NAE_glob"] + best["SE"])
keep = df[df["NAE_glob"] <= lim].sort_values("n_mg")
print(f"\n=== 1-SE rule: best={best['conjunto']} ({best['NAE_glob']:.3f}); "
      f"limit = {best['NAE_glob']:.3f}+{best['SE']:.3f} = {lim:.3f} ===")
print(f"  within 1-SE: {list(keep['conjunto'])}  | smallest of them: {keep.iloc[0]['conjunto']}")

# PAIRED comparison against reference_48 (same seeds)
if "reference_48" in glob:
    ref = glob["reference_48"]
    print("\n=== Paired delta vs reference_48 (mean over seeds; <0 = better than 48) ===")
    for s in present:
        c = ref.index.intersection(glob[s].index)
        d = glob[s][c] - ref[c]
        print(f"  {s:<13} delta={d.mean():+.3f} (+/-{d.std(ddof=1) if len(d)>1 else float('nan'):.3f})")

# per-output GUARD: a candidate "degrades" an output if it is >5% worse than the BEST set on that output
# in >=4/5 seeds
print("\n=== Per-output guard (>5% worse than the best set, in >=4 of the 5 seeds) ===")
outs = list(po[present[0]].columns)
best_o = {o: min(po[k][o].mean() for k in present) for o in outs}
for s in present:
    deg = [o for o in outs if int((po[s][o] > best_o[o] * 1.05).sum()) >= 4]
    deg_guard = [o for o in deg if o in GUARD_OUT]
    print(f"  {s:<13} degraded={len(deg)} | critical(M15/M16B/G6C)={deg_guard if deg_guard else 'none'}")

print("\nDecision: n_adopted is NOT closed here; combine the 1-SE rule + per-output guard + parsimony.")
print("State: n_K=8  n95=19  n_union=29  n_adopted=PENDING (confirmation delivered).")
