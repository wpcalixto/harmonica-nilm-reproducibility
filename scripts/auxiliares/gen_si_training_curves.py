#!/usr/bin/env python3
"""
Title: gen_si_training_curves.py

File type: Figure-generation script (Supplementary Information; reads pipeline artifacts only)

Purpose:
    Training/validation loss curves: 5 SEPARATE PANELS (one per trainable architecture) +
    shared legend, in ENGLISH, following the protocol of the predobs figures.
    Mean over the 5 seeds per epoch. Source: figdata/23_training_history.csv.
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
from pathlib import Path
import os
import numpy as np, pandas as pd, shutil
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 18, "axes.labelsize": 18, "xtick.labelsize": 18, "ytick.labelsize": 18,
    "legend.fontsize": 18, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False})
if shutil.which("latex") is None: plt.rcParams["text.usetex"] = False
ROOT=str(Path(__file__).resolve().parents[2])  # repository root
FR=ROOT; ART=ROOT+"/figures"; os.makedirs(ART, exist_ok=True)
OI={"train":"#0072B2","val":"#D55E00"}
ORDER=["LSTM","RCNN_att","PE_ES","SPEC","MoTE_v2"]
d=pd.read_csv(FR+"/figdata/23_training_history.csv")
def save(fig,name): fig.savefig(ART+"/"+name,format="pdf",dpi=300); plt.close(fig); print("  fig",name)
for m in ORDER:
    dm=d[d.model==m]; fig,ax=plt.subplots(figsize=(5.2,3.7))
    for sp in ["train","val"]:
        g=dm[dm.split==sp].groupby("epoch").loss.mean()
        ax.plot(g.index,g.values,color=OI[sp],lw=2.0,label=("training" if sp=="train" else "validation"))
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    if m=="LSTM":                       # legend ONLY in the first panel
        ax.legend(frameon=False,loc="center right")
    save(fig,f"fig_si_train_{m}.pdf")
print("max epochs:",{m:int(d[d.model==m].epoch.max()) for m in ORDER})
