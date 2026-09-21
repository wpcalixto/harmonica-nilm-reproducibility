#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: ablation_pilot.py

File type: Auxiliary training script (feature-set ablation; decision on the validation split only)

Purpose:
    ABLATION (pilot/confirmation): the decision is taken ONLY on the VALIDATION split
    (the test split is locked).

    For the feature set FS_SET (environment variable), trains the LSTM baseline (same
    architecture as script 14: 2x BiLSTM(100) + TimeDistributed Dense) x NN_SEEDS, with early
    stopping on val_loss. For each seed it computes:
      - GLOBAL validation NAE (aggregated over the 18 outputs): sum|y_hat - y| / sum|y|
        (primary metric);
      - per-output NAE (stored per output).
    Architecture/hyperparameters are CONSTANT across feature sets (isolates the effect of the
    set). X_test is NEVER used.
    Writes results/ablation/<FS_SET>/{val_per_output.csv, val_global.csv, summary.json}.

    Overrides: NN_SEEDS, NN_EPOCHS, NN_PATIENCE, NN_LR. FS_SET is mandatory.
    Confirmation run = defaults (5 seeds 42,123,456,789,1024; 500 epochs; patience 30).

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
import os, sys, json, time, importlib.util
from pathlib import Path
SCRIPTS_DIR = Path(__file__).resolve().parents[1]   # scripts/ (holds _nn_common and the pipeline stages)
sys.path.insert(0, str(SCRIPTS_DIR))
import numpy as np, pandas as pd
import _nn_common as C

FS = os.environ["FS_SET"]
_spec = importlib.util.spec_from_file_location("m14", SCRIPTS_DIR / "14_train_baselines.py")
m14 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(m14)   # build_lstm, LSTM_BATCH
import keras
from keras.callbacks import EarlyStopping

RES = C.PROJECT_ROOT / "results" / "ablation" / FS
RES.mkdir(parents=True, exist_ok=True)
data = C.load_splits()
loss = C.make_masked_loss(C.phase_groups(data["output_labels"]))
labels = data["output_labels"]
MID = C.MID


def val_nae(model):
    """(global NAE %, per-output NAE % array) on the VALIDATION split, mid-point alignment."""
    preds = model.predict(data["X_val"], verbose=0)
    yv = data["scaler_Y"].inverse_transform(preds[:, MID, :])
    yt = data["Y_raw_val"]; n = min(len(yv), len(yt) - MID)
    yt = yt[MID:MID + n]; yv = yv[:n]
    g = float(np.sum(np.abs(yv - yt)) / np.sum(np.abs(yt)) * 100)          # aggregated GLOBAL
    po = []
    for j in range(yv.shape[1]):
        d = np.sum(np.abs(yt[:, j]))
        po.append(np.nan if d == 0 else float(np.sum(np.abs(yv[:, j] - yt[:, j])) / d * 100))
    return g, np.array(po)


print(f"[{FS}] n_feat={data['n_features']} (MG={data['n_features']-12}) outputs={data['output_dim']} "
      f"seeds={C.SEEDS} epochs_max={C.EPOCHS_MAX} patience={C.PATIENCE}", flush=True)
po_rows, g_rows = [], []
for seed in C.SEEDS:
    C.set_seed(seed)
    model = m14.build_lstm(data["n_features"], data["output_dim"])
    model.compile(optimizer=keras.optimizers.Adam(C.LR), loss=loss)
    t0 = time.time()
    h = model.fit(data["X_train"], data["Y_train"], validation_data=(data["X_val"], data["Y_val"]),
                  batch_size=m14.LSTM_BATCH, epochs=C.EPOCHS_MAX,
                  callbacks=[EarlyStopping(monitor="val_loss", patience=C.PATIENCE,
                                           restore_best_weights=True, verbose=0)], verbose=2)
    g, po = val_nae(model)
    g_rows.append({"fs_set": FS, "seed": seed, "val_global_nae": round(g, 4),
                   "epochs": len(h.history["loss"])})
    for j, lbl in enumerate(labels):
        po_rows.append({"fs_set": FS, "seed": seed, "output": lbl, "val_nae_pct": round(float(po[j]), 4)})
    print(f"[{FS}] seed {seed}: GLOBAL_val_nae={g:.3f}  epochs={len(h.history['loss'])}  {time.time()-t0:.0f}s", flush=True)

pd.DataFrame(po_rows).to_csv(RES / "val_per_output.csv", index=False)
gdf = pd.DataFrame(g_rows); gdf.to_csv(RES / "val_global.csv", index=False)
gv = gdf["val_global_nae"].to_numpy()
n = len(gv); sd = float(np.std(gv, ddof=1)) if n > 1 else float("nan")
piv = pd.DataFrame(po_rows).groupby("output")["val_nae_pct"].mean()   # mean over seeds, per output
summary = {
    "fs_set": FS, "n_mg": int(data["n_features"] - 12), "n_seeds": n,
    "epochs_max": C.EPOCHS_MAX,
    "mean_global_nae": float(np.mean(gv)), "sd_global_nae": sd,
    "se_global_nae": (sd / np.sqrt(n)) if n > 1 else float("nan"),
    "per_seed_global": {int(r["seed"]): float(r["val_global_nae"]) for r in g_rows},
    "worst_output": str(piv.idxmax()), "worst_val_nae": float(piv.max()),
    "M15_G5": {k: float(piv[k]) for k in labels if k.startswith("G5")},
    "M16B_G6_B": float(piv.get("G6_B", float("nan"))),
    "G6_C": float(piv.get("G6_C", float("nan"))),
    "per_output": {k: float(v) for k, v in piv.items()},
}
json.dump(summary, open(RES / "summary.json", "w"), indent=1, ensure_ascii=False)
print(f"[{FS}] SUMMARY mean_global_nae={summary['mean_global_nae']:.3f} "
      f"(SE={summary['se_global_nae']:.3f}) worst={summary['worst_output']}({summary['worst_val_nae']:.3f}) "
      f"G6_C={summary['G6_C']:.3f} M16B={summary['M16B_G6_B']:.3f}", flush=True)
