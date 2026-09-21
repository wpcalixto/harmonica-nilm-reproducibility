#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 13_create_windows_splits.py

File type: Data preprocessing script (pipeline stage 13: windowing and chronological split)

Purpose:
    Windowing and chronological 80/10/10 split.

    Inputs (Block B):
        data/processed/10_original_features.parquet   (N, 177) features of M_G
        data/processed/09_R1.parquet                  Y (scenario R1; the M_G_* columns are
                                                      discarded -> 18 loads in Watts)
        manifests/11_feature_selection_rfecv_svr_kneedle_params.json  48 MG + 12 DG = 60

    Outputs (n_outputs = number of loads of R1, DYNAMIC = 18):
        data/processed/13_X_train.npy  (N_tr, W, 60)
        data/processed/13_Y_train.npy  (N_tr, W, 18)
        data/processed/13_X_val.npy    (N_va, W, 60)
        data/processed/13_Y_val.npy    (N_va, W, 18)
        data/processed/13_X_test.npy   (N_te, W, 60)
        data/processed/13_Y_test.npy   (N_te, W, 18)
        data/processed/13_Y_raw_test.npy  (n_test_ts, 18) Watts
        data/processed/13_scaler_X.pkl    MinMaxScaler fitted on the 48 MG features (train)
        data/processed/13_scaler_Y.pkl    MinMaxScaler fitted on the 18 loads (train, Watts)
        audits/13_split_index.csv
        audits/13_window_metadata.csv
        figdata/13_split_timeline.csv
        logs/13_create_windows_splits.log
        manifests/13_create_windows_splits_params.json

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
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # repository root (parent of scripts/)

DATA_DIR      = PROJECT_ROOT / "data" / "processed"
AUDITS_DIR    = PROJECT_ROOT / "audits"
FIGDATA_DIR   = PROJECT_ROOT / "figdata"
LOGS_DIR      = PROJECT_ROOT / "logs"
MANIFESTS_DIR = PROJECT_ROOT / "manifests"

for d in (DATA_DIR, AUDITS_DIR, FIGDATA_DIR, LOGS_DIR, MANIFESTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── logging ────────────────────────────────────────────────────────────────────
log_file = LOGS_DIR / "13_create_windows_splits.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────
W            = 12
STRIDE       = 1
TRAIN_RATIO  = 0.8
VAL_RATIO    = 0.1   # test = 1 - 0.8 - 0.1 = 0.1

# ── helpers ────────────────────────────────────────────────────────────────────

def make_windows(X: np.ndarray, Y: np.ndarray, W: int, stride: int):
    """Sliding windows; returns (n_windows, W, n_cols) arrays."""
    n = X.shape[0]
    starts = range(0, n - W + 1, stride)
    X_out = np.stack([X[i : i + W] for i in starts]).astype(np.float32)
    Y_out = np.stack([Y[i : i + W] for i in starts]).astype(np.float32)
    return X_out, Y_out


# -- 1. feature set (ABLATION): FS_SET selects the candidate set -----------------------
# While the adoption is pending on the ablation, script 13 windows the set requested by FS_SET
# (kneedle_8 | n95_19 | union_29 | reference_48 | all_177). Default = provisional union.
import os
CAND = json.loads((DATA_DIR / "11_candidate_sets.json").read_text())
FS_SET = os.environ.get("FS_SET", "")
if FS_SET not in CAND:
    FS_SET = next(k for k in CAND if k.startswith("union_"))   # default = provisional union
adopted_features   = list(CAND[FS_SET])
N_DG               = 12
dg_features        = [f"dg_unit_{i:02d}" for i in range(1, N_DG + 1)]
all_model_features = adopted_features + dg_features
N_MG               = len(adopted_features)
log.info("Set FS_SET=%s: %d MG + %d DG = %d total", FS_SET, N_MG, N_DG, N_MG + N_DG)

# isolated ablation directory and ORDER-dependent hash of the features
import hashlib
AB_DIR = DATA_DIR / "windows" / FS_SET
AB_DIR.mkdir(parents=True, exist_ok=True)
feature_hash = hashlib.sha1("|".join(all_model_features).encode()).hexdigest()[:12]
log.info("AB_DIR=%s | feature_hash=%s (order included)", AB_DIR, feature_hash)

# ── 2. load data ───────────────────────────────────────────────────────────────
log.info("Loading parquet files...")
X_raw_df = pd.read_parquet(DATA_DIR / "10_original_features.parquet")
Y_raw_df  = pd.read_parquet(DATA_DIR / "09_R1.parquet")

# Normalises the indices to UTC (09 writes tz-aware; 08 writes tz-naive); without this
# the intersection is empty.
X_raw_df.index = pd.to_datetime(X_raw_df.index, utc=True)
Y_raw_df.index = pd.to_datetime(Y_raw_df.index, utc=True)

# Align on common index
common_idx = X_raw_df.index.intersection(Y_raw_df.index)
X_raw_df   = X_raw_df.loc[common_idx]
Y_raw_df   = Y_raw_df.loc[common_idx]

n = len(common_idx)
log.info("Aligned samples: %d", n)

# Build X: 48 adopted MG features + 12 DG zero columns
X_mg  = X_raw_df[adopted_features].values.astype(np.float32)   # (N, 48)
X_dg  = np.zeros((n, N_DG), dtype=np.float32)                  # (N, 12)

# Y: loads in Watts (09_R1 is consolidated: discard the M_G_* columns, which are X)
output_cols = [c for c in Y_raw_df.columns
               if not str(c).startswith("M_G") and c not in ("datetime_read",)]
Y_watts = Y_raw_df[output_cols].values.astype(np.float32)       # (N, 18)
n_outputs = Y_watts.shape[1]
log.info("X_mg=%s  Y_watts=%s  range=[%.1f, %.1f] W",
         X_mg.shape, Y_watts.shape, Y_watts.min(), Y_watts.max())

# ── 3. chronological split at timestep level ───────────────────────────────────
n1 = int(n * TRAIN_RATIO)                        # train end
n2 = int(n * (TRAIN_RATIO + VAL_RATIO))          # val end

X_mg_tr,  X_mg_va,  X_mg_te  = X_mg[:n1],  X_mg[n1:n2],  X_mg[n2:]
Y_watts_tr, Y_watts_va, Y_watts_te = Y_watts[:n1], Y_watts[n1:n2], Y_watts[n2:]

log.info("Timestep split: train: %d  val: %d  test: %d", n1, n2 - n1, n - n2)

# ── 4. fit scalers on train only ───────────────────────────────────────────────
scaler_X = MinMaxScaler()
scaler_X.fit(X_mg_tr)         # fitted on 48 MG train features only

scaler_Y = MinMaxScaler()
scaler_Y.fit(Y_watts_tr)      # fitted on n_outputs Watt train targets

# ── 5. scale ──────────────────────────────────────────────────────────────────
X_mg_tr_s = scaler_X.transform(X_mg_tr).astype(np.float32)
X_mg_va_s = scaler_X.transform(X_mg_va).astype(np.float32)
X_mg_te_s = scaler_X.transform(X_mg_te).astype(np.float32)

X_dg_tr = np.zeros((n1,       N_DG), dtype=np.float32)
X_dg_va = np.zeros((n2 - n1,  N_DG), dtype=np.float32)
X_dg_te = np.zeros((n - n2,   N_DG), dtype=np.float32)

X_tr_s = np.concatenate([X_mg_tr_s, X_dg_tr], axis=1)  # (n1, 60)
X_va_s = np.concatenate([X_mg_va_s, X_dg_va], axis=1)
X_te_s = np.concatenate([X_mg_te_s, X_dg_te], axis=1)

Y_tr_s = scaler_Y.transform(Y_watts_tr).astype(np.float32)
Y_va_s = scaler_Y.transform(Y_watts_va).astype(np.float32)
Y_te_s = scaler_Y.transform(Y_watts_te).astype(np.float32)

log.info("X_tr_s=%s  Y_tr_s=%s", X_tr_s.shape, Y_tr_s.shape)

# ── 6. create sliding windows ─────────────────────────────────────────────────
log.info("Creating windows W=%d, stride=%d...", W, STRIDE)
X_train, Y_train = make_windows(X_tr_s, Y_tr_s, W, STRIDE)
X_val,   Y_val   = make_windows(X_va_s, Y_va_s, W, STRIDE)
X_test,  Y_test  = make_windows(X_te_s, Y_te_s, W, STRIDE)

log.info("Windows: train: %s  val: %s  test: %s",
         X_train.shape, X_val.shape, X_test.shape)

# y_raw_test: unscaled Watts for test period (no windowing — full test time series)
Y_raw_test = Y_watts_te   # (n_test_ts, n_outputs)

# ── 7. save numpy arrays + scalers ────────────────────────────────────────────
log.info("Saving numpy arrays...")
np.save(AB_DIR / "X_train.npy",    X_train)
np.save(AB_DIR / "Y_train.npy",    Y_train)
np.save(AB_DIR / "X_val.npy",      X_val)
np.save(AB_DIR / "Y_val.npy",      Y_val)
np.save(AB_DIR / "X_test.npy",     X_test)
np.save(AB_DIR / "Y_test.npy",     Y_test)
np.save(AB_DIR / "Y_raw_test.npy", Y_raw_test)
# compat.: mirrors the canonical names at the root for the current set (default = provisional union)
for nm, arr in [("13_X_train", X_train), ("13_Y_train", Y_train), ("13_X_val", X_val),
                ("13_Y_val", Y_val), ("13_X_test", X_test), ("13_Y_test", Y_test),
                ("13_Y_raw_test", Y_raw_test)]:
    np.save(DATA_DIR / f"{nm}.npy", arr)

joblib.dump(scaler_X, AB_DIR / "scaler_X.pkl")
joblib.dump(scaler_Y, AB_DIR / "scaler_Y.pkl")
joblib.dump(scaler_X, DATA_DIR / "13_scaler_X.pkl")
joblib.dump(scaler_Y, DATA_DIR / "13_scaler_Y.pkl")
log.info("Scalers saved.")

# ── 8. audits ──────────────────────────────────────────────────────────────────
n_tr_ts = n1
n_va_ts = n2 - n1
n_te_ts = n - n2

split_index = pd.DataFrame([
    {"split": "train", "ts_start": 0,    "ts_end": n1 - 1,  "n_timesteps": n_tr_ts, "n_windows": X_train.shape[0]},
    {"split": "val",   "ts_start": n1,   "ts_end": n2 - 1,  "n_timesteps": n_va_ts, "n_windows": X_val.shape[0]},
    {"split": "test",  "ts_start": n2,   "ts_end": n - 1,   "n_timesteps": n_te_ts, "n_windows": X_test.shape[0]},
])
split_index.to_csv(AUDITS_DIR / "13_split_index.csv", index=False)

window_meta = pd.DataFrame([{
    "W":               W,
    "stride":          STRIDE,
    "n_features":      X_train.shape[2],
    "n_mg_features":   N_MG,
    "n_dg_features":   N_DG,
    "n_outputs":       n_outputs,
    "n_total_ts":      n,
    "n_train_ts":      n_tr_ts,
    "n_val_ts":        n_va_ts,
    "n_test_ts":       n_te_ts,
    "n_train_windows": X_train.shape[0],
    "n_val_windows":   X_val.shape[0],
    "n_test_windows":  X_test.shape[0],
}])
window_meta.to_csv(AUDITS_DIR / "13_window_metadata.csv", index=False)

# -- ablation manifest (isolation + traceability per FS_SET) -----------------------------
split_hash = hashlib.sha1(f"{n}_{n1}_{n2}_{W}_{STRIDE}".encode()).hexdigest()[:12]
ab_manifest = {
    "fs_set": FS_SET,
    "n_mg_features": int(N_MG), "n_aux_features": int(N_DG),
    "n_total_features": int(N_MG + N_DG),
    "feature_names": all_model_features,
    "feature_hash": feature_hash,                  # includes the ORDER of the features
    "split_hash": split_hash,
    "window_shape": [int(x) for x in X_train.shape[1:]],
    "n_train_windows": int(X_train.shape[0]),
    "n_val_windows": int(X_val.shape[0]),
    "n_test_windows": int(X_test.shape[0]),
}
(AB_DIR / "manifest.json").write_text(json.dumps(ab_manifest, indent=1, ensure_ascii=False))
log.info("AB manifest: fs_set=%s feature_hash=%s split_hash=%s window=%s",
         FS_SET, feature_hash, split_hash, ab_manifest["window_shape"])

log.info("Audits saved.")

# -- 9. figdata: split timeline ---------------------------------------------------------
splits_ts = (
    ["train"] * n_tr_ts
    + ["val"]  * n_va_ts
    + ["test"] * n_te_ts
)
timeline_df = pd.DataFrame({
    "ts_idx": np.arange(n),
    "split":  splits_ts,
})
if hasattr(common_idx, "to_series"):
    timeline_df["timestamp"] = common_idx.to_numpy()
timeline_df.to_csv(FIGDATA_DIR / "13_split_timeline.csv", index=False)
log.info("figdata/13_split_timeline.csv saved (%d rows)", len(timeline_df))

# ── 10. manifest ───────────────────────────────────────────────────────────────
from datetime import datetime, timezone

manifest = {
    "script":         "13_create_windows_splits.py",
    "run_timestamp":  datetime.now(timezone.utc).isoformat(),
    "parameters": {
        "W":            W,
        "stride":       STRIDE,
        "train_ratio":  TRAIN_RATIO,
        "val_ratio":    VAL_RATIO,
    },
    "inputs": {
        "X_parquet":         str(DATA_DIR / "10_original_features.parquet"),
        "Y_parquet":         str(DATA_DIR / "09_R1.parquet"),
        "candidate_sets":    str(DATA_DIR / "11_candidate_sets.json"),
        "fs_set":            FS_SET,
        "feature_hash":      feature_hash,
    },
    "results": {
        "n_total_ts":      n,
        "n_train_ts":      n_tr_ts,
        "n_val_ts":        n_va_ts,
        "n_test_ts":       n_te_ts,
        "n_train_windows": int(X_train.shape[0]),
        "n_val_windows":   int(X_val.shape[0]),
        "n_test_windows":  int(X_test.shape[0]),
        "X_shape":         list(X_train.shape),
        "Y_shape":         list(Y_train.shape),
        "n_features":      int(X_train.shape[2]),
        "n_mg_features":   N_MG,
        "n_dg_features":   N_DG,
        "n_outputs":       n_outputs,
        "output_labels":   output_cols,
        "Y_raw_test_shape": list(Y_raw_test.shape),
        "Y_watts_train_range": [float(Y_watts_tr.min()), float(Y_watts_tr.max())],
        "Y_scaled_train_range": [float(Y_tr_s.min()), float(Y_tr_s.max())],
    },
    "outputs": {
        "X_train":    "data/processed/13_X_train.npy",
        "Y_train":    "data/processed/13_Y_train.npy",
        "X_val":      "data/processed/13_X_val.npy",
        "Y_val":      "data/processed/13_Y_val.npy",
        "X_test":     "data/processed/13_X_test.npy",
        "Y_test":     "data/processed/13_Y_test.npy",
        "Y_raw_test": "data/processed/13_Y_raw_test.npy",
        "scaler_X":   "data/processed/13_scaler_X.pkl",
        "scaler_Y":   "data/processed/13_scaler_Y.pkl",
        "split_index":    "audits/13_split_index.csv",
        "window_metadata": "audits/13_window_metadata.csv",
        "split_timeline": "figdata/13_split_timeline.csv",
        "log":        "logs/13_create_windows_splits.log",
        "manifest":   "manifests/13_create_windows_splits_params.json",
    },
}
(MANIFESTS_DIR / "13_create_windows_splits_params.json").write_text(
    json.dumps(manifest, indent=2)
)

log.info("Manifest saved.")
log.info(
    "Done: train=%d  val=%d  test=%d windows  |  X(W,F)=(%d,%d)  Y(W,O)=(%d,%d)",
    X_train.shape[0], X_val.shape[0], X_test.shape[0],
    W, X_train.shape[2], W, n_outputs,
)
