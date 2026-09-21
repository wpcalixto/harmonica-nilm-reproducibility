#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: _nn_common.py

File type: Shared utility module imported by the training scripts (14, 15, 18) and by the auxiliary training scripts; not an executable pipeline stage

Purpose:
    Shared utilities of the Block B trainers.

    Centralises what the training scripts have in common:
      - constants (SEEDS, W, MID, EPOCHS_MAX, PATIENCE), with environment overrides (smoke test);
      - set_seed, nae_percent;
      - load_splits(): reads the 13_*.npy arrays + 13_scaler_Y.pkl + the output labels from the
        manifest; defines n_features and output_dim DYNAMICALLY;
      - phase_groups(): groups of 3 phases per load, derived from the labels;
      - make_masked_loss(): factory of masked_mae_phase_constraint for the actual number of outputs;
      - train_and_evaluate(): seed loop + checkpoint + metrics/predictions/figdata.

    TensorFlow/Keras imports are DEFERRED (inside the functions) so that the scaffolding
    without TF (load_splits, phase_groups, nae_percent) is importable/testable without TF.

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

import json
import os
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # repository root (parent of scripts/)
DATA_DIR      = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"
PREDS_DIR     = PROJECT_ROOT / "predictions"
METRICS_DIR   = PROJECT_ROOT / "metrics"
FIGDATA_DIR   = PROJECT_ROOT / "figdata"
MANIFESTS_DIR = PROJECT_ROOT / "manifests"
AUDITS_DIR    = PROJECT_ROOT / "audits"      # shared (not redirected by FS_SET)
CKPT_ROOT     = PROJECT_ROOT / "checkpoints"

# ABLATION mode: FS_SET isolates windows/models/predictions/metrics per feature set.
# Without FS_SET (empty) -> normal pipeline (canonical 13_*.npy names, models/ root).
_FS = os.environ.get("FS_SET", "")
if _FS:
    MODELS_DIR  = MODELS_DIR / "ablation" / _FS
    PREDS_DIR   = PREDS_DIR / "ablation" / _FS
    METRICS_DIR = METRICS_DIR / "ablation" / _FS

# Constants (environment overrides for smoke tests in an environment with TF):
W          = 12
MID        = W // 2
SEEDS      = [int(s) for s in os.environ.get("NN_SEEDS", "42,123,456,789,1024").split(",")]
EPOCHS_MAX = int(os.environ.get("NN_EPOCHS", "500"))
PATIENCE   = int(os.environ.get("NN_PATIENCE", "30"))
LR         = float(os.environ.get("NN_LR", "1e-3"))


def set_seed(seed: int) -> None:
    import keras
    os.environ["PYTHONHASHSEED"] = str(seed)
    keras.utils.set_random_seed(seed)   # Keras 3: numpy + tf + random


def nae_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_pred - y_true)) / denom * 100)


def load_splits() -> dict:
    """Loads the 13_*.npy arrays + scaler_Y + output labels from the manifest."""
    keys = ["X_train", "Y_train", "X_val", "Y_val", "X_test", "Y_test", "Y_raw_test"]
    if _FS:                                            # ablation: windows/<FS_SET>/*.npy
        wdir = DATA_DIR / "windows" / _FS
        d = {k: np.load(wdir / f"{k}.npy") for k in keys}
        d["scaler_Y"] = joblib.load(wdir / "scaler_Y.pkl")
    else:                                              # normal pipeline (canonical windows of script 13)
        d = {k: np.load(DATA_DIR / f"13_{k}.npy") for k in keys}
        d["scaler_Y"] = joblib.load(DATA_DIR / "13_scaler_Y.pkl")
    d["n_features"] = int(d["X_train"].shape[-1])     # 60
    d["output_dim"] = int(d["Y_train"].shape[-1])     # 18 (DYNAMIC)
    # VALIDATION targets in Watts (for the selection of W* and of the SPEC loads; NEVER test):
    # un-window the validation series and undo the scaling.
    d["Y_raw_val"] = d["scaler_Y"].inverse_transform(unwindow(d["Y_val"])).astype("float32")
    man = json.loads((MANIFESTS_DIR / "13_create_windows_splits_params.json").read_text())
    labels = man.get("results", {}).get("output_labels")
    d["output_labels"] = labels if labels else [f"out_{i}" for i in range(d["output_dim"])]
    return d


def unwindow(arr_w: np.ndarray) -> np.ndarray:
    """Rebuilds the base series (ts, feat) from stride-1 windows (N, W, feat).
    Exact for stride=1: first step of every window + remaining steps of the last one."""
    return np.vstack([arr_w[:, 0, :], arr_w[-1, 1:, :]])


def rewindow(X_ts: np.ndarray, Y_ts: np.ndarray, W: int, stride: int = 1):
    """Sliding windows (N, W, feat) from the base series (ts, feat)."""
    n = (len(X_ts) - W) // stride + 1
    Xw = np.stack([X_ts[i:i + W] for i in range(0, n * stride, stride)]).astype(np.float32)
    Yw = np.stack([Y_ts[i:i + W] for i in range(0, n * stride, stride)]).astype(np.float32)
    return Xw, Yw


def mid_align(preds_w: np.ndarray, W: int):
    """Rebuilds the predicted series by the mid-point of each window (offset = W//2).
    Returns (predicted_series (N, out), offset). The aligned reference series is
    y_raw[offset : offset + N]."""
    mid = W // 2
    return preds_w[:, mid, :], mid


def phase_groups(labels: list) -> list:
    """Indices grouped by load (prefix G{n}); only groups with 3 phases enter the
    inter-phase penalty. Generic: derived from the labels, not hard-coded."""
    groups: dict = {}
    for i, lab in enumerate(labels):
        g = str(lab).split("_")[0]
        groups.setdefault(g, []).append(i)
    return [idx for idx in groups.values() if len(idx) == 3]


def make_masked_loss(phase_groups_idx: list):
    """Factory of masked_mae_phase_constraint for the actual number of outputs."""
    import tensorflow as tf
    import keras

    @keras.saving.register_keras_serializable(package="Custom")
    def masked_mae_phase_constraint(y_true, y_pred):
        mask = tf.cast(
            tf.reduce_any(tf.not_equal(y_true, 0.0), axis=-1, keepdims=True),
            tf.float32,
        )
        mae = tf.reduce_mean(mask * tf.abs(y_pred - y_true))
        pen = tf.constant(0.0)
        for sl in phase_groups_idx:
            y_load = tf.gather(y_pred, sl, axis=-1)        # (batch, W, 3)
            pen = pen + tf.reduce_mean(tf.math.reduce_std(y_load, axis=-1))
        if phase_groups_idx:
            pen = (pen / len(phase_groups_idx)) * 0.05
        return mae + pen

    return masked_mae_phase_constraint


def make_masked_loss_weighted(phase_groups_idx: list, weight_idx: list, weight: float = 5.0):
    """Like make_masked_loss, but with extra weight (SPEC) on the outputs in weight_idx."""
    import tensorflow as tf
    import keras
    n_out = None  # weight built on the first call (output_dim is known from y_true)

    @keras.saving.register_keras_serializable(package="Custom")
    def masked_mae_spec(y_true, y_pred):
        w = tf.ones_like(y_true[..., :])
        if weight_idx:
            upd = tf.tensor_scatter_nd_update(
                tf.ones([tf.shape(y_true)[-1]]),
                [[i] for i in weight_idx],
                [weight] * len(weight_idx))
            w = w * upd
        mask = tf.cast(tf.reduce_any(tf.not_equal(y_true, 0.0), axis=-1, keepdims=True), tf.float32)
        mae = tf.reduce_mean(mask * tf.abs(y_pred - y_true) * w)
        pen = tf.constant(0.0)
        for sl in phase_groups_idx:
            pen = pen + tf.reduce_mean(tf.math.reduce_std(tf.gather(y_pred, sl, axis=-1), axis=-1))
        if phase_groups_idx:
            pen = (pen / len(phase_groups_idx)) * 0.05
        return mae + pen

    return masked_mae_spec


def per_output_nae_val(model, X_val_w, Y_raw_val, W: int, scaler_Y, mid=None) -> np.ndarray:
    """NAE (%) per output on the VALIDATION split (mid-point alignment)."""
    mid = (W // 2) if mid is None else mid
    preds = model.predict(X_val_w, verbose=0)
    yv = scaler_Y.inverse_transform(preds[:, mid, :])
    n = min(len(yv), len(Y_raw_val) - mid)
    yt = Y_raw_val[mid:mid + n]
    yv = yv[:n]
    out = []
    for j in range(yv.shape[1]):
        den = np.sum(np.abs(yt[:, j]))
        out.append(np.nan if den == 0 else float(np.sum(np.abs(yv[:, j] - yt[:, j])) / den * 100))
    return np.array(out, dtype=float)


def select_W_on_val(build_fn_W, data, grid, batch, log, epochs=None, patience=None, seed=42):
    """Selects W* by the LOWEST mean VALIDATION NAE (never test).
    build_fn_W(W, n_feat, n_out) -> compiled model. Trains 1 model per W of the grid,
    with the SAME seed (fair comparison)."""
    import keras
    from keras.callbacks import EarlyStopping
    epochs = EPOCHS_MAX if epochs is None else epochs
    patience = PATIENCE if patience is None else patience
    Xtr_ts, Ytr_ts = unwindow(data["X_train"]), unwindow(data["Y_train"])
    Xv_ts, Yv_ts = unwindow(data["X_val"]), unwindow(data["Y_val"])
    res = {}
    for W in grid:
        set_seed(seed)
        Xtr, Ytr = rewindow(Xtr_ts, Ytr_ts, W)
        Xv, Yv = rewindow(Xv_ts, Yv_ts, W)
        m = build_fn_W(W, data["n_features"], data["output_dim"])
        m.fit(Xtr, Ytr, validation_data=(Xv, Yv), batch_size=batch, epochs=epochs,
              callbacks=[EarlyStopping(monitor="val_loss", patience=patience,
                                       restore_best_weights=True, verbose=0)], verbose=0)
        nae = per_output_nae_val(m, Xv, data["Y_raw_val"], W, data["scaler_Y"])
        res[W] = float(np.nanmean(nae))
        log.info("  W=%2d -> NAE_val=%.3f%%", W, res[W])
        keras.backend.clear_session()
    W_star = min(res, key=res.get)
    log.info("  W* = %d (NAE_val=%.3f%%)", W_star, res[W_star])
    return W_star, res


def hard_loads_from_val(nae_val_per_output: np.ndarray, k: int = 2) -> list:
    """Indices of the k hardest loads by VALIDATION NAE (descending)."""
    return list(np.argsort(-np.nan_to_num(nae_val_per_output, nan=-1.0))[:k])


def _ckpt_file(name: str) -> Path:
    d = CKPT_ROOT / f"script_12_{name}"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"12_{name}_checkpoint.pkl"


def _load_ckpt(name, log):
    f = _ckpt_file(name)
    if not f.exists():
        return 0, {}, {}
    with open(f, "rb") as fh:
        d = pickle.load(fh)
    log.info("Checkpoint restored (%s); resuming from seed idx %d", name, d["start_idx"])
    return d["start_idx"], d["histories"], d["preds_norm"]


def _save_ckpt(name, start_idx, histories, preds_norm):
    with open(_ckpt_file(name), "wb") as fh:
        pickle.dump({"start_idx": start_idx, "histories": histories,
                     "preds_norm": preds_norm}, fh)


def train_and_evaluate(build_fn, name: str, data: dict, loss_fn, batch: int, log,
                       script_prefix: str = "12") -> dict:
    """Trains build_fn x SEEDS, evaluates NAE (mid-point) and writes the outputs with prefix
    {script_prefix}_{name}_*."""
    import tensorflow as tf  # noqa: F401
    import keras
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    for dd in (MODELS_DIR, PREDS_DIR, METRICS_DIR, FIGDATA_DIR, MANIFESTS_DIR, AUDITS_DIR):
        dd.mkdir(parents=True, exist_ok=True)

    X_train, Y_train = data["X_train"], data["Y_train"]
    X_val, Y_val     = data["X_val"], data["Y_val"]
    X_test           = data["X_test"]
    Y_raw_test       = data["Y_raw_test"]
    scaler_Y         = data["scaler_Y"]
    labels           = data["output_labels"]
    n_features       = data["n_features"]
    output_dim       = data["output_dim"]
    n_win_test       = X_test.shape[0]

    start_idx, hist_all, preds_all = _load_ckpt(name, log)
    val_naes = {}                                        # validation NAE per output, per seed
    for idx in range(start_idx, len(SEEDS)):
        seed = SEEDS[idx]
        log.info("=" * 55)
        log.info("%s: seed %d (%d/%d)", name, seed, idx + 1, len(SEEDS))
        set_seed(seed)
        model = build_fn(n_features, output_dim)
        mpath = MODELS_DIR / f"{script_prefix}_{name}_seed_{seed}.keras"
        cbs = [EarlyStopping(monitor="val_loss", patience=PATIENCE,
                             restore_best_weights=True, verbose=1),
               ModelCheckpoint(str(mpath), monitor="val_loss",
                               save_best_only=True, verbose=0)]
        h = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                      batch_size=batch, epochs=EPOCHS_MAX, callbacks=cbs, verbose=2)
        preds_all[seed] = model.predict(X_test, verbose=0)   # (n_win, W, output_dim)
        if "Y_raw_val" in data:                              # validation NAE per output (for script 15)
            val_naes[seed] = per_output_nae_val(model, X_val, data["Y_raw_val"], W, scaler_Y)
        hist_all[seed] = {"loss": h.history["loss"], "val_loss": h.history["val_loss"],
                          "n_epochs": len(h.history["loss"])}
        log.info("seed %d done: epochs=%d val_loss_best=%.5f",
                 seed, len(h.history["loss"]), min(h.history["val_loss"]))
        _save_ckpt(name, idx + 1, hist_all, preds_all)

    # NAE per seed/output (mid-point alignment)
    y_true = Y_raw_test[MID:MID + n_win_test]
    mrows, prows = [], []
    for seed in SEEDS:
        y_mid = scaler_Y.inverse_transform(preds_all[seed][:, MID, :])
        n = min(len(y_mid), len(y_true))
        for j, lbl in enumerate(labels):
            mrows.append({"seed": seed, "output": lbl,
                          "nae_pct": round(nae_percent(y_true[:n, j], y_mid[:n, j]), 4)})
        for wi in range(len(y_mid)):
            row = {"seed": seed, "window_idx": wi}
            row.update({lbl: float(y_mid[wi, j]) for j, lbl in enumerate(labels)})
            prows.append(row)

    mdf = pd.DataFrame(mrows)
    summary = mdf.groupby("output")["nae_pct"].agg(nae_mean="mean", nae_std="std").reset_index()
    mdf.to_csv(METRICS_DIR / f"{script_prefix}_{name}_metrics.csv", index=False)
    pd.DataFrame(prows).to_parquet(PREDS_DIR / f"{script_prefix}_{name}_predictions.parquet", index=False)

    hrows = []
    for seed in SEEDS:
        for ep, (tr, va) in enumerate(zip(hist_all[seed]["loss"], hist_all[seed]["val_loss"])):
            hrows.append({"seed": seed, "epoch": ep + 1, "split": "train", "loss": tr})
            hrows.append({"seed": seed, "epoch": ep + 1, "split": "val", "loss": va})
    pd.DataFrame(hrows).to_csv(FIGDATA_DIR / f"{script_prefix}_{name}_training_history_long.csv", index=False)

    # VALIDATION NAE per output (mean over seeds), consumed by script 15 (SPEC/Ensemble).
    # Accumulates in {script_prefix}_val_nae_per_output.csv (append: lstm, then rcnn_att).
    if val_naes:
        vmean = np.nanmean(np.vstack([val_naes[s] for s in val_naes]), axis=0)
        vrows = [{"model": name, "idx": j, "output": labels[j],
                  "nae_val_pct": round(float(vmean[j]), 4)} for j in range(output_dim)]
        vpath = AUDITS_DIR / f"{script_prefix}_val_nae_per_output.csv"
        pd.DataFrame(vrows).to_csv(vpath, mode="a", header=not vpath.exists(), index=False)

    res = {"name": name, "n_outputs": output_dim,
           "overall_nae_mean": round(float(summary["nae_mean"].mean()), 4),
           "overall_nae_std": round(float(summary["nae_std"].mean()), 4),
           "epochs_per_seed": {int(s): hist_all[s]["n_epochs"] for s in SEEDS}}
    log.info("%s: mean NAE: %.3f%% +/- %.3f%%", name, res["overall_nae_mean"], res["overall_nae_std"])
    return res
