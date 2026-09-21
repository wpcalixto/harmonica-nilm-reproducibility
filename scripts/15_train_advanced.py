#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 15_train_advanced.py

File type: Training script (pipeline stage 15: advanced architectures, SPEC, MoTE_v2, Ensemble_G4; optional HPO sensitivity)

Purpose:
    Advanced training of Block B + PE-ES-Optuna HPO sensitivity in the SAME script.

    MAIN FLOW (default): 5 seeds each, DYNAMIC output_dim (18), series reconstruction before
    the metrics:
      - PE-ES   : Conv->BiLSTM->Attention; W* selected on VALIDATION (grid {6,12,18,24,30,36});
                  frozen default hyperparameters; NO Optuna.
      - SPEC    : PE-ES with 5x weight on the 2 hardest loads (RCNN-att on VALIDATION).
      - MoTE_v2 : mixture of 3 encoders (shallow/deep BiLSTM + Conv), fixed W=36.
      - Ensemble_G4 : per output, best of {14_lstm,14_rcnn} by VALIDATION NAE.
      Outputs: models/15_{pe_es,spec,mote_v2}_seed_*.keras . predictions/15_*_predictions.parquet
               metrics/15_*_metrics.csv . figdata/15_*_history_long.csv . audits/15_* . manifest.

    HPO SENSITIVITY (optional; only with env RUN_OPTUNA=1):
      PE-ES search with Optuna (TPE 200 + CMA-ES 100), score = val_loss on VALIDATION; best
      hyperparameters retrained x 5 seeds. It is an HPO-intensive SENSITIVITY analysis (~18 h
      of GPU), outside the main ranking; hence behind a flag and NOT run by default.
      Outputs (prefix kept so that 16/17/18 read them alike): models/15_exp_pe_es_optuna_seed_*.keras
               predictions/15_exp_pe_es_optuna_predictions.parquet . metrics/15_exp_pe_es_optuna_metrics.csv
               figdata/15_exp_pe_es_optuna_history_long.csv . audits/15_exp_optuna_trials.csv
               manifests/15_exp_pe_es_optuna_params.json . checkpoints/15_exp_pe_es_optuna/optuna_study.db
      Overrides for smoke tests: OPTUNA_TPE, OPTUNA_CMA, OPTUNA_SEARCH_PATIENCE, NN_EPOCHS, NN_PATIENCE, NN_SEEDS.

    Window: the selection of W is done ONLY on validation; W* is frozen; predictions are
    reconstructed on the original time axis (offset = W//2) before the metrics (test).

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

import gc
import json
import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import _nn_common as C

for d in (C.MODELS_DIR, C.PREDS_DIR, C.METRICS_DIR, C.FIGDATA_DIR, C.MANIFESTS_DIR,
          C.PROJECT_ROOT / "audits", C.PROJECT_ROOT / "logs"):
    d.mkdir(parents=True, exist_ok=True)
AUDITS_DIR = C.PROJECT_ROOT / "audits"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(C.PROJECT_ROOT / "logs" / "15_train_advanced.log",
                                                  mode="w", encoding="utf-8")])
log = logging.getLogger(__name__)

W_GRID      = [6, 12, 18, 24, 30, 36]
W_MOTE      = 36
N_SPEC      = 2
SPEC_WEIGHT = 5.0
# frozen default hyperparameters of PE-ES (no search):
PE = dict(conv1=64, conv2=128, kernel=3, lstm=100, d1=100, d2=50, drop=0.20, batch=64)
MOTE_BATCH = 32

# -- HPO sensitivity (PE-ES-Optuna), only with RUN_OPTUNA=1 --------------------------------
OPT_W_CANDIDATES    = W_GRID            # same W grid as the main flow
OPT_TPE             = int(os.environ.get("OPTUNA_TPE", "200"))
OPT_CMA             = int(os.environ.get("OPTUNA_CMA", "100"))
OPT_SEED            = 42
OPT_SEARCH_PATIENCE = int(os.environ.get("OPTUNA_SEARCH_PATIENCE", "15"))


_CUSTOM = None


def _custom_layers():
    """Defines and registers the custom layers ONCE (avoids re-registration at every call)."""
    global _CUSTOM
    if _CUSTOM is not None:
        return _CUSTOM
    import tensorflow as tf, keras
    from keras import layers

    @keras.saving.register_keras_serializable(package="Custom")
    class TemporalAttention(layers.Layer):
        def __init__(self, units=64, **kw):
            super().__init__(**kw); self.units = units
            self.W_att = layers.Dense(units, use_bias=False, activation="tanh")
            self.v_att = layers.Dense(1, use_bias=False)
        def call(self, h):
            a = keras.activations.softmax(self.v_att(self.W_att(h)), axis=1)
            ctx = tf.reduce_sum(a * h, axis=1)
            return tf.tile(tf.expand_dims(ctx, 1), [1, tf.shape(h)[1], 1]) + h
        def get_config(self):
            c = super().get_config(); c.update({"units": self.units}); return c

    @keras.saving.register_keras_serializable(package="Custom")
    class StackHeads(layers.Layer):
        def call(self, inputs): return keras.ops.stack(inputs, axis=-1)
        def compute_output_shape(self, s): t = list(s[0]); t.append(3); return tuple(t)

    @keras.saving.register_keras_serializable(package="Custom")
    class WeightedMixture(layers.Layer):
        def call(self, inputs): g, st = inputs; return keras.ops.sum(g * st, axis=-1)
        def compute_output_shape(self, s): return tuple(list(s[0])[:-1])

    _CUSTOM = {"TemporalAttention": TemporalAttention, "StackHeads": StackHeads,
               "WeightedMixture": WeightedMixture}
    return _CUSTOM


def build_pe_es(W, n_feat, n_out, loss=None):
    import keras
    from keras import layers
    TA = _custom_layers()["TemporalAttention"]
    loss = loss if loss is not None else LOSS
    inp = keras.Input(shape=(W, n_feat))
    x = inp
    for f in (PE["conv1"], PE["conv2"]):
        x = layers.Conv1D(f, PE["kernel"], padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2, padding="same")(x)
        x = layers.UpSampling1D(2)(x)
        x = layers.Dropout(PE["drop"])(x)
    x = layers.Bidirectional(layers.LSTM(PE["lstm"], return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(PE["lstm"], return_sequences=True))(x)
    x = TA(units=PE["lstm"])(x)
    x = layers.TimeDistributed(layers.Dense(PE["d1"], activation="relu"))(x)
    x = layers.TimeDistributed(layers.Dense(PE["d2"], activation="relu"))(x)
    out = layers.TimeDistributed(layers.Dense(n_out, activation="linear"))(x)
    m = keras.Model(inp, out, name="PE_ES")
    m.compile(optimizer=keras.optimizers.Adam(C.LR), loss=loss)
    return m


def build_pe_es_hp(W, n_feat, n_out, hp, loss):
    """Parameterised PE-ES (same family; hyperparameters from the Optuna trial)."""
    import keras
    from keras import layers
    TA = _custom_layers()["TemporalAttention"]
    inp = keras.Input(shape=(W, n_feat))
    x = inp
    for f in (hp["conv1"], hp["conv2"]):
        x = layers.Conv1D(f, hp["kernel"], padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2, padding="same")(x)
        x = layers.UpSampling1D(2)(x)
        x = layers.Dropout(hp["dropout"])(x)
    x = layers.Bidirectional(layers.LSTM(hp["lstm"], return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(hp["lstm"], return_sequences=True))(x)
    x = TA(units=hp["lstm"])(x)
    x = layers.TimeDistributed(layers.Dense(hp["dense1"], activation="relu"))(x)
    x = layers.TimeDistributed(layers.Dense(hp["dense2"], activation="relu"))(x)
    out = layers.TimeDistributed(layers.Dense(n_out, activation="linear"))(x)
    m = keras.Model(inp, out, name="PE_ES_Optuna")
    m.compile(optimizer=keras.optimizers.Adam(hp["lr"]), loss=loss)
    return m


def build_mote(W, n_feat, n_out, loss=None):
    import keras
    from keras import layers
    loss = loss if loss is not None else LOSS
    cl = _custom_layers()
    StackHeads, WeightedMixture = cl["StackHeads"], cl["WeightedMixture"]

    inp = keras.Input(shape=(W, n_feat), name="input")
    e1 = layers.Bidirectional(layers.LSTM(50, return_sequences=True))(inp)
    e1 = layers.Bidirectional(layers.LSTM(50, return_sequences=True))(e1)
    e2 = layers.Bidirectional(layers.LSTM(100, return_sequences=True))(inp)
    e2 = layers.Bidirectional(layers.LSTM(100, return_sequences=True))(e2)
    e3 = layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
    e3 = layers.BatchNormalization()(e3)
    e3 = layers.Conv1D(128, 3, padding="same", activation="relu")(e3)
    e3 = layers.BatchNormalization()(e3)
    o1 = layers.TimeDistributed(layers.Dense(n_out))(e1)
    o2 = layers.TimeDistributed(layers.Dense(n_out))(e2)
    o3 = layers.TimeDistributed(layers.Dense(n_out))(e3)
    gate = layers.TimeDistributed(layers.Dense(n_out * 3))(layers.Concatenate(axis=-1)([e1, e2, e3]))
    gate = layers.Reshape((W, n_out, 3))(gate)
    gate = layers.Softmax(axis=-1)(gate)
    stack = StackHeads()([o1, o2, o3])
    out = WeightedMixture()([gate, stack])
    m = keras.Model(inp, out, name="MoTE_v2")
    m.compile(optimizer=keras.optimizers.Adam(C.LR), loss=loss)
    return m


def train_family(build_at_W, name, W, base, X_test_ts, data, batch, loss, log):
    """Trains build_at_W() (= model at the given W) x SEEDS; saves predictions/metrics/figdata."""
    import keras
    from keras.callbacks import EarlyStopping
    Xtr, Ytr = C.rewindow(base["Xtr_ts"], base["Ytr_ts"], W)
    Xv, Yv = C.rewindow(base["Xv_ts"], base["Yv_ts"], W)
    Xte, _ = C.rewindow(X_test_ts, X_test_ts, W)
    mid = W // 2
    labels, scaler_Y = data["output_labels"], data["scaler_Y"]
    yt = data["Y_raw_test"][mid:mid + len(Xte)]
    mrows, prows, hrows, seed_nae = [], [], [], {}
    for seed in C.SEEDS:
        C.set_seed(seed)
        m = build_at_W(W, data["n_features"], data["output_dim"], loss=loss)
        h = m.fit(Xtr, Ytr, validation_data=(Xv, Yv), batch_size=batch, epochs=C.EPOCHS_MAX,
                  callbacks=[EarlyStopping(monitor="val_loss", patience=C.PATIENCE,
                                           restore_best_weights=True, verbose=0)], verbose=2)
        m.save(C.MODELS_DIR / f"15_{name}_seed_{seed}.keras")
        yp = scaler_Y.inverse_transform(m.predict(Xte, verbose=0)[:, mid, :])
        n = min(len(yp), len(yt))
        for j, lbl in enumerate(labels):
            den = np.sum(np.abs(yt[:n, j]))
            nae = np.nan if den == 0 else float(np.sum(np.abs(yp[:n, j] - yt[:n, j])) / den * 100)
            mrows.append({"seed": seed, "output": lbl, "nae_pct": round(nae, 4)})
        for wi in range(n):
            prows.append({"seed": seed, "window_idx": wi, **{lbl: float(yp[wi, j]) for j, lbl in enumerate(labels)}})
        for ep, (tr, va) in enumerate(zip(h.history["loss"], h.history["val_loss"])):
            hrows.append({"seed": seed, "epoch": ep + 1, "split": "train", "loss": tr})
            hrows.append({"seed": seed, "epoch": ep + 1, "split": "val", "loss": va})
        seed_nae[seed] = float(np.nanmean([r["nae_pct"] for r in mrows if r["seed"] == seed]))
        keras.backend.clear_session()
    pd.DataFrame(mrows).to_csv(C.METRICS_DIR / f"15_{name}_metrics.csv", index=False)
    pd.DataFrame(prows).to_parquet(C.PREDS_DIR / f"15_{name}_predictions.parquet", index=False)
    pd.DataFrame(hrows).to_csv(C.FIGDATA_DIR / f"15_{name}_training_history_long.csv", index=False)
    nae_mean = float(np.mean(list(seed_nae.values())))
    log.info("%s (W=%d): mean test NAE: %.3f%%", name, W, nae_mean)
    return {"W": W, "nae_test_mean": round(nae_mean, 4), "per_seed": seed_nae}


# ══════════════════════════════════════════════════════════════════════════════
#  HPO SENSITIVITY: PE-ES-Optuna (only with RUN_OPTUNA=1)
# ══════════════════════════════════════════════════════════════════════════════
def run_optuna_sensitivity(data, base, X_test_ts, loss):
    """Optuna search (TPE+CMA-ES) of PE-ES + final training x seeds.
    Reuses data/base/loss already loaded in main. Outputs with prefix 15_exp_*."""
    import keras
    from keras.callbacks import EarlyStopping
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler
    from optuna.trial import TrialState
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    log.info("\n########## HPO SENSITIVITY: PE-ES-Optuna (RUN_OPTUNA=1) ##########")
    log.info("TPE=%d  CMA-ES=%d  search_patience=%d", OPT_TPE, OPT_CMA, OPT_SEARCH_PATIENCE)
    ckpt_dir = C.CKPT_ROOT / "15_exp_pe_es_optuna"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{ckpt_dir / 'optuna_study.db'}"

    n_feat, n_out = data["n_features"], data["output_dim"]
    labels, scaler_Y = data["output_labels"], data["scaler_Y"]
    Xtr_ts, Ytr_ts = base["Xtr_ts"], base["Ytr_ts"]
    Xv_ts, Yv_ts = base["Xv_ts"], base["Yv_ts"]
    Xte_ts = X_test_ts

    def objective(trial):
        hp = {"W": trial.suggest_categorical("W", OPT_W_CANDIDATES),
              "conv1": trial.suggest_categorical("conv1", [32, 64, 128]),
              "conv2": trial.suggest_categorical("conv2", [64, 128, 256]),
              "kernel": trial.suggest_categorical("kernel", [3, 5]),
              "lstm": trial.suggest_categorical("lstm", [64, 100, 128, 200]),
              "dense1": trial.suggest_categorical("dense1", [64, 100, 128]),
              "dense2": trial.suggest_categorical("dense2", [32, 50, 64]),
              "dropout": trial.suggest_float("dropout", 0.10, 0.40, step=0.05),
              "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
              "batch": trial.suggest_categorical("batch", [32, 64, 128])}
        W = hp["W"]
        Xtr, Ytr = C.rewindow(Xtr_ts, Ytr_ts, W)
        Xv, Yv = C.rewindow(Xv_ts, Yv_ts, W)
        C.set_seed(OPT_SEED)
        m = build_pe_es_hp(W, n_feat, n_out, hp, loss)
        h = m.fit(Xtr, Ytr, validation_data=(Xv, Yv), batch_size=hp["batch"], epochs=C.EPOCHS_MAX,
                  callbacks=[EarlyStopping(monitor="val_loss", patience=OPT_SEARCH_PATIENCE,
                                           restore_best_weights=True, verbose=0)], verbose=0)
        val_loss = float(min(h.history["val_loss"]))
        trial.set_user_attr("n_epochs", len(h.history["loss"]))
        del m; keras.backend.clear_session(); gc.collect()
        return val_loss

    def run_study(study, n_target):
        done = len([t for t in study.trials if t.state == TrialState.COMPLETE])
        remaining = max(0, n_target - done)
        if remaining:
            study.optimize(objective, n_trials=remaining, show_progress_bar=False)
        else:
            log.info("  (study %s already has %d trials; nothing to do)", study.study_name, done)

    log.info("[Phase 1] TPE: %d trials", OPT_TPE)
    study_tpe = optuna.create_study(direction="minimize", sampler=TPESampler(seed=OPT_SEED),
                                    storage=storage, study_name="PE_ES_TPE", load_if_exists=True)
    run_study(study_tpe, OPT_TPE)
    log.info("[Phase 2] CMA-ES: %d trials", OPT_CMA)
    study_cma = optuna.create_study(direction="minimize",
                                    sampler=CmaEsSampler(x0=study_tpe.best_params, seed=OPT_SEED),
                                    storage=storage, study_name="PE_ES_CMA", load_if_exists=True)
    run_study(study_cma, OPT_CMA)

    if study_cma.best_value < study_tpe.best_value:
        best_params, best_val, best_study = study_cma.best_params, study_cma.best_value, "CMA-ES"
    else:
        best_params, best_val, best_study = study_tpe.best_params, study_tpe.best_value, "TPE"
    log.info("GLOBAL BEST (%s) val_loss=%.6f params=%s", best_study, best_val, best_params)

    rows = []
    for s in (study_tpe, study_cma):
        for t in s.trials:
            row = {"study": s.study_name, "trial": t.number, "val_loss": t.value,
                   "n_epochs": t.user_attrs.get("n_epochs", -1)}
            row.update(t.params); rows.append(row)
    pd.DataFrame(rows).to_csv(AUDITS_DIR / "15_exp_optuna_trials.csv", index=False)

    hp = {**best_params}; W = hp["W"]; mid = W // 2
    Xtr, Ytr = C.rewindow(Xtr_ts, Ytr_ts, W)
    Xv, Yv = C.rewindow(Xv_ts, Yv_ts, W)
    Xte, _ = C.rewindow(Xte_ts, Xte_ts, W)
    yt = data["Y_raw_test"][mid:mid + len(Xte)]
    log.info("[Final] W*=%d HP=%s Xtest=%s", W, hp, Xte.shape)

    mrows, prows, hrows, seed_nae = [], [], [], {}
    for seed in C.SEEDS:
        C.set_seed(seed)
        m = build_pe_es_hp(W, n_feat, n_out, hp, loss)
        h = m.fit(Xtr, Ytr, validation_data=(Xv, Yv), batch_size=hp["batch"], epochs=C.EPOCHS_MAX,
                  callbacks=[EarlyStopping(monitor="val_loss", patience=C.PATIENCE,
                                           restore_best_weights=True, verbose=0)], verbose=2)
        m.save(C.MODELS_DIR / f"15_exp_pe_es_optuna_seed_{seed}.keras")
        yp = scaler_Y.inverse_transform(m.predict(Xte, verbose=0)[:, mid, :])
        n = min(len(yp), len(yt))
        for j, lbl in enumerate(labels):
            den = np.sum(np.abs(yt[:n, j]))
            nae = np.nan if den == 0 else float(np.sum(np.abs(yp[:n, j] - yt[:n, j])) / den * 100)
            mrows.append({"seed": seed, "output": lbl, "nae_pct": round(nae, 4)})
        for wi in range(n):
            prows.append({"seed": seed, "window_idx": wi, **{lbl: float(yp[wi, j]) for j, lbl in enumerate(labels)}})
        for ep, (tr, va) in enumerate(zip(h.history["loss"], h.history["val_loss"])):
            hrows.append({"seed": seed, "epoch": ep + 1, "split": "train", "loss": tr})
            hrows.append({"seed": seed, "epoch": ep + 1, "split": "val", "loss": va})
        seed_nae[seed] = float(np.nanmean([r["nae_pct"] for r in mrows if r["seed"] == seed]))
        log.info("seed %d done: NAE=%.3f%%", seed, seed_nae[seed])
        keras.backend.clear_session(); gc.collect()

    pd.DataFrame(mrows).to_csv(C.METRICS_DIR / "15_exp_pe_es_optuna_metrics.csv", index=False)
    pd.DataFrame(prows).to_parquet(C.PREDS_DIR / "15_exp_pe_es_optuna_predictions.parquet", index=False)
    pd.DataFrame(hrows).to_csv(C.FIGDATA_DIR / "15_exp_pe_es_optuna_history_long.csv", index=False)
    nae_mean = float(np.mean(list(seed_nae.values())))
    log.info("PE-ES-Optuna: mean test NAE: %.3f%%", nae_mean)
    manifest = {"script": "15_train_advanced.py::run_optuna_sensitivity",
                "tipo": "sensibilidade HPO (fora do fluxo principal)",
                "analysis_group": "hpo_sensitivity", "is_primary": False,
                "run_timestamp": datetime.now(timezone.utc).isoformat(),
                "busca": {"tpe_trials": OPT_TPE, "cma_trials": OPT_CMA,
                          "search_patience": OPT_SEARCH_PATIENCE, "W_candidates": OPT_W_CANDIDATES},
                "best": {"study": best_study, "val_loss": round(best_val, 6), "params": best_params},
                "resultado": {"nae_test_mean": round(nae_mean, 4), "per_seed": seed_nae, "W_star": int(W)}}
    (C.MANIFESTS_DIR / "15_exp_pe_es_optuna_params.json").write_text(json.dumps(manifest, indent=2, default=str))
    log.info("HPO sensitivity done (15_exp_*).")
    return {"nae_test_mean": round(nae_mean, 4), "W_star": int(W), "best_study": best_study}


def main():
    import keras
    log.info("=== 15_train_advanced.py ===")
    data = C.load_splits()
    pg = C.phase_groups(data["output_labels"])
    global LOSS
    LOSS = C.make_masked_loss(pg)
    base = dict(Xtr_ts=C.unwindow(data["X_train"]), Ytr_ts=C.unwindow(data["Y_train"]),
                Xv_ts=C.unwindow(data["X_val"]), Yv_ts=C.unwindow(data["Y_val"]))
    X_test_ts = C.unwindow(data["X_test"])
    log.info("output_dim=%d | phase groups=%d", data["output_dim"], len(pg))

    # -- 1) W* on validation (reduced grid) ----------------------------------------------------
    log.info("[W*] selection on validation, grid %s", W_GRID)
    W_star, w_res = C.select_W_on_val(build_pe_es, data, W_GRID, PE["batch"], log)
    pd.DataFrame([{"W": w, "nae_val_pct": round(v, 4), "is_W_star": w == W_star}
                  for w, v in w_res.items()]).to_csv(AUDITS_DIR / "15_window_selection.csv", index=False)

    results = {}
    # -- 2) PE-ES at W* -------------------------------------------------------------------------
    results["pe_es"] = train_family(build_pe_es, "pe_es", W_star, base, X_test_ts, data, PE["batch"], LOSS, log)

    # -- 3) SPEC: hard loads by RCNN-att on VALIDATION (read from 14_val_nae_*) -----------------
    vdf = pd.read_csv(AUDITS_DIR / "14_val_nae_per_output.csv")
    # normalises names (accepts rcnn/rcnn_att/RCNN-att, lstm/LSTM); avoids an empty filter
    vdf["model_norm"] = (vdf["model"].astype(str).str.lower()
                         .replace({"rcnn_att": "rcnn", "rcnn-att": "rcnn"}))
    rcnn_val = (vdf[vdf.model_norm == "rcnn"].set_index("idx")["nae_val_pct"]
                .reindex(range(data["output_dim"])).to_numpy())
    lstm_val = (vdf[vdf.model_norm == "lstm"].set_index("idx")["nae_val_pct"]
                .reindex(range(data["output_dim"])).to_numpy())
    hard = C.hard_loads_from_val(rcnn_val, k=N_SPEC)
    pd.DataFrame([{"idx": int(i), "output": data["output_labels"][i],
                   "nae_val_rcnn_pct": round(float(rcnn_val[i]), 4), "is_spec": True} for i in hard]
                 ).to_csv(AUDITS_DIR / "15_spec_hard_loads.csv", index=False)
    log.info("[SPEC] hard loads (val, RCNN): %s", [data["output_labels"][i] for i in hard])
    LOSS_SPEC = C.make_masked_loss_weighted(pg, list(hard), SPEC_WEIGHT)
    results["spec"] = train_family(build_pe_es, "spec", W_star, base, X_test_ts, data, PE["batch"], LOSS_SPEC, log)

    # -- 4) MoTE_v2 (fixed W=36) ----------------------------------------------------------------
    results["mote_v2"] = train_family(build_mote, "mote_v2", W_MOTE, base, X_test_ts, data, MOTE_BATCH, LOSS, log)

    # -- 5) Ensemble_G4: per output, best of {14_lstm,14_rcnn} by validation NAE ----------------
    log.info("[Ensemble] choice per output via validation NAE of script 14 (lstm vs rcnn)")
    choice = np.array(["rcnn" if rcnn_val[j] < lstm_val[j] else "lstm"
                       for j in range(data["output_dim"])])
    pd.DataFrame([{"output": data["output_labels"][j], "choice": choice[j],
                   "nae_val_lstm": round(float(lstm_val[j]), 4),
                   "nae_val_rcnn": round(float(rcnn_val[j]), 4)} for j in range(data["output_dim"])]
                 ).to_csv(AUDITS_DIR / "15_ensemble_choice.csv", index=False)
    labels, mid12 = data["output_labels"], 6
    pl = pd.read_parquet(C.PREDS_DIR / "14_lstm_predictions.parquet")
    pr = pd.read_parquet(C.PREDS_DIR / "14_rcnn_att_predictions.parquet")
    erows, eprows, eseed = [], [], {}
    for seed in C.SEEDS:
        sl = pl[pl.seed == seed].sort_values("window_idx")
        sr = pr[pr.seed == seed].sort_values("window_idx")
        yl = sl[labels].to_numpy(dtype=float); yr = sr[labels].to_numpy(dtype=float)
        ye = np.where(choice[None, :] == "rcnn", yr, yl)
        yt = data["Y_raw_test"][mid12:mid12 + len(ye)]
        n = min(len(ye), len(yt))
        for j, lbl in enumerate(labels):
            den = np.sum(np.abs(yt[:n, j]))
            nae = np.nan if den == 0 else float(np.sum(np.abs(ye[:n, j] - yt[:n, j])) / den * 100)
            erows.append({"seed": seed, "output": lbl, "nae_pct": round(nae, 4)})
        for wi in range(n):
            eprows.append({"seed": seed, "window_idx": wi, **{lbl: float(ye[wi, j]) for j, lbl in enumerate(labels)}})
        eseed[seed] = float(np.nanmean([r["nae_pct"] for r in erows if r["seed"] == seed]))
    pd.DataFrame(erows).to_csv(C.METRICS_DIR / "15_ensemble_g4_metrics.csv", index=False)
    pd.DataFrame(eprows).to_parquet(C.PREDS_DIR / "15_ensemble_g4_predictions.parquet", index=False)
    results["ensemble_g4"] = {"W": 12, "nae_test_mean": round(float(np.mean(list(eseed.values()))), 4)}
    log.info("ensemble_g4: mean test NAE: %.3f%%", results["ensemble_g4"]["nae_test_mean"])

    manifest = {
        "script": "15_train_advanced.py", "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "merge_de": ["20_train_pe_es_spec.py", "21_train_mote_v2_and_ensemble_g4.py",
                     "15_exp_pe_es_optuna.py (sensibilidade, via RUN_OPTUNA=1)"],
        "methodology": {"pe_es": "W* by validation (grid %s), no Optuna" % W_GRID,
                        "spec": "PE-ES with weight %g on %d hard loads (RCNN, validation)" % (SPEC_WEIGHT, N_SPEC),
                        "mote_v2": "W=%d fixo" % W_MOTE,
                        "ensemble_g4": "per output, best of {14_lstm,14_rcnn} by validation NAE",
                        "W_star": int(W_star),
                        "hpo_sensitivity": "PE-ES-Optuna via RUN_OPTUNA=1 (fora do fluxo principal)"},
        "results": results,
    }
    (C.MANIFESTS_DIR / "15_train_advanced_params.json").write_text(json.dumps(manifest, indent=2, default=str))
    log.info("Manifest saved. Main flow done.")

    # -- optional HPO sensitivity (only with RUN_OPTUNA=1) --------------------------------------
    if os.environ.get("RUN_OPTUNA") == "1":
        run_optuna_sensitivity(data, base, X_test_ts, LOSS)
    else:
        log.info("HPO sensitivity (PE-ES-Optuna) NOT executed (set RUN_OPTUNA=1 to run it).")


if __name__ == "__main__":
    LOSS = None
    main()
