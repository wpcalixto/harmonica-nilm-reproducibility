#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 14_train_baselines.py

File type: Training script (pipeline stage 14: baseline architectures)

Purpose:
    Training of the baselines (BiLSTM and RCNN-att), 5 seeds each.

    Both architectures share the same training structure; only the model builder differs.
    The common skeleton (data loading, set_seed, masked loss, seed loop, checkpoint,
    metrics/predictions/figdata) lives in `_nn_common.py`.

    Inputs (Block B):
        data/processed/13_X_{train,val,test}.npy, 13_Y_{train,val,test}.npy
        data/processed/13_Y_raw_test.npy, 13_scaler_Y.pkl
        manifests/13_create_windows_splits_params.json   (output labels)

    Outputs (per architecture {lstm, rcnn_att}):
        models/14_{arch}_seed_*.keras
        predictions/14_{arch}_predictions.parquet
        metrics/14_{arch}_metrics.csv
        figdata/14_{arch}_training_history_long.csv
        logs/14_train_baselines.log
        manifests/14_train_baselines_params.json

    The number of outputs (output_dim) and the phase groups are DYNAMIC (derived from the shape
    of Y and from the labels); there is no hard-coded Dense layer size.

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
import logging
import sys
from datetime import datetime, timezone

import _nn_common as C

# -- logging ------------------------------------------------------------------------------
C.PROJECT_ROOT.joinpath("logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(C.PROJECT_ROOT / "logs" / "14_train_baselines.log",
                                  mode="w", encoding="utf-8")],
)
log = logging.getLogger(__name__)

# -- constants per architecture -----------------------------------------------------------
LSTM_UNITS     = 100
LSTM_BATCH     = 32
CONV_FILTERS_1 = 64
CONV_FILTERS_2 = 128
CONV_KERNEL    = 3
POOL_SIZE      = 2
DROPOUT_RATE   = 0.20
DENSE_UNITS_1  = 100
DENSE_UNITS_2  = 50
RCNN_BATCH     = 64


def build_lstm(n_features: int, n_out: int):
    import keras
    from keras import layers
    inp = keras.Input(shape=(C.W, n_features), name="input")
    h = layers.Bidirectional(layers.LSTM(LSTM_UNITS, return_sequences=True), name="bilstm_1")(inp)
    h = layers.Bidirectional(layers.LSTM(LSTM_UNITS, return_sequences=True), name="bilstm_2")(h)
    out = layers.TimeDistributed(layers.Dense(n_out, activation="linear"), name="output")(h)
    model = keras.Model(inp, out, name="LSTM_baseline")
    model.compile(optimizer=keras.optimizers.Adam(C.LR), loss=LOSS_FN)
    return model


_TEMPORAL_ATT_CLS = None


def _temporal_attention_cls():
    """Defines and registers TemporalAttention ONCE (avoids re-registration at every seed)."""
    global _TEMPORAL_ATT_CLS
    if _TEMPORAL_ATT_CLS is not None:
        return _TEMPORAL_ATT_CLS
    import tensorflow as tf
    import keras
    from keras import layers

    @keras.saving.register_keras_serializable(package="Custom")
    class TemporalAttention(layers.Layer):
        def __init__(self, units=64, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.W_att = layers.Dense(units, use_bias=False, activation="tanh")
            self.v_att = layers.Dense(1, use_bias=False)

        def call(self, h):
            score = self.v_att(self.W_att(h))
            alpha = keras.activations.softmax(score, axis=1)
            context = tf.reduce_sum(alpha * h, axis=1)
            context_seq = tf.tile(tf.expand_dims(context, 1), [1, tf.shape(h)[1], 1])
            return context_seq + h

        def get_config(self):
            cfg = super().get_config(); cfg.update({"units": self.units}); return cfg

    _TEMPORAL_ATT_CLS = TemporalAttention
    return _TEMPORAL_ATT_CLS


def build_rcnn_att(n_features: int, n_out: int):
    import keras
    from keras import layers
    TemporalAttention = _temporal_attention_cls()

    inp = keras.Input(shape=(C.W, n_features), name="input")
    x = layers.Conv1D(CONV_FILTERS_1, CONV_KERNEL, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(POOL_SIZE, padding="same")(x)
    x = layers.UpSampling1D(POOL_SIZE)(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Conv1D(CONV_FILTERS_2, CONV_KERNEL, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(POOL_SIZE, padding="same")(x)
    x = layers.UpSampling1D(POOL_SIZE)(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Bidirectional(layers.LSTM(LSTM_UNITS, return_sequences=True), name="bilstm_1")(x)
    x = layers.Bidirectional(layers.LSTM(LSTM_UNITS, return_sequences=True), name="bilstm_2")(x)
    x = TemporalAttention(units=LSTM_UNITS, name="temporal_attention")(x)
    x = layers.TimeDistributed(layers.Dense(DENSE_UNITS_1, activation="relu"), name="dense_1")(x)
    x = layers.TimeDistributed(layers.Dense(DENSE_UNITS_2, activation="relu"), name="dense_2")(x)
    out = layers.TimeDistributed(layers.Dense(n_out, activation="linear"), name="output")(x)
    model = keras.Model(inp, out, name="RCNN_att")
    model.compile(optimizer=keras.optimizers.Adam(C.LR), loss=LOSS_FN)
    return model


ARCHITECTURES = {"lstm": (build_lstm, LSTM_BATCH), "rcnn_att": (build_rcnn_att, RCNN_BATCH)}
LOSS_FN = None   # defined in main (after loading the data -> actual phase groups)


def main() -> None:
    global LOSS_FN
    import tensorflow as tf
    log.info("TF %s | GPUs: %s", tf.__version__, tf.config.list_physical_devices("GPU"))

    data = C.load_splits()
    pgroups = C.phase_groups(data["output_labels"])
    LOSS_FN = C.make_masked_loss(pgroups)
    log.info("X=%s Y=%s | n_features=%d output_dim=%d | phase groups=%d",
             data["X_train"].shape, data["Y_train"].shape,
             data["n_features"], data["output_dim"], len(pgroups))

    # start the val-NAE accumulator clean (avoids duplicated rows on re-runs)
    _vpath = C.AUDITS_DIR / "14_val_nae_per_output.csv"
    if _vpath.exists():
        _vpath.unlink()

    results = {}
    for name, (build_fn, batch) in ARCHITECTURES.items():
        log.info("\n########## Architecture: %s (batch=%d) ##########", name, batch)
        results[name] = C.train_and_evaluate(build_fn, name, data, LOSS_FN, batch, log,
                                              script_prefix="14")

    manifest = {
        "script": "14_train_baselines.py",
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "merge_de": ["18_train_lstm_baseline.py", "19_train_rcnn_attention.py"],
        "parameters": {"W": C.W, "seeds": C.SEEDS, "epochs_max": C.EPOCHS_MAX,
                       "patience": C.PATIENCE, "lr": C.LR,
                       "lstm_units": LSTM_UNITS, "lstm_batch": LSTM_BATCH,
                       "rcnn_batch": RCNN_BATCH, "loss": "masked_mae_phase_constraint"},
        "data": {"n_features": data["n_features"], "output_dim": data["output_dim"],
                 "output_labels": data["output_labels"]},
        "results": results,
    }
    (C.MANIFESTS_DIR / "14_train_baselines_params.json").write_text(json.dumps(manifest, indent=2))
    log.info("Manifest saved. Done.")


if __name__ == "__main__":
    main()
