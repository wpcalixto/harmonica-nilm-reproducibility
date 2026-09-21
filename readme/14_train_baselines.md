# 14_train_baselines.py

## Purpose

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

## Pipeline role

- File type: Training script (pipeline stage 14: baseline architectures)
- Position in the canonical sequence: 15 of 29
- Preceding stage: `13_create_windows_splits.py`
- Following stage: `15_train_advanced.py`

## Inputs

Declared in the module header:

- `data/processed/13_X_{train,val,test}.npy, 13_Y_{train,val,test}.npy`
- `data/processed/13_Y_raw_test.npy, 13_scaler_Y.pkl`
- `manifests/13_create_windows_splits_params.json   (output labels)`

Files read by the source code (resolved from the path expressions in the script):

- none detected by static analysis (inputs may be passed through shared helpers or environment variables)

## Outputs

Declared in the module header:

- `models/14_{arch}_seed_*.keras`
- `predictions/14_{arch}_predictions.parquet`
- `metrics/14_{arch}_metrics.csv`
- `figdata/14_{arch}_training_history_long.csv`
- `logs/14_train_baselines.log`
- `manifests/14_train_baselines_params.json`

Files written by the source code (resolved from the path expressions in the script):

- `manifests/14_train_baselines_params.json`

## Configuration

- No configuration file is read directly.

## Dependencies

- Local module: `_nn_common.py`
- Third-party packages: `keras`, `tensorflow`
- A CUDA-capable GPU is required for a practical runtime (model training).

## Usage

From the repository root:

```bash
python scripts/14_train_baselines.py
```

The script has no command-line interface.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `audits/14_val_nae_per_output.csv`
- `logs/14_train_baselines.log`
- `manifests/14_train_baselines_params.json`

## Reproducibility considerations

- Random seeds are fixed in the code/configuration (5 seeds: 42, 123, 456, 789, 1024 where applicable); do not change them.
- Trained models and predictions are provided in `models/` and `predictions/`; retraining on other hardware may differ at floating-point level.
- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
