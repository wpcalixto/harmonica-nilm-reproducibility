# 13_create_windows_splits.py

## Purpose

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

## Pipeline role

- File type: Data preprocessing script (pipeline stage 13: windowing and chronological split)
- Position in the canonical sequence: 14 of 29
- Preceding stage: `12_input_blocks_from_02b.py`
- Following stage: `14_train_baselines.py`

## Inputs

Declared in the module header:

- `data/processed/10_original_features.parquet   (N, 177) features of M_G`
- `data/processed/09_R1.parquet                  Y (scenario R1; the M_G_* columns are`
- `discarded -> 18 loads in Watts)`
- `manifests/11_feature_selection_rfecv_svr_kneedle_params.json  48 MG + 12 DG = 60`

Files read by the source code (resolved from the path expressions in the script):

- `data/processed/09_R1.parquet`
- `data/processed/10_original_features.parquet`
- `data/processed/11_candidate_sets.json`

## Outputs

Declared in the module header:

- `data/processed/13_X_train.npy  (N_tr, W, 60)`
- `data/processed/13_Y_train.npy  (N_tr, W, 18)`
- `data/processed/13_X_val.npy    (N_va, W, 60)`
- `data/processed/13_Y_val.npy    (N_va, W, 18)`
- `data/processed/13_X_test.npy   (N_te, W, 60)`
- `data/processed/13_Y_test.npy   (N_te, W, 18)`
- `data/processed/13_Y_raw_test.npy  (n_test_ts, 18) Watts`
- `data/processed/13_scaler_X.pkl    MinMaxScaler fitted on the 48 MG features (train)`
- `data/processed/13_scaler_Y.pkl    MinMaxScaler fitted on the 18 loads (train, Watts)`
- `audits/13_split_index.csv`
- `audits/13_window_metadata.csv`
- `figdata/13_split_timeline.csv`
- `logs/13_create_windows_splits.log`
- `manifests/13_create_windows_splits_params.json`

Files written by the source code (resolved from the path expressions in the script):

- `audits/13_split_index.csv`
- `audits/13_window_metadata.csv`
- `data/processed/windows/{FS_SET}/X_test.npy`
- `data/processed/windows/{FS_SET}/X_train.npy`
- `data/processed/windows/{FS_SET}/X_val.npy`
- `data/processed/windows/{FS_SET}/Y_raw_test.npy`
- `data/processed/windows/{FS_SET}/Y_test.npy`
- `data/processed/windows/{FS_SET}/Y_train.npy`
- `data/processed/windows/{FS_SET}/Y_val.npy`
- `data/processed/windows/{FS_SET}/manifest.json`
- `data/processed/{…}.npy`
- `figdata/13_split_timeline.csv`
- `manifests/13_create_windows_splits_params.json`

## Configuration

- No configuration file is read directly.
- Environment variables recognised: `FS_SET`

## Dependencies

- Third-party packages: `joblib`, `numpy`, `pandas`, `sklearn`

## Usage

From the repository root:

```bash
python scripts/13_create_windows_splits.py
```

The script has no command-line options; behaviour is controlled by the environment variables listed above.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `audits/13_split_index.csv`
- `audits/13_window_metadata.csv`
- `data/processed/09_R1.parquet`
- `data/processed/10_original_features.parquet`
- `data/processed/11_candidate_sets.json`
- `data/processed/13_X_test.npy`
- `data/processed/13_X_train.npy`
- `data/processed/13_X_val.npy`
- `data/processed/13_Y_raw_test.npy`
- `data/processed/13_Y_test.npy`
- `data/processed/13_Y_train.npy`
- `data/processed/13_Y_val.npy`
- `data/processed/13_scaler_X.pkl`
- `data/processed/13_scaler_Y.pkl`
- `data/processed/windows/{FS_SET}/X_test.npy`
- `data/processed/windows/{FS_SET}/X_train.npy`
- `data/processed/windows/{FS_SET}/X_val.npy`
- `data/processed/windows/{FS_SET}/Y_raw_test.npy`
- `data/processed/windows/{FS_SET}/Y_test.npy`
- `data/processed/windows/{FS_SET}/Y_train.npy`
- `data/processed/windows/{FS_SET}/Y_val.npy`
- `data/processed/windows/{FS_SET}/manifest.json`
- `data/processed/windows/{FS_SET}/scaler_X.pkl`
- `data/processed/windows/{FS_SET}/scaler_Y.pkl`
- `data/processed/{…}.npy`
- `figdata/13_split_timeline.csv`
- `logs/13_create_windows_splits.log`
- `manifests/13_create_windows_splits_params.json`

## Reproducibility considerations

- The environment variables above alter the run (e.g. smoke tests); leave them unset to reproduce the reference run.
- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
