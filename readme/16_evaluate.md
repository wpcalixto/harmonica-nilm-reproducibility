# 16_evaluate.py

## Purpose

Predictive and energy evaluation, consuming saved PREDICTIONS.

Evaluates the 6 models of Block B from the saved predictions (Watts), WITHOUT loading
.keras models (independent of the Keras version):
    LSTM         predictions/14_lstm_predictions.parquet         (W=12)
    RCNN_att     predictions/14_rcnn_att_predictions.parquet     (W=12)
    PE_ES        predictions/15_pe_es_predictions.parquet        (W=W*)
    SPEC         predictions/15_spec_predictions.parquet         (W=W*)
    MoTE_v2      predictions/15_mote_v2_predictions.parquet      (W=36)
    Ensemble_G4  predictions/15_ensemble_g4_predictions.parquet  (W=12)

Metrics per output (mean +/- sd over seeds), per three-phase load and systemic:
    NAE% . MAE . energy_abs_error_contrib% . Delta_reduction(kWh) . ERG% (energy).
    (energy_abs_error_contrib% = |E_true - E_pred| / sum|E_true| per output: contribution of
     the absolute energy error, NOT an "accuracy"; AAE = 100 - ERG is derived in script 18.)
The evaluated models include PE-ES-Optuna (15_exp_*, HPO sensitivity, W from result.W_star).
Reconstruction: every predicted series is aligned on the original time axis by the window
mid-point (offset = W//2) before any metric. W* is read from the manifest of script 13.

Inputs: data/processed/13_Y_raw_test.npy + manifest 11 (labels); manifest 13 (W*);
        predictions/{14,15}_*_predictions.parquet.
Outputs: metrics/16_load_metrics_long.csv . 16_metrics_by_output.csv
         metrics/16_energy_by_load.csv . 16_system_erg.csv
         figdata/16_nae_by_model_long.csv . manifests/16_evaluate_params.json

## Pipeline role

- File type: Evaluation script (pipeline stage 16: predictive and energy metrics from saved predictions)
- Position in the canonical sequence: 17 of 29
- Preceding stage: `15_train_advanced.py`
- Following stage: `17_dg_balance.py`

## Inputs

Declared in the module header:

- `data/processed/13_Y_raw_test.npy + manifest 11 (labels)`
- `manifest 13 (W*)`
- `predictions/{14,15}_*_predictions.parquet.`

Files read by the source code (resolved from the path expressions in the script):

- `data/processed/13_Y_raw_test.npy`
- `manifests/13_create_windows_splits_params.json`
- `manifests/15_exp_pe_es_optuna_params.json`
- `manifests/15_train_advanced_params.json`
- `predictions/{…}_predictions.parquet`

## Outputs

Declared in the module header:

- `metrics/16_load_metrics_long.csv`
- `16_metrics_by_output.csv`
- `metrics/16_energy_by_load.csv`
- `16_system_erg.csv`
- `figdata/16_nae_by_model_long.csv`
- `manifests/16_evaluate_params.json`

Files written by the source code (resolved from the path expressions in the script):

- `figdata/16_nae_by_model_long.csv`
- `manifests/16_evaluate_params.json`
- `metrics/16_energy_by_load.csv`
- `metrics/16_load_metrics_long.csv`
- `metrics/16_metrics_by_output.csv`
- `metrics/16_system_erg.csv`

## Configuration

- No configuration file is read directly.

## Dependencies

- Third-party packages: `numpy`, `pandas`

## Usage

From the repository root:

```bash
python scripts/16_evaluate.py
```

The script has no command-line interface.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `data/processed/13_Y_raw_test.npy`
- `figdata/16_nae_by_model_long.csv`
- `logs/16_evaluate.log`
- `manifests/13_create_windows_splits_params.json`
- `manifests/15_exp_pe_es_optuna_params.json`
- `manifests/15_train_advanced_params.json`
- `manifests/16_evaluate_params.json`
- `metrics/16_energy_by_load.csv`
- `metrics/16_load_metrics_long.csv`
- `metrics/16_metrics_by_output.csv`
- `metrics/16_system_erg.csv`
- `predictions/{…}_predictions.parquet`

## Reproducibility considerations

- Random seeds are fixed in the code/configuration (5 seeds: 42, 123, 456, 789, 1024 where applicable); do not change them.
- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
