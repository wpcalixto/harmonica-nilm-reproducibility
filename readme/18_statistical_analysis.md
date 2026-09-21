# 18_statistical_analysis.py

## Purpose

Statistical analysis of the article (18 outputs, 6 models):
  Op1 : Wilcoxon signed-rank + Cohen's d for all pairs of models (NAE per output)
  Op2 : Pearson(NAE, ERG), test of H0: rho = 0
  Op3 : Permutation Feature Importance (PFI), 5 seeds x 5 repetitions x 48 features
         I_f = Delta NAE_f / NAE_baseline   (reference model = LSTM baseline, 14_lstm)

Data sources:
  metrics/16_load_metrics_long.csv (nae_pct, e_true_kwh, e_pred_kwh per model x seed x output;
  systemic ERG per seed = |sum e_pred - sum e_true|/|sum e_true|*100); the 18 output labels
  come from the manifest via _nn_common; the windowed arrays via _nn_common.load_splits;
  LSTM models 14_lstm_seed_*; feature-selection manifest of script 11.

Op3 (PFI) needs to load .keras models, so it runs on the GPU environment where the models
were trained (TF 2.17). If the models cannot be loaded locally (Keras version skew) the PFI
is SKIPPED with a warning; Op1/Op2 always run (they only depend on metrics).

Inputs:  metrics/16_load_metrics_long.csv . 13_*.npy + scaler (via _nn_common) .
         manifests/11_feature_selection_rfecv_svr_kneedle_params.json . models/14_lstm_seed_*.keras
Outputs: metrics/18_statistical_tests.csv . 18_model_comparison_summary.csv . 18_pfi_results.csv
         metrics/18_model_comparison_long.csv . 18_pfi_top_features_long.csv
         manifests/18_statistical_analysis_params.json  (figdata/ is populated downstream)

## Pipeline role

- File type: Statistical analysis script (pipeline stage 18; Op3 requires the trained .keras models)
- Position in the canonical sequence: 19 of 29
- Preceding stage: `17_dg_balance.py`
- Following stage: `19_explainability_ablation.py`

## Inputs

Declared in the module header:

- `metrics/16_load_metrics_long.csv (nae_pct, e_true_kwh, e_pred_kwh per model x seed x output;`
- `systemic ERG per seed = |sum e_pred - sum e_true|/|sum e_true|*100); the 18 output labels`
- `come from the manifest via _nn_common; the windowed arrays via _nn_common.load_splits;`
- `LSTM models 14_lstm_seed_*; feature-selection manifest of script 11.`
- `metrics/16_load_metrics_long.csv`
- `13_*.npy + scaler (via _nn_common) .`
- `manifests/11_feature_selection_rfecv_svr_kneedle_params.json`
- `models/14_lstm_seed_*.keras`

Files read by the source code (resolved from the path expressions in the script):

- `checkpoints/18_pfi/18_pfi_checkpoint.json`
- `data/processed/11_candidate_sets.json`
- `metrics/16_load_metrics_long.csv`

## Outputs

Declared in the module header:

- `metrics/18_statistical_tests.csv`
- `18_model_comparison_summary.csv`
- `18_pfi_results.csv`
- `metrics/18_model_comparison_long.csv`
- `18_pfi_top_features_long.csv`
- `manifests/18_statistical_analysis_params.json  (figdata/ is populated downstream)`

Files written by the source code (resolved from the path expressions in the script):

- `checkpoints/18_pfi/18_pfi_checkpoint.json`
- `manifests/18_statistical_analysis_params.json`
- `metrics/18_model_comparison_long.csv`
- `metrics/18_model_comparison_summary.csv`
- `metrics/18_pfi_results.csv`
- `metrics/18_pfi_top_features_long.csv`
- `metrics/18_statistical_tests.csv`
- `metrics/18_statistical_tests_hpo_sensitivity.csv`

## Configuration

- No configuration file is read directly.

## Dependencies

- Local module: `_nn_common.py`
- Third-party packages: `keras`, `numpy`, `pandas`, `scipy`, `tensorflow`

## Usage

From the repository root:

```bash
python scripts/18_statistical_analysis.py
```

The script has no command-line interface.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `checkpoints/18_pfi/18_pfi_checkpoint.json`
- `data/processed/11_candidate_sets.json`
- `logs/18_statistical_analysis.log`
- `manifests/18_statistical_analysis_params.json`
- `metrics/16_load_metrics_long.csv`
- `metrics/18_model_comparison_long.csv`
- `metrics/18_model_comparison_summary.csv`
- `metrics/18_pfi_results.csv`
- `metrics/18_pfi_top_features_long.csv`
- `metrics/18_statistical_tests.csv`
- `metrics/18_statistical_tests_hpo_sensitivity.csv`
- `models/{…}_seed_{…}.keras`

## Reproducibility considerations

- Random seeds are fixed in the code/configuration (5 seeds: 42, 123, 456, 789, 1024 where applicable); do not change them.
- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
