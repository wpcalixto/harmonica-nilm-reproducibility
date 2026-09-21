# 15_train_advanced.py

## Purpose

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

## Pipeline role

- File type: Training script (pipeline stage 15: advanced architectures, SPEC, MoTE_v2, Ensemble_G4; optional HPO sensitivity)
- Position in the canonical sequence: 16 of 29
- Preceding stage: `14_train_baselines.py`
- Following stage: `16_evaluate.py`

## Inputs

Files read by the source code (resolved from the path expressions in the script):

- `audits/14_val_nae_per_output.csv`
- `predictions/14_lstm_predictions.parquet`
- `predictions/14_rcnn_att_predictions.parquet`

## Outputs

Declared in the module header:

- `models/15_{pe_es,spec,mote_v2}_seed_*.keras`
- `predictions/15_*_predictions.parquet`
- `metrics/15_*_metrics.csv`
- `figdata/15_*_history_long.csv`
- `audits/15_*`
- `manifest.`
- `models/15_exp_pe_es_optuna_seed_*.keras`
- `predictions/15_exp_pe_es_optuna_predictions.parquet`
- `metrics/15_exp_pe_es_optuna_metrics.csv`
- `figdata/15_exp_pe_es_optuna_history_long.csv`
- `audits/15_exp_optuna_trials.csv`
- `manifests/15_exp_pe_es_optuna_params.json`
- `checkpoints/15_exp_pe_es_optuna/optuna_study.db`
- `Overrides for smoke tests: OPTUNA_TPE, OPTUNA_CMA, OPTUNA_SEARCH_PATIENCE, NN_EPOCHS, NN_PATIENCE, NN_SEEDS.`

Files written by the source code (resolved from the path expressions in the script):

- `audits/15_ensemble_choice.csv`
- `audits/15_exp_optuna_trials.csv`
- `audits/15_spec_hard_loads.csv`
- `audits/15_window_selection.csv`
- `figdata/15_exp_pe_es_optuna_history_long.csv`
- `figdata/15_{…}_training_history_long.csv`
- `manifests/15_exp_pe_es_optuna_params.json`
- `manifests/15_train_advanced_params.json`
- `metrics/15_ensemble_g4_metrics.csv`
- `metrics/15_exp_pe_es_optuna_metrics.csv`
- `metrics/15_{…}_metrics.csv`
- `models/15_exp_pe_es_optuna_seed_{…}.keras`
- `models/15_{…}_seed_{…}.keras`
- `predictions/15_ensemble_g4_predictions.parquet`
- `predictions/15_exp_pe_es_optuna_predictions.parquet`
- `predictions/15_{…}_predictions.parquet`

## Configuration

- No configuration file is read directly.
- Environment variables recognised: `OPTUNA_CMA`, `OPTUNA_SEARCH_PATIENCE`, `OPTUNA_TPE`, `RUN_OPTUNA`

## Dependencies

- Local module: `_nn_common.py`
- Third-party packages: `keras`, `numpy`, `optuna`, `pandas`, `tensorflow`
- A CUDA-capable GPU is required for a practical runtime (model training).

## Usage

From the repository root:

```bash
python scripts/15_train_advanced.py
```

The script has no command-line options; behaviour is controlled by the environment variables listed above.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `audits/14_val_nae_per_output.csv`
- `audits/15_ensemble_choice.csv`
- `audits/15_exp_optuna_trials.csv`
- `audits/15_spec_hard_loads.csv`
- `audits/15_window_selection.csv`
- `checkpoints/15_exp_pe_es_optuna/optuna_study.db`
- `figdata/15_exp_pe_es_optuna_history_long.csv`
- `figdata/15_{…}_training_history_long.csv`
- `logs/15_train_advanced.log`
- `manifests/15_exp_pe_es_optuna_params.json`
- `manifests/15_train_advanced_params.json`
- `metrics/15_ensemble_g4_metrics.csv`
- `metrics/15_exp_pe_es_optuna_metrics.csv`
- `metrics/15_{…}_metrics.csv`
- `models/15_exp_pe_es_optuna_seed_{…}.keras`
- `models/15_{…}_seed_{…}.keras`
- `predictions/14_lstm_predictions.parquet`
- `predictions/14_rcnn_att_predictions.parquet`
- `predictions/15_ensemble_g4_predictions.parquet`
- `predictions/15_exp_pe_es_optuna_predictions.parquet`
- `predictions/15_{…}_predictions.parquet`

## Reproducibility considerations

- Random seeds are fixed in the code/configuration (5 seeds: 42, 123, 456, 789, 1024 where applicable); do not change them.
- Trained models and predictions are provided in `models/` and `predictions/`; retraining on other hardware may differ at floating-point level.
- The environment variables above alter the run (e.g. smoke tests); leave them unset to reproduce the reference run.
- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
