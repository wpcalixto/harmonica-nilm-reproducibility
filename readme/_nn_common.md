# _nn_common.py

## Purpose

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

## Role

- File type: Shared utility module imported by the training scripts (14, 15, 18) and by the auxiliary training scripts; not an executable pipeline stage
- Not an executable stage: imported by `14_train_baselines.py`, `15_train_advanced.py`, `18_statistical_analysis.py`, `scripts/auxiliares/ablation_pilot.py` and `scripts/auxiliares/hpo_two_models.py`.

## Inputs

Files read by the source code (resolved from the path expressions in the script):

- `data/processed/13_scaler_Y.pkl`
- `data/processed/13_{…}.npy`
- `data/processed/windows/{_FS}/scaler_Y.pkl`
- `data/processed/windows/{_FS}/{…}.npy`
- `manifests/13_create_windows_splits_params.json`

## Outputs

Files written by the source code (resolved from the path expressions in the script):

- `audits/{…}_val_nae_per_output.csv`
- `figdata/{…}_{…}_training_history_long.csv`
- `metrics/ablation/{_FS}/{…}_{…}_metrics.csv`
- `predictions/ablation/{_FS}/{…}_{…}_predictions.parquet`

## Configuration

- No configuration file is read directly.
- Environment variables recognised: `FS_SET`, `NN_EPOCHS`, `NN_LR`, `NN_PATIENCE`, `NN_SEEDS`, `PYTHONHASHSEED`

## Dependencies

- Third-party packages: `joblib`, `keras`, `numpy`, `pandas`, `tensorflow`

## Usage

Not executed directly. Imported by the training scripts, which add `scripts/` to `sys.path` when needed:

```python
import _nn_common as C
```

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `audits/{…}_val_nae_per_output.csv`
- `checkpoints/script_12_{…}/12_{…}_checkpoint.pkl`
- `data/processed/13_scaler_Y.pkl`
- `data/processed/13_{…}.npy`
- `data/processed/windows/{_FS}/scaler_Y.pkl`
- `data/processed/windows/{_FS}/{…}.npy`
- `figdata/{…}_{…}_training_history_long.csv`
- `manifests/13_create_windows_splits_params.json`
- `metrics/ablation/{_FS}/{…}_{…}_metrics.csv`
- `models/ablation/{_FS}/{…}_{…}_seed_{…}.keras`
- `predictions/ablation/{_FS}/{…}_{…}_predictions.parquet`

## Reproducibility considerations

- Random seeds are fixed in the code/configuration (5 seeds: 42, 123, 456, 789, 1024 where applicable); do not change them.
- The environment variables above alter the run (e.g. smoke tests); leave them unset to reproduce the reference run.
- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
