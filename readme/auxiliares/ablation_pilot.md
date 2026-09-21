# ablation_pilot.py

## Purpose

ABLATION (pilot/confirmation): the decision is taken ONLY on the VALIDATION split
(the test split is locked).

For the feature set FS_SET (environment variable), trains the LSTM baseline (same
architecture as script 14: 2x BiLSTM(100) + TimeDistributed Dense) x NN_SEEDS, with early
stopping on val_loss. For each seed it computes:
  - GLOBAL validation NAE (aggregated over the 18 outputs): sum|y_hat - y| / sum|y|
    (primary metric);
  - per-output NAE (stored per output).
Architecture/hyperparameters are CONSTANT across feature sets (isolates the effect of the
set). X_test is NEVER used.
Writes results/ablation/<FS_SET>/{val_per_output.csv, val_global.csv, summary.json}.

Overrides: NN_SEEDS, NN_EPOCHS, NN_PATIENCE, NN_LR. FS_SET is mandatory.
Confirmation run = defaults (5 seeds 42,123,456,789,1024; 500 epochs; patience 30).

## Role

- File type: Auxiliary training script (feature-set ablation; decision on the validation split only)
- Auxiliary script: not part of the canonical sequential pipeline (01-28). It reads artifacts produced by the pipeline and writes its own outputs.

## Inputs

Files read by the source code (resolved from the path expressions in the script):

- none detected by static analysis (inputs may be passed through shared helpers or environment variables)

## Outputs

Files written by the source code (resolved from the path expressions in the script):

- `results/ablation/{FS}/summary.json`
- `results/ablation/{FS}/val_global.csv`
- `results/ablation/{FS}/val_per_output.csv`

## Configuration

- No configuration file is read directly.
- Environment variables recognised: `FS_SET`

## Dependencies

- Local module: `_nn_common.py`
- Loads by path: `14_train_baselines.py`
- Third-party packages: `keras`, `numpy`, `pandas`
- A CUDA-capable GPU is required for a practical runtime (model training).

## Usage

From the repository root:

```bash
python scripts/auxiliares/ablation_pilot.py
```

The script has no command-line options; behaviour is controlled by the environment variables listed above.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `results/ablation/{FS}/summary.json`
- `results/ablation/{FS}/val_global.csv`
- `results/ablation/{FS}/val_per_output.csv`

## Reproducibility considerations

- Random seeds are fixed in the code/configuration (5 seeds: 42, 123, 456, 789, 1024 where applicable); do not change them.
- Trained models and predictions are provided in `models/` and `predictions/`; retraining on other hardware may differ at floating-point level.
- The environment variables above alter the run (e.g. smoke tests); leave them unset to reproduce the reference run.
- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
