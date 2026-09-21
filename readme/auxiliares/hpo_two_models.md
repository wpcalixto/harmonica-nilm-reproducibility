# hpo_two_models.py

## Purpose

Supplementary FAIR hyperparameter optimisation (HPO): PE-ES (proposed) + LSTM (competitor
selected by ERG_val).

PRIMARY objective = VALIDATION ERG (aligned with the selection criterion of the competitor);
NAE_val as secondary metric. EQUAL GPU-hour budget per architecture (timeout), same sampler
(TPE, fixed seed), same pruner (none), same early-stopping patience (fixed), test split LOCKED
during the search. Protocol against single-seed bias:
  search (1 seed) -> top-3 -> re-evaluate top-3 on 3 seeds (ERG_val) -> freeze the best
  -> retrain 5 seeds -> only then TEST.
Default x HPO comparison per model and between finalists (delta ERG abs/rel, paired by seed).

Framing: SUPPLEMENTARY/EXPLORATORY analysis; conclusions restricted to the two models.

Env: SMOKE=1 (short run), GPU_BUDGET_S=<seconds per model, default 5400>.
Outputs: audits/hpo2_*.csv/json ; models/hpo2_*_seed_*.keras

## Role

- File type: Auxiliary hyperparameter-optimisation and training script (supplementary analysis; requires GPU)
- Auxiliary script: not part of the canonical sequential pipeline (01-28). It reads artifacts produced by the pipeline and writes its own outputs.

## Inputs

Files read by the source code (resolved from the path expressions in the script):

- `manifests/15_train_advanced_params.json`

## Outputs

Declared in the module header:

- `audits/hpo2_*.csv/json`
- `models/hpo2_*_seed_*.keras`

Files written by the source code (resolved from the path expressions in the script):

- `audits/hpo2_comparison.csv`
- `audits/hpo2_delta_by_model.csv`
- `audits/hpo2_final_per_seed.csv`
- `audits/hpo2_finalists_delta.csv`
- `audits/hpo2_manifest.json`
- `audits/hpo2_trials_{…}.csv`
- `models/hpo2_{…}_seed_{…}.keras`

## Configuration

- No configuration file is read directly.
- Environment variables recognised: `GPU_BUDGET_S`, `SMOKE`

## Dependencies

- Local module: `_nn_common.py`
- Third-party packages: `keras`, `numpy`, `optuna`, `pandas`
- A CUDA-capable GPU is required for a practical runtime (model training).

## Usage

From the repository root:

```bash
python scripts/auxiliares/hpo_two_models.py
```

The script has no command-line options; behaviour is controlled by the environment variables listed above.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `audits/hpo2_comparison.csv`
- `audits/hpo2_delta_by_model.csv`
- `audits/hpo2_final_per_seed.csv`
- `audits/hpo2_finalists_delta.csv`
- `audits/hpo2_manifest.json`
- `audits/hpo2_trials_{…}.csv`
- `checkpoints/hpo2/{…}_study.db`
- `logs/hpo_two_models.log`
- `manifests/15_train_advanced_params.json`
- `models/hpo2_{…}_seed_{…}.keras`
- `models/{…}_seed_{…}.keras`

## Reproducibility considerations

- Random seeds are fixed in the code/configuration (5 seeds: 42, 123, 456, 789, 1024 where applicable); do not change them.
- Trained models and predictions are provided in `models/` and `predictions/`; retraining on other hardware may differ at floating-point level.
- The environment variables above alter the run (e.g. smoke tests); leave them unset to reproduce the reference run.
- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
