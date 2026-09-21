# 11_feature_selection_rfecv_svr_kneedle.py

## Purpose

Feature selection RFECV-SVR + Kneedle.

Runs linear RFECV-SVR per output on the training split and applies the Kneedle elbow on
the normalised cumulative-importance curve to determine the diagnostic subsets of
features.

Methodology (MAIN variant):
  - X scaled on the training split (StandardScaler, no leakage);
  - Y scaled PER OUTPUT on the training split (StandardScaler) -> SVR coefficients
    comparable across loads of different magnitude;
  - TEMPORAL cross-validation (TimeSeriesSplit, n_splits=5) in the RFECV, with the folds
    defined on the ORIGINAL time grid and the target-validity mask applied inside each fold;
  - linear SVR with max_iter=20000 (convergence);
  - operational threshold: n* MG (>= IMPORTANCE_THRESHOLD cumulative) + 12 DG.
The predominance of voltage / voltage-harmonic variables persisted in all variants
(75-81%), confirming it as a finding rather than a scale artifact.

Inputs:
  data/processed/10_original_features.parquet   X (M_G, 177 features)
  data/processed/09_R1.parquet                  Y (scenario R1; the M_G_* columns are
                                                discarded: Y = only the loads Gamma_i)

Outputs:
  audits/11_selected_features_by_output.csv
  audits/11_feature_ranking.csv
  audits/11_rfecv_scores.csv
  audits/11_kneedle_selection.csv
  audits/11_global_importance.csv
  data/processed/11_candidate_sets.json          (candidate sets for the ablation)
  figdata/11_feature_ranking_long.csv            consumed by script 23
  logs/11_feature_selection_rfecv_svr_kneedle.log
  manifests/11_feature_selection_rfecv_svr_kneedle_params.json

Operations:
  1. Apply the selection on the training split only (chronological, no leakage).
  2. Run linear RFECV-SVR per output.
  3. Compute the performance curve per number of features.
  4. Apply Kneedle to determine n*.
  5. Rank by the absolute value of the SVR coefficients.
  6. Consolidate the candidate subsets (n* MG + 12 DG).
  7. Report n* and the total number of inputs.

## Pipeline role

- File type: Pipeline script (stage 11: feature selection RFECV-SVR + Kneedle)
- Position in the canonical sequence: 12 of 29
- Preceding stage: `10_post_audit_and_features.py`
- Following stage: `12_input_blocks_from_02b.py`

## Inputs

Declared in the module header:

- `data/processed/10_original_features.parquet   X (M_G, 177 features)`
- `data/processed/09_R1.parquet                  Y (scenario R1; the M_G_* columns are`
- `discarded: Y = only the loads Gamma_i)`

Files read by the source code (resolved from the path expressions in the script):

- `data/processed/08_fill_confidence.parquet`
- `data/processed/09_R1.parquet`
- `data/processed/10_original_features.parquet`

## Outputs

Declared in the module header:

- `audits/11_selected_features_by_output.csv`
- `audits/11_feature_ranking.csv`
- `audits/11_rfecv_scores.csv`
- `audits/11_kneedle_selection.csv`
- `audits/11_global_importance.csv`
- `data/processed/11_candidate_sets.json          (candidate sets for the ablation)`
- `figdata/11_feature_ranking_long.csv            consumed by script 23`
- `logs/11_feature_selection_rfecv_svr_kneedle.log`
- `manifests/11_feature_selection_rfecv_svr_kneedle_params.json`

Files written by the source code (resolved from the path expressions in the script):

- `audits/11_convergence_by_output.csv`
- `audits/11_feature_ranking.csv`
- `audits/11_global_importance.csv`
- `audits/11_kneedle_selection.csv`
- `audits/11_rfecv_scores.csv`
- `audits/11_selected_features_by_output.csv`
- `data/processed/11_candidate_sets.json`
- `figdata/11_feature_ranking_long.csv`
- `manifests/11_feature_selection_rfecv_svr_kneedle_params.json`

## Configuration

- No configuration file is read directly.
- Environment variables recognised: `FS_MAX_ITER`, `FS_SCOPE`

## Dependencies

- Third-party packages: `numpy`, `pandas`, `sklearn`

## Usage

From the repository root:

```bash
python scripts/11_feature_selection_rfecv_svr_kneedle.py
```

The script has no command-line options; behaviour is controlled by the environment variables listed above.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `audits/11_convergence_by_output.csv`
- `audits/11_feature_ranking.csv`
- `audits/11_global_importance.csv`
- `audits/11_kneedle_selection.csv`
- `audits/11_rfecv_scores.csv`
- `audits/11_selected_features_by_output.csv`
- `data/processed/08_fill_confidence.parquet`
- `data/processed/09_R1.parquet`
- `data/processed/10_original_features.parquet`
- `data/processed/11_candidate_sets.json`
- `figdata/11_feature_ranking_long.csv`
- `logs/11_feature_selection_rfecv_svr_kneedle.log`
- `manifests/11_feature_selection_rfecv_svr_kneedle_params.json`

## Reproducibility considerations

- The environment variables above alter the run (e.g. smoke tests); leave them unset to reproduce the reference run.
- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
