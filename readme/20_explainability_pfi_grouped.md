# 20_explainability_pfi_grouped.py

## Purpose

GROUPED permutation importance (I_g) per feature family G.

Complements the per-feature PFI of script 18 (which, under correlated features,
underestimates redundant families) with the JOINT permutation of each family G on the
ADOPTED MODEL:

    I_g = (NAE_perm(g) - NAE_base) / NAE_base

permuting in a block all columns of family g present in the adopted set
(F_g^(48) = G_g intersect S48). Preserves the intra-family correlation and breaks only the
family-target relation. Reports mean and dispersion over seeds x repetitions.

DESIGN (v2, without proxies). The channel audit of v2 did NOT classify any channel as a
proxy (is_proxy=0); the proxy-based analysis is N/A. The families use ALL their features
of the adopted set (without the `& is_proxy` filter, which produced empty groups). The
base model is the surrogate trained on the adopted S48 (48 features + 12 auxiliary
columns), mirroring the input of the deployed model. Confirmation with the deep model is a
separate GPU run. SMOKE reduces the cost.

MANDATORY audit of the partition of the 177 features by family (aborts if invalid).

Inputs:
    data/processed/10_original_features.parquet . data/processed/09_R1.parquet
    data/processed/11_candidate_sets.json (adopted set S48) . audits/12_input_blocks_from_02b.csv
Outputs:
    metrics/20_pfi_grouped.csv . figdata/20_pfi_grouped_long.csv
    manifests/20_explainability_pfi_grouped_params.json . logs/20_explainability_pfi_grouped.log

## Pipeline role

- File type: Pipeline script (stage 20: grouped permutation importance per feature family)
- Position in the canonical sequence: 21 of 29
- Preceding stage: `19_explainability_ablation.py`
- Following stage: `21_economic_analysis.py`

## Inputs

Declared in the module header:

- `data/processed/10_original_features.parquet`
- `data/processed/09_R1.parquet`
- `data/processed/11_candidate_sets.json (adopted set S48)`
- `audits/12_input_blocks_from_02b.csv`

Files read by the source code (resolved from the path expressions in the script):

- `audits/12_input_blocks_from_02b.csv`
- `data/processed/09_R1.parquet`
- `data/processed/10_original_features.parquet`
- `data/processed/11_candidate_sets.json`

## Outputs

Declared in the module header:

- `metrics/20_pfi_grouped.csv`
- `figdata/20_pfi_grouped_long.csv`
- `manifests/20_explainability_pfi_grouped_params.json`
- `logs/20_explainability_pfi_grouped.log`

Files written by the source code (resolved from the path expressions in the script):

- `figdata/20_pfi_grouped_long.csv`
- `manifests/20_explainability_pfi_grouped_params.json`
- `metrics/20_pfi_grouped.csv`

## Configuration

- No configuration file is read directly.
- Environment variables recognised: `SMOKE`

## Dependencies

- Third-party packages: `numpy`, `pandas`, `sklearn`

## Usage

From the repository root:

```bash
python scripts/20_explainability_pfi_grouped.py
```

The script has no command-line options; behaviour is controlled by the environment variables listed above.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `audits/12_input_blocks_from_02b.csv`
- `data/processed/09_R1.parquet`
- `data/processed/10_original_features.parquet`
- `data/processed/11_candidate_sets.json`
- `figdata/20_pfi_grouped_long.csv`
- `logs/20_explainability_pfi_grouped.log`
- `manifests/20_explainability_pfi_grouped_params.json`
- `metrics/20_pfi_grouped.csv`

## Reproducibility considerations

- Random seeds are fixed in the code/configuration (5 seeds: 42, 123, 456, 789, 1024 where applicable); do not change them.
- The environment variables above alter the run (e.g. smoke tests); leave them unset to reproduce the reference run.
- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
