# 19_explainability_ablation.py

## Purpose

Ablation of electrical feature families (two levels) + controls.

EXPLAINABILITY runner. Consumes the input-block map produced by script 12
(`audits/12_input_blocks_from_02b.csv`) and measures the effect of each electrical family G
on the disaggregation error, with chronological validation.

DESIGN (v2, without proxies). The channel audit of v2 did NOT classify any channel as a
proxy (is_proxy=0). Therefore the proxy-based analysis (X1 = X0+proxies; X2 = X0+residuals)
is NOT APPLICABLE and is reported as N/A, never as a numerical result. The families use ALL
their features (without the `& is_proxy` filter, which produced empty sets).

Two levels of analysis (12 auxiliary columns kept CONSTANT in every configuration):
  Main: restricted to the adopted model S48 (48 features), F_g^(48) = G_g intersect S48:
      . Addition    A_g   = X0 union F_g^(48)     (marginal value of the family over the base)
      . Removal     D_g   = S48 \ F_g^(48)        (loss when the family is removed from the final model)
  Supplementary: informational capacity with the raw features of the 177:
      . A_g^(177) = X0 union F_g^(177)            (NOT an ablation of the final model; reintroduces
                                                  attributes rejected in the selection)
  X0 = electrical base (powers G_P) + 12 auxiliary columns.

Controls (anti-artifact, proxy-independent):
  A0t   = X0 + normalised time index (spurious clock)
  Aperm = S48 with the non-base features PERMUTED in time (destroys the signal -> confirms
          genuine dependence; NAE must worsen)

Criterion: addition/supplement, keep if Delta J = J(A_g) - J(X0) <= -delta_J; removal,
  family IMPORTANT if Delta J = J(D_g) - J(S48) > delta_J. delta_J = 0 (main); sensitivity 0.01.

MANDATORY partition audit (aborts on failure): X0 union G_I union G_V union G_THD union
  G_hV union G_hI union G_sec = 177, no feature in two families, no orphan feature,
  q/s/cos (G_sec) present.

SURROGATE: HistGradientBoostingRegressor (fast, CPU) as a consistent instrument across
configurations. Confirmation with the deep models (Block B) is a separate GPU run.
SMOKE mode (env SMOKE=1): subsamples and reduces iterations/seeds.

Inputs:
    data/processed/10_original_features.parquet   (X: 177 features of M_G)
    data/processed/09_R1.parquet                  (Y: 18 loads; discards M_G_*)
    data/processed/11_candidate_sets.json         (adopted set S48)
    audits/12_input_blocks_from_02b.csv           (family map)
Outputs:
    metrics/19_ablation_blocks.csv
    figdata/19_ablation_blocks_long.csv
    manifests/19_explainability_ablation_params.json
    logs/19_explainability_ablation.log

## Pipeline role

- File type: Pipeline script (stage 19: ablation of electrical feature families + controls; surrogate model)
- Position in the canonical sequence: 20 of 29
- Preceding stage: `18_statistical_analysis.py`
- Following stage: `20_explainability_pfi_grouped.py`

## Inputs

Declared in the module header:

- `data/processed/10_original_features.parquet   (X: 177 features of M_G)`
- `data/processed/09_R1.parquet                  (Y: 18 loads; discards M_G_*)`
- `data/processed/11_candidate_sets.json         (adopted set S48)`
- `audits/12_input_blocks_from_02b.csv           (family map)`

Files read by the source code (resolved from the path expressions in the script):

- `audits/12_input_blocks_from_02b.csv`
- `data/processed/09_R1.parquet`
- `data/processed/10_original_features.parquet`
- `data/processed/11_candidate_sets.json`

## Outputs

Declared in the module header:

- `metrics/19_ablation_blocks.csv`
- `figdata/19_ablation_blocks_long.csv`
- `manifests/19_explainability_ablation_params.json`
- `logs/19_explainability_ablation.log`

Files written by the source code (resolved from the path expressions in the script):

- `figdata/19_ablation_blocks_long.csv`
- `manifests/19_explainability_ablation_params.json`
- `metrics/19_ablation_blocks.csv`

## Configuration

- No configuration file is read directly.
- Environment variables recognised: `SMOKE`

## Dependencies

- Third-party packages: `numpy`, `pandas`, `sklearn`

## Usage

From the repository root:

```bash
python scripts/19_explainability_ablation.py
```

The script has no command-line options; behaviour is controlled by the environment variables listed above.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `audits/12_input_blocks_from_02b.csv`
- `data/processed/09_R1.parquet`
- `data/processed/10_original_features.parquet`
- `data/processed/11_candidate_sets.json`
- `figdata/19_ablation_blocks_long.csv`
- `logs/19_explainability_ablation.log`
- `manifests/19_explainability_ablation_params.json`
- `metrics/19_ablation_blocks.csv`

## Reproducibility considerations

- Random seeds are fixed in the code/configuration (5 seeds: 42, 123, 456, 789, 1024 where applicable); do not change them.
- The environment variables above alter the run (e.g. smoke tests); leave them unset to reproduce the reference run.
- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
