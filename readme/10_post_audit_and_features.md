# 10_post_audit_and_features.py

## Purpose

Post-filling audits + feature matrix X.

Consolidates three pieces of the pipeline design:

  SECTION A: energy audit ACROSS STAGES.
    Measures the electrical energy (kWh) per meter x phase along the actual stages:
      raw       -> data/interim/02_raw_standardized.parquet   (IRREGULAR timestamps)
      gridded   -> data/interim/06_para_preencher.csv          (1-min grid; sentinels/outliers/gaps = NaN)
      sanitized -> data/interim/06_para_preencher_sanitizado.csv
      filled    -> data/processed/08_preenchido.parquet
    On the raw data it uses trapezoidal integration over the actual dt (energy_nonuniform);
    on the others, the sum over the 1-min grid (energy_uniform). Captures distortion
    introduced by the gridding itself, which script 08 (pre-fill vs fill, both on the grid)
    does not detect.

  SECTION B: energy composition of P_other / P01.
    Breaks down the energy of P01 (meter 100 = M10+M12+M13+M14) per component and per
    phase, from the raw data (operational rows i_an>0.5, secondary scale of v2), and compares
    it with the final energy of P01 after filling.

  SECTION C: feature matrix X.
    Extracts the 177 original features of M_G (meter 1) from the filled data (script 08),
    without inserting new attributes.

Outputs:
  audits/10_energy_preservation_by_stage.csv
  audits/10_energy_preservation_alerts.csv
  figdata/10_raw_vs_processed_energy_long.csv
  audits/10_p_other_components.csv
  audits/10_p_other_energy_by_phase.csv
  data/processed/10_original_features.parquet
  audits/10_feature_dictionary.csv
  logs/10_post_audit_and_features.log
  manifests/10_post_audit_and_features_params.json

Each section is independent: if an input is missing, the section is skipped with a warning.

## Pipeline role

- File type: Pipeline script (stage 10: post-filling audits + M_G feature matrix)
- Position in the canonical sequence: 11 of 29
- Preceding stage: `09_build_model_datasets.py`
- Following stage: `11_feature_selection_rfecv_svr_kneedle.py`

## Inputs

Files read by the source code (resolved from the path expressions in the script):

- none detected by static analysis (inputs may be passed through shared helpers or environment variables)

## Outputs

Declared in the module header:

- `audits/10_energy_preservation_by_stage.csv`
- `audits/10_energy_preservation_alerts.csv`
- `figdata/10_raw_vs_processed_energy_long.csv`
- `audits/10_p_other_components.csv`
- `audits/10_p_other_energy_by_phase.csv`
- `data/processed/10_original_features.parquet`
- `audits/10_feature_dictionary.csv`
- `logs/10_post_audit_and_features.log`
- `manifests/10_post_audit_and_features_params.json`

Files written by the source code (resolved from the path expressions in the script):

- `audits/10_energy_preservation_alerts.csv`
- `audits/10_energy_preservation_by_stage.csv`
- `audits/10_feature_dictionary.csv`
- `audits/10_p_other_components.csv`
- `audits/10_p_other_energy_by_phase.csv`
- `data/processed/10_original_features.parquet`
- `figdata/10_raw_vs_processed_energy_long.csv`
- `manifests/10_post_audit_and_features_params.json`

## Configuration

- No configuration file is read directly.

## Dependencies

- Third-party packages: `numpy`, `pandas`

## Usage

From the repository root:

```bash
python scripts/10_post_audit_and_features.py
```

The script has no command-line interface.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `audits/10_energy_preservation_alerts.csv`
- `audits/10_energy_preservation_by_stage.csv`
- `audits/10_feature_dictionary.csv`
- `audits/10_p_other_components.csv`
- `audits/10_p_other_energy_by_phase.csv`
- `data/interim/02_raw_standardized.parquet`
- `data/interim/06_para_preencher.csv`
- `data/interim/06_para_preencher_sanitizado.csv`
- `data/processed/08_preenchido.parquet`
- `data/processed/10_original_features.parquet`
- `figdata/10_raw_vs_processed_energy_long.csv`
- `logs/10_post_audit_and_features.log`
- `manifests/10_post_audit_and_features_params.json`

## Reproducibility considerations

- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
