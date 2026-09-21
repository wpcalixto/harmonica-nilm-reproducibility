# 09_build_model_datasets.py

## Purpose

Final construction of the modelling datasets.

Definitions
-----------
D_raw   = raw data of the file data/raw/dados_maior_v2_transformed.csv.
D_final = final dataset after alignment, cleaning, sentinel removal, outlier treatment
          and filling.
R0      = initial reference scenario, RESTRICTED to the meters that reach this stage
          (loads of the article {M2,M5,M6,M7,M10,M11,M14,M16} intersect available data =
          M5,M6,M7,M11,M16). It is not the full reproduction of the initial configuration:
          M2, M10 and M14 do not reach this stage (M2 ~ duplicate of M_G; M14 is an
          ELECTRICALLY DISTINCT load from M15 -> part of P_other; M10 sparse -> P_other).
R1      = corrected scenario: uses only the audited/selected independent loads
          (candidates of script 05): M5, M6, M7, M11, M15, M16.
R2      = corrected scenario with energy closure: R1 + P_other, where P_other = P01
          (meter 100 = M10+M12+M13+M14 aggregated in script 05).

Objective
---------
Execute only the tasks that still belong to the final post-processing:
  1. audit the energy preservation of the filling;
  2. mask dead/blocked channels in the model-ready base;
  3. build the 3 consolidated datasets R0, R1 and R2 (X = M_G; Y = loads of the scenario);
  4. write the scenario mapping and the manifest.

What this script does NOT do
----------------------------
This script ONLY audits the previous scope; it does NOT exclude meters anew. The scope
selection (including the representative of M8/M9/M17/M18 and the inclusion of M12/M13)
is decided by scripts 03-05. This stage uses the meters that reach it.

Expected official inputs
------------------------
  data/interim/06_para_preencher.csv       (pre-fill; or --pre-fill-file)
  data/processed/08_preenchido.parquet     (filled; or --filled-file)
  audits/05_meter_scope_preselection.csv   (scope; audit only)
  audits/06_dead_channels.csv              (or the equivalent produced upstream)

Main outputs
------------
  data/processed/09_R0.parquet             R0 (M_G + loads {5,6,7,11,16})
  data/processed/09_R1.parquet             R1 (M_G + loads {5,6,7,11,15,16})
  data/processed/09_R2.parquet             R2 (R1 + P_other = P01/meter 100)
  data/processed/09_model_ready.parquet    filled base (masked for dead channels)

  audits/09_scenario_mapping.csv           authoritative roles per column/scenario
  audits/09_energy_preservation_audit.csv  energy audit of the filling
  audits/09_gamma5_electrical.csv          M14 vs M15 electrical diagnostic

  manifests/09_build_model_datasets_params.json

Common usage
------------
  python scripts/09_build_model_datasets.py

With a specific filled file:
  python scripts/09_build_model_datasets.py --filled-file data/processed/08_preenchido.parquet

Automatically filter meters outside the approved scope:
  python scripts/09_build_model_datasets.py --filter-to-scope

## Pipeline role

- File type: Pipeline script (stage 09: final construction of the modelling datasets R0/R1/R2)
- Position in the canonical sequence: 10 of 29
- Preceding stage: `08_fill_gaps_no_clipping.py`
- Following stage: `10_post_audit_and_features.py`

## Inputs

Declared in the module header:

- `data/interim/06_para_preencher.csv       (pre-fill; or --pre-fill-file)`
- `data/processed/08_preenchido.parquet     (filled; or --filled-file)`
- `audits/05_meter_scope_preselection.csv   (scope; audit only)`
- `audits/06_dead_channels.csv              (or the equivalent produced upstream)`

Files read by the source code (resolved from the path expressions in the script):

- `data/interim/02_raw_standardized.parquet`

## Outputs

Declared in the module header:

- `data/processed/09_R0.parquet             R0 (M_G + loads {5,6,7,11,16})`
- `data/processed/09_R1.parquet             R1 (M_G + loads {5,6,7,11,15,16})`
- `data/processed/09_R2.parquet             R2 (R1 + P_other = P01/meter 100)`
- `data/processed/09_model_ready.parquet    filled base (masked for dead channels)`
- `audits/09_scenario_mapping.csv           authoritative roles per column/scenario`
- `audits/09_energy_preservation_audit.csv  energy audit of the filling`
- `audits/09_gamma5_electrical.csv          M14 vs M15 electrical diagnostic`
- `manifests/09_build_model_datasets_params.json`

Files written by the source code (resolved from the path expressions in the script):

- `audits/09_energy_preservation_audit.csv`
- `audits/09_gamma5_electrical.csv`
- `audits/09_scenario_mapping.csv`
- `data/processed/09_R0.parquet`
- `data/processed/09_R1.parquet`
- `data/processed/09_R2.parquet`
- `data/processed/09_model_ready.parquet`
- `manifests/09_build_model_datasets_params.json`

## Configuration

- No configuration file is read directly.
- Environment variables recognised: `HARMONIC_PROJECT_ROOT`

Command-line arguments (from `argparse` in the source):

| Argument | Default | Description |
|---|---|---|
| `--pre-fill-file` | None | Pre-filling file. Default: data/interim/06_para_preencher.csv |
| `--filled-file` | None | Filled file. Default: 08_preenchido.parquet; then 05_preenchido.parquet |
| `--scope-file` | None | Scope file. Default: audits/05_meter_scope_preselection.csv |
| `--dead-file` | None | Dead-channel file. Default: searches 05/04/03_dead_channels.csv |
| `--filter-to-scope` | off (flag) | Automatically filters meters outside the scope instead of aborting. |
| `--dead-pct` | DEAD_PCT_DEFAULT | Pre-fill NaN percentage to declare a channel dead by inference. Default: 99. |
| `--preserve-alert-pct` | PRESERVE_ALERT_PCT_DEFAULT | Alert for energy change at observed points. Default: 1%. |
| `--preserve-block-pct` | PRESERVE_BLOCK_PCT_DEFAULT | Block for energy change at observed points. Default: 5%. |
| `--near-zero-kwh` | NEAR_ZERO_KWH_DEFAULT | Energy below which the relative error does not block. Default: 1 kWh. |

## Dependencies

- Third-party packages: `numpy`, `pandas`

## Usage

From the repository root:

```bash
python scripts/09_build_model_datasets.py [options]
```

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `audits/03_dead_channels.csv`
- `audits/04_dead_channels.csv`
- `audits/05_meter_scope_preselection.csv`
- `audits/06A_dead_channels.csv`
- `audits/06_dead_channels.csv`
- `audits/07_dead_channels.csv`
- `audits/09_energy_preservation_audit.csv`
- `audits/09_gamma5_electrical.csv`
- `audits/09_scenario_mapping.csv`
- `data/interim/02_raw_standardized.parquet`
- `data/interim/06_para_preencher.csv`
- `data/processed/05_preenchido.parquet`
- `data/processed/08_preenchido.parquet`
- `data/processed/09_R0.parquet`
- `data/processed/09_R1.parquet`
- `data/processed/09_R2.parquet`
- `data/processed/09_model_ready.parquet`
- `logs/09_build_model_datasets.log`
- `manifests/09_build_model_datasets_params.json`

## Reproducibility considerations

- The environment variables above alter the run (e.g. smoke tests); leave them unset to reproduce the reference run.
- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
