# 21_economic_analysis.py

## Purpose

Economic analysis with the corrected balance.

  Op1   : energy per load (from 16_load_metrics_long)
  Op2-4 : C_real = E_tot*c_Gr ; C_hat = E_hat_tot*c_Gr ; C_error = E_error*c_Gr ;
          Delta_C = |C_real - C_hat|/C_real ; C_yearly = C_error*(T_yearly/T_eval)
  Op5   : split observed / predicted / computed_by_balance / inferred_P_other / M_G
  Op6   : propagate the D_G uncertainty into the cost (from 17_dg_uncertainty_interval)

Data sources:
  metrics/16_load_metrics_long.csv (e_true_kwh, e_pred_kwh)
  dg/17_dg_energy_by_phase.csv (E_load_obs_kWh, E_load_pred_kWh, E_DG_*_kWh)
  dg/17_dg_uncertainty_interval.csv (u_total_kWh per phase)
  T_eval = n_test/60 (from manifest 11)
  Outputs in metrics/ (analysis CSVs); figdata/tabdata are populated downstream.

Inputs:  metrics/16_load_metrics_long.csv . dg/17_dg_energy_by_phase.csv .
         dg/17_dg_uncertainty_interval.csv . manifests/11_*_params.json
Outputs: metrics/21_economic_results.csv . 21_economic_sensitivity.csv .
         21_economic_sensitivity_long.csv . manifests/21_economic_analysis_params.json

## Pipeline role

- File type: Pipeline script (stage 21: economic analysis)
- Position in the canonical sequence: 22 of 29
- Preceding stage: `20_explainability_pfi_grouped.py`
- Following stage: `22_build_tables.py`

## Inputs

Declared in the module header:

- `metrics/16_load_metrics_long.csv (e_true_kwh, e_pred_kwh)`
- `dg/17_dg_energy_by_phase.csv (E_load_obs_kWh, E_load_pred_kWh, E_DG_*_kWh)`
- `dg/17_dg_uncertainty_interval.csv (u_total_kWh per phase)`
- `T_eval = n_test/60 (from manifest 11)`
- `metrics/16_load_metrics_long.csv`
- `dg/17_dg_energy_by_phase.csv .`
- `dg/17_dg_uncertainty_interval.csv`
- `manifests/11_*_params.json`

Files read by the source code (resolved from the path expressions in the script):

- `dg/17_dg_energy_by_phase.csv`
- `dg/17_dg_uncertainty_interval.csv`
- `manifests/13_create_windows_splits_params.json`
- `metrics/16_load_metrics_long.csv`

## Outputs

Declared in the module header:

- `metrics/21_economic_results.csv`
- `21_economic_sensitivity.csv .`
- `21_economic_sensitivity_long.csv`
- `manifests/21_economic_analysis_params.json`

Files written by the source code (resolved from the path expressions in the script):

- `manifests/21_economic_analysis_params.json`
- `metrics/21_economic_results.csv`
- `metrics/21_economic_sensitivity.csv`
- `metrics/21_economic_sensitivity_long.csv`

## Configuration

- No configuration file is read directly.

## Dependencies

- Third-party packages: `numpy`, `pandas`

## Usage

From the repository root:

```bash
python scripts/21_economic_analysis.py
```

The script has no command-line interface.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `dg/17_dg_energy_by_phase.csv`
- `dg/17_dg_uncertainty_interval.csv`
- `logs/21_economic_analysis.log`
- `manifests/13_create_windows_splits_params.json`
- `manifests/21_economic_analysis_params.json`
- `metrics/16_load_metrics_long.csv`
- `metrics/21_economic_results.csv`
- `metrics/21_economic_sensitivity.csv`
- `metrics/21_economic_sensitivity_long.csv`

## Reproducibility considerations

- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
