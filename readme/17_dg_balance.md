# 17_dg_balance.py

## Purpose

Load-consistency balance (historically labelled D_G) + sensitivity + sensitivity budget.

Design: 18 three-phase load outputs, P_other = P01 = M10+M12+M13+M14.

Section A  BALANCE
    Reconstructed M_G = sum of observed loads(phase) + P_other(phase)   (closure, DG=0).
    For each model x scenario (R1/R2) x phase (A/B/C):
        R1: load side WITHOUT P_other -> D_G = P_L - M_G   (level ~ -P_other)
        R2: load side WITH P_other    -> D_G = (P_L+P_other) - M_G
    D_G_obs uses the observed loads; D_G_pred uses the predictions (mean of the 5 seeds).
    Energy by rectangular integration (dt = 1/60 h).
    ERG_bal = |E_DG_pred - E_DG_obs| / |E_MG| x 100 (energy closure error).

Section B  SENSITIVITY: scenarios S0-S7, at the energy level:
    S0 baseline . S1-S3 losses 1/2/5% (M_G x (1-l)) . S4-S7 P_other without M12/M13/M10/M14.
    S1-S3 = sensitivity to LOSSES (appear in R2: D_G += mg x loss).
    S4-S7 = STRUCTURAL sensitivity of P01 (appear in R1: D_G changes by E_component).
    Component removal via the energy fraction of 10_p_other_components (sum to 100%/phase;
    M12 is negative -> removing it INCREASES P_other).

Section C  SENSITIVITY BUDGET of R_cl, PER MODEL, directly in energy (three terms):
      u_met_L,m,phi = 0.5% x |E_L^obs| on the support of the model   (assumed perturbation of the loads)
      u_sync,m,phi  = sqrt((d_-30^2 + d_+30^2)/2)                    (actual shift of the observations, +/-30 s)
      u_seed,m,phi  = sd_s(E_R,m,s,phi)                              (directly in energy)
    Three-phase total: sums the phases WITHIN each seed before the sd, preserving the
    covariance between phases. Losses and P_other stay out (zero derivative in R_cl).
    Predictions FROZEN; +/-30 s by linear interpolation between minutes.
    The structural impact of P01 (S4-S7 vs S0) is reported SEPARATELY (scope decision,
    not random fluctuation; it does NOT enter the quadrature).

Data sources:
    13_Y_raw_test.npy (18 loads, Watts) + timestamps of the split; P_other from 09_R2
    (P_other_A/B/C), aligned by timestamp; predictions 14_/15_*_predictions.parquet
    (per seed); the 18 output labels from the manifest (DYNAMIC); W* from the manifest
    of script 13; imputed fraction of P01 from 10_p_other_energy_by_phase.

Outputs (dg/):
    17_dg_balance_timeseries.parquet . 17_dg_energy_by_phase.csv
    17_dg_sensitivity.csv . 17_dg_structural_impact.csv
    17_dg_uncertainty.csv . 17_dg_uncertainty_interval.csv
    manifests/17_dg_balance_params.json

## Pipeline role

- File type: Pipeline script (stage 17: load-consistency balance, sensitivity scenarios and sensitivity budget)
- Position in the canonical sequence: 18 of 29
- Preceding stage: `16_evaluate.py`
- Following stage: `18_statistical_analysis.py`

## Inputs

Declared in the module header:

- `13_Y_raw_test.npy (18 loads, Watts) + timestamps of the split; P_other from 09_R2`
- (P_other_A/B/C), aligned by timestamp; predictions 14_/15_*_predictions.parquet
- (per seed); the 18 output labels from the manifest (DYNAMIC); W* from the manifest
- `of script 13; imputed fraction of P01 from 10_p_other_energy_by_phase.`

Files read by the source code (resolved from the path expressions in the script):

- `audits/10_p_other_components.csv`
- `audits/10_p_other_energy_by_phase.csv`
- `data/processed/09_R2.parquet`
- `data/processed/13_Y_raw_test.npy`
- `figdata/13_split_timeline.csv`
- `manifests/13_create_windows_splits_params.json`
- `manifests/15_exp_pe_es_optuna_params.json`
- `manifests/15_train_advanced_params.json`
- `predictions/{filename}`

## Outputs

Declared in the module header:

- `17_dg_balance_timeseries.parquet`
- `17_dg_energy_by_phase.csv`
- `17_dg_sensitivity.csv`
- `17_dg_structural_impact.csv`
- `17_dg_uncertainty.csv`
- `17_dg_uncertainty_interval.csv`
- `manifests/17_dg_balance_params.json`

Files written by the source code (resolved from the path expressions in the script):

- `dg/17_dg_balance_timeseries.parquet`
- `dg/17_dg_balance_timeseries_primary_models.parquet`
- `dg/17_dg_balance_timeseries_sensitivity.parquet`
- `dg/17_dg_energy_by_phase.csv`
- `dg/17_dg_energy_by_phase_primary_models.csv`
- `dg/17_dg_energy_by_phase_sensitivity.csv`
- `dg/17_dg_sensitivity.csv`
- `dg/17_dg_sensitivity_hpo_sensitivity.csv`
- `dg/17_dg_sensitivity_primary_models.csv`
- `dg/17_dg_structural_impact.csv`
- `dg/17_dg_structural_impact_hpo_sensitivity.csv`
- `dg/17_dg_structural_impact_primary_models.csv`
- `dg/17_dg_uncertainty.csv`
- `dg/17_dg_uncertainty_interval.csv`
- `dg/17_dg_uncertainty_interval_hpo_sensitivity.csv`
- `dg/17_dg_uncertainty_interval_primary_models.csv`
- `manifests/17_dg_balance_params.json`

## Configuration

- No configuration file is read directly.
- Environment variables recognised: `HARMONIC_PROJECT_ROOT`

## Dependencies

- Third-party packages: `numpy`, `pandas`

## Usage

From the repository root:

```bash
python scripts/17_dg_balance.py
```

The script has no command-line options; behaviour is controlled by the environment variables listed above.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `audits/10_p_other_components.csv`
- `audits/10_p_other_energy_by_phase.csv`
- `data/processed/09_R2.parquet`
- `data/processed/13_Y_raw_test.npy`
- `dg/17_dg_balance_timeseries.parquet`
- `dg/17_dg_balance_timeseries_primary_models.parquet`
- `dg/17_dg_balance_timeseries_sensitivity.parquet`
- `dg/17_dg_energy_by_phase.csv`
- `dg/17_dg_energy_by_phase_primary_models.csv`
- `dg/17_dg_energy_by_phase_sensitivity.csv`
- `dg/17_dg_sensitivity.csv`
- `dg/17_dg_sensitivity_hpo_sensitivity.csv`
- `dg/17_dg_sensitivity_primary_models.csv`
- `dg/17_dg_structural_impact.csv`
- `dg/17_dg_structural_impact_hpo_sensitivity.csv`
- `dg/17_dg_structural_impact_primary_models.csv`
- `dg/17_dg_uncertainty.csv`
- `dg/17_dg_uncertainty_interval.csv`
- `dg/17_dg_uncertainty_interval_hpo_sensitivity.csv`
- `dg/17_dg_uncertainty_interval_primary_models.csv`
- `figdata/13_split_timeline.csv`
- `logs/17_dg_balance.log`
- `manifests/13_create_windows_splits_params.json`
- `manifests/15_exp_pe_es_optuna_params.json`
- `manifests/15_train_advanced_params.json`
- `manifests/17_dg_balance_params.json`
- `predictions/{filename}`

## Reproducibility considerations

- Random seeds are fixed in the code/configuration (5 seeds: 42, 123, 456, 789, 1024 where applicable); do not change them.
- The environment variables above alter the run (e.g. smoke tests); leave them unset to reproduce the reference run.
- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
