# 04_inverse_physical_compatibility.py

## Purpose

Inverse physical-compatibility problem (read-only physical control).

READ-ONLY physical control between the audit (03) and the scope selection (05). Does NOT
recalibrate the data automatically. For each meter x phase, evaluates whether the
electrical relations can be reconciled by a MINIMAL ADJUSTMENT close to the identity:

    V~ = a_V*V + b_V ,  I~ = a_I*I + b_I ,  S~ = a_S*S ,  Q~ = a_Q*Q ,  (a_P=1, b_P=0)

    J(theta) = w_vi*sum rho(S~ - V~*I~) + w_pcos*sum rho(P - S~*cos(phi))
             + w_tri*sum rho(S~^2 - P^2 - Q~^2) + lambda*||theta - theta0||^2 ,
    theta0 = (1,0,1,0,1,1)

rho = robust loss (Huber). The active power P is the fixed energy quantity (not adjusted).
The result is DIAGNOSTIC: it measures compatibility and triggers alert/sensitivity
flags, without altering the data.

Status: COMPATIBLE (low residuals and theta close to the identity) . ALERT (moderate
residual or theta far from the identity) . INCOMPATIBLE_PERSISTENT (high residual after
the minimal adjustment).

Input:
    data/interim/02_raw_standardized.parquet
Parameters: block `inverse_compatibility` of preprocessing_config.yaml (built-in fallback).

Outputs (prefix 04_):
    audits/04_inverse_residuals_by_phase.csv     (residuals before/after + status)
    audits/04_inverse_parameter_shifts.csv       (theta* per meter x phase)
    audits/04_inverse_identifiability.csv        (n_real, convergence, identifiability)
    figdata/04_inverse_residuals_long.csv        (long: term x stage)
    logs/04_inverse_physical_compatibility.log
    manifests/04_inverse_physical_compatibility_params.json

## Pipeline role

- File type: Diagnostic script (pipeline stage 04: inverse physical-compatibility control; read-only)
- Position in the canonical sequence: 4 of 29
- Preceding stage: `03_audit_physical_informational.py`
- Following stage: `05_select_meter_scope_candidate.py`

## Inputs

Declared in the module header:

- `data/interim/02_raw_standardized.parquet`

Files read by the source code (resolved from the path expressions in the script):

- `config/preprocessing_config.yaml`
- `data/interim/02_raw_standardized.parquet`

## Outputs

Declared in the module header:

- `audits/04_inverse_residuals_by_phase.csv     (residuals before/after + status)`
- `audits/04_inverse_parameter_shifts.csv       (theta* per meter x phase)`
- `audits/04_inverse_identifiability.csv        (n_real, convergence, identifiability)`
- `figdata/04_inverse_residuals_long.csv        (long: term x stage)`
- `logs/04_inverse_physical_compatibility.log`
- `manifests/04_inverse_physical_compatibility_params.json`

Files written by the source code (resolved from the path expressions in the script):

- `audits/04_inverse_identifiability.csv`
- `audits/04_inverse_parameter_shifts.csv`
- `audits/04_inverse_residuals_by_phase.csv`
- `figdata/04_inverse_residuals_long.csv`
- `manifests/04_inverse_physical_compatibility_params.json`

## Configuration

- Reads `config/preprocessing_config.yaml`

## Dependencies

- Third-party packages: `numpy`, `pandas`, `scipy`, `yaml`

## Usage

From the repository root:

```bash
python scripts/04_inverse_physical_compatibility.py
```

The script has no command-line interface.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `audits/04_inverse_identifiability.csv`
- `audits/04_inverse_parameter_shifts.csv`
- `audits/04_inverse_residuals_by_phase.csv`
- `config/preprocessing_config.yaml`
- `data/interim/02_raw_standardized.parquet`
- `figdata/04_inverse_residuals_long.csv`
- `logs/04_inverse_physical_compatibility.log`
- `manifests/04_inverse_physical_compatibility_params.json`

## Reproducibility considerations

- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
