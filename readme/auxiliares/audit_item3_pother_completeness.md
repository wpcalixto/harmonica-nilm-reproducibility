# audit_item3_pother_completeness.py

## Purpose

Layer 3 / item 3 (Tier A): completeness sensitivity of P_other.

EXACT mask of P01 (05_select_meter_scope: operational = i_an > 0.5 A) applied on the
1-minute GRID (mean per meter-minute cell, ingestion rule). C(t) = number of operational
components {10,12,13,14} at minute t. Validated against the manuscript (phase A: C>=1=2196,
C=1=1349, C=4=97; canonical artifact 06_alignment_coverage). Reports, per phase and
completeness level (exact strata C=1..4 and nested supports C>=1,>=2,>=3,=4): n, % of grid,
duration, E_MG and E_loads on the SAME support, energy of the partial aggregate, closure
residual (kWh, kWh/h, %|E_MG|). Read-only; uncertainty is descriptive (n and duration
reported).

## Role

- File type: Auxiliary analysis script (read-only re-analysis of pipeline artifacts)
- Auxiliary script: not part of the canonical sequential pipeline (01-28). It reads artifacts produced by the pipeline and writes its own outputs.

## Inputs

Files read by the source code (resolved from the path expressions in the script):

- `data/interim/02_raw_standardized.parquet`

## Outputs

Files written by the source code (resolved from the path expressions in the script):

- `audits/item3_pother_completeness.csv`

## Configuration

- No configuration file is read directly.

## Dependencies

- Third-party packages: `numpy`, `pandas`

## Usage

From the repository root:

```bash
python scripts/auxiliares/audit_item3_pother_completeness.py
```

The script has no command-line interface.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `audits/item3_pother_completeness.csv`
- `data/interim/02_raw_standardized.parquet`

## Reproducibility considerations

- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
