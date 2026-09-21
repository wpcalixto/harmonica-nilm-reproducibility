# 28_leakage_control_wobs.py

## Purpose

LATERAL DIAGNOSTIC (does not alter the pipeline).

Strict control of the influence of reconstructed inputs: recomputes NAE/ERG only on the test
windows in which ALL 48 M_G input features are originally observed over the W instants AND
the 18 targets are observed at the evaluated instant (W_obs). The 12 constant-zero auxiliary
columns are excluded from the check (they are never filled).

Sources (read-only): 08_fill_confidence mask, predictions/reference targets of the current
run, list of the 48 features (windows/reference_48/manifest.json). Outputs with prefix 28_.

## Pipeline role

- File type: Diagnostic script (leakage control; does not alter the pipeline; stage 28)
- Position in the canonical sequence: 29 of 29
- Preceding stage: `27_generate_si_gap_duration_histogram.py`
- Following stage: `none (last stage)`

## Inputs

Files read by the source code (resolved from the path expressions in the script):

- `data/processed/08_fill_confidence.parquet`
- `data/processed/13_Y_raw_test.npy`
- `data/processed/windows/reference_48/manifest.json`

## Outputs

Files written by the source code (resolved from the path expressions in the script):

- none detected by static analysis (outputs may be written through shared helpers)

## Configuration

- No configuration file is read directly.

Command-line arguments (from `argparse` in the source):

| Argument | Default | Description |
|---|---|---|
| `--root` | str(ROOT) |  |
| `--full` | str(FULL) |  |

## Dependencies

- Third-party packages: `numpy`, `pandas`

## Usage

From the repository root:

```bash
python scripts/28_leakage_control_wobs.py [options]
```

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `audits/28_wobs_control.csv`
- `data/processed/08_fill_confidence.parquet`
- `data/processed/13_Y_raw_test.npy`
- `data/processed/windows/reference_48/manifest.json`
- `predictions/{pf}_predictions.parquet`
- `tabdata/28_wobs_control.csv`

## Reproducibility considerations

- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
