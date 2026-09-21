# 26_sanitization_diagnostics.py

## Purpose

LATERAL DIAGNOSTIC (does not alter the pipeline 01-25).

Reads existing artifacts, reuses the actual filling functions (analog_fill/freq of script
08) and the validation helpers (script 07) and produces, with prefix 26_ and WITHOUT
overwriting anything:

  Task 1 (gap-duration histogram):
    audits/26_gap_blocks.csv
    tabdata/26_gap_duration_summary.csv
    figdata/26_gap_duration_histogram.csv
    figdata/26_gap_duration_thresholds.csv

  Task 3 (held-out with 50 individual windows + Wilcoxon/Holm + effect size):
    audits/26_heldout_window_level.csv
    tabdata/26_heldout_summary.csv
    tabdata/26_heldout_wilcoxon_holm.csv
    figdata/26_heldout_window_level.csv

The 50 windows per duration reproduce the protocol of script 07 (same seed=7, same
inputs), so that the MEDIANS coincide with 07_fill_validation; this script only preserves
the PER-WINDOW values (previously discarded) and extends the inference.

## Pipeline role

- File type: Diagnostic script (stage 26: sanitisation diagnostics; does not alter the pipeline 01-25)
- Position in the canonical sequence: 27 of 29
- Preceding stage: `25_reproducibility_report.py`
- Following stage: `27_generate_si_gap_duration_histogram.py`

## Inputs

Files read by the source code (resolved from the path expressions in the script):

- `audits/07_fill_validation.csv`
- `audits/07_gap_profile_by_channel.csv`
- `audits/07_gap_size_distribution.csv`
- `data/interim/06_para_preencher.csv`
- `data/processed/08_fill_confidence.parquet`

## Outputs

Files written by the source code (resolved from the path expressions in the script):

- `audits/26_confidence_reconciliation.csv`
- `audits/26_gap_blocks.csv`
- `audits/26_heldout_window_level.csv`
- `figdata/26_confidence_by_split.csv`
- `figdata/26_gap_duration_histogram.csv`
- `figdata/26_gap_duration_thresholds.csv`
- `figdata/26_heldout_window_level.csv`
- `tabdata/26_confidence_by_split.csv`
- `tabdata/26_gap_duration_summary.csv`
- `tabdata/26_heldout_summary.csv`
- `tabdata/26_heldout_wilcoxon_holm.csv`

## Configuration

- No configuration file is read directly.

Command-line arguments (from `argparse` in the source):

| Argument | Default | Description |
|---|---|---|
| `--project-root` | str(PROJECT_ROOT) |  |
| `--run-all` | off (flag) |  |
| `--run-gap-histogram` | off (flag) |  |
| `--run-heldout` | off (flag) |  |
| `--run-confidence` | off (flag) |  |

## Dependencies

- Third-party packages: `numpy`, `pandas`, `scipy`

## Usage

From the repository root:

```bash
python scripts/26_sanitization_diagnostics.py [options]
```

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `audits/07_fill_validation.csv`
- `audits/07_gap_profile_by_channel.csv`
- `audits/07_gap_size_distribution.csv`
- `audits/26_confidence_reconciliation.csv`
- `audits/26_gap_blocks.csv`
- `audits/26_heldout_window_level.csv`
- `data/interim/06_para_preencher.csv`
- `data/processed/08_fill_confidence.parquet`
- `figdata/26_confidence_by_split.csv`
- `figdata/26_gap_duration_histogram.csv`
- `figdata/26_gap_duration_thresholds.csv`
- `figdata/26_heldout_window_level.csv`
- `tabdata/26_confidence_by_split.csv`
- `tabdata/26_gap_duration_summary.csv`
- `tabdata/26_heldout_summary.csv`
- `tabdata/26_heldout_wilcoxon_holm.csv`

## Reproducibility considerations

- Random seeds are fixed in the code/configuration (5 seeds: 42, 123, 456, 789, 1024 where applicable); do not change them.
- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
