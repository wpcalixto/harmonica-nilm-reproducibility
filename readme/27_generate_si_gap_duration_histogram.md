# 27_generate_si_gap_duration_histogram.py

## Purpose

Supplementary figure (drawing only).

Reads exclusively figdata/26_gap_duration_histogram.csv and
figdata/26_gap_duration_thresholds.csv (produced by script 26) and generates
figures/figure_S_gap_duration_histogram.pdf. Does NOT recompute any result (project rule:
figure scripts never recompute data).

## Pipeline role

- File type: Figure-generation script (Supplementary Information; pipeline stage 27)
- Position in the canonical sequence: 28 of 29
- Preceding stage: `26_sanitization_diagnostics.py`
- Following stage: `28_leakage_control_wobs.py`

## Inputs

Files read by the source code (resolved from the path expressions in the script):

- `figdata/26_gap_duration_histogram.csv`
- `figdata/26_gap_duration_thresholds.csv`

## Outputs

Files written by the source code (resolved from the path expressions in the script):

- `figures/figure_S_gap_duration_histogram.pdf`

## Configuration

- No configuration file is read directly.

Command-line arguments (from `argparse` in the source):

| Argument | Default | Description |
|---|---|---|
| `--project-root` | str(PROJECT_ROOT) |  |

## Dependencies

- Third-party packages: `matplotlib`, `numpy`, `pandas`

## Usage

From the repository root:

```bash
python scripts/27_generate_si_gap_duration_histogram.py [options]
```

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `figdata/26_gap_duration_histogram.csv`
- `figdata/26_gap_duration_thresholds.csv`
- `figures/figure_S_gap_duration_histogram.pdf`

## Reproducibility considerations

- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
