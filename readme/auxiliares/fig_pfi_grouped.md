# fig_pfi_grouped.py

## Purpose

Reproduces fig_resultado_11.pdf: GROUPED permutation importance (I_g) per electrical
family, for the LSTM.

Closes the provenance gap of the figure: script 20 writes only the CSV
(metrics/20_pfi_grouped.csv) and no generator in the repository drew this figure. Here the
drawing is traceable to the data.

Axis label written in full ("Grouped permutation importance"), without the abbreviation
PFI, which is not defined in the body of the article.

Input:  figdata/20_pfi_grouped_long.csv (and metrics/20_pfi_grouped.csv)
Output: figures/fig_resultado_11.pdf

## Role

- File type: Figure-generation script (auxiliary; reads pipeline artifacts only)
- Auxiliary script: not part of the canonical sequential pipeline (01-28). It reads artifacts produced by the pipeline and writes its own outputs.

## Inputs

Declared in the module header:

- `figdata/20_pfi_grouped_long.csv (and metrics/20_pfi_grouped.csv)`

Files read by the source code (resolved from the path expressions in the script):

- `figdata/20_pfi_grouped_long.csv`

## Outputs

Declared in the module header:

- `figures/fig_resultado_11.pdf`

Files written by the source code (resolved from the path expressions in the script):

- `figures/fig_resultado_11.pdf`

## Configuration

- No configuration file is read directly.

## Dependencies

- Third-party packages: `matplotlib`, `numpy`, `pandas`

## Usage

From the repository root:

```bash
python scripts/auxiliares/fig_pfi_grouped.py
```

The script has no command-line interface.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `figdata/20_pfi_grouped_long.csv`
- `figures/fig_resultado_11.pdf`

## Reproducibility considerations

- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
