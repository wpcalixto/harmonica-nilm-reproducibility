# fig_predobs_seed.py

## Purpose

Predicted-vs-observed panels of the LSTM showing the INDIVIDUAL prediction of each seed
(not the mean over seeds).

Generates:
  - seed 42  -> fig_resultado_predobs_{1..8}.pdf   (ARTICLE, main text)
  - seeds 123/456/789/1024 -> fig_si_predobs_s{seed}_{1..8}.pdf  (Supplementary Information)
  - GENERIC legend (reusable in all panels): fig_resultado_predobs_legend.pdf
    and fig_si_predobs_legend.pdf

In each panel: orange = reference target (observed + reconstructed); blue = individual
prediction of the seed; grey band = min-max envelope of the 5 seeds.

Inputs:
    figdata/rev_predobs.csv                    (obs_{o} = reference target)
    predictions/14_lstm_predictions.parquet    (prediction per seed x window)
Output directory: figures/

## Role

- File type: Figure-generation script (auxiliary; reads pipeline artifacts only)
- Auxiliary script: not part of the canonical sequential pipeline (01-28). It reads artifacts produced by the pipeline and writes its own outputs.

## Inputs

Declared in the module header:

- `figdata/rev_predobs.csv                    (obs_{o} = reference target)`
- `predictions/14_lstm_predictions.parquet    (prediction per seed x window)`

Files read by the source code (resolved from the path expressions in the script):

- `predictions/14_lstm_predictions.parquet`

## Outputs

Declared in the module header:

- `figures/`

Files written by the source code (resolved from the path expressions in the script):

- `figures/{name}`

## Configuration

- No configuration file is read directly.

## Dependencies

- Third-party packages: `matplotlib`, `numpy`, `pandas`

## Usage

From the repository root:

```bash
python scripts/auxiliares/fig_predobs_seed.py
```

The script has no command-line interface.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `figdata/rev_predobs.csv`
- `figures/{name}`
- `predictions/14_lstm_predictions.parquet`

## Reproducibility considerations

- Random seeds are fixed in the code/configuration (5 seeds: 42, 123, 456, 789, 1024 where applicable); do not change them.
- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
