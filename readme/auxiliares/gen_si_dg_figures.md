# gen_si_dg_figures.py

## Purpose

Supplementary-Information figures for the balance, sensitivity and sensitivity-budget block
of the load-consistency residual R_cl.

Same protocol as gen_review_figures.py: ONE PDF per panel, single 18 pt font, English, no
title (goes in the LaTeX caption), Okabe-Ito palette, symbols identical to the manuscript.
Reads ONLY the artifacts of 17_dg_balance.py (dg/), never fixed values.

Outputs (figures/):
    fig_si_dg_loss.pdf         sensitivity to the loss hypothesis (S0-S3)
    fig_si_dg_structural.pdf   structural impact of the P_other composition (S4-S7)
    fig_si_dg_ubudget.pdf      components of the sensitivity budget per phase
    fig_si_dg_intervals.pdf    expanded intervals (k_u=2) per model x phase

## Role

- File type: Figure-generation script (Supplementary Information; reads pipeline artifacts only)
- Auxiliary script: not part of the canonical sequential pipeline (01-28). It reads artifacts produced by the pipeline and writes its own outputs.

## Inputs

Files read by the source code (resolved from the path expressions in the script):

- `dg/17_dg_sensitivity.csv`
- `dg/17_dg_structural_impact.csv`
- `dg/17_dg_uncertainty.csv`
- `dg/17_dg_uncertainty_interval.csv`

## Outputs

Declared in the module header:

- `fig_si_dg_loss.pdf         sensitivity to the loss hypothesis (S0-S3)`
- `fig_si_dg_structural.pdf   structural impact of the P_other composition (S4-S7)`
- `fig_si_dg_ubudget.pdf      components of the sensitivity budget per phase`
- `fig_si_dg_intervals.pdf    expanded intervals (k_u=2) per model x phase`

Files written by the source code (resolved from the path expressions in the script):

- `figures/{name}`

## Configuration

- No configuration file is read directly.

## Dependencies

- Third-party packages: `matplotlib`, `numpy`, `pandas`

## Usage

From the repository root:

```bash
python scripts/auxiliares/gen_si_dg_figures.py
```

The script has no command-line interface.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `dg/17_dg_sensitivity.csv`
- `dg/17_dg_structural_impact.csv`
- `dg/17_dg_uncertainty.csv`
- `dg/17_dg_uncertainty_interval.csv`
- `figures/{name}`

## Reproducibility considerations

- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
