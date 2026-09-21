# gen_review_figures.py

## Purpose

SINGLE GENERATOR of the review figures, following the figure protocol:
  - ONE PDF per panel (no internal subplots/subfigures);
  - reads ONLY from figdata/ (rev_*.csv), prepared by --prep;
  - single 18 pt font in every figure (never smaller); English; no title (goes in the LaTeX caption);
  - symbols identical to the manuscript; discrete colourblind-safe palette (Okabe-Ito) + cividis for maps.

Usage:
  python scripts/auxiliares/gen_review_figures.py --prep   # (re)populates figdata/rev_*.csv from the artifacts
  python scripts/auxiliares/gen_review_figures.py          # draws all PDFs into figures/

## Role

- File type: Figure-generation script (auxiliary; also prepares figdata/rev_*.csv from pipeline artifacts)
- Auxiliary script: not part of the canonical sequential pipeline (01-28). It reads artifacts produced by the pipeline and writes its own outputs.

## Inputs

Files read by the source code (resolved from the path expressions in the script):

- `audits/03_channel_classification.csv`
- `audits/03_phys_audit.csv`
- `audits/04_inverse_residuals_by_phase.csv`
- `audits/11_feature_ranking.csv`
- `audits/11_global_importance.csv`
- `audits/26_heldout_window_level.csv`
- `data/processed/13_Y_raw_test.npy`
- `data/raw/dados_maior_v2_transformed.csv`
- `dg/17_dg_energy_by_phase_primary_models.csv`
- `dg/17_dg_uncertainty.csv`
- `figdata/04_inverse_residuals_long.csv`
- `figdata/26_gap_duration_histogram.csv`
- `figdata/26_gap_duration_thresholds.csv`
- `figdata/rev_audit.csv`
- `figdata/rev_balanco.csv`
- `figdata/rev_cov_order.json`
- `figdata/rev_cov_presence.npy`
- `figdata/rev_heldout.csv`
- `figdata/rev_inverse_residuals.csv`
- `figdata/rev_inverse_status.csv`
- `figdata/rev_m2_cumenergy.csv`
- `figdata/rev_m2_xcorr.csv`
- `figdata/rev_perout.csv`
- `figdata/rev_sel_family.csv`
- `figdata/rev_sel_global.csv`
- `figdata/rev_sel_union.csv`
- `metrics/16_metrics_by_output.csv`
- `predictions/14_lstm_predictions.parquet`

## Outputs

Files written by the source code (resolved from the path expressions in the script):

- `figdata/rev_audit.csv`
- `figdata/rev_balanco.csv`
- `figdata/rev_cov_order.json`
- `figdata/rev_cov_presence.npy`
- `figdata/rev_heldout.csv`
- `figdata/rev_inverse_residuals.csv`
- `figdata/rev_inverse_status.csv`
- `figdata/rev_m2_cumenergy.csv`
- `figdata/rev_m2_xcorr.csv`
- `figdata/rev_perout.csv`
- `figdata/rev_predobs.csv`
- `figdata/rev_sel_family.csv`
- `figdata/rev_sel_global.csv`
- `figdata/rev_sel_union.csv`
- `figures/{name}`

## Configuration

- No configuration file is read directly.

Command-line arguments (from `argparse` in the source):

| Argument | Default | Description |
|---|---|---|
| `--prep` | off (flag) |  |

## Dependencies

- Third-party packages: `matplotlib`, `numpy`, `pandas`

## Usage

From the repository root:

```bash
python scripts/auxiliares/gen_review_figures.py [options]
```

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `audits/03_channel_classification.csv`
- `audits/03_phys_audit.csv`
- `audits/04_inverse_residuals_by_phase.csv`
- `audits/11_feature_ranking.csv`
- `audits/11_global_importance.csv`
- `audits/26_heldout_window_level.csv`
- `data/processed/13_Y_raw_test.npy`
- `data/raw/dados_maior_v2_transformed.csv`
- `dg/17_dg_energy_by_phase_primary_models.csv`
- `dg/17_dg_uncertainty.csv`
- `figdata/04_inverse_residuals_long.csv`
- `figdata/26_gap_duration_histogram.csv`
- `figdata/26_gap_duration_thresholds.csv`
- `figdata/rev_audit.csv`
- `figdata/rev_balanco.csv`
- `figdata/rev_cov_order.json`
- `figdata/rev_cov_presence.npy`
- `figdata/rev_heldout.csv`
- `figdata/rev_inverse_residuals.csv`
- `figdata/rev_inverse_status.csv`
- `figdata/rev_m2_cumenergy.csv`
- `figdata/rev_m2_xcorr.csv`
- `figdata/rev_perout.csv`
- `figdata/rev_predobs.csv`
- `figdata/rev_sel_family.csv`
- `figdata/rev_sel_global.csv`
- `figdata/rev_sel_union.csv`
- `figures/{name}`
- `metrics/16_metrics_by_output.csv`
- `predictions/14_lstm_predictions.parquet`

## Reproducibility considerations

- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
