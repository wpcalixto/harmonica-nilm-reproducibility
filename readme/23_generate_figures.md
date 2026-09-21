# 23_generate_figures.py

## Purpose

RESULT figures of the article.

Rules (figure protocol):
  - Two phases: (A) prepare_figdata (reads dg/metrics/predictions -> writes figdata/23_*),
    (B) plot (reads ONLY figdata/ -> figures/*.pdf). No figure recomputes results.
  - Image protocol: usetex, serif/cm, EVERYTHING at 18 pt, constrained_layout, PDF 300 dpi,
    no title in the figure (title via the LaTeX caption), 1 file per figure, spines off.
  - Consistent palette. 6 MAIN models in the main figures; PE-ES-Optuna* enters as a
    SENSITIVITY SERIES (dashed style / distinct marker), not as a main model.
  - "Report when there is no data": a figure without input is logged and skipped.

Methodological conclusion kept: MoTE_v2 = best main model (point-wise NAE);
PE-ES-Optuna* = best energy sensitivity (ERG), with an additional HPO budget.

Result figures generated:
  figure_10_pfi.pdf . figure_11_pred_obs_*.pdf . figure_16_nae_by_model.pdf .
  figure_13_p_other.pdf . figure_17_dg_balance.pdf . figure_17_dg_sensitivity.pdf .
  figure_16_uncertainty.pdf . figure_17A_cohen_d.pdf . figure_17B_nae_erg.pdf .
  figure_18A_c_error.pdf . figure_18B_c_yearly.pdf

## Pipeline role

- File type: Figure-generation script (pipeline stage 23: result figures of the article)
- Position in the canonical sequence: 24 of 29
- Preceding stage: `22_build_tables.py`
- Following stage: `24_export_article_results.py`

## Inputs

Files read by the source code (resolved from the path expressions in the script):

- `audits/10_p_other_energy_by_phase.csv`
- `data/interim/02_raw_standardized.parquet`
- `data/processed/13_Y_raw_test.npy`
- `figdata/06_para_preencher.csv`
- `figdata/08_fill_confidence.parquet`
- `figdata/08_preenchido.parquet`
- `figdata/{name}`
- `figdata/{…}_training_history_long.csv`
- `manifests/13_create_windows_splits_params.json`
- `metrics/16_load_metrics_long.csv`
- `predictions/15_mote_v2_predictions.parquet`

## Outputs

Files written by the source code (resolved from the path expressions in the script):

- `figdata/23_anomaly_regions.csv`
- `figdata/23_anomaly_series.csv`
- `figdata/23_cov_roles.csv`
- `figdata/23_nae.csv`
- `figdata/23_p_other.csv`
- `figdata/23_pred_obs.csv`
- `figdata/23_training_history.csv`
- `figdata/{dst}`
- `figdata/{out_csv}`
- `figures/{name}`
- `manifests/23_generate_figures_params.json`

## Configuration

- No configuration file is read directly.

## Dependencies

- Third-party packages: `matplotlib`, `numpy`, `pandas`

## Usage

From the repository root:

```bash
python scripts/23_generate_figures.py
```

The script has no command-line interface.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `audits/04_inverse_residuals_by_phase.csv`
- `audits/07_fill_validation.csv`
- `audits/10_p_other_components.csv`
- `audits/10_p_other_energy_by_phase.csv`
- `audits/11_feature_ranking.csv`
- `audits/11_kneedle_selection.csv`
- `audits/11_rfecv_scores.csv`
- `audits/11_selected_features_by_output.csv`
- `data/interim/02_raw_standardized.parquet`
- `data/processed/13_Y_raw_test.npy`
- `dg/17_dg_energy_by_phase.csv`
- `dg/17_dg_sensitivity.csv`
- `dg/17_dg_uncertainty.csv`
- `dg/17_dg_uncertainty_interval.csv`
- `figdata/02_timestamp_coverage.csv`
- `figdata/06_para_preencher.csv`
- `figdata/06_para_preencher_sanitizado.csv`
- `figdata/07_fill_validation_examples.csv`
- `figdata/08_fill_confidence.parquet`
- `figdata/08_preenchido.parquet`
- `figdata/10_raw_vs_processed_energy_long.csv`
- `figdata/11_feature_ranking_long.csv`
- `figdata/13_split_timeline.csv`
- `figdata/23_anomaly_regions.csv`
- `figdata/23_anomaly_series.csv`
- `figdata/23_cov_roles.csv`
- `figdata/23_nae.csv`
- `figdata/23_p_other.csv`
- `figdata/23_pred_obs.csv`
- `figdata/23_training_history.csv`
- `figdata/{dst}`
- `figdata/{name}`
- `figdata/{out_csv}`
- `figdata/{…}_training_history_long.csv`
- `figures/{name}`
- `logs/23_generate_figures.log`
- `manifests/13_create_windows_splits_params.json`
- `manifests/23_generate_figures_params.json`
- `metrics/16_energy_by_load.csv`
- `metrics/16_load_metrics_long.csv`
- `metrics/16_pfi_top_features_long.csv`
- `metrics/18_model_comparison_long.csv`
- `metrics/18_statistical_tests.csv`
- `metrics/21_economic_sensitivity_long.csv`
- `predictions/15_mote_v2_predictions.parquet`

## Reproducibility considerations

- Random seeds are fixed in the code/configuration (5 seeds: 42, 123, 456, 789, 1024 where applicable); do not change them.
- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
