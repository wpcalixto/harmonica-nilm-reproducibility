# 25_reproducibility_report.py

## Purpose

Reproducibility report: library versions, parameters, seeds, scripts, checksums,
scenarios and decisions, collected from the manifests and artifacts of the pipeline.

Anchors checked: 09_R1/R2, 11_*, 10_kneedle, 14_load_metrics, 17_dg_energy,
17_dg_uncertainty_interval, table_05, 20_article_values; scenarios from
09_scenario_mapping / 17_dg_energy / 17_dg_sensitivity; training stages
14_train_baselines / 15_train_advanced (including the PE-ES-Optuna sensitivity run).

Outputs: manifests/25_reproducibility_manifest.json, article/25_pipeline_report.md, log.

## Pipeline role

- File type: Reproducibility validation script (pipeline stage 25)
- Position in the canonical sequence: 26 of 29
- Preceding stage: `24_export_article_results.py`
- Following stage: `26_sanitization_diagnostics.py`

## Inputs

Files read by the source code (resolved from the path expressions in the script):

- none detected by static analysis (inputs may be passed through shared helpers or environment variables)

## Outputs

Declared in the module header:

- `manifests/25_reproducibility_manifest.json, article/25_pipeline_report.md, log.`

Files written by the source code (resolved from the path expressions in the script):

- `article/25_pipeline_report.md`
- `manifests/25_reproducibility_manifest.json`

## Configuration

- No configuration file is read directly.

## Dependencies

- Third-party packages: none (standard library only)

## Usage

From the repository root:

```bash
python scripts/25_reproducibility_report.py
```

The script has no command-line interface.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `article/2*_*.md`
- `article/2*_*.yaml`
- `article/24_article_values.yaml`
- `article/25_pipeline_report.md`
- `audits/09_scenario_mapping.csv`
- `audits/11_kneedle_selection.csv`
- `data/processed/09_R1.parquet`
- `data/processed/09_R2.parquet`
- `data/processed/13_X_test.npy`
- `data/processed/13_X_train.npy`
- `data/processed/13_Y_raw_test.npy`
- `dg/17_dg_*.csv`
- `dg/17_dg_energy_by_phase.csv`
- `dg/17_dg_sensitivity.csv`
- `dg/17_dg_uncertainty_interval.csv`
- `figures/figure_*.pdf`
- `logs/25_reproducibility_report.log`
- `manifests/25_reproducibility_manifest.json`
- `metrics/16_load_metrics_long.csv`
- `metrics/18_*.csv`
- `metrics/21_economic_*.csv`
- `tabdata/table_05_model_metrics.csv`
- `tables/table_*.tex`

## Reproducibility considerations

- Random seeds are fixed in the code/configuration (5 seeds: 42, 123, 456, 789, 1024 where applicable); do not change them.
- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
