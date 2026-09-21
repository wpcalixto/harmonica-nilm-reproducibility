# 22_build_tables.py

## Purpose

Tables of the article.

Part 1: consolidates the results of the pipeline (01-21) into tabdata/.
Part 2: generates the .tex tables (booktabs) reading ONLY from tabdata/.

Methodology rule: final tables read only tabdata/ (never audits/). Here Part 1 reads
audits/metrics/dg/manifests and writes tabdata/; Part 2 reads only tabdata/ and writes
tables/*.tex. "Report when there is no data": every table whose input is missing is
logged and skipped (without aborting the script).

Data sources: 13_window_metadata; 11_kneedle_selection / 11_selected_features_by_output;
03_channel_eligibility; 09_scenario_mapping; 06_gap_profile_summary; 06_outlier_locations /
08_fill_traceability / 10_energy_preservation_by_stage; 16_load_metrics_long /
16_energy_by_load; 17_dg_energy_by_phase / 17_dg_sensitivity / 17_dg_structural_impact /
17_dg_uncertainty / 17_dg_uncertainty_interval; 18_statistical_tests / 18_pfi_results;
21_economic_results / 21_economic_sensitivity.

Outputs: tabdata/table_*.csv + table_manifest.csv ; tables/table_*.tex ; manifest/log.

## Pipeline role

- File type: Pipeline script (stage 22: consolidation of results into tabdata/ and generation of the LaTeX tables)
- Position in the canonical sequence: 23 of 29
- Preceding stage: `21_economic_analysis.py`
- Following stage: `23_generate_figures.py`

## Inputs

Declared in the module header:

- `13_window_metadata`
- `11_kneedle_selection / 11_selected_features_by_output`

Files read by the source code (resolved from the path expressions in the script):

- `tabdata/{n}`

## Outputs

Declared in the module header:

- `tabdata/table_*.csv + table_manifest.csv`
- `tables/table_*.tex`
- `manifest/log.`

Files written by the source code (resolved from the path expressions in the script):

- `manifests/22_build_tables_params.json`
- `tabdata/table_manifest.csv`
- `tabdata/{name}`
- `tables/tables_manifest.csv`

## Configuration

- No configuration file is read directly.

## Dependencies

- Third-party packages: `numpy`, `pandas`

## Usage

From the repository root:

```bash
python scripts/22_build_tables.py
```

The script has no command-line interface.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `audits/03_channel_eligibility.csv`
- `audits/06_gap_profile_summary.csv`
- `audits/06_outlier_locations.csv`
- `audits/08_fill_traceability.csv`
- `audits/10_energy_preservation_by_stage.csv`
- `audits/11_kneedle_selection.csv`
- `audits/11_selected_features_by_output.csv`
- `audits/13_window_metadata.csv`
- `dg/17_dg_energy_by_phase.csv`
- `dg/17_dg_sensitivity.csv`
- `dg/17_dg_structural_impact.csv`
- `dg/17_dg_uncertainty.csv`
- `dg/17_dg_uncertainty_interval.csv`
- `logs/22_build_tables.log`
- `manifests/22_build_tables_params.json`
- `metrics/16_energy_by_load.csv`
- `metrics/16_load_metrics_long.csv`
- `metrics/18_pfi_results.csv`
- `metrics/18_statistical_tests.csv`
- `metrics/21_economic_results.csv`
- `metrics/21_economic_sensitivity.csv`
- `tabdata/table_manifest.csv`
- `tabdata/{name}`
- `tabdata/{n}`
- `tables/tables_manifest.csv`

## Reproducibility considerations

- Random seeds are fixed in the code/configuration (5 seeds: 42, 123, 456, 789, 1024 where applicable); do not change them.
- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
