# 24_export_article_results.py

## Purpose

Consolidation for the manuscript: exports values/decisions ready for the article.
EVERYTHING is derived from the files of the pipeline (no hard-coded numbers). The
PE-ES-Optuna model appears as SENSITIVITY (analysis_group), never in the main ranking.

Data sources: 09_scenario_mapping (Gamma map per scenario), 11_kneedle_selection +
11_candidate_sets (feature selection), 10_p_other_energy_by_phase (P_other), 17_dg_*
(balance and uncertainty), table_05 (metrics); table_01 uses "parameter/value".

Outputs: article/24_article_values.yaml . 24_article_results_summary.md .
         24_methods_values_checklist.md . manifests/24_*.json

## Pipeline role

- File type: Pipeline script (stage 24: consolidation of values for the manuscript)
- Position in the canonical sequence: 25 of 29
- Preceding stage: `23_generate_figures.py`
- Following stage: `25_reproducibility_report.py`

## Inputs

Declared in the module header:

- `09_scenario_mapping (Gamma map per scenario), 11_kneedle_selection +`

Files read by the source code (resolved from the path expressions in the script):

- `data/processed/11_candidate_sets.json`

## Outputs

Declared in the module header:

- `article/24_article_values.yaml`
- `24_article_results_summary.md .`
- `24_methods_values_checklist.md`
- `manifests/24_*.json`

Files written by the source code (resolved from the path expressions in the script):

- `article/24_article_results_summary.md`
- `article/24_article_values.yaml`
- `article/24_methods_values_checklist.md`
- `manifests/24_export_article_results_params.json`

## Configuration

- No configuration file is read directly.

## Dependencies

- Third-party packages: `numpy`, `pandas`, `yaml`

## Usage

From the repository root:

```bash
python scripts/24_export_article_results.py
```

The script has no command-line interface.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `article/24_article_results_summary.md`
- `article/24_article_values.yaml`
- `article/24_methods_values_checklist.md`
- `audits/09_scenario_mapping.csv`
- `audits/10_p_other_energy_by_phase.csv`
- `audits/11_global_importance.csv`
- `audits/11_kneedle_selection.csv`
- `data/processed/11_candidate_sets.json`
- `dg/17_dg_energy_by_phase.csv`
- `dg/17_dg_uncertainty_interval.csv`
- `logs/24_export_article_results.log`
- `manifests/24_export_article_results_params.json`
- `tabdata/table_01_dataset_summary.csv`
- `tabdata/table_05_model_metrics.csv`

## Reproducibility considerations

- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
