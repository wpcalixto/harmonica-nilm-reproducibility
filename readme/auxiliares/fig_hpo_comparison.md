# fig_hpo_comparison.py

## Purpose

Reproduces fig_resultado_12.pdf: test ERG_net in the default and tuned (HPO) configurations,
for PE-ES and LSTM.

Closes the provenance gap of the hyperparameter-sensitivity figure: reads
audits/hpo2_comparison.csv (produced by hpo_two_models.py) and draws the bars with error
bars (+/- 1 standard deviation across the 5 seeds). The comparison is DESCRIPTIVE: the default
and HPO configurations also differ in window length W (PE-ES 30->12, LSTM 12->36) and are
evaluated on distinct temporal supports, so the difference does not isolate the effect of
the optimisation.

Input:  audits/hpo2_comparison.csv (and audits/hpo2_final_per_seed.csv for the per-seed points)
Output: figures/fig_resultado_12.pdf

## Role

- File type: Figure-generation script (auxiliary; reads pipeline artifacts only)
- Auxiliary script: not part of the canonical sequential pipeline (01-28). It reads artifacts produced by the pipeline and writes its own outputs.

## Inputs

Declared in the module header:

- `audits/hpo2_comparison.csv (and audits/hpo2_final_per_seed.csv for the per-seed points)`

Files read by the source code (resolved from the path expressions in the script):

- `audits/hpo2_final_per_seed.csv`

## Outputs

Declared in the module header:

- `figures/fig_resultado_12.pdf`

Files written by the source code (resolved from the path expressions in the script):

- `figures/fig_resultado_12.pdf`

## Configuration

- No configuration file is read directly.

## Dependencies

- Third-party packages: `matplotlib`, `numpy`, `pandas`

## Usage

From the repository root:

```bash
python scripts/auxiliares/fig_hpo_comparison.py
```

The script has no command-line interface.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `audits/hpo2_final_per_seed.csv`
- `figures/fig_resultado_12.pdf`

## Reproducibility considerations

- Random seeds are fixed in the code/configuration (5 seeds: 42, 123, 456, 789, 1024 where applicable); do not change them.
- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
