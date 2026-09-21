# collect_ablation.py

## Purpose

Consolidates the ablation (VALIDATION split): mean +/- SE of the global NAE per feature
set, one-standard-error rule, PAIRED comparison against reference_48, and a per-output GUARD
(degradation >5% in >=4/5 seeds). Does NOT decide n_adopted automatically: it delivers the
evidence for the decision. The test split is locked.

## Role

- File type: Auxiliary analysis script (consolidation of the feature-set ablation results)
- Auxiliary script: not part of the canonical sequential pipeline (01-28). It reads artifacts produced by the pipeline and writes its own outputs.

## Inputs

Files read by the source code (resolved from the path expressions in the script):

- `results/ablation/{s}/summary.json`
- `results/ablation/{s}/val_global.csv`
- `results/ablation/{s}/val_per_output.csv`

## Outputs

Files written by the source code (resolved from the path expressions in the script):

- `results/ablation/confirmation_comparison.csv`

## Configuration

- No configuration file is read directly.

## Dependencies

- Third-party packages: `numpy`, `pandas`

## Usage

From the repository root:

```bash
python scripts/auxiliares/collect_ablation.py
```

The script has no command-line interface.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `results/ablation/confirmation_comparison.csv`
- `results/ablation/{s}/summary.json`
- `results/ablation/{s}/val_global.csv`
- `results/ablation/{s}/val_per_output.csv`

## Reproducibility considerations

- Random seeds are fixed in the code/configuration (5 seeds: 42, 123, 456, 789, 1024 where applicable); do not change them.
- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
