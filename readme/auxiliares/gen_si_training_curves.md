# gen_si_training_curves.py

## Purpose

Training/validation loss curves: 5 SEPARATE PANELS (one per trainable architecture) +
shared legend, in ENGLISH, following the protocol of the predobs figures.
Mean over the 5 seeds per epoch. Source: figdata/23_training_history.csv.
Output directory: figures/

## Role

- File type: Figure-generation script (Supplementary Information; reads pipeline artifacts only)
- Auxiliary script: not part of the canonical sequential pipeline (01-28). It reads artifacts produced by the pipeline and writes its own outputs.

## Inputs

Files read by the source code (resolved from the path expressions in the script):

- `figdata/23_training_history.csv`

## Outputs

Declared in the module header:

- `figures/`

Files written by the source code (resolved from the path expressions in the script):

- none detected by static analysis (outputs may be written through shared helpers)

## Configuration

- No configuration file is read directly.

## Dependencies

- Third-party packages: `matplotlib`, `numpy`, `pandas`

## Usage

From the repository root:

```bash
python scripts/auxiliares/gen_si_training_curves.py
```

The script has no command-line interface.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `figdata/23_training_history.csv`

## Reproducibility considerations

- Random seeds are fixed in the code/configuration (5 seeds: 42, 123, 456, 789, 1024 where applicable); do not change them.
- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
