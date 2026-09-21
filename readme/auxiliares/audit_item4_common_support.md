# audit_item4_common_support.py

## Purpose

Layer 3 / item 4 (Tier A): NAE on the TEMPORAL SUPPORT COMMON to the 6 models.

Intersection of the window centres: test instant in [max offset, min(offset+n-1)].
Recomputes the active NAE per model on that common support and checks whether the ranking
is preserved. Also repeats the OBSERVED-TARGET control on the common support. Read-only.

## Role

- File type: Auxiliary analysis script (read-only re-analysis of pipeline artifacts)
- Auxiliary script: not part of the canonical sequential pipeline (01-28). It reads artifacts produced by the pipeline and writes its own outputs.

## Inputs

Files read by the source code (resolved from the path expressions in the script):

- `data/processed/13_Y_raw_test.npy`
- `figdata/08_fill_confidence.parquet`
- `predictions/{…}.parquet`

## Outputs

Files written by the source code (resolved from the path expressions in the script):

- `audits/item4_common_support.csv`

## Configuration

- No configuration file is read directly.

## Dependencies

- Third-party packages: `numpy`, `pandas`

## Usage

From the repository root:

```bash
python scripts/auxiliares/audit_item4_common_support.py
```

The script has no command-line interface.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `audits/item4_common_support.csv`
- `data/processed/13_Y_raw_test.npy`
- `figdata/08_fill_confidence.parquet`
- `predictions/{…}.parquet`

## Reproducibility considerations

- Random seeds are fixed in the code/configuration (5 seeds: 42, 123, 456, 789, 1024 where applicable); do not change them.
- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
