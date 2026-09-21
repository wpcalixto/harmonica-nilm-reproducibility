# audit_item2_observed_nae.py

## Purpose

Layer 3 / item 2 (Tier A): active NAE restricted to ORIGINALLY OBSERVED targets, 6 models.

Re-analysis of existing artifacts (no retraining). Compares the current active NAE with
the NAE computed only on the windows whose CENTRAL target of each output was originally
observed (confidence class 0 in 08_fill_confidence). Read-only; writes a summary CSV to
audits/. Does not modify the pipeline.

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

- `audits/item2_observed_nae.csv`

## Configuration

- No configuration file is read directly.

## Dependencies

- Third-party packages: `numpy`, `pandas`

## Usage

From the repository root:

```bash
python scripts/auxiliares/audit_item2_observed_nae.py
```

The script has no command-line interface.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `audits/item2_observed_nae.csv`
- `data/processed/13_Y_raw_test.npy`
- `figdata/08_fill_confidence.parquet`
- `predictions/{…}.parquet`

## Reproducibility considerations

- Random seeds are fixed in the code/configuration (5 seeds: 42, 123, 456, 789, 1024 where applicable); do not change them.
- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
