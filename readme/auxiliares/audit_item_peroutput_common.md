# audit_item_peroutput_common.py

## Purpose

Per-output NAE (18) x [6 models + constant predictor] on the COMMON SUPPORT.
Same logic as item 4 (seed-averaged, common support of 965 windows), reported per output.
Read-only. Output: audits/item_peroutput_common.csv (18x8).

## Role

- File type: Auxiliary analysis script (read-only re-analysis of pipeline artifacts)
- Auxiliary script: not part of the canonical sequential pipeline (01-28). It reads artifacts produced by the pipeline and writes its own outputs.

## Inputs

Files read by the source code (resolved from the path expressions in the script):

- `audits/item_null_predictor_peroutput.csv`
- `data/processed/13_Y_raw_test.npy`
- `predictions/{…}.parquet`

## Outputs

Files written by the source code (resolved from the path expressions in the script):

- `audits/item_peroutput_common.csv`

## Configuration

- No configuration file is read directly.

## Dependencies

- Third-party packages: `numpy`, `pandas`

## Usage

From the repository root:

```bash
python scripts/auxiliares/audit_item_peroutput_common.py
```

The script has no command-line interface.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `audits/item_null_predictor_peroutput.csv`
- `audits/item_peroutput_common.csv`
- `data/processed/13_Y_raw_test.npy`
- `predictions/{…}.parquet`

## Reproducibility considerations

- Random seeds are fixed in the code/configuration (5 seeds: 42, 123, 456, 789, 1024 where applicable); do not change them.
- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
