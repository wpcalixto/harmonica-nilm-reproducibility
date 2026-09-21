# audit_item9_clustered_bootstrap.py

## Purpose

Layer 3 / item 9 (Tier A): inference CLUSTERED BY LOAD (bootstrap).

Delta_Gi = (1/3) sum_phi (NAE_competitor,i,phi - NAE_MoTE,i,phi);  Delta>0 favours MoTE.
- COMMON temporal support (965 windows) + ORIGINALLY OBSERVED target (joins controls 2 and 4).
- Paired differences per seed and output; mean of the 3 phases within each load.
- Resamples the 6 LOADS with replacement (phase triplet kept together) AND the 5 seeds (paired).
- 20,000 replicates, fixed seed; 95% percentile CI; shows the 6 per-load effects.
- Secondary control: exact sign test on the 6 per-load effects.

## Role

- File type: Auxiliary statistical analysis script (read-only re-analysis of pipeline artifacts)
- Auxiliary script: not part of the canonical sequential pipeline (01-28). It reads artifacts produced by the pipeline and writes its own outputs.

## Inputs

Files read by the source code (resolved from the path expressions in the script):

- `data/processed/13_Y_raw_test.npy`
- `figdata/08_fill_confidence.parquet`
- `predictions/{…}.parquet`

## Outputs

Files written by the source code (resolved from the path expressions in the script):

- `audits/item9_clustered_bootstrap.csv`

## Configuration

- No configuration file is read directly.

## Dependencies

- Third-party packages: `numpy`, `pandas`

## Usage

From the repository root:

```bash
python scripts/auxiliares/audit_item9_clustered_bootstrap.py
```

The script has no command-line interface.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `audits/item9_clustered_bootstrap.csv`
- `data/processed/13_Y_raw_test.npy`
- `figdata/08_fill_confidence.parquet`
- `predictions/{…}.parquet`

## Reproducibility considerations

- Random seeds are fixed in the code/configuration (5 seeds: 42, 123, 456, 789, 1024 where applicable); do not change them.
- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
