# audit_item_null_predictor.py

## Purpose

Layer 3 / NULL predictor (baseline without M_G): control of the first hypothesis.

y_null_j(t) = median{ y_j(t) : t in TRAIN, originally observed target },
constant per output, estimated ONLY on the training split, ONLY on observed targets,
per output, WITHOUT any M_G feature.

Evaluation on the TEMPORAL SUPPORT COMMON to the 6 models and on OBSERVED TARGETS,
same active NAE as item 4. Descriptive comparison with the 6 models. Read-only.

## Role

- File type: Auxiliary analysis script (null-predictor control; read-only re-analysis of pipeline artifacts)
- Auxiliary script: not part of the canonical sequential pipeline (01-28). It reads artifacts produced by the pipeline and writes its own outputs.

## Inputs

Files read by the source code (resolved from the path expressions in the script):

- `audits/item4_common_support.csv`
- `data/processed/13_Y_raw_test.npy`
- `data/processed/13_Y_train.npy`
- `data/processed/13_scaler_Y.pkl`
- `figdata/08_fill_confidence.parquet`
- `predictions/{…}.parquet`

## Outputs

Files written by the source code (resolved from the path expressions in the script):

- `audits/item_null_predictor_compare.csv`
- `audits/item_null_predictor_peroutput.csv`

## Configuration

- No configuration file is read directly.

## Dependencies

- Third-party packages: `joblib`, `numpy`, `pandas`

## Usage

From the repository root:

```bash
python scripts/auxiliares/audit_item_null_predictor.py
```

The script has no command-line interface.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `audits/item4_common_support.csv`
- `audits/item_null_predictor_compare.csv`
- `audits/item_null_predictor_peroutput.csv`
- `data/processed/13_Y_raw_test.npy`
- `data/processed/13_Y_train.npy`
- `data/processed/13_scaler_Y.pkl`
- `figdata/08_fill_confidence.parquet`
- `predictions/{…}.parquet`

## Reproducibility considerations

- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
