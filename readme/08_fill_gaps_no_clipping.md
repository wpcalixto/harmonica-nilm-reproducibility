# 08_fill_gaps_no_clipping.py

## Purpose

Analog/fractal gap filling by wave portions.

Fills the gaps AFTER the sanitisation of stage 06/06b. The main input is the sanitised base
with NaN at the points to fill. The category rule / number of wave portions is NOT fixed
in this script: it is read from the file produced upstream (`audits/06_gap_rule.csv`).

WITHOUT CLIPPING (this version): the amplitude guard NO LONGER uses the hard cut
`np.clip(filled, obs.min, obs.max)`, which flattened ~214k filled points at the observed
extremes. Instead, the texture of the gap is smoothly rescaled around the base line /
anchor; only a negligible residual clip remains. Result: filling with natural amplitude
(not capped), with no points left unfilled.

Default inputs:
    data/interim/06_para_preencher_sanitizado.csv
    audits/06_gap_rule.csv
    audits/06_dead_channels.csv
    audits/06_long_gaps.csv        (optional, used for traceability)

Default outputs:
    data/processed/08_preenchido.parquet
    data/processed/08_fill_confidence.parquet     (per-point confidence mask)
    data/processed/08_preenchido_referencia.parquet (scientific reference matrix)
    audits/08_fill_traceability.csv
    audits/08_gap_rule_used.csv
    figdata/08_preenchido.parquet
    logs/08_fill_gaps.log
    manifests/08_fill_gaps_params.json

## Pipeline role

- File type: Data preprocessing script (pipeline stage 08: analog/fractal gap filling by wave portions)
- Position in the canonical sequence: 9 of 29
- Preceding stage: `07_characterize_gaps_dynamic_rule.py`
- Following stage: `09_build_model_datasets.py`

## Inputs

Declared in the module header:

- `data/interim/06_para_preencher_sanitizado.csv`
- `audits/06_gap_rule.csv`
- `audits/06_dead_channels.csv`
- `audits/06_long_gaps.csv        (optional, used for traceability)`

Files read by the source code (resolved from the path expressions in the script):

- none detected by static analysis (inputs may be passed through shared helpers or environment variables)

## Outputs

Declared in the module header:

- `data/processed/08_preenchido.parquet`
- `data/processed/08_fill_confidence.parquet     (per-point confidence mask)`
- `data/processed/08_preenchido_referencia.parquet (scientific reference matrix)`
- `audits/08_fill_traceability.csv`
- `audits/08_gap_rule_used.csv`
- `figdata/08_preenchido.parquet`
- `logs/08_fill_gaps.log`
- `manifests/08_fill_gaps_params.json`

Files written by the source code (resolved from the path expressions in the script):

- `audits/08_fill_traceability.csv`
- `manifests/08_fill_gaps_params.json`

## Configuration

- No configuration file is read directly.

Command-line arguments (from `argparse` in the source):

| Argument | Default | Description |
|---|---|---|
| `--input` | DEFAULT_INPUT_CSV | Sanitised CSV to fill. Default: data/interim/06_para_preencher_sanitizado.csv |
| `--output` | DEFAULT_OUTPUT_PARQUET | Filled parquet. Default: data/processed/08_preenchido.parquet |
| `--gap-rule` | DEFAULT_GAP_RULE_CSV | Dynamic gap rule produced by stage 06. Default: audits/06_gap_rule.csv |
| `--dead-channels` | DEFAULT_DEAD_CHANNELS_CSV | Dead channels produced by stage 06. Default: audits/06_dead_channels.csv |
| `--long-gaps` | DEFAULT_LONG_GAPS_CSV | Long gaps produced by stage 06, for traceability. Default: audits/06_long_gaps.csv |
| `--seed` | SEED | Seed of the random generator. |

## Dependencies

- Third-party packages: `numpy`, `pandas`

## Usage

From the repository root:

```bash
python scripts/08_fill_gaps_no_clipping.py [options]
```

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `audits/06_dead_channels.csv`
- `audits/06_gap_rule.csv`
- `audits/06_long_gaps.csv`
- `audits/08_fill_traceability.csv`
- `audits/08_gap_rule_used.csv`
- `data/interim/06_para_preencher_sanitizado.csv`
- `data/processed/08_preenchido.parquet`
- `figdata/08_fill_confidence.parquet`
- `figdata/08_preenchido.parquet`
- `figdata/08_supported_reference.parquet`
- `logs/08_fill_gaps.log`
- `manifests/08_fill_gaps_params.json`

## Reproducibility considerations

- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
