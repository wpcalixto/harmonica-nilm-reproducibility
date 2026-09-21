# 07_characterize_gaps_dynamic_rule.py

## Purpose

Gap characterisation and dynamic computation of the filling rule.

Objective
---------
Read the consolidated base prepared for filling, find contiguous NaN blocks per
(meter, variable), identify dead channels and compute dynamically the GAP_RULE from the
largest fillable gap observed in the non-dead channels.

The script does NOT fill gaps. It sizes the gaps and produces the inputs required by the
subsequent filling stage (script 08).

Default input (output of script 06)
-----------------------------------
    data/interim/06_para_preencher.csv

Default outputs, prefix configurable with --output-prefix
---------------------------------------------------------
    audits/07_gap_rule.csv
    audits/07_gap_size_distribution.csv
    audits/07_gap_profile_by_channel.csv
    audits/07_gap_profile_summary.csv
    audits/07_long_gaps.csv
    audits/07_dead_channels.csv
    logs/07_characterize_gaps.log
    manifests/07_characterize_gaps_params.json
    (with --run-heldout: audits/07_fill_validation, figdata/07_fill_validation_examples.csv)

Dynamic rule
------------
Category 1 keeps the operational interpretation of a short gap:
    1 .. short_gap_max_min  -> 1 wave portion

Categories 2, 3 and 4 are computed from the largest fillable gap, i.e. the largest gap
found in non-dead channels. By default:
    base = floor(max_fillable_gap_min / 3)
    cat 2: short_gap_max_min + 1 .. base
    cat 3: base + 1 .. 2*base
    cat 4: 2*base + 1 .. open

With max_fillable_gap_min = 5282 and short_gap_max_min = 60, the rule yields:
    1..60, 61..1760, 1761..3520, >=3521.

## Pipeline role

- File type: Pipeline script (stage 07: gap characterisation and dynamic filling rule)
- Position in the canonical sequence: 8 of 29
- Preceding stage: `06b_defrag_envelope.py`
- Following stage: `08_fill_gaps_no_clipping.py`

## Inputs

Declared in the module header:

- `data/interim/06_para_preencher.csv`

Files read by the source code (resolved from the path expressions in the script):

- none detected by static analysis (inputs may be passed through shared helpers or environment variables)

## Outputs

Declared in the module header:

- `audits/07_gap_rule.csv`
- `audits/07_gap_size_distribution.csv`
- `audits/07_gap_profile_by_channel.csv`
- `audits/07_gap_profile_summary.csv`
- `audits/07_long_gaps.csv`
- `audits/07_dead_channels.csv`
- `logs/07_characterize_gaps.log`
- `manifests/07_characterize_gaps_params.json`
- (with --run-heldout: audits/07_fill_validation, figdata/07_fill_validation_examples.csv)

Files written by the source code (resolved from the path expressions in the script):

- `audits/{…}.csv`
- `audits/{…}_gap_profile_by_channel.csv`
- `audits/{…}_gap_rule.csv`
- `audits/{…}_long_gaps.csv`
- `figdata/07_fill_validation_examples.csv`
- `manifests/{…}_characterize_gaps_params.json`
- `tables/{…}.tex`

## Configuration

- No configuration file is read directly.

Command-line arguments (from `argparse` in the source):

| Argument | Default | Description |
|---|---|---|
| `--input` | str(DEFAULT_INPUT_CSV) | Consolidated CSV with gaps as NaN. Default: data/interim/06_para_preencher.csv. |
| `--output-prefix` | OUTPUT_PREFIX | Prefix of the outputs in audits/tables/logs/manifests. Default: 07. |
| `--dead-pct` | DEAD_PCT | Gap percentage to classify a dead channel. Default: 99.0. |
| `--short-gap-max-min` | SHORT_GAP_MAX_MIN | Upper limit of the short category, in minutes. Default: 60. |
| `--run-heldout` | off (flag) | Runs the held-out proof of the filling. Not executed by default. |

## Dependencies

- Third-party packages: `numpy`, `pandas`, `scipy`, `sklearn`

## Usage

From the repository root:

```bash
python scripts/07_characterize_gaps_dynamic_rule.py [options]
```

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `audits/{…}.csv`
- `audits/{…}_gap_profile_by_channel.csv`
- `audits/{…}_gap_rule.csv`
- `audits/{…}_long_gaps.csv`
- `data/interim/06_para_preencher.csv`
- `figdata/07_fill_validation_examples.csv`
- `logs/{…}_characterize_gaps.log`
- `manifests/{…}_characterize_gaps_params.json`
- `tables/{…}.tex`

## Reproducibility considerations

- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
