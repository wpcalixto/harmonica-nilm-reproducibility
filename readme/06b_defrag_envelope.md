# 06b_defrag_envelope.py

## Purpose

Defragmentation + envelope sanitisation (single self-contained script; runs between 06 and 07).

STEP 1: Defragmentation. For each fragmented series (meter x variable), reconstructs by
    the most correlated DONOR (a*donor+b) + overlap-add of real islands (Hanning window).
    No linear interpolation. Fills only where M1 has a reading (global-gap lock).
    Intermediate output: data/interim/06A_desfragmentado.csv.

STEP 2: Envelope (approved method). Per series, uses a clean REFERENCE SEGMENT as the
    amplitude pattern. The window is anchored by TIMESTAMP (--ref-start-time/--ref-end-time,
    default 2020-10-28 15:59 -> 2020-10-29 05:18 UTC), resolved by temporal search in each
    series; generic, independent of the grid origin.
    Envelope = [min_ref - k*std, max_ref + k*std]. Points outside:
       - 1 isolated point -> clip at the bound;
       - block (2+) -> replacement by a real fragment of the reference segment;
       - initial block (< --startup-scan-max-points) -> a single synthetic segment.
    Iterative passes + final envelope guard. Runs with --min-ref-obs-frac 0.45.
    Recomputes the gap files (gap_rule/long_gaps/dead_channels) for script 07.

Default input (output of script 06):
    data/interim/06_para_preencher.csv

Main outputs:
    data/interim/06A_desfragmentado.csv             (intermediate, after STEP 1)
    data/interim/06_para_preencher_sanitizado.csv   (final, input of script 07)
    audits/06A_defrag_map.csv
    audits/06_reference_amplitude_thresholds.csv
    audits/06_reference_amplitude_repair_map.csv
    audits/06_reference_amplitude_action_summary.csv
    audits/06_gap_rule.csv . 06_long_gaps.csv . 06_dead_channels.csv
    figdata/06_para_preencher_sanitizado.csv
    logs/06_sanitize_waveforms_reference_segment_repair.log
    manifests/06_sanitize_waveforms_reference_segment_repair_params.json

## Pipeline role

- File type: Data preprocessing script (pipeline stage 06b: defragmentation + envelope sanitisation)
- Position in the canonical sequence: 7 of 29
- Preceding stage: `06_validate_sentinels_outliers.py`
- Following stage: `07_characterize_gaps_dynamic_rule.py`

## Inputs

Declared in the module header:

- `data/interim/06_para_preencher.csv`

Files read by the source code (resolved from the path expressions in the script):

- `data/interim/06A_desfragmentado.csv`

## Outputs

Declared in the module header:

- `data/interim/06A_desfragmentado.csv             (intermediate, after STEP 1)`
- `data/interim/06_para_preencher_sanitizado.csv   (final, input of script 07)`
- `audits/06A_defrag_map.csv`
- `audits/06_reference_amplitude_thresholds.csv`
- `audits/06_reference_amplitude_repair_map.csv`
- `audits/06_reference_amplitude_action_summary.csv`
- `audits/06_gap_rule.csv`
- `06_long_gaps.csv`
- `06_dead_channels.csv`
- `figdata/06_para_preencher_sanitizado.csv`
- `logs/06_sanitize_waveforms_reference_segment_repair.log`
- `manifests/06_sanitize_waveforms_reference_segment_repair_params.json`

Files written by the source code (resolved from the path expressions in the script):

- `audits/{…}_dead_channels.csv`
- `audits/{…}_defrag_map.csv`
- `audits/{…}_gap_profile_by_channel.csv`
- `audits/{…}_gap_rule.csv`
- `audits/{…}_gap_size_distribution.csv`
- `audits/{…}_long_gaps.csv`
- `audits/{…}_reference_amplitude_action_summary.csv`
- `audits/{…}_reference_amplitude_repair_map.csv`
- `audits/{…}_reference_amplitude_thresholds.csv`
- `manifests/{…}_sanitize_waveforms_reference_segment_repair_params.json`

## Configuration

- No configuration file is read directly.

Command-line arguments (from `argparse` in the source):

| Argument | Default | Description |
|---|---|---|
| `--input` | str(DEFAULT_INPUT) | Input CSV with gaps as NaN. |
| `--output` | str(DEFAULT_OUTPUT) | Sanitised output CSV. |
| `--prefix` | '06' | Prefix of the audits. |
| `--ref-start-time` | '2020-10-28 15:59:00' | Start of the reference window as a UTC TIMESTAMP (where it is measured). |
| `--ref-end-time` | '2020-10-29 05:18:00' | End of the reference window as a UTC TIMESTAMP. |
| `--ref-start` | None | (Optional override) fixed start index; if omitted, uses --ref-start-time. |
| `--ref-end` | None | (Optional override) fixed end index; if omitted, uses --ref-end-time. |
| `--sd-k` | 1.0 | Multiplier of the standard deviation of the reference segment. |
| `--min-ref-obs-frac` | 0.45 | Minimum fraction of finite values in the reference segment of the series. |
| `--startup-scan-max-points` | 700 | Initial region for grouping the start-up block. |
| `--startup-min-bad-points` | 30 | Minimum number of points outside the envelope to form the initial block. |
| `--startup-min-span-points` | 60 | Minimum extent of the initial block. |
| `--merge-block-gap-points` | 10 | Closes small good intervals between violations. |
| `--clip-max-run-points` | 1 | Runs up to this length are clipped, not replaced. |
| `--repair-padding-points` | 3 | Margin added before/after replaced blocks. Does not affect clipped single points. Default: 3. |
| `--max-repair-passes` | 2 | Maximum number of repair passes per series. Default: 2. |
| `--final-envelope-guard` | off (flag) | Applies the final guard: any finite value still outside the envelope is clipped. |
| `--dead-pct` | 99.0 | Gap percentage for a dead channel. |
| `--short-gap-max-min` | 60 | Short-gap limit in the dynamic rule. |
| `--dry-run` | off (flag) | Generates the maps, but does not apply the repairs to the output CSV. |

## Dependencies

- Third-party packages: `numpy`, `pandas`

## Usage

From the repository root:

```bash
python scripts/06b_defrag_envelope.py [options]
```

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `audits/{…}_dead_channels.csv`
- `audits/{…}_defrag_map.csv`
- `audits/{…}_gap_profile_by_channel.csv`
- `audits/{…}_gap_rule.csv`
- `audits/{…}_gap_size_distribution.csv`
- `audits/{…}_long_gaps.csv`
- `audits/{…}_reference_amplitude_action_summary.csv`
- `audits/{…}_reference_amplitude_repair_map.csv`
- `audits/{…}_reference_amplitude_thresholds.csv`
- `data/interim/06A_desfragmentado.csv`
- `data/interim/06_para_preencher.csv`
- `data/interim/06_para_preencher_sanitizado.csv`
- `logs/06_sanitize_waveforms_reference_segment_repair.log`
- `manifests/{…}_sanitize_waveforms_reference_segment_repair_params.json`

## Reproducibility considerations

- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
