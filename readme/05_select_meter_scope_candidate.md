# 05_select_meter_scope_candidate.py

## Purpose

Generic pre-selection of the scope per meter.

Objective
---------
Read the four CSVs produced by the ingestion (02) and reduce the scope per meter,
preserving the positional alignment between the files.

This version does NOT fix roles such as Gamma, P_other, sensitivity or redundancy.
It only knows:

    METER_MG  = general meter, preserved by design;
    CANDIDATE = meters that are candidates to remain in the flow;

The final decision on the role of each meter is taken in later stages.

Default inputs in data/interim/ (outputs of 02)
-----------------------------------------------
02_primarias.csv
02_secundarias.csv
02_distorcao_meta.csv
02_harmonicas.csv

Outputs in data/interim/ (does NOT overwrite the 02_* files)
-----------------------------------------------------------
05_primarias.csv
05_secundarias.csv
05_distorcao_meta.csv
05_harmonicas.csv

The script writes the 05_*.csv files only with the kept meters and does NOT overwrite
the 02_* files. The decision table per meter goes to audits/05_meter_scope_preselection.csv.

P_other / P01 (pseudo-meter)
----------------------------
At the end, appends to the 05_*.csv a pseudo-meter P01 (meter_id=100) = aggregate of the
rejected real meters M10+M12+M13+M14 (additive quantities summed: i/p/q/s/hrm_i; the others
averaged). It is the closure term used later as P_other (scenario R2 of script 09).

Preliminary quality criterion
-----------------------------
For each candidate meter, the quality of the energy signal is computed using current as
anchor:

  - fraction of finite cells classified as preliminary sentinel;
  - fraction of rows with at least one preliminarily useful current;
  - temporal coverage relative to the meter with most rows.

By default, a candidate is dropped if:

  - pct_prelim_sentinel_energy >= 60%; or
  - pct_useful_energy < 40%; or
  - coverage_frac < 70%.

These limits can be changed by command-line arguments.

Common usage
------------
Main flow:
    python scripts/05_select_meter_scope_candidate.py

Audit only, without activating the scope:
    python scripts/05_select_meter_scope_candidate.py --dry-run

Generate the scope-filtered files without replacing the originals:
    python scripts/05_select_meter_scope_candidate.py --no-activate

Restore the full original files:
    python scripts/05_select_meter_scope_candidate.py --restore-full

## Pipeline role

- File type: Pipeline script (stage 05: generic pre-selection of the meter scope + P_other pseudo-meter)
- Position in the canonical sequence: 5 of 29
- Preceding stage: `04_inverse_physical_compatibility.py`
- Following stage: `06_validate_sentinels_outliers.py`

## Inputs

Declared in the module header:

- `02_primarias.csv`
- `02_secundarias.csv`
- `02_distorcao_meta.csv`
- `02_harmonicas.csv`

Files read by the source code (resolved from the path expressions in the script):

- `config/meter_map.yaml`

## Outputs

Declared in the module header:

- `05_primarias.csv`
- `05_secundarias.csv`
- `05_distorcao_meta.csv`
- `05_harmonicas.csv`

Files written by the source code (resolved from the path expressions in the script):

- `audits/05_duplicate_groups.csv`
- `audits/05_duplicate_similarity.csv`
- `audits/05_meter_scope_preselection.csv`
- `audits/05_preliminary_energy_quality_by_meter.csv`
- `audits/05_preliminary_sentinel_by_meter_variable.csv`
- `audits/05_row_filter_summary.csv`
- `audits/05_useful_energy_points_by_meter.csv`
- `manifests/05_select_meter_scope_candidate_only_params.json`

## Configuration

- Reads `config/meter_map.yaml`

Command-line arguments (from `argparse` in the source):

| Argument | Default | Description |
|---|---|---|
| `--main-meter` | METER_MG | General meter(s), e.g. --main-meter 1. Default: 1. |
| `--candidate-meters` | CANDIDATE | List of candidates, e.g. --candidate-meters 2,5,6,7,10,11,15,16. |
| `--apply-quality-to-main-meter` | off (flag) | Also applies the quality criteria to the general meter. By default, M_G is preserved. |
| `--min-useful-energy-frac` | 0.4 | Minimum fraction of rows with a useful energy signal for candidates. Default: 0.40. |
| `--max-sentinel-energy-frac` | 0.6 | Maximum allowed fraction of finite energy cells classified as preliminary sentinel. Default: 0.60. |
| `--min-coverage-frac` | 0.5 | Minimum temporal coverage relative to the meter with most rows. Default: 0.50 (admits the partial meters M14/M15 ~51%; the deduplication keeps M15 and removes M14). |
| `--sentinel-dominance-frac` | 0.8 | Horizontal value dominance for a preliminary sentinel. Default: 0.80. |
| `--min-unique-for-real` | 4 | Minimum number of rounded unique values to consider real variation. Default: 4. |
| `--placeholder-round-decimals` | 6 | Decimal places used to detect the horizontal dominant value. Default: 6. |
| `--robust-amp-eps` | 1e-09 | Minimum robust amplitude q99-q01 to not consider a column constant. Default: 1e-9. |
| `--min-finite-per-column` | 10 | Minimum number of finite points to evaluate an energy column. Default: 10. |
| `--force-keep` | set() | List of meters to keep, e.g. --force-keep 3,4,8 |
| `--force-drop` | set() | List of meters to drop, e.g. --force-drop 9,17,18 |
| `--dry-run` | off (flag) | Computes the selection and writes audits/manifest, but neither generates 01_scope_*.csv nor activates the scope. |
| `--no-activate` | off (flag) | Generates 01_scope_*.csv, but does not replace the 01_*.csv files. |
| `--restore-full` | off (flag) | Restores the complete 01_*.csv files from the 01_full_*.csv backups and exits. |
| `--no-dedup` | off (flag) | Disables the similarity-based deduplication among kept meters. |
| `--dup-corr-threshold` | 0.999 | Minimum correlation (mean of the 3 currents, common minutes) to consider a duplicate. Default: 0.999. |
| `--dup-magn-tol` | 0.05 | Maximum relative magnitude difference to consider a duplicate. Default: 0.05 (5%). |
| `--dup-min-overlap` | 200 | Minimum number of common minutes between two meters to assess OVERLAPPING duplication. Default: 200. |
| `--dup-interleave-max-overlap-frac` | 0.05 | Maximum overlap fraction to consider an INTERLEAVED duplicate (disjoint windows). Default: 0.05. |
| `--dup-interleave-min-union` | 0.8 | Minimum coverage union of the pair to consider an INTERLEAVED duplicate. Default: 0.80. |

## Dependencies

- Third-party packages: `numpy`, `pandas`, `yaml`

## Usage

From the repository root:

```bash
python scripts/05_select_meter_scope_candidate.py [options]
```

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `audits/05_duplicate_groups.csv`
- `audits/05_duplicate_similarity.csv`
- `audits/05_meter_scope_preselection.csv`
- `audits/05_preliminary_energy_quality_by_meter.csv`
- `audits/05_preliminary_sentinel_by_meter_variable.csv`
- `audits/05_row_filter_summary.csv`
- `audits/05_useful_energy_points_by_meter.csv`
- `config/meter_map.yaml`
- `logs/05_select_meter_scope_candidate_only.log`
- `manifests/05_select_meter_scope_candidate_only_params.json`

## Reproducibility considerations

- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
