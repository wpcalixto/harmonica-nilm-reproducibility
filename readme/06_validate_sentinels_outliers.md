# 06_validate_sentinels_outliers.py

## Purpose

Validation, alignment, real gaps, sentinels and outliers.

Reads the 4 per-type CSVs produced by the scope pre-selection (script 05) and:
  1. Counts the variables (columns) per CSV.
  2. Lists the columns (and the type of each).
  3. Checks the decimal separator (comma vs dot).
  4. Converts NaN -> 0 in the numeric columns (upstream of the alignment).
  5. ALIGNS everything to a COMMON 1-min GRID, with configurable or automatic origin
     (by default, the latest start among meters; derived pseudo-meters, e.g. P01=100,
     are excluded from the origin computation; see --pseudo-meter-ids).
     Mean per minute; minutes without reading remain GAPS.
  The categories are located and saved IN ORDER, in 3 SEPARATE files:
  6. REAL ALIGNMENT GAPS: minutes without reading after the common 1-min grid, located in
     the aligned raw data before any conversion of sentinel/outlier to NaN.
     -> 06_gap_locations.csv.
  7. SENTINELS: (a) any point equal to the exact global floor of the variable, even
     isolated, and (b) approximate/almost horizontal lower plateau, confirmed by a run
     >=3 and then applied to all points of the same local lower group. For confirmed
     blocks, one finite sample at each edge is also removed, so that the first/last
     point of the sentinel does not remain as an anchor for the filling. Only LOCATES;
     does not convert to NaN at this step.
     -> 06_sentinel_locations.csv.
  8. OUTLIERS: short points outside the robust moving median, located after masking
     sentinels and gap edges only in a local working copy. Only LOCATES.
     -> 06_outlier_locations.csv.
  9. AT THE END, after saving the three locations: sentinels + outliers BECOME NaN in
     the consolidated file for filling. Real gaps were already NaN by absence of reading.

Rules: figures read from figdata/ -> this script COPIES the source files there; tables
read from tabdata/ -> this script writes the DATA there (the .tex is produced by script 22).
Never audits/.

Inputs (from script 05):
    data/interim/05_{primarias,secundarias,distorcao_meta,harmonicas}.csv

Outputs:
    data/interim/06_{primarias,secundarias,distorcao_meta,harmonicas}.csv   (aligned)
    data/interim/06_*_sentinelas.csv . 06_*_limpo.csv
    data/interim/06_para_preencher.csv                       (consolidated, 177 vars)
    audits/06_sentinel_locations.csv . 06_outlier_locations.csv . 06_gap_locations.csv   (3 locations)
    audits/06_variable_inventory.csv . 06_columns_long.csv . 06_alignment_coverage.csv
    tabdata/06_sentinels_by_variable.csv . 06_sentinels_currents.csv . 06_outliers.csv
    figdata/06_{primarias,secundarias,para_preencher,sentinel_locations,outlier_locations,gap_locations}
    logs/06_validate_and_zero.log . manifests/06_validate_and_zero_params.json

## Pipeline role

- File type: Data preprocessing script (pipeline stage 06: validation, 1-min alignment, real gaps, sentinels and outliers)
- Position in the canonical sequence: 6 of 29
- Preceding stage: `05_select_meter_scope_candidate.py`
- Following stage: `06b_defrag_envelope.py`

## Inputs

Declared in the module header:

- `data/interim/05_{primarias,secundarias,distorcao_meta,harmonicas}.csv`

Files read by the source code (resolved from the path expressions in the script):

- `data/interim/06_{…}.csv`

## Outputs

Declared in the module header:

- `data/interim/06_{primarias,secundarias,distorcao_meta,harmonicas}.csv   (aligned)`
- `data/interim/06_*_sentinelas.csv`
- `06_*_limpo.csv`
- `data/interim/06_para_preencher.csv                       (consolidated, 177 vars)`
- `audits/06_sentinel_locations.csv`
- `06_outlier_locations.csv`
- `06_gap_locations.csv   (3 locations)`
- `audits/06_variable_inventory.csv`
- `06_columns_long.csv`
- `06_alignment_coverage.csv`
- `tabdata/06_sentinels_by_variable.csv`
- `06_sentinels_currents.csv`
- `06_outliers.csv`
- `figdata/06_{primarias,secundarias,para_preencher,sentinel_locations,outlier_locations,gap_locations}`
- `logs/06_validate_and_zero.log`
- `manifests/06_validate_and_zero_params.json`

Files written by the source code (resolved from the path expressions in the script):

- `audits/06_alignment_coverage.csv`
- `audits/06_columns_long.csv`
- `audits/06_gap_locations.csv`
- `audits/06_outlier_locations.csv`
- `audits/06_sentinel_global_candidates.csv`
- `audits/06_sentinel_locations.csv`
- `audits/06_variable_inventory.csv`
- `data/interim/06_para_preencher.csv`
- `data/interim/06_{…}.csv`
- `data/interim/06_{…}_limpo.csv`
- `data/interim/06_{…}_sentinelas.csv`
- `manifests/06_validate_and_zero_params.json`
- `tabdata/{…}.csv`

## Configuration

- No configuration file is read directly.

Command-line arguments (from `argparse` in the source):

| Argument | Default | Description |
|---|---|---|
| `--grid-origin-mode` | GRID_ORIGIN_MODE | Grid-origin mode. auto_latest_start uses the latest start among meters; auto_first_start uses the first global timestamp; manual uses --grid-origin. Default: auto_latest_start. |
| `--grid-origin` | GRID_ORIGIN | Manual grid origin, e.g. '2020-10-22 15:21:00+00:00'. Used when --grid-origin-mode manual. |
| `--main-meter-id` | MAIN_METER_ID | ID of the general meter for the physical lock. Default: 1 (lock enabled). Use 'none' to disable. |
| `--pseudo-meter-ids` | PSEUDO_METER_IDS | IDs of derived pseudo-meters (e.g. P01=100), comma-separated, excluded from the grid-origin computation (auto_latest_start). They are aligned normally. 'none' for none. Default: 100. |

## Dependencies

- Third-party packages: `numpy`, `pandas`

## Usage

From the repository root:

```bash
python scripts/06_validate_sentinels_outliers.py [options]
```

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `audits/06_alignment_coverage.csv`
- `audits/06_columns_long.csv`
- `audits/06_gap_locations.csv`
- `audits/06_outlier_locations.csv`
- `audits/06_sentinel_global_candidates.csv`
- `audits/06_sentinel_locations.csv`
- `audits/06_variable_inventory.csv`
- `data/interim/06_para_preencher.csv`
- `data/interim/06_primarias.csv`
- `data/interim/06_secundarias.csv`
- `data/interim/06_{…}.csv`
- `data/interim/06_{…}_limpo.csv`
- `data/interim/06_{…}_sentinelas.csv`
- `logs/06_validate_and_zero.log`
- `manifests/06_validate_and_zero_params.json`
- `tabdata/{…}.csv`

## Reproducibility considerations

- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
