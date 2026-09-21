# 02_ingest_raw.py

## Purpose

Reading of the raw file + split by variable TYPE (parquet + 4 CSVs).

Reads ALL variables of the raw file and builds the outputs WITHOUT any correction (no
sentinel decoding, no outlier removal, no imputation): read-only, structural
standardisation and split by type. Corrections start in script 06 (sentinels and
outliers), 06b (envelope sanitisation) and 08 (gap filling).

Operations:
  1. Read the raw CSV (data/raw/dados_maior_v2_transformed.csv): all columns.
  2. Convert the timestamp to datetime; sort by time (stable).
  3. Remove only EXACT duplicates (no electrical aggregation).
  4. Write the standardised parquet.
  5. Split the columns into 4 TYPES and write 1 CSV per type.

Types (182 columns in total):
  - primary     (6):  i_an,i_bn,i_cn, v_an,v_bn,v_cn               (direct measurements)
  - secondary   (12): p_*,q_*,s_*,cos_*                            (derived)
  - distortion+meta (11): thdi_*,thdv_* (6) + meta (5: Unnamed:0,time,
                          datetime_read,meter_id,tag_meter_id)
  - harmonics   (153): hrm_i_* (75) + hrm_v_* (78)                 (spectrum)

Alignment NOTE: the meta columns (time, meter_id) go ONLY in the distortion+meta CSV.
The other 3 CSVs are aligned by ROW ORDER (all come from the same sorted table); to
rebuild a record, read the 4 files and concatenate by position.

Outputs (all with prefix 02_):
    data/interim/02_raw_standardized.parquet
    data/interim/02_primarias.csv
    data/interim/02_secundarias.csv
    data/interim/02_distorcao_meta.csv
    data/interim/02_harmonicas.csv
    audits/02_raw_overview.csv
    figdata/02_timestamp_coverage.csv
    logs/02_ingest_raw.log
    manifests/02_ingest_raw_params.json

## Pipeline role

- File type: Data ingestion script (pipeline stage 02)
- Position in the canonical sequence: 2 of 29
- Preceding stage: `01_config.py`
- Following stage: `03_audit_physical_informational.py`

## Inputs

Files read by the source code (resolved from the path expressions in the script):

- `config/project_config.yaml`

## Outputs

Declared in the module header:

- `data/interim/02_raw_standardized.parquet`
- `data/interim/02_primarias.csv`
- `data/interim/02_secundarias.csv`
- `data/interim/02_distorcao_meta.csv`
- `data/interim/02_harmonicas.csv`
- `audits/02_raw_overview.csv`
- `figdata/02_timestamp_coverage.csv`
- `logs/02_ingest_raw.log`
- `manifests/02_ingest_raw_params.json`

Files written by the source code (resolved from the path expressions in the script):

- `audits/02_raw_overview.csv`
- `data/interim/02_raw_standardized.parquet`
- `data/interim/{fname}`
- `figdata/02_timestamp_coverage.csv`
- `manifests/02_ingest_raw_params.json`

## Configuration

- Reads `config/project_config.yaml`

## Dependencies

- Third-party packages: `pandas`, `yaml`

## Usage

From the repository root:

```bash
python scripts/02_ingest_raw.py
```

The script has no command-line interface.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `audits/02_raw_overview.csv`
- `config/project_config.yaml`
- `data/interim/02_raw_standardized.parquet`
- `data/interim/{fname}`
- `data/raw/dados_maior_v2_transformed.csv`
- `figdata/02_timestamp_coverage.csv`
- `logs/02_ingest_raw.log`
- `manifests/02_ingest_raw_params.json`

## Reproducibility considerations

- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
