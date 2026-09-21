# 01_config.py

## Purpose

Central project configuration loader and validator.

Loads all 6 YAML configuration files (config/), validates that all locked methods have
status 'locked_main_method', creates the output directories, checks that the raw data
file exists, and writes the resolved manifest + log.

Outputs:
    manifests/01_config_resolved.json
    logs/01_config.log

Success criterion:  all locked methods appear in the manifest with status locked_main_method
Failure criterion:  the script raises SystemExit if any method is generic/equivalent/to_be_decided

## Pipeline role

- File type: Configuration loader and validator (pipeline stage 01; imported by no other script)
- Position in the canonical sequence: 1 of 29
- Preceding stage: `none (first stage)`
- Following stage: `02_ingest_raw.py`

## Inputs

Files read by the source code (resolved from the path expressions in the script):

- none detected by static analysis (inputs may be passed through shared helpers or environment variables)

## Outputs

Declared in the module header:

- `manifests/01_config_resolved.json`
- `logs/01_config.log`

Files written by the source code (resolved from the path expressions in the script):

- `manifests/01_config_resolved.json`

## Configuration

- Reads `config/dg_config.yaml`
- Reads `config/meter_map.yaml`
- Reads `config/model_config.yaml`
- Reads `config/preprocessing_config.yaml`
- Reads `config/project_config.yaml`
- Reads `config/scenario_config.yaml`

## Dependencies

- Third-party packages: `yaml`

## Usage

From the repository root:

```bash
python scripts/01_config.py
```

The script has no command-line interface.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `config/dg_config.yaml`
- `config/meter_map.yaml`
- `config/model_config.yaml`
- `config/preprocessing_config.yaml`
- `config/project_config.yaml`
- `config/scenario_config.yaml`
- `logs/01_config.log`
- `manifests/01_config_resolved.json`

## Reproducibility considerations

- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
