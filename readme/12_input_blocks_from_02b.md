# 12_input_blocks_from_02b.py

## Purpose

Link between the channel audit (script 03) and the explainability runners: map of the
input blocks X0/X1/X2.

Materialises the INPUT BLOCKS of the explainability analysis from the channel audit
(script 03), WITHOUT touching the locked selection logic (script 11) or the training. It
is a read-only stage that only ANNOTATES each M_G feature with its channel class, its
family G, and its membership in the blocks X0/X1/X2 and in the ablation families A0-A7.

Block definition (residual version, canonical):
    X0 = audited active powers                    (channel_class = ACTIVE_POWER_VALID)
    X1 = X0 + normalised instrumental proxies     (approved proxies, robust z-score)
    X2 = X0 + proxy residuals                     (r_x = z_x - f(P,Q,S); only the NON-redundant ones)

Redundancy rule (consistent with the channel audit): a proxy with R2_vs_Pref >= redund_r2
(0.95) is redundant -> leaves X2 (what is already explained by the power is not
residualised); the others enter as residuals. R2 >= 0.99 is only a marker of strong
redundancy.

Families G (= ablation families A0-A7):
    G_P=A0 (powers) . G_I=A1 (currents) . G_V=A2 (voltages) .
    G_hI=A3 (current harmonics) . G_hV=A4 (voltage harmonics) . G_THD=A5 (THD) .
    G_sec (q/s/cos, secondary proxy) . X1=A6 (all proxies) . X2=A7 (residuals).

This script does NOT decide the exclusion of any feature from the main flow; that is a
separate, approved decision. It only delivers the advisory map consumed by the
explainability runner.

Inputs:
    audits/03_channel_classification.csv            (classes per channel; filters M_G)
    audits/10_feature_dictionary.csv                (universe of M_G features)
    audits/11_selected_features_by_output.csv       (selected features; flag)
    config/meter_map.yaml (roles.m_g_input) . config/preprocessing_config.yaml (redund_r2)

Outputs:
    audits/12_input_blocks_from_02b.csv
    manifests/12_input_blocks_from_02b_params.json
    logs/12_input_blocks_from_02b.log

## Pipeline role

- File type: Pipeline script (stage 12: input-block map for explainability; read-only annotation)
- Position in the canonical sequence: 13 of 29
- Preceding stage: `11_feature_selection_rfecv_svr_kneedle.py`
- Following stage: `13_create_windows_splits.py`

## Inputs

Declared in the module header:

- `audits/03_channel_classification.csv            (classes per channel; filters M_G)`
- `audits/10_feature_dictionary.csv                (universe of M_G features)`
- `audits/11_selected_features_by_output.csv       (selected features; flag)`
- `config/meter_map.yaml (roles.m_g_input)`
- `config/preprocessing_config.yaml (redund_r2)`

Files read by the source code (resolved from the path expressions in the script):

- `audits/03_channel_classification.csv`
- `audits/10_feature_dictionary.csv`
- `audits/11_selected_features_by_output.csv`
- `config/meter_map.yaml`
- `config/preprocessing_config.yaml`
- `data/processed/11_candidate_sets.json`

## Outputs

Declared in the module header:

- `audits/12_input_blocks_from_02b.csv`
- `manifests/12_input_blocks_from_02b_params.json`
- `logs/12_input_blocks_from_02b.log`

Files written by the source code (resolved from the path expressions in the script):

- `audits/12_input_blocks_from_02b.csv`
- `manifests/12_input_blocks_from_02b_params.json`

## Configuration

- Reads `config/meter_map.yaml`
- Reads `config/preprocessing_config.yaml`

## Dependencies

- Third-party packages: `pandas`, `yaml`

## Usage

From the repository root:

```bash
python scripts/12_input_blocks_from_02b.py
```

The script has no command-line interface.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `audits/03_channel_classification.csv`
- `audits/10_feature_dictionary.csv`
- `audits/11_selected_features_by_output.csv`
- `audits/12_input_blocks_from_02b.csv`
- `config/meter_map.yaml`
- `config/preprocessing_config.yaml`
- `data/processed/11_candidate_sets.json`
- `logs/12_input_blocks_from_02b.log`
- `manifests/12_input_blocks_from_02b_params.json`

## Reproducibility considerations

- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
