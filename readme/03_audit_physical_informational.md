# 03_audit_physical_informational.py

## Purpose

Physical and informational audit (read-only).

DIAGNOSTIC stage between ingestion (02) and scope selection (05). Does NOT correct, does
NOT remove, does NOT impute: reads the standardised parquet (faithful to the raw data),
measures and classifies every channel (meter x variable x phase). The per-channel verdict
governs the downstream policy.

TWO-LEVEL DESIGN:
  * PHASE LEVEL: `phase_physical_consistency` measures whether the phase CLOSES as an
    electrical system (triangle P^2+Q^2<=S^2, cos in [-1,1], THD>=0, harmonics 0-100 %,
    voltage). It is a MARKER; it is NOT automatically inherited by the channels.
  * CHANNEL LEVEL: `channel_class`; each QUANTITY is evaluated by its own criterion.
    A phase can be NON_PHYSICAL and still have `p_a` as ACTIVE_POWER_VALID.

ACTIVE POWER has its own rule (not downgraded by Q/S/cos/V/I): P_phi is a physical
candidate if it is not a dominant sentinel, not stuck, has a temporal dynamics compatible
with a load and a finite accumulated energy. A triangle violation is attributed to Q/S
(own check S>=|P|, |Q|<=S), never to P.

METHODOLOGICAL DECISION: the magnitude test against an external reference (P_lim) is
WITHDRAWN from the main study: there is no source independent of the CSV, so any limit
extracted from the data itself would be circular. With external_power_limit_kw null, one
always gets energy_plausibility=WARN and scale_status=UNKNOWN_NO_REFERENCE, and
FAIL_EXTERNAL_REF is never applied. The per-meter/phase mechanism remains in the code,
inert, only activatable if a reliable external reference appears. Admissible conclusion:
"P_phi was the most reliable quantity after the internal audit", NOT "the absolute scale
of P_phi is confirmed".

REDUNDANCY tested against a HIERARCHICAL REFERENCE of AUDITED ACTIVE powers:
  P_ref = { P_phi ACTIVE_POWER_VALID } union { P_tot = sum of valid P_phi }.
  R2_{x|P_ref} = R2( x(t) ~ f(P_ref(t)) ); redundant if >= redund_r2 (strong >= redund_strong).
Thus a current/harmonic is not declared redundant because of a weak or broken phase power.

CHANNEL TAXONOMY (real physical data):
  PHYSICAL_VALID        quantity passes its own physical checks (valid measurement)
  ACTIVE_POWER_VALID    active power passes its own audit (physical basis / balance)
  PHYSICAL_ALERT        live quantity with a physical inconsistency (outside the instrument
                        range, cos>1, THD<0, S<|P|): alert, not exclusion
  REDUNDANT_WITH_POWER  explained by P_ref (R2>=0.95)
  DEAD_OR_STUCK         sentinel, constant or without dynamics (exclude)
  EXCLUDE               no physical nor informational value (exclude)

The voltage band is the INSTRUMENT RANGE (Parametros.jpeg): L-N 1-300 Vrms.

Method guarantees:
  * SENTINEL-AWARE: physics measured only on real points (NaN/zero/dominant floor excluded).
  * PRE-FILL: informational statistics before the filling; MAD robust to offset.

Input:
    data/interim/02_raw_standardized.parquet   (produced by 02, without repair)

Outputs (prefix 03_):
    audits/03_phys_audit.csv                 (meter x phase; phase consistency)
    audits/03_info_audit.csv                 (meter x variable x phase; metrics)
    audits/03_channel_classification.csv     (consolidated verdict per channel)
    audits/03_channel_eligibility.csv        (aggregated eligibility; source of table_02)
    audits/03_energy_coherence_alert.csv
    audits/03_audit_summary.md
    logs/03_audit_physical_informational.log
    manifests/03_audit_physical_informational_params.json

## Pipeline role

- File type: Diagnostic script (pipeline stage 03: physical and informational channel audit; read-only)
- Position in the canonical sequence: 3 of 29
- Preceding stage: `02_ingest_raw.py`
- Following stage: `04_inverse_physical_compatibility.py`

## Inputs

Declared in the module header:

- `data/interim/02_raw_standardized.parquet   (produced by 02, without repair)`

Files read by the source code (resolved from the path expressions in the script):

- `config/meter_map.yaml`
- `config/preprocessing_config.yaml`
- `data/interim/02_raw_standardized.parquet`

## Outputs

Declared in the module header:

- `audits/03_phys_audit.csv                 (meter x phase; phase consistency)`
- `audits/03_info_audit.csv                 (meter x variable x phase; metrics)`
- `audits/03_channel_classification.csv     (consolidated verdict per channel)`
- `audits/03_channel_eligibility.csv        (aggregated eligibility; source of table_02)`
- `audits/03_energy_coherence_alert.csv`
- `audits/03_audit_summary.md`
- `logs/03_audit_physical_informational.log`
- `manifests/03_audit_physical_informational_params.json`

Files written by the source code (resolved from the path expressions in the script):

- `audits/03_audit_summary.md`
- `audits/03_channel_classification.csv`
- `audits/03_channel_eligibility.csv`
- `audits/03_energy_coherence_alert.csv`
- `audits/03_info_audit.csv`
- `audits/03_phys_audit.csv`
- `manifests/03_audit_physical_informational_params.json`

## Configuration

- Reads `config/meter_map.yaml`
- Reads `config/preprocessing_config.yaml`

## Dependencies

- Third-party packages: `numpy`, `pandas`, `yaml`

## Usage

From the repository root:

```bash
python scripts/03_audit_physical_informational.py
```

The script has no command-line interface.

## Notes

All repository paths referenced in the source code (relative to the repository root; `{…}` denotes a runtime value):

- `audits/03_audit_summary.md`
- `audits/03_channel_classification.csv`
- `audits/03_channel_eligibility.csv`
- `audits/03_energy_coherence_alert.csv`
- `audits/03_info_audit.csv`
- `audits/03_phys_audit.csv`
- `config/meter_map.yaml`
- `config/preprocessing_config.yaml`
- `data/interim/02_raw_standardized.parquet`
- `logs/03_audit_physical_informational.log`
- `manifests/03_audit_physical_informational_params.json`

## Reproducibility considerations

- The scientific logic (thresholds, windows, metrics, statistical procedures) is locked; only paths and documentation were adapted for the public release.
- The public dataset `data/raw/dados_maior_v2_transformed.csv` carries reversibly transformed values; exact numerical reproduction of the article requires the inverse-transformation key (see the main README).
