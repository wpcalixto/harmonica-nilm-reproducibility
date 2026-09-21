# Reproducibility notes

## Reference run

The shipped `manifests/` directory records the reference run: Python 3.11.0, TensorFlow 2.17.0,
Keras 3.4.1, NumPy 1.26.4, pandas 3.0.3, scikit-learn 1.9.0, SciPy 1.17.1, Optuna 4.9.0,
Matplotlib 3.11.0 (see `manifests/25_reproducibility_manifest.json`). Training stages were
executed on a cloud GPU (Linux); the CPU stages on macOS/Linux.

## Seeds and splits

- Seeds: 42, 123, 456, 789, 1024 (five seeds per model), fixed in `config/model_config.yaml`
  and in `scripts/_nn_common.py`; every training script calls `set_seed()` per seed.
- Chronological split 80/10/10 at time-step level (`13_create_windows_splits.py`); the test
  split is never used for selection (window W*, SPEC loads, ablation decisions are taken on
  validation).
- Window length W = 12 for LSTM/RCNN-att/Ensemble_G4, W* (selected on validation) for PE-ES
  and SPEC, W = 36 for MoTE_v2. Predictions are aligned on the original time axis by the
  window mid-point (offset W//2) before any metric.

## What can be reproduced without a GPU

Stages 16–28 and every auxiliary script consume the shipped `predictions/` (and, for PFI and
the HPO figures, `models/` and `audits/hpo2_*`); they reproduce the metrics, balance,
statistics, explainability, economics, tables and figures of the article on a CPU.

## What requires the original data

The public dataset is a reversible transformation of the physical measurements. Stages 02–13
run on it end-to-end, but their numerical outputs (audits, features, windows) differ from the
article. Exact reproduction of stages 02–15 requires the inverse-transformation key
(available under the Data Use Agreement, `docs/DUA_template.md`); the restored file is
byte-identical to the original (verified by SHA-256).

## Numerical determinism

GPU training is not bit-reproducible across hardware/driver versions; differences are within
the inter-seed dispersion reported in the article. CPU stages are deterministic given the
pinned library versions.

## Checksums

`checksums/checksums_sha256.txt` lists the SHA-256 of every file of the package; verify with
`sha256sum -c checksums/checksums_sha256.txt` (Linux) or `shasum -a 256 -c` (macOS).
