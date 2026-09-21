# Changelog

## [2.0.0] - 2026-09-21

Complete replacement of the package. Version 1.0.1 (DOI 10.5281/zenodo.20681042) contained an
earlier pipeline that no longer corresponds to the published article and an uncorrected dataset;
it is superseded by this release.

### Added
- Canonical pipeline of the article: 29 stages (`scripts/01_*.py` … `28_*.py`, including `06b`),
  shared module `_nn_common.py`, 15 auxiliary scripts (`scripts/auxiliares/`).
- Trained models (`models/`, 35 `.keras`), per-window predictions of every model and seed
  (`predictions/`) and execution manifests of the reference run (`manifests/`).
- Six YAML configuration files (`config/`).
- Public dataset `data/raw/dados_maior_v2_transformed.csv` (reversible per-column bilinear
  transformation of the physical dataset; see README, "Data transformation notice").
- One README per Python file (`readme/`), main README, `docs/`, `CITATION.cff`, `.zenodo.json`,
  `LICENSE` (MIT, code) and `LICENSE-DATA` (CC BY 4.0, data/models/predictions/manifests).

### Changed
- All human-readable content (headers, docstrings, comments, messages, YAML comments, READMEs)
  in English; standardised provenance header in every Python file.
- Every script resolves the repository root from its own location; no dependency on
  developer-specific paths or on historical execution directories.
- Version, date and DOI updated (2.0.0, 2026-09-21, 10.5281/zenodo.22878924).

### Removed
- The 29 scripts of the 1.0.x pipeline (`codigos/`), the uncorrected raw file `dados_maior.csv`,
  and the documentation that described them.
