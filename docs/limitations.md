# Known limitations

1. **Single facility, seven days.** The dataset covers one industrial installation over 6.94
   days at 1-minute resolution; results are not transferable without re-running the pipeline
   on new data.
2. **Unmeasured support source.** The auxiliary source is not metered. The scale-admissibility
   gate refuses the physical topological balance (no absolute energy of the support source is
   estimated); the load-consistency residual R_cl is a consistency diagnostic, not a measurement.
3. **Transformed public dataset.** Absolute values and linear correlations between columns are
   not preserved in `data/raw/dados_maior_v2_transformed.csv`; only ordering and temporal
   structure are. Exact numerical reproduction needs the inverse-transformation key.
4. **Reconstructed inputs.** Part of the input series is gap-filled (script 08); the strict
   leakage control (script 28) evaluates only windows whose 48 inputs and 18 targets are
   originally observed, which is possible for the W = 12 models only.
5. **Statistical separation.** The three best models are not statistically separable under
   the clustered bootstrap (confidence intervals include zero); model ranking is descriptive.
6. **Null predictor.** Three of the six models do not outperform the constant (median)
   predictor on the common support; see the article's supplementary material.
7. **HPO sensitivity.** The PE-ES-Optuna run and `scripts/auxiliares/hpo_two_models.py` are
   sensitivity analyses outside the main ranking; their configurations differ in window length
   and support from the default ones.
8. **GPU requirement.** Stages 14–15 and the auxiliary training scripts need a CUDA-capable
   GPU for a practical runtime.
9. **Meters M14/M15.** Sampled at 2-minute cadence and without temporal overlap; M14 is
   treated as part of P_other and M15 as load Gamma_5 after the electrical audit (script 09).
