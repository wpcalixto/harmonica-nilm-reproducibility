#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: 04_inverse_physical_compatibility.py

File type: Diagnostic script (pipeline stage 04: inverse physical-compatibility control; read-only)

Purpose:
    Inverse physical-compatibility problem (read-only physical control).

    READ-ONLY physical control between the audit (03) and the scope selection (05). Does NOT
    recalibrate the data automatically. For each meter x phase, evaluates whether the
    electrical relations can be reconciled by a MINIMAL ADJUSTMENT close to the identity:

        V~ = a_V*V + b_V ,  I~ = a_I*I + b_I ,  S~ = a_S*S ,  Q~ = a_Q*Q ,  (a_P=1, b_P=0)

        J(theta) = w_vi*sum rho(S~ - V~*I~) + w_pcos*sum rho(P - S~*cos(phi))
                 + w_tri*sum rho(S~^2 - P^2 - Q~^2) + lambda*||theta - theta0||^2 ,
        theta0 = (1,0,1,0,1,1)

    rho = robust loss (Huber). The active power P is the fixed energy quantity (not adjusted).
    The result is DIAGNOSTIC: it measures compatibility and triggers alert/sensitivity
    flags, without altering the data.

    Status: COMPATIBLE (low residuals and theta close to the identity) . ALERT (moderate
    residual or theta far from the identity) . INCOMPATIBLE_PERSISTENT (high residual after
    the minimal adjustment).

    Input:
        data/interim/02_raw_standardized.parquet
    Parameters: block `inverse_compatibility` of preprocessing_config.yaml (built-in fallback).

    Outputs (prefix 04_):
        audits/04_inverse_residuals_by_phase.csv     (residuals before/after + status)
        audits/04_inverse_parameter_shifts.csv       (theta* per meter x phase)
        audits/04_inverse_identifiability.csv        (n_real, convergence, identifiability)
        figdata/04_inverse_residuals_long.csv        (long: term x stage)
        logs/04_inverse_physical_compatibility.log
        manifests/04_inverse_physical_compatibility_params.json

Repository: harmonica-nilm-reproducibility
Version: v2.0.0
Date: 2026-09-21

Developer:
    Prof. Wesley Pacheco Calixto, Dr.

Authors:
    1. Wesley Pacheco Calixto
       Federal University of Goias / University of Coimbra /
       Federal Institute of Goias, Brazil

    2. Jose A. Gobbes Cararo
       Federal University of Goias / Federal Institute Goiano, Brazil

    3. Guilherme A. Sousa Ribeiro
       Federal University of Goias / Federal Institute of Goias, Brazil

    4. Paulo Victor Santos
       Federal University of Goias / Federal Institute of Goias, Brazil

Corresponding author:
    Wesley Pacheco Calixto
    wesley.pacheco@isr.uc.pt

License:
    MIT

Citation:
    Calixto, W. P., Cararo, J. A. G., Ribeiro, G. A. S., & Santos, P. V.
    (2026). Reproducibility package for non-intrusive load monitoring
    in industrial three-phase multi-source systems (Version 2.0.0)
    [Computer software]. Zenodo.
    https://doi.org/10.5281/zenodo.22878924

Related publication:
    Scale-admissibility gate for non-intrusive load monitoring in
    three-phase industrial energy systems with an unmeasured support
    source. Applied Energy.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT  = SCRIPT_DIR.parent  # repository root (parent of scripts/)
CONFIG_DIR    = PROJECT_ROOT / "config"
INTERIM_DIR   = PROJECT_ROOT / "data" / "interim"
AUDITS_DIR    = PROJECT_ROOT / "audits"
FIGDATA_DIR   = PROJECT_ROOT / "figdata"
LOGS_DIR      = PROJECT_ROOT / "logs"
MANIFESTS_DIR = PROJECT_ROOT / "manifests"

PARQUET_IN    = INTERIM_DIR / "02_raw_standardized.parquet"
METER_COL     = "meter_id"
PHASES        = ["a", "b", "c"]
THETA0        = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0])   # (a_V,b_V,a_I,b_I,a_S,a_Q)
MIN_POINTS    = 30

_DEFAULTS = {
    "w_vi": 1.0, "w_pcos": 1.0, "w_tri": 1.0, "lambda_reg": 0.10, "huber_delta": 0.05,
    "resid_compatible": 0.05, "resid_incompatible": 0.20, "param_dist_alert": 0.20,
}


def _load_cfg() -> dict:
    vals = dict(_DEFAULTS)
    try:
        import yaml
        cfg = yaml.safe_load((CONFIG_DIR / "preprocessing_config.yaml").read_text(encoding="utf-8"))
        blk = (cfg or {}).get("inverse_compatibility", {}) or {}
        for k in vals:
            if blk.get(k) is not None:
                vals[k] = float(blk[k])
    except Exception:
        pass
    return vals


CFG = _load_cfg()


def setup_logger(p: Path) -> logging.Logger:
    p.parent.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("04_inverse"); lg.setLevel(logging.DEBUG); lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(p, mode="w", encoding="utf-8"); fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt)
    lg.addHandler(fh); lg.addHandler(ch); return lg


def rel_residuals(theta, V, I, P, Q, S, C, sV, sI, sS, sP):
    """POINT-BY-POINT relative residuals (bounded, robust to scale outliers).
    Denominator = the quantity itself at each point (as in script 03); sV/sI only scale the offset."""
    aV, bV, aI, bI, aS, aQ = theta
    eps = 1e-9
    Vt = aV * V + bV * sV
    It = aI * I + bI * sI
    St = aS * S
    Qt = aQ * Q
    r_vi   = (St - Vt * It) / (np.abs(S) + eps)
    r_pcos = (P - St * C) / (np.abs(P) + eps)
    r_tri  = (St ** 2 - P ** 2 - Qt ** 2) / (S ** 2 + eps)
    return r_vi, r_pcos, r_tri


def _huber(r, d):
    a = np.abs(r)
    return np.where(a <= d, 0.5 * r * r, d * (a - 0.5 * d))


def fit_phase(V, I, P, Q, S, C):
    """Minimal adjustment close to the identity. Scalar objective: Huber (MEAN per point) on the
    data only + pure quadratic penalty ||theta - theta0||^2 (not robustified). L-BFGS-B with bounds.

    Note: using least_squares(loss='huber') would ALSO robustify the penalty, flattening it;
    hence the scalar objective via minimize."""
    from scipy.optimize import minimize
    sV = float(np.median(np.abs(V))) or 1.0
    sI = float(np.median(np.abs(I))) or 1.0
    sS = float(np.median(np.abs(S))) or 1.0
    sP = float(np.median(np.abs(P))) or 1.0
    w = (CFG["w_vi"], CFG["w_pcos"], CFG["w_tri"])
    lam = CFG["lambda_reg"]; d = CFG["huber_delta"]
    med = lambda x: float(np.median(np.abs(x)))

    # Residuals at the IDENTITY (theta0). If already compatible, do NOT adjust (avoids moving theta needlessly).
    b_vi, b_pcos, b_tri = rel_residuals(THETA0, V, I, P, Q, S, C, sV, sI, sS, sP)
    before_max = max(med(b_vi), med(b_pcos), med(b_tri))
    if before_max <= CFG["resid_compatible"]:
        return {"theta": THETA0.copy(), "converged": True, "cost": before_max,
                "resid_vi_before": med(b_vi), "resid_pcos_before": med(b_pcos),
                "resid_tri_before": med(b_tri), "resid_vi_after": med(b_vi),
                "resid_pcos_after": med(b_pcos), "resid_tri_after": med(b_tri),
                "param_distance": 0.0, "fitted": False}

    def obj(theta):
        r_vi, r_pcos, r_tri = rel_residuals(theta, V, I, P, Q, S, C, sV, sI, sS, sP)
        data = (w[0] * _huber(r_vi, d).mean() + w[1] * _huber(r_pcos, d).mean()
                + w[2] * _huber(r_tri, d).mean())
        pen = lam * float(np.sum((np.asarray(theta) - THETA0) ** 2))
        return data + pen

    # Bounds: MINIMAL adjustment (a in [0.5,2], b in [-0.5,0.5] p.u.); prevents a degenerate theta.
    bounds = [(0.5, 2.0), (-0.5, 0.5), (0.5, 2.0), (-0.5, 0.5), (0.5, 2.0), (0.5, 2.0)]
    res = minimize(obj, THETA0.copy(), method="L-BFGS-B", bounds=bounds)
    theta = res.x
    a_vi, a_pcos, a_tri = rel_residuals(theta, V, I, P, Q, S, C, sV, sI, sS, sP)
    return {
        "theta": theta, "converged": bool(res.success), "cost": float(res.fun), "fitted": True,
        "resid_vi_before": med(b_vi), "resid_pcos_before": med(b_pcos), "resid_tri_before": med(b_tri),
        "resid_vi_after": med(a_vi), "resid_pcos_after": med(a_pcos), "resid_tri_after": med(a_tri),
        "param_distance": float(np.linalg.norm(theta - THETA0)),
    }


def classify(r_after: float, param_dist: float) -> str:
    if r_after > CFG["resid_incompatible"]:
        return "INCOMPATIBLE_PERSISTENT"
    if r_after <= CFG["resid_compatible"] and param_dist <= CFG["param_dist_alert"]:
        return "COMPATIBLE"
    return "ALERT"


def main() -> None:
    log = setup_logger(LOGS_DIR / "04_inverse_physical_compatibility.log")
    log.info("=== 04_inverse_physical_compatibility.py: inverse compatibility ===")
    if not PARQUET_IN.exists():
        log.error(f"Parquet missing (run script 02): {PARQUET_IN}"); sys.exit(1)
    try:
        import scipy  # noqa: F401
    except Exception:
        log.error("scipy unavailable: required for least_squares."); sys.exit(1)

    df = pd.read_parquet(PARQUET_IN)
    meters = sorted(df[METER_COL].dropna().unique().tolist())
    cols = set(df.columns)
    log.info(f"Meters: {meters} | thresholds: {CFG}")

    res_rows, par_rows, ident_rows, long_rows = [], [], [], []
    for meter in meters:
        dm = df[df[METER_COL] == meter]
        for ph in PHASES:
            need = [f"v_{ph}n", f"i_{ph}n", f"p_{ph}", f"q_{ph}", f"s_{ph}", f"cos_{ph}"]
            if not all(c in cols for c in need):
                continue
            V, I, P, Q, S, C = (dm[c].to_numpy(float) for c in need)
            ok = np.isfinite(V) & np.isfinite(I) & np.isfinite(P) & np.isfinite(Q) \
                & np.isfinite(S) & np.isfinite(C) & (S > 0)
            n = int(ok.sum())
            PH = ph.upper()
            if n < MIN_POINTS:
                ident_rows.append({"meter_id": meter, "phase": PH, "n_real": n,
                                   "converged": False, "identifiable": False, "cost": np.nan})
                res_rows.append({"meter_id": meter, "phase": PH, "n_real": n,
                                 "inverse_status": "DEAD_OR_INSUFFICIENT"})
                continue
            f = fit_phase(V[ok], I[ok], P[ok], Q[ok], S[ok], C[ok])
            r_after = max(f["resid_vi_after"], f["resid_pcos_after"], f["resid_tri_after"])
            status = classify(r_after, f["param_distance"])
            res_rows.append({"meter_id": meter, "phase": PH, "n_real": n,
                             "resid_vi_before": round(f["resid_vi_before"], 5),
                             "resid_vi_after": round(f["resid_vi_after"], 5),
                             "resid_pcos_before": round(f["resid_pcos_before"], 5),
                             "resid_pcos_after": round(f["resid_pcos_after"], 5),
                             "resid_tri_before": round(f["resid_tri_before"], 5),
                             "resid_tri_after": round(f["resid_tri_after"], 5),
                             "resid_after_max": round(r_after, 5),
                             "param_distance": round(f["param_distance"], 5),
                             "inverse_status": status})
            aV, bV, aI, bI, aS, aQ = f["theta"]
            par_rows.append({"meter_id": meter, "phase": PH, "a_V": round(aV, 4),
                             "b_V": round(bV, 4), "a_I": round(aI, 4), "b_I": round(bI, 4),
                             "a_S": round(aS, 4), "a_Q": round(aQ, 4)})
            ident_rows.append({"meter_id": meter, "phase": PH, "n_real": n,
                               "converged": f["converged"],
                               "identifiable": bool(f["converged"] and n >= MIN_POINTS),
                               "cost": round(f["cost"], 6)})
            for term, bfr, aft in [("vi", f["resid_vi_before"], f["resid_vi_after"]),
                                   ("pcos", f["resid_pcos_before"], f["resid_pcos_after"]),
                                   ("tri", f["resid_tri_before"], f["resid_tri_after"])]:
                long_rows.append({"meter_id": meter, "phase": PH, "term": term,
                                  "stage": "before", "residual": round(bfr, 5)})
                long_rows.append({"meter_id": meter, "phase": PH, "term": term,
                                  "stage": "after", "residual": round(aft, 5)})
        log.info(f"  meter {meter}: phases processed")

    AUDITS_DIR.mkdir(parents=True, exist_ok=True); FIGDATA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    res_df = pd.DataFrame(res_rows)
    pd.DataFrame(par_rows).to_csv(AUDITS_DIR / "04_inverse_parameter_shifts.csv", index=False)
    pd.DataFrame(ident_rows).to_csv(AUDITS_DIR / "04_inverse_identifiability.csv", index=False)
    res_df.to_csv(AUDITS_DIR / "04_inverse_residuals_by_phase.csv", index=False)
    pd.DataFrame(long_rows).to_csv(FIGDATA_DIR / "04_inverse_residuals_long.csv", index=False)

    counts = res_df["inverse_status"].value_counts().to_dict()
    log.info(f"Inverse status: {counts}")
    for _, r in res_df[res_df["inverse_status"].isin(["ALERT", "INCOMPATIBLE_PERSISTENT"])].iterrows():
        log.warning(f"  {r['inverse_status']}: M{r['meter_id']} phase {r['phase']} "
                    f"(resid_after={r.get('resid_after_max')}, param_dist={r.get('param_distance')})")

    manifest = {
        "script": "04_inverse_physical_compatibility.py", "section": "§5 / §10 (v8)",
        "run_timestamp": datetime.now().isoformat(),
        "role": "read-only physical control between 03 and 05; does not recalibrate the data",
        "model": "minimal_identity_adjustment: theta=(a_V,b_V,a_I,b_I,a_S,a_Q), a_P=1,b_P=0",
        "thresholds": CFG, "status_counts": counts,
        "inputs": {"parquet": str(PARQUET_IN.relative_to(PROJECT_ROOT))},
        "outputs": {
            "residuals_by_phase": "audits/04_inverse_residuals_by_phase.csv",
            "parameter_shifts": "audits/04_inverse_parameter_shifts.csv",
            "identifiability": "audits/04_inverse_identifiability.csv",
            "residuals_long": "figdata/04_inverse_residuals_long.csv",
        },
        "status": "success",
    }
    (MANIFESTS_DIR / "04_inverse_physical_compatibility_params.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log.info("=== 04 done ===")


if __name__ == "__main__":
    main()
