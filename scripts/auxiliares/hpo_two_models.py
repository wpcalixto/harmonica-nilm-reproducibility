#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: hpo_two_models.py

File type: Auxiliary hyperparameter-optimisation and training script (supplementary analysis; requires GPU)

Purpose:
    Supplementary FAIR hyperparameter optimisation (HPO): PE-ES (proposed) + LSTM (competitor
    selected by ERG_val).

    PRIMARY objective = VALIDATION ERG (aligned with the selection criterion of the competitor);
    NAE_val as secondary metric. EQUAL GPU-hour budget per architecture (timeout), same sampler
    (TPE, fixed seed), same pruner (none), same early-stopping patience (fixed), test split LOCKED
    during the search. Protocol against single-seed bias:
      search (1 seed) -> top-3 -> re-evaluate top-3 on 3 seeds (ERG_val) -> freeze the best
      -> retrain 5 seeds -> only then TEST.
    Default x HPO comparison per model and between finalists (delta ERG abs/rel, paired by seed).

    Framing: SUPPLEMENTARY/EXPLORATORY analysis; conclusions restricted to the two models.

    Env: SMOKE=1 (short run), GPU_BUDGET_S=<seconds per model, default 5400>.
    Outputs: audits/hpo2_*.csv/json ; models/hpo2_*_seed_*.keras

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
import os, sys, json, gc, time, importlib.util, subprocess
from datetime import datetime, timezone
from pathlib import Path
SCRIPTS_DIR = Path(__file__).resolve().parents[1]   # scripts/ (holds _nn_common and the pipeline stages)
sys.path.insert(0, str(SCRIPTS_DIR))
import numpy as np, pandas as pd
import _nn_common as C

def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
m14 = _imp("m14", str(SCRIPTS_DIR / "14_train_baselines.py"))
m15 = _imp("m15", str(SCRIPTS_DIR / "15_train_advanced.py"))

import keras
from keras import layers
from keras.callbacks import EarlyStopping

DT = 1.0 / 60.0 / 1000.0
SMOKE = os.environ.get("SMOKE") == "1"
GPU_BUDGET_S = int(os.environ.get("GPU_BUDGET_S", "5400"))          # 1.5 h per model (real run)
SEARCH_SEED  = 42
REEVAL_SEEDS = [42, 123, 456]
FINAL_SEEDS  = [42, 123, 456, 789, 1024]
if SMOKE:
    GPU_BUDGET_S = int(os.environ.get("GPU_BUDGET_S", "90"))
    REEVAL_SEEDS, FINAL_SEEDS = [42, 123], [42, 123]
    EPOCHS = 2; PATIENCE = 2
else:
    EPOCHS = C.EPOCHS_MAX; PATIENCE = C.PATIENCE

AUD = C.PROJECT_ROOT / "audits"; MODELS = C.MODELS_DIR
CKPT = C.CKPT_ROOT / "hpo2"; CKPT.mkdir(parents=True, exist_ok=True)
for d in (AUD, MODELS): d.mkdir(parents=True, exist_ok=True)

# -- data (base series; test only after freezing) -------------------------------------------
data = C.load_splits()
n_feat, n_out = data["n_features"], data["output_dim"]
labels, scaler_Y = data["output_labels"], data["scaler_Y"]
Y_raw_val, Y_raw_test = data["Y_raw_val"], data["Y_raw_test"]
Xtr_ts, Ytr_ts = C.unwindow(data["X_train"]), C.unwindow(data["Y_train"])
Xv_ts,  Yv_ts  = C.unwindow(data["X_val"]),   C.unwindow(data["Y_val"])
Xte_ts         = C.unwindow(data["X_test"])
LOSS = C.make_masked_loss(C.phase_groups(labels))
m15._custom_layers(); m14._temporal_attention_cls()
W_STAR = json.loads((C.MANIFESTS_DIR / "15_train_advanced_params.json").read_text())["methodology"]["W_star"]

def _vram_mb():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                       "--format=csv,noheader,nounits"], timeout=8)
        return int(out.decode().split("\n")[0])
    except Exception:
        return -1

def erg_nae_on(model, W, Y_raw):
    """Systemic ERG (%) and mean active NAE (%) on a split (val/test), mid-point aligned."""
    Xw, _ = C.rewindow((Xv_ts if Y_raw is Y_raw_val else Xte_ts),
                       (Xv_ts if Y_raw is Y_raw_val else Xte_ts), W)
    mid = W // 2
    pw = model.predict(Xw, verbose=0); pm, _ = C.mid_align(pw, W)
    yp = scaler_Y.inverse_transform(pm)
    n = min(len(yp), len(Y_raw) - mid); yt = Y_raw[mid:mid + n]; yp = yp[:n]
    et = yt.sum(0) * DT; ep = yp.sum(0) * DT
    erg = float(abs(et.sum() - ep.sum()) / (abs(et.sum()) + 1e-9) * 100)
    active = [j for j in range(n_out) if abs(et[j]) >= 1.0]
    nae = float(np.mean([abs(yp[:, j].sum() - yt[:, j].sum()) / (abs(yt[:, j].sum()) + 1e-9) * 100
                         for j in active])) if active else float("nan")
    return erg, nae, len(Xw)

# -- search spaces (comparable extent, architecture-specific) --------------------------------
W_GRID = [6, 12, 18, 24, 30, 36]

def build_lstm_hp(W, hp, loss):
    inp = keras.Input(shape=(W, n_feat))
    x = inp
    for _ in range(hp["n_layers"]):
        x = layers.Bidirectional(layers.LSTM(hp["units"], return_sequences=True))(x)
        if hp["dropout"] > 0:
            x = layers.Dropout(hp["dropout"])(x)
    out = layers.TimeDistributed(layers.Dense(n_out, activation="linear"))(x)
    m = keras.Model(inp, out, name="LSTM_HPO")
    m.compile(optimizer=keras.optimizers.Adam(hp["lr"]), loss=loss)
    return m

def sample_hp(trial, kind):
    if kind == "LSTM":
        return {"W": trial.suggest_categorical("W", W_GRID),
                "n_layers": trial.suggest_categorical("n_layers", [1, 2]),
                "units": trial.suggest_categorical("units", [50, 100, 150, 200]),
                "dropout": trial.suggest_float("dropout", 0.0, 0.4, step=0.1),
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "batch": trial.suggest_categorical("batch", [16, 32, 64])}
    return {"W": trial.suggest_categorical("W", W_GRID),
            "conv1": trial.suggest_categorical("conv1", [32, 64, 128]),
            "conv2": trial.suggest_categorical("conv2", [64, 128, 256]),
            "kernel": trial.suggest_categorical("kernel", [3, 5]),
            "lstm": trial.suggest_categorical("lstm", [64, 100, 128, 200]),
            "dense1": trial.suggest_categorical("dense1", [64, 100, 128]),
            "dense2": trial.suggest_categorical("dense2", [32, 50, 64]),
            "dropout": trial.suggest_float("dropout", 0.10, 0.40, step=0.05),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch": trial.suggest_categorical("batch", [32, 64, 128])}

def build_model(kind, W, hp, loss):
    return build_lstm_hp(W, hp, loss) if kind == "LSTM" else m15.build_pe_es_hp(W, n_feat, n_out, hp, loss)

def train_one(kind, hp, seed, epochs=None):
    """Trains a model (kind,hp,seed) on the training split; returns trained model + number of windows."""
    W = hp["W"]
    Xtr, Ytr = C.rewindow(Xtr_ts, Ytr_ts, W); Xv, Yv = C.rewindow(Xv_ts, Yv_ts, W)
    C.set_seed(seed)
    m = build_model(kind, W, hp, loss=LOSS)
    m.fit(Xtr, Ytr, validation_data=(Xv, Yv), batch_size=hp["batch"], epochs=epochs or EPOCHS,
          callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                   restore_best_weights=True, verbose=0)], verbose=0)
    return m, len(Xtr), len(Xv)

# -- Optuna search per model (objective = ERG_val; equal timeout) ---------------------------
def search(kind, log):
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = f"sqlite:///{CKPT / f'{kind}_study.db'}"
    peak = {"vram": 0}
    def objective(trial):
        hp = sample_hp(trial, kind)
        try:
            m, ntr, nva = train_one(kind, hp, SEARCH_SEED)
        except Exception as e:
            raise optuna.TrialPruned() from e
        erg, nae, _ = erg_nae_on(m, hp["W"], Y_raw_val)
        trial.set_user_attr("nae_val", nae); trial.set_user_attr("n_win_tr", ntr)
        trial.set_user_attr("n_win_val", nva); trial.set_user_attr("W", hp["W"])
        peak["vram"] = max(peak["vram"], _vram_mb())
        del m; keras.backend.clear_session(); gc.collect()
        return erg
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=SEARCH_SEED),
                                storage=storage, study_name=f"{kind}_ERGval", load_if_exists=True,
                                pruner=optuna.pruners.NopPruner())
    t0 = time.time()
    study.optimize(objective, timeout=GPU_BUDGET_S, show_progress_bar=False)
    wall = time.time() - t0
    from optuna.trial import TrialState
    comp = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == TrialState.PRUNED]
    rows = [{"kind": kind, "trial": t.number, "erg_val": t.value,
             "nae_val": t.user_attrs.get("nae_val"), "n_win_tr": t.user_attrs.get("n_win_tr"),
             "n_win_val": t.user_attrs.get("n_win_val"), **t.params} for t in study.trials]
    pd.DataFrame(rows).to_csv(AUD / f"hpo2_trials_{kind}.csv", index=False)
    top3 = sorted(comp, key=lambda t: t.value)[:3]
    med = float(np.median([t.duration.total_seconds() for t in comp])) if comp else -1
    log(f"[{kind}] search: wall={wall:.0f}s trials_ok={len(comp)} pruned={len(pruned)} "
        f"median/trial={med:.1f}s VRAM~{peak['vram']}MB | top3 ERG_val="
        f"{[round(t.value,4) for t in top3]}")
    return {"kind": kind, "wall_s": round(wall, 1), "n_complete": len(comp), "n_pruned": len(pruned),
            "median_trial_s": round(med, 1), "peak_vram_mb": peak["vram"],
            "top3": [{"params": t.params, "erg_val_search": t.value} for t in top3]}

# -- re-evaluate top-3 on 3 seeds -> freeze -> retrain 5 seeds -----------------------------
def pick_and_finalize(kind, top3, log):
    reeval = []
    for i, cand in enumerate(top3):
        hp = cand["params"]; ergs = []
        for s in REEVAL_SEEDS:
            m, _, _ = train_one(kind, hp, s)
            e, _, _ = erg_nae_on(m, hp["W"], Y_raw_val); ergs.append(e)
            del m; keras.backend.clear_session(); gc.collect()
        reeval.append({"cand": i, "params": hp, "erg_val_mean": float(np.mean(ergs)),
                       "erg_val_std": float(np.std(ergs)), "erg_seeds": ergs})
    best = min(reeval, key=lambda r: r["erg_val_mean"])
    log(f"[{kind}] top-3 re-evaluation ({len(REEVAL_SEEDS)} seeds): "
        f"{[round(r['erg_val_mean'],4) for r in reeval]} -> FREEZE h* ERG_val="
        f"{best['erg_val_mean']:.4f}")
    # final retraining on 5 seeds + TEST (only now)
    rows = []
    for s in FINAL_SEEDS:
        m, _, _ = train_one(kind, best["params"], s)
        ev, nv, _ = erg_nae_on(m, best["params"]["W"], Y_raw_val)
        et, nt, _ = erg_nae_on(m, best["params"]["W"], Y_raw_test)
        m.save(MODELS / f"hpo2_{kind}_seed_{s}.keras")
        rows.append({"kind": kind, "config": "HPO", "seed": s, "W": best["params"]["W"],
                     "erg_val": ev, "nae_val": nv, "erg_test": et, "nae_test": nt})
        del m; keras.backend.clear_session(); gc.collect()
    return best, reeval, rows

# -- DEFAULT configuration: reuses existing models (val + test per seed) -------------------
def eval_standard(kind, prefix, W, log):
    rows = []
    for s in FINAL_SEEDS:
        p = MODELS / f"{prefix}_seed_{s}.keras"
        if not p.exists(): continue
        m = keras.models.load_model(str(p), compile=False, safe_mode=False,
                                    custom_objects=m15._custom_layers())
        ev, nv, _ = erg_nae_on(m, W, Y_raw_val)
        et, nt, _ = erg_nae_on(m, W, Y_raw_test)
        rows.append({"kind": kind, "config": "standard", "seed": s, "W": W,
                     "erg_val": ev, "nae_val": nv, "erg_test": et, "nae_test": nt})
        del m; keras.backend.clear_session(); gc.collect()
    log(f"[{kind}] default evaluated ({len(rows)} seeds, W={W})")
    return rows


def main():
    import logging, sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(C.PROJECT_ROOT / "logs" / "hpo_two_models.log",
                                                      mode="w")])
    log = logging.getLogger("hpo2").info
    log(f"=== HPO 2 models (PE-ES + LSTM) | SMOKE={SMOKE} budget={GPU_BUDGET_S}s/model "
        f"| objective=ERG_val | W*_pe_es(default)={W_STAR} ===")
    KINDS = [("PE_ES", "15_pe_es", W_STAR), ("LSTM", "14_lstm", 12)]

    all_seed_rows, search_meta, sel_meta = [], {}, {}
    for kind, std_prefix, std_W in KINDS:
        smeta = search(kind, log)
        best, reeval, hpo_rows = pick_and_finalize(kind, smeta["top3"], log)
        std_rows = eval_standard(kind, std_prefix, std_W, log)
        all_seed_rows += hpo_rows + std_rows
        search_meta[kind] = smeta
        sel_meta[kind] = {"frozen_hp": best["params"], "erg_val_mean_reeval": best["erg_val_mean"],
                          "reeval": reeval, "std_W": std_W}

    seed_df = pd.DataFrame(all_seed_rows)
    seed_df.to_csv(AUD / "hpo2_final_per_seed.csv", index=False)

    # default x HPO comparison per model (delta ERG abs/rel) + paired by seed
    comp_rows = []
    for kind, _, _ in KINDS:
        for cfg in ("standard", "HPO"):
            sub = seed_df[(seed_df.kind == kind) & (seed_df.config == cfg)]
            comp_rows.append({"kind": kind, "config": cfg,
                              "erg_val_mean": round(sub.erg_val.mean(), 4), "erg_val_std": round(sub.erg_val.std(), 4),
                              "erg_test_mean": round(sub.erg_test.mean(), 4), "erg_test_std": round(sub.erg_test.std(), 4),
                              "nae_val_mean": round(sub.nae_val.mean(), 4), "nae_test_mean": round(sub.nae_test.mean(), 4)})
    comp_df = pd.DataFrame(comp_rows)
    # delta ERG per model (val)
    delta_rows = []
    for kind, _, _ in KINDS:
        st = seed_df[(seed_df.kind == kind) & (seed_df.config == "standard")].set_index("seed")
        hp = seed_df[(seed_df.kind == kind) & (seed_df.config == "HPO")].set_index("seed")
        common = sorted(set(st.index) & set(hp.index))
        for split in ("val", "test"):
            es = st.loc[common, f"erg_{split}"].to_numpy(); eh = hp.loc[common, f"erg_{split}"].to_numpy()
            d_abs = es - eh; d_rel = 100 * (es - eh) / (es + 1e-12)
            delta_rows.append({"kind": kind, "split": split,
                               "erg_std_mean": round(float(es.mean()), 4), "erg_hpo_mean": round(float(eh.mean()), 4),
                               "dERG_abs_mean": round(float(d_abs.mean()), 4), "dERG_rel_pct_mean": round(float(d_rel.mean()), 3),
                               "dERG_abs_by_seed": [round(x, 4) for x in d_abs.tolist()],
                               "n_seeds": len(common)})
    pd.DataFrame(delta_rows).to_csv(AUD / "hpo2_delta_by_model.csv", index=False)
    comp_df.to_csv(AUD / "hpo2_comparison.csv", index=False)

    # delta between finalists per seed (val and test), default and HPO
    pe = seed_df[seed_df.kind == "PE_ES"].set_index(["config", "seed"])
    ls = seed_df[seed_df.kind == "LSTM"].set_index(["config", "seed"])
    fin_rows = []
    for cfg in ("standard", "HPO"):
        for split in ("val", "test"):
            common = sorted(set(pe.loc[cfg].index) & set(ls.loc[cfg].index))
            dpe = pe.loc[cfg].loc[common, f"erg_{split}"].to_numpy()
            dls = ls.loc[cfg].loc[common, f"erg_{split}"].to_numpy()
            fin_rows.append({"config": cfg, "split": split,
                             "erg_PE_ES_mean": round(float(dpe.mean()), 4), "erg_LSTM_mean": round(float(dls.mean()), 4),
                             "delta_PE_minus_LSTM_by_seed": [round(x, 4) for x in (dpe - dls).tolist()],
                             "ranking": "PE_ES<LSTM" if dpe.mean() < dls.mean() else "LSTM<=PE_ES"})
    pd.DataFrame(fin_rows).to_csv(AUD / "hpo2_finalists_delta.csv", index=False)

    manifest = {"script": "hpo_two_models.py", "run_timestamp": datetime.now(timezone.utc).isoformat(),
                "mode": "SMOKE" if SMOKE else "full", "proposed": "PE_ES", "competitor": "LSTM",
                "competitor_criterion": "argmin ERG_val (individual, non-ensemble); post-hoc/exploratory",
                "primary_objective": "ERG_val", "secondary_metric": "NAE_val",
                "gpu_budget_seconds_per_model": GPU_BUDGET_S, "same_budget": True, "sampler": "TPE",
                "optuna_seed": SEARCH_SEED, "pruner": "NopPruner", "es_patience": PATIENCE, "epochs_max": EPOCHS,
                "reeval_seeds": REEVAL_SEEDS, "final_seeds": FINAL_SEEDS,
                "protocol": "search(1 seed)->top3->reeval(3 seeds ERG_val)->freeze->retrain(5 seeds)->TEST",
                "test_accessed_during_search": False, "objective_scope": "validation_only",
                "search_space_dims": {"LSTM": 6, "PE_ES": 10},
                "search": search_meta, "selection": sel_meta}
    (AUD / "hpo2_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log("=== HPO 2 models finished ===")
    log("comparison:\n" + comp_df.to_string(index=False))


if __name__ == "__main__":
    main()
