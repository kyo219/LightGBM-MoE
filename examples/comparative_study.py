#!/usr/bin/env python
# coding: utf-8
"""
comparative_study.py — naive vs naive-ensemble vs MoE 大規模比較 + ハイパラ重要度分析

`benchmark.py` の data generators (Synthetic / Hamilton / VIX) と独自の CV を再利用しつつ、
3 variants で Optuna を回す:

  - **naive-lightgbm**: 単一の標準 GBDT
  - **naive-ensemble**: K (=2〜4) 個の標準 GBDT を異なる seed で訓練して予測を平均。
    MoE と同じ K-way 多モデル容量を持ち「単純平均で十分か / gating が真に効くか」の
    fair な ablation。同一ハイパラ + per-member seed のみ変える seed-ensemble。
  - **moe**: K experts + gate (token / expert choice 横断)

  - どっちが精度が良いか (best / median RMSE)
  - どっちが速いか (per-trial train time の中央値)
  - **MoE で「明らかに良い」設定は何か** を分析:
      A. fANOVA importance ランキング     (各パラメータが RMSE 分散にどれだけ寄与したか)
      B. categorical の値別平均 RMSE + t検定 (best vs runner-up)
      D. numeric の quartile 別平均 RMSE  (sweet spot 検出)
      E. Optuna slice plot                (各パラメータ値 vs RMSE 散布図)

mixture_init は {random, gmm, tree_hierarchical} に限定 (uniform/quantile/balanced_kmeans 除外)。

Determinism note (--n-jobs default=1):
    Earlier versions defaulted to n_jobs=6 to "speed up" Optuna. In practice,
    n_jobs>1 makes best RMSE non-reproducible across runs / builds even with
    seed fixed: the TPE sampler's recommendations depend on the order parallel
    workers report observations back, and that order is non-deterministic. On
    a regression-checking run (synthetic, 500 trials, 3 seeds × 2 builds, see
    `bench_logs/regression_check_*`) the n_jobs=6 best-RMSE std was ±0.31 —
    enough to make a single-seed n_jobs=6 result swing by ±0.6, large enough
    to falsely flag any code change as a regression. Worse, n_jobs=6 was *not*
    actually faster: with each LightGBM trial saturating all cores via OMP,
    6 concurrent trials oversubscribe the CPU and run ~18% slower per dataset
    than a single sequential pipeline (234s vs 287s on synthetic in that run).

    The default is now n_jobs=1: deterministic best RMSE across runs and
    builds, AND faster wall time. Override with --n-jobs >1 only if you
    explicitly need the (noisy) speed/coverage tradeoff and don't care about
    cross-run comparison.

Usage:
    # 小規模 sanity check
    python examples/comparative_study.py --trials 50 --out bench_results/study_smoke.json

    # 本番 500 trials × 2 variants × 6 datasets (deterministic)
    python examples/comparative_study.py --trials 500 --out bench_results/study_500.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import optuna
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# Build override for A/B comparisons: LGBM_MOE_PACKAGE_DIR points at an
# alternate python-package directory (e.g. a worktree build of an older
# release). An editable install registers a meta-path finder that wins over
# sys.path, so it must be disarmed for the override to take effect. The
# resolved package path + lib sha256 are recorded in the output provenance.
_PKG_DIR = os.environ.get("LGBM_MOE_PACKAGE_DIR")
if _PKG_DIR:
    sys.meta_path = [f for f in sys.meta_path
                     if "editable" not in type(f).__module__.lower()]
    sys.path.insert(0, _PKG_DIR)
else:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python-package"))
import lightgbm_moe as lgb

if _PKG_DIR and not os.path.abspath(lgb.__file__).startswith(os.path.abspath(_PKG_DIR)):
    raise SystemExit(f"LGBM_MOE_PACKAGE_DIR={_PKG_DIR} requested but lightgbm_moe "
                     f"resolved to {lgb.__file__}")

sys.path.insert(0, os.path.dirname(__file__))
from benchmark import (  # noqa: E402
    BenchmarkConfig,
    generate_fred_gdp_data,
    generate_hmm_data,
    generate_sp500_basic_data,
    generate_sp500_data,
    generate_synthetic_data,
    generate_vix_data,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# matplotlib for slice plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Search space constants (per user spec)
# =============================================================================
INIT_CHOICES = ["random", "gmm", "tree_hierarchical", "uniform"]
GATE_CHOICES = ["gbdt", "none", "leaf_reuse"]
ROUTING_CHOICES = ["token_choice", "expert_choice"]
E_STEP_MODES = ["em", "loss_only", "gate_only"]
SMOOTHING_CHOICES = ["none", "ema", "markov"]


# =============================================================================
# CV with per-trial timing
# =============================================================================
# Audit fixes baked into the evaluation protocol:
#
#   1. Early stopping no longer validates on the scoring fold. The old code
#      passed the fold's validation set to `early_stopping(50)` and then
#      scored predictions on that same set — best_iteration was tuned
#      per-fold on the evaluation data, optimistically biasing every fold
#      score (and differentially favoring high-capacity variants). Each
#      fold's training window now donates its chronological tail
#      (ES_FRACTION) as the early-stopping set; the fold's validation
#      window is only ever predicted once.
#   2. TimeSeriesSplit(gap=1): with next-step targets, the last training
#      row's label is the first thing the adjacent validation rows' lag
#      features encode. A 1-step embargo removes that boundary reuse.
#   3. Per-fold RMSEs and fold failure counts are returned so the study can
#      report crash rates and run paired per-fold comparisons, instead of
#      silently folding failures into an `inf` mean.
ES_FRACTION = 0.15
CV_GAP = 1


def _es_tail_split(n_rows: int):
    """Chronological (fit, early-stop) split of a training window."""
    n_es = max(20, int(n_rows * ES_FRACTION))
    n_es = min(n_es, n_rows // 2)  # degenerate-window guard
    return n_rows - n_es


def _train_with_es(params, Xt, yt, num_boost_round):
    """Train with the ES set carved from the tail of the training window."""
    cut = _es_tail_split(len(Xt))
    train = lgb.Dataset(Xt[:cut], label=yt[:cut])
    es_set = lgb.Dataset(Xt[cut:], label=yt[cut:], reference=train)
    return lgb.train(
        params,
        train,
        num_boost_round=num_boost_round,
        valid_sets=[es_set],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )


def evaluate_cv_timed(X, y, params, n_splits: int, num_boost_round: int):
    """Time-series CV (gap=1, ES on train tail).

    Returns (rmse_mean, mean_train_seconds_per_fold, fold_rmses, n_failed_folds).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=CV_GAP)
    rmses = []
    times = []
    n_failed = 0
    for tr_idx, va_idx in tscv.split(X):
        try:
            t0 = time.perf_counter()
            model = _train_with_es(params, X[tr_idx], y[tr_idx], num_boost_round)
            t1 = time.perf_counter()
            pred = model.predict(X[va_idx])
            rmses.append(float(np.sqrt(mean_squared_error(y[va_idx], pred))))
            times.append(t1 - t0)
        except Exception:
            rmses.append(float("inf"))
            n_failed += 1
    mean_time = float(np.mean(times)) if times else 0.0
    return float(np.mean(rmses)), mean_time, rmses, n_failed


def evaluate_ensemble_cv_timed(X, y, params, n_models: int, base_seed: int,
                               n_splits: int, num_boost_round: int):
    """Time-series CV with a K-way naive seed-ensemble of LightGBMs.

    Each ensemble member uses identical hyperparameters but a per-member seed
    offset, so bagging / feature-fraction / extra_trees randomness diverges
    across members. Predictions are averaged. Total tree budget per fold is
    `n_models * num_boost_round`, matching MoE's K * num_boost_round.
    Same ES-tail / gap protocol as `evaluate_cv_timed`.

    Returns (rmse_mean, mean_total_train_seconds_per_fold, fold_rmses, n_failed_folds).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=CV_GAP)
    rmses = []
    times = []
    n_failed = 0
    for tr_idx, va_idx in tscv.split(X):
        try:
            t0 = time.perf_counter()
            preds = np.zeros(len(va_idx), dtype=np.float64)
            for k in range(n_models):
                p = dict(params)
                # `seed` is LightGBM's master seed and propagates to bagging /
                # feature-fraction / extra-trees per the docs. Setting it
                # per-member is the cleanest way to diverge member k from j.
                p["seed"] = base_seed + 1009 * k  # arbitrary multiplier to
                                                  # avoid 1-step adjacency
                model = _train_with_es(p, X[tr_idx], y[tr_idx], num_boost_round)
                preds += model.predict(X[va_idx])
            t1 = time.perf_counter()
            preds /= n_models
            rmses.append(float(np.sqrt(mean_squared_error(y[va_idx], preds))))
            times.append(t1 - t0)
        except Exception:
            rmses.append(float("inf"))
            n_failed += 1
    mean_time = float(np.mean(times)) if times else 0.0
    return float(np.mean(rmses)), mean_time, rmses, n_failed


# =============================================================================
# Optuna objectives
# =============================================================================
def make_naive_lightgbm_objective(X, y, cfg: BenchmarkConfig, trial_log: list):
    def objective(trial):
        params = {
            "objective": "regression",
            "boosting": "gbdt",
            "verbose": -1,
            "num_threads": 4,
            "seed": cfg.seed,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
            "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
        }
        rmse, train_s, fold_rmses, n_failed = evaluate_cv_timed(
            X, y, params, cfg.n_splits, cfg.num_boost_round)
        trial_log.append({"variant": "naive-lightgbm", "rmse": rmse, "train_s": train_s,
                          "fold_rmses": fold_rmses, "n_failed_folds": n_failed,
                          "params": dict(trial.params),
                          "lgbm_params": dict(params)})
        return rmse

    return objective


def make_naive_ensemble_objective(X, y, cfg: BenchmarkConfig, trial_log: list):
    """K-way seed-ensemble of standard LightGBM models. Same hyperparam search
    space as `make_naive_lightgbm_objective` plus `n_models` ∈ {2, 3, 4} so
    the ensemble has the same K-way capacity range as MoE. Per-fold compute
    is K × naive's, matching MoE's K × num_boost_round tree budget.
    """
    def objective(trial):
        n_models = trial.suggest_int("n_models", 2, 4)
        params = {
            "objective": "regression",
            "boosting": "gbdt",
            "verbose": -1,
            "num_threads": 4,
            # `seed` is overridden per-member inside evaluate_ensemble_cv_timed;
            # this top-level value is not actually used by the ensemble.
            "seed": cfg.seed,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
            "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
        }
        rmse, train_s, fold_rmses, n_failed = evaluate_ensemble_cv_timed(
            X, y, params, n_models, cfg.seed, cfg.n_splits, cfg.num_boost_round)
        trial_log.append({"variant": "naive-ensemble", "rmse": rmse, "train_s": train_s,
                          "fold_rmses": fold_rmses, "n_failed_folds": n_failed,
                          "params": dict(trial.params),
                          "lgbm_params": dict(params), "n_models": n_models})
        return rmse

    return objective


def make_moe_objective(X, y, cfg: BenchmarkConfig, trial_log: list,
                       refit_mode: str = "search", refit_decay: float = 0.0):
    """Build an Optuna objective that trains MoE.

    `refit_mode` controls how the v0.7 / v0.8 refit + regrow machinery is
    handled across the search:

      - "search"  : (v0.8 default) refit_leaves / refit_trigger / regrow*
                    are themselves Optuna search variables. Each trial
                    samples its own refit/regrow configuration along with
                    the rest of the MoE hyperparameters. This is what the
                    v0.8 study uses to ask: "if Optuna can pick refit and
                    regrow alongside everything else, do v0.8 features
                    rise to the top?"
      - "off"     : force ``mixture_refit_leaves=false`` (v0.6 baseline,
                    used to reproduce the v0.6 README headline numbers
                    without contamination)
      - "always" / "elbo" / "every_n":
                    force a specific refit trigger; regrow / decay are
                    NOT searched (compatible with the legacy v0.7
                    ablation runs in bench_results/study_v07_*_report.md)

    `refit_decay` is only used when refit_mode is one of the legacy
    forced-on triggers; in "search" mode the decay is itself a search
    variable.
    """
    def objective(trial):
        smoothing = trial.suggest_categorical("mixture_r_smoothing", SMOOTHING_CHOICES)
        routing_mode = trial.suggest_categorical("mixture_routing_mode", ROUTING_CHOICES)
        gate_type = trial.suggest_categorical("mixture_gate_type", GATE_CHOICES)
        num_experts = trial.suggest_int("mixture_num_experts", 2, 4)

        params = {
            "objective": "regression",
            "boosting": "mixture",
            "verbose": -1,
            "num_threads": 4,
            "seed": cfg.seed,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
            "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
            "mixture_init": trial.suggest_categorical("mixture_init", INIT_CHOICES),
            "mixture_num_experts": num_experts,
            # NOTE (audit fix): mixture_e_step_alpha used to be searched here
            # (0.1–3.0), but it is IGNORED whenever mixture_estimate_variance
            # is true — which is the default and was never in the search
            # space. Every trial therefore carried a dead dimension that
            # wasted TPE budget and produced meaningless fANOVA / quartile
            # rows for `mixture_e_step_alpha` in past reports.
            "mixture_e_step_mode": trial.suggest_categorical("mixture_e_step_mode", E_STEP_MODES),
            "mixture_warmup_iters": trial.suggest_int("mixture_warmup_iters", 5, 50),
            "mixture_balance_factor": trial.suggest_int("mixture_balance_factor", 2, 10),
            "mixture_r_smoothing": smoothing,
            "mixture_smoothing_lambda": (
                trial.suggest_float("mixture_smoothing_lambda", 0.0, 0.9) if smoothing != "none" else 0.0
            ),
            "mixture_diversity_lambda": trial.suggest_float("mixture_diversity_lambda", 0.0, 0.5),
            "mixture_routing_mode": routing_mode,
            "mixture_hard_m_step": trial.suggest_categorical("mixture_hard_m_step", [True, False]),
            "mixture_gate_type": gate_type,
        }

        if gate_type in ("gbdt", "leaf_reuse"):
            params["mixture_gate_max_depth"] = trial.suggest_int("mixture_gate_max_depth", 2, 10)
            params["mixture_gate_num_leaves"] = trial.suggest_int("mixture_gate_num_leaves", 4, 64)
            params["mixture_gate_learning_rate"] = trial.suggest_float("mixture_gate_learning_rate", 0.01, 0.5, log=True)
            params["mixture_gate_lambda_l2"] = trial.suggest_float("mixture_gate_lambda_l2", 1e-3, 10.0, log=True)
            params["mixture_gate_iters_per_round"] = trial.suggest_int("mixture_gate_iters_per_round", 1, 3)
        if gate_type == "leaf_reuse":
            params["mixture_gate_retrain_interval"] = trial.suggest_int("mixture_gate_retrain_interval", 5, 30)

        if routing_mode == "expert_choice":
            params["mixture_expert_capacity_factor"] = trial.suggest_float("mixture_expert_capacity_factor", 0.8, 1.5)
            params["mixture_expert_choice_score"] = "combined"
            params["mixture_expert_choice_boost"] = trial.suggest_float("mixture_expert_choice_boost", 5.0, 30.0)
            params["mixture_expert_choice_hard"] = trial.suggest_categorical("mixture_expert_choice_hard", [True, False])

        # v0.7 leaf-refit + v0.8 partition re-grow.
        #
        # Three modes for this block:
        #   - refit_mode == "off":     leaf refit forced off, no regrow.
        #                              Reproduces v0.6 baseline.
        #   - refit_mode == "search":  Optuna samples the refit/regrow
        #                              knobs as part of the trial. This is
        #                              the v0.8 study mode — lets the
        #                              optimizer decide whether a tuned
        #                              config benefits from refit/regrow
        #                              and at what configuration.
        #   - refit_mode in {always,elbo,every_n}: legacy v0.7 ablation
        #                              path; trigger forced, regrow off.
        if refit_mode == "search":
            refit_on = trial.suggest_categorical("mixture_refit_leaves",
                                                 [True, False])
            if refit_on:
                trigger = trial.suggest_categorical(
                    "mixture_refit_trigger", ["always", "elbo", "every_n"])
                params["mixture_refit_leaves"] = True
                params["mixture_refit_trigger"] = trigger
                # decay is mostly best at 0.0 (full replace) per v0.7
                # acceptance — search a narrow range around it.
                params["mixture_refit_decay_rate"] = trial.suggest_float(
                    "mixture_refit_decay_rate", 0.0, 0.5)

                if trigger == "elbo":
                    # v0.8 elbo trigger has its own threshold + window
                    # knobs. Search log-uniform over a sensible range —
                    # 0.001-0.05 for drop, 0.0001-0.01 for plateau.
                    params["mixture_elbo_drop_threshold"] = trial.suggest_float(
                        "mixture_elbo_drop_threshold", 1e-3, 5e-2, log=True)
                    params["mixture_elbo_plateau_threshold"] = trial.suggest_float(
                        "mixture_elbo_plateau_threshold", 1e-4, 1e-2, log=True)
                    params["mixture_elbo_window"] = trial.suggest_int(
                        "mixture_elbo_window", 5, 20)
                elif trigger == "every_n":
                    params["mixture_refit_every_n"] = trial.suggest_int(
                        "mixture_refit_every_n", 5, 30)

                # v0.8 partition re-grow — only meaningful when refit is on.
                # leaf_reuse + regrow is auto-disabled at C++ Init (warning),
                # so no Python-side guard is needed.
                regrow_on = trial.suggest_categorical(
                    "mixture_regrow_oldest_trees", [True, False])
                if regrow_on:
                    params["mixture_regrow_oldest_trees"] = True
                    params["mixture_regrow_per_fire"] = trial.suggest_int(
                        "mixture_regrow_per_fire", 1, 5)
                    params["mixture_regrow_mode"] = trial.suggest_categorical(
                        "mixture_regrow_mode", ["replace", "delete"])
        elif refit_mode != "off":
            # Legacy v0.7 forced-trigger path (back-compat for older
            # ablation scripts).
            params["mixture_refit_leaves"] = True
            params["mixture_refit_trigger"] = refit_mode
            params["mixture_refit_decay_rate"] = refit_decay

        rmse, train_s, fold_rmses, n_failed = evaluate_cv_timed(
            X, y, params, cfg.n_splits, cfg.num_boost_round)
        trial_log.append({"variant": "moe", "rmse": rmse, "train_s": train_s,
                          "fold_rmses": fold_rmses, "n_failed_folds": n_failed,
                          "params": dict(trial.params),
                          "lgbm_params": dict(params)})
        return rmse

    return objective


# =============================================================================
# Hyperparameter analysis
# =============================================================================
def fanova_importance(study) -> dict:
    """Optuna param importance (default: MeanDecreaseImpurity)."""
    try:
        imp = optuna.importance.get_param_importances(study)
        return {k: round(float(v), 4) for k, v in imp.items()}
    except Exception as e:
        return {"_error": str(e)[:80]}


def categorical_value_stats(trials: list[dict], param: str) -> dict:
    """For each value of `param`, compute mean RMSE / std / count.
    Then identify the best value and run a t-test against the runner-up.
    Returns {"per_value": {...}, "best": {"value":..., "vs_runner_up_p":...}}.
    """
    finite = [t for t in trials if np.isfinite(t["rmse"]) and param in t["params"]]
    if not finite:
        return {"per_value": {}, "best": None}

    by_val: dict = {}
    for t in finite:
        by_val.setdefault(t["params"][param], []).append(t["rmse"])

    per_value = {}
    for v, rmses in by_val.items():
        per_value[str(v)] = {
            "n": len(rmses),
            "mean": round(float(np.mean(rmses)), 4),
            "std": round(float(np.std(rmses)), 4),
            "min": round(float(np.min(rmses)), 4),
        }

    if len(per_value) < 2:
        return {"per_value": per_value, "best": None}

    sorted_vals = sorted(per_value.items(), key=lambda kv: kv[1]["mean"])
    best_val, runner_up = sorted_vals[0][0], sorted_vals[1][0]
    best_rmses = by_val[type(list(by_val.keys())[0])(best_val) if not isinstance(list(by_val.keys())[0], str) else best_val]
    runner_rmses = by_val[type(list(by_val.keys())[0])(runner_up) if not isinstance(list(by_val.keys())[0], str) else runner_up]

    try:
        t_stat, p_val = stats.ttest_ind(best_rmses, runner_rmses, equal_var=False)
    except Exception:
        t_stat, p_val = float("nan"), float("nan")

    return {
        "per_value": per_value,
        "best": {
            "value": str(best_val),
            "runner_up": str(runner_up),
            "delta_mean": round(per_value[str(runner_up)]["mean"] - per_value[str(best_val)]["mean"], 4),
            "p_value": round(float(p_val), 6) if np.isfinite(p_val) else None,
            "significant": (np.isfinite(p_val) and p_val < 0.01),
        },
    }


def numeric_quartile_stats(trials: list[dict], param: str) -> dict:
    """Sort trials by `param` value into Q1..Q4, return mean RMSE per quartile."""
    finite = [t for t in trials if np.isfinite(t["rmse"]) and param in t["params"]]
    if len(finite) < 8:
        return {}
    vals = np.array([float(t["params"][param]) for t in finite])
    rmses = np.array([t["rmse"] for t in finite])
    qs = np.quantile(vals, [0.25, 0.5, 0.75])

    out = {}
    for label, lo, hi in [("Q1", -np.inf, qs[0]), ("Q2", qs[0], qs[1]), ("Q3", qs[1], qs[2]), ("Q4", qs[2], np.inf)]:
        mask = (vals >= lo) & (vals < hi) if label != "Q4" else (vals >= lo)
        if not mask.any():
            continue
        out[label] = {
            "range": [round(float(lo), 4) if np.isfinite(lo) else None, round(float(hi), 4) if np.isfinite(hi) else None],
            "n": int(mask.sum()),
            "mean_rmse": round(float(rmses[mask].mean()), 4),
        }
    out["best_quartile"] = min(("Q1", "Q2", "Q3", "Q4"), key=lambda q: out.get(q, {"mean_rmse": float("inf")})["mean_rmse"])
    return out


def make_slice_plot(study, out_path: str, title: str):
    """Save an optuna slice plot as PNG."""
    try:
        from optuna.visualization.matplotlib import plot_slice
        ax = plot_slice(study)
        # plot_slice returns a single Axes or array; gather a Figure
        if hasattr(ax, "figure"):
            fig = ax.figure
        elif hasattr(ax, "__iter__"):
            fig = ax[0].figure if len(ax) > 0 else None
        else:
            fig = plt.gcf()
        if fig is not None:
            fig.suptitle(title, fontsize=14)
            fig.set_size_inches(max(12, 2 * len(study.best_params)), 8)
            fig.tight_layout()
            fig.savefig(out_path, dpi=80, bbox_inches="tight")
            plt.close(fig)
        return True
    except Exception as e:
        print(f"    [warn] slice plot for {title} failed: {e}")
        return False


# =============================================================================
# Aggregator
# =============================================================================
def aggregate_variant(name: str, trials: list[dict], study) -> dict:
    finite = [t for t in trials if np.isfinite(t["rmse"])]
    if not finite:
        return {"variant": name, "n_trials": len(trials), "n_finite": 0}

    rmses = np.array([t["rmse"] for t in finite])
    times = np.array([t["train_s"] for t in finite])

    out = {
        "variant": name,
        "n_trials": len(trials),
        "n_finite": len(finite),
        "rmse_best": float(rmses.min()),
        "rmse_p10": float(np.quantile(rmses, 0.10)),
        "rmse_median": float(np.median(rmses)),
        "train_s_median": float(np.median(times)),
        "train_s_mean": float(np.mean(times)),
        "train_s_p90": float(np.quantile(times, 0.90)),
        "best_params": dict(study.best_params),
        "importance": fanova_importance(study),
    }

    cat_params = (
        ["mixture_gate_type", "mixture_routing_mode", "mixture_e_step_mode", "mixture_init",
         "mixture_r_smoothing", "mixture_hard_m_step", "extra_trees"]
        if name == "moe" else ["extra_trees"]
    )
    out["categorical_stats"] = {p: categorical_value_stats(trials, p) for p in cat_params}

    num_params = (
        ["mixture_num_experts", "mixture_diversity_lambda",
         "mixture_warmup_iters", "mixture_balance_factor", "learning_rate",
         "num_leaves", "max_depth", "min_data_in_leaf"]
        if name == "moe"
        else ["learning_rate", "num_leaves", "max_depth", "min_data_in_leaf",
              "lambda_l1", "lambda_l2", "feature_fraction", "bagging_fraction"]
    )
    out["numeric_quartiles"] = {p: numeric_quartile_stats(trials, p) for p in num_params}

    return out


# =============================================================================
# Markdown report
# =============================================================================
def render_markdown(results: dict, out_path: str, slice_paths: dict):
    lines = []
    lines.append("# Comparative Study Report — naive vs naive-ensemble vs MoE\n")
    cfg = results.get("config", {})
    lines.append(f"- **Trials per (variant × dataset × seed)**: {cfg.get('trials')}\n")
    lines.append(f"- **Datasets**: {cfg.get('datasets')}, **seeds**: {cfg.get('seeds')}\n")
    lines.append(f"- **n_splits**: {cfg.get('splits')} (gap={cfg.get('cv_gap')}), "
                 f"**rounds**: {cfg.get('rounds')}, "
                 f"**holdout**: final {cfg.get('holdout_frac', 0):.0%} (never seen by Optuna), "
                 f"**ES**: chronological tail {cfg.get('es_fraction', 0):.0%} of each train window\n")
    prov = results.get("provenance", {})
    if prov:
        lines.append(f"- **Build**: commit `{prov.get('git_commit', '?')[:12]}`"
                     f"{' (dirty)' if prov.get('git_dirty') else ''}, "
                     f"lib sha256 `{prov.get('lib_sha256', '?')[:12]}…`, "
                     f"package `{prov.get('lightgbm_moe_package', '?')}`\n")
    lines.append("\n---\n")

    # Headline: holdout metric (mean ± std across seeds)
    lines.append("## Headline: holdout RMSE (chronological tail, evaluated once per seed)\n")
    lines.append("Selection happened on CV inside the search region; this table is the "
                 "unbiased comparison. `cv_best` is the (optimistic) selection metric, "
                 "shown for reference.\n")
    lines.append("| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |")
    lines.append("|---|---|---|---|---|---|")
    for ds_name, variants in results.get("summary", {}).items():
        for v, e in variants.items():
            per_seed = e.get("per_seed", {})
            crash = np.mean([s.get("crash_rate", 0) for s in per_seed.values()]) if per_seed else 0
            retrain = np.mean([s.get("retrain_s") or 0 for s in per_seed.values()]) if per_seed else 0
            hm = e.get("holdout_mean")
            hs = e.get("holdout_std", 0)
            hm_str = f"**{hm:.5f}** ± {hs:.5f}" if hm is not None else "—"
            cb = e.get("cv_best_mean")
            cb_str = f"{cb:.5f}" if cb is not None else "—"
            lines.append(f"| {ds_name} | {v} | {hm_str} | {cb_str} | "
                         f"{crash:.1%} | {retrain:.2f} |")
    lines.append("\n")

    # Secondary: CV selection-metric table per run
    lines.append("## Selection metric per run (CV over the search region)\n")
    lines.append("| Run | Variant | cv best | cv median | median train s/fold | wall s |")
    lines.append("|---|---|---|---|---|---|")
    for run_tag, ds in results.get("runs", {}).items():
        for v in ds.get("_variants", []):
            r = ds.get(v, {})
            lines.append(
                f"| {run_tag} | {v} | {r.get('rmse_best', float('nan')):.4f} "
                f"| {r.get('rmse_median', float('nan')):.4f} "
                f"| {r.get('train_s_median', 0):.3f} | {r.get('wall_s', 0):.0f} |"
            )
    lines.append("\n")

    # Per-run detail
    for run_tag, ds in results.get("runs", {}).items():
        ds_name = run_tag
        lines.append(f"\n---\n\n## {run_tag}  (search X={ds.get('X_search_shape')}, "
                     f"holdout n={ds.get('n_holdout')})\n")

        for v in ds.get("_variants", []):
            r = ds.get(v, {})
            if not r:
                continue
            lines.append(f"\n### {v}\n")
            hold = r.get("holdout", {})
            if hold:
                lines.append(f"- **holdout RMSE: {hold.get('holdout_rmse', float('nan')):.5f}** "
                             f"(winner retrained in {hold.get('retrain_s', 0):.2f}s, "
                             f"cv score of winner: {hold.get('cv_rmse_of_winner', float('nan')):.4f})")
            lines.append(f"- cv best RMSE: {r.get('rmse_best', float('nan')):.4f}, "
                         f"median: {r.get('rmse_median', float('nan')):.4f}, "
                         f"p10: {r.get('rmse_p10', float('nan')):.4f}")
            lines.append(f"- train: median {r.get('train_s_median', 0):.3f}s/fold, "
                         f"mean {r.get('train_s_mean', 0):.3f}s, p90 {r.get('train_s_p90', 0):.3f}s")
            lines.append(f"- finite trials: {r.get('n_finite', 0)} / {r.get('n_trials', 0)}")

            # Importance
            imp = r.get("importance", {})
            if imp and "_error" not in imp:
                lines.append(f"\n#### A. fANOVA importance (top 10)\n")
                lines.append("| param | importance |")
                lines.append("|---|---|")
                for k, val in list(imp.items())[:10]:
                    lines.append(f"| `{k}` | {val:.3f} |")

            # Categorical
            cat = r.get("categorical_stats", {})
            sig_rows = []
            for p, info in cat.items():
                if info.get("best") and info["best"].get("significant"):
                    b = info["best"]
                    sig_rows.append(
                        f"| `{p}` | **{b['value']}** | {info['per_value'][b['value']]['mean']:.4f} "
                        f"(n={info['per_value'][b['value']]['n']}) | {b['runner_up']} | "
                        f"Δ +{b['delta_mean']:.4f} | p={b['p_value']:.2e} |"
                    )
            if sig_rows:
                lines.append(f"\n#### B. Categorical: clearly best values (p<0.01)\n")
                lines.append("| param | best | mean RMSE | runner-up | Δ | p |")
                lines.append("|---|---|---|---|---|---|")
                lines.extend(sig_rows)

            # All categorical detail
            if cat:
                lines.append(f"\n<details><summary>All categorical breakdowns</summary>\n")
                for p, info in cat.items():
                    if not info.get("per_value"):
                        continue
                    lines.append(f"\n**`{p}`**")
                    lines.append("| value | n | mean RMSE | std | min |")
                    lines.append("|---|---|---|---|---|")
                    for val, st in sorted(info["per_value"].items(), key=lambda kv: kv[1]["mean"]):
                        lines.append(f"| {val} | {st['n']} | {st['mean']:.4f} | {st['std']:.4f} | {st['min']:.4f} |")
                lines.append("\n</details>\n")

            # Numeric quartiles
            num = r.get("numeric_quartiles", {})
            if num:
                lines.append(f"\n#### D. Numeric: quartile mean RMSE (sweet spot)\n")
                lines.append("| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |")
                lines.append("|---|---|---|---|---|---|")
                for p, info in num.items():
                    if not info or "best_quartile" not in info:
                        continue
                    best_q = info["best_quartile"]
                    rng = info[best_q].get("range", [None, None])
                    cells = [f"{info[q]['mean_rmse']:.4f}" if q in info else "—" for q in ("Q1", "Q2", "Q3", "Q4")]
                    rng_str = f"[{rng[0]}, {rng[1]}]" if rng[1] is not None else f"[{rng[0]}, ∞)"
                    lines.append(f"| `{p}` | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | **{best_q}** {rng_str} |")

            # E. Slice plot reference
            sp = slice_paths.get(f"{run_tag}/{v}")
            if sp:
                lines.append(f"\n#### E. Slice plot\n")
                lines.append(f"![{run_tag}/{v}]({os.path.basename(sp)})\n")

    # Overall recommendations summary
    lines.append("\n---\n\n## Overall recommendations\n")
    moe_recs = []
    for run_tag, ds in results.get("runs", {}).items():
        moe_keys = [v for v in ds.get("_variants", []) if v.startswith("moe")]
        for mk in moe_keys:
            cat = ds.get(mk, {}).get("categorical_stats", {})
            for p, info in cat.items():
                if info.get("best") and info["best"].get("significant"):
                    moe_recs.append((run_tag, p, info["best"]["value"],
                                     info["best"]["delta_mean"], info["best"]["p_value"]))
    if moe_recs:
        lines.append("**Categorical settings that are statistically significant winners (p<0.01):**\n")
        lines.append("| dataset | param | best value | Δ vs runner-up | p |")
        lines.append("|---|---|---|---|---|")
        for ds_name, p, val, d, pv in moe_recs:
            lines.append(f"| {ds_name} | `{p}` | **{val}** | +{d:.4f} | {pv:.2e} |")
    else:
        lines.append("(no categorical settings were universally significant — see per-dataset breakdown)\n")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved → {out_path}")


# =============================================================================
# Holdout evaluation (audit fix — the selection metric is not the headline)
# =============================================================================
# The old protocol reported "best RMSE" = the minimum over N Optuna trials of
# the SAME CV score the optimizer minimized. That is a selection metric, not
# a performance estimate: comparing minima across variants with differently
# sized / differently noisy search spaces rewards whichever space buys more
# lottery tickets (winner's curse). The headline metric is now a
# chronological holdout — the final `holdout_frac` of every series is never
# seen by Optuna; after the search, the best-CV config is retrained once on
# the search region (with the same ES-tail protocol) and scored once on the
# holdout.


def chronological_split(X, y, holdout_frac: float):
    n_hold = int(len(X) * holdout_frac)
    if n_hold < 10:
        raise SystemExit(f"holdout too small ({n_hold} rows) — reduce --holdout-frac")
    cut = len(X) - n_hold
    return X[:cut], y[:cut], X[cut:], y[cut:]


def best_finite_trial(trial_log: list) -> dict | None:
    finite = [t for t in trial_log if np.isfinite(t["rmse"])]
    return min(finite, key=lambda t: t["rmse"]) if finite else None


def holdout_eval(best: dict, X_search, y_search, X_hold, y_hold, cfg) -> dict:
    """Retrain the winning config on the full search region, score the holdout once."""
    try:
        t0 = time.perf_counter()
        if best["variant"] == "naive-ensemble":
            preds = np.zeros(len(X_hold), dtype=np.float64)
            for k in range(best["n_models"]):
                p = dict(best["lgbm_params"])
                p["seed"] = cfg.seed + 1009 * k
                model = _train_with_es(p, X_search, y_search, cfg.num_boost_round)
                preds += model.predict(X_hold)
            preds /= best["n_models"]
            t1 = time.perf_counter()
            pred_s = 0.0  # ensemble latency folded into retrain timing
        else:
            model = _train_with_es(dict(best["lgbm_params"]), X_search, y_search,
                                   cfg.num_boost_round)
            t1 = time.perf_counter()
            preds = model.predict(X_hold)
            pred_s = time.perf_counter() - t1
        return {
            "holdout_rmse": float(np.sqrt(mean_squared_error(y_hold, preds))),
            "retrain_s": round(t1 - t0, 4),
            "predict_s": round(pred_s, 4),
            "cv_rmse_of_winner": best["rmse"],
        }
    except Exception as e:
        return {"holdout_rmse": float("nan"), "error": str(e)[:200]}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_provenance() -> dict:
    """Everything needed to reproduce / attribute a run: build, code, data."""
    prov = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "optuna": optuna.__version__,
        "platform": platform.platform(),
        "lightgbm_moe_package": os.path.dirname(os.path.abspath(lgb.__file__)),
    }
    lib_path = os.path.join(os.path.dirname(os.path.abspath(lgb.__file__)),
                            "lib", "lib_lightgbm.so")
    if os.path.exists(lib_path):
        prov["lib_sha256"] = _sha256(lib_path)
    try:
        prov["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)), text=True).strip()
        prov["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=os.path.dirname(os.path.abspath(__file__)), text=True).strip())
    except Exception:
        pass
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
    if os.path.isdir(cache_dir):
        prov["data_cache"] = {
            f: {"sha256": _sha256(os.path.join(cache_dir, f)),
                "bytes": os.path.getsize(os.path.join(cache_dir, f))}
            for f in sorted(os.listdir(cache_dir)) if f.endswith(".csv")
        }
    return prov


# =============================================================================
# Driver
# =============================================================================
def run_study(dataset_name: str, X_search, y_search, X_hold, y_hold,
              n_trials: int, n_jobs: int, cfg: BenchmarkConfig, slice_dir: str,
              run_tag: str) -> tuple[dict, dict]:
    print(f"\n{'=' * 60}")
    print(f"  Run: {run_tag}  search={X_search.shape}  holdout={len(X_hold)}  "
          f"trials={n_trials}  n_jobs={n_jobs}")
    print(f"{'=' * 60}")

    out = {"dataset": dataset_name, "seed": cfg.seed,
           "X_search_shape": list(X_search.shape), "n_holdout": int(len(X_hold)),
           "y_stats": {"mean": float(y_search.mean()), "std": float(y_search.std())}}
    slice_paths: dict = {}
    X, y = X_search, y_search  # Optuna objectives close over the search region only

    # When refit ablation is requested, the moe variant is renamed to "moe-refit"
    # so two runs (--moe-refit-mode off vs --moe-refit-mode <other>) produce
    # JSONs whose keys distinguish the rows in the merged report. The naive
    # variants are unaffected — so for an ablation it's typical to skip them
    # via --variants moe to save 2/3 of the compute.
    refit_mode = getattr(cfg, "moe_refit_mode", "search")
    refit_decay = getattr(cfg, "moe_refit_decay", 0.0)
    # "search" and "off" both keep the canonical "moe" variant name so the
    # v0.8 study output is directly comparable to the v0.6 README JSON.
    # Forced-trigger ablation modes get a distinguishing suffix.
    moe_variant_name = ("moe" if refit_mode in ("search", "off")
                        else f"moe-refit-{refit_mode}")

    all_variants = [
        ("naive-lightgbm", lambda log: make_naive_lightgbm_objective(X, y, cfg, log)),
        ("naive-ensemble", lambda log: make_naive_ensemble_objective(X, y, cfg, log)),
        (moe_variant_name, lambda log: make_moe_objective(X, y, cfg, log,
                                                          refit_mode=refit_mode,
                                                          refit_decay=refit_decay)),
    ]
    selected_variants = getattr(cfg, "variants", None)
    if selected_variants:
        # Match by prefix so "moe" selects "moe-refit-elbo" too.
        all_variants = [v for v in all_variants
                         if any(v[0] == s or v[0].startswith(s + "-") for s in selected_variants)]
    out["_variants"] = [v[0] for v in all_variants]
    for variant, make_obj in all_variants:
        print(f"\n  → {variant} ({n_trials} trials)...")
        trial_log: list = []
        sampler = optuna.samplers.TPESampler(seed=cfg.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        t0 = time.perf_counter()
        study.optimize(make_obj(trial_log), n_trials=n_trials, n_jobs=n_jobs)
        wall = time.perf_counter() - t0

        agg = aggregate_variant(variant, trial_log, study)
        agg["wall_s"] = round(wall, 1)

        # Holdout: retrain the CV winner once, score the untouched tail once.
        best = best_finite_trial(trial_log)
        if best is not None:
            agg["holdout"] = holdout_eval(best, X_search, y_search, X_hold, y_hold, cfg)
        out[variant] = agg
        out[f"{variant}_trials"] = trial_log

        # Slice plot
        sp_path = os.path.join(slice_dir, f"slice_{run_tag}_{variant}.png")
        if make_slice_plot(study, sp_path, f"{run_tag} / {variant}"):
            slice_paths[f"{run_tag}/{variant}"] = sp_path

        hold = agg.get("holdout", {})
        print(f"    cv-best RMSE = {agg.get('rmse_best', float('nan')):.4f}, "
              f"holdout RMSE = {hold.get('holdout_rmse', float('nan')):.4f}, "
              f"crash rate = {1 - agg.get('n_finite', 0) / max(1, agg.get('n_trials', 1)):.1%}, "
              f"wall = {wall:.0f}s")

    return out, slice_paths


DATASET_GENERATORS = {
    "synthetic": lambda seed: generate_synthetic_data(seed=seed),
    "fred_gdp": lambda seed: generate_fred_gdp_data(seed=seed),
    "sp500_basic": lambda seed: generate_sp500_basic_data(seed=seed),  # ~13 features
    "sp500": lambda seed: generate_sp500_data(seed=seed),              # ~28 features (enriched)
    "vix": lambda seed: generate_vix_data(seed=seed),
    "hmm": lambda seed: generate_hmm_data(seed=seed),
}


def summarize_across_seeds(results: dict) -> dict:
    """Cross-seed summary: per (dataset, variant), holdout RMSE mean ± std.

    The holdout number is the headline; cv_best is retained as the selection
    metric for reference. With a single seed the std fields are 0 and should
    be read as "no variance estimate", not "no variance".
    """
    summary: dict = {}
    for run_tag, run in results.get("runs", {}).items():
        ds = run["dataset"]
        for v in run.get("_variants", []):
            agg = run.get(v, {})
            hold = agg.get("holdout", {})
            entry = summary.setdefault(ds, {}).setdefault(v, {"per_seed": {}})
            entry["per_seed"][str(run["seed"])] = {
                "holdout_rmse": hold.get("holdout_rmse"),
                "cv_best": agg.get("rmse_best"),
                "crash_rate": round(1 - agg.get("n_finite", 0) / max(1, agg.get("n_trials", 1)), 4),
                "retrain_s": hold.get("retrain_s"),
            }
    for ds, variants in summary.items():
        for v, entry in variants.items():
            hs = [s["holdout_rmse"] for s in entry["per_seed"].values()
                  if s["holdout_rmse"] is not None and np.isfinite(s["holdout_rmse"])]
            cs = [s["cv_best"] for s in entry["per_seed"].values() if s["cv_best"] is not None]
            if hs:
                entry["holdout_mean"] = round(float(np.mean(hs)), 6)
                entry["holdout_std"] = round(float(np.std(hs)), 6)
            if cs:
                entry["cv_best_mean"] = round(float(np.mean(cs)), 6)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=500)
    # n_jobs=1 is deliberately the default — see the module docstring for the
    # determinism / speed rationale. tl;dr Optuna n_jobs>1 makes best-RMSE
    # non-reproducible across runs even with seed fixed (TPE observation
    # ordering depends on worker scheduling), AND oversubscribes the CPU
    # against per-trial OMP threading so it's not actually faster.
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--seeds", type=str, default="42",
                   help="comma-separated seeds; each seed redraws synthetic/hmm "
                        "data AND reseeds the TPE sampler + models. Multi-seed "
                        "runs report holdout mean±std — single-seed numbers "
                        "have no variance estimate.")
    p.add_argument("--holdout-frac", type=float, default=0.2,
                   help="chronological tail fraction never seen by Optuna; the "
                        "CV winner is retrained on the rest and scored on it once")
    p.add_argument(
        "--datasets",
        type=str,
        default="synthetic,fred_gdp,sp500_basic,sp500,vix,hmm",
        help=f"Comma-separated subset of: {','.join(DATASET_GENERATORS.keys())}",
    )
    p.add_argument("--out", type=str, required=True)
    # v0.7 leaf-refit + v0.8 partition-regrow handling.
    #
    # "search" (v0.8 default): Optuna samples mixture_refit_leaves /
    #     mixture_refit_trigger / mixture_regrow_oldest_trees / etc.
    #     alongside the rest of the MoE hyperparameters. Used for the v0.8
    #     500-trial study — answers "do v0.8 features rise to the top?"
    # "off": force ``mixture_refit_leaves=false``, no regrow. Reproduces
    #     the v0.6 baseline. Use this to re-run the v0.6 README headline.
    # "always" / "elbo" / "every_n": legacy v0.7 forced-trigger ablation
    #     (the variant key becomes "moe-refit-<mode>" so a baseline run
    #     plus an ablation run can be merged). Regrow is OFF in this path.
    p.add_argument("--moe-refit-mode",
                   choices=["search", "off", "always", "elbo", "every_n"],
                   default="search",
                   help="how to handle the v0.7/v0.8 refit + regrow knobs. "
                        "default 'search' = let Optuna pick (v0.8 study mode)")
    p.add_argument("--moe-refit-decay", type=float, default=0.0,
                   help="leaf blend factor (0=replace, 1=no-op); used when mode != off")
    p.add_argument("--variants", type=str, default=None,
                   help="comma-separated subset of {naive-lightgbm,naive-ensemble,moe}; "
                        "useful with --moe-refit-mode to skip naive baselines on ablation runs")
    args = p.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    selected = [s.strip() for s in args.datasets.split(",")]
    unknown = [s for s in selected if s not in DATASET_GENERATORS]
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {unknown}. Known: {list(DATASET_GENERATORS)}")

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    slice_dir = out_dir

    results = {
        "config": {"trials": args.trials, "n_jobs": args.n_jobs, "rounds": args.rounds,
                   "splits": args.splits, "seeds": seeds, "holdout_frac": args.holdout_frac,
                   "datasets": selected, "es_fraction": ES_FRACTION, "cv_gap": CV_GAP,
                   "moe_refit_mode": args.moe_refit_mode},
        "provenance": collect_provenance(),
        "runs": {},
    }
    all_slice_paths: dict = {}

    for seed in seeds:
        cfg = BenchmarkConfig(n_trials=args.trials, seed=seed, n_splits=args.splits,
                              num_boost_round=args.rounds)
        cfg.moe_refit_mode = args.moe_refit_mode
        cfg.moe_refit_decay = args.moe_refit_decay
        cfg.variants = [v.strip() for v in args.variants.split(",")] if args.variants else None
        for ds_name in selected:
            X, y, _ = DATASET_GENERATORS[ds_name](seed)
            X_search, y_search, X_hold, y_hold = chronological_split(X, y, args.holdout_frac)
            run_tag = f"{ds_name}@s{seed}"
            ds_out, sp = run_study(ds_name, X_search, y_search, X_hold, y_hold,
                                   args.trials, args.n_jobs, cfg, slice_dir, run_tag)
            results["runs"][run_tag] = ds_out
            all_slice_paths.update(sp)

    results["summary"] = summarize_across_seeds(results)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved JSON → {args.out}")

    md_path = args.out.replace(".json", "_report.md")
    render_markdown(results, md_path, all_slice_paths)


if __name__ == "__main__":
    main()
