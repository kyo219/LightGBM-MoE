#!/usr/bin/env python
# coding: utf-8
"""
comparative_study.py — Standard GBDT vs MoE vs MoE-PE 大規模比較

`benchmark.py` の data generators (Synthetic / Hamilton) と evaluate_cv を再利用しつつ、
3バリアント × 2データセットで Optuna を回し、

  - どのバリアントが最も精度が良いか (best RMSE)
  - どのバリアントが速いか (per-trial train time の中央値)
  - 各バリアントで推奨される設定 (gate_type, routing_mode, e_step_mode など
    の上位 trials 分布から経験則を抽出)

を一気に得る用途。

注:
  - mixture_init は {random, gmm, tree_hierarchical} に限定 (uniform/quantile/balanced_kmeans 除外)
  - mixture_gate_type は {gbdt, none, leaf_reuse} を全探索
  - 全 trial の時間とパラメータを JSON に保存し、後で再分析可能

Usage:
    # 小規模 sanity check
    python examples/comparative_study.py --trials 20 --out bench_results/study_smoke.json

    # 本番 1000 trials × 3 variants × 2 datasets
    python examples/comparative_study.py --trials 1000 --out bench_results/study_1k.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python-package"))
import lightgbm_moe as lgb

# Reuse data generators and CV evaluator from benchmark.py
sys.path.insert(0, os.path.dirname(__file__))
from benchmark import (  # noqa: E402
    BenchmarkConfig,
    generate_hamilton_gnp_data,
    generate_synthetic_data,
    generate_vix_data,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# Init choices restricted per user request
# =============================================================================
INIT_CHOICES = ["random", "gmm", "tree_hierarchical"]
GATE_CHOICES = ["gbdt", "none", "leaf_reuse"]
ROUTING_CHOICES = ["token_choice", "expert_choice"]
E_STEP_MODES = ["em", "loss_only", "gate_only"]
SMOOTHING_CHOICES = ["none", "ema", "markov"]


# =============================================================================
# CV with per-trial timing capture
# =============================================================================
def evaluate_cv_timed(X, y, params, n_splits: int, num_boost_round: int):
    """5-fold time-series CV. Returns (rmse_mean, mean_train_seconds_per_fold)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses = []
    times = []
    for tr_idx, va_idx in tscv.split(X):
        Xt, Xv = X[tr_idx], X[va_idx]
        yt, yv = y[tr_idx], y[va_idx]
        train = lgb.Dataset(Xt, label=yt)
        valid = lgb.Dataset(Xv, label=yv, reference=train)
        try:
            t0 = time.perf_counter()
            model = lgb.train(
                params,
                train,
                num_boost_round=num_boost_round,
                valid_sets=[valid],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )
            t1 = time.perf_counter()
            pred = model.predict(Xv)
            rmse = float(np.sqrt(mean_squared_error(yv, pred)))
            rmses.append(rmse)
            times.append(t1 - t0)
        except Exception:
            rmses.append(float("inf"))
            times.append(0.0)
    return float(np.mean(rmses)), float(np.mean(times))


# =============================================================================
# Per-variant Optuna objectives
# =============================================================================
def make_standard_objective(X, y, cfg: BenchmarkConfig, trial_log: list):
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
        rmse, train_s = evaluate_cv_timed(X, y, params, cfg.n_splits, cfg.num_boost_round)
        trial_log.append({"variant": "standard", "rmse": rmse, "train_s": train_s, "params": dict(trial.params)})
        return rmse

    return objective


def _moe_common_params(trial, cfg: BenchmarkConfig, num_experts: int):
    """Shared search space across MoE variants. Returns base param dict."""
    smoothing = trial.suggest_categorical("mixture_r_smoothing", SMOOTHING_CHOICES)
    routing_mode = trial.suggest_categorical("mixture_routing_mode", ROUTING_CHOICES)
    gate_type = trial.suggest_categorical("mixture_gate_type", GATE_CHOICES)

    params = {
        "objective": "regression",
        "boosting": "mixture",
        "verbose": -1,
        "num_threads": 4,
        "seed": cfg.seed,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
        "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
        # Init restricted to {random, gmm, tree_hierarchical} per user request
        "mixture_init": trial.suggest_categorical("mixture_init", INIT_CHOICES),
        "mixture_num_experts": num_experts,
        "mixture_e_step_alpha": trial.suggest_float("mixture_e_step_alpha", 0.1, 3.0),
        "mixture_e_step_mode": trial.suggest_categorical("mixture_e_step_mode", E_STEP_MODES),
        "mixture_warmup_iters": trial.suggest_int("mixture_warmup_iters", 5, 50),
        "mixture_balance_factor": trial.suggest_int("mixture_balance_factor", 2, 10),
        "mixture_r_smoothing": smoothing,
        "mixture_smoothing_lambda": trial.suggest_float("mixture_smoothing_lambda", 0.0, 0.9) if smoothing != "none" else 0.0,
        "mixture_diversity_lambda": trial.suggest_float("mixture_diversity_lambda", 0.0, 0.5),
        "mixture_routing_mode": routing_mode,
        "mixture_hard_m_step": trial.suggest_categorical("mixture_hard_m_step", [True, False]),
        # Gate type and gate-specific parameters
        "mixture_gate_type": gate_type,
    }

    # Gate parameters apply only when there's actually a gate model
    if gate_type in ("gbdt", "leaf_reuse"):
        params["mixture_gate_max_depth"] = trial.suggest_int("mixture_gate_max_depth", 2, 10)
        params["mixture_gate_num_leaves"] = trial.suggest_int("mixture_gate_num_leaves", 4, 64)
        params["mixture_gate_learning_rate"] = trial.suggest_float("mixture_gate_learning_rate", 0.01, 0.5, log=True)
        params["mixture_gate_lambda_l2"] = trial.suggest_float("mixture_gate_lambda_l2", 1e-3, 10.0, log=True)
        params["mixture_gate_iters_per_round"] = trial.suggest_int("mixture_gate_iters_per_round", 1, 3)
    if gate_type == "leaf_reuse":
        params["mixture_gate_retrain_interval"] = trial.suggest_int("mixture_gate_retrain_interval", 5, 30)

    # Expert Choice routing-specific params
    if routing_mode == "expert_choice":
        params["mixture_expert_capacity_factor"] = trial.suggest_float("mixture_expert_capacity_factor", 0.8, 1.5)
        params["mixture_expert_choice_score"] = "combined"
        params["mixture_expert_choice_boost"] = trial.suggest_float("mixture_expert_choice_boost", 5.0, 30.0)
        params["mixture_expert_choice_hard"] = trial.suggest_categorical("mixture_expert_choice_hard", [True, False])

    return params


def make_moe_objective(X, y, cfg: BenchmarkConfig, trial_log: list, per_expert: bool):
    variant_name = "moe-pe" if per_expert else "moe"

    def objective(trial):
        num_experts = trial.suggest_int("mixture_num_experts", 2, 4)
        params = _moe_common_params(trial, cfg, num_experts)

        if per_expert:
            max_depths = [trial.suggest_int(f"max_depth_{k}", 3, 12) for k in range(num_experts)]
            num_leaves = [trial.suggest_int(f"num_leaves_{k}", 8, 128) for k in range(num_experts)]
            min_data = [trial.suggest_int(f"min_data_in_leaf_{k}", 5, 100) for k in range(num_experts)]
            params["mixture_expert_max_depths"] = ",".join(map(str, max_depths))
            params["mixture_expert_num_leaves"] = ",".join(map(str, num_leaves))
            params["mixture_expert_min_data_in_leaf"] = ",".join(map(str, min_data))
        else:
            params["num_leaves"] = trial.suggest_int("num_leaves", 8, 128)
            params["max_depth"] = trial.suggest_int("max_depth", 3, 12)
            params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 5, 100)

        rmse, train_s = evaluate_cv_timed(X, y, params, cfg.n_splits, cfg.num_boost_round)
        trial_log.append({"variant": variant_name, "rmse": rmse, "train_s": train_s, "params": dict(trial.params)})
        return rmse

    return objective


# =============================================================================
# Per-variant aggregator: best RMSE, time stats, top-trial categorical distributions
# =============================================================================
def categorical_top_distribution(trials: list[dict], param: str, top_pct: int = 10) -> dict:
    """Among the top-N% trials by RMSE, what fraction picked each value of `param`?"""
    finite = [t for t in trials if np.isfinite(t["rmse"])]
    if not finite:
        return {}
    finite.sort(key=lambda t: t["rmse"])
    cutoff = max(1, len(finite) * top_pct // 100)
    top = finite[:cutoff]
    cnt = Counter(t["params"].get(param) for t in top if param in t["params"])
    n = sum(cnt.values())
    if n == 0:
        return {}
    return {str(k): round(v / n, 3) for k, v in cnt.most_common()}


def numeric_top_stats(trials: list[dict], param: str, top_pct: int = 10) -> dict:
    """Median / IQR of a numeric param among the top trials."""
    finite = [t for t in trials if np.isfinite(t["rmse"]) and param in t["params"]]
    if not finite:
        return {}
    finite.sort(key=lambda t: t["rmse"])
    cutoff = max(1, len(finite) * top_pct // 100)
    vals = np.array([t["params"][param] for t in finite[:cutoff]], dtype=float)
    return {
        "median": round(float(np.median(vals)), 4),
        "q25": round(float(np.quantile(vals, 0.25)), 4),
        "q75": round(float(np.quantile(vals, 0.75)), 4),
    }


def aggregate_variant(name: str, trials: list[dict]) -> dict:
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
    }

    # For MoE variants, summarize categorical knobs in top 10% trials
    if name in ("moe", "moe-pe"):
        cat_params = [
            "mixture_gate_type",
            "mixture_routing_mode",
            "mixture_e_step_mode",
            "mixture_init",
            "mixture_r_smoothing",
            "mixture_hard_m_step",
            "extra_trees",
        ]
        out["top10pct_categorical"] = {p: categorical_top_distribution(trials, p) for p in cat_params}

        num_params = [
            "mixture_num_experts",
            "mixture_e_step_alpha",
            "mixture_diversity_lambda",
            "mixture_warmup_iters",
            "mixture_balance_factor",
            "learning_rate",
        ]
        out["top10pct_numeric"] = {p: numeric_top_stats(trials, p) for p in num_params}

    return out


# =============================================================================
# Main
# =============================================================================
def run_study(dataset_name: str, X, y, n_trials: int, n_jobs: int, cfg: BenchmarkConfig) -> dict:
    print(f"\n{'=' * 60}")
    print(f"  Dataset: {dataset_name}  (X={X.shape}, n_trials={n_trials}, n_jobs={n_jobs})")
    print(f"{'=' * 60}")

    out = {"dataset": dataset_name, "X_shape": list(X.shape), "y_stats": {"mean": float(y.mean()), "std": float(y.std())}}

    for variant, make_obj in [
        ("standard", lambda log: make_standard_objective(X, y, cfg, log)),
        ("moe", lambda log: make_moe_objective(X, y, cfg, log, per_expert=False)),
        ("moe-pe", lambda log: make_moe_objective(X, y, cfg, log, per_expert=True)),
    ]:
        print(f"\n  → {variant} ({n_trials} trials)...")
        trial_log: list[dict] = []
        sampler = optuna.samplers.TPESampler(seed=cfg.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        t0 = time.perf_counter()
        study.optimize(make_obj(trial_log), n_trials=n_trials, n_jobs=n_jobs)
        wall = time.perf_counter() - t0

        agg = aggregate_variant(variant, trial_log)
        agg["wall_s"] = round(wall, 1)
        agg["best_params"] = study.best_params
        out[variant] = agg
        out[f"{variant}_trials"] = trial_log
        print(f"    best RMSE = {agg.get('rmse_best', float('nan')):.4f}, "
              f"median train = {agg.get('train_s_median', 0):.2f}s/fold, "
              f"wall = {wall:.1f}s")

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--n-jobs", type=int, default=6)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--datasets", type=str, default="synthetic,hamilton,vix",
                   help="Comma-separated subset of: synthetic,hamilton,vix")
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    cfg = BenchmarkConfig(
        n_trials=args.trials,
        seed=args.seed,
        n_splits=args.splits,
        num_boost_round=args.rounds,
    )

    selected = [s.strip() for s in args.datasets.split(",")]
    results = {"config": {"trials": args.trials, "n_jobs": args.n_jobs, "rounds": args.rounds, "splits": args.splits, "seed": args.seed}}

    if "synthetic" in selected:
        X_syn, y_syn, _regime_syn = generate_synthetic_data(seed=cfg.seed)
        results["synthetic"] = run_study("synthetic", X_syn, y_syn, args.trials, args.n_jobs, cfg)

    if "hamilton" in selected:
        X_ham, y_ham, _regime_ham = generate_hamilton_gnp_data(seed=cfg.seed)
        results["hamilton"] = run_study("hamilton", X_ham, y_ham, args.trials, args.n_jobs, cfg)

    if "vix" in selected:
        X_vix, y_vix, _regime_vix = generate_vix_data(seed=cfg.seed)
        results["vix"] = run_study("vix", X_vix, y_vix, args.trials, args.n_jobs, cfg)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
