#!/usr/bin/env python
# coding: utf-8
"""
Benchmark: Gate Type Comparison (gbdt vs none vs leaf_reuse)

3つのgate typeをOptunaで公平に比較。
gate_type以外のMoEパラメータはOptuna探索。

Usage:
    python examples/benchmark_gate.py                    # Full (50 trials)
    python examples/benchmark_gate.py --trials 10        # Quick test
"""

import argparse
import time
import warnings
from dataclasses import dataclass

import numpy as np
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

import lightgbm_moe as lgb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# Config
# =============================================================================
@dataclass
class BenchConfig:
    n_trials: int = 50
    n_splits: int = 5
    num_boost_round: int = 100
    seed: int = 42
    num_threads: int = 4
    n_jobs: int = 1


# =============================================================================
# Data Generation (from benchmark_2.py)
# =============================================================================
def generate_synthetic(n_samples=2000, seed=42):
    """Regime determinable from X (MoE ideal case)."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, 5)
    regime = (0.5 * X[:, 1] + 0.3 * X[:, 2] - 0.2 * X[:, 3] > 0).astype(int)
    y = np.zeros(n_samples)
    m0 = regime == 0
    y[m0] = 5 * X[m0, 0] + 3 * X[m0, 0] * X[m0, 2] + 2 * np.sin(2 * X[m0, 3]) + 10
    m1 = regime == 1
    y[m1] = -5 * X[m1, 0] - 2 * X[m1, 1] ** 2 + 3 * np.cos(2 * X[m1, 4]) - 10
    y += np.random.randn(n_samples) * 0.5
    return X, y, regime


def generate_hamilton(n_samples=500, seed=42):
    """Latent regime (time-varying)."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, 4)
    t = np.arange(n_samples)
    regime = (np.random.rand(n_samples) < 0.5 + 0.3 * np.sin(2 * np.pi * t / 100)).astype(int)
    y = np.zeros(n_samples)
    m0 = regime == 0
    y[m0] = 0.8 + 0.3 * X[m0, 0] + 0.2 * X[m0, 1]
    m1 = regime == 1
    y[m1] = -0.5 + 0.1 * X[m1, 0] - 0.3 * X[m1, 2]
    y += np.random.randn(n_samples) * 0.3
    return X, y, regime


DATASETS = {
    "Synthetic": {"generator": generate_synthetic, "params": {"n_samples": 2000}},
    "Hamilton": {"generator": generate_hamilton, "params": {"n_samples": 500}},
}


# =============================================================================
# CV Evaluation
# =============================================================================
def evaluate_cv(X, y, params, cfg: BenchConfig):
    tscv = TimeSeriesSplit(n_splits=cfg.n_splits)
    scores = []
    for tr, va in tscv.split(X):
        ds_tr = lgb.Dataset(X[tr], label=y[tr], free_raw_data=False)
        ds_va = lgb.Dataset(X[va], label=y[va], reference=ds_tr, free_raw_data=False)
        try:
            bst = lgb.train(
                params, ds_tr, num_boost_round=cfg.num_boost_round,
                valid_sets=[ds_va],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            pred = bst.predict(X[va])
            scores.append(np.sqrt(mean_squared_error(y[va], pred)))
        except Exception:
            scores.append(float("inf"))
    return np.mean(scores)


# =============================================================================
# Optuna Objectives
# =============================================================================
def create_objective_standard(X, y, cfg):
    def objective(trial):
        params = {
            "objective": "regression", "boosting": "gbdt", "verbose": -1,
            "num_threads": cfg.num_threads, "seed": cfg.seed,
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
        }
        return evaluate_cv(X, y, params, cfg)
    return objective


def create_objective_moe(X, y, cfg, gate_type):
    """Create MoE objective with fixed gate_type."""
    def objective(trial):
        params = {
            "objective": "regression", "verbose": -1,
            "num_threads": cfg.num_threads, "seed": cfg.seed,
            # Tree params
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
            # MoE params
            "boosting": "mixture",
            "mixture_num_experts": trial.suggest_int("mixture_num_experts", 2, 4),
            "mixture_e_step_alpha": trial.suggest_float("mixture_e_step_alpha", 0.1, 2.0),
            "mixture_warmup_iters": trial.suggest_int("mixture_warmup_iters", 5, 50),
            "mixture_hard_m_step": True,
            "mixture_r_smoothing": "none",
            "mixture_init": trial.suggest_categorical(
                "mixture_init", ["uniform", "quantile", "balanced_kmeans"]),
            # Gate config (fixed per study)
            "mixture_gate_type": gate_type,
            "mixture_gate_max_depth": trial.suggest_int("mixture_gate_max_depth", 2, 6),
            "mixture_gate_num_leaves": trial.suggest_int("mixture_gate_num_leaves", 4, 32),
            "mixture_gate_learning_rate": trial.suggest_float(
                "mixture_gate_learning_rate", 0.01, 0.3, log=True),
        }
        return evaluate_cv(X, y, params, cfg)
    return objective


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Gate type benchmark")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4,
                        help="OMP threads per LightGBM call")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Optuna parallel trials (uses Python threads)")
    args = parser.parse_args()

    cfg = BenchConfig(n_trials=args.trials, seed=args.seed,
                      num_boost_round=args.rounds, n_splits=args.splits,
                      num_threads=args.threads, n_jobs=args.n_jobs)

    print("=" * 72)
    print("  Gate Type Benchmark: gbdt vs none vs leaf_reuse")
    print(f"  Optuna trials={cfg.n_trials}, CV splits={cfg.n_splits}, "
          f"rounds={cfg.num_boost_round}")
    print("=" * 72)

    all_results = {}

    for ds_name, ds_info in DATASETS.items():
        gen = ds_info["generator"]
        X, y, regime = gen(**ds_info["params"], seed=cfg.seed)
        print(f"\n{'─' * 72}")
        print(f"  Dataset: {ds_name}  ({X.shape[0]} samples, {X.shape[1]} features)")
        print(f"{'─' * 72}")

        results = {}

        # Standard GBDT baseline
        t0 = time.perf_counter()
        study = optuna.create_study(direction="minimize",
                                     sampler=optuna.samplers.TPESampler(seed=cfg.seed))
        study.optimize(create_objective_standard(X, y, cfg),
                       n_trials=cfg.n_trials, n_jobs=cfg.n_jobs)
        elapsed = time.perf_counter() - t0
        results["Standard"] = {
            "rmse": study.best_value, "time": elapsed, "params": study.best_params}
        print(f"  Standard GBDT:    RMSE={study.best_value:.4f}  ({elapsed:.1f}s)")

        # MoE with each gate type
        for gate_type in ["gbdt", "none", "leaf_reuse"]:
            t0 = time.perf_counter()
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=cfg.seed))
            study.optimize(
                create_objective_moe(X, y, cfg, gate_type),
                n_trials=cfg.n_trials, n_jobs=cfg.n_jobs)
            elapsed = time.perf_counter() - t0
            results[f"MoE_{gate_type}"] = {
                "rmse": study.best_value, "time": elapsed, "params": study.best_params}
            print(f"  MoE gate={gate_type:<12s} RMSE={study.best_value:.4f}  ({elapsed:.1f}s)")

        all_results[ds_name] = results

    # Summary table
    print(f"\n{'=' * 72}")
    print("  Summary")
    print(f"{'=' * 72}")
    print(f"  {'Dataset':<12s}  {'Method':<20s}  {'RMSE':>8s}  {'Time':>8s}  {'vs Std':>8s}")
    print(f"  {'─' * 12}  {'─' * 20}  {'─' * 8}  {'─' * 8}  {'─' * 8}")
    for ds_name, results in all_results.items():
        std_rmse = results["Standard"]["rmse"]
        for method, res in results.items():
            improvement = (std_rmse - res["rmse"]) / std_rmse * 100
            imp_str = f"{improvement:+.1f}%" if method != "Standard" else "—"
            print(f"  {ds_name:<12s}  {method:<20s}  {res['rmse']:8.4f}  "
                  f"{res['time']:7.1f}s  {imp_str:>8s}")
        print()


if __name__ == "__main__":
    main()
