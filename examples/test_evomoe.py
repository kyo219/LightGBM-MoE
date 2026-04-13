#!/usr/bin/env python
# coding: utf-8
"""
EvoMoE Progressive Training + Gate Temperature Annealing 効果検証

既存MoE vs EvoMoE vs Temperature Annealing vs Both の4パターン比較。
Optuna で各パターンの最適パラメータを探索し、公平に比較する。

Usage:
    python examples/test_evomoe.py              # 20 trials (default)
    python examples/test_evomoe.py --trials 50  # 50 trials
"""

import argparse
import time
import warnings

import numpy as np
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

import lightgbm_moe as lgb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# Data Generation (same as benchmark.py)
# =============================================================================
def generate_synthetic_data(n_samples=2000, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n_samples, 5)
    regime_score = 0.5 * X[:, 1] + 0.3 * X[:, 2] - 0.2 * X[:, 3]
    regime_true = (regime_score > 0).astype(int)
    y = np.zeros(n_samples)
    mask0 = regime_true == 0
    y[mask0] = 5.0 * X[mask0, 0] + 3.0 * X[mask0, 0] * X[mask0, 2] + 2.0 * np.sin(2 * X[mask0, 3]) + 10.0
    mask1 = regime_true == 1
    y[mask1] = -5.0 * X[mask1, 0] - 2.0 * X[mask1, 1] ** 2 + 3.0 * np.cos(2 * X[mask1, 4]) - 10.0
    y += np.random.randn(n_samples) * 0.5
    return X, y, regime_true


def generate_hamilton_data(n_samples=500, seed=42):
    np.random.seed(seed)
    X_base = np.random.randn(n_samples, 4)
    t = np.arange(n_samples)
    regime_prob = 0.5 + 0.3 * np.sin(2 * np.pi * t / 100)
    regime_true = (np.random.rand(n_samples) < regime_prob).astype(int)
    y = np.zeros(n_samples)
    mask0 = regime_true == 0
    y[mask0] = 0.8 + 0.3 * X_base[mask0, 0] + 0.2 * X_base[mask0, 1]
    mask1 = regime_true == 1
    y[mask1] = -0.5 + 0.1 * X_base[mask1, 0] - 0.3 * X_base[mask1, 2]
    y += np.random.randn(n_samples) * 0.3

    # Time-series features
    ts_features = []
    for window in [5, 10, 20]:
        ma = np.zeros(n_samples)
        for i in range(1, n_samples):
            start = max(0, i - window)
            ma[i] = np.mean(y[start:i])
        ts_features.append(ma)
    for window in [5, 10]:
        vol = np.zeros(n_samples)
        for i in range(2, n_samples):
            start = max(0, i - window)
            vol[i] = np.std(y[start:i])
        ts_features.append(vol)
    ts_features.append(ts_features[0] - ts_features[2])  # MA diff
    ts_features.append(np.sign(ts_features[0]))  # Sign of MA5
    frac_pos = np.zeros(n_samples)
    for i in range(1, n_samples):
        start = max(0, i - 10)
        frac_pos[i] = np.mean(y[start:i] > 0)
    ts_features.append(frac_pos)

    X = np.column_stack([X_base] + ts_features)
    return X, y, regime_true


# =============================================================================
# CV Evaluation
# =============================================================================
def evaluate_cv(X, y, params, n_splits=5, num_boost_round=100, seed=42):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        try:
            model = lgb.train(
                params, train_data,
                num_boost_round=num_boost_round,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )
            pred = model.predict(X_val)
            scores.append(np.sqrt(mean_squared_error(y_val, pred)))
        except Exception:
            scores.append(float("inf"))
    return np.mean(scores)


# =============================================================================
# Optuna Objectives
# =============================================================================
def _common_moe_params(trial, seed):
    """Shared MoE params across all variants."""
    return {
        "objective": "regression",
        "boosting": "mixture",
        "verbose": -1,
        "num_threads": 4,
        "seed": seed,
        "mixture_num_experts": trial.suggest_int("mixture_num_experts", 2, 4),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
        "mixture_e_step_alpha": trial.suggest_float("mixture_e_step_alpha", 0.1, 3.0),
        "mixture_warmup_iters": trial.suggest_int("mixture_warmup_iters", 5, 50),
        "mixture_r_smoothing": "none",
        "mixture_routing_mode": "expert_choice",
        "mixture_expert_capacity_factor": trial.suggest_float("mixture_expert_capacity_factor", 0.8, 1.5),
        "mixture_expert_choice_score": "combined",
        "mixture_expert_choice_boost": trial.suggest_float("mixture_expert_choice_boost", 5.0, 30.0),
        "mixture_diversity_lambda": trial.suggest_float("mixture_diversity_lambda", 0.0, 0.5),
        "mixture_gate_max_depth": trial.suggest_int("mixture_gate_max_depth", 2, 10),
        "mixture_gate_num_leaves": trial.suggest_int("mixture_gate_num_leaves", 4, 64),
        "mixture_gate_learning_rate": trial.suggest_float("mixture_gate_learning_rate", 0.01, 0.5, log=True),
        "mixture_gate_lambda_l2": trial.suggest_float("mixture_gate_lambda_l2", 1e-3, 10.0, log=True),
    }


def create_objective_baseline(X, y, seed, num_boost_round):
    """Baseline MoE (no progressive, no temperature annealing)."""
    def objective(trial):
        params = _common_moe_params(trial, seed)
        return evaluate_cv(X, y, params, num_boost_round=num_boost_round, seed=seed)
    return objective


def create_objective_evomoe(X, y, seed, num_boost_round):
    """EvoMoE progressive training."""
    def objective(trial):
        params = _common_moe_params(trial, seed)
        params["mixture_progressive_mode"] = "evomoe"
        params["mixture_seed_iterations"] = trial.suggest_int("mixture_seed_iterations", 10, 60)
        params["mixture_spawn_perturbation"] = trial.suggest_float("mixture_spawn_perturbation", 0.1, 0.9)
        return evaluate_cv(X, y, params, num_boost_round=num_boost_round, seed=seed)
    return objective


def create_objective_temperature(X, y, seed, num_boost_round):
    """Temperature annealing only."""
    def objective(trial):
        params = _common_moe_params(trial, seed)
        params["mixture_gate_temperature_init"] = trial.suggest_float("mixture_gate_temperature_init", 1.0, 5.0)
        params["mixture_gate_temperature_final"] = trial.suggest_float("mixture_gate_temperature_final", 0.1, 1.0)
        return evaluate_cv(X, y, params, num_boost_round=num_boost_round, seed=seed)
    return objective


def create_objective_both(X, y, seed, num_boost_round):
    """EvoMoE + Temperature annealing."""
    def objective(trial):
        params = _common_moe_params(trial, seed)
        params["mixture_progressive_mode"] = "evomoe"
        params["mixture_seed_iterations"] = trial.suggest_int("mixture_seed_iterations", 10, 60)
        params["mixture_spawn_perturbation"] = trial.suggest_float("mixture_spawn_perturbation", 0.1, 0.9)
        params["mixture_gate_temperature_init"] = trial.suggest_float("mixture_gate_temperature_init", 1.0, 5.0)
        params["mixture_gate_temperature_final"] = trial.suggest_float("mixture_gate_temperature_final", 0.1, 1.0)
        return evaluate_cv(X, y, params, num_boost_round=num_boost_round, seed=seed)
    return objective


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="EvoMoE effect verification")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=100)
    args = parser.parse_args()

    n_trials = args.trials
    seed = args.seed
    num_boost_round = args.rounds

    DATASETS = {
        "Synthetic": generate_synthetic_data(n_samples=2000, seed=seed),
        "Hamilton":  generate_hamilton_data(n_samples=500, seed=seed),
    }

    VARIANTS = {
        "Baseline MoE":   create_objective_baseline,
        "EvoMoE":         create_objective_evomoe,
        "TempAnneal":     create_objective_temperature,
        "EvoMoE+Temp":    create_objective_both,
    }

    print("=" * 80)
    print("EvoMoE Progressive Training + Gate Temperature Annealing 効果検証")
    print("=" * 80)
    print(f"Trials: {n_trials}, Boost Rounds: {num_boost_round}, Seed: {seed}")
    print()

    all_results = {}

    for ds_name, (X, y, regime) in DATASETS.items():
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds_name}  (N={len(y)}, F={X.shape[1]})")
        print(f"{'=' * 60}")

        ds_results = {}

        for var_name, obj_factory in VARIANTS.items():
            print(f"\n  [{var_name}] ({n_trials} trials)...", end="", flush=True)
            start = time.time()
            study = optuna.create_study(direction="minimize")
            study.optimize(
                obj_factory(X, y, seed, num_boost_round),
                n_trials=n_trials,
                show_progress_bar=False,
            )
            elapsed = time.time() - start
            ds_results[var_name] = {
                "rmse": study.best_value,
                "params": study.best_params,
                "time": elapsed,
            }
            print(f"  RMSE={study.best_value:.4f}  ({elapsed:.1f}s)")

        all_results[ds_name] = ds_results

    # Summary
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    header = f"{'Dataset':<12}"
    for var_name in VARIANTS:
        header += f" {var_name:>14}"
    header += f" {'Best Impr':>12}"
    print(header)
    print("-" * 80)

    for ds_name, ds_results in all_results.items():
        row = f"{ds_name:<12}"
        baseline_rmse = ds_results["Baseline MoE"]["rmse"]
        best_new_rmse = baseline_rmse

        for var_name in VARIANTS:
            rmse = ds_results[var_name]["rmse"]
            if var_name != "Baseline MoE":
                best_new_rmse = min(best_new_rmse, rmse)
            # Bold the best
            row += f" {rmse:>14.4f}"
        improvement = (baseline_rmse - best_new_rmse) / baseline_rmse * 100
        row += f" {improvement:>+11.1f}%"
        print(row)

    print("-" * 80)
    print("Baseline MoE = 既存 (no progressive, no temp annealing)")
    print("EvoMoE = progressive training only")
    print("TempAnneal = temperature annealing only")
    print("EvoMoE+Temp = both combined")
    print("Best Impr = best new variant vs baseline")
    print("=" * 80)

    # Print best params for new features
    for ds_name, ds_results in all_results.items():
        print(f"\n[{ds_name}] Best new feature params:")
        for var_name in ["EvoMoE", "TempAnneal", "EvoMoE+Temp"]:
            res = ds_results[var_name]
            p = res["params"]
            extra = []
            if "mixture_seed_iterations" in p:
                extra.append(f"seed_iters={p['mixture_seed_iterations']}")
            if "mixture_spawn_perturbation" in p:
                extra.append(f"perturb={p['mixture_spawn_perturbation']:.2f}")
            if "mixture_gate_temperature_init" in p:
                extra.append(f"temp_init={p['mixture_gate_temperature_init']:.2f}")
            if "mixture_gate_temperature_final" in p:
                extra.append(f"temp_final={p['mixture_gate_temperature_final']:.2f}")
            print(f"  {var_name:<14} RMSE={res['rmse']:.4f}  {', '.join(extra)}")

    print("\nDone!")


if __name__ == "__main__":
    main()
