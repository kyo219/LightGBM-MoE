#!/usr/bin/env python
# coding: utf-8
"""
Benchmark: Standard GBDT vs MoE GBDT

3つのデータセットで Standard GBDT と MoE GBDT を比較。
MoEは全パラメータ（smoothing含む）をOptunaで最適化。

Usage:
    python examples/benchmark.py
    python examples/benchmark.py --trials 50
    python examples/benchmark.py --trials 30 --seed 123
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
# Configuration
# =============================================================================
@dataclass
class BenchmarkConfig:
    n_trials: int = 50
    n_splits: int = 5
    num_boost_round: int = 100
    seed: int = 42


# =============================================================================
# Data Generation
# =============================================================================
def generate_synthetic_data(
    n_samples: int = 2000, noise_level: float = 0.5, seed: int = 42
):
    """
    Synthetic regime-switching data (regime determinable from X).
    MoEが得意とするケース：特徴量からregimeが決まる。
    """
    np.random.seed(seed)
    n_features = 5
    X = np.random.randn(n_samples, n_features)

    # Regime is determined by features
    regime_score = 0.5 * X[:, 1] + 0.3 * X[:, 2] - 0.2 * X[:, 3]
    regime_true = (regime_score > 0).astype(int)

    y = np.zeros(n_samples)

    # Regime 0: positive, nonlinear
    mask0 = regime_true == 0
    y[mask0] = (
        5.0 * X[mask0, 0]
        + 3.0 * X[mask0, 0] * X[mask0, 2]
        + 2.0 * np.sin(2 * X[mask0, 3])
        + 10.0
    )

    # Regime 1: negative, different nonlinearity
    mask1 = regime_true == 1
    y[mask1] = (
        -5.0 * X[mask1, 0]
        - 2.0 * X[mask1, 1] ** 2
        + 3.0 * np.cos(2 * X[mask1, 4])
        - 10.0
    )

    y += np.random.randn(n_samples) * noise_level

    return X, y, regime_true


def generate_hamilton_gnp_data(n_samples: int = 500, seed: int = 42):
    """
    Hamilton GNP-like regime-switching data (latent regime).
    時系列的なregimeスイッチング。
    """
    np.random.seed(seed)
    n_features = 4
    X = np.random.randn(n_samples, n_features)

    # Regime is latent (time-based probability)
    t = np.arange(n_samples)
    regime_prob = 0.5 + 0.3 * np.sin(2 * np.pi * t / 100)
    regime_true = (np.random.rand(n_samples) < regime_prob).astype(int)

    y = np.zeros(n_samples)

    # Expansion regime
    mask0 = regime_true == 0
    y[mask0] = 0.8 + 0.3 * X[mask0, 0] + 0.2 * X[mask0, 1]

    # Recession regime
    mask1 = regime_true == 1
    y[mask1] = -0.5 + 0.1 * X[mask1, 0] - 0.3 * X[mask1, 2]

    y += np.random.randn(n_samples) * 0.3

    return X, y, regime_true


def generate_vix_data(n_samples: int = 1000, seed: int = 42):
    """
    VIX-like volatility regime data.
    ボラティリティの高低でregimeが変わる。
    """
    np.random.seed(seed)
    n_features = 5
    X = np.random.randn(n_samples, n_features)

    # Volatility regime (high/low)
    t = np.arange(n_samples)
    regime_prob = 0.3 + 0.4 * (np.sin(2 * np.pi * t / 200) > 0)
    regime_true = (np.random.rand(n_samples) < regime_prob).astype(int)

    y = np.zeros(n_samples)

    # Low volatility regime
    mask0 = regime_true == 0
    y[mask0] = 0.01 + 0.002 * np.abs(X[mask0, 0])

    # High volatility regime
    mask1 = regime_true == 1
    y[mask1] = 0.025 + 0.005 * np.abs(X[mask1, 0]) + 0.003 * X[mask1, 1] ** 2

    y += np.random.randn(n_samples) * 0.005

    return X, y, regime_true


DATASETS = {
    "Synthetic": {
        "generator": generate_synthetic_data,
        "params": {"n_samples": 2000},
        "description": "X→Regime (MoE向き)",
    },
    "Hamilton": {
        "generator": generate_hamilton_gnp_data,
        "params": {"n_samples": 500},
        "description": "Latent regime (時系列)",
    },
    "VIX": {
        "generator": generate_vix_data,
        "params": {"n_samples": 1000},
        "description": "Volatility regime",
    },
}


# =============================================================================
# Evaluation
# =============================================================================
def evaluate_cv(X, y, params, config: BenchmarkConfig):
    """Time-series cross-validation."""
    tscv = TimeSeriesSplit(n_splits=config.n_splits)
    scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)

        try:
            model = lgb.train(
                params, train_data, num_boost_round=config.num_boost_round
            )
            pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            scores.append(rmse)
        except Exception:
            scores.append(float("inf"))

    return np.mean(scores)


# =============================================================================
# Optuna Objectives
# =============================================================================
def create_objective_standard(X, y, config: BenchmarkConfig):
    """Standard GBDT objective."""

    def objective(trial):
        params = {
            "objective": "regression",
            "boosting": "gbdt",
            "verbose": -1,
            "num_threads": 4,
            "seed": config.seed,
            # Tree structure
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
            # Learning
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            # Regularization
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            # Sampling
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
        }
        return evaluate_cv(X, y, params, config)

    return objective


def create_objective_moe(X, y, config: BenchmarkConfig, per_expert: bool = False):
    """MoE GBDT objective (full search including smoothing).

    Args:
        per_expert: If True, use per-expert hyperparameters (different tree structure per expert)
    """

    def objective(trial):
        smoothing = trial.suggest_categorical(
            "mixture_r_smoothing", ["none", "ema", "markov", "momentum"]
        )
        num_experts = trial.suggest_int("mixture_num_experts", 2, 4)

        params = {
            "objective": "regression",
            "boosting": "mixture",
            "verbose": -1,
            "num_threads": 4,
            "seed": config.seed,
            # Learning
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            # Regularization
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            # Sampling
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
            # MoE specific
            "mixture_num_experts": num_experts,
            "mixture_e_step_alpha": trial.suggest_float(
                "mixture_e_step_alpha", 0.1, 2.0
            ),
            "mixture_warmup_iters": trial.suggest_int("mixture_warmup_iters", 5, 30),
            "mixture_balance_factor": trial.suggest_int(
                "mixture_balance_factor", 2, 10
            ),
            "mixture_r_smoothing": smoothing,
        }

        if per_expert:
            # Per-expert tree structure (different for each expert)
            max_depths = [
                trial.suggest_int(f"max_depth_{k}", 3, 12) for k in range(num_experts)
            ]
            num_leaves = [
                trial.suggest_int(f"num_leaves_{k}", 8, 128) for k in range(num_experts)
            ]
            min_data = [
                trial.suggest_int(f"min_data_in_leaf_{k}", 5, 100)
                for k in range(num_experts)
            ]

            params["mixture_expert_max_depths"] = ",".join(map(str, max_depths))
            params["mixture_expert_num_leaves"] = ",".join(map(str, num_leaves))
            params["mixture_expert_min_data_in_leaf"] = ",".join(map(str, min_data))
        else:
            # Shared tree structure (same for all experts)
            params["num_leaves"] = trial.suggest_int("num_leaves", 8, 128)
            params["max_depth"] = trial.suggest_int("max_depth", 3, 12)
            params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 5, 100)

        # Smoothing lambda (if smoothing enabled)
        if smoothing != "none":
            params["mixture_smoothing_lambda"] = trial.suggest_float(
                "mixture_smoothing_lambda", 0.1, 0.9
            )

        return evaluate_cv(X, y, params, config)

    return objective


# =============================================================================
# Analysis & Visualization
# =============================================================================
def analyze_regime_prediction(X, y, regime_true, params, config: BenchmarkConfig):
    """Regimeの予測精度を分析。"""
    full_params = {
        "objective": "regression",
        "boosting": "mixture",
        "verbose": -1,
        "num_threads": 4,
        "seed": config.seed,
    }
    full_params.update(params)

    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(full_params, train_data, num_boost_round=config.num_boost_round)

    # Predicted regime (argmax of gate probabilities)
    regime_pred = model.predict_regime(X)
    K = full_params.get("mixture_num_experts", 2)
    n_true_regimes = len(np.unique(regime_true))

    # Confusion matrix: rows = true regime, cols = predicted expert
    confusion = np.zeros((n_true_regimes, K))
    for r in range(n_true_regimes):
        mask = regime_true == r
        for k in range(K):
            confusion[r, k] = (regime_pred[mask] == k).sum()

    # Normalize to percentages
    confusion_pct = confusion / confusion.sum(axis=1, keepdims=True) * 100

    # Expert usage statistics
    expert_counts = np.bincount(regime_pred, minlength=K)
    expert_pct = expert_counts / len(regime_pred) * 100

    # Regime-expert mapping (which expert corresponds to which true regime)
    regime_to_expert = {}
    for r in range(n_true_regimes):
        regime_to_expert[r] = np.argmax(confusion[r])

    # Prediction accuracy (if K matches true regimes)
    accuracy = None
    if K == n_true_regimes:
        # Try to find best mapping
        from itertools import permutations

        best_acc = 0
        for perm in permutations(range(K)):
            mapped = np.array([perm[p] for p in regime_pred])
            acc = (mapped == regime_true).mean()
            best_acc = max(best_acc, acc)
        accuracy = best_acc

    return {
        "confusion": confusion,
        "confusion_pct": confusion_pct,
        "expert_pct": expert_pct,
        "regime_to_expert": regime_to_expert,
        "accuracy": accuracy,
        "K": K,
        "n_true_regimes": n_true_regimes,
    }


def print_regime_analysis(name: str, analysis: dict):
    """Regime分析結果を出力。"""
    K = analysis["K"]
    n_true = analysis["n_true_regimes"]
    confusion_pct = analysis["confusion_pct"]
    expert_pct = analysis["expert_pct"]

    print(f"\n  [Regime Analysis for {name}]")
    print(f"  True regimes: {n_true}, MoE experts: {K}")

    # Confusion matrix
    header = "  True\\Pred |" + "".join([f" E{k:>5}" for k in range(K)]) + " |"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in range(n_true):
        row = f"  Regime {r}  |"
        dominant = np.argmax(confusion_pct[r])
        for k in range(K):
            val = confusion_pct[r, k]
            if k == dominant:
                row += f" *{val:>4.1f}%"
            else:
                row += f"  {val:>4.1f}%"
        row += f" | → E{dominant}"
        print(row)

    # Expert usage
    print(
        "\n  Expert usage: "
        + ", ".join([f"E{k}={expert_pct[k]:.1f}%" for k in range(K)])
    )

    # Accuracy if available
    if analysis["accuracy"] is not None:
        print(f"  Regime prediction accuracy: {analysis['accuracy']:.1%}")


def print_prediction_stats(X, y, params_std, params_moe, config: BenchmarkConfig):
    """予測統計を出力。"""
    # Train both models on full data
    train_data = lgb.Dataset(X, label=y)

    full_params_std = {
        "objective": "regression",
        "boosting": "gbdt",
        "verbose": -1,
        "num_threads": 4,
        "seed": config.seed,
    }
    full_params_std.update(params_std)

    full_params_moe = {
        "objective": "regression",
        "boosting": "mixture",
        "verbose": -1,
        "num_threads": 4,
        "seed": config.seed,
    }
    full_params_moe.update(params_moe)

    model_std = lgb.train(
        full_params_std, train_data, num_boost_round=config.num_boost_round
    )
    model_moe = lgb.train(
        full_params_moe, train_data, num_boost_round=config.num_boost_round
    )

    pred_std = model_std.predict(X)
    pred_moe = model_moe.predict(X)

    # Residual statistics
    resid_std = y - pred_std
    resid_moe = y - pred_moe

    print("\n  [Prediction Statistics (on training data)]")
    print(f"  {'':20} {'Standard':>12} {'MoE':>12}")
    print(
        f"  {'RMSE':20} {np.sqrt((resid_std**2).mean()):>12.4f} {np.sqrt((resid_moe**2).mean()):>12.4f}"
    )
    print(
        f"  {'MAE':20} {np.abs(resid_std).mean():>12.4f} {np.abs(resid_moe).mean():>12.4f}"
    )
    print(
        f"  {'Max Error':20} {np.abs(resid_std).max():>12.4f} {np.abs(resid_moe).max():>12.4f}"
    )
    print(f"  {'Residual Std':20} {resid_std.std():>12.4f} {resid_moe.std():>12.4f}")


# =============================================================================
# Benchmark Runner
# =============================================================================
def run_benchmark(
    name: str, X, y, regime_true, config: BenchmarkConfig, test_per_expert: bool = False
):
    """1データセットのベンチマーク実行。

    Args:
        test_per_expert: If True, also test MoE with per-expert hyperparameters
    """
    print(f"\n{'=' * 70}")
    print(f"Dataset: {name}")
    print(f"{'=' * 70}")
    print(f"Samples: {len(y)}, Features: {X.shape[1]}")
    print(
        f"Regime distribution: {(regime_true == 0).mean():.1%} / {(regime_true == 1).mean():.1%}"
    )

    n_methods = 3 if test_per_expert else 2
    results = {}

    # Standard GBDT
    print(f"\n[1/{n_methods}] Standard GBDT ({config.n_trials} trials)...")
    start = time.time()
    study_std = optuna.create_study(direction="minimize")
    study_std.optimize(
        create_objective_standard(X, y, config),
        n_trials=config.n_trials,
        show_progress_bar=True,
    )
    elapsed_std = time.time() - start
    results["Standard"] = {
        "rmse": study_std.best_value,
        "params": study_std.best_params,
        "time": elapsed_std,
    }
    print(f"  Best RMSE: {study_std.best_value:.4f}")

    # MoE GBDT (shared hyperparameters)
    print(f"\n[2/{n_methods}] MoE GBDT ({config.n_trials} trials)...")
    start = time.time()
    study_moe = optuna.create_study(direction="minimize")
    study_moe.optimize(
        create_objective_moe(X, y, config, per_expert=False),
        n_trials=config.n_trials,
        show_progress_bar=True,
    )
    elapsed_moe = time.time() - start
    results["MoE"] = {
        "rmse": study_moe.best_value,
        "params": study_moe.best_params,
        "time": elapsed_moe,
    }
    print(f"  Best RMSE: {study_moe.best_value:.4f}")

    # MoE GBDT (per-expert hyperparameters)
    if test_per_expert:
        print(f"\n[3/{n_methods}] MoE per-expert ({config.n_trials} trials)...")
        start = time.time()
        study_moe_pe = optuna.create_study(direction="minimize")
        study_moe_pe.optimize(
            create_objective_moe(X, y, config, per_expert=True),
            n_trials=config.n_trials,
            show_progress_bar=True,
        )
        elapsed_moe_pe = time.time() - start
        results["MoE-PerExp"] = {
            "rmse": study_moe_pe.best_value,
            "params": study_moe_pe.best_params,
            "time": elapsed_moe_pe,
        }
        print(f"  Best RMSE: {study_moe_pe.best_value:.4f}")

    # Comparison
    std_rmse = results["Standard"]["rmse"]
    moe_rmse = results["MoE"]["rmse"]
    improvement = (std_rmse - moe_rmse) / std_rmse * 100

    print(f"\n  {'Standard RMSE:':<20} {std_rmse:.4f}")
    print(f"  {'MoE RMSE:':<20} {moe_rmse:.4f}")
    if improvement > 0:
        print(f"  {'Improvement:':<20} +{improvement:.1f}% ✓")
    else:
        print(f"  {'Improvement:':<20} {improvement:.1f}%")

    if test_per_expert:
        moe_pe_rmse = results["MoE-PerExp"]["rmse"]
        improvement_pe = (std_rmse - moe_pe_rmse) / std_rmse * 100
        print(f"  {'MoE-PerExp RMSE:':<20} {moe_pe_rmse:.4f}")
        if improvement_pe > 0:
            print(f"  {'PerExp Improvement:':<20} +{improvement_pe:.1f}% ✓")
        else:
            print(f"  {'PerExp Improvement:':<20} {improvement_pe:.1f}%")

    # MoE best params summary
    moe_params = results["MoE"]["params"]
    print("\n  MoE Best Params:")
    print(f"    K={moe_params.get('mixture_num_experts')}, ", end="")
    print(f"alpha={moe_params.get('mixture_e_step_alpha', 0):.2f}, ", end="")
    print(f"balance={moe_params.get('mixture_balance_factor')}, ", end="")
    print(f"smoothing={moe_params.get('mixture_r_smoothing')}")
    if moe_params.get("mixture_r_smoothing") != "none":
        print(
            f"    smoothing_lambda={moe_params.get('mixture_smoothing_lambda', 0):.3f}"
        )

    if test_per_expert:
        moe_pe_params = results["MoE-PerExp"]["params"]
        K = moe_pe_params.get("mixture_num_experts", 2)
        print("\n  MoE-PerExp Best Params:")
        print(f"    K={K}, ", end="")
        print(f"alpha={moe_pe_params.get('mixture_e_step_alpha', 0):.2f}, ", end="")
        print(f"balance={moe_pe_params.get('mixture_balance_factor')}, ", end="")
        print(f"smoothing={moe_pe_params.get('mixture_r_smoothing')}")
        # Per-expert structure
        depths = [moe_pe_params.get(f"max_depth_{k}", "?") for k in range(K)]
        leaves = [moe_pe_params.get(f"num_leaves_{k}", "?") for k in range(K)]
        min_data = [moe_pe_params.get(f"min_data_in_leaf_{k}", "?") for k in range(K)]
        print(f"    max_depths: {depths}")
        print(f"    num_leaves: {leaves}")
        print(f"    min_data:   {min_data}")

    # Regime analysis (use best MoE)
    best_moe_key = "MoE"
    if test_per_expert and results["MoE-PerExp"]["rmse"] < results["MoE"]["rmse"]:
        best_moe_key = "MoE-PerExp"

    try:
        analysis = analyze_regime_prediction(
            X, y, regime_true, results[best_moe_key]["params"], config
        )
        print_regime_analysis(f"{name} ({best_moe_key})", analysis)
        results["regime_analysis"] = analysis
    except Exception as e:
        print(f"\n  [Regime analysis failed: {e}]")

    return results


def print_final_summary(all_results: dict, test_per_expert: bool = False):
    """最終サマリーを出力。"""
    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)

    # Results table
    if test_per_expert:
        print(
            f"\n{'Dataset':<12} {'Std':>10} {'MoE':>10} {'MoE-PE':>10} {'Best':>12} {'vs Std':>10}"
        )
        print("-" * 70)

        for name, results in all_results.items():
            std_rmse = results["Standard"]["rmse"]
            moe_rmse = results["MoE"]["rmse"]
            moe_pe_rmse = results.get("MoE-PerExp", {}).get("rmse", float("inf"))

            best_rmse = min(std_rmse, moe_rmse, moe_pe_rmse)
            if best_rmse == std_rmse:
                winner = "Standard"
            elif best_rmse == moe_rmse:
                winner = "MoE ✓"
            else:
                winner = "MoE-PE ✓"

            improvement = (std_rmse - best_rmse) / std_rmse * 100

            print(
                f"{name:<12} {std_rmse:>10.4f} {moe_rmse:>10.4f} {moe_pe_rmse:>10.4f} {winner:>12} {improvement:>+9.1f}%"
            )
    else:
        print(
            f"\n{'Dataset':<15} {'Std RMSE':>12} {'MoE RMSE':>12} {'Improve':>10} {'Winner':>10}"
        )
        print("-" * 60)

        for name, results in all_results.items():
            std_rmse = results["Standard"]["rmse"]
            moe_rmse = results["MoE"]["rmse"]
            improvement = (std_rmse - moe_rmse) / std_rmse * 100
            winner = "MoE ✓" if improvement > 0 else "Standard"

            print(
                f"{name:<15} {std_rmse:>12.4f} {moe_rmse:>12.4f} {improvement:>+9.1f}% {winner:>10}"
            )

    # MoE hyperparameters summary
    print("\n" + "-" * 70)
    print("Best MoE Configurations:")
    print("-" * 70)

    for name, results in all_results.items():
        params = results["MoE"]["params"]
        print(f"\n{name} (MoE):")
        print(f"  K={params.get('mixture_num_experts')}, ", end="")
        print(f"alpha={params.get('mixture_e_step_alpha', 0):.2f}, ", end="")
        print(f"balance={params.get('mixture_balance_factor')}, ", end="")
        print(f"smoothing={params.get('mixture_r_smoothing')}", end="")
        if params.get("mixture_r_smoothing") != "none":
            print(f", λ={params.get('mixture_smoothing_lambda', 0):.2f}")
        else:
            print()

        if test_per_expert and "MoE-PerExp" in results:
            pe_params = results["MoE-PerExp"]["params"]
            K = pe_params.get("mixture_num_experts", 2)
            print(f"{name} (MoE-PerExp):")
            print(f"  K={K}, ", end="")
            print(f"alpha={pe_params.get('mixture_e_step_alpha', 0):.2f}, ", end="")
            print(f"balance={pe_params.get('mixture_balance_factor')}, ", end="")
            print(f"smoothing={pe_params.get('mixture_r_smoothing')}")
            depths = [pe_params.get(f"max_depth_{k}", "?") for k in range(K)]
            leaves = [pe_params.get(f"num_leaves_{k}", "?") for k in range(K)]
            print(f"  max_depths={depths}, num_leaves={leaves}")

    # Regime prediction summary
    print("\n" + "-" * 70)
    print("Regime Prediction Summary:")
    print("-" * 70)

    for name, results in all_results.items():
        if "regime_analysis" in results:
            analysis = results["regime_analysis"]
            acc_str = f"{analysis['accuracy']:.1%}" if analysis["accuracy"] else "N/A"
            print(f"{name}: K={analysis['K']}, accuracy={acc_str}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark: Standard GBDT vs MoE GBDT")
    parser.add_argument(
        "--trials", type=int, default=50, help="Number of Optuna trials"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--splits", type=int, default=5, help="CV splits")
    parser.add_argument("--rounds", type=int, default=100, help="Boosting rounds")
    parser.add_argument(
        "--per-expert", action="store_true", help="Also test per-expert hyperparameters"
    )
    args = parser.parse_args()

    config = BenchmarkConfig(
        n_trials=args.trials,
        n_splits=args.splits,
        num_boost_round=args.rounds,
        seed=args.seed,
    )

    print("=" * 70)
    print("Benchmark: Standard GBDT vs MoE GBDT")
    if args.per_expert:
        print("(including per-expert hyperparameters)")
    print("=" * 70)
    print(f"Trials: {config.n_trials}, CV Splits: {config.n_splits}, ")
    print(f"Boost Rounds: {config.num_boost_round}, Seed: {config.seed}")

    all_results = {}

    for name, dataset_info in DATASETS.items():
        generator = dataset_info["generator"]
        params = dataset_info["params"]
        params["seed"] = config.seed

        X, y, regime = generator(**params)
        all_results[name] = run_benchmark(
            name, X, y, regime, config, test_per_expert=args.per_expert
        )

    print_final_summary(all_results, test_per_expert=args.per_expert)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
