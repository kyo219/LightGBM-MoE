#!/usr/bin/env python
# coding: utf-8
"""
Benchmark 2: Standard GBDT vs MoE GBDT (with init search)

Standard GBDT と MoE GBDT を比較。
- mixture_init を探索対象に追加
- routing_mode を探索対象に追加

Usage:
    python examples/benchmark_2.py                    # Full benchmark (100 trials)
    python examples/benchmark_2.py --trials 10        # Quick test
"""

import argparse
import time
import warnings
from dataclasses import dataclass
from itertools import permutations

import matplotlib.pyplot as plt
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
    n_trials: int = 100
    n_splits: int = 5
    num_boost_round: int = 100
    seed: int = 42


# =============================================================================
# Data Generation
# =============================================================================
def generate_synthetic_data(n_samples: int = 2000, noise_level: float = 0.5, seed: int = 42):
    """
    Synthetic regime-switching data (regime determinable from X).
    MoEが得意とするケース：特徴量からregimeが決まる。
    """
    np.random.seed(seed)
    n_features = 5
    X = np.random.randn(n_samples, n_features)

    regime_score = 0.5 * X[:, 1] + 0.3 * X[:, 2] - 0.2 * X[:, 3]
    regime_true = (regime_score > 0).astype(int)

    y = np.zeros(n_samples)

    mask0 = regime_true == 0
    y[mask0] = 5.0 * X[mask0, 0] + 3.0 * X[mask0, 0] * X[mask0, 2] + 2.0 * np.sin(2 * X[mask0, 3]) + 10.0

    mask1 = regime_true == 1
    y[mask1] = -5.0 * X[mask1, 0] - 2.0 * X[mask1, 1] ** 2 + 3.0 * np.cos(2 * X[mask1, 4]) - 10.0

    y += np.random.randn(n_samples) * noise_level

    return X, y, regime_true


def generate_hamilton_gnp_data(n_samples: int = 500, seed: int = 42):
    """
    Hamilton GNP-like regime-switching data (latent regime).
    """
    np.random.seed(seed)
    n_features = 4
    X = np.random.randn(n_samples, n_features)

    t = np.arange(n_samples)
    regime_prob = 0.5 + 0.3 * np.sin(2 * np.pi * t / 100)
    regime_true = (np.random.rand(n_samples) < regime_prob).astype(int)

    y = np.zeros(n_samples)

    mask0 = regime_true == 0
    y[mask0] = 0.8 + 0.3 * X[mask0, 0] + 0.2 * X[mask0, 1]

    mask1 = regime_true == 1
    y[mask1] = -0.5 + 0.1 * X[mask1, 0] - 0.3 * X[mask1, 2]

    y += np.random.randn(n_samples) * 0.3

    return X, y, regime_true


DATASETS = {
    "Synthetic": {
        "generator": generate_synthetic_data,
        "params": {"n_samples": 2000},
        "description": "X->Regime (MoE ideal)",
    },
    "Hamilton": {
        "generator": generate_hamilton_gnp_data,
        "params": {"n_samples": 500},
        "description": "Latent regime",
    },
}


# =============================================================================
# Expert Collapse Early Stopping
# =============================================================================
def expert_collapse_stopper(
    X_sample: np.ndarray,
    check_every: int = 20,
    corr_threshold: float = 0.7,
    min_expert_ratio: float = 0.05,
    min_iters: int = 50,
    verbose: bool = False,
):
    """
    Expert collapseを検知したら学習を打ち切るcallback。
    """

    def callback(env):
        iteration = env.iteration

        if iteration < min_iters or iteration % check_every != 0:
            return

        try:
            model = env.model
            expert_preds = model.predict_expert_pred(X_sample)
            K = expert_preds.shape[1]

            # 1. Expert間の相関をチェック（max pairwise）
            max_corr = 0.0
            for i in range(K):
                for j in range(i + 1, K):
                    corr = abs(np.corrcoef(expert_preds[:, i], expert_preds[:, j])[0, 1])
                    max_corr = max(max_corr, corr)

            if max_corr > corr_threshold:
                if verbose:
                    print(f"[iter {iteration}] Expert collapse detected: max_corr={max_corr:.3f} > {corr_threshold}")
                raise lgb.EarlyStopException(iteration, [])

            # 2. Expert利用率をチェック
            regime_pred = model.predict_regime(X_sample)
            n_samples = len(regime_pred)
            for k in range(K):
                expert_ratio = (regime_pred == k).sum() / n_samples
                if expert_ratio < min_expert_ratio:
                    if verbose:
                        print(f"[iter {iteration}] Expert underutilization: E{k} ratio={expert_ratio:.1%} < {min_expert_ratio:.1%}")
                    raise lgb.EarlyStopException(iteration, [])

        except lgb.EarlyStopException:
            raise
        except Exception:
            pass

    return callback


# =============================================================================
# Evaluation
# =============================================================================
def evaluate_cv(
    X,
    y,
    params,
    config: BenchmarkConfig,
    use_collapse_stopper: bool = False,
    collapse_stopper_kwargs: dict = None,
):
    """Time-series cross-validation with early stopping."""
    tscv = TimeSeriesSplit(n_splits=config.n_splits)
    scores = []

    if collapse_stopper_kwargs is None:
        collapse_stopper_kwargs = {}

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]

        if use_collapse_stopper:
            sample_size = min(500, len(X_train))
            sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
            X_sample = X_train[sample_idx]
            callbacks.append(expert_collapse_stopper(X_sample, **collapse_stopper_kwargs))

        try:
            model = lgb.train(
                params,
                train_data,
                num_boost_round=config.num_boost_round,
                valid_sets=[valid_data],
                callbacks=callbacks,
            )
            pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            scores.append(rmse)
        except lgb.EarlyStopException:
            scores.append(float("inf"))
        except Exception:
            scores.append(float("inf"))

    return np.mean(scores)


# =============================================================================
# Optuna Objectives
# =============================================================================
def create_objective_standard(X, y, config: BenchmarkConfig):
    def objective(trial):
        params = {
            "objective": "regression",
            "boosting": "gbdt",
            "verbose": -1,
            "num_threads": 4,
            "seed": config.seed,
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
        return evaluate_cv(X, y, params, config)

    return objective


def create_objective_moe(
    X,
    y,
    config: BenchmarkConfig,
    use_collapse_stopper: bool = False,
    collapse_stopper_kwargs: dict = None,
):
    """Create MoE objective for Optuna with init and routing search."""
    if collapse_stopper_kwargs is None:
        collapse_stopper_kwargs = {}

    def objective(trial):
        # Search initialization method
        mixture_init = trial.suggest_categorical(
            "mixture_init",
            ["uniform", "quantile", "random", "balanced_kmeans", "gmm", "tree_hierarchical"]
        )

        # Search routing mode
        routing_mode = trial.suggest_categorical(
            "mixture_routing_mode",
            ["token_choice", "expert_choice"]
        )

        num_experts = trial.suggest_int("mixture_num_experts", 2, 4)

        params = {
            "objective": "regression",
            "boosting": "mixture",
            "verbose": -1,
            "num_threads": 4,
            "seed": config.seed,
            # Tree structure
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
            # MoE specific
            "mixture_num_experts": num_experts,
            "mixture_init": mixture_init,
            "mixture_routing_mode": routing_mode,
            "mixture_e_step_alpha": trial.suggest_float("mixture_e_step_alpha", 0.1, 2.0),
            "mixture_warmup_iters": trial.suggest_int("mixture_warmup_iters", 5, 50),
            "mixture_balance_factor": trial.suggest_int("mixture_balance_factor", 2, 10),
            "mixture_r_smoothing": "none",
            "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
            # Gate parameters
            "mixture_gate_max_depth": trial.suggest_int("mixture_gate_max_depth", 2, 6),
            "mixture_gate_num_leaves": trial.suggest_int("mixture_gate_num_leaves", 4, 32),
            "mixture_gate_learning_rate": trial.suggest_float("mixture_gate_learning_rate", 0.01, 0.3, log=True),
        }

        # Expert Choice specific parameters
        if routing_mode == "expert_choice":
            params["mixture_expert_capacity_factor"] = trial.suggest_float("mixture_expert_capacity_factor", 0.8, 1.5)
            params["mixture_expert_choice_score"] = "gate"
            params["mixture_expert_choice_boost"] = trial.suggest_float("mixture_expert_choice_boost", 5.0, 30.0)
            params["mixture_expert_choice_hard"] = trial.suggest_categorical("mixture_expert_choice_hard", [True, False])

        return evaluate_cv(
            X, y, params, config,
            use_collapse_stopper=use_collapse_stopper,
            collapse_stopper_kwargs=collapse_stopper_kwargs,
        )

    return objective


# =============================================================================
# MoE Analysis
# =============================================================================
def analyze_moe(X, y, regime_true, best_params, config: BenchmarkConfig):
    """Analyze MoE model: regime separation & expert differentiation."""
    full_params = {
        "objective": "regression",
        "boosting": "mixture",
        "verbose": -1,
        "num_threads": 4,
        "seed": config.seed,
    }
    full_params.update(best_params)

    # Add expert_choice fixed params if needed
    if full_params.get("mixture_routing_mode") == "expert_choice":
        if "mixture_expert_choice_score" not in full_params:
            full_params["mixture_expert_choice_score"] = "gate"

    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(full_params, train_data, num_boost_round=config.num_boost_round)

    final_pred = model.predict(X)
    regime_pred = model.predict_regime(X)
    regime_proba = model.predict_regime_proba(X)
    expert_preds = model.predict_expert_pred(X)
    K = full_params.get("mixture_num_experts", 2)

    # Gate check
    gate_working = not np.allclose(regime_proba, 1.0 / K, atol=0.01)
    proba_range = (regime_proba.min(), regime_proba.max())

    # Expert differentiation
    corrs = []
    for i in range(K):
        for j in range(i + 1, K):
            corrs.append(np.corrcoef(expert_preds[:, i], expert_preds[:, j])[0, 1])
    expert_corr_max = max(corrs)
    expert_corr_min = min(corrs)
    experts_different = expert_corr_max < 0.99

    # Regime accuracy
    n_true_regimes = len(np.unique(regime_true))
    best_acc = 0
    for perm in permutations(range(K)):
        mapped = np.array([perm[p] % n_true_regimes for p in regime_pred])
        acc = (mapped == regime_true).mean()
        if acc > best_acc:
            best_acc = acc

    # Expert utilization
    expert_utilization = {}
    for k in range(K):
        util = (regime_pred == k).mean()
        expert_utilization[f"expert_{k}"] = util

    return {
        "K": K,
        "gate_working": gate_working,
        "proba_range": proba_range,
        "experts_different": experts_different,
        "expert_corr_max": expert_corr_max,
        "expert_corr_min": expert_corr_min,
        "regime_accuracy": best_acc,
        "expert_utilization": expert_utilization,
        "final_pred": final_pred,
        "y": y,
    }


# =============================================================================
# Benchmark Runner
# =============================================================================
def run_benchmark(
    name: str,
    X,
    y,
    regime_true,
    config: BenchmarkConfig,
    use_collapse_stopper: bool = False,
    collapse_stopper_kwargs: dict = None,
):
    """Run benchmark for one dataset."""
    print(f"\n{'=' * 70}")
    print(f"Dataset: {name}")
    print(f"{'=' * 70}")
    print(f"Samples: {len(y)}, Features: {X.shape[1]}")
    print(f"Regime distribution: {(regime_true == 0).mean():.1%} / {(regime_true == 1).mean():.1%}")

    if use_collapse_stopper:
        print(f"Expert Collapse Stopper: ENABLED {collapse_stopper_kwargs}")

    results = {}

    # 1. Standard GBDT
    print(f"\n[1] Standard GBDT ({config.n_trials} trials)...")
    start = time.time()
    study_std = optuna.create_study(direction="minimize")
    study_std.optimize(
        create_objective_standard(X, y, config),
        n_trials=config.n_trials,
        show_progress_bar=True,
    )
    results["Standard"] = {
        "rmse": study_std.best_value,
        "params": study_std.best_params,
        "time": time.time() - start,
    }
    print(f"  Best RMSE: {study_std.best_value:.4f}")

    # 2. MoE
    print(f"\n[2] MoE ({config.n_trials} trials)...")
    start = time.time()
    study_moe = optuna.create_study(direction="minimize")
    study_moe.optimize(
        create_objective_moe(
            X, y, config,
            use_collapse_stopper=use_collapse_stopper,
            collapse_stopper_kwargs=collapse_stopper_kwargs,
        ),
        n_trials=config.n_trials,
        show_progress_bar=True,
    )
    results["MoE"] = {
        "rmse": study_moe.best_value,
        "params": study_moe.best_params,
        "time": time.time() - start,
    }
    print(f"  Best RMSE: {study_moe.best_value:.4f}")

    # MoE Analysis
    print("\n[3] MoE Analysis...")
    moe_params = study_moe.best_params
    moe_routing = moe_params.get("mixture_routing_mode", "token_choice")
    moe_init = moe_params.get("mixture_init", "uniform")
    analysis_moe = analyze_moe(X, y, regime_true, moe_params, config)
    results["analysis_moe"] = analysis_moe

    print(f"  Routing: {moe_routing}, Init: {moe_init}")
    print(f"  Gate working: {'Yes' if analysis_moe['gate_working'] else 'NO!'}")
    print(f"  Expert corr (max/min): {analysis_moe['expert_corr_max']:.2f} / {analysis_moe['expert_corr_min']:.2f}")
    print(f"  Regime accuracy: {analysis_moe['regime_accuracy']:.1%}")

    # Expert utilization
    K = analysis_moe["K"]
    util_str = ", ".join([f"E{k}={analysis_moe['expert_utilization'][f'expert_{k}']:.1%}" for k in range(K)])
    print(f"  Expert utilization: {util_str}")

    # Summary
    std_rmse = results["Standard"]["rmse"]
    moe_rmse = results["MoE"]["rmse"]
    improvement = (std_rmse - moe_rmse) / std_rmse * 100

    print("\n  --- Summary ---")
    print(f"  Standard RMSE: {std_rmse:.4f}")
    print(f"  MoE RMSE:      {moe_rmse:.4f}")
    print(f"  Improvement:   {improvement:+.1f}%")

    return results


# =============================================================================
# Visualization
# =============================================================================
def create_visualization(all_results: dict, output_path: str):
    """Create visualization of benchmark results."""
    n_datasets = len(all_results)
    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 5))

    if n_datasets == 1:
        axes = [axes]

    for i, (name, results) in enumerate(all_results.items()):
        ax = axes[i]

        methods = ["Standard", "MoE"]
        rmses = [results["Standard"]["rmse"], results["MoE"]["rmse"]]
        colors = ["gray", "C0"]

        bars = ax.bar(methods, rmses, color=colors, alpha=0.7, edgecolor="black")
        ax.set_ylabel("RMSE")
        ax.set_title(f"{name}")

        for bar, rmse in zip(bars, rmses, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{rmse:.4f}",
                    ha="center", va="bottom", fontsize=10)

        # Add MoE info
        moe_params = results["MoE"]["params"]
        routing = moe_params.get("mixture_routing_mode", "?")
        init = moe_params.get("mixture_init", "?")
        ax.text(0.5, 0.02, f"routing={routing}, init={init}",
                transform=ax.transAxes, ha="center", fontsize=9, style="italic")

    plt.suptitle("Standard vs MoE Benchmark", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved: {output_path}")
    plt.close()


def print_final_summary(all_results: dict):
    """Print final summary table."""
    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    print(f"\n{'Dataset':<12} {'Standard':>10} {'MoE':>10} {'Impr':>8} {'K':>4} {'Routing':<14} {'Init':<18} {'Corr':>10}")
    print("-" * 100)

    for name, results in all_results.items():
        std_rmse = results["Standard"]["rmse"]
        moe_rmse = results["MoE"]["rmse"]
        improvement = (std_rmse - moe_rmse) / std_rmse * 100

        moe_params = results["MoE"]["params"]
        K = moe_params.get("mixture_num_experts", 2)
        routing = moe_params.get("mixture_routing_mode", "?")
        init = moe_params.get("mixture_init", "?")

        analysis = results["analysis_moe"]
        corr = f"{analysis['expert_corr_min']:.2f}/{analysis['expert_corr_max']:.2f}"

        print(f"{name:<12} {std_rmse:>10.4f} {moe_rmse:>10.4f} {improvement:>+7.1f}% {K:>4} {routing:<14} {init:<18} {corr:>10}")

    print("=" * 100)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Benchmark 2: Standard vs MoE (with init search)")
    parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--splits", type=int, default=5, help="CV splits")
    parser.add_argument("--rounds", type=int, default=100, help="Boosting rounds")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    parser.add_argument("--collapse-stopper", action="store_true", help="Enable expert collapse early stopping")
    parser.add_argument("--corr-threshold", type=float, default=0.7, help="Expert correlation threshold")
    parser.add_argument("--min-expert-ratio", type=float, default=0.05, help="Minimum expert utilization ratio")
    args = parser.parse_args()

    config = BenchmarkConfig(
        n_trials=args.trials,
        n_splits=args.splits,
        num_boost_round=args.rounds,
        seed=args.seed,
    )

    print("=" * 70)
    print("Benchmark 2: Standard vs MoE (with init search)")
    print("=" * 70)
    print(f"Trials: {config.n_trials}, CV Splits: {config.n_splits}, Boost Rounds: {config.num_boost_round}")

    collapse_stopper_kwargs = {
        "corr_threshold": args.corr_threshold,
        "min_expert_ratio": args.min_expert_ratio,
        "check_every": 20,
        "min_iters": 50,
        "verbose": True,
    }
    if args.collapse_stopper:
        print(f"Expert Collapse Stopper: ENABLED")
        print(f"  corr_threshold={args.corr_threshold}, min_expert_ratio={args.min_expert_ratio}")

    all_results = {}

    for name, dataset_info in DATASETS.items():
        generator = dataset_info["generator"]
        params = dataset_info["params"]
        params["seed"] = config.seed

        X, y, regime = generator(**params)
        all_results[name] = run_benchmark(
            name, X, y, regime, config,
            use_collapse_stopper=args.collapse_stopper,
            collapse_stopper_kwargs=collapse_stopper_kwargs,
        )

    print_final_summary(all_results)

    if not args.no_viz:
        create_visualization(all_results, "examples/benchmark_2_results.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
