#!/usr/bin/env python
# coding: utf-8
"""
Benchmark: Standard GBDT vs MoE GBDT

3つのデータセットで Standard GBDT と MoE GBDT を比較。
- RMSE比較
- Regime分離の確認（Gate機能）
- Expert予測差の確認
- 可視化

Usage:
    python examples/benchmark.py                    # Full benchmark (100 trials)
    python examples/benchmark.py --trials 10        # Quick test
"""

import argparse
import time
import warnings
from dataclasses import dataclass
from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np
import optuna
import shap
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


def generate_vix_data(n_samples: int = 1000, seed: int = 42):
    """
    VIX-like volatility regime data.
    """
    np.random.seed(seed)
    n_features = 5
    X = np.random.randn(n_samples, n_features)

    t = np.arange(n_samples)
    regime_prob = 0.3 + 0.4 * (np.sin(2 * np.pi * t / 200) > 0)
    regime_true = (np.random.rand(n_samples) < regime_prob).astype(int)

    y = np.zeros(n_samples)

    mask0 = regime_true == 0
    y[mask0] = 0.01 + 0.002 * np.abs(X[mask0, 0])

    mask1 = regime_true == 1
    y[mask1] = 0.025 + 0.005 * np.abs(X[mask1, 0]) + 0.003 * X[mask1, 1] ** 2

    y += np.random.randn(n_samples) * 0.005

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
    """Time-series cross-validation with early stopping."""
    tscv = TimeSeriesSplit(n_splits=config.n_splits)
    scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        try:
            model = lgb.train(
                params,
                train_data,
                num_boost_round=config.num_boost_round,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
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


def create_objective_moe(X, y, config: BenchmarkConfig, per_expert: bool = False):
    def objective(trial):
        # smoothing=none fixed to prevent expert collapse
        smoothing = "none"
        num_experts = trial.suggest_int("mixture_num_experts", 2, 4)

        params = {
            "objective": "regression",
            "boosting": "mixture",
            "verbose": -1,
            "num_threads": 4,
            "seed": config.seed,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
            "mixture_num_experts": num_experts,
            "mixture_e_step_alpha": trial.suggest_float("mixture_e_step_alpha", 0.1, 2.0),
            "mixture_warmup_iters": trial.suggest_int("mixture_warmup_iters", 5, 50),
            "mixture_balance_factor": trial.suggest_int("mixture_balance_factor", 2, 10),
            "mixture_r_smoothing": smoothing,
        }

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

        # smoothing_lambda not needed when smoothing=none

        return evaluate_cv(X, y, params, config)

    return objective


# =============================================================================
# MoE Analysis (Regime Separation & Expert Differentiation)
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

    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(full_params, train_data, num_boost_round=config.num_boost_round)

    regime_pred = model.predict_regime(X)
    regime_proba = model.predict_regime_proba(X)
    expert_preds = model.predict_expert_pred(X)
    K = full_params.get("mixture_num_experts", 2)

    # Gate check
    gate_working = not np.allclose(regime_proba, 1.0 / K, atol=0.01)
    proba_range = (regime_proba.min(), regime_proba.max())

    # Expert differentiation (correlation between experts)
    corrs = []
    for i in range(K):
        for j in range(i + 1, K):
            corrs.append(np.corrcoef(expert_preds[:, i], expert_preds[:, j])[0, 1])
    expert_corr_max = max(corrs)  # Most similar pair (collapse detection)
    expert_corr_min = min(corrs)  # Most differentiated pair
    experts_different = expert_corr_max < 0.99

    # Regime accuracy (with best permutation)
    # Map predicted experts to true regimes, find best mapping
    n_true_regimes = len(np.unique(regime_true))
    best_acc = 0
    for perm in permutations(range(K)):
        # Map expert k to regime perm[k] % n_true_regimes
        mapped = np.array([perm[p] % n_true_regimes for p in regime_pred])
        acc = (mapped == regime_true).mean()
        best_acc = max(best_acc, acc)

    return {
        "K": K,
        "gate_working": gate_working,
        "proba_range": proba_range,
        "experts_different": experts_different,
        "expert_corr_max": expert_corr_max,
        "expert_corr_min": expert_corr_min,
        "regime_accuracy": best_acc,
        "regime_pred": regime_pred,
        "regime_proba": regime_proba,
        "expert_preds": expert_preds,
        "regime_true": regime_true,
    }


# =============================================================================
# Benchmark Runner
# =============================================================================
def run_benchmark(name: str, X, y, regime_true, config: BenchmarkConfig):
    """Run benchmark for one dataset."""
    print(f"\n{'=' * 70}")
    print(f"Dataset: {name}")
    print(f"{'=' * 70}")
    print(f"Samples: {len(y)}, Features: {X.shape[1]}")
    print(f"Regime distribution: {(regime_true == 0).mean():.1%} / {(regime_true == 1).mean():.1%}")

    results = {}

    # Standard GBDT
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

    # MoE GBDT
    print(f"\n[2] MoE GBDT ({config.n_trials} trials)...")
    start = time.time()
    study_moe = optuna.create_study(direction="minimize")
    study_moe.optimize(
        create_objective_moe(X, y, config, per_expert=False),
        n_trials=config.n_trials,
        show_progress_bar=True,
    )
    results["MoE"] = {
        "rmse": study_moe.best_value,
        "params": study_moe.best_params,
        "time": time.time() - start,
    }
    print(f"  Best RMSE: {study_moe.best_value:.4f}")

    # MoE Per-Expert
    print(f"\n[3] MoE per-expert ({config.n_trials} trials)...")
    start = time.time()
    study_moe_pe = optuna.create_study(direction="minimize")
    study_moe_pe.optimize(
        create_objective_moe(X, y, config, per_expert=True),
        n_trials=config.n_trials,
        show_progress_bar=True,
    )
    results["MoE-PE"] = {
        "rmse": study_moe_pe.best_value,
        "params": study_moe_pe.best_params,
        "time": time.time() - start,
    }
    print(f"  Best RMSE: {study_moe_pe.best_value:.4f}")

    # MoE Analysis
    print("\n[4] MoE Analysis (Regime Separation & Expert Differentiation)...")
    analysis_moe = analyze_moe(X, y, regime_true, study_moe.best_params, config)
    analysis_moe_pe = analyze_moe(X, y, regime_true, study_moe_pe.best_params, config)
    results["analysis_moe"] = analysis_moe
    results["analysis_moe_pe"] = analysis_moe_pe

    # Print analysis
    print("  MoE:")
    print(f"    Gate working: {'Yes' if analysis_moe['gate_working'] else 'NO!'}")
    print(f"    Expert corr (max/min): {analysis_moe['expert_corr_max']:.2f} / {analysis_moe['expert_corr_min']:.2f}")
    print(f"    Regime accuracy: {analysis_moe['regime_accuracy']:.1%}")
    print("  MoE-PE:")
    print(f"    Gate working: {'Yes' if analysis_moe_pe['gate_working'] else 'NO!'}")
    print(
        f"    Expert corr (max/min): {analysis_moe_pe['expert_corr_max']:.2f} / {analysis_moe_pe['expert_corr_min']:.2f}"
    )
    print(f"    Regime accuracy: {analysis_moe_pe['regime_accuracy']:.1%}")

    # Summary
    std_rmse = results["Standard"]["rmse"]
    moe_rmse = results["MoE"]["rmse"]
    moe_pe_rmse = results["MoE-PE"]["rmse"]
    best_rmse = min(moe_rmse, moe_pe_rmse)
    improvement = (std_rmse - best_rmse) / std_rmse * 100

    print("\n  --- Summary ---")
    print(f"  Standard RMSE:  {std_rmse:.4f}")
    print(f"  MoE RMSE:       {moe_rmse:.4f}")
    print(f"  MoE-PE RMSE:    {moe_pe_rmse:.4f}")
    print(f"  Best MoE Impr:  {improvement:+.1f}%")

    return results


# =============================================================================
# Visualization
# =============================================================================
def create_visualization(all_results: dict, output_path: str):
    """Create visualization of regime separation and expert differentiation."""
    n_datasets = len(all_results)
    fig, axes = plt.subplots(n_datasets, 3, figsize=(15, 4 * n_datasets))

    if n_datasets == 1:
        axes = axes.reshape(1, -1)

    for i, (name, results) in enumerate(all_results.items()):
        # Use best MoE analysis for visualization
        if results["MoE"]["rmse"] <= results["MoE-PE"]["rmse"]:
            analysis = results["analysis_moe"]
        else:
            analysis = results["analysis_moe_pe"]
        regime_true = analysis["regime_true"]
        regime_pred = analysis["regime_pred"]
        expert_preds = analysis["expert_preds"]
        K = analysis["K"]

        # Plot 1: Regime Separation
        ax1 = axes[i, 0]
        for true_r in [0, 1]:
            mask = regime_true == true_r
            for pred_e in range(min(K, 2)):
                pct = (regime_pred[mask] == pred_e).mean() * 100
                if pred_e == 0:
                    ax1.barh(f"True R{true_r}", pct, color="C0", alpha=0.7)
                    ax1.text(pct / 2, true_r, f"{pct:.0f}%", ha="center", va="center", fontweight="bold")
                else:
                    ax1.barh(f"True R{true_r}", -pct, color="C1", alpha=0.7)
                    ax1.text(-pct / 2, true_r, f"{pct:.0f}%", ha="center", va="center", fontweight="bold")
        ax1.axvline(0, color="black", linewidth=0.5)
        ax1.set_xlim(-110, 110)
        ax1.set_xlabel("% to E0 (left) / E1 (right)")
        ax1.set_title(f"{name}: Regime Separation\n(Acc: {analysis['regime_accuracy']:.1%})")

        # Plot 2: Expert Predictions Scatter
        ax2 = axes[i, 1]
        colors = ["C0" if r == 0 else "C1" for r in regime_true]
        ax2.scatter(expert_preds[:, 0], expert_preds[:, 1], c=colors, alpha=0.3, s=10)
        lims = [expert_preds.min() - 1, expert_preds.max() + 1]
        ax2.plot(lims, lims, "k--", alpha=0.5)
        ax2.set_xlabel("Expert 0 Prediction")
        ax2.set_ylabel("Expert 1 Prediction")
        ax2.set_title(
            f"{name}: Expert Preds\n(corr: {analysis['expert_corr_min']:.2f}~{analysis['expert_corr_max']:.2f})"
        )

        # Plot 3: RMSE Comparison
        ax3 = axes[i, 2]
        methods = ["Standard", "MoE", "MoE-PE"]
        rmses = [results["Standard"]["rmse"], results["MoE"]["rmse"], results["MoE-PE"]["rmse"]]
        bars = ax3.bar(methods, rmses, color=["gray", "C0", "C2"], alpha=0.7)
        ax3.set_ylabel("RMSE")
        ax3.set_title(f"{name}: RMSE Comparison")
        for bar, rmse in zip(bars, rmses, strict=True):
            ax3.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{rmse:.3f}", ha="center", va="bottom", fontsize=9
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved: {output_path}")
    plt.close()


def create_regime_demo(X, y, regime_true, config: BenchmarkConfig, output_path: str):
    """
    Create regime-switching demo visualization on Synthetic dataset.
    Shows MoE's ability to detect and separate regimes.
    """
    print("\n[Regime Demo] Creating demo visualization on Synthetic data...")

    # Train/Test split (time-based)
    train_size = int(len(y) * 0.7)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    regime_test = regime_true[train_size:]
    t_test = np.arange(len(y_test))

    # Train Standard GBDT
    params_std = {
        "objective": "regression",
        "boosting": "gbdt",
        "verbose": -1,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "seed": config.seed,
    }
    train_data = lgb.Dataset(X_train, label=y_train)
    model_std = lgb.train(params_std, train_data, num_boost_round=100)
    pred_std = model_std.predict(X_test)

    # Train MoE GBDT
    params_moe = {
        "objective": "regression",
        "boosting": "mixture",
        "verbose": -1,
        "mixture_num_experts": 2,
        "mixture_e_step_alpha": 1.0,
        "mixture_warmup_iters": 30,
        "mixture_r_smoothing": "none",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "seed": config.seed,
    }
    model_moe = lgb.train(params_moe, train_data, num_boost_round=100)
    pred_moe = model_moe.predict(X_test)
    regime_pred = model_moe.predict_regime(X_test)
    regime_proba = model_moe.predict_regime_proba(X_test)

    # Check if regime labels need inversion
    acc_normal = (regime_pred == regime_test).mean()
    acc_inverted = ((1 - regime_pred) == regime_test).mean()
    if acc_inverted > acc_normal:
        regime_pred = 1 - regime_pred
        regime_proba = regime_proba[:, ::-1]
        regime_acc = acc_inverted
    else:
        regime_acc = acc_normal

    # Calculate metrics
    rmse_std = np.sqrt(mean_squared_error(y_test, pred_std))
    rmse_moe = np.sqrt(mean_squared_error(y_test, pred_moe))

    # Create visualization (6 subplots)
    fig = plt.figure(figsize=(16, 12))
    regime_colors = ["#2ecc71", "#e74c3c"]
    regime_names = ["Regime 0 (Bull)", "Regime 1 (Bear)"]

    # 1. Actual vs Predicted Scatter
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(y_test, pred_std, alpha=0.5, s=20, label=f"Std (RMSE={rmse_std:.3f})")
    ax1.scatter(y_test, pred_moe, alpha=0.5, s=20, label=f"MoE (RMSE={rmse_moe:.3f})")
    lims = [min(y_test.min(), pred_std.min(), pred_moe.min()), max(y_test.max(), pred_std.max(), pred_moe.max())]
    ax1.plot(lims, lims, "k--", alpha=0.5)
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.set_title("1. Actual vs Predicted")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Time Series: Standard GBDT
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(t_test, y_test, "k-", alpha=0.7, label="Actual", linewidth=1)
    ax2.plot(t_test, pred_std, "b-", alpha=0.7, label="Predicted", linewidth=1)
    ax2.fill_between(t_test, y_test, pred_std, alpha=0.2, color="red")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Value")
    ax2.set_title(f"2. Standard GBDT (RMSE={rmse_std:.3f})")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Time Series: MoE GBDT
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(t_test, y_test, "k-", alpha=0.7, label="Actual", linewidth=1)
    ax3.plot(t_test, pred_moe, "g-", alpha=0.7, label="Predicted", linewidth=1)
    ax3.fill_between(t_test, y_test, pred_moe, alpha=0.2, color="red")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Value")
    ax3.set_title(f"3. MoE GBDT (RMSE={rmse_moe:.3f})")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Regime Estimation
    ax4 = fig.add_subplot(2, 3, 4)
    for r in range(2):
        mask = regime_test == r
        ax4.scatter(t_test[mask], y_test[mask], c=regime_colors[r], alpha=0.6, s=15, label=f"Actual {regime_names[r]}")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Value")
    ax4.set_title(f"4. Regime Estimation (Acc: {regime_acc:.1%})")
    ax4.legend(fontsize=7, loc="upper right")
    ax4.grid(True, alpha=0.3)

    # 5. Regime Probability Over Time
    ax5 = fig.add_subplot(2, 3, 5)
    for r in range(2):
        ax5.fill_between(t_test, 0, regime_proba[:, r], alpha=0.5, color=regime_colors[r], label=regime_names[r])
    regime_changes = np.where(np.diff(regime_test) != 0)[0]
    for rc in regime_changes:
        ax5.axvline(x=t_test[rc], color="black", linestyle="--", alpha=0.5)
    ax5.set_xlabel("Time")
    ax5.set_ylabel("Probability")
    ax5.set_title("5. Gate Probabilities Over Time")
    ax5.legend(fontsize=8, loc="upper right")
    ax5.set_ylim(0, 1)
    ax5.grid(True, alpha=0.3)

    # 6. MAE by Regime
    ax6 = fig.add_subplot(2, 3, 6)
    errors_std = [np.abs(y_test[regime_test == r] - pred_std[regime_test == r]).mean() for r in range(2)]
    errors_moe = [np.abs(y_test[regime_test == r] - pred_moe[regime_test == r]).mean() for r in range(2)]
    x_pos = np.arange(2)
    width = 0.35
    ax6.bar(x_pos - width / 2, errors_std, width, label="Standard", color="steelblue", alpha=0.7)
    ax6.bar(x_pos + width / 2, errors_moe, width, label="MoE", color="forestgreen", alpha=0.7)
    ax6.set_xlabel("True Regime")
    ax6.set_ylabel("MAE")
    ax6.set_title("6. MAE by Regime")
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(["Regime 0", "Regime 1"])
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Regime-Switching Demo: Standard GBDT vs MoE GBDT", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[Regime Demo] Saved: {output_path}")
    plt.close()

    improvement = (rmse_std - rmse_moe) / rmse_std * 100
    print(f"[Regime Demo] Standard RMSE: {rmse_std:.4f}, MoE RMSE: {rmse_moe:.4f} ({improvement:+.1f}%)")
    print(f"[Regime Demo] Regime Accuracy: {regime_acc:.1%}")


def _booster_to_shap_compatible(booster):
    """Convert lightgbm_moe Booster to SHAP-compatible format.

    SHAP's TreeExplainer checks for specific class names like 'lightgbm.basic.Booster'.
    This function converts our Booster to a format SHAP can understand by saving
    to a temporary file and loading with standard lightgbm (if available) or
    creating a compatible wrapper.

    Parameters
    ----------
    booster : lightgbm_moe.Booster
        The booster to convert.

    Returns
    -------
    model : object
        A SHAP-compatible model object.
    """
    import tempfile

    # Try to use standard lightgbm if available
    try:
        import lightgbm as standard_lgb

        # Save to temp file and reload with standard lightgbm
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            model_str = booster.model_to_string(num_iteration=-1)
            f.write(model_str)
            temp_path = f.name

        try:
            return standard_lgb.Booster(model_file=temp_path)
        finally:
            import os

            os.unlink(temp_path)
    except ImportError:
        # Standard lightgbm not available, try direct approach
        # SHAP can sometimes work with the model dump directly
        pass

    # Fallback: return the booster and hope SHAP's internals handle it
    # This may not work, but we'll let the error propagate
    return booster


def create_shap_visualization(
    X, y, config: BenchmarkConfig, best_moe_params: dict, output_dir: str = "examples"
):
    """
    Create SHAP beeswarm plots for MoE components (Gate and Experts).

    This function trains a MoE model with the optimized parameters from the benchmark
    and generates SHAP beeswarm plots for each component (gate and experts).

    Parameters
    ----------
    X : numpy array
        Feature matrix.
    y : numpy array
        Target values.
    config : BenchmarkConfig
        Configuration for training.
    best_moe_params : dict
        Best hyperparameters from Optuna optimization for MoE model.
    output_dir : str
        Directory to save output PNG files.
    """
    print("\n[SHAP] Creating SHAP beeswarm plots for MoE components...")
    print(f"[SHAP] Using optimized MoE params: num_experts={best_moe_params.get('mixture_num_experts', 2)}")

    # Build full params from optimized hyperparameters
    params_moe = {
        "objective": "regression",
        "boosting": "mixture",
        "verbose": -1,
        "num_threads": 4,
        "seed": config.seed,
    }
    params_moe.update(best_moe_params)

    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(params_moe, train_data, num_boost_round=config.num_boost_round)

    # Get component boosters
    boosters = model.get_all_boosters()
    num_experts = model.num_experts()

    # Create feature names
    feature_names = [f"X{i}" for i in range(X.shape[1])]

    # Generate individual SHAP plots
    shap_values_dict = {}

    for name, booster in boosters.items():
        print(f"[SHAP] Computing SHAP values for {name}...")

        # Convert to SHAP-compatible format
        shap_model = _booster_to_shap_compatible(booster)
        explainer = shap.TreeExplainer(shap_model)
        shap_values = explainer.shap_values(X)

        # Handle multi-output models (like gate which outputs probabilities for each expert)
        # shap_values can be:
        # - 2D array (n_samples, n_features) for single output
        # - 3D array (n_samples, n_features, n_classes) for multi-class
        # - list of 2D arrays for multi-class (older SHAP versions)
        if isinstance(shap_values, list):
            # For multi-class, use the first class for simplicity
            shap_values_for_plot = shap_values[0]
            print(f"[SHAP] {name} is multi-output ({len(shap_values)} classes), using class 0")
        elif shap_values.ndim == 3:
            # 3D array: (n_samples, n_features, n_classes)
            # Use the first class for plotting
            shap_values_for_plot = shap_values[:, :, 0]
            print(f"[SHAP] {name} is multi-output ({shap_values.shape[2]} classes), using class 0")
        else:
            shap_values_for_plot = shap_values

        shap_values_dict[name] = shap_values_for_plot

        # Create individual beeswarm plot
        # shap.summary_plot handles figure creation internally, so we close all first
        plt.close("all")
        shap.summary_plot(
            shap_values_for_plot,
            X,
            feature_names=feature_names,
            plot_type="dot",  # "dot" is the beeswarm-style plot
            show=False,
        )
        plt.gcf().set_size_inches(10, 6)
        plt.title(f"SHAP Beeswarm: {name}")
        plt.tight_layout()
        output_path = f"{output_dir}/shap_{name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"[SHAP] Saved: {output_path}")

    # Create combined figure showing mean absolute SHAP values (bar chart)
    # Note: shap.summary_plot doesn't work well with subplots, so we use bar charts
    n_components = 1 + num_experts  # gate + experts
    fig, axes = plt.subplots(1, n_components, figsize=(5 * n_components, 5))

    component_names = ["gate"] + [f"expert_{k}" for k in range(num_experts)]

    for idx, name in enumerate(component_names):
        shap_vals = shap_values_dict[name]
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)

        # Sort by importance
        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        sorted_names = [feature_names[i] for i in sorted_idx]
        sorted_values = mean_abs_shap[sorted_idx]

        ax = axes[idx]
        y_pos = np.arange(len(feature_names))
        ax.barh(y_pos, sorted_values, color="steelblue", alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"{name}")
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle("MoE Component Feature Importance (Mean |SHAP|)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    combined_path = f"{output_dir}/moe_shap_beeswarm.png"
    plt.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Saved combined plot: {combined_path}")

    print("[SHAP] SHAP visualization complete.")


def generate_markdown_report(all_results: dict, config: BenchmarkConfig) -> str:
    """Generate markdown report of benchmark results."""
    lines = []
    lines.append("## Benchmark Results")
    lines.append("")
    lines.append(f"- **Optuna Trials**: {config.n_trials}")
    lines.append(f"- **CV Splits**: {config.n_splits}")
    lines.append(f"- **Boosting Rounds**: {config.num_boost_round}")
    lines.append(f"- **Seed**: {config.seed}")
    lines.append("")

    # RMSE Table
    lines.append("### RMSE Comparison")
    lines.append("")
    lines.append("| Dataset | Standard | MoE | MoE-PE | Best Improvement |")
    lines.append("|---------|----------|-----|--------|------------------|")

    for name, results in all_results.items():
        std_rmse = results["Standard"]["rmse"]
        moe_rmse = results["MoE"]["rmse"]
        moe_pe_rmse = results["MoE-PE"]["rmse"]
        best_rmse = min(moe_rmse, moe_pe_rmse)
        improvement = (std_rmse - best_rmse) / std_rmse * 100

        best_marker_moe = " **" if moe_rmse <= moe_pe_rmse else ""
        best_marker_pe = " **" if moe_pe_rmse < moe_rmse else ""
        end_marker_moe = "**" if moe_rmse <= moe_pe_rmse else ""
        end_marker_pe = "**" if moe_pe_rmse < moe_rmse else ""

        lines.append(
            f"| {name} | {std_rmse:.4f} | {best_marker_moe}{moe_rmse:.4f}{end_marker_moe} | "
            f"{best_marker_pe}{moe_pe_rmse:.4f}{end_marker_pe} | {improvement:+.1f}% |"
        )

    lines.append("")

    # Expert Differentiation Table
    lines.append("### Expert Differentiation (Regime Separation)")
    lines.append("")
    lines.append("| Dataset | MoE Corr (min/max) | MoE-PE Corr (min/max) | MoE Regime Acc | MoE-PE Regime Acc |")
    lines.append("|---------|--------------------|-----------------------|----------------|-------------------|")

    for name, results in all_results.items():
        analysis_moe = results["analysis_moe"]
        analysis_pe = results["analysis_moe_pe"]

        moe_corr = f"{analysis_moe['expert_corr_min']:.2f}/{analysis_moe['expert_corr_max']:.2f}"
        pe_corr = f"{analysis_pe['expert_corr_min']:.2f}/{analysis_pe['expert_corr_max']:.2f}"
        moe_acc = analysis_moe["regime_accuracy"]
        pe_acc = analysis_pe["regime_accuracy"]

        lines.append(f"| {name} | {moe_corr} | {pe_corr} | {moe_acc:.1%} | {pe_acc:.1%} |")

    lines.append("")
    lines.append("**Notes**:")
    lines.append("- **Expert Corr (min/max)**: Min and max pairwise correlation between expert predictions")
    lines.append("  - min: Most differentiated pair (lower = better separation)")
    lines.append("  - max: Most similar pair (if ~1.0, some experts may have collapsed)")
    lines.append(
        "- **Regime Acc**: Classification accuracy of predicted regime vs true regime (with best label permutation)"
    )
    lines.append("")

    # Selected Hyperparameters Section
    lines.append("### Selected Hyperparameters")
    lines.append("")

    for name, results in all_results.items():
        lines.append(f"#### {name}")
        lines.append("")

        # Standard GBDT params
        std_params = results["Standard"]["params"]
        lines.append("**Standard GBDT:**")
        lines.append(f"- max_depth: {std_params.get('max_depth', 'N/A')}")
        lines.append(f"- num_leaves: {std_params.get('num_leaves', 'N/A')}")
        lines.append(f"- min_data_in_leaf: {std_params.get('min_data_in_leaf', 'N/A')}")
        lines.append(f"- learning_rate: {std_params.get('learning_rate', 'N/A'):.4f}")
        lines.append("")

        # MoE params (shared tree structure)
        moe_params = results["MoE"]["params"]
        lines.append("**MoE (Shared Tree Structure):**")
        lines.append(f"- num_experts: {moe_params.get('mixture_num_experts', 2)}")
        lines.append(f"- max_depth: {moe_params.get('max_depth', 'N/A')}")
        lines.append(f"- num_leaves: {moe_params.get('num_leaves', 'N/A')}")
        lines.append(f"- min_data_in_leaf: {moe_params.get('min_data_in_leaf', 'N/A')}")
        lines.append(f"- learning_rate: {moe_params.get('learning_rate', 'N/A'):.4f}")
        lines.append(f"- smoothing: {moe_params.get('mixture_r_smoothing', 'none')}")
        lines.append("")

        # MoE-PE params (per-expert tree structure)
        pe_params = results["MoE-PE"]["params"]
        num_experts = pe_params.get("mixture_num_experts", 2)
        lines.append("**MoE-PerExpert (Per-Expert Tree Structure):**")
        lines.append(f"- num_experts: {num_experts}")

        # Extract per-expert tree params
        max_depths = [pe_params.get(f"max_depth_{k}", "?") for k in range(num_experts)]
        num_leaves = [pe_params.get(f"num_leaves_{k}", "?") for k in range(num_experts)]
        min_data = [pe_params.get(f"min_data_in_leaf_{k}", "?") for k in range(num_experts)]

        # Create a table for per-expert params
        lines.append("")
        lines.append("| Expert | max_depth | num_leaves | min_data_in_leaf |")
        lines.append("|--------|-----------|------------|------------------|")
        for k in range(num_experts):
            lines.append(f"| E{k} | {max_depths[k]} | {num_leaves[k]} | {min_data[k]} |")
        lines.append("")

        lines.append(f"- learning_rate: {pe_params.get('learning_rate', 'N/A'):.4f}")
        lines.append(f"- smoothing: {pe_params.get('mixture_r_smoothing', 'none')}")
        lines.append("")

    return "\n".join(lines)


def print_final_summary(all_results: dict):
    """Print final summary table with hyperparameters."""
    print("\n" + "=" * 140)
    print("FINAL SUMMARY")
    print("=" * 140)

    # Results table with expert count
    print(
        f"\n{'Dataset':<12} {'Standard':>10} {'MoE':>10} {'MoE-PE':>10} {'Best Impr':>10} {'MoE K':>6} {'PE K':>5} {'MoE Corr':>12} {'PE Corr':>12} {'MoE Acc':>8} {'PE Acc':>8}"
    )
    print("-" * 140)

    for name, results in all_results.items():
        std_rmse = results["Standard"]["rmse"]
        moe_rmse = results["MoE"]["rmse"]
        moe_pe_rmse = results["MoE-PE"]["rmse"]
        best_rmse = min(moe_rmse, moe_pe_rmse)
        improvement = (std_rmse - best_rmse) / std_rmse * 100
        analysis_moe = results["analysis_moe"]
        analysis_pe = results["analysis_moe_pe"]

        moe_k = analysis_moe["K"]
        pe_k = analysis_pe["K"]
        moe_corr = f"{analysis_moe['expert_corr_min']:.2f}/{analysis_moe['expert_corr_max']:.2f}"
        pe_corr = f"{analysis_pe['expert_corr_min']:.2f}/{analysis_pe['expert_corr_max']:.2f}"
        moe_acc = f"{analysis_moe['regime_accuracy']:.1%}"
        pe_acc = f"{analysis_pe['regime_accuracy']:.1%}"

        print(
            f"{name:<12} {std_rmse:>10.4f} {moe_rmse:>10.4f} {moe_pe_rmse:>10.4f} {improvement:>+9.1f}% {moe_k:>6} {pe_k:>5} {moe_corr:>12} {pe_corr:>12} {moe_acc:>8} {pe_acc:>8}"
        )

    print("\n" + "-" * 140)
    print("K: Number of experts | Corr: Expert correlation (min/max) | Acc: Regime classification accuracy")
    print("=" * 140)

    # Detailed hyperparameters for each dataset
    print("\nSELECTED HYPERPARAMETERS")
    print("=" * 140)

    for name, results in all_results.items():
        print(f"\n[{name}]")

        # Standard GBDT
        std_params = results["Standard"]["params"]
        print(
            f"  Standard: depth={std_params.get('max_depth', '?')}, leaves={std_params.get('num_leaves', '?')}, "
            f"min_data={std_params.get('min_data_in_leaf', '?')}, lr={std_params.get('learning_rate', 0):.4f}"
        )

        # MoE
        moe_params = results["MoE"]["params"]
        moe_k = moe_params.get("mixture_num_experts", 2)
        print(
            f"  MoE:      K={moe_k}, depth={moe_params.get('max_depth', '?')}, leaves={moe_params.get('num_leaves', '?')}, "
            f"min_data={moe_params.get('min_data_in_leaf', '?')}, lr={moe_params.get('learning_rate', 0):.4f}, "
            f"alpha={moe_params.get('mixture_e_step_alpha', '?'):.2f}, warmup={moe_params.get('mixture_warmup_iters', '?')}"
        )

        # MoE-PE (per-expert)
        pe_params = results["MoE-PE"]["params"]
        pe_k = pe_params.get("mixture_num_experts", 2)
        depths = [pe_params.get(f"max_depth_{k}", "?") for k in range(pe_k)]
        leaves = [pe_params.get(f"num_leaves_{k}", "?") for k in range(pe_k)]
        min_data = [pe_params.get(f"min_data_in_leaf_{k}", "?") for k in range(pe_k)]
        print(
            f"  MoE-PE:   K={pe_k}, lr={pe_params.get('learning_rate', 0):.4f}, "
            f"alpha={pe_params.get('mixture_e_step_alpha', '?'):.2f}, warmup={pe_params.get('mixture_warmup_iters', '?')}"
        )
        for k in range(pe_k):
            print(f"            E{k}: depth={depths[k]}, leaves={leaves[k]}, min_data={min_data[k]}")

    print("\n" + "=" * 140)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Benchmark: Standard GBDT vs MoE GBDT")
    parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--splits", type=int, default=5, help="CV splits")
    parser.add_argument("--rounds", type=int, default=100, help="Boosting rounds")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    parser.add_argument("--no-demo", action="store_true", help="Skip regime demo visualization")
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP beeswarm visualization")
    parser.add_argument("--output-md", type=str, help="Output markdown file path (e.g., BENCHMARK.md)")
    args = parser.parse_args()

    config = BenchmarkConfig(
        n_trials=args.trials,
        n_splits=args.splits,
        num_boost_round=args.rounds,
        seed=args.seed,
    )

    print("=" * 70)
    print("Benchmark: Standard GBDT vs MoE GBDT vs MoE-PerExpert")
    print("=" * 70)
    print(f"Trials: {config.n_trials}, CV Splits: {config.n_splits}, Boost Rounds: {config.num_boost_round}")

    all_results = {}
    synthetic_data = None  # Store for regime demo

    for name, dataset_info in DATASETS.items():
        generator = dataset_info["generator"]
        params = dataset_info["params"]
        params["seed"] = config.seed

        X, y, regime = generator(**params)
        all_results[name] = run_benchmark(name, X, y, regime, config)

        # Store Synthetic data for regime demo
        if name == "Synthetic":
            synthetic_data = (X, y, regime)

    print_final_summary(all_results)

    if not args.no_viz:
        create_visualization(all_results, "examples/benchmark_results.png")

    # Regime demo on Synthetic data
    if not args.no_demo and synthetic_data is not None:
        X, y, regime = synthetic_data
        create_regime_demo(X, y, regime, config, "examples/regime_demo.png")

    # SHAP visualization on Synthetic data using optimized MoE params
    if not args.no_shap and synthetic_data is not None and "Synthetic" in all_results:
        X, y, regime = synthetic_data
        best_moe_params = all_results["Synthetic"]["MoE"]["params"]
        create_shap_visualization(X, y, config, best_moe_params, output_dir="examples")

    if args.output_md:
        md_content = generate_markdown_report(all_results, config)
        with open(args.output_md, "w") as f:
            f.write(md_content)
        print(f"\nMarkdown report saved: {args.output_md}")

    print("\nDone!")


if __name__ == "__main__":
    main()
