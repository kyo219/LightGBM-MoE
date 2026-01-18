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

    regime_score = 0.5 * X[:, 1] + 0.3 * X[:, 2] - 0.2 * X[:, 3]
    regime_true = (regime_score > 0).astype(int)

    y = np.zeros(n_samples)

    mask0 = regime_true == 0
    y[mask0] = (
        5.0 * X[mask0, 0]
        + 3.0 * X[mask0, 0] * X[mask0, 2]
        + 2.0 * np.sin(2 * X[mask0, 3])
        + 10.0
    )

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
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
            "mixture_num_experts": num_experts,
            "mixture_e_step_alpha": trial.suggest_float(
                "mixture_e_step_alpha", 0.1, 2.0
            ),
            "mixture_warmup_iters": trial.suggest_int("mixture_warmup_iters", 5, 50),
            "mixture_balance_factor": trial.suggest_int(
                "mixture_balance_factor", 2, 10
            ),
            "mixture_r_smoothing": smoothing,
        }

        if per_expert:
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
            params["num_leaves"] = trial.suggest_int("num_leaves", 8, 128)
            params["max_depth"] = trial.suggest_int("max_depth", 3, 12)
            params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 5, 100)

        if smoothing != "none":
            params["mixture_smoothing_lambda"] = trial.suggest_float(
                "mixture_smoothing_lambda", 0.1, 0.9
            )

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
    if K == 2:
        expert_corr = np.corrcoef(expert_preds[:, 0], expert_preds[:, 1])[0, 1]
    else:
        corrs = []
        for i in range(K):
            for j in range(i + 1, K):
                corrs.append(np.corrcoef(expert_preds[:, i], expert_preds[:, j])[0, 1])
        expert_corr = np.mean(corrs)
    experts_different = expert_corr < 0.99

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
        "expert_corr": expert_corr,
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
    print(
        f"Regime distribution: {(regime_true == 0).mean():.1%} / {(regime_true == 1).mean():.1%}"
    )

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
    print(f"  MoE:")
    print(f"    Gate working: {'Yes' if analysis_moe['gate_working'] else 'NO!'}")
    print(f"    Expert correlation: {analysis_moe['expert_corr']:.4f}")
    print(f"    Regime accuracy: {analysis_moe['regime_accuracy']:.1%}")
    print(f"  MoE-PE:")
    print(f"    Gate working: {'Yes' if analysis_moe_pe['gate_working'] else 'NO!'}")
    print(f"    Expert correlation: {analysis_moe_pe['expert_corr']:.4f}")
    print(f"    Regime accuracy: {analysis_moe_pe['regime_accuracy']:.1%}")

    # Summary
    std_rmse = results["Standard"]["rmse"]
    moe_rmse = results["MoE"]["rmse"]
    moe_pe_rmse = results["MoE-PE"]["rmse"]
    best_rmse = min(moe_rmse, moe_pe_rmse)
    improvement = (std_rmse - best_rmse) / std_rmse * 100

    print(f"\n  --- Summary ---")
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
            best_label = "MoE"
        else:
            analysis = results["analysis_moe_pe"]
            best_label = "MoE-PE"
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
        ax2.set_title(f"{name}: Expert Preds\n(corr={analysis['expert_corr']:.2f})")

        # Plot 3: RMSE Comparison
        ax3 = axes[i, 2]
        methods = ["Standard", "MoE", "MoE-PE"]
        rmses = [results["Standard"]["rmse"], results["MoE"]["rmse"], results["MoE-PE"]["rmse"]]
        bars = ax3.bar(methods, rmses, color=["gray", "C0", "C2"], alpha=0.7)
        ax3.set_ylabel("RMSE")
        ax3.set_title(f"{name}: RMSE Comparison")
        for bar, rmse in zip(bars, rmses):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{rmse:.3f}",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved: {output_path}")
    plt.close()


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
    lines.append("| Dataset | MoE Corr | MoE-PE Corr | MoE Regime Acc | MoE-PE Regime Acc |")
    lines.append("|---------|----------|-------------|----------------|-------------------|")

    for name, results in all_results.items():
        analysis_moe = results["analysis_moe"]
        analysis_pe = results["analysis_moe_pe"]

        moe_corr = analysis_moe["expert_corr"]
        pe_corr = analysis_pe["expert_corr"]
        moe_acc = analysis_moe["regime_accuracy"]
        pe_acc = analysis_pe["regime_accuracy"]

        lines.append(
            f"| {name} | {moe_corr:.2f} | {pe_corr:.2f} | {moe_acc:.1%} | {pe_acc:.1%} |"
        )

    lines.append("")
    lines.append("**Notes**:")
    lines.append("- **Expert Corr**: Correlation between expert predictions (lower = more differentiated, negative = opposite predictions)")
    lines.append("- **Regime Acc**: Classification accuracy of predicted regime vs true regime (with best label permutation)")
    lines.append("")

    return "\n".join(lines)


def print_final_summary(all_results: dict):
    """Print final summary table."""
    print("\n" + "=" * 105)
    print("FINAL SUMMARY")
    print("=" * 105)

    # Results table
    print(f"\n{'Dataset':<12} {'Standard':>10} {'MoE':>10} {'MoE-PE':>10} {'Best Impr':>10} {'MoE Corr':>9} {'PE Corr':>8} {'MoE Acc':>8} {'PE Acc':>8}")
    print("-" * 105)

    for name, results in all_results.items():
        std_rmse = results["Standard"]["rmse"]
        moe_rmse = results["MoE"]["rmse"]
        moe_pe_rmse = results["MoE-PE"]["rmse"]
        best_rmse = min(moe_rmse, moe_pe_rmse)
        improvement = (std_rmse - best_rmse) / std_rmse * 100
        analysis_moe = results["analysis_moe"]
        analysis_pe = results["analysis_moe_pe"]

        moe_corr = f"{analysis_moe['expert_corr']:.2f}"
        pe_corr = f"{analysis_pe['expert_corr']:.2f}"
        moe_acc = f"{analysis_moe['regime_accuracy']:.1%}"
        pe_acc = f"{analysis_pe['regime_accuracy']:.1%}"

        print(f"{name:<12} {std_rmse:>10.4f} {moe_rmse:>10.4f} {moe_pe_rmse:>10.4f} {improvement:>+9.1f}% {moe_corr:>9} {pe_corr:>8} {moe_acc:>8} {pe_acc:>8}")

    print("\n" + "=" * 105)
    print("MoE Corr / PE Corr: Correlation between Expert predictions (lower = more differentiated)")
    print("MoE Acc / PE Acc: Best regime classification accuracy with label permutation")
    print("=" * 105)


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

    for name, dataset_info in DATASETS.items():
        generator = dataset_info["generator"]
        params = dataset_info["params"]
        params["seed"] = config.seed

        X, y, regime = generator(**params)
        all_results[name] = run_benchmark(name, X, y, regime, config)

    print_final_summary(all_results)

    if not args.no_viz:
        create_visualization(all_results, "examples/benchmark_results.png")

    if args.output_md:
        md_content = generate_markdown_report(all_results, config)
        with open(args.output_md, "w") as f:
            f.write(md_content)
        print(f"\nMarkdown report saved: {args.output_md}")

    print("\nDone!")


if __name__ == "__main__":
    main()
