#!/usr/bin/env python
# coding: utf-8
"""
Visualization: MoE Regime Prediction on Time Series

Creates a publication-quality plot showing:
- Actual vs Predicted values
- Background colored by predicted regime
- Test set only (last 20% of data)

Uses the same synthetic data from benchmark_moe_vs_standard.py
"""

import sys
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import lightgbm_moe as lgb

sys.path.insert(0, "examples")

# Import from benchmark script
from benchmark_moe_vs_standard import generate_synthetic_data

warnings.filterwarnings("ignore")

# Style settings
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14


def main():
    print("Generating synthetic regime-switching data...")
    X, y, regime_true = generate_synthetic_data(n_samples=600, noise_level=0.5, seed=42)
    t = np.arange(len(y))  # Time index

    # Train/Test split (80/20)
    split_idx = int(len(y) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    t_test = t[split_idx:]
    regime_true_test = regime_true[split_idx:]

    # Train MoE model
    print("Training MoE model...")
    params = {
        "boosting": "mixture",
        "objective": "regression",
        "mixture_num_experts": 2,
        "mixture_e_step_alpha": 1.0,
        "mixture_balance_factor": 5,
        "mixture_r_smoothing": "ema",
        "mixture_smoothing_lambda": 0.3,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "verbose": -1,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=100)

    # Predictions on test set
    y_pred = model.predict(X_test)
    regime_pred = model.predict_regime(X_test)
    regime_proba = model.predict_regime_proba(X_test)

    # Calculate metrics
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

    # Check if regime labels are inverted (expert assignment is arbitrary)
    acc_normal = np.mean(regime_pred == regime_true_test)
    acc_inverted = np.mean((1 - regime_pred) == regime_true_test)

    if acc_inverted > acc_normal:
        # Invert the predictions to match true labels
        regime_pred = 1 - regime_pred
        regime_proba = regime_proba[:, ::-1]  # Swap columns
        regime_acc = acc_inverted * 100
        print("(Regime labels inverted to match true labels)")
    else:
        regime_acc = acc_normal * 100

    print(f"Test RMSE: {rmse:.4f}")
    print(f"Regime Accuracy: {regime_acc:.1f}%")

    # Create visualization
    print("Creating visualization...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])

    # Colors for regimes
    colors = ["#3498db", "#e74c3c"]  # Blue for regime 0, Red for regime 1
    color_names = ["Regime 0 (Bull)", "Regime 1 (Bear)"]

    # === Top plot: Actual vs Predicted with regime background ===
    ax1 = axes[0]

    # Fill background with regime colors
    for i in range(len(t_test) - 1):
        regime = regime_pred[i]
        ax1.axvspan(t_test[i], t_test[i + 1], alpha=0.2, color=colors[regime], linewidth=0)

    # Plot actual and predicted
    ax1.plot(t_test, y_test, "o-", color="#2c3e50", markersize=4, linewidth=1.5, label="Actual", alpha=0.8)
    ax1.plot(t_test, y_pred, "s--", color="#27ae60", markersize=4, linewidth=1.5, label="Predicted", alpha=0.8)

    # Add legend for regimes
    regime_patches = [mpatches.Patch(color=colors[i], alpha=0.3, label=color_names[i]) for i in range(2)]

    # Combine legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles1 + regime_patches, loc="upper left", framealpha=0.9, ncol=2)

    ax1.set_ylabel("Value")
    ax1.set_title(
        f"MoE Regime-Switching Prediction (Test Set)\nRMSE: {rmse:.4f} | Regime Accuracy: {regime_acc:.1f}%",
        fontweight="bold",
    )
    ax1.set_xlim(t_test[0], t_test[-1])

    # === Bottom plot: Regime probability ===
    ax2 = axes[1]

    # Stacked area for regime probabilities
    ax2.fill_between(t_test, 0, regime_proba[:, 0], color=colors[0], alpha=0.6, label="P(Regime 0)")
    ax2.fill_between(t_test, regime_proba[:, 0], 1, color=colors[1], alpha=0.6, label="P(Regime 1)")

    # Add true regime markers
    for i, (ti, r_true) in enumerate(zip(t_test, regime_true_test)):
        if i % 5 == 0:  # Plot every 5th point
            marker = "v" if r_true == 0 else "^"
            ax2.scatter(ti, 0.5, marker=marker, color="black", s=30, zorder=5)

    ax2.set_ylabel("Probability")
    ax2.set_xlabel("Time")
    ax2.set_ylim(0, 1)
    ax2.set_xlim(t_test[0], t_test[-1])
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.set_title("Gate Probability (▼=True Regime 0, ▲=True Regime 1)", fontsize=11)

    plt.tight_layout()

    # Save
    output_path = "examples/regime_switching_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved: {output_path}")

    plt.close()

    print("Done!")


if __name__ == "__main__":
    main()
