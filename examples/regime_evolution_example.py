"""Demo: visualize how MoE's EM redistributes regime assignments.

Runs two scenarios that the visualization adapts to differently:

  1. **Time-series** (synthetic regime-switching series, Markov smoothing on)
     → x-axis is time, top panel is y(t).
  2. **Tabular** (HMM data with no temporal structure exploited at training)
     → x-axis is "samples sorted by final regime", top panel is per-regime y.

Each run writes a PNG you can eyeball:

  bench_results/regime_evolution_timeseries.png
  bench_results/regime_evolution_tabular.png

The interesting questions to answer from the figures:

  * Does the iter=0 snapshot (= GMM/kmeans init) already look meaningful?
    Compare it against the final snapshot — large redistribution means the
    init was off and EM did real work; small redistribution means the init
    was already close.
  * Does the entropy curve drop monotonically? If it stalls or rises late,
    something is fighting EM (over-aggressive dropout, lr-scaling, etc.).
  * Does the expert load chart show one band collapsing to 0 or 1? That's
    expert collapse — the load-balance bias didn't catch it.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np

import lightgbm_moe as lgb
from lightgbm_moe import RegimeEvolutionRecorder


OUT_DIR = Path(__file__).parent.parent / "bench_results"
OUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# 1. Time-series demo                                                         #
# --------------------------------------------------------------------------- #

def make_regime_switching_series(n=800, seed=0):
    """A toy series with three persistent regimes the gate can learn."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5)).astype(np.float64)
    # Regime persists in chunks of ~80 samples and rotates 0 → 1 → 2 → 0…
    regime = (np.arange(n) // 80) % 3
    coeffs = np.array([
        [+2.0, +0.0, -1.0, +0.5, +0.0],   # regime 0
        [-2.0, +1.0, +0.0, +0.0, +0.5],   # regime 1
        [+0.0, -1.5, +1.5, +0.0, -0.5],   # regime 2
    ])
    y = (X * coeffs[regime]).sum(axis=1) + 0.15 * rng.normal(size=n)
    return X, y, regime


def run_timeseries():
    X, y, true_regime = make_regime_switching_series()
    rec = RegimeEvolutionRecorder(every=5, max_snapshots=20)
    params = {
        "boosting": "mixture",
        "mixture_num_experts": 3,
        "objective": "regression",
        "verbose": -1,
        "mixture_init": "kmeans_features",
        "mixture_warmup_iters": 5,
        "mixture_r_smoothing": "markov",
        "mixture_smoothing_lambda": 0.3,
        "mixture_diversity_lambda": 0.05,
        "learning_rate": 0.05,
    }
    lgb.train(
        params, lgb.Dataset(X, label=y),
        num_boost_round=120, callbacks=[rec],
    )

    fig = rec.plot(y=y, title="Regime evolution (time-series, Markov on)")
    out = OUT_DIR / "regime_evolution_timeseries.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  wrote {out}")
    print(f"  init→final flip rate sum: {rec.flip_rate().sum():.3f} "
          f"({rec.num_snapshots} snapshots)")


# --------------------------------------------------------------------------- #
# 2. Tabular demo                                                             #
# --------------------------------------------------------------------------- #

def make_tabular_clusters(n=600, seed=1):
    """Three Gaussian clusters in feature space; y depends on cluster."""
    rng = np.random.default_rng(seed)
    centers = np.array([
        [+2.0, +0.0, +0.0],
        [+0.0, +2.0, +0.0],
        [-1.5, -1.5, +1.0],
    ])
    cluster = rng.integers(0, 3, size=n)
    X = centers[cluster] + 0.4 * rng.normal(size=(n, 3))
    coeffs = np.array([+3.0, -3.0, +1.5])
    y = coeffs[cluster] * X[:, 0] + 0.2 * rng.normal(size=n)
    # Shuffle so row order carries no time info.
    perm = rng.permutation(n)
    return X[perm], y[perm], cluster[perm]


def run_tabular():
    X, y, true_cluster = make_tabular_clusters()
    rec = RegimeEvolutionRecorder(every=5, max_snapshots=20)
    params = {
        "boosting": "mixture",
        "mixture_num_experts": 3,
        "objective": "regression",
        "verbose": -1,
        "mixture_init": "kmeans_features",
        "mixture_warmup_iters": 5,
        "mixture_diversity_lambda": 0.05,
        "learning_rate": 0.05,
    }
    lgb.train(
        params, lgb.Dataset(X, label=y),
        num_boost_round=120, callbacks=[rec],
    )

    fig = rec.plot(y=y, title="Regime evolution (tabular, sorted by final regime)")
    out = OUT_DIR / "regime_evolution_tabular.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("[1/2] Time-series scenario...")
    run_timeseries()
    print("[2/2] Tabular scenario...")
    run_tabular()
