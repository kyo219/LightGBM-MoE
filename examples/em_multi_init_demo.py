"""Multi-init restart on the feature-determined 3-regime synthetic.

The init sensitivity sweep showed that final RMSE varies by 30%+ across
init schemes — picking a single init is gambling. ``train_multi_init``
runs N attempts in series and returns the best.

Output:
  bench_results/em_multi_init_demo.txt   (per-attempt summary)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import lightgbm_moe as lgb
from lightgbm_moe import RegimeEvolutionRecorder, train_multi_init

OUT_DIR = Path(__file__).parent.parent / "bench_results"
OUT_DIR.mkdir(exist_ok=True)


def make_data(n=800, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5)).astype(np.float64)
    score = 0.7 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2]
    thr = np.quantile(score, [1.0 / 3, 2.0 / 3])
    regime = np.searchsorted(thr, score)
    coeffs = np.array([
        [+2.0, +0.0, -1.0, +0.5, +0.0],
        [-2.0, +1.0, +0.0, +0.0, +0.5],
        [+0.0, -1.5, +1.5, +0.0, -0.5],
    ])
    y = (X * coeffs[regime]).sum(axis=1) + 0.15 * rng.normal(size=n)
    return X, y


def main():
    X, y = make_data()

    base_params = {
        "boosting": "mixture",
        "mixture_num_experts": 3,
        "objective": "regression",
        "verbose": -1,
        "mixture_warmup_iters": 5,
        "mixture_diversity_lambda": 0.05,
        "learning_rate": 0.05,
    }

    print("Running 5 attempts with varied init schemes…")
    res = train_multi_init(
        base_params,
        lgb.Dataset(X, label=y),
        num_boost_round=120,
        n_inits=5,
        init_schemes=["uniform", "random", "quantile", "kmeans_features", "gmm"],
        score_data=(X, y),
        verbose=True,
    )
    print()
    print(res.summary_table())

    # Compare best vs worst
    best, worst = res.best_trial, max(res.trials, key=lambda t: t.score)
    print()
    print(f"Best:  {best.init_scheme:>16s}  RMSE={best.score:.4f}")
    print(f"Worst: {worst.init_scheme:>16s}  RMSE={worst.score:.4f}")
    print(f"Gap: {(worst.score - best.score) / best.score * 100:.1f}% worse "
          "if you'd picked the wrong init.")

    out = OUT_DIR / "em_multi_init_demo.txt"
    with out.open("w") as f:
        f.write(res.summary_table())
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
