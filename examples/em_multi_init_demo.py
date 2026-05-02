"""Multi-init restart on the feature-determined 3-regime synthetic.

The init sensitivity sweep showed that final RMSE varies by 30%+ across
init schemes — picking a single init is gambling. ``train_multi_init``
runs N attempts in series and returns the best.

Output:
  bench_results/em_multi_init_demo.txt   (per-attempt summary)
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import lightgbm_moe as lgb
from lightgbm_moe import RegimeEvolutionRecorder, train_multi_init

OUT_DIR = Path(__file__).parent.parent / "bench_results"
OUT_DIR.mkdir(exist_ok=True)


def make_data(n=8000, seed=0):
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
    import time

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

    inits = ["uniform", "random", "quantile", "kmeans_features", "gmm"]

    # ----------------------------------------------------------------------- #
    # Sequential baseline — sanity check + reference timing.                  #
    # ----------------------------------------------------------------------- #
    print(f">>> Sequential (n_jobs=1, no prescreen): 5 × full  (n={X.shape[0]}, rounds=300)")
    t0 = time.time()
    res_seq = train_multi_init(
        base_params, (X, y),
        num_boost_round=300, n_inits=5, init_schemes=inits,
        score_data=(X, y), verbose=True,
    )
    seq_wall = time.time() - t0
    print(res_seq.summary_table())
    print(f"sequential wall = {seq_wall:.2f}s  best RMSE = {res_seq.best_trial.score:.4f}\n")

    # ----------------------------------------------------------------------- #
    # Parallel — A: just n_jobs                                               #
    # ----------------------------------------------------------------------- #
    print(">>> Parallel (n_jobs=4, no prescreen): 5 × full in subprocesses")
    t0 = time.time()
    res_par = train_multi_init(
        base_params, (X, y),
        num_boost_round=300, n_inits=5, init_schemes=inits,
        score_data=(X, y), n_jobs=4, verbose=True,
    )
    par_wall = time.time() - t0
    print(f"parallel wall   = {par_wall:.2f}s  best RMSE = {res_par.best_trial.score:.4f}")
    print(f"speedup vs seq  = {seq_wall / par_wall:.2f}×\n")

    # ----------------------------------------------------------------------- #
    # Prescreen — C: cheap pass + full retrain on top K                       #
    # ----------------------------------------------------------------------- #
    print(">>> Prescreen + parallel (n_jobs=4, prescreen=40 rounds, keep=2)")
    t0 = time.time()
    res_pre = train_multi_init(
        base_params, (X, y),
        num_boost_round=300, n_inits=5, init_schemes=inits,
        score_data=(X, y), n_jobs=4,
        prescreen_rounds=40, prescreen_keep=2,
        verbose=True,
    )
    pre_wall = time.time() - t0
    print(f"prescreen wall  = {pre_wall:.2f}s  best RMSE = {res_pre.best_trial.score:.4f}")
    print(f"speedup vs seq  = {seq_wall / pre_wall:.2f}×")

    # ----------------------------------------------------------------------- #
    # Verdict                                                                 #
    # ----------------------------------------------------------------------- #
    best_seq = res_seq.best_trial
    worst_seq = max(res_seq.trials, key=lambda t: t.score)
    print()
    print("=" * 60)
    print(f"Best init found      : {best_seq.init_scheme} (RMSE {best_seq.score:.4f})")
    print(f"Worst init avoided   : {worst_seq.init_scheme} (RMSE {worst_seq.score:.4f})")
    print(f"Gap (cost of bad pick): "
          f"{(worst_seq.score - best_seq.score) / best_seq.score * 100:.1f}%")
    print()
    print(f"Wall-clock summary on {os.cpu_count()} cores:")
    print(f"  sequential (5×full)               : {seq_wall:.2f}s  baseline")
    print(f"  parallel n_jobs=4 (5×full)        : {par_wall:.2f}s  "
          f"({seq_wall/par_wall:.1f}× faster)")
    print(f"  parallel + prescreen=20 keep=2    : {pre_wall:.2f}s  "
          f"({seq_wall/pre_wall:.1f}× faster)")

    out = OUT_DIR / "em_multi_init_demo.txt"
    with out.open("w") as f:
        f.write("Sequential\n")
        f.write(res_seq.summary_table())
        f.write("\n\nParallel n_jobs=4\n")
        f.write(res_par.summary_table())
        f.write("\n\nPrescreen + parallel\n")
        f.write(res_pre.summary_table())
        f.write(f"\n\nWall: seq={seq_wall:.2f}s  par={par_wall:.2f}s  pre={pre_wall:.2f}s\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
