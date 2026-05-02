"""Visualize how leaf-refit changes the EM regime trajectory.

Trains four boosters on the same synthetic two-regime regression, varying
only `(mixture_init, mixture_refit_leaves)`:

  1. init=random, refit=off
  2. init=random, refit=on  (always trigger, decay=0)
  3. init=gmm,    refit=off
  4. init=gmm,    refit=on  (always trigger, decay=0)

For each run, ``RegimeEvolutionRecorder`` (PR #31) snapshots ``r_ik`` every
few iters and renders the standard 4-panel diagnostic. The four resulting
PNGs sit side-by-side at ``bench_results/em_refit_regime_evolution_*.png``,
showing the EM trajectory under each config.

Gate type is fixed at ``gbdt`` (refit is force-disabled for ``leaf_reuse``;
this script uses gbdt to keep the comparison clean).
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python-package"))

import lightgbm_moe as lgb  # noqa: E402
from lightgbm_moe import RegimeEvolutionRecorder  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


N_TRAIN = 1500
N_FEATURES = 5
NUM_BOOST_ROUND = 80
WARMUP = 5
SEED = 42


def make_synthetic_two_regime(n: int, seed: int):
    """Same regime structure as `examples/em_refit_demo.py`:
    regime is a deterministic function of X[:, 0]; the two regimes use
    opposite-sign coefficients on the same features.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, N_FEATURES)
    regime = (X[:, 0] > 0).astype(int)
    y = np.where(
        regime == 0,
        +5 * X[:, 1] - 2 * X[:, 2],
        -5 * X[:, 1] + 2 * X[:, 2],
    ) + 0.3 * rng.randn(n)
    return X, y, regime


def common_params() -> dict:
    return {
        "boosting": "mixture",
        "mixture_enable": True,
        "mixture_num_experts": 2,
        "mixture_gate_type": "gbdt",
        "mixture_estimate_variance": True,
        "mixture_warmup_iters": WARMUP,
        "mixture_diversity_lambda": 0.1,
        "objective": "regression",
        "num_leaves": 16,
        "learning_rate": 0.1,
        "verbose": -1,
        "num_threads": 1,
        "seed": SEED,
        "deterministic": True,
        "force_col_wise": True,
    }


def run_one(name: str, init: str, refit: bool, X, y, out_dir: str) -> tuple[float, float]:
    """Train + record + plot. Returns (final train RMSE, final ||r-r_init||_F norm)."""
    params = dict(common_params(), mixture_init=init)
    if refit:
        params["mixture_refit_leaves"] = True
        params["mixture_refit_decay_rate"] = 0.0
        params["mixture_refit_trigger"] = "always"
    rec = RegimeEvolutionRecorder(every=2, max_snapshots=40)
    bst = lgb.train(params, lgb.Dataset(X, label=y),
                    num_boost_round=NUM_BOOST_ROUND, callbacks=[rec])
    pred = bst.predict(X)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))

    # Drift metric: ||r_t - r_init|| / ||r_init||
    r0 = rec.snapshots[0][1]
    rT = rec.snapshots[-1][1]
    drift = float(np.linalg.norm(rT - r0) / max(np.linalg.norm(r0), 1e-9))

    fig = rec.plot(
        y=y,
        params=params,
        title=f"{name}  (final RMSE = {rmse:.3f},  ||rT−r0|| = {drift:.3f})",
        figsize=(11, 7),
    )
    out = os.path.join(out_dir, f"em_refit_regime_evolution_{name}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{name}]  RMSE={rmse:.4f}  drift={drift:.4f}  → {out}")
    return rmse, drift


def main():
    out_dir = "bench_results"
    os.makedirs(out_dir, exist_ok=True)

    X, y, regime = make_synthetic_two_regime(N_TRAIN, seed=SEED)
    print(f"synthetic two-regime  N={N_TRAIN}  K=2  rounds={NUM_BOOST_ROUND}")
    print(f"true regime balance: {np.bincount(regime)}")
    print()

    cfgs = [
        ("init-random_refit-off", "random", False),
        ("init-random_refit-on",  "random", True),
        ("init-gmm_refit-off",    "gmm",    False),
        ("init-gmm_refit-on",     "gmm",    True),
    ]
    results = []
    for name, init, refit in cfgs:
        results.append((name, *run_one(name, init, refit, X, y, out_dir)))

    print()
    print(f"{'config':<28} {'final RMSE':>12} {'drift ||rT-r0||':>18}")
    print("-" * 60)
    for name, rmse, drift in results:
        print(f"{name:<28} {rmse:>12.4f} {drift:>18.4f}")


if __name__ == "__main__":
    sys.exit(main() or 0)
