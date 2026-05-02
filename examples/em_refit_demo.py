"""Visual proof that v0.7 leaf-refit lets EM escape a bad ``r_init``.

The "additive-only EM" limitation written up in the README (and tracked in
issue #37) says that boosted-tree experts can only inch their predictions per
EM round, so when the gate / responsibilities lock onto a bad initial
assignment the model can't escape mid-training. The v0.7 ``mixture_refit_leaves``
flag closes that gap: the M-step re-derives every existing tree's leaf values
in closed form against the current ``r_ik`` before the next tree is appended.

This script demonstrates the difference empirically:

1. Generate a synthetic regression with two clearly-separated regimes
   (regime is a deterministic function of ``X[:, 0]``).
2. Force a "wrong" initialization by using ``mixture_init=random`` with a
   seed that lands far from the true regime structure.
3. Train two boosters with otherwise-identical configs:
       (a) baseline:  ``mixture_refit_leaves=False``  (v0.6 behavior)
       (b) refit on:  ``mixture_refit_leaves=True, decay=0.0``  (v0.7)
4. Snapshot ``model.get_responsibilities()`` each iter via callback, then
   compute Frobenius distance ||r_t − r_init|| and plot it over iters.

Expected behavior: refit-on diverges from r_init (escaping the bad basin);
refit-off stays close to r_init (stuck — the expert trees calibrated against
r_init can't be unmade). Final train / held-out RMSEs are also reported.

Output: prints a comparison table; if matplotlib is installed, also writes
a plot to ``bench_results/em_refit_demo.png``.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

import lightgbm_moe as lgb


N_TRAIN = 600
N_VALID = 200
N_FEATURES = 5
N_BOOST_ROUND = 60
WARMUP = 5
SEED = 42


def make_two_regime_data(n: int, seed: int):
    """Two regimes with opposite-sign coefficients on the same features.

    Regime is a deterministic step on ``X[:, 0]``; under correct init, MoE
    routes perfectly. Under a wrong init, EM has to flip mass between the
    two experts to recover — exactly what refit unlocks.
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


def base_params() -> dict:
    return {
        "boosting": "mixture",
        "mixture_enable": True,
        "mixture_num_experts": 2,
        # Wrong init: random partition that ignores the actual regime structure.
        # ``uniform`` would be broken by the symmetry breaker; ``random`` lands
        # on a non-uniform but mostly-wrong assignment, which is the harder
        # case the refit fix is meant to address.
        "mixture_init": "random",
        "objective": "regression",
        "num_leaves": 8,
        "learning_rate": 0.1,
        "verbose": -1,
        "num_threads": 1,
        "mixture_warmup_iters": WARMUP,
        "seed": SEED,
        "deterministic": True,
        "force_col_wise": True,
        # Variance estimation is on — it interacts with refit in a meaningful
        # way (the Bayesian-posterior r becomes consistent with the post-refit
        # leaves), so showing the demo with the v0.6 default is correct.
        "mixture_estimate_variance": True,
    }


def train_with_responsibility_log(params: dict, X_train, y_train, X_valid, y_valid):
    """Train a booster, recording r_ik snapshots at each EM iter."""
    snapshots: list[tuple[int, np.ndarray]] = []
    rmse_train: list[float] = []
    rmse_valid: list[float] = []

    def cb(env):
        # env.iteration is 0-based on the just-completed iter
        r = env.model.get_responsibilities()
        if r.size > 0:
            snapshots.append((env.iteration, r.copy()))
        # Track RMSE on the fly for a per-iter learning curve.
        pred_t = env.model.predict(X_train)
        pred_v = env.model.predict(X_valid)
        rmse_train.append(float(np.sqrt(np.mean((pred_t - y_train) ** 2))))
        rmse_valid.append(float(np.sqrt(np.mean((pred_v - y_valid) ** 2))))

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    bst = lgb.train(
        params, train_data,
        num_boost_round=N_BOOST_ROUND,
        valid_sets=[valid_data], valid_names=["valid"],
        callbacks=[cb],
    )
    return bst, snapshots, rmse_train, rmse_valid


def frob_distance_from_init(snapshots):
    """||r_t − r_init||_F per iter, normalized by ||r_init||_F + ε."""
    if not snapshots:
        return np.array([]), np.array([])
    iters = np.array([it for it, _ in snapshots])
    r_init = snapshots[0][1]
    norm_init = float(np.linalg.norm(r_init))
    dists = np.array([
        float(np.linalg.norm(r - r_init)) / max(norm_init, 1e-9)
        for _, r in snapshots
    ])
    return iters, dists


def main():
    os.makedirs("bench_results", exist_ok=True)
    X_train, y_train, _ = make_two_regime_data(N_TRAIN, seed=SEED)
    X_valid, y_valid, _ = make_two_regime_data(N_VALID, seed=SEED + 1)

    print("=" * 72)
    print("v0.7 leaf-refit demo: escape a bad r_init")
    print(f"N_train={N_TRAIN}  N_valid={N_VALID}  K=2  rounds={N_BOOST_ROUND}")
    print(f"warmup={WARMUP}  init=random  seed={SEED}")
    print("=" * 72)

    # (a) baseline v0.6 behavior
    t0 = time.perf_counter()
    bst_off, snap_off, rmse_t_off, rmse_v_off = train_with_responsibility_log(
        dict(base_params(), mixture_refit_leaves=False),
        X_train, y_train, X_valid, y_valid,
    )
    t_off = time.perf_counter() - t0

    # (b) v0.7 refit on, full replace per iter
    t0 = time.perf_counter()
    bst_on, snap_on, rmse_t_on, rmse_v_on = train_with_responsibility_log(
        dict(base_params(), mixture_refit_leaves=True,
             mixture_refit_decay_rate=0.0,
             mixture_refit_trigger="always"),
        X_train, y_train, X_valid, y_valid,
    )
    t_on = time.perf_counter() - t0

    iters_off, dist_off = frob_distance_from_init(snap_off)
    iters_on, dist_on = frob_distance_from_init(snap_on)

    print()
    print(f"{'metric':<36} {'refit=off':>12} {'refit=on':>12} {'Δ':>10}")
    print("-" * 72)
    print(f"{'final train RMSE':<36} {rmse_t_off[-1]:>12.4f} {rmse_t_on[-1]:>12.4f} "
          f"{rmse_t_on[-1]-rmse_t_off[-1]:>+10.4f}")
    print(f"{'final valid RMSE':<36} {rmse_v_off[-1]:>12.4f} {rmse_v_on[-1]:>12.4f} "
          f"{rmse_v_on[-1]-rmse_v_off[-1]:>+10.4f}")
    print(f"{'max ||r_t − r_init||_F (norm)':<36} {dist_off.max():>12.4f} "
          f"{dist_on.max():>12.4f} {dist_on.max()-dist_off.max():>+10.4f}")
    print(f"{'wall time (s)':<36} {t_off:>12.2f} {t_on:>12.2f} "
          f"{t_on-t_off:>+10.2f}")
    print("=" * 72)
    print()
    if dist_on.max() > dist_off.max() + 0.05:
        print("[OK] refit-on escaped the r_init basin — see plot for trajectory.")
    else:
        print("[WARN] refit-on did NOT escape r_init meaningfully on this seed. "
              "Try a larger num_boost_round or a different init seed.")

    # Optional plot
    try:
        import matplotlib  # noqa: F401
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(iters_off, dist_off, label="refit=off (v0.6)", lw=2)
        axes[0].plot(iters_on, dist_on, label="refit=on (v0.7)", lw=2)
        axes[0].set_xlabel("EM iteration")
        axes[0].set_ylabel("||r_t − r_init||_F  (normalized)")
        axes[0].set_title("Responsibility drift from initialization")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[1].plot(rmse_v_off, label="refit=off (v0.6)", lw=2)
        axes[1].plot(rmse_v_on, label="refit=on (v0.7)", lw=2)
        axes[1].set_xlabel("EM iteration")
        axes[1].set_ylabel("validation RMSE")
        axes[1].set_title("Held-out RMSE per iteration")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        out = "bench_results/em_refit_demo.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        print(f"plot written: {out}")
    except ImportError:
        print("(matplotlib not available — plot skipped)")


if __name__ == "__main__":
    sys.exit(main() or 0)
