"""Strong-EM sweep — does turning on hard_m_step / variance / higher diversity
let EM recover from a bad init?

We sweep a 4×4 grid of (init scheme × EM strength config) on a
feature-determined 3-regime synthetic dataset (same as
em_init_sensitivity.py). For each cell we measure:

  - Final train RMSE
  - Final mean responsibility entropy (lower = more confident)
  - ARI vs ground-truth regime (init and final)
  - "init→final unchanged" — fraction of samples whose argmax never moved

If "strong EM" actually does its job, we should see:

  - From bad inits (uniform/random), strong configs recover comparable
    ARI vs truth to good inits (kmeans/gmm).
  - Across configs, RMSE differences shrink (less init-dependence).

If we don't see that, the EM machinery in this codebase is not strong
enough to redistribute samples — the init basically determines the answer.

Output:
  bench_results/em_strength_sweep.png   (heatmaps + bar comparison)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import lightgbm_moe as lgb

OUT_DIR = Path(__file__).parent.parent / "bench_results"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Same feature-determined data
# ---------------------------------------------------------------------------

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
    return X, y, regime


# ---------------------------------------------------------------------------
# ARI (no sklearn)
# ---------------------------------------------------------------------------

def adjusted_rand_score(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    ka, kb = a.max() + 1, b.max() + 1
    cont = np.zeros((ka, kb), dtype=np.int64)
    for x, y in zip(a, b):
        cont[x, y] += 1
    comb2 = lambda v: v * (v - 1) // 2
    sij = comb2(cont).sum()
    sa = comb2(cont.sum(axis=1)).sum()
    sb = comb2(cont.sum(axis=0)).sum()
    total = comb2(n)
    if total == 0:
        return 0.0
    expected = sa * sb / total
    max_idx = (sa + sb) / 2
    if max_idx == expected:
        return 0.0
    return float((sij - expected) / (max_idx - expected))


# ---------------------------------------------------------------------------
# Configurations to sweep
# ---------------------------------------------------------------------------

INITS = ["uniform", "random", "kmeans_features", "gmm"]

CONFIGS = {
    "C0_baseline": dict(
        mixture_hard_m_step=False,
        mixture_estimate_variance=False,
        mixture_diversity_lambda=0.05,
    ),
    "C1_hard_M": dict(
        mixture_hard_m_step=True,
        mixture_estimate_variance=False,
        mixture_diversity_lambda=0.05,
    ),
    "C2_var_est": dict(
        mixture_hard_m_step=False,
        mixture_estimate_variance=True,
        mixture_diversity_lambda=0.05,
    ),
    "C3_hard+var+div": dict(
        mixture_hard_m_step=True,
        mixture_estimate_variance=True,
        mixture_diversity_lambda=0.2,
    ),
    # DA-EM (Ueda & Nakano 1998): anneal the responsibility softmax temperature
    # high → low. Mild schedule (T=2 → 1) preserves init information while still
    # softening the early E-step.
    "C4_DA-EM_mild": dict(
        mixture_hard_m_step=False,
        mixture_estimate_variance=True,
        mixture_diversity_lambda=0.05,
        mixture_e_step_temperature_init=2.0,
        mixture_e_step_temperature_final=1.0,
    ),
    # Aggressive DA-EM: T=5 → 0.5. Good for bad inits (uniform/random) but
    # forgets good inits.
    "C5_DA-EM_strong": dict(
        mixture_hard_m_step=False,
        mixture_estimate_variance=True,
        mixture_diversity_lambda=0.05,
        mixture_e_step_temperature_init=5.0,
        mixture_e_step_temperature_final=0.5,
    ),
}


def _capture_init_argmax(X, y, init_scheme, base):
    """Run a 1-iteration training to extract InitResponsibilities (during
    warmup), then read the recorder's iter-0 snapshot."""
    from lightgbm_moe import RegimeEvolutionRecorder
    rec = RegimeEvolutionRecorder(every=1, max_snapshots=2)
    params = {
        "boosting": "mixture",
        "mixture_num_experts": 3,
        "objective": "regression",
        "verbose": -1,
        "mixture_init": init_scheme,
        "mixture_warmup_iters": 5,
        "learning_rate": 0.05,
        "seed": 42,
        "deterministic": True,
        **base,
    }
    lgb.train(params, lgb.Dataset(X, label=y),
              num_boost_round=1, callbacks=[rec])
    return rec.snapshots[0][1].argmax(axis=1)


def train_one(X, y, init_scheme, cfg, n_rounds=120):
    from lightgbm_moe import RegimeEvolutionRecorder
    rec = RegimeEvolutionRecorder(every=10, max_snapshots=20)
    params = {
        "boosting": "mixture",
        "mixture_num_experts": 3,
        "objective": "regression",
        "verbose": -1,
        "mixture_init": init_scheme,
        "mixture_warmup_iters": 5,
        "learning_rate": 0.05,
        "seed": 42,
        "deterministic": True,
        **cfg,
    }
    m = lgb.train(params, lgb.Dataset(X, label=y),
                  num_boost_round=n_rounds, callbacks=[rec])
    yhat = m.predict(X)
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    init_arg = rec.snapshots[0][1].argmax(axis=1)
    final_arg = rec.snapshots[-1][1].argmax(axis=1)
    return {
        "rmse": rmse,
        "final_entropy": float(rec.mean_entropy()[-1]),
        "init_argmax": init_arg,
        "final_argmax": final_arg,
        "final_load": rec.expert_load()[-1],
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    X, y, true_regime = make_data()
    n_init = len(INITS)
    n_cfg = len(CONFIGS)

    rmse_grid = np.zeros((n_init, n_cfg))
    ari_init_grid = np.zeros((n_init, n_cfg))
    ari_final_grid = np.zeros((n_init, n_cfg))
    ent_grid = np.zeros((n_init, n_cfg))
    unchanged_grid = np.zeros((n_init, n_cfg))

    print(f"{'init':>16s}  {'config':>18s}  {'RMSE':>6s}  "
          f"{'H_fin':>6s}  {'ARI_init':>9s}  {'ARI_fin':>8s}  {'Δ ARI':>7s}  "
          f"{'unch%':>6s}")
    for i, init in enumerate(INITS):
        for j, (cname, cfg) in enumerate(CONFIGS.items()):
            r = train_one(X, y, init, cfg)
            ari_i = adjusted_rand_score(r["init_argmax"], true_regime)
            ari_f = adjusted_rand_score(r["final_argmax"], true_regime)
            unch = float((r["init_argmax"] == r["final_argmax"]).mean())

            rmse_grid[i, j] = r["rmse"]
            ari_init_grid[i, j] = ari_i
            ari_final_grid[i, j] = ari_f
            ent_grid[i, j] = r["final_entropy"]
            unchanged_grid[i, j] = unch

            print(f"{init:>16s}  {cname:>18s}  {r['rmse']:>6.3f}  "
                  f"{r['final_entropy']:>6.2f}  {ari_i:>9.3f}  {ari_f:>8.3f}  "
                  f"{ari_f - ari_i:>+7.3f}  {unch*100:>5.1f}%")

    # ---- Plot --------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    def _heatmap(ax, grid, title, fmt="{:.2f}", cmap="viridis", vmin=None, vmax=None):
        im = ax.imshow(grid, aspect="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(n_cfg))
        ax.set_xticklabels(list(CONFIGS.keys()), rotation=20, ha="right",
                           fontsize=8)
        ax.set_yticks(range(n_init))
        ax.set_yticklabels(INITS, fontsize=8)
        ax.set_title(title)
        for i in range(n_init):
            for j in range(n_cfg):
                ax.text(j, i, fmt.format(grid[i, j]),
                        ha="center", va="center",
                        color="white" if grid[i, j] < (vmax + vmin) / 2 else "black",
                        fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.7)

    _heatmap(axes[0, 0], rmse_grid,
             "Final train RMSE  (lower = better)",
             cmap="viridis_r",
             vmin=rmse_grid.min(), vmax=rmse_grid.max())
    _heatmap(axes[0, 1], ari_final_grid,
             "Final ARI vs ground truth  (1.0 = perfect)",
             cmap="viridis", vmin=0, vmax=max(0.5, float(ari_final_grid.max())))
    _heatmap(axes[1, 0], ari_final_grid - ari_init_grid,
             "Δ ARI = final − init  (positive = EM moved toward truth)",
             fmt="{:+.3f}", cmap="RdBu",
             vmin=-0.1, vmax=0.1)
    _heatmap(axes[1, 1], unchanged_grid,
             "init → final argmax unchanged  (1.0 = EM did nothing)",
             cmap="cividis", vmin=0, vmax=1)

    fig.suptitle("EM strength sweep — can hard-M / variance / diversity break "
                 "init dependence?")
    out = OUT_DIR / "em_strength_sweep.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
