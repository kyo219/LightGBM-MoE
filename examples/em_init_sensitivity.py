"""Init sensitivity probe — does EM converge to the same place from different starts?

Hypotheses we're trying to discriminate:

    A. EM is healthy: GMM init was already near a local optimum, so EM
       polished it. From a *bad* init (uniform / random) EM should still find
       a comparable solution — same final RMSE, similar regime structure
       (high pairwise ARI between final argmaxes after label permutation).
    B. EM is weak: bad inits stay bad — final RMSE degrades sharply,
       responsibilities never sharpen, expert load collapses to whoever
       got lucky.
    C. Data property: there's no alternative regime structure for EM to
       discover, so init = data clustering ≈ final regardless. We can't
       fully rule this out from one dataset; flag it as ambiguous.

This script trains the same model with each init scheme and reports:
  - final train RMSE
  - final mean responsibility entropy (lower = more confident)
  - expert load (collapse if any expert ≈ 0% or ≈ 100%)
  - pairwise ARI between final argmaxes across init schemes
  - iter-0 vs iter-final argmax overlap (how much EM moved samples)

Side-by-side regime tapes are written to
  bench_results/em_init_sensitivity.png

Reading the output:
  * If RMSE rows are within ~1% of each other AND pairwise ARI is mostly
    high (>0.7) → hypothesis A. EM converges to the same neighborhood.
  * If uniform/random RMSE is much worse than kmeans/gmm → hypothesis B.
  * If RMSE is similar but ARI is low → either A with lots of label
    symmetry, or C (EM doesn't really care which clustering it picks).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import lightgbm_moe as lgb
from lightgbm_moe import RegimeEvolutionRecorder

OUT_DIR = Path(__file__).parent.parent / "bench_results"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Data — same regime-switching synthetic as regime_evolution_example.py
# ---------------------------------------------------------------------------

def make_data(n=800, seed=0):
    """Feature-determined regime: any feature-aware init *should* be able to
    recover the truth. So if some inits fail to converge to the true regime
    here, that's a real EM weakness — not a "data has no recoverable
    structure from features" excuse.

    Regime is `sign(linear combination of X) → 3 buckets`. Conditional on
    regime, y is a different polynomial of X. Identical to the spirit of
    the `synthetic` dataset in benchmark.py (Microsoft LightGBM-style
    MoE-ideal case).
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5)).astype(np.float64)
    score = 0.7 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2]
    # 3 regimes via feature-driven thresholds (terciles of `score`).
    thr = np.quantile(score, [1.0 / 3, 2.0 / 3])
    regime = np.searchsorted(thr, score)  # 0, 1, or 2
    coeffs = np.array([
        [+2.0, +0.0, -1.0, +0.5, +0.0],
        [-2.0, +1.0, +0.0, +0.0, +0.5],
        [+0.0, -1.5, +1.5, +0.0, -0.5],
    ])
    y = (X * coeffs[regime]).sum(axis=1) + 0.15 * rng.normal(size=n)
    return X, y, regime


# ---------------------------------------------------------------------------
# ARI (no sklearn dependency)
# ---------------------------------------------------------------------------

def adjusted_rand_score(a, b):
    """Hubert-Arabie ARI from contingency table — minimal numpy impl."""
    a = np.asarray(a)
    b = np.asarray(b)
    n = len(a)
    ka = a.max() + 1
    kb = b.max() + 1
    cont = np.zeros((ka, kb), dtype=np.int64)
    for x, y in zip(a, b):
        cont[x, y] += 1

    def comb2(v):
        return v * (v - 1) // 2

    sum_ij = comb2(cont).sum()
    sum_a = comb2(cont.sum(axis=1)).sum()
    sum_b = comb2(cont.sum(axis=0)).sum()
    total = comb2(n)
    expected = sum_a * sum_b / total if total > 0 else 0.0
    max_idx = (sum_a + sum_b) / 2
    if max_idx == expected:
        return 0.0
    return float((sum_ij - expected) / (max_idx - expected))


# ---------------------------------------------------------------------------
# One training run
# ---------------------------------------------------------------------------

def train_one(X, y, init_scheme, n_rounds=120, seed=42):
    rec = RegimeEvolutionRecorder(every=5, max_snapshots=25)
    params = {
        "boosting": "mixture",
        "mixture_num_experts": 3,
        "objective": "regression",
        "verbose": -1,
        "mixture_init": init_scheme,
        "mixture_warmup_iters": 5,
        "mixture_diversity_lambda": 0.05,
        "learning_rate": 0.05,
        "seed": seed,
        "deterministic": True,
    }
    m = lgb.train(params, lgb.Dataset(X, label=y),
                  num_boost_round=n_rounds, callbacks=[rec])
    yhat = m.predict(X)
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))

    # Snapshots: r at iter 0 (init) and iter-final
    init_r = rec.snapshots[0][1]
    final_r = rec.snapshots[-1][1]
    init_arg = init_r.argmax(axis=1)
    final_arg = final_r.argmax(axis=1)
    final_entropy = float(rec.mean_entropy()[-1])
    final_load = rec.expert_load()[-1]
    init_to_final_unchanged = float((init_arg == final_arg).mean())

    return {
        "init": init_scheme,
        "rmse": rmse,
        "final_entropy": final_entropy,
        "final_load": final_load,
        "init_argmax": init_arg,
        "final_argmax": final_arg,
        "init_unchanged": init_to_final_unchanged,
        "recorder": rec,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    INITS = ["uniform", "random", "quantile", "kmeans_features", "gmm"]
    X, y, true_regime = make_data()

    print("Training with each init scheme...")
    results = []
    for init in INITS:
        try:
            r = train_one(X, y, init)
            results.append(r)
            print(f"  {init:>16s}  RMSE={r['rmse']:.4f}  "
                  f"final_entropy={r['final_entropy']:.3f}  "
                  f"load=[{', '.join(f'{l:.2f}' for l in r['final_load'])}]  "
                  f"init→final unchanged={r['init_unchanged']*100:.1f}%")
        except Exception as exc:
            print(f"  {init:>16s}  FAILED: {exc}")

    if not results:
        print("All inits failed.")
        return

    # Pairwise ARI between final argmaxes
    print("\nPairwise ARI between final argmaxes (1.0 = same clustering up to relabel)")
    n = len(results)
    print(f"{'':>16s}  " + "  ".join(f"{r['init'][:8]:>8s}" for r in results))
    for i, ri in enumerate(results):
        row = [f"{ri['init']:>16s}"]
        for j, rj in enumerate(results):
            if i == j:
                row.append(f"{'-':>8s}")
            else:
                ari = adjusted_rand_score(ri["final_argmax"], rj["final_argmax"])
                row.append(f"{ari:>8.3f}")
        print("  ".join(row))

    # ARI vs ground truth
    print("\nARI vs true_regime (ground truth, 1.0 = perfect recovery)")
    for r in results:
        ari = adjusted_rand_score(r["final_argmax"], true_regime)
        ari_init = adjusted_rand_score(r["init_argmax"], true_regime)
        print(f"  {r['init']:>16s}  init={ari_init:.3f}  final={ari:.3f}  "
              f"Δ={ari-ari_init:+.3f}")

    # ---- Plot: side-by-side regime tapes ---------------------------------
    fig, axes = plt.subplots(len(results), 1,
                             figsize=(11, 1.6 * len(results)),
                             sharex=True, constrained_layout=True)
    if len(results) == 1:
        axes = [axes]

    K = 3
    cmap = plt.get_cmap("tab10", K)
    from matplotlib.colors import BoundaryNorm, ListedColormap
    listed = ListedColormap(cmap(np.arange(K)))
    norm = BoundaryNorm(np.arange(K + 1) - 0.5, ncolors=K)

    for ax, r in zip(axes, results):
        am = r["recorder"].regime_argmax()
        ax.imshow(am, aspect="auto", interpolation="nearest",
                  cmap=listed, norm=norm,
                  extent=(0, am.shape[1], am.shape[0] - 0.5, -0.5))
        ax.set_title(f"{r['init']}  RMSE={r['rmse']:.3f}  "
                     f"H_final={r['final_entropy']:.2f}  "
                     f"unchanged={r['init_unchanged']*100:.0f}%",
                     fontsize=9)
        ax.set_yticks([0, am.shape[0] - 1])
        ax.set_yticklabels([f"iter={r['recorder'].iterations[0]}",
                            f"iter={r['recorder'].iterations[-1]}"],
                           fontsize=8)
    axes[-1].set_xlabel("time / sample index")
    fig.suptitle("EM init sensitivity — final RMSE / regime tape per init")

    out = OUT_DIR / "em_init_sensitivity.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
