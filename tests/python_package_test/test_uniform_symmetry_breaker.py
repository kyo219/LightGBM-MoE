# coding: utf-8
"""Tests for the uniform-r symmetry breaker.

Background — why this matters:

  Without the symmetry breaker, ``mixture_init=uniform`` leaves the
  responsibility matrix at exactly ``1/K`` for every (i, k). Identical
  responsibility weights produce identical gradients across experts, which
  produce identical trees, which produce identical predictions, which the
  E-step then maps back to uniform r. Empirically (see
  ``examples/em_init_sensitivity.py``) this fixed-point trap survived every
  combination of ``mixture_hard_m_step`` / ``mixture_estimate_variance`` /
  ``mixture_diversity_lambda`` we tried.

  The breaker injects a small deterministic per-(sample, expert) sinusoidal
  perturbation when r is detected uniform within ``1e-6``. It's a no-op when
  r is already non-uniform (kmeans/gmm/quantile/tree-hierarchical etc.).
"""

from __future__ import annotations

import numpy as np

import lightgbm_moe as lgb
from lightgbm_moe import RegimeEvolutionRecorder


def _toy_data(n=200, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X[:, 0] * 2.0 - X[:, 1] + 0.1 * rng.normal(size=n)
    return X, y


class TestUniformSymmetryBreaker:
    def test_uniform_init_produces_non_uniform_r(self):
        """With the breaker, `mixture_init=uniform` no longer leaves r at 1/K."""
        X, y = _toy_data()
        rec = RegimeEvolutionRecorder(every=1, max_snapshots=2)
        lgb.train(
            {"boosting": "mixture", "mixture_num_experts": 3,
             "objective": "regression", "verbose": -1,
             "mixture_init": "uniform",
             "mixture_warmup_iters": 2},
            lgb.Dataset(X, label=y),
            num_boost_round=3, callbacks=[rec],
        )
        r0 = rec.snapshots[0][1]
        per_sample_max = r0.max(axis=1)
        # Sinusoidal eps=0.05 → max r should be ~ 1/3 + ~0.033 after renorm.
        # Just check it's clearly above uniform.
        assert per_sample_max.max() > 1.0 / 3 + 1e-3, (
            "Uniform-init symmetry breaker did not perturb r "
            f"(max(per-sample max r) = {per_sample_max.max()})"
        )

    def test_breaker_preserves_simplex(self):
        """Each row must still sum to 1.0 after perturbation+renorm."""
        X, y = _toy_data()
        rec = RegimeEvolutionRecorder(every=1, max_snapshots=2)
        lgb.train(
            {"boosting": "mixture", "mixture_num_experts": 4,
             "objective": "regression", "verbose": -1,
             "mixture_init": "uniform",
             "mixture_warmup_iters": 2},
            lgb.Dataset(X, label=y),
            num_boost_round=2, callbacks=[rec],
        )
        r0 = rec.snapshots[0][1]
        np.testing.assert_allclose(r0.sum(axis=1), 1.0, atol=1e-9)
        assert (r0 >= 0).all() and (r0 <= 1).all()

    def test_breaker_no_op_for_non_uniform_init(self):
        """The breaker is gated on detection — it must not perturb a real
        init like kmeans_features whose r already varies across rows."""
        X, y = _toy_data()
        rec = RegimeEvolutionRecorder(every=1, max_snapshots=2)
        lgb.train(
            {"boosting": "mixture", "mixture_num_experts": 3,
             "objective": "regression", "verbose": -1,
             "mixture_init": "kmeans_features",
             "mixture_warmup_iters": 2},
            lgb.Dataset(X, label=y),
            num_boost_round=2, callbacks=[rec],
        )
        # kmeans_features assigns each row to one cluster (one-hot-ish).
        # Maximum-entropy under K=3 is ≈1.099. We check that init r is far
        # from uniform — i.e. the breaker did not over-write it.
        r0 = rec.snapshots[0][1]
        per_row_entropy = -(np.clip(r0, 1e-12, 1.0)
                            * np.log(np.clip(r0, 1e-12, 1.0))).sum(axis=1)
        # If the breaker overwrote a kmeans init, mean entropy would be
        # near max (~1.099). It should be much lower.
        assert per_row_entropy.mean() < 0.5, (
            f"breaker may have overwritten non-uniform init "
            f"(mean entropy = {per_row_entropy.mean():.3f})"
        )
