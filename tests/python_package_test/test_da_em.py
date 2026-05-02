# coding: utf-8
"""Tests for the DA-EM responsibility-softmax annealing and the matching
uniform-symmetry-breaker that ships alongside it.

Background — why these matter:

  Without the symmetry breaker, ``mixture_init=uniform`` leaves the
  responsibility matrix at exactly ``1/K`` for every (i, k). Identical
  responsibility weights produce identical gradients across experts, which
  produce identical trees, which produce identical predictions, which the
  E-step then maps back to uniform r. Empirically (see
  ``examples/em_init_sensitivity.py``) this fixed-point trap survived every
  combination of ``mixture_hard_m_step`` / ``mixture_estimate_variance`` /
  ``mixture_diversity_lambda`` we tried.

  Without DA-EM, EM is heavily init-dependent: across init schemes the
  pairwise ARI of final argmaxes was < 0.15 on a feature-determined 3-regime
  synthetic, indicating each init lands in its own local optimum and stays
  there. Annealing the responsibility softmax temperature high → low (DA-EM,
  Ueda & Nakano 1998) lets EM walk past those local optima during the soft
  early phase.
"""

from __future__ import annotations

import numpy as np
import pytest

import lightgbm_moe as lgb
from lightgbm_moe import RegimeEvolutionRecorder


def _toy_data(n=200, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X[:, 0] * 2.0 - X[:, 1] + 0.1 * rng.normal(size=n)
    return X, y


# --------------------------------------------------------------------------- #
# Uniform symmetry breaker                                                    #
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# DA-EM responsibility-softmax annealing                                      #
# --------------------------------------------------------------------------- #

def _train_with_t(t_init, t_final, init="kmeans_features", n_rounds=10):
    X, y = _toy_data()
    rec = RegimeEvolutionRecorder(every=2)
    m = lgb.train(
        {"boosting": "mixture", "mixture_num_experts": 3,
         "objective": "regression", "verbose": -1,
         "mixture_init": init,
         "mixture_warmup_iters": 2,
         "mixture_e_step_temperature_init": t_init,
         "mixture_e_step_temperature_final": t_final,
         "seed": 42, "deterministic": True},
        lgb.Dataset(X, label=y),
        num_boost_round=n_rounds, callbacks=[rec],
    )
    return m, rec


class TestDAEMAnnealing:
    def test_default_is_no_annealing(self):
        """T_init = T_final = 1.0 (default) reproduces standard EM exactly,
        so two runs with identical seeds must yield identical responsibilities."""
        _, rec1 = _train_with_t(1.0, 1.0)
        _, rec2 = _train_with_t(1.0, 1.0)
        np.testing.assert_allclose(
            rec1.snapshots[-1][1], rec2.snapshots[-1][1], atol=1e-12,
        )

    def test_annealing_changes_outcome(self):
        """Turning DA-EM on must change the trained model (otherwise the
        params are silently ignored — the bug we hit during development
        when config_auto.cpp wasn't regenerated)."""
        _, rec_off = _train_with_t(1.0, 1.0)
        _, rec_on = _train_with_t(5.0, 0.5)
        diff = np.abs(
            rec_off.snapshots[-1][1] - rec_on.snapshots[-1][1]
        ).max()
        assert diff > 1e-3, (
            f"DA-EM (T=5→0.5) produced an identical model to T=1→1; "
            f"max responsibility diff was {diff:.2e}"
        )

    def test_high_t_softens_responsibilities(self):
        """A constant high temperature should produce softer r (higher
        entropy) than the default at the same iteration."""
        _, rec_T1 = _train_with_t(1.0, 1.0)
        _, rec_Thi = _train_with_t(10.0, 10.0)
        # Entropy at the final captured snapshot.
        ent_T1 = rec_T1.mean_entropy()[-1]
        ent_Thi = rec_Thi.mean_entropy()[-1]
        assert ent_Thi > ent_T1, (
            f"high T_em should produce softer r: ent(T=10) = {ent_Thi:.3f}, "
            f"ent(T=1) = {ent_T1:.3f}"
        )

    def test_temperature_must_be_positive(self):
        """The auto-generated config check should reject T_em ≤ 0."""
        X, y = _toy_data()
        with pytest.raises(lgb.basic.LightGBMError):
            lgb.train(
                {"boosting": "mixture", "mixture_num_experts": 3,
                 "objective": "regression", "verbose": -1,
                 "mixture_e_step_temperature_init": 0.0},
                lgb.Dataset(X, label=y), num_boost_round=2,
            )


# --------------------------------------------------------------------------- #
# Combined: DA-EM rescues a bad init                                          #
# --------------------------------------------------------------------------- #

class TestDAEMOnUniformInit:
    """End-to-end smoke: with mixture_init=uniform the symmetry breaker fires
    AND DA-EM softens the early E-step. Compared to default settings, this
    combination should yield (a) non-trivial expert load and (b) lower
    final entropy than constant T=10 (because annealing → sharp at end)."""

    def test_uniform_with_da_em_settles_to_non_uniform(self):
        X, y = _toy_data(n=400)
        rec = RegimeEvolutionRecorder(every=5)
        lgb.train(
            {"boosting": "mixture", "mixture_num_experts": 3,
             "objective": "regression", "verbose": -1,
             "mixture_init": "uniform",
             "mixture_warmup_iters": 3,
             "mixture_estimate_variance": True,
             "mixture_e_step_temperature_init": 5.0,
             "mixture_e_step_temperature_final": 0.5},
            lgb.Dataset(X, label=y),
            num_boost_round=30, callbacks=[rec],
        )
        # Final load should not be exactly uniform (which would prove EM
        # never broke from the init's symmetry).
        load = rec.expert_load()[-1]
        max_dev = np.abs(load - 1.0 / 3).max()
        assert max_dev > 0.01, (
            f"After DA-EM + symmetry breaker on uniform init, expert load "
            f"is still essentially uniform (max deviation = {max_dev:.4f})"
        )
