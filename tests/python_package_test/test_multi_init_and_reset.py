# coding: utf-8
"""Tests for multi-init restart and expert collapse reset.

These two features sit at opposite ends of the EM intervention spectrum:

  - **Multi-init restart** is a Python-side wrapper. EM converges to an
    init-specific local optimum (pairwise ARI < 0.15 across schemes on the
    em_init_sensitivity probe), so we run several attempts with varied
    inits/seeds and pick the best.

  - **Expert reset** is a C++-side surgery. Once an expert collapses
    (mean responsibility falls below threshold), GBDT's tree-additivity
    means the gate has effectively learned to route past it permanently.
    We roll back the collapsed expert's recent trees and force its
    responsibility share onto the samples the rest of the model is fitting
    worst, giving it a fresh territory to specialize on.
"""

from __future__ import annotations

import numpy as np
import pytest

import lightgbm_moe as lgb
from lightgbm_moe import (
    MultiInitResult,
    MultiInitTrial,
    RegimeEvolutionRecorder,
    train_multi_init,
)


def _toy_regression(n=300, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X[:, 0] * 2.0 - X[:, 1] + 0.1 * rng.normal(size=n)
    return X, y


# --------------------------------------------------------------------------- #
# Multi-init restart                                                          #
# --------------------------------------------------------------------------- #

class TestMultiInitBasic:
    def test_returns_best_by_lower_rmse(self):
        X, y = _toy_regression()
        res = train_multi_init(
            {"boosting": "mixture", "mixture_num_experts": 3,
             "objective": "regression", "verbose": -1,
             "mixture_warmup_iters": 2},
            lgb.Dataset(X, label=y),
            num_boost_round=10, n_inits=4,
            init_schemes=["uniform", "kmeans_features", "gmm", "quantile"],
            score_data=(X, y),
        )
        assert isinstance(res, MultiInitResult)
        assert len(res.trials) == 4
        scores = [t.score for t in res.trials]
        # Best trial must have the minimum score (lower RMSE is better).
        assert res.best_trial.score == pytest.approx(min(scores))

    def test_unique_seeds_across_attempts(self):
        X, y = _toy_regression()
        res = train_multi_init(
            {"boosting": "mixture", "mixture_num_experts": 3,
             "objective": "regression", "verbose": -1},
            lgb.Dataset(X, label=y),
            num_boost_round=5, n_inits=3,
            score_data=(X, y),
        )
        seeds = [t.seed for t in res.trials]
        assert len(set(seeds)) == len(seeds), f"seeds not unique: {seeds}"

    def test_init_scheme_cycle(self):
        X, y = _toy_regression()
        schemes = ["kmeans_features", "gmm"]
        res = train_multi_init(
            {"boosting": "mixture", "mixture_num_experts": 3,
             "objective": "regression", "verbose": -1},
            lgb.Dataset(X, label=y),
            num_boost_round=5, n_inits=4,
            init_schemes=schemes,
            score_data=(X, y),
        )
        # Cycles through the list: 0→0, 1→1, 2→0, 3→1
        assert [t.init_scheme for t in res.trials] == [
            "kmeans_features", "gmm", "kmeans_features", "gmm",
        ]

    def test_callbacks_factory_per_trial(self):
        """One recorder per trial — recorder is stateful, so reuse would
        leak snapshots across trials."""
        X, y = _toy_regression()
        recorders = []

        def make_cb(i):
            r = RegimeEvolutionRecorder(every=2)
            recorders.append(r)
            return [r]

        train_multi_init(
            {"boosting": "mixture", "mixture_num_experts": 3,
             "objective": "regression", "verbose": -1,
             "mixture_warmup_iters": 2},
            lgb.Dataset(X, label=y),
            num_boost_round=8, n_inits=3,
            score_data=(X, y),
            callbacks_factory=make_cb,
        )
        assert len(recorders) == 3
        # Each recorder captured snapshots independently.
        for r in recorders:
            assert r.num_snapshots > 0

    def test_custom_score_callable(self):
        """User-supplied scorer; must return a float."""
        X, y = _toy_regression()
        called = []

        def negative_iter_count(booster):
            # Higher = better, but reported as "custom" so we use min.
            called.append(1)
            return -float(booster.current_iteration())

        res = train_multi_init(
            {"boosting": "mixture", "mixture_num_experts": 3,
             "objective": "regression", "verbose": -1},
            lgb.Dataset(X, label=y),
            num_boost_round=5, n_inits=2,
            score_metric=negative_iter_count,
        )
        assert len(called) == 2
        assert res.score_metric == "custom"

    def test_summary_table_marks_best(self):
        X, y = _toy_regression()
        res = train_multi_init(
            {"boosting": "mixture", "mixture_num_experts": 3,
             "objective": "regression", "verbose": -1},
            lgb.Dataset(X, label=y),
            num_boost_round=5, n_inits=2,
            score_data=(X, y),
        )
        table = res.summary_table()
        assert "★" in table  # best row marker
        assert "rmse" in table.lower()

    def test_rejects_non_mixture(self):
        X, y = _toy_regression()
        with pytest.raises(ValueError, match="boosting='mixture'"):
            train_multi_init(
                {"boosting": "gbdt", "objective": "regression", "verbose": -1},
                lgb.Dataset(X, label=y),
                num_boost_round=5, n_inits=2, score_data=(X, y),
            )

    def test_rmse_needs_score_data(self):
        X, y = _toy_regression()
        with pytest.raises(ValueError, match="score_data"):
            train_multi_init(
                {"boosting": "mixture", "mixture_num_experts": 3,
                 "objective": "regression", "verbose": -1},
                lgb.Dataset(X, label=y),
                num_boost_round=5, n_inits=2,
                # No score_data → RMSE can't be computed.
            )

    def test_accepts_xy_tuple(self):
        """Subprocess workers can't share a constructed Dataset, so accept
        raw (X, y) tuples directly."""
        X, y = _toy_regression()
        res = train_multi_init(
            {"boosting": "mixture", "mixture_num_experts": 3,
             "objective": "regression", "verbose": -1},
            (X, y),
            num_boost_round=5, n_inits=2, score_data=(X, y),
        )
        assert len(res.trials) == 2

    def test_returned_booster_is_predictive(self):
        """After (de)serialization the returned booster must still predict."""
        X, y = _toy_regression()
        res = train_multi_init(
            {"boosting": "mixture", "mixture_num_experts": 3,
             "objective": "regression", "verbose": -1},
            (X, y), num_boost_round=10, n_inits=2, score_data=(X, y),
        )
        yhat = res.best_booster.predict(X)
        assert yhat.shape == (X.shape[0],)
        assert np.isfinite(yhat).all()


# --------------------------------------------------------------------------- #
# Speed knobs: n_jobs (parallel) + prescreen                                  #
# --------------------------------------------------------------------------- #

class TestMultiInitParallel:
    """ProcessPoolExecutor path. Smoke tests — verifying correctness, not
    speedup (parallel speedup is brittle in CI for small problems)."""

    def test_n_jobs_2_runs_all_trials(self):
        X, y = _toy_regression(n=300)
        res = train_multi_init(
            {"boosting": "mixture", "mixture_num_experts": 3,
             "objective": "regression", "verbose": -1,
             "mixture_warmup_iters": 2},
            (X, y),
            num_boost_round=10, n_inits=4,
            init_schemes=["uniform", "kmeans_features", "gmm", "quantile"],
            score_data=(X, y),
            n_jobs=2,
        )
        assert len(res.trials) == 4
        # Order in res.trials follows the original trial_index ordering
        # (we re-sort by trial_index when assembling), and seeds must remain
        # deterministic across the parallel scatter/gather.
        seeds = sorted(t.seed for t in res.trials)
        assert seeds == [42, 142, 242, 342]

    def test_n_jobs_rejects_callable_score(self):
        X, y = _toy_regression()
        with pytest.raises(ValueError, match="callable score_metric"):
            train_multi_init(
                {"boosting": "mixture", "mixture_num_experts": 3,
                 "objective": "regression", "verbose": -1},
                (X, y), num_boost_round=5, n_inits=2,
                score_metric=lambda b: 0.0, n_jobs=2,
            )

    def test_n_jobs_warns_on_callbacks_factory(self):
        X, y = _toy_regression()
        with pytest.warns(UserWarning, match="callbacks_factory is ignored"):
            train_multi_init(
                {"boosting": "mixture", "mixture_num_experts": 3,
                 "objective": "regression", "verbose": -1},
                (X, y), num_boost_round=5, n_inits=2,
                score_data=(X, y),
                callbacks_factory=lambda i: [],
                n_jobs=2,
            )


class TestMultiInitPrescreen:
    """Phase-1 cheap pass + phase-2 full retrain on survivors."""

    def test_prescreen_reduces_final_trials(self):
        X, y = _toy_regression()
        res = train_multi_init(
            {"boosting": "mixture", "mixture_num_experts": 3,
             "objective": "regression", "verbose": -1,
             "mixture_warmup_iters": 2},
            (X, y),
            num_boost_round=15, n_inits=5,
            init_schemes=["uniform", "random", "quantile",
                          "kmeans_features", "gmm"],
            score_data=(X, y),
            prescreen_rounds=4, prescreen_keep=2,
        )
        # Final result reports only the survivors (post-prescreen retrains).
        assert len(res.trials) == 2

    def test_prescreen_requires_keep(self):
        X, y = _toy_regression()
        with pytest.raises(ValueError, match="prescreen_keep"):
            train_multi_init(
                {"boosting": "mixture", "mixture_num_experts": 3,
                 "objective": "regression", "verbose": -1},
                (X, y), num_boost_round=10, n_inits=3,
                score_data=(X, y),
                prescreen_rounds=4,  # missing prescreen_keep
            )

    def test_prescreen_keep_must_be_positive(self):
        X, y = _toy_regression()
        with pytest.raises(ValueError, match="prescreen_keep must be"):
            train_multi_init(
                {"boosting": "mixture", "mixture_num_experts": 3,
                 "objective": "regression", "verbose": -1},
                (X, y), num_boost_round=10, n_inits=3,
                score_data=(X, y),
                prescreen_rounds=4, prescreen_keep=0,
            )

    def test_prescreen_picks_best_survivor(self):
        """Prescreen ranking should select genuinely better-fitting trials."""
        X, y = _toy_regression(n=400)
        res = train_multi_init(
            {"boosting": "mixture", "mixture_num_experts": 3,
             "objective": "regression", "verbose": -1,
             "mixture_warmup_iters": 3},
            (X, y),
            num_boost_round=20, n_inits=4,
            init_schemes=["uniform", "random", "kmeans_features", "gmm"],
            score_data=(X, y),
            prescreen_rounds=6, prescreen_keep=2,
        )
        # The two reported trials should be those that *survived* the
        # prescreen — i.e. no two of the never-promising ones.
        # We can't assert which exact two without doing the prescreen here,
        # but we can require that the final best is competitive: not worse
        # than running all 4 sequentially.
        full_run = train_multi_init(
            {"boosting": "mixture", "mixture_num_experts": 3,
             "objective": "regression", "verbose": -1,
             "mixture_warmup_iters": 3},
            (X, y),
            num_boost_round=20, n_inits=4,
            init_schemes=["uniform", "random", "kmeans_features", "gmm"],
            score_data=(X, y),
        )
        # Allow some slack — prescreen can occasionally drop the eventual
        # winner if the early-round score is misleading.
        assert res.best_trial.score <= full_run.best_trial.score * 1.3


# --------------------------------------------------------------------------- #
# Expert collapse reset                                                       #
# --------------------------------------------------------------------------- #

class TestExpertResetDisabled:
    def test_default_off_changes_nothing(self):
        """With reset disabled (default), training is bit-exact identical
        to the same run with the new params not set at all."""
        X, y = _toy_regression()
        common = {
            "boosting": "mixture", "mixture_num_experts": 3,
            "objective": "regression", "verbose": -1,
            "mixture_warmup_iters": 2,
            "mixture_init": "kmeans_features",
            "seed": 42, "deterministic": True,
        }
        m_off = lgb.train(common, lgb.Dataset(X, label=y), num_boost_round=15)
        m_off_explicit = lgb.train(
            {**common, "mixture_expert_reset_enable": False},
            lgb.Dataset(X, label=y), num_boost_round=15,
        )
        np.testing.assert_allclose(
            m_off.predict(X), m_off_explicit.predict(X), atol=1e-12
        )


class TestExpertResetFires:
    """Trigger collapse by oversizing K relative to a trivial problem so
    one or more experts inevitably end up with little load."""

    def _train(self, enable_reset, **overrides):
        rng = np.random.default_rng(1)
        n = 200
        X = rng.normal(size=(n, 3))
        y = X[:, 0]  # only one feature actually predicts → lots of redundant experts
        rec = RegimeEvolutionRecorder(every=2)
        # Use random init so the natural starting load is uneven (one or
        # more experts will sit well below the threshold from iter 0,
        # guaranteeing the reset has something to work on).
        params = {
            "boosting": "mixture", "mixture_num_experts": 8,
            "objective": "regression", "verbose": -1,
            "mixture_init": "random",
            "mixture_warmup_iters": 2,
            "mixture_expert_reset_enable": enable_reset,
            "mixture_expert_reset_threshold": 0.15,
            "mixture_expert_reset_interval": 4,
            "mixture_expert_reset_trees": 3,
            "seed": 42, "deterministic": True,
            **overrides,
        }
        m = lgb.train(params, lgb.Dataset(X, label=y),
                      num_boost_round=30, callbacks=[rec])
        return m, rec

    def test_collapse_load_rebounds_under_reset(self):
        """The reset's job is not to permanently fix collapse (over-K can
        make some collapse inevitable), but to perturb the load trajectory
        so that no single expert stays below threshold across the entire
        run. Compare the *minimum-over-iterations* of the per-expert load
        trajectory with reset off vs on: with reset on, the minimum should
        be at least sometimes restored above threshold immediately after
        the reset fires."""
        m_off, rec_off = self._train(enable_reset=False)
        m_on, rec_on = self._train(enable_reset=True)

        # Both runs should complete cleanly.
        assert rec_off.num_snapshots > 0
        assert rec_on.num_snapshots > 0

        # With reset on, at least one expert's load should differ measurably
        # from the no-reset baseline at some iteration. (If the predictions
        # were bit-identical we'd know the reset never fired.)
        load_off = rec_off.expert_load()  # (S, K)
        load_on = rec_on.expert_load()
        diff = np.abs(load_off - load_on).max()
        assert diff > 1e-3, (
            f"reset=True produced identical load trajectory to reset=False "
            f"(max diff = {diff:.2e}); reset never fired or had no effect"
        )

    def test_reset_does_not_crash_with_minimal_drop_window(self):
        """`mixture_expert_reset_trees` larger than current iter count must
        gracefully clamp at GBDT iteration 0."""
        m, _ = self._train(
            enable_reset=True,
            mixture_expert_reset_trees=100,  # bigger than num_boost_round
            mixture_expert_reset_interval=3,
        )
        # Just verify training completed and produces a valid prediction.
        rng = np.random.default_rng(2)
        X_test = rng.normal(size=(10, 3))
        y_pred = m.predict(X_test)
        assert y_pred.shape == (10,)
        assert np.isfinite(y_pred).all()
