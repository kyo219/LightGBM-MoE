# coding: utf-8
"""Tests for RegimeEvolutionRecorder and the underlying get_responsibilities binding."""

from __future__ import annotations

import numpy as np
import pytest

import lightgbm_moe as lgb
from lightgbm_moe import RegimeEvolutionRecorder


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #

def _two_regime_data(n=300, seed=0):
    """Toy time-series with a deterministic 2-regime structure."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    # Regime flips at row n//2 — gives the gate something to learn.
    y = np.where(np.arange(n) < n // 2, 2.0 * X[:, 0], -2.0 * X[:, 0])
    y = y + 0.1 * rng.normal(size=n)
    return X, y


def _train_with_recorder(X, y, recorder, params=None, num_boost_round=20):
    p = {
        "boosting": "mixture",
        "mixture_num_experts": 2,
        "objective": "regression",
        "verbose": -1,
        "mixture_warmup_iters": 2,
        "mixture_init": "kmeans_features",
    }
    if params:
        p.update(params)
    dset = lgb.Dataset(X, label=y)
    return lgb.train(p, dset, num_boost_round=num_boost_round, callbacks=[recorder])


# --------------------------------------------------------------------------- #
# get_responsibilities binding                                                #
# --------------------------------------------------------------------------- #

class TestGetResponsibilitiesBinding:
    def test_returns_valid_stochastic_matrix(self):
        X, y = _two_regime_data()
        captured = []

        def cb(env):
            r = env.model.get_responsibilities()
            captured.append((env.iteration, r))

        _train_with_recorder(X, y, cb, num_boost_round=5)

        assert len(captured) == 5
        for it, r in captured:
            assert r.shape == (300, 2)
            np.testing.assert_allclose(r.sum(axis=1), 1.0, atol=1e-9)
            assert (r >= 0).all() and (r <= 1).all()

    def test_raises_for_non_mixture(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 3))
        y = rng.normal(size=100)
        m = lgb.train(
            {"objective": "regression", "verbose": -1},
            lgb.Dataset(X, label=y), num_boost_round=5,
        )
        with pytest.raises(lgb.basic.LightGBMError):
            m.get_responsibilities()

    def test_post_load_returns_empty(self, tmp_path):
        """responsibilities_ is not serialized — loaded model returns empty."""
        X, y = _two_regime_data()
        m = _train_with_recorder(X, y, lambda env: None, num_boost_round=5)
        f = tmp_path / "model.txt"
        m.save_model(str(f))
        loaded = lgb.Booster(model_file=str(f))
        r = loaded.get_responsibilities()
        assert r.shape == (0, 2)


# --------------------------------------------------------------------------- #
# Recorder behavior                                                           #
# --------------------------------------------------------------------------- #

class TestRecorderCapture:
    def test_every_n_snapshots(self):
        X, y = _two_regime_data()
        rec = RegimeEvolutionRecorder(every=5)
        _train_with_recorder(X, y, rec, num_boost_round=20)
        # iter 0, 5, 10, 15 → 4 snapshots
        iters = rec.iterations
        assert list(iters) == [0, 5, 10, 15]

    def test_capture_iter_zero_when_every_large(self):
        X, y = _two_regime_data()
        rec = RegimeEvolutionRecorder(every=100, capture_iter_zero=True)
        _train_with_recorder(X, y, rec, num_boost_round=10)
        # iter 0 forced, no others reach 100 → 1 snapshot
        assert rec.num_snapshots == 1
        assert rec.iterations[0] == 0

    def test_max_snapshots_trims(self):
        X, y = _two_regime_data()
        rec = RegimeEvolutionRecorder(every=1, max_snapshots=4)
        _train_with_recorder(X, y, rec, num_boost_round=20)
        # 20 iters captured, trimmed to 4. First and last must be retained.
        assert rec.num_snapshots == 4
        assert rec.iterations[0] == 0
        assert rec.iterations[-1] == 19

    def test_iter_zero_holds_init_responsibilities_under_warmup(self):
        """During warmup the E-step is skipped, so the iter=0 snapshot equals
        InitResponsibilities (kmeans_features in this case)."""
        X, y = _two_regime_data()
        rec = RegimeEvolutionRecorder(every=1)
        _train_with_recorder(X, y, rec,
                             params={"mixture_warmup_iters": 3},
                             num_boost_round=5)
        # During warmup r should match the *init* — the same stochastic matrix
        # for iters 0, 1, 2. After warmup it diverges.
        r0 = rec.snapshots[0][1]
        r1 = rec.snapshots[1][1]
        r2 = rec.snapshots[2][1]
        np.testing.assert_allclose(r0, r1, atol=1e-12)
        np.testing.assert_allclose(r1, r2, atol=1e-12)

    def test_callback_does_not_break_training(self):
        X, y = _two_regime_data()
        # Smaller K=2, small dataset — just verify training completes.
        rec = RegimeEvolutionRecorder(every=2)
        m = _train_with_recorder(X, y, rec, num_boost_round=10)
        assert m.is_mixture()
        assert rec.num_snapshots > 0


# --------------------------------------------------------------------------- #
# Derived metrics                                                             #
# --------------------------------------------------------------------------- #

class TestDerivedMetrics:
    def setup_method(self):
        X, y = _two_regime_data()
        self.X, self.y = X, y
        self.rec = RegimeEvolutionRecorder(every=2)
        _train_with_recorder(X, y, self.rec, num_boost_round=20)

    def test_regime_argmax_shape_and_range(self):
        am = self.rec.regime_argmax()
        assert am.shape == (self.rec.num_snapshots, 300)
        assert am.min() >= 0 and am.max() <= 1  # K=2

    def test_expert_load_sums_to_one_per_snapshot(self):
        load = self.rec.expert_load()
        np.testing.assert_allclose(load.sum(axis=1), 1.0, atol=1e-9)

    def test_mean_entropy_nonneg_and_bounded(self):
        ent = self.rec.mean_entropy()
        assert (ent >= 0).all()
        assert (ent <= np.log(2) + 1e-9).all()  # K=2 max entropy

    def test_flip_rate_shape(self):
        fr = self.rec.flip_rate()
        assert fr.shape == (self.rec.num_snapshots - 1,)
        assert (fr >= 0).all() and (fr <= 1).all()

    def test_empty_recorder_raises(self):
        rec = RegimeEvolutionRecorder()
        with pytest.raises(RuntimeError):
            rec.regime_argmax()


# --------------------------------------------------------------------------- #
# Mode resolution                                                             #
# --------------------------------------------------------------------------- #

class TestModeResolution:
    def test_explicit_mode_overrides_auto(self):
        rec = RegimeEvolutionRecorder(mode="tabular")
        assert rec._resolve_mode({"mixture_r_smoothing": "markov"}) == "tabular"

    def test_auto_picks_timeseries_for_markov(self):
        rec = RegimeEvolutionRecorder(mode="auto")
        assert rec._resolve_mode({"mixture_r_smoothing": "markov"}) == "timeseries"
        assert rec._resolve_mode({"mixture_r_smoothing": "ema"}) == "timeseries"

    def test_auto_picks_tabular_otherwise(self):
        rec = RegimeEvolutionRecorder(mode="auto")
        assert rec._resolve_mode({"mixture_r_smoothing": "none"}) == "tabular"
        assert rec._resolve_mode({}) == "tabular"

    def test_time_axis_array_pushes_to_timeseries(self):
        rec = RegimeEvolutionRecorder(mode="auto", time_axis=np.arange(10))
        assert rec._resolve_mode({}) == "timeseries"

    def test_time_axis_length_validated(self):
        rec = RegimeEvolutionRecorder(time_axis=np.arange(7))
        with pytest.raises(ValueError, match="length 7"):
            rec._resolve_time_axis(10)


# --------------------------------------------------------------------------- #
# plot() — smoke tests (matplotlib in non-interactive backend)                #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def trained_recorder():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    X, y = _two_regime_data()
    rec = RegimeEvolutionRecorder(every=4)
    _train_with_recorder(
        X, y, rec,
        params={"mixture_r_smoothing": "markov",
                "mixture_smoothing_lambda": 0.3},
        num_boost_round=20,
    )
    return rec, y


class TestPlotSmoke:
    def test_plot_timeseries_returns_figure(self, trained_recorder):
        rec, y = trained_recorder
        fig = rec.plot(y=y)
        # Sanity: 4 axes (top, tape, diag, load) + 1 colorbar = 5
        assert len(fig.axes) >= 4

    def test_plot_tabular_mode_explicit(self, trained_recorder):
        rec, y = trained_recorder
        fig = rec.plot(y=y, mode="tabular")
        assert len(fig.axes) >= 4

    def test_plot_without_y_still_works(self, trained_recorder):
        rec, _ = trained_recorder
        fig = rec.plot()
        assert len(fig.axes) >= 3  # top is empty stub

    def test_plot_y_length_mismatch_raises(self, trained_recorder):
        rec, _ = trained_recorder
        with pytest.raises(ValueError, match="length"):
            rec.plot(y=np.zeros(7))

    def test_plot_empty_recorder_raises(self):
        rec = RegimeEvolutionRecorder()
        with pytest.raises(RuntimeError, match="No snapshots"):
            rec.plot()
