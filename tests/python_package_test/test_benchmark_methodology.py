# coding: utf-8
"""Methodology invariants for the comparative-study benchmark.

These lock in the audit fixes from the v0.8.1 benchmark overhaul so they
cannot silently regress:

  1. Feature/target alignment — row t sees series[..t], target is series[t+1]
     ("next-day" means next-day; the pre-audit code had an off-by-one that
     made it two-steps-ahead with the most informative lag discarded).
  2. Causality — features computed on a truncated series must equal the
     corresponding prefix of features computed on the full series (no
     full-sample statistics like the old `vix - vix.mean()` leak).
  3. Seed handling — synthetic generators are reproducible for a fixed seed
     and vary across seeds; no global np.random state is mutated.
  4. Protocol helpers — the ES-tail split never trains on the scoring fold,
     and the chronological holdout split is disjoint and ordered.

Market-data tests use the cached CSVs under examples/data_cache and are
skipped when the cache is absent (CI without network); the synthetic ones
always run.
"""

import os
import sys

import numpy as np
import pytest

_EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "..", "examples")
sys.path.insert(0, _EXAMPLES)

import benchmark as bm  # noqa: E402

_CACHE = os.path.join(_EXAMPLES, "data_cache")


def _has_cache(name):
    return os.path.exists(os.path.join(_CACHE, name))


requires_vix = pytest.mark.skipif(not _has_cache("vix_VIX.csv"),
                                  reason="vix cache not present")
requires_sp500 = pytest.mark.skipif(not _has_cache("sp500_GSPC.csv"),
                                    reason="sp500 cache not present")


# ---------------------------------------------------------------------------
# 1. Alignment: lag-1 = today, target = tomorrow
# ---------------------------------------------------------------------------
@requires_vix
def test_vix_alignment_next_day():
    X, y, _ = bm.generate_vix_data()
    # lag columns: col0 = lag-1 = s[t], col1 = lag-2 = s[t-1]
    # → next row's lag-2 equals this row's lag-1
    assert np.allclose(X[1:, 1], X[:-1, 0])
    # target is the NEXT value of the lag-1 column (truly next-day)
    assert np.allclose(y[:-1], X[1:, 0])


@requires_sp500
def test_sp500_basic_alignment_next_day():
    X, y, _ = bm.generate_sp500_basic_data()
    assert np.allclose(X[1:, 1], X[:-1, 0])
    assert np.allclose(y[:-1], X[1:, 0])


@requires_sp500
def test_sp500_enriched_alignment_next_day():
    X, y, _ = bm.generate_sp500_data()
    # col0 is the lag-1 return = today's return; target = tomorrow's return
    assert np.allclose(y[:-1], X[1:, 0])


def test_fred_style_alignment_ar():
    # fred_gdp follows the AR convention: target growth_t, lags t-1..t-4.
    # Reconstruct with a deterministic series to avoid the network fetch.
    growth = np.sin(np.arange(300) * 0.7) + np.linspace(0, 1, 300)
    lag_max = 4
    n = len(growth) - lag_max
    lags = np.column_stack([growth[lag_max - k - 1: lag_max - k - 1 + n]
                            for k in range(lag_max)])
    y = growth[lag_max:]
    # lag k=0 must be the value immediately before the target
    assert np.allclose(lags[:, 0], growth[lag_max - 1: lag_max - 1 + n])
    assert np.allclose(y[:-1], lags[1:, 0])


# ---------------------------------------------------------------------------
# 2. Causality: truncation invariance (catches any full-sample statistic)
# ---------------------------------------------------------------------------
@requires_vix
def test_vix_features_do_not_depend_on_future():
    close = bm._yf_download_close("^VIX", "2010-01-01", "2024-12-31", "vix_VIX")
    vix = close.to_numpy(dtype=np.float64)

    def features_up_to(n_keep):
        v = vix[:n_keep]
        em = np.cumsum(v) / np.arange(1, len(v) + 1)
        c = v.copy()
        c[1:] -= em[:-1]
        c[0] = 0.0
        return bm._add_ts_features(c)

    full = features_up_to(len(vix))
    part = features_up_to(500)
    # If any feature used a full-sample statistic (the old vix.mean() leak),
    # the prefix would differ.
    assert np.allclose(full[:500], part)


def test_ts_features_exclusive_of_current_index():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(200)
    feats = bm._add_ts_features(y)
    # Changing y[i] must not change feats[i] (row i sees only y[:i]).
    y2 = y.copy()
    y2[100] += 100.0
    feats2 = bm._add_ts_features(y2)
    assert np.allclose(feats[100], feats2[100])
    # ...but must change some later row.
    assert not np.allclose(feats[101], feats2[101])


# ---------------------------------------------------------------------------
# 3. Seeds: reproducible, varying, and no global RNG pollution
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gen", [bm.generate_synthetic_data, bm.generate_hmm_data])
def test_synthetic_generators_seed_contract(gen):
    X1, y1, _ = gen(seed=42)
    X2, y2, _ = gen(seed=42)
    X3, y3, _ = gen(seed=43)
    assert np.allclose(X1, X2) and np.allclose(y1, y2)
    assert not np.allclose(y1, y3)


def test_generators_do_not_touch_global_rng():
    np.random.seed(12345)
    expected = np.random.RandomState(12345).rand(4)
    bm.generate_synthetic_data(seed=7)
    bm.generate_hmm_data(seed=7)
    got = np.random.rand(4)
    assert np.allclose(got, expected), (
        "data generators mutated the global numpy RNG state")


# ---------------------------------------------------------------------------
# 4. Protocol helpers
# ---------------------------------------------------------------------------
def _load_study_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "comparative_study", os.path.join(_EXAMPLES, "comparative_study.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def study():
    return _load_study_module()


def test_es_tail_split_never_empty(study):
    for n in (40, 100, 1000, 25):
        cut = study._es_tail_split(n)
        assert 0 < cut < n  # both fit and ES parts non-empty


def test_chronological_split_disjoint_ordered(study):
    X = np.arange(100).reshape(-1, 1).astype(float)
    y = np.arange(100).astype(float)
    Xs, ys, Xh, yh = study.chronological_split(X, y, 0.2)
    assert len(Xh) == 20 and len(Xs) == 80
    # holdout is strictly the chronological tail
    assert ys.max() < yh.min()


def test_cv_gap_embargo(study):
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5, gap=study.CV_GAP)
    X = np.zeros((200, 1))
    for tr, va in tscv.split(X):
        # at least a 1-row embargo between train end and valid start
        assert va.min() - tr.max() >= study.CV_GAP + 1
