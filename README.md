<img src="docs/logo.png" width=300 />

LightGBM-MoE
============

**A regime-switching / Mixture-of-Experts extension of LightGBM.**

[English](README.md) | [日本語](README.ja.md)

---

## Overview

LightGBM-MoE is a fork of [Microsoft LightGBM](https://github.com/microsoft/LightGBM) that implements **Mixture-of-Experts (MoE) / Regime-Switching GBDT** natively in C++:

```
ŷ(x) = Σₖ gₖ(x) · fₖ(x)
```

- `fₖ(x)`: Expert k's prediction (K regression GBDTs)
- `gₖ(x)`: Gate's routing probability for expert k (softmax)
- `K`: Number of experts (hyperparameter)

## Requirements

- **Python**: 3.10+
- **OS**: Linux (x86_64, aarch64), macOS (Intel, Apple Silicon)
- **Build**: CMake 3.16+ and a C++ compiler (GCC, Clang, or Apple Clang) when installing from source

## Installation

```bash
# From GitHub (builds from source)
pip install git+https://github.com/kyo219/LightGBM-MoE.git

# Editable install for development
git clone https://github.com/kyo219/LightGBM-MoE.git
cd LightGBM-MoE/python-package
pip install -e .
```

## Quick Start

```python
import lightgbm_moe as lgb

params = {
    'boosting': 'mixture',           # Enable MoE mode
    'mixture_num_experts': 3,        # Number of experts (search 2-6 for your data)
    'mixture_gate_type': 'gbdt',     # Best on 4/6 datasets; search 'leaf_reuse' too
    'mixture_routing_mode': 'token_choice',  # Tied 3/3 with 'expert_choice' across datasets
    'objective': 'regression',
    'metric': 'rmse',
    'verbose': -1,
}

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

model = lgb.train(
    params, train_data,
    num_boost_round=500,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50)],
)

# Predictions
y_pred = model.predict(X_test)                      # Weighted mixture
regime = model.predict_regime(X_test)               # Regime index (argmax)
regime_proba = model.predict_regime_proba(X_test)   # Gate probabilities (N, K)
expert_preds = model.predict_expert_pred(X_test)    # Expert predictions (N, K)
```

## How regime estimation works

Each boosting round runs one **soft-EM step** (Jordan & Jacobs 1991, GBDT-ified). After a short warmup (default 5 rounds — gate fits `r_init` first so the first real E-step has a meaningful prior):

```
per round:
  Forward    →  expert preds f_k(x),  gate prior π_k(x)
  E-step     →  r_ik = softmax_k[ log π_k(x_i) + log p(y_i | f_k(x_i), σ_k²) ]
  M-step σ²  →  σ_k² = Σ_i r_ik (y_i − f_k)² / Σ_i r_ik
  M-step f   →  expert k gets +1 tree, gradient = r_ik · ∂L/∂f_k
  M-step g   →  gate gets +K trees, soft CE  ∇z = (p − r) / T
```

`r_ik` is the **posterior probability that sample i was emitted by regime k** under the current mixture, not just a routing weight — this is what makes `predict_regime_proba` well-calibrated rather than just "where the gate happens to point". Three implementation choices distinguish this from a textbook GBDT-MoE — each was a real bug at some point:

| Choice | What it fixes | PR |
|---|---|---|
| Per-expert noise scale `σ_k²` is **estimated each round** (default since #24) | Without it, a hand-tuned `mixture_e_step_alpha` is silently doing what `1/(2σ²)` should — y-scale variant, broken across datasets. With it, the responsibility softmax is the actual Bayesian posterior — y-scale invariant. | [#24](https://github.com/kyo219/LightGBM-MoE/pull/24) |
| **Bias-free prior split**: `π_k = softmax(z/T)` for the E-step, `softmax((z+b)/T)` for the actual routing | DeepSeek "loss-free LB" bias is a routing nudge, not part of the probabilistic model; mixing it into the prior makes the gate spend each iter unlearning the bias the load balancer just added. | [#25](https://github.com/kyo219/LightGBM-MoE/pull/25) |
| **Soft CE gate** with `∇z = (p − r)/T`, Hessian = `K/(K−1) · p(1−p) / T²` (Friedman) | Earlier code argmax'd `r` to hard pseudo-labels (dropped the soft routing signal); missing chain-rule mis-scaled Newton steps by T at non-unit temperatures; missing Friedman factor mis-sized leaf values by `(K−1)/K`. | [#23](https://github.com/kyo219/LightGBM-MoE/pull/23) |

Three pathology guards worth knowing about:

- **Uniform `r` is an EM fixed point** — same gradient → identical trees → `r` stays uniform forever. After init, a deterministic sin perturbation breaks the symmetry (no-op when `r` is non-uniform; only `mixture_init=uniform` actually triggers it). Reproducer: `examples/em_init_sensitivity.py` ([#36](https://github.com/kyo219/LightGBM-MoE/pull/36)).
- **Diversity regularizer sign** was inverted in earlier code (pushed experts *together*). Sign + Huber clip + Hessian damping fix in [#26](https://github.com/kyo219/LightGBM-MoE/pull/26) — the empirical "`mixture_diversity_lambda` is the only universal MoE knob" finding ([Settings that won](#settings-that-won--search-every-knob)) only became real after this.
- **ELBO is logged every 10 iters** (`Σ_i log Σ_k π_k p(y_i | f_k, σ_k²)`). Approximate M-step makes small dips normal; persistent / >5% drops have historically meant high expert dropout, adaptive-LR decoupling experts from EM, or a recent gradient/Hessian regression — flagged loudly so we don't miss it.

Full deep dive with code refs: [docs/moe/architecture.md](docs/moe/architecture.md).

## Limitations

Two practical caveats of regime estimation in this model. The first is universal to switching models; the second is specific to this implementation.

### Init dependence (universal to switching models / MoE)

EM converges to a *local* fixed point of `r`, and which fixed point you reach depends on `r_init`. This is the same multimodality that makes GMM and HMM fits init-dependent — not unique to MoE-GBDT. Practical implications:

- If the regime is observable from X, prefer `mixture_init=gmm_features` or `kmeans_features` over the `[X, y]`-aware variants. Cluster on raw features, not on the label.
- `mixture_init` is worth searching over with Optuna — it's not interchangeable across datasets, and there is no universal best init (verified in the 500-trial study).
- **Don't read `r_ik` as "the true regime" of sample i.** It's "the regime *this* run of EM converged to under *this* init". Different runs with different inits are valid alternative explanations of the same data.

### EM updates are additive-only (model-specific)

Classical EM rewrites parameters freely between iterations — at iter t+1 the component means/variances can be radically different from t, and that is how EM escapes one basin and finds another mode. **Here the experts are gradient-boosted ensembles, so each EM round only *appends* one tree per expert** (historical trees are frozen). The expert function `f_k(x)` therefore evolves incrementally (one tree of `learning_rate`-scaled correction per round); `r_ik` shifts only as fast as the new tree's contribution to `loss(y_i, f_k(x_i))` shifts; and the gate's accumulated trees lock in routing toward whichever expert `r` concentrated on early.

Two consequences worth being honest about:

- **Large regime re-assignments don't happen mid-training.** Some refinement happens — that's why the [#26](https://github.com/kyo219/LightGBM-MoE/pull/26) diversity term and the [#36](https://github.com/kyo219/LightGBM-MoE/pull/36) symmetry breaker matter — but you don't see the "EM flipped two components" behavior of free-parameter EM. Snapshotting `model.get_responsibilities()` from a per-iter callback (see the docstring in `python-package/lightgbm_moe/basic.py:4994`) shows `r` smoothing toward its fixed point, not bouncing between modes.
- **This compounds init dependence above** — the basin set by `r_init` is sticky because the model cannot take big jumps in `r` to escape it.

If your problem genuinely needs mode-discovery rather than mode-refinement (e.g. unsupervised regime detection where the regime structure is unknown a priori), this implementation will under-deliver versus a free-parameter EM with K random restarts. **The root-cause fix — leaf-value refit-on-r-update, which restores the closed-form M-step on each tree's existing partition structure — is planned for v0.7** (tracked in [#37](https://github.com/kyo219/LightGBM-MoE/issues/37)). Until then, this is a real limitation; track it via `model.get_responsibilities()` snapshots and watch for ELBO-monotonicity warnings.

## When to use MoE

MoE's gating mechanism wins on regime-structured data — but the **honest baseline is not single-model naive LightGBM, it's a K-way ensemble of LightGBMs** with the same total tree budget. Simply averaging K independent models with different seeds gives variance reduction "for free"; the question is whether MoE's learned routing beats that. The 6-row study below answers it: **MoE clearly wins on `synthetic` (−24.8 %), `vix` (−6.9 %), `hmm` (−1.6 %) — datasets where the regime is observable from `X` —, ties on the `sp500` rows, and *loses to the K-way ensemble on `fred_gdp`* (+3.2 %)**. On `fred_gdp` (~310 quarterly samples) the K-way capacity helps but gating does not — the regime is too noisy or the dataset too small for the gate to learn it, and uniform averaging extracts the available capacity better than learned routing. Compute trade-off: MoE costs **1.3–8.4 ×** naive single-model time, and **0.5–2.8 ×** naive-ensemble time. Net rule: try MoE when (a) you believe the regime is observable from your features and (b) your wall time has 5–10 × headroom over single-model LightGBM.

## Benchmark — 500-trial study (naive vs naive-ensemble vs MoE, 5 datasets)

5-fold time-series CV, 500 Optuna trials per (variant × dataset), 5 datasets spanning synthetic-ideal → real macro/financial → controlled-latent (sp500 is included in two parallel feature configurations, so 6 rows). Numbers are **deterministic** — `--n-jobs 1` (the default since [PR #30](https://github.com/kyo219/LightGBM-MoE/pull/30)) makes Optuna's TPE sampler reproducible across runs and builds with the same seed, so build-to-build comparisons are not contaminated by parallel-worker scheduling noise (which historically had ±0.3 RMSE-std on synthetic). Full report: [`bench_results/study_500_report.md`](bench_results/study_500_report.md). Methodology and dataset-specific recommendations: [docs/moe/benchmark.md](docs/moe/benchmark.md).

The third variant — **`naive-ensemble`** — is a K-way (K ∈ {2, 3, 4}) seed-ensemble of standard LightGBMs that share hyperparameters but diverge per-member via the `seed` master-seed override. Same total tree budget as MoE (K × `num_boost_round`), same Optuna search space as `naive-lightgbm` plus K. It is the fair ablation for "is gating doing real work, or would any K-way ensemble suffice?" — see [PR #33](https://github.com/kyo219/LightGBM-MoE/pull/33).

| Dataset | Shape | naive best | ensemble best | MoE best | MoE vs naive | **MoE vs ensemble** *(the fair test)* |
|---|---|---|---|---|---|---|
| `synthetic` | 2000 × 5 | 5.0233 | 4.8899 | **3.6779** | −26.8 % | **−24.8 %** 🎯 |
| `fred_gdp` | 311 × 12 | 0.9311 | **0.9094** | 0.9381 | +0.75 % | +3.2 % *(ensemble wins)* |
| `sp500_basic` (13 feat) | 3761 × 13 | 0.01003 | 0.01002 | **0.01001** | −0.24 % | −0.12 % |
| `sp500` (28 feat, enriched) | 3711 × 28 | 0.01002 | 0.01003 | 0.01002 | ±0.00 % | −0.06 % |
| `vix` | 3762 × 13 | 2.8869 | 2.8724 | **2.6745** | −7.4 % | **−6.9 %** 🎯 |
| `hmm` | 2000 × 5 | 2.1913 | 2.1818 | **2.1465** | −2.1 % | **−1.6 %** |

**Read the "MoE vs ensemble" column** — that's where the gating hypothesis is actually tested. MoE clearly beats the K-way average on the three datasets where the regime is structurally observable from features (synthetic, vix, hmm), is a wash on the irreducibly hard sp500 rows, and *loses* on the smallest-data fred_gdp where the gate cannot learn the regime reliably enough to beat uniform averaging.

| Dataset | naive (s/fold) | ensemble (s/fold) | MoE (s/fold) | ensemble / naive | MoE / naive | MoE / ensemble |
|---|---|---|---|---|---|---|
| `synthetic` | 0.033 | 0.083 | 0.044 | 2.5 × | 1.3 × | **0.53 ×** |
| `fred_gdp` | 0.003 | 0.017 | 0.020 | 5.5 × | 6.5 × | 1.2 × |
| `sp500_basic` | 0.012 | 0.029 | 0.081 | 2.4 × | 6.7 × | 2.8 × |
| `sp500` | 0.009 | 0.037 | 0.055 | 4.2 × | 6.2 × | 1.5 × |
| `vix` | 0.010 | 0.035 | 0.062 | 3.4 × | 6.0 × | 1.8 × |
| `hmm` | 0.005 | 0.023 | 0.045 | 4.3 × | 8.4 × | 2.0 × |

On `synthetic` MoE is actually **faster** than the naive-ensemble (0.53 ×) — when the regime is clearly identifiable, sparse routing means each expert sees only its share of samples and converges quickly; the ensemble has to fit all K members on full data. On harder datasets the ratio flips back.

> **The sp500 pair is a controlled feature-engineering ablation on the same raw series**: only the feature set changes between rows, identical CV / Optuna budget / seed. All three variants are essentially tied at ~0.01001 across both feature sets — the next-day-log-return objective appears to hit the irreducible noise floor under any architecture and any feature set we've tried.

### Datasets

Five datasets spanning the regime-switching applicability spectrum: one MoE-ideal synthetic, three real-world series (the canonical references in the regime-switching literature), and one HMM with known ground-truth labels for measuring regime recovery.

#### `synthetic` — feature-driven regime, MoE-ideal case (2000 × 5)

- **Source**: internal generator in this repo (`generate_synthetic_data` in `examples/benchmark.py`).
- **Construction**: five i.i.d. Gaussian features; the regime is a deterministic function of `X`, so the gate can route perfectly:

  ```
  regime = (0.5·X1 + 0.3·X2 − 0.2·X3 > 0)
  y | regime=0 :   5·X0 + 3·X0·X2 + 2·sin(2·X3) + 10  +  ε
  y | regime=1 :  −5·X0 − 2·X1²   + 3·cos(2·X4) − 10  +  ε     ε ~ N(0, 0.5²)
  ```

  The two regimes use *opposite-sign* coefficients on the same features — a single GBDT is forced to average them, which is exactly the failure mode MoE is built to avoid.

#### `fred_gdp` — US Real GDP, Hamilton-style MS-AR(4) (~310 × 12)

- **Source**: FRED series [`GDPC1`](https://fred.stlouisfed.org/series/GDPC1) — Real Gross Domestic Product, Chained 2017 Dollars, Quarterly, Seasonally Adjusted Annual Rate (BEA via FRED, no auth, CSV endpoint).
- **Methodology cite**: Hamilton, J. D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle.* **Econometrica** 57(2), 357-384. <https://www.jstor.org/stable/1912559>.
- **Construction**: target is the quarterly growth rate `100·Δlog(GDP)`; features are 4 lags of growth (Hamilton's MS-AR(4)) plus engineered MA / volatility / regime-proxy features. The regime (expansion / recession) is genuinely latent — there is no oracle column. Generator: `generate_fred_gdp_data`.

#### `sp500_basic` & `sp500` — S&P 500 daily log returns, two feature sets

The S&P 500 series is included in **two parallel configurations** to demonstrate how MoE's lift scales with how observable the regime is from features:

- **`sp500_basic`** (~3760 × 13): the minimal feature set — lagged returns at {1, 2, 3, 5, 10} plus the standard MA / rolling-vol / regime-proxy block (`generate_sp500_basic_data`).
- **`sp500`** (~3710 × 28): a practitioner-grade feature set — multi-horizon lags {1, 2, 3, 5, 10, 20, 60}, cumulative momentum at {5, 20, 60}, realized volatility at {5, 20, 60} plus the short/long ratio, multi-window MAs and MA crossovers, RSI(14)/RSI(30), rolling skewness and kurtosis (20-day), Bollinger band z-score, drawdown from the 20- and 60-day rolling high, and fraction of positive returns over {5, 20} days (`generate_sp500_data`).

Common to both:

- **Source**: Yahoo Finance, symbol [`^GSPC`](https://finance.yahoo.com/quote/%5EGSPC/history) (default range `2010-01-01` to `2024-12-31`).
- **Index methodology**: [S&P Dow Jones Indices, S&P 500](https://www.spglobal.com/spdji/en/indices/equity/sp-500/).
- **Target**: next-day log return (deliberately hard predictive setup).
- **Regime structure**: latent (low-vol vs high-vol periods).

The empirical takeaway from the side-by-side: with the basic feature set the result is a true tie; with the enriched set MoE picks up a sub-percent edge *and* trains faster than naive. **As features make the regime more predictable, MoE's advantage grows** — consistent with the broader thesis that MoE wins scale with regime observability.

#### `vix` — CBOE Volatility Index, daily level (~3760 × 13)

- **Source**: Yahoo Finance, symbol [`^VIX`](https://finance.yahoo.com/quote/%5EVIX/history) (same date range as `sp500`).
- **Index methodology**: [CBOE VIX](https://www.cboe.com/tradable_products/vix/).
- **Construction**: target is next-day VIX level. Features: lagged VIX at lags {1, 2, 3, 5, 10} plus MA / rolling-volatility features. Same latent low-vol / high-vol regime structure as `sp500`, viewed through the implied-vol lens. Generator: `generate_vix_data`.

#### `hmm` — 3-state Gaussian HMM with known regime labels (2000 × 5)

- **Source**: internal generator in this repo (`generate_hmm_data`).
- **Methodology cite**: Rabiner, L. R. (1989). *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition.* **Proceedings of the IEEE** 77(2), 257-286. <https://www.cs.ubc.ca/~murphyk/Bayes/rabiner.pdf>.
- **Construction**: hidden state evolves as a 3-state Markov chain with persistent (95 % diagonal) transitions; emissions are Gaussian with well-separated means `{−3, 0, +3}` and varying scales `{0.4, 0.7, 1.0}`. Two of the five feature columns carry a weak linear signal of the hidden state (so the gate has *some* observable hint, not a free lunch); the remaining three columns are pure noise. **Returns the ground-truth regime labels** — usable for measuring regime recovery, not just RMSE.

### Caching

Real-world fetches (FRED, yfinance) are cached under `examples/data_cache/` (gitignored). First run pulls from the network; subsequent runs are offline.

### Settings that won — search every knob

In the deterministic 500-trial study, **no MoE knob was a universal winner**: the best `mixture_gate_type` is `gbdt` on 4 of 6 rows but `leaf_reuse` on `fred_gdp` and `vix`; the best `mixture_routing_mode` is split 3 / 3 between `token_choice` and `expert_choice`; the best `mixture_num_experts` is spread across {2, 3, 4}. The only knob that's **non-zero on every dataset** is `mixture_diversity_lambda` (range [0.07, 0.36] across the 6 winners), which validates that the diversity regularizer is doing real work — search it explicitly. Full per-dataset table in [docs/moe/benchmark.md](docs/moe/benchmark.md).

| Parameter | Universal? | Notes |
|---|---|---|
| `mixture_gate_type` | **No** | `gbdt` wins on synthetic, sp500_basic, sp500, hmm; `leaf_reuse` wins on fred_gdp, vix. Search both. |
| `mixture_routing_mode` | **No** | `token_choice` wins on synthetic, sp500, hmm; `expert_choice` wins on fred_gdp, sp500_basic, vix. Search both. |
| `mixture_num_experts` | **No** | K=4 most common (3 / 6), but K=2 wins on synthetic and vix, K=3 on hmm. Search {2, 3, 4, 6}. |
| `mixture_diversity_lambda` | search 0.05–0.4 | Always non-zero in the per-dataset best params after the [PR #26](https://github.com/kyo219/LightGBM-MoE/pull/26) sign / Hessian fix. Top-5 in fANOVA importance. |
| `mixture_estimate_variance` | **leave at default `true`** | Default since [PR #24](https://github.com/kyo219/LightGBM-MoE/pull/24). Setting `false` re-enables the legacy fixed-`alpha` E-step temperature, which is y-scale dependent and emits a warning at Init. |

Dataset-dependent knobs (`mixture_e_step_mode`, `mixture_init`, `mixture_r_smoothing`, `mixture_hard_m_step`, `extra_trees`, `learning_rate`) need per-problem search — see [docs/moe/benchmark.md](docs/moe/benchmark.md) for the full breakdown table.

```bash
# Reproduce the full headline study (~20 min on 12-core / 24-thread, 6 dataset rows;
# default --n-jobs 1 makes the result deterministic across runs and builds — see PR #30)
python examples/comparative_study.py --trials 500 --out bench_results/study_500.json

# Smoke test (~1 min, all 6 dataset rows; --n-jobs 2 trades reproducibility for speed)
python examples/comparative_study.py --trials 10 --n-jobs 2 --out bench_results/smoke.json

# Subset of datasets
python examples/comparative_study.py --trials 500 \
    --datasets sp500_basic,sp500 --out bench_results/sp500_ablation.json
```

## Documentation

| Topic | Doc |
|---|---|
| Full parameter reference (MoE core, Gate, Smoothing, Prediction APIs) | [docs/moe/parameters.md](docs/moe/parameters.md) |
| Optuna search templates | [docs/moe/optuna-recipes.md](docs/moe/optuna-recipes.md) |
| 500-trial / 5-dataset benchmark methodology & per-dataset recommendations | [docs/moe/benchmark.md](docs/moe/benchmark.md) |
| Per-expert hyperparameters & role-based recipe | [docs/moe/per-expert-hp.md](docs/moe/per-expert-hp.md) |
| Expert Choice routing | [docs/moe/advanced-routing.md](docs/moe/advanced-routing.md) |
| Progressive training (EvoMoE) & gate temperature annealing | [docs/moe/advanced-progressive.md](docs/moe/advanced-progressive.md) |
| Expert collapse prevention & `diagnose_moe` | [docs/moe/advanced-collapse.md](docs/moe/advanced-collapse.md) |
| SHAP analysis for MoE components | [docs/moe/shap.md](docs/moe/shap.md) |
| `int8` & `use_quantized_grad` compatibility (8-axis matrix) | [docs/moe/int8-compat.md](docs/moe/int8-compat.md) |
| Architecture & EM-loop deep dive | [docs/moe/architecture.md](docs/moe/architecture.md) |

## License

MIT license. Based on [Microsoft LightGBM](https://github.com/microsoft/LightGBM).
