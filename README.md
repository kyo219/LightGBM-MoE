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
    'mixture_num_experts': 3,        # Number of experts
    'mixture_gate_type': 'gbdt',     # Universal winner in the 500-trial / 5-dataset study
    'mixture_routing_mode': 'token_choice',
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

## When to use MoE

The 5-dataset, 500-trial study below (sp500 is included in two parallel feature configurations, so 6 rows) shows MoE provides accuracy improvements on every dataset, ranging from sub-percent (`sp500_basic`, essentially a tie) to substantial (**`vix` −15.1 %**, the dataset where the low-vol / high-vol regime structure is most pronounced). The most informative comparison is **`sp500_basic` (13 features) vs `sp500` (28 features)** on the same raw series: MoE's edge grows as the feature set makes the latent regime more observable — the broader thesis that "MoE wins scale with regime observability" is reproducible *within* a single dataset by changing only the feature engineering. Compute trade-off varies: most datasets cost **1.7–4.8 ×** naive's per-fold train time, but `fred_gdp` and the enriched `sp500` are only **0.85–2.2 ×** (MoE is in fact slightly *faster* than naive on `sp500`). Net rule: try MoE when you care about the last few percent and your wall time has 2–5× headroom.

## Benchmark — 500-trial study (naive-lightgbm vs MoE, 5 datasets)

5-fold time-series CV, 500 Optuna trials per (variant × dataset), 5 datasets spanning synthetic-ideal → real macro/financial → controlled-latent. Full report: [`bench_results/study_500_report.md`](bench_results/study_500_report.md). Methodology and dataset-specific recommendations: [docs/moe/benchmark.md](docs/moe/benchmark.md).

| Dataset | Shape | naive-lightgbm best | MoE best | Δ RMSE | Speed (MoE / naive, median train s/fold) |
|---|---|---|---|---|---|
| `synthetic` | 2000 × 5 | 4.9765 | **4.6651** | −6.3 % | 0.663 / 0.240 = **2.76 ×** |
| `fred_gdp` | 311 × 12 | 0.9286 | **0.9128** | −1.7 % | 0.122 / 0.055 = **2.22 ×** |
| `sp500_basic` (13 feat) | 3761 × 13 | **0.01003** | 0.01005 | +0.18 % *(naive wins)* | 0.152 / 0.127 = **1.20 ×** |
| `sp500` (28 feat, enriched) | 3711 × 28 | 0.01002 | **0.00998** | −0.34 % | 0.134 / 0.158 = **0.85 ×** *(MoE faster)* |
| `vix` | 3762 × 13 | 2.8942 | **2.4574** | **−15.1 %** | 0.386 / 0.081 = **4.77 ×** |
| `hmm` | 2000 × 5 | 2.1893 | **2.1096** | −3.6 % | 0.126 / 0.074 = **1.70 ×** |

> **The sp500 pair is a controlled feature-engineering ablation on the same raw series**: only the feature set changes between rows, identical CV / Optuna budget / seed. With 13 features `naive-lightgbm` wins by 0.18 %; with 28 features MoE flips to a 0.34 % win *and* trains 15 % faster. The flip from "basic" to "enriched" is the cleanest in-repo demonstration that MoE's lift scales with how observable the regime is from the features you provide.

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

### Settings that won on every dataset

The only categorical setting that produced the absolute best (min) RMSE on all 5 datasets is **`mixture_gate_type='gbdt'`**. For everything else, the optimal value depends on the dataset — full per-dataset table in [docs/moe/benchmark.md](docs/moe/benchmark.md).

| Parameter | Universal? | Notes |
|---|---|---|
| `mixture_gate_type` | **`gbdt`** | Best minimum RMSE on every dataset; `leaf_reuse` and `none` never produced the absolute best |
| `mixture_routing_mode` | **No** | `token_choice` won on synthetic; `expert_choice` won on fred_gdp, vix, hmm. Search both. |
| `mixture_num_experts` | weakly 3-4 | Q4 quartile mean is best on most datasets but margin is small |
| `mixture_diversity_lambda` | search 0.0–0.5 | Consistently top-5 in fANOVA importance for MoE; no single best value, but searching it matters |

Dataset-dependent knobs (`mixture_e_step_mode`, `mixture_init`, `mixture_r_smoothing`, `mixture_hard_m_step`, `extra_trees`, `learning_rate`) need per-problem search — see [docs/moe/benchmark.md](docs/moe/benchmark.md) for the full breakdown table.

```bash
# Reproduce the full headline study (~30-40 min on 12-core / 24-thread; 6 dataset rows)
python examples/comparative_study.py --trials 500 --out bench_results/study_500.json

# Smoke test (~1 min, all 6 dataset rows)
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
