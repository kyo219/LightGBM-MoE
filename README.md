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

The 5-dataset, 500-trial study below shows MoE provides **modest accuracy improvements on 4 of 5 datasets** (only sp500 ties). The biggest single win is **real VIX (−15.1 % RMSE)**, which is also the place where the regime structure (low-vol / high-vol periods) is most pronounced. The catch: MoE pays a **1.5–4.8 × compute penalty per CV fold** for these gains. So the rule is "use MoE when accuracy says yes *and* the extra wall time is acceptable" — it is no longer a strict subset of "regime must be observable from features."

## Benchmark — 500-trial study (naive-lightgbm vs MoE, 5 datasets)

5-fold time-series CV, 500 Optuna trials per (variant × dataset), 5 datasets spanning synthetic-ideal → real macro/financial → controlled-latent. Full report: [`bench_results/study_500_report.md`](bench_results/study_500_report.md). Methodology and dataset-specific recommendations: [docs/moe/benchmark.md](docs/moe/benchmark.md).

| Dataset | Shape | naive-lightgbm best | MoE best | Δ RMSE | Speed (MoE / naive, median train s/fold) |
|---|---|---|---|---|---|
| synthetic | 2000 × 5 | 4.9765 | **4.6651** | −6.3 % | 0.663 / 0.240 = **2.76 ×** |
| real_hamilton | 311 × 12 | 0.9286 | **0.9128** | −1.7 % | 0.122 / 0.055 = **2.22 ×** |
| sp500 | 3761 × 13 | 0.0100 | 0.0100 | tie | 0.136 / 0.091 = **1.49 ×** |
| real_vix | 3762 × 13 | 2.8942 | **2.4574** | **−15.1 %** | 0.386 / 0.081 = **4.77 ×** |
| hmm | 2000 × 5 | 2.1893 | **2.1096** | −3.6 % | 0.126 / 0.074 = **1.70 ×** |

### Datasets

Five datasets spanning the regime-switching applicability spectrum: one MoE-ideal synthetic, three real-world series (the canonical references in the regime-switching literature), and one HMM with known ground-truth labels for measuring regime recovery.

**Synthetic** — *feature-driven regime, MoE-ideal case (2000 × 5)*

Five i.i.d. Gaussian features; the regime is a deterministic function of `X`, so the gate can route perfectly:

```
regime = (0.5·X1 + 0.3·X2 − 0.2·X3 > 0)
y | regime=0 :   5·X0 + 3·X0·X2 + 2·sin(2·X3) + 10  +  ε
y | regime=1 :  −5·X0 − 2·X1²   + 3·cos(2·X4) − 10  +  ε     ε ~ N(0, 0.5²)
```

The two regimes use *opposite-sign* coefficients on the same features — a single GBDT is forced to average them, which is exactly the failure mode MoE is built to avoid. Generator: `generate_synthetic_data` in `examples/benchmark.py`.

**Real Hamilton GDP** — *Hamilton 1989's MS-AR(4) setup on real US macro data (~310 × 12)*

Quarterly US Real GDP (`GDPC1`) fetched from the FRED CSV endpoint. Following [Hamilton (1989, *Econometrica*)](https://www.jstor.org/stable/1912559), the target is the quarterly growth rate `100·Δlog(GDP)`; features are 4 lags of growth (Hamilton's MS-AR(4)) plus engineered MA / volatility / regime-proxy features. The regime (expansion / recession) is genuinely latent — there is no oracle column. Generator: `generate_real_hamilton_gnp_data`.

**S&P 500 daily returns** — *the canonical volatility-regime testbed (~3760 × 13)*

Daily Close of `^GSPC` from Yahoo Finance (default range `2010-01-01` to `2024-12-31`), converted to log returns. Target: next-day log return (a deliberately hard predictive setup). Features: lagged returns at lags {1, 2, 3, 5, 10} plus MA / rolling-volatility / MA-crossover features. Regime is latent (low-vol vs high-vol periods). Generator: `generate_sp500_data`.

**Real VIX** — *implied-volatility level prediction (~3760 × 13)*

Daily Close of CBOE `^VIX` from Yahoo Finance (same date range as S&P). Target: next-day VIX level. Features: lagged VIX + MA / rolling stats. Same latent low-vol / high-vol regime structure as the S&P series, viewed through the implied-vol lens. Generator: `generate_real_vix_data`.

**HMM synthetic** — *3-state Gaussian HMM with known regime labels (2000 × 5)*

Hidden state evolves as a 3-state Markov chain with persistent (95 % diagonal) transitions; emissions are Gaussian with well-separated means (`{−3, 0, +3}`) and varying scales (`{0.4, 0.7, 1.0}`). Two of the five feature columns carry a weak linear signal of the hidden state (so the gate has *some* observable hint, not a free lunch); the remaining three columns are pure noise. **Returns the ground-truth regime labels** — used by `diagnose_moe` to measure regime recovery, not just RMSE. Generator: `generate_hmm_data`.

### Caching

Real-world fetches (FRED, yfinance) are cached under `examples/data_cache/` (gitignored). First run pulls from the network; subsequent runs are offline.

### Settings that won on every dataset

The only categorical setting that produced the absolute best (min) RMSE on all 5 datasets is **`mixture_gate_type='gbdt'`**. For everything else, the optimal value depends on the dataset — full per-dataset table in [docs/moe/benchmark.md](docs/moe/benchmark.md).

| Parameter | Universal? | Notes |
|---|---|---|
| `mixture_gate_type` | **`gbdt`** | Best minimum RMSE on every dataset; `leaf_reuse` and `none` never produced the absolute best |
| `mixture_routing_mode` | **No** | `token_choice` won on synthetic; `expert_choice` won on real_hamilton, real_vix, hmm. Search both. |
| `mixture_num_experts` | weakly 3-4 | Q4 quartile mean is best on most datasets but margin is small |
| `mixture_diversity_lambda` | search 0.0–0.5 | Consistently top-5 in fANOVA importance for MoE; no single best value, but searching it matters |

Dataset-dependent knobs (`mixture_e_step_mode`, `mixture_init`, `mixture_r_smoothing`, `mixture_hard_m_step`, `extra_trees`, `learning_rate`) need per-problem search — see [docs/moe/benchmark.md](docs/moe/benchmark.md) for the full breakdown table.

```bash
# Reproduce the headline study (~25-35 min on 12-core / 24-thread)
python examples/comparative_study.py --trials 500 --out bench_results/study_500.json

# Smoke test (~1 min, all 5 datasets)
python examples/comparative_study.py --trials 10 --n-jobs 2 --out bench_results/smoke.json

# Subset of datasets
python examples/comparative_study.py --trials 500 \
    --datasets synthetic,hmm --out bench_results/study_two.json
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
