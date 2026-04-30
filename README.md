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
    'mixture_gate_type': 'gbdt',     # Universal winner in the 1000-trial study
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

**MoE wins when the regime is observable from the features** — and only then. The 1000-trial study below makes this concrete:

- **Synthetic** (regime determinable from X): MoE 3.41 vs Standard 4.96 RMSE → **+31% improvement**
- **Hamilton** (latent regime + engineered TS features): MoE 0.6985 vs Standard 0.6990 → tie
- **VIX**: MoE 0.0115 vs Standard 0.0115 → tie

When MoE doesn't lift accuracy, it is also 1.5-1.8× slower per fold. Use MoE when accuracy says yes, not by default.

## Benchmark — 1000-trial study (Standard vs MoE)

5-fold time-series CV, 1000 Optuna trials per (variant × dataset). Full report: [`bench_results/study_1k_report.md`](bench_results/study_1k_report.md). Methodology and dataset-specific recommendations: [docs/moe/benchmark.md](docs/moe/benchmark.md).

| Dataset | Shape | Standard best | MoE best | MoE penalty (speed) |
|---|---|---|---|---|
| Synthetic (feature-driven regime) | 2000 × 5 | 4.96 | **3.41** | 1.09 × |
| Hamilton (latent regime) | 500 × 12 | 0.6990 | 0.6985 | 1.79 × |
| VIX | 1000 × 5 | 0.0115 | 0.0115 | 1.53 × |

### Datasets (synthetic generators in `examples/benchmark.py`)

All three are synthetic with a known regime structure, chosen to span the spectrum from "MoE-ideal" to "latent". Fixed seed (`42`), 5-fold time-series CV.

**Synthetic** — *feature-driven regime, MoE-ideal case (2000 × 5)*

Five i.i.d. Gaussian features. The regime is a deterministic function of `X`, so the gate can route perfectly:

```
regime = (0.5·X1 + 0.3·X2 − 0.2·X3 > 0)
y | regime=0 :   5·X0 + 3·X0·X2 + 2·sin(2·X3) + 10  +  ε
y | regime=1 :  −5·X0 − 2·X1²   + 3·cos(2·X4) − 10  +  ε     ε ~ N(0, 0.5²)
```

The two regimes use *opposite-sign* coefficients on the same features, so a single GBDT must average them — this is exactly what MoE is built to avoid.

**Hamilton** — *latent regime + engineered TS features (500 × 12)*

Hamilton GNP–style: the regime is **latent** (not in the features). It evolves over time as a Bernoulli with sinusoidally-modulated probability `P(regime=1) = 0.5 + 0.3·sin(2π·t/100)`. Targets:

```
y | regime=0 :   0.8 + 0.3·X0 + 0.2·X1  +  ε
y | regime=1 :  −0.5 + 0.1·X0 − 0.3·X2  +  ε                 ε ~ N(0, 0.3²)
```

Four base Gaussian features are augmented with **8 derived time-series features** computed from past `y`: moving averages over windows {5, 10, 20}, rolling stdev over {5, 10}, MA(5)−MA(20) crossover, sign(MA(5)), and rolling fraction of positive `y`. These make the latent regime *partially* observable from history. Even with this engineering, the gate cannot fully separate the regimes — hence the tie with Standard.

**VIX** — *latent volatility regime, small magnitude (1000 × 5)*

VIX-like: a low-volatility / high-volatility regime alternates with `P(high) = 0.3 + 0.4·𝟙[sin(2π·t/200) > 0]`. Targets are positive and small:

```
y | regime=0 :   0.01 + 0.002·|X0|                          +  ε
y | regime=1 :   0.025 + 0.005·|X0| + 0.003·X1²             +  ε     ε ~ N(0, 0.005²)
```

Like Hamilton, the regime is latent but **no TS features are added** here — the only signal the gate has is the noise-dominated `X`. MoE has nothing to route on, so it ties with Standard.

### Settings that won on every dataset (universal)

| Parameter | Recommended |
|---|---|
| `mixture_num_experts` | 3-4 |
| `mixture_gate_type` | `gbdt` |
| `mixture_routing_mode` | `token_choice` |
| `extra_trees` | `true` |
| `mixture_diversity_lambda` | search 0.0–0.5 (top-3 fANOVA importance, no single best value) |

Dataset-dependent knobs (`mixture_e_step_mode`, `mixture_init`, `mixture_r_smoothing`, `mixture_hard_m_step`, `learning_rate`) need per-problem search — see the benchmark doc for the table.

```bash
# Reproduce the headline study (~17 min on 12-core / 24-thread)
python examples/comparative_study.py --trials 1000 --out bench_results/study_1k.json

# Smoke test (~30 s)
python examples/comparative_study.py --trials 30 --out bench_results/smoke.json
```

## Documentation

| Topic | Doc |
|---|---|
| Full parameter reference (MoE core, Gate, Smoothing, Prediction APIs) | [docs/moe/parameters.md](docs/moe/parameters.md) |
| Optuna search templates | [docs/moe/optuna-recipes.md](docs/moe/optuna-recipes.md) |
| 1000-trial benchmark methodology & per-dataset recommendations | [docs/moe/benchmark.md](docs/moe/benchmark.md) |
| Per-expert hyperparameters & role-based recipe | [docs/moe/per-expert-hp.md](docs/moe/per-expert-hp.md) |
| Expert Choice routing | [docs/moe/advanced-routing.md](docs/moe/advanced-routing.md) |
| Progressive training (EvoMoE) & gate temperature annealing | [docs/moe/advanced-progressive.md](docs/moe/advanced-progressive.md) |
| Expert collapse prevention & `diagnose_moe` | [docs/moe/advanced-collapse.md](docs/moe/advanced-collapse.md) |
| SHAP analysis for MoE components | [docs/moe/shap.md](docs/moe/shap.md) |
| `int8` & `use_quantized_grad` compatibility (8-axis matrix) | [docs/moe/int8-compat.md](docs/moe/int8-compat.md) |
| Architecture & EM-loop deep dive | [docs/moe/architecture.md](docs/moe/architecture.md) |

## License

MIT license. Based on [Microsoft LightGBM](https://github.com/microsoft/LightGBM).
