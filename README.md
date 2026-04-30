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
