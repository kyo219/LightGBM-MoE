# v0.7 leaf-refit ablation — 500-trial / 6-dataset study

**? Optuna trials per (variant × dataset), 5-fold time-series CV.**
Refit ablation runs the same MoE search space with `mixture_refit_leaves=True` fixed; only the trigger differs:

- `moe` (baseline): v0.6 behavior, refit off
- `moe-refit-elbo`: refit fires only when the per-10-iter ELBO log shows >5% drop
- `moe-refit-every_n`: refit fires every 10 post-warmup iters

## Best RMSE per dataset

| Dataset | naive | ensemble | **moe (off)** | moe-refit-elbo | moe-refit-every_n | Δ elbo vs off | Δ every_n vs off |
|---|---|---|---|---|---|---|---|
| `fred_gdp` | 0.9311 | 0.9094 | **0.9381** | 0.9381 | 0.9292 | +0.00% | -0.94% |
| `hmm` | 2.1913 | 2.1818 | **2.1465** | 2.1465 | 2.1202 | +0.00% | -1.23% |
| `sp500` | 0.0100 | 0.0100 | **0.0100** | 0.0100 | 0.0101 | +0.15% | +0.33% |
| `sp500_basic` | 0.0100 | 0.0100 | **0.0100** | 0.0099 | 0.0100 | -0.34% | +0.34% |
| `synthetic` | 5.0233 | 4.8899 | **3.6779** | 4.3544 | 4.8498 | +18.39% | +31.86% |
| `vix` | 2.8869 | 2.8724 | **2.6745** | 2.7134 | 2.7190 | +1.46% | +1.66% |

## Cost (median per-fold training, seconds)

| Dataset | moe (off) | moe-refit-elbo | moe-refit-every_n | elbo / off | every_n / off |
|---|---|---|---|---|---|
| `fred_gdp` | 0.036 | 0.037 | 0.035 | 1.02× | 0.96× |
| `hmm` | 0.049 | 0.076 | 0.072 | 1.54× | 1.46× |
| `sp500` | 0.075 | 0.095 | 0.158 | 1.27× | 2.13× |
| `sp500_basic` | 0.190 | 0.171 | 0.091 | 0.90× | 0.48× |
| `synthetic` | 0.073 | 0.170 | 0.446 | 2.32× | 6.10× |
| `vix` | 0.063 | 0.046 | 0.129 | 0.72× | 2.03× |

## Wall-clock budget per variant (seconds, all 500 trials)

| Dataset | moe (off) | moe-refit-elbo | moe-refit-every_n |
|---|---|---|---|
| `fred_gdp` | 136 | 139 | 168 |
| `hmm` | 147 | 229 | 237 |
| `sp500` | 248 | 296 | 471 |
| `sp500_basic` | 499 | 458 | 319 |
| `synthetic` | 240 | 465 | 1125 |
| `vix` | 182 | 189 | 416 |

## Acceptance summary (issue #37)

- elbo trigger: better than off on **1/6** datasets, within ±1% on **4/6**

- every_n trigger: better than off on **2/6** datasets
