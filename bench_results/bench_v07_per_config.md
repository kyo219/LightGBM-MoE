# v0.7 leaf-refit per-config ablation (TPE divergence eliminated)

For each dataset, the v0.6 winning MoE config from the headline ?-trial study is held FIXED across the four refit-mode columns. Only `mixture_refit_*` differs.
5-fold time-series CV; mean ± std across folds.

## Mean validation RMSE (lower better)

| Dataset | refit=off | refit=elbo | refit=every_n=10 | refit=always | Δ elbo vs off | Δ every_n vs off | Δ always vs off |
|---|---|---|---|---|---|---|---|
| `synthetic` | 3.6779 ± 0.6804 | 3.6779 ± 0.6804 | 3.8405 ± 0.7413 | 3.9578 ± 0.6658 | +0.00% | +4.42% | +7.61% |
| `fred_gdp` | 0.9381 ± 0.4362 | 0.9381 ± 0.4362 | 0.9361 ± 0.4353 | 0.9368 ± 0.4309 | +0.00% | -0.21% | -0.13% |
| `sp500_basic` | 0.0100 ± 0.0037 | 0.0100 ± 0.0037 | 0.0100 ± 0.0037 | 0.0100 ± 0.0037 | +0.00% | +0.37% | +0.36% |
| `sp500` | 0.0100 ± 0.0035 | 0.0100 ± 0.0035 | 0.0101 ± 0.0038 | 0.0101 ± 0.0037 | +0.00% | +1.16% | +0.86% |
| `vix` | 2.6745 ± 1.4604 | 2.6745 ± 1.4604 | 2.7814 ± 1.6694 | 2.7837 ± 1.5991 | +0.00% | +4.00% | +4.08% |
| `hmm` | 2.1465 ± 0.2405 | 2.1465 ± 0.2405 | 2.2459 ± 0.2053 | 2.2424 ± 0.2081 | +0.00% | +4.63% | +4.47% |

## Mean training time per fold (seconds)

| Dataset | off | elbo | every_n=10 | always | elbo / off | every_n / off | always / off |
|---|---|---|---|---|---|---|---|
| `synthetic` | 0.07 | 0.07 | 0.12 | 0.65 | 0.92× | 1.74× | 9.22× |
| `fred_gdp` | 0.03 | 0.03 | 0.04 | 0.17 | 1.16× | 1.58× | 6.25× |
| `sp500_basic` | 0.12 | 0.13 | 0.17 | 0.61 | 1.12× | 1.47× | 5.15× |
| `sp500` | 0.16 | 0.15 | 0.28 | 0.68 | 0.95× | 1.79× | 4.32× |
| `vix` | 0.12 | 0.11 | 0.14 | 0.33 | 0.94× | 1.19× | 2.80× |
| `hmm` | 0.08 | 0.08 | 0.16 | 0.84 | 1.01× | 2.09× | 10.89× |

## Summary

- `elbo`: better than off on **0/6** datasets, bit-identical (Δ < 1e-6) on **6/6** (no-fire = no-op as designed)
- `every_n=10`: better than off on **1/6**
- `always`: better than off on **1/6**
