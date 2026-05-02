# v0.7 leaf-refit per-config ablation (TPE divergence eliminated)

For each dataset, the v0.6 winning MoE config from the headline ?-trial study is held FIXED across the four refit-mode columns. Only `mixture_refit_*` differs.
5-fold time-series CV; mean ± std across folds.

## Mean validation RMSE (lower better)

| Dataset | refit=off | refit=elbo | refit=every_n=10 | refit=always | Δ elbo vs off | Δ every_n vs off | Δ always vs off |
|---|---|---|---|---|---|---|---|
| `synthetic` | 5.6859 ± 0.8244 | 5.6859 ± 0.8244 | 5.5442 ± 0.7338 | 5.2939 ± 1.1426 | +0.00% | -2.49% | -6.89% |
| `fred_gdp` | 0.9391 ± 0.4281 | 0.9391 ± 0.4281 | 0.9435 ± 0.4363 | 0.9426 ± 0.4374 | +0.00% | +0.47% | +0.38% |
| `sp500_basic` | 0.0101 ± 0.0037 | 0.0101 ± 0.0037 | 0.0101 ± 0.0037 | 0.0101 ± 0.0037 | +0.00% | -0.04% | +0.02% |
| `sp500` | 0.0101 ± 0.0037 | 0.0101 ± 0.0037 | 0.0101 ± 0.0037 | 0.0101 ± 0.0037 | +0.00% | +0.07% | +0.12% |
| `vix` | 2.7660 ± 1.5084 | 2.7660 ± 1.5084 | 2.9156 ± 1.6629 | 2.9587 ± 1.5781 | +0.00% | +5.41% | +6.97% |
| `hmm` | 2.2448 ± 0.2080 | 2.2448 ± 0.2080 | 2.2164 ± 0.2047 | 2.2182 ± 0.2046 | +0.00% | -1.27% | -1.19% |

## Mean training time per fold (seconds)

| Dataset | off | elbo | every_n=10 | always | elbo / off | every_n / off | always / off |
|---|---|---|---|---|---|---|---|
| `synthetic` | 0.05 | 0.04 | 0.09 | 0.59 | 0.88× | 1.90× | 12.24× |
| `fred_gdp` | 0.02 | 0.02 | 0.03 | 0.12 | 0.87× | 1.46× | 5.75× |
| `sp500_basic` | 0.08 | 0.07 | 0.12 | 0.53 | 0.99× | 1.63× | 7.03× |
| `sp500` | 0.12 | 0.12 | 0.16 | 0.54 | 1.00× | 1.31× | 4.52× |
| `vix` | 0.07 | 0.07 | 0.09 | 0.27 | 1.01× | 1.36× | 3.97× |
| `hmm` | 0.02 | 0.02 | 0.07 | 0.35 | 1.06× | 3.23× | 16.46× |

## Summary

- `elbo`: better than off on **0/6** datasets, bit-identical (Δ < 1e-6) on **6/6** (no-fire = no-op as designed)
- `every_n=10`: better than off on **3/6**
- `always`: better than off on **2/6**
