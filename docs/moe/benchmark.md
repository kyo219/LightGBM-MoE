# Benchmark — 500-trial naive-lightgbm vs MoE Study

Headline study: 500 Optuna trials × 2 variants (`naive-lightgbm`, `moe`) × **5 datasets** spanning the regime-switching applicability spectrum (one MoE-ideal synthetic, three real macro/financial series, one HMM with known regime labels). 5-fold time-series CV, early stopping. Full per-trial dump in [`bench_results/study_500.json`](../../bench_results/study_500.json), generated report in [`bench_results/study_500_report.md`](../../bench_results/study_500_report.md).

## Datasets

| Dataset | Source / Generator | Approx. shape | Regime story | Ground truth? |
|---|---|---|---|---|
| **synthetic** | `generate_synthetic_data` (synthetic, K=2, oracle features) | 2000 × 5 | Regime is a deterministic function of `X` — MoE-ideal | yes |
| **fred_gdp** | FRED `GDPC1` via `generate_fred_gdp_data` (Hamilton 1989's MS-AR(4) on 100·Δlog(GDP)) | ~310 × 12 | Quarterly US Real GDP — regime (expansion/recession) is latent | no (latent) |
| **sp500** | yfinance `^GSPC` (2010-2024) via `generate_sp500_data` | ~3760 × 13 | Daily log returns, low-vol / high-vol regimes | no (latent) |
| **vix** | yfinance `^VIX` (2010-2024) via `generate_vix_data` | ~3760 × 13 | Daily VIX level, same vol regimes from the implied-vol angle | no (latent) |
| **hmm** | `generate_hmm_data` (3-state Gaussian HMM, 95 % diagonal transition) | 2000 × 5 | Hidden Markov state with weak observable signal in 2 of 5 features | **yes** |

Real-world fetches are cached under `examples/data_cache/` (gitignored). First run pulls from network; subsequent runs are offline.

The `hmm` dataset is the only one where `diagnose_moe`'s regime-recovery metrics can be validated against ground truth — useful for catching gates that learn to *predict y* without actually *separating regimes*.

## Headline results

| Dataset | naive-lightgbm best RMSE | MoE best RMSE | Δ | naive median train s/fold | MoE median train s/fold | Speed × |
|---|---|---|---|---|---|---|
| synthetic | 4.9765 | **4.6651** | −6.3 % | 0.240 | 0.663 | 2.76 × |
| fred_gdp | 0.9286 | **0.9128** | −1.7 % | 0.055 | 0.122 | 2.22 × |
| sp500 | 0.0100 | 0.0100 | tie | 0.091 | 0.136 | 1.49 × |
| vix | 2.8942 | **2.4574** | **−15.1 %** | 0.081 | 0.386 | 4.77 × |
| hmm | 2.1096 | **2.1096** | −3.6 % | 0.074 | 0.126 | 1.70 × |

See [`bench_results/study_500_report.md`](../../bench_results/study_500_report.md) for the full auto-generated report (median / p10 RMSE, fANOVA importance per dataset, all categorical breakdowns, slice plots).

## Recommended hyperparameters

Below uses **best (min) RMSE per value** rather than mean — Optuna runs always produce a long tail of catastrophic configurations whose mean is misleading. fANOVA importance gives the per-parameter contribution to RMSE variance.

### Universal — produced the best (min) RMSE on every dataset

| Parameter | Recommended | Notes |
|---|---|---|
| `mixture_gate_type` | **`gbdt`** | The only categorical setting whose best value is the same on all 5 datasets. `leaf_reuse` and `none` never produced the absolute best. |
| `mixture_diversity_lambda` | **search 0.0–0.5** | Top-5 fANOVA contributor on every MoE run; no single best value, but searching it consistently matters. |
| `mixture_num_experts` | **3-4** (weak) | Q4 quartile mean is best on most datasets, but the margin is small — within noise on sp500. |

### Dataset-dependent — winning value varies, search per problem

| Parameter | synthetic | fred_gdp | sp500 | vix | hmm |
|---|---|---|---|---|---|
| `mixture_routing_mode` | `token_choice` | `expert_choice` | `token_choice` | `expert_choice` | `expert_choice` |
| `mixture_e_step_mode` | `gate_only`* | `loss_only` | `gate_only` | `gate_only` | `gate_only` |
| `mixture_init` | `tree_hierarchical`* | `gmm` | `random` | `random` | `gmm` |
| `mixture_r_smoothing` | `none`* | `markov` | `markov` | `ema` | `markov` |
| `mixture_hard_m_step` | `False` | `True` | `False` | `True` | `True` |
| `extra_trees` | `False` | `True` | `True` | `False` | `True` |

\* On synthetic, Optuna's TPE happened to find the absolute best inside parameter combinations whose categorical-mean tail was dominated by failures. The "*statistically* best mean" categorical winners on synthetic are different — `mixture_e_step_mode='loss_only'`, `mixture_init='random'`, `mixture_r_smoothing='ema'` (all p<0.01). Treat the synthetic absolute-min row as "what worked once," not as a recommendation. The non-synthetic winners are robust because every value was sampled enough times for the min to be meaningful.

### fANOVA importance: top contributors per (dataset, variant)

| Dataset | Variant | #1 | #2 | #3 |
|---|---|---|---|---|
| synthetic | naive | `min_data_in_leaf` | `learning_rate` | `extra_trees` |
| synthetic | moe | `feature_fraction` (0.66) | `mixture_hard_m_step` (0.16) | `mixture_diversity_lambda` (0.04) |
| fred_gdp | naive | `min_data_in_leaf` (0.80) | `learning_rate` | `feature_fraction` |
| fred_gdp | moe | (see report) | | |
| sp500 | naive | (see report) | | |
| sp500 | moe | (see report) | | |
| vix | naive | `learning_rate` (0.58) | `min_data_in_leaf` (0.37) | `bagging_fraction` |
| vix | moe | `bagging_fraction` (0.25) | `mixture_init` (0.19) | `min_data_in_leaf` (0.16) |
| hmm | naive | `learning_rate` (0.42) | `extra_trees` (0.41) | `min_data_in_leaf` (0.16) |
| hmm | moe | `lambda_l1` (0.55) | `mixture_warmup_iters` (0.07) | `mixture_hard_m_step` (0.07) |

For the canonical Optuna template that bakes in `mixture_gate_type='gbdt'` and searches the rest, see [optuna-recipes.md](optuna-recipes.md).

## Reproduction

```bash
# Full study (~25-35 min on 12-core / 24-thread machine, n_jobs=6)
python examples/comparative_study.py --trials 500 --out bench_results/study_500.json

# Smoke test (~1 min, all 5 datasets)
python examples/comparative_study.py --trials 10 --n-jobs 2 --out bench_results/smoke.json

# Subset of datasets
python examples/comparative_study.py --trials 500 \
    --datasets synthetic,hmm --out bench_results/study_two.json
```

The script writes:

- `bench_results/study_500.json` — full per-trial dump (every trial's RMSE + sampled params, for re-analysis or fANOVA reruns)
- `bench_results/study_500_report.md` — auto-generated analysis (headline table + per-dataset fANOVA + categorical / quartile breakdowns + slice plot links)
- `bench_results/slice_<dataset>_<variant>.png` — Optuna slice plots showing each parameter's value vs RMSE for every (dataset, variant) pair
