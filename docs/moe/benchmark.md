# Benchmark — 1000-trial Standard vs MoE Study

Headline study: 1000 Optuna trials × 2 variants (Standard, MoE) × 3 datasets (Synthetic, Hamilton, VIX), 5-fold time-series CV, early stopping. Full per-trial dump in [`bench_results/study_1k.json`](../../bench_results/study_1k.json), generated report in [`bench_results/study_1k_report.md`](../../bench_results/study_1k_report.md).

## Accuracy

| Dataset | Shape | Standard best RMSE | MoE best RMSE | Improvement |
|---|---|---|---|---|
| **Synthetic** (feature-driven regime) | 2000 × 5 | 4.96 | **3.41** | **+31 % MoE** |
| Hamilton (latent regime + TS features) | 500 × 12 | 0.6990 | 0.6985 | tie |
| VIX | 1000 × 5 | 0.0115 | 0.0115 | tie |

**MoE only helps when the regime is observable from the features.** On Hamilton (latent regime, even after engineered TS features) and VIX, MoE matches Standard but does not beat it; the extra machinery costs compute without buying accuracy.

## Speed (median train time per CV fold)

| Dataset | Standard | MoE | MoE penalty |
|---|---|---|---|
| Synthetic | 0.231 s | 0.251 s | 1.09 × |
| Hamilton | 0.077 s | 0.138 s | 1.79 × |
| VIX | 0.072 s | 0.110 s | 1.53 × |

So on the two datasets where MoE doesn't lift accuracy, it is also 1.5-1.8× slower per fold. **Use MoE when accuracy says yes, not by default.**

## Recommended hyperparameters from the study

The breakdown below uses **best (min) RMSE per value** rather than mean — the per-variant Optuna run produced a long tail of catastrophic configurations whose mean would mislead.

### Universal — same value won across all 3 datasets (in MoE)

| Parameter | Recommended | Notes |
|---|---|---|
| `mixture_num_experts` | **3-4** | Q4 quartile mean wins on all 3 datasets |
| `mixture_gate_type` | **`gbdt`** | Best minimum RMSE on every dataset; `leaf_reuse` and `none` never produced the absolute best |
| `mixture_routing_mode` | **`token_choice`** | Best minimum RMSE on every dataset |
| `extra_trees` | **`true`** | Best minimum RMSE on every dataset (also Standard's choice on Hamilton/VIX) |
| `mixture_diversity_lambda` | **search 0.0–0.5** | Consistently top-3 in fANOVA importance for MoE (no single best value, but matters) |

### Dataset-dependent — search these per problem

| Parameter | Synthetic | Hamilton | VIX |
|---|---|---|---|
| `mixture_e_step_mode` | `em` | `gate_only` | `gate_only` |
| `mixture_init` | `gmm` | `random` | `gmm` |
| `mixture_r_smoothing` | `markov` | `markov` | `ema` |
| `mixture_hard_m_step` | `true` | `true` | `false` |
| `learning_rate` (best Q) | 0.20-0.24 | 0.10-0.13 | 0.26+ |

### fANOVA importance (top contributors)

For **Standard GBDT**, `min_data_in_leaf` dominates (importance 0.48-0.80 across the 3 datasets), with `learning_rate` distantly second. **A well-tuned `min_data_in_leaf` carries most of the win.**

For **MoE**, the picture is more spread:

| Dataset | Top contributor | Second | Third |
|---|---|---|---|
| Synthetic | `min_data_in_leaf` (0.87) | `mixture_gate_type` (0.034) | `mixture_diversity_lambda` (0.017) |
| Hamilton | `learning_rate` (0.53) | `mixture_diversity_lambda` (0.16) | `bagging_fraction` (0.077) |
| VIX | `lambda_l1` (0.44) | `mixture_diversity_lambda` (0.20) | `mixture_gate_type` (0.088) |

Across all three MoE runs, `mixture_diversity_lambda` is in the top 3 — **searching it is critical, the value is not.**

## Reproduction

```bash
# Full study (~17 min on 12-core / 24-thread machine, n_jobs=6)
python examples/comparative_study.py --trials 1000 --out bench_results/study_1k.json

# Quick smoke check (~30 seconds)
python examples/comparative_study.py --trials 30 --out bench_results/smoke.json

# Subset of datasets
python examples/comparative_study.py --trials 1000 \
    --datasets synthetic,hamilton --out bench_results/two_ds.json
```

The script writes:
- `bench_results/study_1k.json` — full per-trial dump for re-analysis
- `bench_results/study_1k_report.md` — analysis above, automatically generated
- `slice_<dataset>_<variant>.png` — Optuna slice plots (each parameter's value vs RMSE)
