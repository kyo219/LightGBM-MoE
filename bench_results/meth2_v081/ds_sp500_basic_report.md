# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 300

- **Datasets**: ['sp500_basic'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `d25c06cf3b86`, lib sha256 `a3c30fcc7fd3…`, package `/home/mokumoku/Develop/LightGBM-MoE/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| sp500_basic | naive-lightgbm | **0.01104** ± 0.00001 | 0.00969 | 0.0% | 0.03 |
| sp500_basic | naive-ensemble | **0.01103** ± 0.00001 | 0.00970 | 0.0% | 0.09 |
| sp500_basic | moe | **0.01102** ± 0.00000 | 0.00970 | 0.0% | 0.25 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| sp500_basic@s42 | naive-lightgbm | 0.0097 | 0.0097 | 0.022 | 38 |
| sp500_basic@s42 | naive-ensemble | 0.0097 | 0.0097 | 0.053 | 90 |
| sp500_basic@s42 | moe | 0.0097 | 0.0097 | 0.147 | 312 |
| sp500_basic@s43 | naive-lightgbm | 0.0097 | 0.0097 | 0.021 | 38 |
| sp500_basic@s43 | naive-ensemble | 0.0097 | 0.0097 | 0.069 | 112 |
| sp500_basic@s43 | moe | 0.0097 | 0.0097 | 0.103 | 222 |
| sp500_basic@s44 | naive-lightgbm | 0.0097 | 0.0097 | 0.025 | 43 |
| sp500_basic@s44 | naive-ensemble | 0.0097 | 0.0097 | 0.049 | 87 |
| sp500_basic@s44 | moe | 0.0097 | 0.0097 | 0.180 | 390 |



---

## sp500_basic@s42  (search X=[3010, 13], holdout n=752)


### naive-lightgbm

- **holdout RMSE: 0.01103** (winner retrained in 0.03s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0097, p10: 0.0097
- train: median 0.022s/fold, mean 0.022s, p90 0.028s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.561 |
| `learning_rate` | 0.184 |
| `lambda_l2` | 0.099 |
| `feature_fraction` | 0.082 |
| `extra_trees` | 0.031 |
| `bagging_fraction` | 0.020 |
| `num_leaves` | 0.011 |
| `bagging_freq` | 0.010 |
| `max_depth` | 0.002 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 280 | 0.0097 | 0.0000 | 0.0097 |
| False | 20 | 0.0097 | 0.0000 | 0.0097 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0464] |
| `num_leaves` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 19.0] |
| `max_depth` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 4.0] |
| `min_data_in_leaf` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 59.0] |
| `lambda_l1` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0004] |
| `lambda_l2` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0] |
| `feature_fraction` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.6917] |
| `bagging_fraction` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.6112] |

#### E. Slice plot

![sp500_basic@s42/naive-lightgbm](slice_sp500_basic@s42_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 0.01103** (winner retrained in 0.07s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0097, p10: 0.0097
- train: median 0.053s/fold, mean 0.056s, p90 0.074s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.460 |
| `min_data_in_leaf` | 0.221 |
| `extra_trees` | 0.147 |
| `bagging_fraction` | 0.052 |
| `feature_fraction` | 0.048 |
| `num_leaves` | 0.039 |
| `bagging_freq` | 0.021 |
| `max_depth` | 0.011 |
| `lambda_l2` | 0.001 |
| `n_models` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 22 | 0.0097 | 0.0000 | 0.0097 |
| True | 278 | 0.0097 | 0.0000 | 0.0097 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0327] |
| `num_leaves` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 43.0] |
| `max_depth` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 8.0] |
| `min_data_in_leaf` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 76.0] |
| `lambda_l1` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0] |
| `lambda_l2` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0006] |
| `feature_fraction` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.6492] |
| `bagging_fraction` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.8138] |

#### E. Slice plot

![sp500_basic@s42/naive-ensemble](slice_sp500_basic@s42_naive-ensemble.png)


### moe

- **holdout RMSE: 0.01101** (winner retrained in 0.19s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0097, p10: 0.0097
- train: median 0.147s/fold, mean 0.200s, p90 0.296s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `feature_fraction` | 0.234 |
| `extra_trees` | 0.230 |
| `mixture_hard_m_step` | 0.081 |
| `bagging_fraction` | 0.075 |
| `mixture_warmup_iters` | 0.053 |
| `mixture_gate_type` | 0.048 |
| `mixture_num_experts` | 0.041 |
| `lambda_l2` | 0.035 |
| `bagging_freq` | 0.029 |
| `min_data_in_leaf` | 0.027 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_routing_mode` | **token_choice** | 0.0097 (n=226) | expert_choice | Δ +0.0000 | p=0.00e+00 |
| `mixture_e_step_mode` | **loss_only** | 0.0097 (n=69) | gate_only | Δ +0.0000 | p=0.00e+00 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 36 | 0.0097 | 0.0000 | 0.0097 |
| none | 16 | 0.0097 | 0.0000 | 0.0097 |
| gbdt | 248 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 226 | 0.0097 | 0.0000 | 0.0097 |
| expert_choice | 74 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 69 | 0.0097 | 0.0000 | 0.0097 |
| gate_only | 215 | 0.0097 | 0.0000 | 0.0097 |
| em | 16 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| random | 16 | 0.0097 | 0.0000 | 0.0097 |
| gmm | 254 | 0.0097 | 0.0000 | 0.0097 |
| tree_hierarchical | 16 | 0.0097 | 0.0000 | 0.0097 |
| uniform | 14 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 36 | 0.0097 | 0.0000 | 0.0097 |
| markov | 16 | 0.0097 | 0.0001 | 0.0097 |
| none | 248 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 279 | 0.0097 | 0.0000 | 0.0097 |
| False | 21 | 0.0098 | 0.0000 | 0.0097 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 279 | 0.0097 | 0.0000 | 0.0097 |
| False | 21 | 0.0098 | 0.0000 | 0.0097 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.0098 | — | — | 0.0097 | **Q4** [3.0, ∞) |
| `mixture_diversity_lambda` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.3292] |
| `mixture_warmup_iters` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 24.0] |
| `mixture_balance_factor` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 5.0] |
| `learning_rate` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0462] |
| `num_leaves` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 67.0] |
| `max_depth` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 6.0] |
| `min_data_in_leaf` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 64.0] |

#### E. Slice plot

![sp500_basic@s42/moe](slice_sp500_basic@s42_moe.png)


---

## sp500_basic@s43  (search X=[3010, 13], holdout n=752)


### naive-lightgbm

- **holdout RMSE: 0.01105** (winner retrained in 0.04s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0097, p10: 0.0097
- train: median 0.021s/fold, mean 0.022s, p90 0.027s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.508 |
| `min_data_in_leaf` | 0.275 |
| `learning_rate` | 0.078 |
| `bagging_fraction` | 0.046 |
| `feature_fraction` | 0.032 |
| `num_leaves` | 0.030 |
| `max_depth` | 0.014 |
| `bagging_freq` | 0.013 |
| `lambda_l2` | 0.003 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 279 | 0.0097 | 0.0000 | 0.0097 |
| False | 21 | 0.0097 | 0.0000 | 0.0097 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.1027] |
| `num_leaves` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 87.5] |
| `max_depth` | 0.0097 | 0.0097 | — | 0.0097 | **Q1** [None, 10.0] |
| `min_data_in_leaf` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 66.0] |
| `lambda_l1` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0] |
| `lambda_l2` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0] |
| `feature_fraction` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.5668] |
| `bagging_fraction` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.5882] |

#### E. Slice plot

![sp500_basic@s43/naive-lightgbm](slice_sp500_basic@s43_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 0.01104** (winner retrained in 0.10s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0097, p10: 0.0097
- train: median 0.069s/fold, mean 0.071s, p90 0.088s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.339 |
| `bagging_fraction` | 0.333 |
| `feature_fraction` | 0.164 |
| `min_data_in_leaf` | 0.089 |
| `learning_rate` | 0.022 |
| `num_leaves` | 0.020 |
| `max_depth` | 0.013 |
| `bagging_freq` | 0.011 |
| `n_models` | 0.004 |
| `lambda_l2` | 0.002 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 21 | 0.0097 | 0.0000 | 0.0097 |
| True | 279 | 0.0097 | 0.0000 | 0.0097 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.1532] |
| `num_leaves` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 23.0] |
| `max_depth` | 0.0097 | 0.0097 | — | 0.0097 | **Q1** [None, 5.0] |
| `min_data_in_leaf` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 57.0] |
| `lambda_l1` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0004] |
| `lambda_l2` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0] |
| `feature_fraction` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.5239] |
| `bagging_fraction` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.5522] |

#### E. Slice plot

![sp500_basic@s43/naive-ensemble](slice_sp500_basic@s43_naive-ensemble.png)


### moe

- **holdout RMSE: 0.01102** (winner retrained in 0.18s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0097, p10: 0.0097
- train: median 0.103s/fold, mean 0.140s, p90 0.235s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `max_depth` | 0.192 |
| `num_leaves` | 0.144 |
| `mixture_r_smoothing` | 0.132 |
| `mixture_balance_factor` | 0.096 |
| `mixture_refit_leaves` | 0.089 |
| `mixture_routing_mode` | 0.088 |
| `learning_rate` | 0.039 |
| `mixture_diversity_lambda` | 0.035 |
| `feature_fraction` | 0.023 |
| `mixture_warmup_iters` | 0.022 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_r_smoothing` | **ema** | 0.0097 (n=240) | none | Δ +0.0001 | p=4.30e-05 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 49 | 0.0097 | 0.0000 | 0.0097 |
| none | 235 | 0.0097 | 0.0000 | 0.0097 |
| leaf_reuse | 16 | 0.0098 | 0.0000 | 0.0097 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 279 | 0.0097 | 0.0000 | 0.0097 |
| token_choice | 21 | 0.0098 | 0.0000 | 0.0097 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| em | 235 | 0.0097 | 0.0000 | 0.0097 |
| gate_only | 16 | 0.0097 | 0.0000 | 0.0097 |
| loss_only | 49 | 0.0098 | 0.0000 | 0.0097 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gmm | 24 | 0.0097 | 0.0000 | 0.0097 |
| tree_hierarchical | 240 | 0.0097 | 0.0000 | 0.0097 |
| random | 15 | 0.0097 | 0.0000 | 0.0097 |
| uniform | 21 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 240 | 0.0097 | 0.0000 | 0.0097 |
| none | 44 | 0.0098 | 0.0000 | 0.0097 |
| markov | 16 | 0.0098 | 0.0000 | 0.0097 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 258 | 0.0097 | 0.0000 | 0.0097 |
| False | 42 | 0.0097 | 0.0000 | 0.0097 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 277 | 0.0097 | 0.0000 | 0.0097 |
| False | 23 | 0.0098 | 0.0000 | 0.0097 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.0097 | — | — | 0.0097 | **Q1** [None, 4.0] |
| `mixture_diversity_lambda` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0668] |
| `mixture_warmup_iters` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 43.0] |
| `mixture_balance_factor` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 4.75] |
| `learning_rate` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0706] |
| `num_leaves` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 52.5] |
| `max_depth` | 0.0097 | — | 0.0097 | 0.0097 | **Q1** [None, 10.0] |
| `min_data_in_leaf` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 76.0] |

#### E. Slice plot

![sp500_basic@s43/moe](slice_sp500_basic@s43_moe.png)


---

## sp500_basic@s44  (search X=[3010, 13], holdout n=752)


### naive-lightgbm

- **holdout RMSE: 0.01102** (winner retrained in 0.03s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0097, p10: 0.0097
- train: median 0.025s/fold, mean 0.025s, p90 0.030s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.399 |
| `min_data_in_leaf` | 0.307 |
| `feature_fraction` | 0.177 |
| `bagging_freq` | 0.037 |
| `extra_trees` | 0.029 |
| `bagging_fraction` | 0.024 |
| `num_leaves` | 0.015 |
| `max_depth` | 0.010 |
| `lambda_l2` | 0.001 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 36 | 0.0097 | 0.0000 | 0.0097 |
| True | 264 | 0.0097 | 0.0000 | 0.0097 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0144] |
| `num_leaves` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 60.0] |
| `max_depth` | — | — | 0.0097 | 0.0097 | **Q3** [3.0, 4.25] |
| `min_data_in_leaf` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 89.0] |
| `lambda_l1` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0] |
| `lambda_l2` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0] |
| `feature_fraction` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.518] |
| `bagging_fraction` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.8957] |

#### E. Slice plot

![sp500_basic@s44/naive-lightgbm](slice_sp500_basic@s44_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 0.01101** (winner retrained in 0.09s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0097, p10: 0.0097
- train: median 0.049s/fold, mean 0.055s, p90 0.070s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.733 |
| `extra_trees` | 0.068 |
| `bagging_fraction` | 0.057 |
| `learning_rate` | 0.052 |
| `lambda_l2` | 0.032 |
| `feature_fraction` | 0.018 |
| `num_leaves` | 0.017 |
| `bagging_freq` | 0.015 |
| `n_models` | 0.004 |
| `max_depth` | 0.004 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 279 | 0.0097 | 0.0000 | 0.0097 |
| False | 21 | 0.0097 | 0.0000 | 0.0097 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0552] |
| `num_leaves` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 62.0] |
| `max_depth` | 0.0097 | — | 0.0097 | 0.0097 | **Q1** [None, 10.0] |
| `min_data_in_leaf` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 64.0] |
| `lambda_l1` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0] |
| `lambda_l2` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0002] |
| `feature_fraction` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.52] |
| `bagging_fraction` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.6034] |

#### E. Slice plot

![sp500_basic@s44/naive-ensemble](slice_sp500_basic@s44_naive-ensemble.png)


### moe

- **holdout RMSE: 0.01101** (winner retrained in 0.37s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0097, p10: 0.0097
- train: median 0.180s/fold, mean 0.252s, p90 0.268s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.308 |
| `bagging_fraction` | 0.129 |
| `mixture_balance_factor` | 0.101 |
| `num_leaves` | 0.084 |
| `mixture_diversity_lambda` | 0.073 |
| `learning_rate` | 0.064 |
| `extra_trees` | 0.064 |
| `feature_fraction` | 0.046 |
| `max_depth` | 0.036 |
| `mixture_warmup_iters` | 0.025 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_routing_mode` | **token_choice** | 0.0097 (n=125) | expert_choice | Δ +0.0000 | p=0.00e+00 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 220 | 0.0097 | 0.0000 | 0.0097 |
| none | 16 | 0.0097 | 0.0000 | 0.0097 |
| leaf_reuse | 64 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 125 | 0.0097 | 0.0000 | 0.0097 |
| expert_choice | 175 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 20 | 0.0097 | 0.0000 | 0.0097 |
| em | 48 | 0.0097 | 0.0000 | 0.0097 |
| loss_only | 232 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 22 | 0.0097 | 0.0000 | 0.0097 |
| tree_hierarchical | 216 | 0.0097 | 0.0000 | 0.0097 |
| random | 48 | 0.0097 | 0.0000 | 0.0097 |
| gmm | 14 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 23 | 0.0097 | 0.0000 | 0.0097 |
| ema | 232 | 0.0097 | 0.0000 | 0.0097 |
| markov | 45 | 0.0098 | 0.0001 | 0.0097 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 272 | 0.0097 | 0.0000 | 0.0097 |
| True | 28 | 0.0098 | 0.0001 | 0.0097 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 279 | 0.0097 | 0.0000 | 0.0097 |
| False | 21 | 0.0098 | 0.0000 | 0.0097 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | — | — | 0.0097 | **Q4** [2.0, ∞) |
| `mixture_diversity_lambda` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.1929] |
| `mixture_warmup_iters` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 37.0] |
| `mixture_balance_factor` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 6.0] |
| `learning_rate` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.015] |
| `num_leaves` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 87.0] |
| `max_depth` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 6.0] |
| `min_data_in_leaf` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 55.0] |

#### E. Slice plot

![sp500_basic@s44/moe](slice_sp500_basic@s44_moe.png)


---

## Overall recommendations

**Categorical settings that are statistically significant winners (p<0.01):**

| dataset | param | best value | Δ vs runner-up | p |
|---|---|---|---|---|
| sp500_basic@s42 | `mixture_routing_mode` | **token_choice** | +0.0000 | 0.00e+00 |
| sp500_basic@s42 | `mixture_e_step_mode` | **loss_only** | +0.0000 | 0.00e+00 |
| sp500_basic@s43 | `mixture_r_smoothing` | **ema** | +0.0001 | 4.30e-05 |
| sp500_basic@s44 | `mixture_routing_mode` | **token_choice** | +0.0000 | 0.00e+00 |
