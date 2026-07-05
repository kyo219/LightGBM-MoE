# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 300

- **Datasets**: ['sp500_basic'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `d25c06cf3b86`, lib sha256 `5cec0a0bd5ab…`, package `/tmp/lgbm-moe-v080/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| sp500_basic | naive-lightgbm | **0.01104** ± 0.00001 | 0.00969 | 0.0% | 0.04 |
| sp500_basic | moe | **0.01104** ± 0.00001 | 0.00969 | 0.0% | 0.15 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| sp500_basic@s42 | naive-lightgbm | 0.0097 | 0.0097 | 0.022 | 38 |
| sp500_basic@s42 | moe | 0.0097 | 0.0097 | 0.060 | 240 |
| sp500_basic@s43 | naive-lightgbm | 0.0097 | 0.0097 | 0.021 | 38 |
| sp500_basic@s43 | moe | 0.0097 | 0.0097 | 0.108 | 299 |
| sp500_basic@s44 | naive-lightgbm | 0.0097 | 0.0097 | 0.025 | 43 |
| sp500_basic@s44 | moe | 0.0097 | 0.0097 | 0.148 | 296 |



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
| `min_data_in_leaf` | 0.535 |
| `learning_rate` | 0.170 |
| `feature_fraction` | 0.112 |
| `lambda_l2` | 0.066 |
| `bagging_fraction` | 0.038 |
| `extra_trees` | 0.035 |
| `num_leaves` | 0.022 |
| `bagging_freq` | 0.017 |
| `max_depth` | 0.005 |
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


### moe

- **holdout RMSE: 0.01106** (winner retrained in 0.08s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0097, p10: 0.0097
- train: median 0.060s/fold, mean 0.152s, p90 0.217s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.374 |
| `mixture_init` | 0.160 |
| `learning_rate` | 0.128 |
| `mixture_warmup_iters` | 0.089 |
| `feature_fraction` | 0.088 |
| `bagging_fraction` | 0.043 |
| `mixture_diversity_lambda` | 0.029 |
| `mixture_e_step_mode` | 0.017 |
| `num_leaves` | 0.016 |
| `mixture_routing_mode` | 0.013 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_routing_mode` | **token_choice** | 0.0097 (n=142) | expert_choice | Δ +0.0000 | p=2.16e-03 |
| `mixture_init` | **random** | 0.0097 (n=255) | gmm | Δ +0.0000 | p=1.90e-05 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 33 | 0.0097 | 0.0000 | 0.0097 |
| none | 215 | 0.0097 | 0.0000 | 0.0097 |
| gbdt | 52 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 142 | 0.0097 | 0.0000 | 0.0097 |
| expert_choice | 158 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 138 | 0.0097 | 0.0000 | 0.0097 |
| gate_only | 21 | 0.0097 | 0.0000 | 0.0097 |
| em | 141 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| random | 255 | 0.0097 | 0.0000 | 0.0097 |
| gmm | 15 | 0.0097 | 0.0000 | 0.0097 |
| uniform | 14 | 0.0097 | 0.0000 | 0.0097 |
| tree_hierarchical | 16 | 0.0098 | 0.0000 | 0.0097 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 48 | 0.0097 | 0.0000 | 0.0097 |
| markov | 28 | 0.0097 | 0.0000 | 0.0097 |
| none | 224 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 276 | 0.0097 | 0.0000 | 0.0097 |
| False | 24 | 0.0097 | 0.0000 | 0.0097 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 278 | 0.0097 | 0.0000 | 0.0097 |
| False | 22 | 0.0097 | 0.0000 | 0.0097 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.0097 | — | — | 0.0097 | **Q1** [None, 3.0] |
| `mixture_diversity_lambda` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.2111] |
| `mixture_warmup_iters` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 31.0] |
| `mixture_balance_factor` | 0.0097 | — | — | 0.0097 | **Q1** [None, 5.0] |
| `learning_rate` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.089] |
| `num_leaves` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 16.0] |
| `max_depth` | 0.0097 | 0.0097 | — | 0.0097 | **Q1** [None, 11.0] |
| `min_data_in_leaf` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 68.0] |

#### E. Slice plot

![sp500_basic@s42/moe](slice_sp500_basic@s42_moe.png)


---

## sp500_basic@s43  (search X=[3010, 13], holdout n=752)


### naive-lightgbm

- **holdout RMSE: 0.01105** (winner retrained in 0.04s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0097, p10: 0.0097
- train: median 0.021s/fold, mean 0.022s, p90 0.029s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.497 |
| `min_data_in_leaf` | 0.244 |
| `learning_rate` | 0.080 |
| `bagging_fraction` | 0.063 |
| `num_leaves` | 0.033 |
| `max_depth` | 0.025 |
| `feature_fraction` | 0.025 |
| `bagging_freq` | 0.023 |
| `lambda_l2` | 0.010 |
| `lambda_l1` | 0.002 |

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


### moe

- **holdout RMSE: 0.01104** (winner retrained in 0.16s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0097, p10: 0.0097
- train: median 0.108s/fold, mean 0.191s, p90 0.335s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `num_leaves` | 0.526 |
| `lambda_l2` | 0.224 |
| `mixture_init` | 0.039 |
| `extra_trees` | 0.035 |
| `bagging_fraction` | 0.033 |
| `min_data_in_leaf` | 0.026 |
| `mixture_diversity_lambda` | 0.024 |
| `mixture_gate_type` | 0.017 |
| `learning_rate` | 0.015 |
| `feature_fraction` | 0.014 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 266 | 0.0097 | 0.0000 | 0.0097 |
| none | 17 | 0.0097 | 0.0000 | 0.0097 |
| leaf_reuse | 17 | 0.0098 | 0.0000 | 0.0097 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 237 | 0.0097 | 0.0000 | 0.0097 |
| token_choice | 63 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 197 | 0.0097 | 0.0000 | 0.0097 |
| em | 86 | 0.0097 | 0.0000 | 0.0097 |
| gate_only | 17 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gmm | 237 | 0.0097 | 0.0000 | 0.0097 |
| tree_hierarchical | 33 | 0.0097 | 0.0001 | 0.0097 |
| random | 15 | 0.0097 | 0.0000 | 0.0097 |
| uniform | 15 | 0.0098 | 0.0000 | 0.0097 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 106 | 0.0097 | 0.0000 | 0.0097 |
| none | 38 | 0.0097 | 0.0000 | 0.0097 |
| markov | 156 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 56 | 0.0097 | 0.0000 | 0.0097 |
| False | 244 | 0.0097 | 0.0000 | 0.0097 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 277 | 0.0097 | 0.0000 | 0.0097 |
| False | 23 | 0.0098 | 0.0000 | 0.0097 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | — | — | 0.0097 | **Q4** [2.0, ∞) |
| `mixture_diversity_lambda` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.1447] |
| `mixture_warmup_iters` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 12.0] |
| `mixture_balance_factor` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 8.0] |
| `learning_rate` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.1097] |
| `num_leaves` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 27.0] |
| `max_depth` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 5.0] |
| `min_data_in_leaf` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 59.75] |

#### E. Slice plot

![sp500_basic@s43/moe](slice_sp500_basic@s43_moe.png)


---

## sp500_basic@s44  (search X=[3010, 13], holdout n=752)


### naive-lightgbm

- **holdout RMSE: 0.01102** (winner retrained in 0.04s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0097, p10: 0.0097
- train: median 0.025s/fold, mean 0.025s, p90 0.029s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.407 |
| `min_data_in_leaf` | 0.264 |
| `feature_fraction` | 0.139 |
| `max_depth` | 0.074 |
| `bagging_fraction` | 0.040 |
| `num_leaves` | 0.036 |
| `extra_trees` | 0.025 |
| `bagging_freq` | 0.009 |
| `lambda_l2` | 0.005 |
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


### moe

- **holdout RMSE: 0.01102** (winner retrained in 0.22s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0097, p10: 0.0097
- train: median 0.148s/fold, mean 0.189s, p90 0.290s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_init` | 0.203 |
| `mixture_diversity_lambda` | 0.163 |
| `feature_fraction` | 0.137 |
| `extra_trees` | 0.137 |
| `num_leaves` | 0.105 |
| `min_data_in_leaf` | 0.094 |
| `mixture_warmup_iters` | 0.030 |
| `learning_rate` | 0.027 |
| `bagging_fraction` | 0.021 |
| `mixture_balance_factor` | 0.020 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 215 | 0.0097 | 0.0000 | 0.0097 |
| none | 26 | 0.0097 | 0.0000 | 0.0097 |
| leaf_reuse | 59 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 182 | 0.0097 | 0.0000 | 0.0097 |
| expert_choice | 118 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 59 | 0.0097 | 0.0000 | 0.0097 |
| em | 167 | 0.0097 | 0.0000 | 0.0097 |
| loss_only | 74 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 16 | 0.0097 | 0.0000 | 0.0097 |
| random | 56 | 0.0097 | 0.0000 | 0.0097 |
| gmm | 214 | 0.0097 | 0.0000 | 0.0097 |
| tree_hierarchical | 14 | 0.0098 | 0.0000 | 0.0097 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 51 | 0.0097 | 0.0000 | 0.0097 |
| ema | 211 | 0.0097 | 0.0000 | 0.0097 |
| markov | 38 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 30 | 0.0097 | 0.0000 | 0.0097 |
| False | 270 | 0.0097 | 0.0000 | 0.0097 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 279 | 0.0097 | 0.0000 | 0.0097 |
| False | 21 | 0.0098 | 0.0000 | 0.0097 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.0097 | — | — | 0.0097 | **Q1** [None, 3.0] |
| `mixture_diversity_lambda` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.1376] |
| `mixture_warmup_iters` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 41.0] |
| `mixture_balance_factor` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 5.0] |
| `learning_rate` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.129] |
| `num_leaves` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 87.0] |
| `max_depth` | 0.0097 | — | 0.0097 | 0.0097 | **Q1** [None, 8.0] |
| `min_data_in_leaf` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 57.0] |

#### E. Slice plot

![sp500_basic@s44/moe](slice_sp500_basic@s44_moe.png)


---

## Overall recommendations

**Categorical settings that are statistically significant winners (p<0.01):**

| dataset | param | best value | Δ vs runner-up | p |
|---|---|---|---|---|
| sp500_basic@s42 | `mixture_routing_mode` | **token_choice** | +0.0000 | 2.16e-03 |
| sp500_basic@s42 | `mixture_init` | **random** | +0.0000 | 1.90e-05 |
