# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 300

- **Datasets**: ['sp500'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `d25c06cf3b86`, lib sha256 `5cec0a0bd5ab…`, package `/tmp/lgbm-moe-v080/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| sp500 | naive-lightgbm | **0.01105** ± 0.00001 | 0.00975 | 0.0% | 0.04 |
| sp500 | moe | **0.01104** ± 0.00002 | 0.00976 | 0.0% | 0.31 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| sp500@s42 | naive-lightgbm | 0.0098 | 0.0098 | 0.022 | 39 |
| sp500@s42 | moe | 0.0098 | 0.0098 | 0.250 | 432 |
| sp500@s43 | naive-lightgbm | 0.0097 | 0.0098 | 0.022 | 44 |
| sp500@s43 | moe | 0.0097 | 0.0098 | 0.261 | 638 |
| sp500@s44 | naive-lightgbm | 0.0097 | 0.0098 | 0.026 | 45 |
| sp500@s44 | moe | 0.0098 | 0.0098 | 0.091 | 209 |



---

## sp500@s42  (search X=[2969, 28], holdout n=742)


### naive-lightgbm

- **holdout RMSE: 0.01103** (winner retrained in 0.06s, cv score of winner: 0.0098)
- cv best RMSE: 0.0098, median: 0.0098, p10: 0.0098
- train: median 0.022s/fold, mean 0.023s, p90 0.029s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.475 |
| `num_leaves` | 0.212 |
| `min_data_in_leaf` | 0.164 |
| `max_depth` | 0.075 |
| `bagging_fraction` | 0.033 |
| `extra_trees` | 0.021 |
| `feature_fraction` | 0.016 |
| `bagging_freq` | 0.004 |
| `lambda_l2` | 0.000 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 280 | 0.0098 | 0.0000 | 0.0098 |
| False | 20 | 0.0098 | 0.0000 | 0.0098 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0459] |
| `num_leaves` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 17.0] |
| `max_depth` | 0.0098 | 0.0098 | — | 0.0098 | **Q1** [None, 6.0] |
| `min_data_in_leaf` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 82.0] |
| `lambda_l1` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0] |
| `lambda_l2` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0] |
| `feature_fraction` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.6921] |
| `bagging_fraction` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.6104] |

#### E. Slice plot

![sp500@s42/naive-lightgbm](slice_sp500@s42_naive-lightgbm.png)


### moe

- **holdout RMSE: 0.01102** (winner retrained in 0.65s, cv score of winner: 0.0098)
- cv best RMSE: 0.0098, median: 0.0098, p10: 0.0098
- train: median 0.250s/fold, mean 0.280s, p90 0.327s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_diversity_lambda` | 0.562 |
| `mixture_init` | 0.137 |
| `learning_rate` | 0.053 |
| `bagging_fraction` | 0.051 |
| `min_data_in_leaf` | 0.045 |
| `feature_fraction` | 0.027 |
| `mixture_balance_factor` | 0.022 |
| `mixture_gate_type` | 0.016 |
| `num_leaves` | 0.013 |
| `mixture_num_experts` | 0.013 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **leaf_reuse** | 0.0098 (n=210) | none | Δ +0.0000 | p=5.74e-03 |
| `mixture_routing_mode` | **token_choice** | 0.0098 (n=102) | expert_choice | Δ +0.0000 | p=1.46e-04 |
| `mixture_e_step_mode` | **loss_only** | 0.0098 (n=203) | gate_only | Δ +0.0000 | p=3.28e-03 |
| `mixture_r_smoothing` | **ema** | 0.0098 (n=103) | markov | Δ +0.0000 | p=5.16e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 210 | 0.0098 | 0.0000 | 0.0098 |
| none | 40 | 0.0098 | 0.0000 | 0.0098 |
| gbdt | 50 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 102 | 0.0098 | 0.0000 | 0.0098 |
| expert_choice | 198 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 203 | 0.0098 | 0.0000 | 0.0098 |
| gate_only | 76 | 0.0098 | 0.0000 | 0.0098 |
| em | 21 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| random | 25 | 0.0098 | 0.0000 | 0.0098 |
| gmm | 15 | 0.0098 | 0.0000 | 0.0098 |
| tree_hierarchical | 20 | 0.0098 | 0.0000 | 0.0098 |
| uniform | 240 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 103 | 0.0098 | 0.0000 | 0.0098 |
| markov | 104 | 0.0098 | 0.0000 | 0.0098 |
| none | 93 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 274 | 0.0098 | 0.0000 | 0.0098 |
| False | 26 | 0.0098 | 0.0000 | 0.0098 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 260 | 0.0098 | 0.0000 | 0.0098 |
| False | 40 | 0.0098 | 0.0000 | 0.0098 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.0098 | — | — | 0.0098 | **Q1** [None, 4.0] |
| `mixture_diversity_lambda` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.1379] |
| `mixture_warmup_iters` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 10.0] |
| `mixture_balance_factor` | 0.0098 | 0.0098 | — | 0.0098 | **Q1** [None, 5.0] |
| `learning_rate` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0559] |
| `num_leaves` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 94.0] |
| `max_depth` | — | — | 0.0098 | 0.0098 | **Q3** [3.0, 5.0] |
| `min_data_in_leaf` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 35.0] |

#### E. Slice plot

![sp500@s42/moe](slice_sp500@s42_moe.png)


---

## sp500@s43  (search X=[2969, 28], holdout n=742)


### naive-lightgbm

- **holdout RMSE: 0.01105** (winner retrained in 0.02s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0098, p10: 0.0098
- train: median 0.022s/fold, mean 0.026s, p90 0.039s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `bagging_fraction` | 0.413 |
| `learning_rate` | 0.246 |
| `min_data_in_leaf` | 0.078 |
| `bagging_freq` | 0.067 |
| `feature_fraction` | 0.066 |
| `max_depth` | 0.048 |
| `num_leaves` | 0.031 |
| `lambda_l1` | 0.024 |
| `lambda_l2` | 0.015 |
| `extra_trees` | 0.012 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 21 | 0.0098 | 0.0000 | 0.0098 |
| False | 279 | 0.0098 | 0.0000 | 0.0097 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0873] |
| `num_leaves` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 12.0] |
| `max_depth` | — | — | 0.0098 | 0.0098 | **Q3** [3.0, 6.0] |
| `min_data_in_leaf` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 34.0] |
| `lambda_l1` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0001] |
| `lambda_l2` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0] |
| `feature_fraction` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.9045] |
| `bagging_fraction` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.8034] |

#### E. Slice plot

![sp500@s43/naive-lightgbm](slice_sp500@s43_naive-lightgbm.png)


### moe

- **holdout RMSE: 0.01104** (winner retrained in 0.12s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0098, p10: 0.0098
- train: median 0.261s/fold, mean 0.418s, p90 1.194s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_warmup_iters` | 0.224 |
| `lambda_l2` | 0.156 |
| `max_depth` | 0.155 |
| `learning_rate` | 0.113 |
| `num_leaves` | 0.080 |
| `mixture_num_experts` | 0.074 |
| `mixture_diversity_lambda` | 0.070 |
| `feature_fraction` | 0.029 |
| `bagging_freq` | 0.025 |
| `min_data_in_leaf` | 0.025 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **gbdt** | 0.0098 (n=229) | none | Δ +0.0000 | p=8.00e-06 |
| `mixture_e_step_mode` | **loss_only** | 0.0098 (n=229) | em | Δ +0.0000 | p=5.00e-06 |
| `mixture_r_smoothing` | **ema** | 0.0098 (n=217) | none | Δ +0.0000 | p=3.00e-06 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 229 | 0.0098 | 0.0000 | 0.0097 |
| none | 48 | 0.0098 | 0.0000 | 0.0098 |
| leaf_reuse | 23 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 241 | 0.0098 | 0.0000 | 0.0097 |
| token_choice | 59 | 0.0098 | 0.0000 | 0.0097 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 229 | 0.0098 | 0.0000 | 0.0097 |
| em | 48 | 0.0098 | 0.0000 | 0.0098 |
| gate_only | 23 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gmm | 116 | 0.0098 | 0.0000 | 0.0098 |
| tree_hierarchical | 31 | 0.0098 | 0.0000 | 0.0098 |
| random | 11 | 0.0098 | 0.0000 | 0.0098 |
| uniform | 142 | 0.0098 | 0.0000 | 0.0097 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 217 | 0.0098 | 0.0000 | 0.0097 |
| none | 44 | 0.0098 | 0.0000 | 0.0098 |
| markov | 39 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 24 | 0.0098 | 0.0000 | 0.0098 |
| False | 276 | 0.0098 | 0.0000 | 0.0097 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 252 | 0.0098 | 0.0000 | 0.0097 |
| True | 48 | 0.0098 | 0.0000 | 0.0098 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | — | — | 0.0098 | **Q4** [2.0, ∞) |
| `mixture_diversity_lambda` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.331] |
| `mixture_warmup_iters` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 16.0] |
| `mixture_balance_factor` | — | 0.0098 | 0.0098 | 0.0098 | **Q2** [2.0, 3.0] |
| `learning_rate` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.042] |
| `num_leaves` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 22.0] |
| `max_depth` | — | — | 0.0098 | 0.0098 | **Q3** [3.0, 4.0] |
| `min_data_in_leaf` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 9.0] |

#### E. Slice plot

![sp500@s43/moe](slice_sp500@s43_moe.png)


---

## sp500@s44  (search X=[2969, 28], holdout n=742)


### naive-lightgbm

- **holdout RMSE: 0.01106** (winner retrained in 0.03s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0098, p10: 0.0097
- train: median 0.026s/fold, mean 0.027s, p90 0.035s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.768 |
| `num_leaves` | 0.098 |
| `min_data_in_leaf` | 0.058 |
| `feature_fraction` | 0.028 |
| `bagging_fraction` | 0.019 |
| `extra_trees` | 0.010 |
| `bagging_freq` | 0.009 |
| `lambda_l2` | 0.007 |
| `max_depth` | 0.003 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 280 | 0.0098 | 0.0000 | 0.0097 |
| True | 20 | 0.0098 | 0.0000 | 0.0098 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0152] |
| `num_leaves` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 39.0] |
| `max_depth` | 0.0098 | — | — | 0.0098 | **Q1** [None, 4.0] |
| `min_data_in_leaf` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 8.0] |
| `lambda_l1` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0001] |
| `lambda_l2` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0] |
| `feature_fraction` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.65] |
| `bagging_fraction` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.5556] |

#### E. Slice plot

![sp500@s44/naive-lightgbm](slice_sp500@s44_naive-lightgbm.png)


### moe

- **holdout RMSE: 0.01107** (winner retrained in 0.15s, cv score of winner: 0.0098)
- cv best RMSE: 0.0098, median: 0.0098, p10: 0.0098
- train: median 0.091s/fold, mean 0.133s, p90 0.244s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_init` | 0.473 |
| `extra_trees` | 0.158 |
| `mixture_balance_factor` | 0.086 |
| `feature_fraction` | 0.059 |
| `num_leaves` | 0.046 |
| `mixture_diversity_lambda` | 0.040 |
| `min_data_in_leaf` | 0.033 |
| `learning_rate` | 0.025 |
| `mixture_warmup_iters` | 0.018 |
| `mixture_gate_type` | 0.014 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 243 | 0.0098 | 0.0001 | 0.0098 |
| none | 27 | 0.0098 | 0.0000 | 0.0098 |
| leaf_reuse | 30 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 280 | 0.0098 | 0.0001 | 0.0098 |
| expert_choice | 20 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 67 | 0.0098 | 0.0000 | 0.0098 |
| em | 188 | 0.0098 | 0.0001 | 0.0098 |
| loss_only | 45 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 254 | 0.0098 | 0.0000 | 0.0098 |
| random | 16 | 0.0098 | 0.0000 | 0.0098 |
| gmm | 16 | 0.0098 | 0.0000 | 0.0098 |
| tree_hierarchical | 14 | 0.0099 | 0.0003 | 0.0098 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 230 | 0.0098 | 0.0001 | 0.0098 |
| ema | 29 | 0.0098 | 0.0000 | 0.0098 |
| markov | 41 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 279 | 0.0098 | 0.0001 | 0.0098 |
| False | 21 | 0.0098 | 0.0000 | 0.0098 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 279 | 0.0098 | 0.0000 | 0.0098 |
| False | 21 | 0.0098 | 0.0003 | 0.0098 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | — | — | 0.0098 | **Q4** [2.0, ∞) |
| `mixture_diversity_lambda` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.1487] |
| `mixture_warmup_iters` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 10.0] |
| `mixture_balance_factor` | 0.0098 | — | 0.0098 | 0.0098 | **Q1** [None, 3.0] |
| `learning_rate` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0839] |
| `num_leaves` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 97.75] |
| `max_depth` | 0.0098 | 0.0098 | — | 0.0098 | **Q1** [None, 10.0] |
| `min_data_in_leaf` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 86.0] |

#### E. Slice plot

![sp500@s44/moe](slice_sp500@s44_moe.png)


---

## Overall recommendations

**Categorical settings that are statistically significant winners (p<0.01):**

| dataset | param | best value | Δ vs runner-up | p |
|---|---|---|---|---|
| sp500@s42 | `mixture_gate_type` | **leaf_reuse** | +0.0000 | 5.74e-03 |
| sp500@s42 | `mixture_routing_mode` | **token_choice** | +0.0000 | 1.46e-04 |
| sp500@s42 | `mixture_e_step_mode` | **loss_only** | +0.0000 | 3.28e-03 |
| sp500@s42 | `mixture_r_smoothing` | **ema** | +0.0000 | 5.16e-03 |
| sp500@s43 | `mixture_gate_type` | **gbdt** | +0.0000 | 8.00e-06 |
| sp500@s43 | `mixture_e_step_mode` | **loss_only** | +0.0000 | 5.00e-06 |
| sp500@s43 | `mixture_r_smoothing` | **ema** | +0.0000 | 3.00e-06 |
