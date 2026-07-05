# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 300

- **Datasets**: ['sp500'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `d25c06cf3b86`, lib sha256 `a3c30fcc7fd3…`, package `/home/mokumoku/Develop/LightGBM-MoE/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| sp500 | naive-lightgbm | **0.01105** ± 0.00001 | 0.00975 | 0.0% | 0.04 |
| sp500 | naive-ensemble | **0.01105** ± 0.00000 | 0.00974 | 0.0% | 0.15 |
| sp500 | moe | **0.01108** ± 0.00004 | 0.00976 | 0.0% | 0.20 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| sp500@s42 | naive-lightgbm | 0.0098 | 0.0098 | 0.022 | 39 |
| sp500@s42 | naive-ensemble | 0.0097 | 0.0098 | 0.199 | 334 |
| sp500@s42 | moe | 0.0098 | 0.0098 | 0.079 | 298 |
| sp500@s43 | naive-lightgbm | 0.0097 | 0.0098 | 0.022 | 44 |
| sp500@s43 | naive-ensemble | 0.0098 | 0.0098 | 0.073 | 116 |
| sp500@s43 | moe | 0.0098 | 0.0098 | 0.063 | 152 |
| sp500@s44 | naive-lightgbm | 0.0097 | 0.0098 | 0.025 | 45 |
| sp500@s44 | naive-ensemble | 0.0097 | 0.0098 | 0.128 | 202 |
| sp500@s44 | moe | 0.0098 | 0.0098 | 0.166 | 282 |



---

## sp500@s42  (search X=[2969, 28], holdout n=742)


### naive-lightgbm

- **holdout RMSE: 0.01103** (winner retrained in 0.07s, cv score of winner: 0.0098)
- cv best RMSE: 0.0098, median: 0.0098, p10: 0.0098
- train: median 0.022s/fold, mean 0.023s, p90 0.029s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.434 |
| `num_leaves` | 0.259 |
| `min_data_in_leaf` | 0.127 |
| `max_depth` | 0.101 |
| `extra_trees` | 0.031 |
| `bagging_fraction` | 0.022 |
| `bagging_freq` | 0.015 |
| `feature_fraction` | 0.010 |
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


### naive-ensemble

- **holdout RMSE: 0.01105** (winner retrained in 0.24s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0098, p10: 0.0097
- train: median 0.199s/fold, mean 0.219s, p90 0.374s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.840 |
| `min_data_in_leaf` | 0.053 |
| `lambda_l2` | 0.022 |
| `num_leaves` | 0.019 |
| `feature_fraction` | 0.017 |
| `n_models` | 0.013 |
| `max_depth` | 0.011 |
| `bagging_fraction` | 0.011 |
| `bagging_freq` | 0.007 |
| `extra_trees` | 0.006 |

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
| `learning_rate` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0116] |
| `num_leaves` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 25.75] |
| `max_depth` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 7.0] |
| `min_data_in_leaf` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 6.0] |
| `lambda_l1` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0019] |
| `lambda_l2` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0] |
| `feature_fraction` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.7093] |
| `bagging_fraction` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.9263] |

#### E. Slice plot

![sp500@s42/naive-ensemble](slice_sp500@s42_naive-ensemble.png)


### moe

- **holdout RMSE: 0.01113** (winner retrained in 0.24s, cv score of winner: 0.0098)
- cv best RMSE: 0.0098, median: 0.0098, p10: 0.0098
- train: median 0.079s/fold, mean 0.191s, p90 0.435s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_init` | 0.397 |
| `mixture_diversity_lambda` | 0.243 |
| `min_data_in_leaf` | 0.075 |
| `bagging_fraction` | 0.069 |
| `learning_rate` | 0.049 |
| `mixture_gate_type` | 0.029 |
| `mixture_warmup_iters` | 0.027 |
| `feature_fraction` | 0.026 |
| `bagging_freq` | 0.016 |
| `num_leaves` | 0.012 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **leaf_reuse** | 0.0098 (n=17) | none | Δ +0.0000 | p=5.49e-03 |
| `mixture_routing_mode` | **token_choice** | 0.0098 (n=104) | expert_choice | Δ +0.0000 | p=3.20e-05 |
| `mixture_r_smoothing` | **ema** | 0.0098 (n=88) | markov | Δ +0.0000 | p=2.40e-05 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 17 | 0.0098 | 0.0000 | 0.0098 |
| none | 214 | 0.0098 | 0.0000 | 0.0098 |
| gbdt | 69 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 104 | 0.0098 | 0.0000 | 0.0098 |
| expert_choice | 196 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 103 | 0.0098 | 0.0000 | 0.0098 |
| gate_only | 52 | 0.0098 | 0.0000 | 0.0098 |
| em | 145 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| random | 16 | 0.0098 | 0.0000 | 0.0098 |
| gmm | 14 | 0.0098 | 0.0000 | 0.0098 |
| tree_hierarchical | 28 | 0.0098 | 0.0000 | 0.0098 |
| uniform | 242 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 88 | 0.0098 | 0.0000 | 0.0098 |
| markov | 120 | 0.0098 | 0.0000 | 0.0098 |
| none | 92 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 278 | 0.0098 | 0.0000 | 0.0098 |
| False | 22 | 0.0098 | 0.0000 | 0.0098 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 278 | 0.0098 | 0.0000 | 0.0098 |
| False | 22 | 0.0098 | 0.0000 | 0.0098 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.0098 | — | — | 0.0098 | **Q1** [None, 4.0] |
| `mixture_diversity_lambda` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.1744] |
| `mixture_warmup_iters` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 32.0] |
| `mixture_balance_factor` | — | — | 0.0098 | 0.0098 | **Q3** [2.0, 3.0] |
| `learning_rate` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.123] |
| `num_leaves` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 96.75] |
| `max_depth` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 5.75] |
| `min_data_in_leaf` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 46.0] |

#### E. Slice plot

![sp500@s42/moe](slice_sp500@s42_moe.png)


---

## sp500@s43  (search X=[2969, 28], holdout n=742)


### naive-lightgbm

- **holdout RMSE: 0.01105** (winner retrained in 0.03s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0098, p10: 0.0098
- train: median 0.022s/fold, mean 0.026s, p90 0.039s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `bagging_fraction` | 0.512 |
| `learning_rate` | 0.214 |
| `feature_fraction` | 0.060 |
| `min_data_in_leaf` | 0.049 |
| `num_leaves` | 0.044 |
| `max_depth` | 0.044 |
| `bagging_freq` | 0.036 |
| `lambda_l2` | 0.022 |
| `lambda_l1` | 0.015 |
| `extra_trees` | 0.004 |

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


### naive-ensemble

- **holdout RMSE: 0.01104** (winner retrained in 0.13s, cv score of winner: 0.0098)
- cv best RMSE: 0.0098, median: 0.0098, p10: 0.0098
- train: median 0.073s/fold, mean 0.074s, p90 0.093s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.429 |
| `learning_rate` | 0.156 |
| `feature_fraction` | 0.112 |
| `num_leaves` | 0.106 |
| `bagging_fraction` | 0.092 |
| `max_depth` | 0.070 |
| `bagging_freq` | 0.018 |
| `n_models` | 0.011 |
| `extra_trees` | 0.004 |
| `lambda_l2` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 20 | 0.0098 | 0.0000 | 0.0098 |
| True | 280 | 0.0098 | 0.0000 | 0.0098 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0541] |
| `num_leaves` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 85.75] |
| `max_depth` | — | 0.0098 | 0.0098 | 0.0098 | **Q2** [3.0, 4.0] |
| `min_data_in_leaf` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 91.0] |
| `lambda_l1` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0] |
| `lambda_l2` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0] |
| `feature_fraction` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.5203] |
| `bagging_fraction` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.684] |

#### E. Slice plot

![sp500@s43/naive-ensemble](slice_sp500@s43_naive-ensemble.png)


### moe

- **holdout RMSE: 0.01103** (winner retrained in 0.09s, cv score of winner: 0.0098)
- cv best RMSE: 0.0098, median: 0.0098, p10: 0.0098
- train: median 0.063s/fold, mean 0.095s, p90 0.152s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l2` | 0.300 |
| `mixture_init` | 0.252 |
| `mixture_balance_factor` | 0.099 |
| `learning_rate` | 0.085 |
| `bagging_fraction` | 0.058 |
| `min_data_in_leaf` | 0.042 |
| `mixture_warmup_iters` | 0.039 |
| `feature_fraction` | 0.027 |
| `num_leaves` | 0.024 |
| `mixture_num_experts` | 0.014 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 30 | 0.0098 | 0.0000 | 0.0098 |
| none | 254 | 0.0098 | 0.0000 | 0.0098 |
| leaf_reuse | 16 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 212 | 0.0098 | 0.0000 | 0.0098 |
| token_choice | 88 | 0.0098 | 0.0001 | 0.0098 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 99 | 0.0098 | 0.0000 | 0.0098 |
| em | 166 | 0.0098 | 0.0000 | 0.0098 |
| gate_only | 35 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gmm | 14 | 0.0098 | 0.0000 | 0.0098 |
| tree_hierarchical | 15 | 0.0098 | 0.0001 | 0.0098 |
| random | 162 | 0.0098 | 0.0000 | 0.0098 |
| uniform | 109 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 219 | 0.0098 | 0.0000 | 0.0098 |
| none | 57 | 0.0098 | 0.0001 | 0.0098 |
| markov | 24 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 220 | 0.0098 | 0.0000 | 0.0098 |
| False | 80 | 0.0098 | 0.0001 | 0.0098 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 23 | 0.0098 | 0.0000 | 0.0098 |
| True | 277 | 0.0098 | 0.0000 | 0.0098 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.0098 | — | — | 0.0098 | **Q1** [None, 3.0] |
| `mixture_diversity_lambda` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.2049] |
| `mixture_warmup_iters` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 9.0] |
| `mixture_balance_factor` | 0.0098 | — | 0.0098 | 0.0098 | **Q1** [None, 3.0] |
| `learning_rate` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.1629] |
| `num_leaves` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 51.75] |
| `max_depth` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 7.0] |
| `min_data_in_leaf` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 87.0] |

#### E. Slice plot

![sp500@s43/moe](slice_sp500@s43_moe.png)


---

## sp500@s44  (search X=[2969, 28], holdout n=742)


### naive-lightgbm

- **holdout RMSE: 0.01106** (winner retrained in 0.02s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0098, p10: 0.0097
- train: median 0.025s/fold, mean 0.027s, p90 0.033s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.742 |
| `min_data_in_leaf` | 0.098 |
| `num_leaves` | 0.077 |
| `feature_fraction` | 0.030 |
| `bagging_fraction` | 0.019 |
| `max_depth` | 0.014 |
| `extra_trees` | 0.013 |
| `bagging_freq` | 0.007 |
| `lambda_l2` | 0.001 |
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


### naive-ensemble

- **holdout RMSE: 0.01105** (winner retrained in 0.09s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0098, p10: 0.0097
- train: median 0.128s/fold, mean 0.131s, p90 0.188s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.471 |
| `min_data_in_leaf` | 0.195 |
| `bagging_fraction` | 0.092 |
| `lambda_l2` | 0.081 |
| `num_leaves` | 0.043 |
| `bagging_freq` | 0.041 |
| `feature_fraction` | 0.033 |
| `max_depth` | 0.022 |
| `n_models` | 0.012 |
| `extra_trees` | 0.010 |

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
| `learning_rate` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0198] |
| `num_leaves` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 53.0] |
| `max_depth` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 7.0] |
| `min_data_in_leaf` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 7.0] |
| `lambda_l1` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0] |
| `lambda_l2` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.9703] |
| `feature_fraction` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.7247] |
| `bagging_fraction` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.6824] |

#### E. Slice plot

![sp500@s44/naive-ensemble](slice_sp500@s44_naive-ensemble.png)


### moe

- **holdout RMSE: 0.01107** (winner retrained in 0.27s, cv score of winner: 0.0098)
- cv best RMSE: 0.0098, median: 0.0098, p10: 0.0098
- train: median 0.166s/fold, mean 0.181s, p90 0.232s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_init` | 0.535 |
| `learning_rate` | 0.259 |
| `mixture_diversity_lambda` | 0.050 |
| `extra_trees` | 0.039 |
| `num_leaves` | 0.027 |
| `mixture_e_step_mode` | 0.021 |
| `min_data_in_leaf` | 0.017 |
| `feature_fraction` | 0.008 |
| `max_depth` | 0.007 |
| `mixture_gate_type` | 0.007 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_init` | **uniform** | 0.0098 (n=24) | tree_hierarchical | Δ +0.0000 | p=1.14e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 27 | 0.0098 | 0.0000 | 0.0098 |
| none | 16 | 0.0098 | 0.0000 | 0.0098 |
| leaf_reuse | 257 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 171 | 0.0098 | 0.0000 | 0.0098 |
| expert_choice | 129 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 15 | 0.0098 | 0.0000 | 0.0098 |
| em | 134 | 0.0098 | 0.0000 | 0.0098 |
| loss_only | 151 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 24 | 0.0098 | 0.0000 | 0.0098 |
| tree_hierarchical | 14 | 0.0098 | 0.0000 | 0.0098 |
| random | 15 | 0.0098 | 0.0000 | 0.0098 |
| gmm | 247 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 101 | 0.0098 | 0.0000 | 0.0098 |
| ema | 17 | 0.0098 | 0.0000 | 0.0098 |
| markov | 182 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 28 | 0.0098 | 0.0000 | 0.0098 |
| False | 272 | 0.0098 | 0.0000 | 0.0098 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 28 | 0.0098 | 0.0000 | 0.0098 |
| False | 272 | 0.0098 | 0.0000 | 0.0098 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.0098 | — | — | 0.0098 | **Q1** [None, 3.0] |
| `mixture_diversity_lambda` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.1507] |
| `mixture_warmup_iters` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 42.0] |
| `mixture_balance_factor` | 0.0098 | 0.0098 | — | 0.0098 | **Q1** [None, 6.0] |
| `learning_rate` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0124] |
| `num_leaves` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 24.75] |
| `max_depth` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 4.0] |
| `min_data_in_leaf` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 83.0] |

#### E. Slice plot

![sp500@s44/moe](slice_sp500@s44_moe.png)


---

## Overall recommendations

**Categorical settings that are statistically significant winners (p<0.01):**

| dataset | param | best value | Δ vs runner-up | p |
|---|---|---|---|---|
| sp500@s42 | `mixture_gate_type` | **leaf_reuse** | +0.0000 | 5.49e-03 |
| sp500@s42 | `mixture_routing_mode` | **token_choice** | +0.0000 | 3.20e-05 |
| sp500@s42 | `mixture_r_smoothing` | **ema** | +0.0000 | 2.40e-05 |
| sp500@s44 | `mixture_init` | **uniform** | +0.0000 | 1.14e-03 |
