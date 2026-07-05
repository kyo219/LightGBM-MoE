# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 300

- **Datasets**: ['elevators'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `5232455cdc8d`, lib sha256 `5cec0a0bd5ab…`, package `/tmp/lgbm-moe-v080/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| elevators | naive-lightgbm | **0.00238** ± 0.00005 | 0.00264 | 0.0% | 0.19 |
| elevators | moe | **0.00229** ± 0.00005 | 0.00256 | 0.0% | 2.07 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| elevators@s42 | naive-lightgbm | 0.0026 | 0.0027 | 0.125 | 208 |
| elevators@s42 | moe | 0.0026 | 0.0027 | 2.033 | 3347 |
| elevators@s43 | naive-lightgbm | 0.0027 | 0.0027 | 0.151 | 245 |
| elevators@s43 | moe | 0.0026 | 0.0027 | 2.691 | 3860 |
| elevators@s44 | naive-lightgbm | 0.0026 | 0.0027 | 0.150 | 228 |
| elevators@s44 | moe | 0.0026 | 0.0027 | 1.982 | 2647 |



---

## elevators@s42  (search X=[8000, 18], holdout n=2000)


### naive-lightgbm

- **holdout RMSE: 0.00245** (winner retrained in 0.14s, cv score of winner: 0.0026)
- cv best RMSE: 0.0026, median: 0.0027, p10: 0.0027
- train: median 0.125s/fold, mean 0.133s, p90 0.185s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.705 |
| `learning_rate` | 0.177 |
| `num_leaves` | 0.033 |
| `bagging_freq` | 0.030 |
| `feature_fraction` | 0.022 |
| `bagging_fraction` | 0.013 |
| `extra_trees` | 0.009 |
| `max_depth` | 0.007 |
| `min_data_in_leaf` | 0.005 |
| `lambda_l2` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 278 | 0.0028 | 0.0004 | 0.0026 |
| True | 22 | 0.0036 | 0.0009 | 0.0028 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0033 | 0.0027 | 0.0027 | 0.0028 | **Q2** [0.1755, 0.2043] |
| `num_leaves` | 0.0029 | 0.0027 | 0.0028 | 0.0031 | **Q2** [14.0, 18.0] |
| `max_depth` | 0.0031 | — | — | 0.0028 | **Q4** [6.0, ∞) |
| `min_data_in_leaf` | 0.0028 | 0.0028 | 0.0029 | 0.0031 | **Q1** [None, 11.0] |
| `lambda_l1` | 0.0029 | 0.0029 | 0.0028 | 0.0030 | **Q3** [0.0003, 0.0015] |
| `lambda_l2` | 0.0031 | 0.0028 | 0.0028 | 0.0029 | **Q2** [0.0001, 0.0003] |
| `feature_fraction` | 0.0031 | 0.0028 | 0.0028 | 0.0028 | **Q2** [0.9043, 0.9313] |
| `bagging_fraction` | 0.0031 | 0.0028 | 0.0028 | 0.0029 | **Q2** [0.9313, 0.9559] |

#### E. Slice plot

![elevators@s42/naive-lightgbm](slice_elevators@s42_naive-lightgbm.png)


### moe

- **holdout RMSE: 0.00236** (winner retrained in 2.03s, cv score of winner: 0.0026)
- cv best RMSE: 0.0026, median: 0.0027, p10: 0.0026
- train: median 2.033s/fold, mean 2.208s, p90 3.592s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.405 |
| `mixture_init` | 0.344 |
| `mixture_diversity_lambda` | 0.039 |
| `mixture_r_smoothing` | 0.022 |
| `mixture_warmup_iters` | 0.022 |
| `max_depth` | 0.022 |
| `mixture_num_experts` | 0.021 |
| `extra_trees` | 0.018 |
| `mixture_refit_leaves` | 0.015 |
| `min_data_in_leaf` | 0.014 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_routing_mode` | **token_choice** | 0.0036 (n=179) | expert_choice | Δ +0.0015 | p=8.29e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 243 | 0.0041 | 0.0043 | 0.0026 |
| gbdt | 30 | 0.0045 | 0.0031 | 0.0026 |
| none | 27 | 0.0050 | 0.0046 | 0.0026 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 179 | 0.0036 | 0.0033 | 0.0026 |
| expert_choice | 121 | 0.0051 | 0.0052 | 0.0026 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 172 | 0.0036 | 0.0035 | 0.0026 |
| loss_only | 104 | 0.0046 | 0.0047 | 0.0026 |
| em | 24 | 0.0066 | 0.0058 | 0.0027 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 236 | 0.0032 | 0.0022 | 0.0026 |
| random | 33 | 0.0035 | 0.0011 | 0.0026 |
| tree_hierarchical | 15 | 0.0112 | 0.0073 | 0.0026 |
| gmm | 16 | 0.0141 | 0.0069 | 0.0026 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 129 | 0.0040 | 0.0040 | 0.0026 |
| none | 90 | 0.0041 | 0.0041 | 0.0026 |
| markov | 81 | 0.0047 | 0.0046 | 0.0027 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 280 | 0.0041 | 0.0043 | 0.0026 |
| True | 20 | 0.0054 | 0.0028 | 0.0033 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 265 | 0.0041 | 0.0042 | 0.0026 |
| True | 35 | 0.0051 | 0.0043 | 0.0028 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.0052 | — | — | 0.0039 | **Q4** [4.0, ∞) |
| `mixture_diversity_lambda` | 0.0034 | 0.0036 | 0.0042 | 0.0056 | **Q1** [None, 0.0667] |
| `mixture_warmup_iters` | 0.0051 | 0.0044 | 0.0035 | 0.0038 | **Q3** [39.5, 46.0] |
| `mixture_balance_factor` | — | 0.0036 | 0.0051 | 0.0045 | **Q2** [2.0, 5.0] |
| `learning_rate` | 0.0054 | 0.0036 | 0.0039 | 0.0039 | **Q2** [0.1944, 0.2472] |
| `num_leaves` | 0.0043 | 0.0033 | 0.0042 | 0.0051 | **Q2** [14.0, 18.5] |
| `max_depth` | 0.0051 | 0.0042 | 0.0038 | 0.0041 | **Q3** [11.0, 12.0] |
| `min_data_in_leaf` | 0.0042 | 0.0034 | 0.0036 | 0.0056 | **Q2** [16.75, 21.0] |

#### E. Slice plot

![elevators@s42/moe](slice_elevators@s42_moe.png)


---

## elevators@s43  (search X=[8000, 18], holdout n=2000)


### naive-lightgbm

- **holdout RMSE: 0.00236** (winner retrained in 0.14s, cv score of winner: 0.0027)
- cv best RMSE: 0.0027, median: 0.0027, p10: 0.0027
- train: median 0.151s/fold, mean 0.158s, p90 0.234s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.552 |
| `learning_rate` | 0.246 |
| `num_leaves` | 0.055 |
| `feature_fraction` | 0.046 |
| `min_data_in_leaf` | 0.026 |
| `bagging_fraction` | 0.025 |
| `max_depth` | 0.021 |
| `extra_trees` | 0.018 |
| `bagging_freq` | 0.007 |
| `lambda_l2` | 0.004 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 279 | 0.0028 | 0.0005 | 0.0027 |
| True | 21 | 0.0037 | 0.0007 | 0.0029 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0033 | 0.0027 | 0.0028 | 0.0027 | **Q2** [0.1548, 0.1835] |
| `num_leaves` | 0.0028 | 0.0028 | 0.0029 | 0.0031 | **Q1** [None, 17.0] |
| `max_depth` | 0.0034 | 0.0029 | — | 0.0029 | **Q2** [5.0, 8.0] |
| `min_data_in_leaf` | 0.0028 | 0.0028 | 0.0027 | 0.0032 | **Q3** [20.5, 24.0] |
| `lambda_l1` | 0.0029 | 0.0028 | 0.0028 | 0.0031 | **Q2** [0.0, 0.0004] |
| `lambda_l2` | 0.0029 | 0.0029 | 0.0029 | 0.0030 | **Q1** [None, 0.0001] |
| `feature_fraction` | 0.0032 | 0.0029 | 0.0028 | 0.0027 | **Q4** [0.9712, ∞) |
| `bagging_fraction` | 0.0031 | 0.0029 | 0.0029 | 0.0028 | **Q4** [0.9656, ∞) |

#### E. Slice plot

![elevators@s43/naive-lightgbm](slice_elevators@s43_naive-lightgbm.png)


### moe

- **holdout RMSE: 0.00225** (winner retrained in 2.56s, cv score of winner: 0.0026)
- cv best RMSE: 0.0026, median: 0.0027, p10: 0.0026
- train: median 2.691s/fold, mean 2.551s, p90 4.247s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.341 |
| `mixture_r_smoothing` | 0.130 |
| `mixture_init` | 0.119 |
| `mixture_diversity_lambda` | 0.081 |
| `mixture_refit_leaves` | 0.063 |
| `mixture_warmup_iters` | 0.050 |
| `min_data_in_leaf` | 0.042 |
| `mixture_gate_type` | 0.034 |
| `learning_rate` | 0.029 |
| `bagging_freq` | 0.024 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 262 | 0.0034 | 0.0028 | 0.0026 |
| gbdt | 22 | 0.0047 | 0.0029 | 0.0028 |
| none | 16 | 0.0061 | 0.0053 | 0.0026 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 279 | 0.0034 | 0.0023 | 0.0026 |
| token_choice | 21 | 0.0075 | 0.0067 | 0.0034 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 219 | 0.0034 | 0.0026 | 0.0026 |
| gate_only | 52 | 0.0041 | 0.0040 | 0.0027 |
| em | 29 | 0.0047 | 0.0041 | 0.0026 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| random | 209 | 0.0033 | 0.0028 | 0.0026 |
| uniform | 59 | 0.0036 | 0.0020 | 0.0027 |
| tree_hierarchical | 14 | 0.0062 | 0.0052 | 0.0028 |
| gmm | 18 | 0.0064 | 0.0044 | 0.0027 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 240 | 0.0035 | 0.0029 | 0.0026 |
| ema | 42 | 0.0041 | 0.0028 | 0.0027 |
| none | 18 | 0.0056 | 0.0043 | 0.0027 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 276 | 0.0034 | 0.0025 | 0.0026 |
| True | 24 | 0.0069 | 0.0061 | 0.0031 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 278 | 0.0036 | 0.0029 | 0.0026 |
| True | 22 | 0.0051 | 0.0045 | 0.0028 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.0051 | — | — | 0.0034 | **Q4** [4.0, ∞) |
| `mixture_diversity_lambda` | 0.0031 | 0.0037 | 0.0032 | 0.0048 | **Q1** [None, 0.0534] |
| `mixture_warmup_iters` | 0.0031 | 0.0033 | 0.0033 | 0.0047 | **Q1** [None, 7.0] |
| `mixture_balance_factor` | 0.0051 | — | 0.0032 | 0.0036 | **Q3** [5.0, 7.0] |
| `learning_rate` | 0.0046 | 0.0032 | 0.0030 | 0.0039 | **Q3** [0.1921, 0.2184] |
| `num_leaves` | 0.0044 | 0.0037 | 0.0031 | 0.0035 | **Q3** [107.0, 116.0] |
| `max_depth` | 0.0041 | — | 0.0035 | 0.0038 | **Q3** [6.0, 8.0] |
| `min_data_in_leaf` | 0.0040 | 0.0034 | 0.0034 | 0.0040 | **Q2** [19.0, 24.0] |

#### E. Slice plot

![elevators@s43/moe](slice_elevators@s43_moe.png)


---

## elevators@s44  (search X=[8000, 18], holdout n=2000)


### naive-lightgbm

- **holdout RMSE: 0.00234** (winner retrained in 0.30s, cv score of winner: 0.0026)
- cv best RMSE: 0.0026, median: 0.0027, p10: 0.0027
- train: median 0.150s/fold, mean 0.147s, p90 0.211s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.604 |
| `learning_rate` | 0.098 |
| `feature_fraction` | 0.079 |
| `num_leaves` | 0.058 |
| `bagging_fraction` | 0.052 |
| `bagging_freq` | 0.046 |
| `max_depth` | 0.030 |
| `min_data_in_leaf` | 0.019 |
| `extra_trees` | 0.013 |
| `lambda_l2` | 0.002 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 279 | 0.0028 | 0.0005 | 0.0026 |
| True | 21 | 0.0035 | 0.0008 | 0.0029 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0032 | 0.0027 | 0.0027 | 0.0029 | **Q2** [0.1153, 0.1319] |
| `num_leaves` | 0.0031 | 0.0029 | 0.0028 | 0.0027 | **Q4** [124.0, ∞) |
| `max_depth` | 0.0031 | 0.0028 | 0.0028 | 0.0028 | **Q2** [6.0, 7.0] |
| `min_data_in_leaf` | 0.0029 | 0.0027 | 0.0028 | 0.0031 | **Q2** [15.0, 19.0] |
| `lambda_l1` | 0.0030 | 0.0027 | 0.0028 | 0.0031 | **Q2** [0.0, 0.0002] |
| `lambda_l2` | 0.0028 | 0.0028 | 0.0028 | 0.0030 | **Q1** [None, 0.0003] |
| `feature_fraction` | 0.0032 | 0.0028 | 0.0028 | 0.0027 | **Q4** [0.9776, ∞) |
| `bagging_fraction` | 0.0031 | 0.0029 | 0.0027 | 0.0027 | **Q3** [0.9628, 0.9815] |

#### E. Slice plot

![elevators@s44/naive-lightgbm](slice_elevators@s44_naive-lightgbm.png)


### moe

- **holdout RMSE: 0.00225** (winner retrained in 1.62s, cv score of winner: 0.0026)
- cv best RMSE: 0.0026, median: 0.0027, p10: 0.0026
- train: median 1.982s/fold, mean 1.747s, p90 2.838s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.420 |
| `mixture_r_smoothing` | 0.070 |
| `bagging_freq` | 0.069 |
| `mixture_hard_m_step` | 0.066 |
| `mixture_init` | 0.059 |
| `mixture_refit_leaves` | 0.050 |
| `mixture_warmup_iters` | 0.049 |
| `mixture_num_experts` | 0.048 |
| `feature_fraction` | 0.040 |
| `bagging_fraction` | 0.036 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_routing_mode` | **expert_choice** | 0.0034 (n=250) | token_choice | Δ +0.0018 | p=9.37e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 247 | 0.0033 | 0.0025 | 0.0026 |
| none | 16 | 0.0054 | 0.0044 | 0.0027 |
| leaf_reuse | 37 | 0.0058 | 0.0057 | 0.0026 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 250 | 0.0034 | 0.0029 | 0.0026 |
| token_choice | 50 | 0.0052 | 0.0046 | 0.0027 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 217 | 0.0034 | 0.0028 | 0.0026 |
| loss_only | 47 | 0.0038 | 0.0027 | 0.0026 |
| em | 36 | 0.0056 | 0.0055 | 0.0026 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 55 | 0.0032 | 0.0011 | 0.0026 |
| gmm | 195 | 0.0034 | 0.0028 | 0.0026 |
| random | 35 | 0.0046 | 0.0039 | 0.0026 |
| tree_hierarchical | 15 | 0.0076 | 0.0076 | 0.0026 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 249 | 0.0031 | 0.0015 | 0.0026 |
| markov | 35 | 0.0060 | 0.0063 | 0.0027 |
| none | 16 | 0.0090 | 0.0064 | 0.0030 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 258 | 0.0033 | 0.0025 | 0.0026 |
| True | 42 | 0.0061 | 0.0057 | 0.0029 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 279 | 0.0037 | 0.0034 | 0.0026 |
| True | 21 | 0.0044 | 0.0017 | 0.0029 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.0042 | 0.0066 | — | 0.0033 | **Q4** [4.0, ∞) |
| `mixture_diversity_lambda` | 0.0028 | 0.0029 | 0.0035 | 0.0056 | **Q1** [None, 0.0104] |
| `mixture_warmup_iters` | 0.0030 | 0.0029 | 0.0035 | 0.0053 | **Q2** [6.0, 8.0] |
| `mixture_balance_factor` | — | 0.0033 | 0.0036 | 0.0044 | **Q2** [2.0, 3.0] |
| `learning_rate` | 0.0052 | 0.0029 | 0.0035 | 0.0033 | **Q2** [0.1798, 0.232] |
| `num_leaves` | 0.0056 | 0.0029 | 0.0030 | 0.0033 | **Q2** [101.75, 115.0] |
| `max_depth` | 0.0054 | 0.0031 | — | 0.0033 | **Q2** [10.0, 11.0] |
| `min_data_in_leaf` | 0.0046 | 0.0031 | 0.0032 | 0.0039 | **Q2** [35.0, 41.0] |

#### E. Slice plot

![elevators@s44/moe](slice_elevators@s44_moe.png)


---

## Overall recommendations

**Categorical settings that are statistically significant winners (p<0.01):**

| dataset | param | best value | Δ vs runner-up | p |
|---|---|---|---|---|
| elevators@s42 | `mixture_routing_mode` | **token_choice** | +0.0015 | 8.29e-03 |
| elevators@s44 | `mixture_routing_mode` | **expert_choice** | +0.0018 | 9.37e-03 |
