# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 500

- **Datasets**: ['elevators'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `747bb4a7b5e6`, lib sha256 `a3c30fcc7fd3…`, package `/home/mokumoku/Develop/LightGBM-MoE/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| elevators | naive-lightgbm | **0.00236** ± 0.00008 | 0.00263 | 0.0% | 0.24 |
| elevators | moe | **0.00230** ± 0.00002 | 0.00256 | 0.0% | 2.09 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| elevators@s42 | naive-lightgbm | 0.0026 | 0.0027 | 0.118 | 324 |
| elevators@s42 | moe | 0.0025 | 0.0028 | 1.216 | 3351 |
| elevators@s43 | naive-lightgbm | 0.0026 | 0.0027 | 0.168 | 455 |
| elevators@s43 | moe | 0.0026 | 0.0026 | 1.543 | 5622 |
| elevators@s44 | naive-lightgbm | 0.0026 | 0.0027 | 0.168 | 418 |
| elevators@s44 | moe | 0.0026 | 0.0028 | 1.422 | 4837 |



---

## elevators@s42  (search X=[8000, 18], holdout n=2000)


### naive-lightgbm

- **holdout RMSE: 0.00245** (winner retrained in 0.15s, cv score of winner: 0.0026)
- cv best RMSE: 0.0026, median: 0.0027, p10: 0.0027
- train: median 0.118s/fold, mean 0.124s, p90 0.163s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.683 |
| `learning_rate` | 0.111 |
| `min_data_in_leaf` | 0.107 |
| `num_leaves` | 0.058 |
| `feature_fraction` | 0.015 |
| `bagging_fraction` | 0.014 |
| `bagging_freq` | 0.007 |
| `max_depth` | 0.004 |
| `extra_trees` | 0.001 |
| `lambda_l2` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 0.0028 | 0.0005 | 0.0026 |
| True | 32 | 0.0034 | 0.0008 | 0.0028 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0031 | 0.0028 | 0.0027 | 0.0028 | **Q3** [0.2162, 0.2435] |
| `num_leaves` | 0.0029 | 0.0028 | 0.0028 | 0.0030 | **Q2** [13.0, 17.0] |
| `max_depth` | 0.0030 | — | — | 0.0028 | **Q4** [6.0, ∞) |
| `min_data_in_leaf` | 0.0027 | 0.0029 | 0.0028 | 0.0030 | **Q1** [None, 11.0] |
| `lambda_l1` | 0.0029 | 0.0028 | 0.0028 | 0.0030 | **Q2** [0.0001, 0.0004] |
| `lambda_l2` | 0.0030 | 0.0028 | 0.0028 | 0.0028 | **Q2** [0.0001, 0.0004] |
| `feature_fraction` | 0.0030 | 0.0028 | 0.0028 | 0.0028 | **Q2** [0.91, 0.9324] |
| `bagging_fraction` | 0.0030 | 0.0028 | 0.0028 | 0.0028 | **Q2** [0.9364, 0.9573] |

#### E. Slice plot

![elevators@s42/naive-lightgbm](slice_elevators@s42_naive-lightgbm.png)


### moe

- **holdout RMSE: 0.00233** (winner retrained in 1.16s, cv score of winner: 0.0025)
- cv best RMSE: 0.0025, median: 0.0028, p10: 0.0026
- train: median 1.216s/fold, mean 1.314s, p90 2.112s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_diversity_lambda` | 0.314 |
| `max_depth` | 0.090 |
| `bagging_fraction` | 0.086 |
| `feature_fraction` | 0.080 |
| `mixture_expert_dropout_rate` | 0.064 |
| `mixture_balance_factor` | 0.062 |
| `lambda_l1` | 0.055 |
| `mixture_init` | 0.049 |
| `num_leaves` | 0.039 |
| `learning_rate` | 0.034 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 435 | 0.0047 | 0.0043 | 0.0025 |
| gbdt | 41 | 0.0061 | 0.0135 | 0.0026 |
| none | 24 | 0.0065 | 0.0058 | 0.0027 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 432 | 0.0045 | 0.0042 | 0.0025 |
| token_choice | 68 | 0.0069 | 0.0112 | 0.0027 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 45 | 0.0047 | 0.0036 | 0.0026 |
| gate_only | 52 | 0.0049 | 0.0043 | 0.0026 |
| em | 403 | 0.0049 | 0.0061 | 0.0025 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 30 | 0.0036 | 0.0015 | 0.0026 |
| random | 92 | 0.0036 | 0.0033 | 0.0025 |
| tree_hierarchical | 17 | 0.0036 | 0.0021 | 0.0026 |
| kmeans_features | 106 | 0.0037 | 0.0031 | 0.0025 |
| gmm | 234 | 0.0055 | 0.0049 | 0.0025 |
| gmm_features | 21 | 0.0115 | 0.0187 | 0.0026 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 82 | 0.0041 | 0.0035 | 0.0026 |
| markov | 389 | 0.0047 | 0.0043 | 0.0025 |
| none | 29 | 0.0094 | 0.0162 | 0.0026 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 453 | 0.0047 | 0.0044 | 0.0025 |
| True | 47 | 0.0065 | 0.0128 | 0.0030 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 0.0046 | 0.0043 | 0.0025 |
| True | 32 | 0.0080 | 0.0154 | 0.0027 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.0056 | — | — | 0.0048 | **Q4** [4.0, ∞) |
| `mixture_diversity_lambda` | 0.0031 | 0.0035 | 0.0051 | 0.0077 | **Q1** [None, 0.0091] |
| `mixture_warmup_iters` | 0.0043 | 0.0039 | 0.0050 | 0.0058 | **Q2** [15.0, 16.0] |
| `mixture_balance_factor` | 0.0050 | — | 0.0044 | 0.0051 | **Q3** [8.0, 9.0] |
| `learning_rate` | 0.0055 | 0.0049 | 0.0045 | 0.0045 | **Q3** [0.2595, 0.2812] |
| `num_leaves` | 0.0042 | 0.0048 | 0.0055 | 0.0049 | **Q1** [None, 12.0] |
| `max_depth` | 0.0056 | 0.0052 | — | 0.0046 | **Q4** [5.0, ∞) |
| `min_data_in_leaf` | 0.0046 | 0.0046 | 0.0060 | 0.0044 | **Q4** [27.0, ∞) |

#### E. Slice plot

![elevators@s42/moe](slice_elevators@s42_moe.png)


---

## elevators@s43  (search X=[8000, 18], holdout n=2000)


### naive-lightgbm

- **holdout RMSE: 0.00237** (winner retrained in 0.27s, cv score of winner: 0.0026)
- cv best RMSE: 0.0026, median: 0.0027, p10: 0.0027
- train: median 0.168s/fold, mean 0.176s, p90 0.252s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.636 |
| `learning_rate` | 0.137 |
| `feature_fraction` | 0.053 |
| `min_data_in_leaf` | 0.053 |
| `num_leaves` | 0.050 |
| `max_depth` | 0.036 |
| `bagging_freq` | 0.021 |
| `extra_trees` | 0.008 |
| `lambda_l2` | 0.005 |
| `bagging_fraction` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 469 | 0.0028 | 0.0004 | 0.0026 |
| True | 31 | 0.0035 | 0.0006 | 0.0028 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0031 | 0.0028 | 0.0028 | 0.0028 | **Q2** [0.1346, 0.1677] |
| `num_leaves` | 0.0028 | 0.0028 | 0.0028 | 0.0030 | **Q1** [None, 19.0] |
| `max_depth` | 0.0030 | 0.0028 | 0.0028 | 0.0028 | **Q2** [7.0, 8.0] |
| `min_data_in_leaf` | 0.0027 | 0.0028 | 0.0028 | 0.0030 | **Q1** [None, 13.0] |
| `lambda_l1` | 0.0028 | 0.0028 | 0.0028 | 0.0030 | **Q1** [None, 0.0] |
| `lambda_l2` | 0.0029 | 0.0028 | 0.0028 | 0.0029 | **Q2** [0.0035, 0.0406] |
| `feature_fraction` | 0.0030 | 0.0027 | 0.0028 | 0.0027 | **Q2** [0.901, 0.9313] |
| `bagging_fraction` | 0.0029 | 0.0028 | 0.0029 | 0.0028 | **Q2** [0.8597, 0.9186] |

#### E. Slice plot

![elevators@s43/naive-lightgbm](slice_elevators@s43_naive-lightgbm.png)


### moe

- **holdout RMSE: 0.00228** (winner retrained in 3.29s, cv score of winner: 0.0026)
- cv best RMSE: 0.0026, median: 0.0026, p10: 0.0026
- train: median 1.543s/fold, mean 2.213s, p90 3.320s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.396 |
| `mixture_load_balance_alpha` | 0.084 |
| `min_data_in_leaf` | 0.081 |
| `mixture_routing_mode` | 0.077 |
| `mixture_gate_type` | 0.060 |
| `mixture_expert_dropout_rate` | 0.047 |
| `bagging_fraction` | 0.044 |
| `feature_fraction` | 0.042 |
| `max_depth` | 0.028 |
| `num_leaves` | 0.025 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_routing_mode` | **expert_choice** | 0.0036 (n=416) | token_choice | Δ +0.0021 | p=1.91e-04 |
| `mixture_e_step_mode` | **loss_only** | 0.0035 (n=384) | em | Δ +0.0015 | p=6.42e-03 |
| `mixture_r_smoothing` | **markov** | 0.0035 (n=361) | ema | Δ +0.0012 | p=5.34e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 367 | 0.0036 | 0.0032 | 0.0026 |
| gbdt | 105 | 0.0042 | 0.0035 | 0.0026 |
| none | 28 | 0.0065 | 0.0055 | 0.0027 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 416 | 0.0036 | 0.0030 | 0.0026 |
| token_choice | 84 | 0.0057 | 0.0048 | 0.0026 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 384 | 0.0035 | 0.0029 | 0.0026 |
| em | 83 | 0.0050 | 0.0046 | 0.0026 |
| gate_only | 33 | 0.0056 | 0.0051 | 0.0026 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 327 | 0.0032 | 0.0025 | 0.0026 |
| tree_hierarchical | 50 | 0.0038 | 0.0035 | 0.0026 |
| random | 29 | 0.0040 | 0.0027 | 0.0026 |
| gmm | 47 | 0.0056 | 0.0050 | 0.0027 |
| kmeans_features | 30 | 0.0060 | 0.0048 | 0.0028 |
| gmm_features | 17 | 0.0101 | 0.0045 | 0.0030 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 361 | 0.0035 | 0.0029 | 0.0026 |
| ema | 113 | 0.0047 | 0.0040 | 0.0026 |
| none | 26 | 0.0065 | 0.0062 | 0.0026 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 0.0038 | 0.0035 | 0.0026 |
| True | 32 | 0.0052 | 0.0040 | 0.0029 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 469 | 0.0037 | 0.0033 | 0.0026 |
| True | 31 | 0.0065 | 0.0053 | 0.0029 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.0066 | — | — | 0.0037 | **Q4** [5.0, ∞) |
| `mixture_diversity_lambda` | 0.0035 | 0.0034 | 0.0033 | 0.0055 | **Q3** [0.0602, 0.0815] |
| `mixture_warmup_iters` | 0.0040 | 0.0035 | 0.0036 | 0.0044 | **Q2** [21.0, 28.0] |
| `mixture_balance_factor` | — | 0.0035 | 0.0040 | 0.0046 | **Q2** [2.0, 3.0] |
| `learning_rate` | 0.0049 | 0.0036 | 0.0037 | 0.0034 | **Q4** [0.2809, ∞) |
| `num_leaves` | 0.0039 | 0.0032 | 0.0040 | 0.0045 | **Q2** [26.0, 36.0] |
| `max_depth` | 0.0035 | — | 0.0034 | 0.0049 | **Q3** [5.0, 6.0] |
| `min_data_in_leaf` | 0.0036 | 0.0035 | 0.0033 | 0.0053 | **Q3** [16.0, 22.25] |

#### E. Slice plot

![elevators@s43/moe](slice_elevators@s43_moe.png)


---

## elevators@s44  (search X=[8000, 18], holdout n=2000)


### naive-lightgbm

- **holdout RMSE: 0.00225** (winner retrained in 0.30s, cv score of winner: 0.0026)
- cv best RMSE: 0.0026, median: 0.0027, p10: 0.0026
- train: median 0.168s/fold, mean 0.162s, p90 0.217s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.696 |
| `num_leaves` | 0.084 |
| `learning_rate` | 0.066 |
| `feature_fraction` | 0.054 |
| `max_depth` | 0.037 |
| `extra_trees` | 0.024 |
| `min_data_in_leaf` | 0.019 |
| `bagging_freq` | 0.009 |
| `bagging_fraction` | 0.007 |
| `lambda_l2` | 0.004 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 469 | 0.0028 | 0.0004 | 0.0026 |
| True | 31 | 0.0034 | 0.0007 | 0.0029 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0030 | 0.0027 | 0.0027 | 0.0028 | **Q2** [0.1225, 0.1411] |
| `num_leaves` | 0.0028 | 0.0029 | 0.0030 | 0.0027 | **Q4** [121.0, ∞) |
| `max_depth` | 0.0031 | 0.0028 | — | 0.0028 | **Q2** [6.0, 8.0] |
| `min_data_in_leaf` | 0.0029 | 0.0028 | 0.0027 | 0.0029 | **Q3** [19.0, 23.0] |
| `lambda_l1` | 0.0029 | 0.0027 | 0.0027 | 0.0030 | **Q2** [0.0, 0.0001] |
| `lambda_l2` | 0.0028 | 0.0027 | 0.0028 | 0.0030 | **Q2** [0.0, 0.0001] |
| `feature_fraction` | 0.0031 | 0.0028 | 0.0027 | 0.0027 | **Q3** [0.9642, 0.9813] |
| `bagging_fraction` | 0.0030 | 0.0028 | 0.0028 | 0.0027 | **Q4** [0.9802, ∞) |

#### E. Slice plot

![elevators@s44/naive-lightgbm](slice_elevators@s44_naive-lightgbm.png)


### moe

- **holdout RMSE: 0.00228** (winner retrained in 1.82s, cv score of winner: 0.0026)
- cv best RMSE: 0.0026, median: 0.0028, p10: 0.0027
- train: median 1.422s/fold, mean 1.904s, p90 3.024s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.211 |
| `num_leaves` | 0.171 |
| `mixture_hard_m_step` | 0.126 |
| `mixture_gate_type` | 0.096 |
| `bagging_fraction` | 0.059 |
| `mixture_diversity_lambda` | 0.046 |
| `mixture_warmup_iters` | 0.044 |
| `min_data_in_leaf` | 0.038 |
| `mixture_balance_factor` | 0.034 |
| `mixture_expert_dropout_rate` | 0.034 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **gbdt** | 0.0036 (n=449) | leaf_reuse | Δ +0.0030 | p=1.39e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 449 | 0.0036 | 0.0026 | 0.0026 |
| leaf_reuse | 25 | 0.0066 | 0.0041 | 0.0032 |
| none | 26 | 0.0069 | 0.0039 | 0.0034 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 468 | 0.0038 | 0.0029 | 0.0026 |
| expert_choice | 32 | 0.0050 | 0.0033 | 0.0031 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 446 | 0.0038 | 0.0028 | 0.0026 |
| em | 29 | 0.0050 | 0.0035 | 0.0027 |
| gate_only | 25 | 0.0054 | 0.0045 | 0.0029 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gmm | 366 | 0.0036 | 0.0026 | 0.0026 |
| uniform | 25 | 0.0036 | 0.0016 | 0.0027 |
| tree_hierarchical | 18 | 0.0048 | 0.0025 | 0.0028 |
| kmeans_features | 31 | 0.0049 | 0.0037 | 0.0027 |
| gmm_features | 41 | 0.0051 | 0.0044 | 0.0027 |
| random | 19 | 0.0054 | 0.0045 | 0.0027 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 248 | 0.0038 | 0.0025 | 0.0026 |
| markov | 230 | 0.0039 | 0.0031 | 0.0027 |
| ema | 22 | 0.0057 | 0.0048 | 0.0029 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 468 | 0.0035 | 0.0023 | 0.0026 |
| False | 32 | 0.0095 | 0.0051 | 0.0033 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 0.0038 | 0.0029 | 0.0026 |
| True | 32 | 0.0056 | 0.0037 | 0.0030 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.0047 | — | — | 0.0038 | **Q4** [4.0, ∞) |
| `mixture_diversity_lambda` | 0.0038 | 0.0041 | 0.0040 | 0.0038 | **Q1** [None, 0.2282] |
| `mixture_warmup_iters` | 0.0035 | 0.0036 | 0.0038 | 0.0047 | **Q1** [None, 12.0] |
| `mixture_balance_factor` | 0.0045 | 0.0033 | 0.0045 | 0.0039 | **Q2** [6.0, 8.0] |
| `learning_rate` | 0.0045 | 0.0033 | 0.0037 | 0.0042 | **Q2** [0.1352, 0.1598] |
| `num_leaves` | 0.0038 | 0.0037 | 0.0039 | 0.0042 | **Q2** [48.0, 61.5] |
| `max_depth` | 0.0045 | — | 0.0038 | 0.0036 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 0.0041 | 0.0036 | 0.0036 | 0.0043 | **Q2** [27.0, 35.0] |

#### E. Slice plot

![elevators@s44/moe](slice_elevators@s44_moe.png)


---

## Overall recommendations

**Categorical settings that are statistically significant winners (p<0.01):**

| dataset | param | best value | Δ vs runner-up | p |
|---|---|---|---|---|
| elevators@s43 | `mixture_routing_mode` | **expert_choice** | +0.0021 | 1.91e-04 |
| elevators@s43 | `mixture_e_step_mode` | **loss_only** | +0.0015 | 6.42e-03 |
| elevators@s43 | `mixture_r_smoothing` | **markov** | +0.0012 | 5.34e-03 |
| elevators@s44 | `mixture_gate_type` | **gbdt** | +0.0030 | 1.39e-03 |
