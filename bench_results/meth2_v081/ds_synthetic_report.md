# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 300

- **Datasets**: ['synthetic'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `0cf3634c544c`, lib sha256 `a3c30fcc7fd3…`, package `/home/mokumoku/Develop/LightGBM-MoE/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| synthetic | naive-lightgbm | **4.81200** ± 0.33985 | 5.55933 | 0.0% | 0.20 |
| synthetic | naive-ensemble | **4.70621** ± 0.37221 | 5.46794 | 0.0% | 0.53 |
| synthetic | moe | **3.84145** ± 0.28986 | 4.44212 | 0.0% | 0.32 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| synthetic@s42 | naive-lightgbm | 5.3381 | 5.5618 | 0.137 | 196 |
| synthetic@s42 | naive-ensemble | 5.3266 | 5.6435 | 0.234 | 368 |
| synthetic@s42 | moe | 4.4217 | 5.0414 | 0.565 | 1392 |
| synthetic@s43 | naive-lightgbm | 5.5380 | 5.8312 | 0.099 | 147 |
| synthetic@s43 | naive-ensemble | 5.4878 | 5.7482 | 0.452 | 617 |
| synthetic@s43 | moe | 4.3519 | 5.1719 | 0.189 | 381 |
| synthetic@s44 | naive-lightgbm | 5.8019 | 6.1616 | 0.093 | 139 |
| synthetic@s44 | naive-ensemble | 5.5894 | 5.8602 | 0.276 | 398 |
| synthetic@s44 | moe | 4.5527 | 4.9532 | 0.226 | 434 |



---

## synthetic@s42  (search X=[1600, 5], holdout n=400)


### naive-lightgbm

- **holdout RMSE: 4.34766** (winner retrained in 0.18s, cv score of winner: 5.3381)
- cv best RMSE: 5.3381, median: 5.5618, p10: 5.4179
- train: median 0.137s/fold, mean 0.127s, p90 0.172s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.557 |
| `learning_rate` | 0.306 |
| `feature_fraction` | 0.072 |
| `extra_trees` | 0.038 |
| `bagging_fraction` | 0.012 |
| `bagging_freq` | 0.007 |
| `num_leaves` | 0.006 |
| `max_depth` | 0.001 |
| `lambda_l1` | 0.001 |
| `lambda_l2` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 278 | 5.7325 | 0.4837 | 5.3381 |
| True | 22 | 7.1469 | 1.5229 | 5.6091 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 6.1776 | 5.6327 | 5.6468 | 5.8878 | **Q2** [0.0423, 0.0493] |
| `num_leaves` | 5.6674 | 5.7401 | 5.6393 | 6.3012 | **Q3** [33.0, 50.25] |
| `max_depth` | 6.5140 | 5.7722 | — | 5.7153 | **Q4** [9.0, ∞) |
| `min_data_in_leaf` | 5.5472 | 5.5529 | 5.6386 | 6.5512 | **Q1** [None, 6.0] |
| `lambda_l1` | 6.3199 | 5.7260 | 5.6546 | 5.6443 | **Q4** [2.3125, ∞) |
| `lambda_l2` | 5.7133 | 5.6388 | 5.9086 | 6.0841 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 6.3681 | 5.6852 | 5.6903 | 5.6013 | **Q4** [0.9868, ∞) |
| `bagging_fraction` | 5.7752 | 5.6650 | 5.8987 | 6.0060 | **Q2** [0.6994, 0.7301] |

#### E. Slice plot

![synthetic@s42/naive-lightgbm](slice_synthetic@s42_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 4.82120** (winner retrained in 0.25s, cv score of winner: 5.3266)
- cv best RMSE: 5.3266, median: 5.6435, p10: 5.4218
- train: median 0.234s/fold, mean 0.241s, p90 0.361s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.771 |
| `learning_rate` | 0.167 |
| `bagging_fraction` | 0.018 |
| `bagging_freq` | 0.014 |
| `feature_fraction` | 0.013 |
| `max_depth` | 0.007 |
| `extra_trees` | 0.004 |
| `num_leaves` | 0.004 |
| `lambda_l1` | 0.002 |
| `n_models` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 250 | 5.8537 | 0.7006 | 5.3266 |
| False | 50 | 6.1501 | 0.6342 | 5.3736 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 6.3160 | 5.5465 | 5.8407 | 5.9091 | **Q2** [0.1005, 0.1213] |
| `num_leaves` | 5.6761 | 5.9371 | 6.0086 | 5.9610 | **Q1** [None, 29.0] |
| `max_depth` | 6.1669 | 5.9259 | — | 5.7924 | **Q4** [9.0, ∞) |
| `min_data_in_leaf` | — | 5.5474 | 5.7937 | 6.6627 | **Q2** [5.0, 8.0] |
| `lambda_l1` | 6.0419 | 5.8762 | 5.9732 | 5.7211 | **Q4** [0.0141, ∞) |
| `lambda_l2` | 6.0263 | 5.5534 | 5.9131 | 6.1196 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 6.3067 | 5.7731 | 5.9482 | 5.5844 | **Q4** [0.974, ∞) |
| `bagging_fraction` | 6.1215 | 5.8272 | 5.5425 | 6.1212 | **Q3** [0.7698, 0.7935] |

#### E. Slice plot

![synthetic@s42/naive-ensemble](slice_synthetic@s42_naive-ensemble.png)


### moe

- **holdout RMSE: 3.55247** (winner retrained in 0.45s, cv score of winner: 4.4217)
- cv best RMSE: 4.4217, median: 5.0414, p10: 4.6087
- train: median 0.565s/fold, mean 0.918s, p90 2.310s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_gate_type` | 0.570 |
| `feature_fraction` | 0.112 |
| `learning_rate` | 0.110 |
| `mixture_r_smoothing` | 0.038 |
| `mixture_diversity_lambda` | 0.033 |
| `min_data_in_leaf` | 0.032 |
| `num_leaves` | 0.016 |
| `mixture_e_step_mode` | 0.016 |
| `max_depth` | 0.016 |
| `mixture_init` | 0.014 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **gbdt** | 5.2636 (n=262) | leaf_reuse | Δ +2.7178 | p=0.00e+00 |
| `mixture_routing_mode` | **token_choice** | 5.5149 (n=251) | expert_choice | Δ +0.8190 | p=2.20e-05 |
| `mixture_e_step_mode` | **em** | 5.2991 (n=202) | loss_only | Δ +0.8846 | p=0.00e+00 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 262 | 5.2636 | 0.9013 | 4.4217 |
| leaf_reuse | 19 | 7.9814 | 1.2613 | 5.7873 |
| none | 19 | 8.6258 | 2.0513 | 6.0807 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 251 | 5.5149 | 1.4782 | 4.4217 |
| expert_choice | 49 | 6.3339 | 1.0856 | 4.5473 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| em | 202 | 5.2991 | 1.3582 | 4.4217 |
| loss_only | 84 | 6.1837 | 1.2667 | 4.9817 |
| gate_only | 14 | 7.4824 | 1.4744 | 5.5663 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| tree_hierarchical | 86 | 5.3702 | 1.4434 | 4.4594 |
| gmm | 171 | 5.5640 | 1.4785 | 4.4217 |
| uniform | 29 | 6.4585 | 0.9675 | 5.5425 |
| random | 14 | 6.7165 | 0.8732 | 5.4890 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 174 | 5.3828 | 1.4454 | 4.4217 |
| none | 69 | 5.8352 | 1.4815 | 4.4607 |
| ema | 57 | 6.2344 | 1.2172 | 5.0315 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 220 | 5.5453 | 1.4356 | 4.4217 |
| False | 80 | 5.9329 | 1.4638 | 4.4607 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 198 | 5.3223 | 1.4170 | 4.4217 |
| False | 102 | 6.2821 | 1.3065 | 4.9817 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | 5.7359 | — | 5.5829 | **Q4** [4.0, ∞) |
| `mixture_diversity_lambda` | 5.3194 | 5.3261 | 5.6359 | 6.3133 | **Q1** [None, 0.1011] |
| `mixture_warmup_iters` | 6.1503 | 5.9056 | 5.3849 | 5.2377 | **Q4** [44.0, ∞) |
| `mixture_balance_factor` | 5.8508 | 6.3514 | 5.4501 | 5.2388 | **Q4** [9.0, ∞) |
| `learning_rate` | 6.5120 | 5.3879 | 5.2027 | 5.4921 | **Q3** [0.2279, 0.2575] |
| `num_leaves` | 6.3066 | 5.3795 | 5.3044 | 5.5983 | **Q3** [73.0, 82.0] |
| `max_depth` | 6.3585 | 6.1610 | 5.2843 | 5.3388 | **Q3** [11.0, 12.0] |
| `min_data_in_leaf` | 5.1412 | 5.3353 | 5.4514 | 6.6116 | **Q1** [None, 8.0] |

#### E. Slice plot

![synthetic@s42/moe](slice_synthetic@s42_moe.png)


---

## synthetic@s43  (search X=[1600, 5], holdout n=400)


### naive-lightgbm

- **holdout RMSE: 5.15155** (winner retrained in 0.18s, cv score of winner: 5.5380)
- cv best RMSE: 5.5380, median: 5.8312, p10: 5.6698
- train: median 0.099s/fold, mean 0.095s, p90 0.139s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.699 |
| `learning_rate` | 0.153 |
| `bagging_fraction` | 0.059 |
| `feature_fraction` | 0.042 |
| `bagging_freq` | 0.021 |
| `max_depth` | 0.011 |
| `num_leaves` | 0.007 |
| `extra_trees` | 0.003 |
| `lambda_l1` | 0.002 |
| `lambda_l2` | 0.002 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 256 | 5.9714 | 0.4971 | 5.5380 |
| False | 44 | 6.4221 | 0.7100 | 5.7045 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 6.4069 | 5.8382 | 5.9328 | 5.9722 | **Q2** [0.1026, 0.1194] |
| `num_leaves` | 6.3075 | 5.9478 | 5.9577 | 5.9388 | **Q4** [121.0, ∞) |
| `max_depth` | 6.2932 | — | — | 5.9612 | **Q4** [9.0, ∞) |
| `min_data_in_leaf` | — | 5.7937 | 5.9089 | 6.5790 | **Q2** [5.0, 8.0] |
| `lambda_l1` | 6.2461 | 5.9448 | 5.9221 | 6.0371 | **Q3** [0.0012, 0.0059] |
| `lambda_l2` | 6.0309 | 5.9283 | 5.8122 | 6.3787 | **Q3** [0.0, 0.0] |
| `feature_fraction` | 6.4122 | 5.9660 | 5.8785 | 5.8934 | **Q3** [0.9478, 0.969] |
| `bagging_fraction` | 6.1562 | 5.9145 | 6.0390 | 6.0404 | **Q2** [0.8748, 0.9009] |

#### E. Slice plot

![synthetic@s43/naive-lightgbm](slice_synthetic@s43_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 4.20386** (winner retrained in 0.77s, cv score of winner: 5.4878)
- cv best RMSE: 5.4878, median: 5.7482, p10: 5.5967
- train: median 0.452s/fold, mean 0.408s, p90 0.613s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.815 |
| `feature_fraction` | 0.066 |
| `bagging_fraction` | 0.040 |
| `max_depth` | 0.028 |
| `learning_rate` | 0.017 |
| `bagging_freq` | 0.011 |
| `extra_trees` | 0.011 |
| `num_leaves` | 0.005 |
| `n_models` | 0.004 |
| `lambda_l2` | 0.003 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 278 | 5.9243 | 0.4901 | 5.4878 |
| True | 22 | 6.8062 | 1.0188 | 5.8158 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 6.1657 | 5.8619 | 5.8495 | 6.0787 | **Q3** [0.0386, 0.0479] |
| `num_leaves` | 6.0424 | 5.7672 | 5.9783 | 6.1498 | **Q2** [45.0, 50.0] |
| `max_depth` | 6.5915 | — | 5.9116 | 5.9378 | **Q3** [7.0, 10.0] |
| `min_data_in_leaf` | — | 5.7720 | 5.8312 | 6.5518 | **Q2** [5.0, 7.5] |
| `lambda_l1` | 6.1045 | 5.9138 | 6.0027 | 5.9347 | **Q2** [0.0036, 0.0555] |
| `lambda_l2` | 6.0069 | 5.8189 | 6.0567 | 6.0732 | **Q2** [0.0001, 0.0009] |
| `feature_fraction` | 6.4252 | 5.7921 | 5.7286 | 6.0099 | **Q3** [0.9296, 0.9526] |
| `bagging_fraction` | 6.1859 | 5.8332 | 5.8762 | 6.0605 | **Q2** [0.8334, 0.8678] |

#### E. Slice plot

![synthetic@s43/naive-ensemble](slice_synthetic@s43_naive-ensemble.png)


### moe

- **holdout RMSE: 3.73416** (winner retrained in 0.24s, cv score of winner: 4.3519)
- cv best RMSE: 4.3519, median: 5.1719, p10: 4.6125
- train: median 0.189s/fold, mean 0.247s, p90 0.331s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_gate_type` | 0.694 |
| `mixture_r_smoothing` | 0.091 |
| `feature_fraction` | 0.040 |
| `learning_rate` | 0.034 |
| `mixture_e_step_mode` | 0.025 |
| `lambda_l2` | 0.022 |
| `mixture_init` | 0.020 |
| `mixture_diversity_lambda` | 0.019 |
| `bagging_fraction` | 0.014 |
| `num_leaves` | 0.007 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **gbdt** | 5.2965 (n=268) | leaf_reuse | Δ +2.1759 | p=2.00e-06 |
| `mixture_routing_mode` | **token_choice** | 5.5015 (n=257) | expert_choice | Δ +1.0540 | p=9.00e-06 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 268 | 5.2965 | 0.7643 | 4.3519 |
| leaf_reuse | 16 | 7.4724 | 1.1596 | 6.1096 |
| none | 16 | 9.7962 | 1.7112 | 6.9344 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 257 | 5.5015 | 1.3577 | 4.3519 |
| expert_choice | 43 | 6.5555 | 1.2935 | 5.3140 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 158 | 5.4934 | 1.4800 | 4.3519 |
| gate_only | 107 | 5.6985 | 1.2310 | 4.6636 |
| em | 35 | 6.2308 | 1.3358 | 4.9673 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gmm | 256 | 5.4714 | 1.3354 | 4.3519 |
| tree_hierarchical | 15 | 6.4194 | 1.8426 | 4.7199 |
| random | 15 | 6.8514 | 1.0270 | 5.6138 |
| uniform | 14 | 6.8599 | 0.5419 | 5.9351 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 202 | 5.4849 | 1.4818 | 4.4695 |
| ema | 52 | 5.9132 | 1.0276 | 4.9673 |
| markov | 46 | 6.0941 | 1.2334 | 4.3519 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 253 | 5.5443 | 1.4285 | 4.3519 |
| False | 47 | 6.2356 | 1.0454 | 4.9673 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 242 | 5.4907 | 1.3538 | 4.3519 |
| False | 58 | 6.3279 | 1.3788 | 4.9673 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | 5.4289 | 6.0378 | 5.7574 | **Q2** [2.0, 3.0] |
| `mixture_diversity_lambda` | 5.5879 | 5.4439 | 5.4403 | 6.1381 | **Q3** [0.1756, 0.2369] |
| `mixture_warmup_iters` | 5.8324 | 5.6598 | 5.5643 | 5.5588 | **Q4** [47.0, ∞) |
| `mixture_balance_factor` | 6.0687 | 5.8410 | 5.5705 | 5.3730 | **Q4** [8.0, ∞) |
| `learning_rate` | 6.2725 | 5.4645 | 5.4221 | 5.4511 | **Q3** [0.1913, 0.2321] |
| `num_leaves` | 5.9291 | 5.5109 | 5.5470 | 5.6210 | **Q2** [87.0, 107.0] |
| `max_depth` | 5.9318 | 5.6242 | — | 5.5781 | **Q4** [7.0, ∞) |
| `min_data_in_leaf` | 5.2794 | 5.4864 | 5.5128 | 6.2745 | **Q1** [None, 9.0] |

#### E. Slice plot

![synthetic@s43/moe](slice_synthetic@s43_moe.png)


---

## synthetic@s44  (search X=[1600, 5], holdout n=400)


### naive-lightgbm

- **holdout RMSE: 4.93679** (winner retrained in 0.24s, cv score of winner: 5.8019)
- cv best RMSE: 5.8019, median: 6.1616, p10: 5.9403
- train: median 0.093s/fold, mean 0.090s, p90 0.142s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.883 |
| `learning_rate` | 0.061 |
| `feature_fraction` | 0.015 |
| `extra_trees` | 0.012 |
| `bagging_fraction` | 0.010 |
| `max_depth` | 0.007 |
| `num_leaves` | 0.005 |
| `lambda_l2` | 0.003 |
| `lambda_l1` | 0.002 |
| `bagging_freq` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 280 | 6.3322 | 0.5460 | 5.8019 |
| True | 20 | 7.2502 | 0.9051 | 6.1956 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 6.6490 | 6.1474 | 6.2803 | 6.4967 | **Q2** [0.04, 0.0469] |
| `num_leaves` | 6.8672 | 6.1468 | 6.3410 | 6.2123 | **Q2** [84.75, 98.0] |
| `max_depth` | 6.8683 | 6.4367 | — | 6.2094 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 6.1255 | 6.1508 | 6.1991 | 7.0654 | **Q1** [None, 7.0] |
| `lambda_l1` | 6.8510 | 6.3558 | 6.1514 | 6.2152 | **Q3** [1.2129, 3.1615] |
| `lambda_l2` | 6.2652 | 6.2468 | 6.4952 | 6.5662 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 6.8683 | 6.2573 | 6.2016 | 6.2462 | **Q3** [0.9281, 0.955] |
| `bagging_fraction` | 6.3800 | 6.2264 | 6.3471 | 6.6200 | **Q2** [0.5741, 0.6123] |

#### E. Slice plot

![synthetic@s44/naive-lightgbm](slice_synthetic@s44_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 5.09358** (winner retrained in 0.57s, cv score of winner: 5.5894)
- cv best RMSE: 5.5894, median: 5.8602, p10: 5.6519
- train: median 0.276s/fold, mean 0.263s, p90 0.434s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.573 |
| `learning_rate` | 0.279 |
| `feature_fraction` | 0.070 |
| `bagging_fraction` | 0.021 |
| `max_depth` | 0.016 |
| `extra_trees` | 0.012 |
| `n_models` | 0.012 |
| `num_leaves` | 0.011 |
| `bagging_freq` | 0.006 |
| `lambda_l2` | 0.002 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 216 | 5.9411 | 0.5217 | 5.5894 |
| True | 84 | 6.4898 | 0.6846 | 5.8148 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 6.3443 | 5.8764 | 6.0718 | 6.0864 | **Q2** [0.0808, 0.1071] |
| `num_leaves` | 6.5896 | 6.1207 | 5.8756 | 5.8012 | **Q4** [123.0, ∞) |
| `max_depth` | 6.7148 | 5.9937 | — | 5.9191 | **Q4** [11.0, ∞) |
| `min_data_in_leaf` | 5.8144 | 5.7974 | 5.9017 | 6.7552 | **Q2** [7.0, 8.0] |
| `lambda_l1` | 6.3266 | 5.9704 | 5.9226 | 6.1592 | **Q3** [0.0001, 0.0022] |
| `lambda_l2` | 6.0433 | 5.9823 | 6.0762 | 6.2772 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 6.6091 | 5.8285 | 5.9776 | 5.9637 | **Q2** [0.9032, 0.9279] |
| `bagging_fraction` | 6.4481 | 5.9652 | 5.8783 | 6.0873 | **Q3** [0.8834, 0.9219] |

#### E. Slice plot

![synthetic@s44/naive-ensemble](slice_synthetic@s44_naive-ensemble.png)


### moe

- **holdout RMSE: 4.23773** (winner retrained in 0.26s, cv score of winner: 4.5527)
- cv best RMSE: 4.5527, median: 4.9532, p10: 4.6572
- train: median 0.226s/fold, mean 0.283s, p90 0.383s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_gate_type` | 0.577 |
| `learning_rate` | 0.105 |
| `mixture_init` | 0.066 |
| `min_data_in_leaf` | 0.051 |
| `extra_trees` | 0.035 |
| `mixture_warmup_iters` | 0.034 |
| `mixture_diversity_lambda` | 0.031 |
| `num_leaves` | 0.013 |
| `lambda_l2` | 0.013 |
| `max_depth` | 0.012 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **gbdt** | 5.2099 (n=266) | leaf_reuse | Δ +2.3475 | p=0.00e+00 |
| `mixture_routing_mode` | **expert_choice** | 5.3759 (n=262) | token_choice | Δ +1.4336 | p=0.00e+00 |
| `mixture_e_step_mode` | **em** | 5.2375 (n=209) | loss_only | Δ +0.8062 | p=2.23e-04 |
| `mixture_init` | **gmm** | 5.2887 (n=243) | tree_hierarchical | Δ +1.2040 | p=1.00e-06 |
| `mixture_r_smoothing` | **markov** | 5.2904 (n=207) | none | Δ +0.6017 | p=1.41e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 266 | 5.2099 | 0.7610 | 4.5527 |
| leaf_reuse | 17 | 7.5574 | 1.1510 | 5.8350 |
| none | 17 | 8.9971 | 1.6482 | 6.0393 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 262 | 5.3759 | 1.2128 | 4.5527 |
| token_choice | 38 | 6.8095 | 1.3616 | 4.6746 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| em | 209 | 5.2375 | 1.1417 | 4.5527 |
| loss_only | 59 | 6.0437 | 1.4674 | 4.8819 |
| gate_only | 32 | 6.7515 | 1.1670 | 4.6687 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gmm | 243 | 5.2887 | 1.2267 | 4.5527 |
| tree_hierarchical | 28 | 6.4927 | 0.9653 | 4.8765 |
| random | 15 | 6.8086 | 0.9539 | 5.9806 |
| uniform | 14 | 7.0125 | 1.2947 | 5.8111 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 207 | 5.2904 | 1.2501 | 4.5527 |
| none | 63 | 5.8921 | 1.2692 | 4.8819 |
| ema | 30 | 6.6982 | 1.1284 | 4.8327 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 264 | 5.3551 | 1.1660 | 4.5527 |
| True | 36 | 7.0418 | 1.4433 | 4.7372 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 264 | 5.3828 | 1.2507 | 4.5527 |
| True | 36 | 6.8388 | 1.1043 | 4.7632 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 6.8096 | — | — | 5.4486 | **Q4** [3.0, ∞) |
| `mixture_diversity_lambda` | 5.6744 | 5.3777 | 5.4219 | 5.7560 | **Q2** [0.1999, 0.233] |
| `mixture_warmup_iters` | 6.3596 | 5.6476 | 5.3026 | 5.0661 | **Q4** [48.0, ∞) |
| `mixture_balance_factor` | — | — | 5.2479 | 6.1768 | **Q3** [2.0, 3.0] |
| `learning_rate` | 5.8129 | 5.2172 | 5.2922 | 5.9078 | **Q2** [0.0645, 0.0761] |
| `num_leaves` | 5.0925 | 5.2305 | 5.7727 | 6.1188 | **Q1** [None, 12.0] |
| `max_depth` | 5.8415 | 6.1093 | 5.4593 | 5.1543 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 5.7206 | 5.3179 | 5.2960 | 5.8497 | **Q3** [22.0, 29.0] |

#### E. Slice plot

![synthetic@s44/moe](slice_synthetic@s44_moe.png)


---

## Overall recommendations

**Categorical settings that are statistically significant winners (p<0.01):**

| dataset | param | best value | Δ vs runner-up | p |
|---|---|---|---|---|
| synthetic@s42 | `mixture_gate_type` | **gbdt** | +2.7178 | 0.00e+00 |
| synthetic@s42 | `mixture_routing_mode` | **token_choice** | +0.8190 | 2.20e-05 |
| synthetic@s42 | `mixture_e_step_mode` | **em** | +0.8846 | 0.00e+00 |
| synthetic@s43 | `mixture_gate_type` | **gbdt** | +2.1759 | 2.00e-06 |
| synthetic@s43 | `mixture_routing_mode` | **token_choice** | +1.0540 | 9.00e-06 |
| synthetic@s44 | `mixture_gate_type` | **gbdt** | +2.3475 | 0.00e+00 |
| synthetic@s44 | `mixture_routing_mode` | **expert_choice** | +1.4336 | 0.00e+00 |
| synthetic@s44 | `mixture_e_step_mode` | **em** | +0.8062 | 2.23e-04 |
| synthetic@s44 | `mixture_init` | **gmm** | +1.2040 | 1.00e-06 |
| synthetic@s44 | `mixture_r_smoothing` | **markov** | +0.6017 | 1.41e-03 |
