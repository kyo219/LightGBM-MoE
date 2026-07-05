# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 300

- **Datasets**: ['cpu_act'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `5232455cdc8d`, lib sha256 `5cec0a0bd5ab…`, package `/tmp/lgbm-moe-v080/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| cpu_act | naive-lightgbm | **2.51321** ± 0.31795 | 2.45180 | 0.0% | 0.22 |
| cpu_act | moe | **2.49613** ± 0.29179 | 2.42641 | 0.0% | 2.20 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| cpu_act@s42 | naive-lightgbm | 2.5168 | 2.5887 | 0.160 | 263 |
| cpu_act@s42 | moe | 2.4848 | 2.5813 | 1.170 | 2132 |
| cpu_act@s43 | naive-lightgbm | 2.3606 | 2.4179 | 0.209 | 324 |
| cpu_act@s43 | moe | 2.3174 | 2.3739 | 0.899 | 2620 |
| cpu_act@s44 | naive-lightgbm | 2.4780 | 2.5416 | 0.167 | 260 |
| cpu_act@s44 | moe | 2.4770 | 2.6136 | 1.005 | 3331 |



---

## cpu_act@s42  (search X=[6554, 21], holdout n=1638)


### naive-lightgbm

- **holdout RMSE: 2.30096** (winner retrained in 0.14s, cv score of winner: 2.5168)
- cv best RMSE: 2.5168, median: 2.5887, p10: 2.5383
- train: median 0.160s/fold, mean 0.170s, p90 0.227s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.596 |
| `min_data_in_leaf` | 0.197 |
| `extra_trees` | 0.125 |
| `feature_fraction` | 0.031 |
| `num_leaves` | 0.025 |
| `max_depth` | 0.015 |
| `bagging_fraction` | 0.005 |
| `bagging_freq` | 0.004 |
| `lambda_l1` | 0.002 |
| `lambda_l2` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 278 | 2.7237 | 0.5702 | 2.5168 |
| True | 22 | 5.1620 | 2.8843 | 2.9974 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 3.5242 | 2.5987 | 2.6638 | 2.8232 | **Q2** [0.0878, 0.1015] |
| `num_leaves` | 2.7932 | 2.6426 | 2.7366 | 3.4317 | **Q2** [17.0, 22.0] |
| `max_depth` | 3.5982 | 2.7181 | — | 2.6996 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 2.8011 | 2.5874 | 2.6193 | 3.6265 | **Q2** [7.0, 9.0] |
| `lambda_l1` | 3.1783 | 3.0481 | 2.5990 | 2.7845 | **Q3** [0.101, 0.2619] |
| `lambda_l2` | 2.7125 | 2.6434 | 2.9249 | 3.3292 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 3.3781 | 2.7630 | 2.6453 | 2.8235 | **Q3** [0.86, 0.8808] |
| `bagging_fraction` | 2.8270 | 2.6803 | 3.2550 | 2.8477 | **Q2** [0.5773, 0.6504] |

#### E. Slice plot

![cpu_act@s42/naive-lightgbm](slice_cpu_act@s42_naive-lightgbm.png)


### moe

- **holdout RMSE: 2.30423** (winner retrained in 1.23s, cv score of winner: 2.4848)
- cv best RMSE: 2.4848, median: 2.5813, p10: 2.5252
- train: median 1.170s/fold, mean 1.406s, p90 2.142s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.307 |
| `feature_fraction` | 0.279 |
| `bagging_fraction` | 0.151 |
| `bagging_freq` | 0.054 |
| `min_data_in_leaf` | 0.052 |
| `mixture_hard_m_step` | 0.035 |
| `mixture_init` | 0.030 |
| `mixture_diversity_lambda` | 0.020 |
| `extra_trees` | 0.018 |
| `lambda_l1` | 0.015 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 138 | 3.7097 | 4.3838 | 2.5208 |
| none | 128 | 3.9698 | 4.6263 | 2.4848 |
| leaf_reuse | 34 | 5.7558 | 8.8337 | 2.6692 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 265 | 3.6859 | 3.9992 | 2.4848 |
| expert_choice | 35 | 6.8290 | 10.1654 | 2.5357 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 226 | 3.7664 | 5.2810 | 2.4848 |
| gate_only | 37 | 4.7635 | 5.2386 | 2.5686 |
| em | 37 | 5.0901 | 4.5471 | 2.5125 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 226 | 3.1780 | 3.3835 | 2.4848 |
| random | 32 | 4.0602 | 2.5049 | 2.6692 |
| tree_hierarchical | 14 | 8.2573 | 4.9157 | 2.6038 |
| gmm | 28 | 9.0005 | 11.8446 | 3.3571 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 136 | 3.7432 | 4.3392 | 2.5208 |
| ema | 127 | 4.0293 | 4.7817 | 2.4848 |
| markov | 37 | 5.2699 | 8.4279 | 2.5686 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 280 | 3.7100 | 4.9416 | 2.4848 |
| True | 20 | 8.8494 | 6.4416 | 2.7503 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 264 | 3.6769 | 4.6736 | 2.4848 |
| True | 36 | 6.8078 | 7.6078 | 3.1139 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 6.3463 | — | — | 3.7787 | **Q4** [3.0, ∞) |
| `mixture_diversity_lambda` | 4.7956 | 3.9424 | 4.0038 | 3.4685 | **Q4** [0.3923, ∞) |
| `mixture_warmup_iters` | 3.4996 | 3.6237 | 4.4049 | 4.6587 | **Q1** [None, 8.0] |
| `mixture_balance_factor` | 3.4063 | — | 3.4891 | 4.9469 | **Q1** [None, 3.0] |
| `learning_rate` | 5.3836 | 3.2309 | 3.4370 | 4.1589 | **Q2** [0.0954, 0.1241] |
| `num_leaves` | 5.0468 | 3.0635 | 3.5512 | 4.6115 | **Q2** [67.0, 81.0] |
| `max_depth` | 6.8295 | — | 3.2250 | 4.0527 | **Q3** [7.0, 9.0] |
| `min_data_in_leaf` | 3.1161 | 3.3995 | 3.6909 | 5.8981 | **Q1** [None, 7.0] |

#### E. Slice plot

![cpu_act@s42/moe](slice_cpu_act@s42_moe.png)


---

## cpu_act@s43  (search X=[6554, 21], holdout n=1638)


### naive-lightgbm

- **holdout RMSE: 2.96263** (winner retrained in 0.33s, cv score of winner: 2.3606)
- cv best RMSE: 2.3606, median: 2.4179, p10: 2.3767
- train: median 0.209s/fold, mean 0.211s, p90 0.286s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.534 |
| `learning_rate` | 0.347 |
| `bagging_fraction` | 0.025 |
| `feature_fraction` | 0.023 |
| `num_leaves` | 0.019 |
| `extra_trees` | 0.019 |
| `bagging_freq` | 0.018 |
| `max_depth` | 0.009 |
| `lambda_l1` | 0.006 |
| `lambda_l2` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 279 | 2.6156 | 0.7424 | 2.3606 |
| True | 21 | 3.9526 | 1.4165 | 2.8650 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 3.0578 | 2.5427 | 2.6238 | 2.6125 | **Q2** [0.0635, 0.0793] |
| `num_leaves` | 2.6429 | 2.5224 | 2.5676 | 3.0855 | **Q2** [20.0, 26.0] |
| `max_depth` | 3.0309 | 2.5906 | 2.5758 | 2.6500 | **Q3** [10.0, 11.0] |
| `min_data_in_leaf` | 2.4133 | 2.5333 | 2.4333 | 3.4038 | **Q1** [None, 6.0] |
| `lambda_l1` | 2.5531 | 2.6273 | 2.9238 | 2.7327 | **Q1** [None, 0.0] |
| `lambda_l2` | 2.7548 | 2.5349 | 2.6795 | 2.8676 | **Q2** [0.0001, 0.009] |
| `feature_fraction` | 2.8955 | 2.6856 | 2.7271 | 2.5286 | **Q4** [0.9543, ∞) |
| `bagging_fraction` | 2.7772 | 2.7105 | 2.6495 | 2.6996 | **Q3** [0.7722, 0.8744] |

#### E. Slice plot

![cpu_act@s43/naive-lightgbm](slice_cpu_act@s43_naive-lightgbm.png)


### moe

- **holdout RMSE: 2.90845** (winner retrained in 4.47s, cv score of winner: 2.3174)
- cv best RMSE: 2.3174, median: 2.3739, p10: 2.3370
- train: median 0.899s/fold, mean 1.733s, p90 4.026s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.662 |
| `num_leaves` | 0.159 |
| `mixture_warmup_iters` | 0.046 |
| `mixture_diversity_lambda` | 0.044 |
| `mixture_init` | 0.031 |
| `mixture_gate_type` | 0.013 |
| `mixture_refit_leaves` | 0.009 |
| `bagging_fraction` | 0.007 |
| `extra_trees` | 0.006 |
| `mixture_routing_mode` | 0.006 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 106 | 3.0408 | 3.4764 | 2.3174 |
| gbdt | 79 | 3.9102 | 4.6244 | 2.3930 |
| none | 115 | 4.4517 | 8.5199 | 2.3286 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 266 | 3.4790 | 4.9914 | 2.3174 |
| expert_choice | 34 | 6.4044 | 11.5607 | 2.3432 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 191 | 3.5155 | 6.2933 | 2.3174 |
| loss_only | 79 | 3.8125 | 4.6376 | 2.3930 |
| em | 30 | 5.6839 | 8.2533 | 2.3247 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| random | 46 | 3.3079 | 4.2808 | 2.3930 |
| uniform | 195 | 3.3327 | 5.7025 | 2.3174 |
| gmm | 27 | 4.9996 | 8.7521 | 2.5003 |
| tree_hierarchical | 32 | 6.4422 | 7.6258 | 2.4584 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 228 | 3.0068 | 4.0365 | 2.3174 |
| ema | 34 | 5.7331 | 11.1313 | 2.3386 |
| none | 38 | 6.9131 | 8.6797 | 2.3978 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 277 | 3.3452 | 5.5157 | 2.3174 |
| True | 23 | 9.4155 | 9.8231 | 2.8489 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 280 | 3.6563 | 6.1640 | 2.3174 |
| True | 20 | 5.9698 | 5.8773 | 2.7662 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 4.1578 | — | — | 3.7477 | **Q4** [3.0, ∞) |
| `mixture_diversity_lambda` | 4.2317 | 2.6267 | 4.2741 | 4.1097 | **Q2** [0.0941, 0.1669] |
| `mixture_warmup_iters` | 5.8469 | 3.1617 | 2.7876 | 3.5569 | **Q3** [43.0, 46.0] |
| `mixture_balance_factor` | 4.9209 | 4.1736 | — | 3.4015 | **Q4** [9.0, ∞) |
| `learning_rate` | 7.5818 | 2.5667 | 2.4806 | 2.6132 | **Q3** [0.1729, 0.2024] |
| `num_leaves` | 3.1524 | 3.6501 | 2.8125 | 5.4270 | **Q3** [33.0, 43.0] |
| `max_depth` | 4.6517 | 4.1283 | 3.0262 | 4.0161 | **Q3** [8.0, 10.0] |
| `min_data_in_leaf` | 2.5337 | 2.9152 | 3.9003 | 5.5045 | **Q1** [None, 7.0] |

#### E. Slice plot

![cpu_act@s43/moe](slice_cpu_act@s43_moe.png)


---

## cpu_act@s44  (search X=[6554, 21], holdout n=1638)


### naive-lightgbm

- **holdout RMSE: 2.27606** (winner retrained in 0.19s, cv score of winner: 2.4780)
- cv best RMSE: 2.4780, median: 2.5416, p10: 2.5063
- train: median 0.167s/fold, mean 0.168s, p90 0.220s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.556 |
| `learning_rate` | 0.242 |
| `extra_trees` | 0.080 |
| `feature_fraction` | 0.053 |
| `bagging_fraction` | 0.026 |
| `num_leaves` | 0.016 |
| `bagging_freq` | 0.012 |
| `max_depth` | 0.011 |
| `lambda_l2` | 0.004 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 280 | 2.7136 | 0.6089 | 2.4780 |
| True | 20 | 3.8921 | 1.1427 | 2.8796 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 3.1451 | 2.5864 | 2.6362 | 2.8009 | **Q2** [0.0864, 0.0971] |
| `num_leaves` | 2.6861 | 2.6349 | 2.6778 | 3.1672 | **Q2** [19.0, 23.0] |
| `max_depth` | 2.9955 | 2.8130 | — | 2.7386 | **Q4** [7.0, ∞) |
| `min_data_in_leaf` | 2.5623 | 2.5740 | 2.5874 | 3.4190 | **Q1** [None, 6.0] |
| `lambda_l1` | 2.6384 | 3.0447 | 2.6391 | 2.8463 | **Q1** [None, 0.0] |
| `lambda_l2` | 2.6267 | 2.8511 | 2.6385 | 3.0522 | **Q1** [None, 0.0] |
| `feature_fraction` | 3.2942 | 2.6950 | 2.5976 | 2.5817 | **Q4** [0.9903, ∞) |
| `bagging_fraction` | 3.1352 | 2.7428 | 2.6153 | 2.6753 | **Q3** [0.9574, 0.9762] |

#### E. Slice plot

![cpu_act@s44/naive-lightgbm](slice_cpu_act@s44_naive-lightgbm.png)


### moe

- **holdout RMSE: 2.27571** (winner retrained in 0.89s, cv score of winner: 2.4770)
- cv best RMSE: 2.4770, median: 2.6136, p10: 2.4974
- train: median 1.005s/fold, mean 2.207s, p90 8.382s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.353 |
| `min_data_in_leaf` | 0.211 |
| `feature_fraction` | 0.191 |
| `mixture_diversity_lambda` | 0.107 |
| `bagging_fraction` | 0.025 |
| `bagging_freq` | 0.020 |
| `mixture_num_experts` | 0.019 |
| `extra_trees` | 0.017 |
| `max_depth` | 0.011 |
| `mixture_r_smoothing` | 0.011 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_r_smoothing` | **ema** | 3.1926 (n=238) | markov | Δ +3.1306 | p=3.61e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 136 | 2.9738 | 1.8523 | 2.4770 |
| none | 109 | 3.9460 | 5.6487 | 2.5051 |
| gbdt | 55 | 6.4395 | 7.3464 | 2.6392 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 270 | 3.6630 | 4.2893 | 2.4770 |
| expert_choice | 30 | 6.6567 | 8.5122 | 2.5943 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| em | 162 | 3.1389 | 2.2196 | 2.4770 |
| loss_only | 72 | 4.0072 | 6.2746 | 2.4780 |
| gate_only | 66 | 5.9350 | 7.1602 | 2.6392 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 147 | 2.8227 | 1.2897 | 2.4770 |
| random | 88 | 4.0831 | 5.2335 | 2.5051 |
| tree_hierarchical | 24 | 4.5702 | 4.3545 | 2.5943 |
| gmm | 41 | 7.4340 | 9.3421 | 2.7447 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 238 | 3.1926 | 3.3147 | 2.4770 |
| markov | 46 | 6.3232 | 6.7175 | 2.7447 |
| none | 16 | 8.6263 | 11.0315 | 2.5797 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 243 | 3.3320 | 4.0673 | 2.4770 |
| True | 57 | 6.6497 | 7.0774 | 2.6722 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 279 | 3.8079 | 5.0394 | 2.4770 |
| True | 21 | 6.0155 | 3.1062 | 3.1574 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 7.2549 | — | — | 3.0695 | **Q4** [4.0, ∞) |
| `mixture_diversity_lambda` | 6.2211 | 3.8125 | 2.7086 | 3.1074 | **Q3** [0.2894, 0.3216] |
| `mixture_warmup_iters` | 3.0516 | 3.4447 | 3.1039 | 5.9921 | **Q1** [None, 12.0] |
| `mixture_balance_factor` | 3.0644 | 4.3002 | 6.9343 | 3.1454 | **Q1** [None, 3.0] |
| `learning_rate` | 6.6621 | 3.1236 | 3.1326 | 2.9313 | **Q4** [0.2025, ∞) |
| `num_leaves` | 2.7317 | 3.1752 | 4.3924 | 5.5503 | **Q1** [None, 27.75] |
| `max_depth` | 3.6869 | 4.2288 | 4.0847 | 3.8812 | **Q1** [None, 7.0] |
| `min_data_in_leaf` | 2.6033 | 3.4869 | 3.0687 | 6.4697 | **Q1** [None, 7.0] |

#### E. Slice plot

![cpu_act@s44/moe](slice_cpu_act@s44_moe.png)


---

## Overall recommendations

**Categorical settings that are statistically significant winners (p<0.01):**

| dataset | param | best value | Δ vs runner-up | p |
|---|---|---|---|---|
| cpu_act@s44 | `mixture_r_smoothing` | **ema** | +3.1306 | 3.61e-03 |
