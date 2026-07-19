# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 500

- **Datasets**: ['cpu_act'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `747bb4a7b5e6`, lib sha256 `a3c30fcc7fd3…`, package `/home/mokumoku/Develop/LightGBM-MoE/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| cpu_act | naive-lightgbm | **2.49817** ± 0.29668 | 2.44834 | 0.0% | 0.17 |
| cpu_act | moe | **2.43085** ± 0.33341 | 2.38000 | 0.0% | 2.69 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| cpu_act@s42 | naive-lightgbm | 2.5168 | 2.5811 | 0.161 | 440 |
| cpu_act@s42 | moe | 2.4156 | 2.5414 | 1.848 | 5580 |
| cpu_act@s43 | naive-lightgbm | 2.3502 | 2.3963 | 0.195 | 499 |
| cpu_act@s43 | moe | 2.2936 | 2.3552 | 1.048 | 4036 |
| cpu_act@s44 | naive-lightgbm | 2.4780 | 2.5377 | 0.121 | 323 |
| cpu_act@s44 | moe | 2.4308 | 2.5295 | 0.601 | 1931 |



---

## cpu_act@s42  (search X=[6554, 21], holdout n=1638)


### naive-lightgbm

- **holdout RMSE: 2.30096** (winner retrained in 0.14s, cv score of winner: 2.5168)
- cv best RMSE: 2.5168, median: 2.5811, p10: 2.5383
- train: median 0.161s/fold, mean 0.170s, p90 0.224s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.523 |
| `min_data_in_leaf` | 0.326 |
| `extra_trees` | 0.080 |
| `num_leaves` | 0.028 |
| `bagging_freq` | 0.015 |
| `bagging_fraction` | 0.013 |
| `feature_fraction` | 0.009 |
| `max_depth` | 0.005 |
| `lambda_l1` | 0.002 |
| `lambda_l2` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 2.7109 | 0.5267 | 2.5168 |
| True | 32 | 4.6123 | 2.5358 | 2.9974 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 3.2347 | 2.6648 | 2.6706 | 2.7602 | **Q2** [0.0888, 0.1003] |
| `num_leaves` | 2.7448 | 2.6924 | 2.7699 | 3.0788 | **Q2** [18.0, 22.0] |
| `max_depth` | 3.4628 | 2.7405 | — | 2.7123 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 2.7492 | 2.5831 | 2.6221 | 3.3756 | **Q2** [7.0, 9.0] |
| `lambda_l1` | 3.0699 | 2.8129 | 2.6887 | 2.7588 | **Q3** [0.1361, 0.2856] |
| `lambda_l2` | 2.7000 | 2.6290 | 2.7748 | 3.2265 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 3.1650 | 2.7474 | 2.6873 | 2.7306 | **Q3** [0.8599, 0.8784] |
| `bagging_fraction` | 2.8267 | 2.6647 | 2.9527 | 2.8862 | **Q2** [0.5678, 0.5935] |

#### E. Slice plot

![cpu_act@s42/naive-lightgbm](slice_cpu_act@s42_naive-lightgbm.png)


### moe

- **holdout RMSE: 2.15948** (winner retrained in 3.91s, cv score of winner: 2.4156)
- cv best RMSE: 2.4156, median: 2.5414, p10: 2.4444
- train: median 1.848s/fold, mean 2.200s, p90 3.727s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.792 |
| `mixture_e_step_mode` | 0.124 |
| `mixture_init` | 0.031 |
| `mixture_diversity_lambda` | 0.011 |
| `mixture_warmup_iters` | 0.008 |
| `max_depth` | 0.004 |
| `bagging_freq` | 0.004 |
| `lambda_l1` | 0.003 |
| `bagging_fraction` | 0.003 |
| `min_data_in_leaf` | 0.003 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 450 | 3.7921 | 5.3561 | 2.4156 |
| gbdt | 25 | 5.3982 | 5.5241 | 2.5298 |
| none | 25 | 7.3902 | 8.0931 | 3.1988 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 468 | 3.8182 | 5.0504 | 2.4156 |
| expert_choice | 32 | 7.4761 | 10.1954 | 2.7236 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 448 | 3.5946 | 4.8581 | 2.4156 |
| loss_only | 25 | 7.2402 | 6.5495 | 2.9788 |
| em | 27 | 8.6954 | 10.6953 | 2.8078 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gmm_features | 382 | 3.4214 | 4.0397 | 2.4156 |
| tree_hierarchical | 26 | 3.9161 | 3.1549 | 2.4782 |
| kmeans_features | 33 | 5.1038 | 5.7075 | 2.4461 |
| random | 18 | 5.9172 | 10.1263 | 2.4755 |
| uniform | 20 | 7.2977 | 11.5074 | 2.4485 |
| gmm | 21 | 9.3571 | 11.0178 | 2.4836 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 306 | 3.5423 | 4.2733 | 2.4156 |
| markov | 170 | 4.3792 | 6.0961 | 2.4317 |
| none | 24 | 8.2402 | 11.6189 | 2.4658 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 468 | 3.7517 | 4.8868 | 2.4156 |
| False | 32 | 8.4497 | 10.9358 | 2.5890 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 3.6865 | 4.7635 | 2.4156 |
| True | 32 | 9.4020 | 11.2728 | 2.9617 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | 4.9951 | — | 3.4238 | **Q4** [6.0, ∞) |
| `mixture_diversity_lambda` | 4.3101 | 4.6922 | 3.2621 | 3.9448 | **Q3** [0.322, 0.3666] |
| `mixture_warmup_iters` | 4.8217 | 3.3383 | 3.4808 | 4.5727 | **Q2** [14.0, 16.0] |
| `mixture_balance_factor` | 3.7468 | 5.0436 | — | 3.7269 | **Q4** [8.0, ∞) |
| `learning_rate` | 7.3765 | 2.8910 | 2.8180 | 3.1238 | **Q3** [0.225, 0.2507] |
| `num_leaves` | 3.5466 | 3.4315 | 4.5939 | 4.6092 | **Q2** [19.75, 34.0] |
| `max_depth` | 4.0821 | 3.5576 | 3.5597 | 5.2373 | **Q2** [5.0, 6.0] |
| `min_data_in_leaf` | 3.7904 | 3.2789 | 3.9784 | 5.0832 | **Q2** [8.0, 11.0] |

#### E. Slice plot

![cpu_act@s42/moe](slice_cpu_act@s42_moe.png)


---

## cpu_act@s43  (search X=[6554, 21], holdout n=1638)


### naive-lightgbm

- **holdout RMSE: 2.91750** (winner retrained in 0.25s, cv score of winner: 2.3502)
- cv best RMSE: 2.3502, median: 2.3963, p10: 2.3683
- train: median 0.195s/fold, mean 0.194s, p90 0.260s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.520 |
| `learning_rate` | 0.337 |
| `extra_trees` | 0.055 |
| `bagging_fraction` | 0.026 |
| `num_leaves` | 0.025 |
| `feature_fraction` | 0.016 |
| `bagging_freq` | 0.014 |
| `lambda_l1` | 0.003 |
| `max_depth` | 0.002 |
| `lambda_l2` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 469 | 2.5500 | 0.6068 | 2.3502 |
| True | 31 | 3.6338 | 1.2558 | 2.8568 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.8419 | 2.5106 | 2.5571 | 2.5593 | **Q2** [0.0693, 0.0844] |
| `num_leaves` | 2.6072 | 2.5236 | 2.5118 | 2.8061 | **Q3** [28.0, 33.0] |
| `max_depth` | 2.8834 | 2.4895 | 2.5351 | 2.5840 | **Q2** [9.0, 10.0] |
| `min_data_in_leaf` | 2.4213 | 2.4891 | 2.4110 | 3.0877 | **Q3** [9.0, 12.0] |
| `lambda_l1` | 2.5797 | 2.7601 | 2.4987 | 2.6305 | **Q3** [0.1564, 1.2329] |
| `lambda_l2` | 2.6582 | 2.5230 | 2.5067 | 2.7811 | **Q3** [0.0026, 0.0157] |
| `feature_fraction` | 2.8475 | 2.5626 | 2.5383 | 2.5205 | **Q4** [0.9563, ∞) |
| `bagging_fraction` | 2.7173 | 2.5851 | 2.5059 | 2.6607 | **Q3** [0.782, 0.8282] |

#### E. Slice plot

![cpu_act@s43/naive-lightgbm](slice_cpu_act@s43_naive-lightgbm.png)


### moe

- **holdout RMSE: 2.90047** (winner retrained in 3.43s, cv score of winner: 2.2936)
- cv best RMSE: 2.2936, median: 2.3552, p10: 2.3178
- train: median 1.048s/fold, mean 1.591s, p90 3.049s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_init` | 0.496 |
| `learning_rate` | 0.147 |
| `mixture_num_experts` | 0.114 |
| `bagging_fraction` | 0.044 |
| `mixture_warmup_iters` | 0.039 |
| `max_depth` | 0.029 |
| `mixture_r_smoothing` | 0.022 |
| `mixture_diversity_lambda` | 0.022 |
| `mixture_balance_factor` | 0.018 |
| `num_leaves` | 0.017 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 232 | 3.6588 | 5.1703 | 2.2936 |
| none | 234 | 3.8429 | 5.9816 | 2.3135 |
| gbdt | 34 | 5.1122 | 5.5190 | 2.3214 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 268 | 3.8033 | 5.6311 | 2.3051 |
| expert_choice | 232 | 3.8905 | 5.5605 | 2.2936 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 342 | 3.4836 | 4.3618 | 2.2936 |
| em | 93 | 4.2277 | 6.6206 | 2.3200 |
| gate_only | 65 | 5.1899 | 8.6942 | 2.3220 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 391 | 2.9344 | 3.3467 | 2.2936 |
| tree_hierarchical | 24 | 3.4164 | 1.5798 | 2.4495 |
| random | 29 | 3.9463 | 4.0877 | 2.4319 |
| kmeans_features | 18 | 9.0574 | 11.4437 | 2.6145 |
| gmm | 19 | 9.1759 | 11.5666 | 2.6927 |
| gmm_features | 19 | 12.6694 | 12.5903 | 3.4985 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 370 | 3.2802 | 3.7560 | 2.2936 |
| markov | 54 | 4.4274 | 6.5096 | 2.3408 |
| none | 76 | 6.1730 | 10.0106 | 2.3270 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 3.6119 | 5.1453 | 2.2936 |
| True | 32 | 7.2351 | 9.5023 | 2.7916 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 470 | 3.6731 | 5.3620 | 2.2936 |
| True | 30 | 6.5180 | 8.0228 | 2.7333 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 5.0737 | — | — | 3.7162 | **Q4** [4.0, ∞) |
| `mixture_diversity_lambda` | 5.0114 | 3.2628 | 3.3566 | 3.7444 | **Q2** [0.2301, 0.383] |
| `mixture_warmup_iters` | 4.8878 | 3.0681 | 3.0147 | 4.1872 | **Q3** [42.0, 44.0] |
| `mixture_balance_factor` | 5.0976 | 4.1828 | — | 3.4774 | **Q4** [8.0, ∞) |
| `learning_rate` | 6.0344 | 3.0598 | 3.3755 | 2.9055 | **Q4** [0.2169, ∞) |
| `num_leaves` | 2.8956 | 3.3935 | 3.8249 | 5.1594 | **Q1** [None, 23.0] |
| `max_depth` | 5.5076 | 3.2262 | 3.6953 | 3.6907 | **Q2** [8.0, 9.0] |
| `min_data_in_leaf` | 3.3174 | 3.1631 | 3.2465 | 5.6075 | **Q2** [7.0, 9.0] |

#### E. Slice plot

![cpu_act@s43/moe](slice_cpu_act@s43_moe.png)


---

## cpu_act@s44  (search X=[6554, 21], holdout n=1638)


### naive-lightgbm

- **holdout RMSE: 2.27606** (winner retrained in 0.14s, cv score of winner: 2.4780)
- cv best RMSE: 2.4780, median: 2.5377, p10: 2.5058
- train: median 0.121s/fold, mean 0.124s, p90 0.166s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.463 |
| `learning_rate` | 0.303 |
| `extra_trees` | 0.136 |
| `feature_fraction` | 0.049 |
| `num_leaves` | 0.016 |
| `max_depth` | 0.014 |
| `bagging_fraction` | 0.012 |
| `bagging_freq` | 0.005 |
| `lambda_l1` | 0.001 |
| `lambda_l2` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 470 | 2.6744 | 0.5240 | 2.4780 |
| True | 30 | 3.6872 | 1.0227 | 2.8796 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.9760 | 2.6160 | 2.6132 | 2.7356 | **Q3** [0.0969, 0.1088] |
| `num_leaves` | 2.6476 | 2.6437 | 2.6443 | 2.9940 | **Q2** [18.0, 21.0] |
| `max_depth` | 2.9005 | — | — | 2.6891 | **Q4** [7.0, ∞) |
| `min_data_in_leaf` | 2.5548 | 2.5718 | 2.6144 | 3.1387 | **Q1** [None, 5.75] |
| `lambda_l1` | 2.7812 | 2.8047 | 2.5968 | 2.7581 | **Q3** [0.0071, 0.017] |
| `lambda_l2` | 2.6496 | 2.7250 | 2.6849 | 2.8814 | **Q1** [None, 0.0] |
| `feature_fraction` | 3.0691 | 2.6737 | 2.6105 | 2.5875 | **Q4** [0.9908, ∞) |
| `bagging_fraction` | 2.8852 | 2.7112 | 2.6835 | 2.6610 | **Q4** [0.969, ∞) |

#### E. Slice plot

![cpu_act@s44/naive-lightgbm](slice_cpu_act@s44_naive-lightgbm.png)


### moe

- **holdout RMSE: 2.23260** (winner retrained in 0.74s, cv score of winner: 2.4308)
- cv best RMSE: 2.4308, median: 2.5295, p10: 2.4677
- train: median 0.601s/fold, mean 0.757s, p90 1.258s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.710 |
| `mixture_balance_factor` | 0.091 |
| `min_data_in_leaf` | 0.046 |
| `mixture_diversity_lambda` | 0.028 |
| `mixture_num_experts` | 0.020 |
| `mixture_r_smoothing` | 0.019 |
| `num_leaves` | 0.013 |
| `mixture_gate_type` | 0.013 |
| `mixture_e_step_mode` | 0.012 |
| `feature_fraction` | 0.010 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 445 | 3.4393 | 3.7775 | 2.4308 |
| gbdt | 28 | 4.9670 | 5.2716 | 2.4965 |
| none | 27 | 8.4730 | 10.7533 | 2.6000 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 470 | 3.5816 | 4.1648 | 2.4308 |
| expert_choice | 30 | 7.1659 | 8.9836 | 2.5578 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 445 | 3.5482 | 4.5879 | 2.4308 |
| em | 31 | 5.7873 | 4.5601 | 2.5557 |
| loss_only | 24 | 5.8327 | 5.3013 | 2.6351 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 16 | 3.1129 | 0.9213 | 2.5221 |
| kmeans_features | 280 | 3.5567 | 4.4359 | 2.4308 |
| gmm_features | 18 | 3.9028 | 1.9495 | 2.5061 |
| gmm | 152 | 3.9141 | 4.9797 | 2.4364 |
| tree_hierarchical | 17 | 5.1778 | 3.9221 | 2.5419 |
| random | 17 | 5.8489 | 8.3442 | 2.5216 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 359 | 3.5529 | 4.3469 | 2.4308 |
| markov | 118 | 3.9606 | 4.4813 | 2.4364 |
| none | 23 | 6.7609 | 8.2249 | 2.4725 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 467 | 3.5447 | 4.1956 | 2.4308 |
| False | 33 | 7.3624 | 8.2861 | 2.6066 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 469 | 3.6272 | 4.5084 | 2.4308 |
| True | 31 | 6.3601 | 6.1854 | 3.3873 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | — | — | 3.7967 | **Q4** [2.0, ∞) |
| `mixture_diversity_lambda` | 3.9010 | 2.9620 | 3.8429 | 4.4808 | **Q2** [0.2864, 0.3452] |
| `mixture_warmup_iters` | 3.6454 | 3.5146 | 3.9352 | 4.0965 | **Q2** [30.0, 33.0] |
| `mixture_balance_factor` | 5.4695 | — | 3.2360 | 3.8461 | **Q3** [7.0, 8.0] |
| `learning_rate` | 5.9971 | 2.9222 | 2.9665 | 3.3009 | **Q2** [0.1208, 0.149] |
| `num_leaves` | 3.9645 | 3.2807 | 3.5504 | 4.4020 | **Q2** [52.0, 61.5] |
| `max_depth` | 5.7290 | — | 3.4524 | 3.7940 | **Q3** [7.0, 8.0] |
| `min_data_in_leaf` | 3.0859 | 3.4483 | 3.4378 | 4.9892 | **Q1** [None, 7.0] |

#### E. Slice plot

![cpu_act@s44/moe](slice_cpu_act@s44_moe.png)


---

## Overall recommendations

(no categorical settings were universally significant — see per-dataset breakdown)
