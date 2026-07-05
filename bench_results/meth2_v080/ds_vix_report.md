# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 300

- **Datasets**: ['vix'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `d25c06cf3b86`, lib sha256 `5cec0a0bd5ab…`, package `/tmp/lgbm-moe-v080/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| vix | naive-lightgbm | **1.79451** ± 0.06340 | 2.64148 | 0.0% | 0.06 |
| vix | moe | **1.92614** ± 0.15307 | 2.50119 | 0.0% | 1.13 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| vix@s42 | naive-lightgbm | 2.6702 | 2.7291 | 0.069 | 110 |
| vix@s42 | moe | 2.4733 | 2.6340 | 0.623 | 1038 |
| vix@s43 | naive-lightgbm | 2.5971 | 2.7157 | 0.038 | 72 |
| vix@s43 | moe | 2.5539 | 2.6418 | 0.208 | 587 |
| vix@s44 | naive-lightgbm | 2.6572 | 2.7332 | 0.043 | 75 |
| vix@s44 | moe | 2.4764 | 2.5709 | 0.196 | 617 |



---

## vix@s42  (search X=[3011, 13], holdout n=752)


### naive-lightgbm

- **holdout RMSE: 1.76644** (winner retrained in 0.07s, cv score of winner: 2.6702)
- cv best RMSE: 2.6702, median: 2.7291, p10: 2.6885
- train: median 0.069s/fold, mean 0.070s, p90 0.095s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.764 |
| `min_data_in_leaf` | 0.115 |
| `extra_trees` | 0.044 |
| `bagging_fraction` | 0.043 |
| `max_depth` | 0.011 |
| `feature_fraction` | 0.011 |
| `num_leaves` | 0.006 |
| `bagging_freq` | 0.005 |
| `lambda_l2` | 0.001 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 278 | 2.7555 | 0.1312 | 2.6702 |
| True | 22 | 3.2067 | 0.5093 | 2.7210 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.8875 | 2.7324 | 2.7454 | 2.7890 | **Q2** [0.039, 0.0468] |
| `num_leaves` | 2.7864 | 2.8612 | 2.7577 | 2.7486 | **Q4** [124.0, ∞) |
| `max_depth` | 2.8009 | — | 2.7644 | 2.8274 | **Q3** [5.0, 7.0] |
| `min_data_in_leaf` | 2.8082 | 2.7191 | 2.7264 | 2.9041 | **Q2** [24.0, 28.0] |
| `lambda_l1` | 2.7204 | 2.7450 | 2.8155 | 2.8734 | **Q1** [None, 0.0] |
| `lambda_l2` | 2.8117 | 2.8035 | 2.7477 | 2.7914 | **Q3** [0.0724, 0.9155] |
| `feature_fraction` | 2.8579 | 2.7613 | 2.7567 | 2.7784 | **Q3** [0.7364, 0.7762] |
| `bagging_fraction` | 2.8140 | 2.7434 | 2.7894 | 2.8075 | **Q2** [0.6679, 0.7205] |

#### E. Slice plot

![vix@s42/naive-lightgbm](slice_vix@s42_naive-lightgbm.png)


### moe

- **holdout RMSE: 1.94873** (winner retrained in 1.17s, cv score of winner: 2.4733)
- cv best RMSE: 2.4733, median: 2.6340, p10: 2.5143
- train: median 0.623s/fold, mean 0.683s, p90 0.842s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.238 |
| `mixture_hard_m_step` | 0.211 |
| `lambda_l2` | 0.199 |
| `mixture_warmup_iters` | 0.070 |
| `min_data_in_leaf` | 0.036 |
| `mixture_init` | 0.034 |
| `bagging_fraction` | 0.034 |
| `bagging_freq` | 0.034 |
| `num_leaves` | 0.031 |
| `extra_trees` | 0.023 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 263 | 2.7514 | 0.5868 | 2.4733 |
| leaf_reuse | 19 | 3.1461 | 0.6717 | 2.6347 |
| none | 18 | 3.2263 | 0.4783 | 2.6754 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 265 | 2.7565 | 0.5098 | 2.4733 |
| expert_choice | 35 | 3.1717 | 1.0031 | 2.6737 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 215 | 2.7292 | 0.4715 | 2.4733 |
| em | 70 | 2.9600 | 0.8288 | 2.5273 |
| loss_only | 15 | 3.1656 | 0.7600 | 2.6404 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| tree_hierarchical | 225 | 2.7200 | 0.4497 | 2.4733 |
| gmm | 33 | 2.9760 | 0.5467 | 2.5273 |
| random | 16 | 3.0022 | 0.3318 | 2.6734 |
| uniform | 26 | 3.2011 | 1.3155 | 2.6347 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 234 | 2.7362 | 0.5084 | 2.4733 |
| markov | 36 | 2.9846 | 0.5970 | 2.5862 |
| none | 30 | 3.1254 | 1.0125 | 2.6020 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 280 | 2.7454 | 0.5318 | 2.4733 |
| True | 20 | 3.6378 | 0.8759 | 2.8363 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 279 | 2.7560 | 0.5478 | 2.4733 |
| True | 21 | 3.4540 | 0.8779 | 2.8968 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 3.0487 | — | — | 2.7375 | **Q4** [4.0, ∞) |
| `mixture_diversity_lambda` | 2.7223 | 2.7573 | 2.7202 | 3.0198 | **Q3** [0.0588, 0.1068] |
| `mixture_warmup_iters` | 2.9073 | 2.8253 | 2.6566 | 2.8082 | **Q3** [22.0, 25.0] |
| `mixture_balance_factor` | 2.8867 | 2.7182 | 3.1081 | 2.8148 | **Q2** [4.0, 7.0] |
| `learning_rate` | 2.9660 | 2.6767 | 2.6801 | 2.8968 | **Q2** [0.0349, 0.0416] |
| `num_leaves` | 2.8594 | 2.7486 | 2.7053 | 2.9124 | **Q3** [36.0, 51.25] |
| `max_depth` | 2.9972 | — | 2.6485 | 2.9602 | **Q3** [8.0, 9.0] |
| `min_data_in_leaf` | 2.8699 | 2.7468 | 2.7820 | 2.8247 | **Q2** [35.0, 44.0] |

#### E. Slice plot

![vix@s42/moe](slice_vix@s42_moe.png)


---

## vix@s43  (search X=[3011, 13], holdout n=752)


### naive-lightgbm

- **holdout RMSE: 1.88229** (winner retrained in 0.05s, cv score of winner: 2.5971)
- cv best RMSE: 2.5971, median: 2.7157, p10: 2.6491
- train: median 0.038s/fold, mean 0.045s, p90 0.067s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.598 |
| `min_data_in_leaf` | 0.363 |
| `feature_fraction` | 0.009 |
| `bagging_freq` | 0.008 |
| `num_leaves` | 0.007 |
| `max_depth` | 0.006 |
| `bagging_fraction` | 0.004 |
| `lambda_l2` | 0.002 |
| `lambda_l1` | 0.002 |
| `extra_trees` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 259 | 2.7476 | 0.1920 | 2.5971 |
| False | 41 | 2.8651 | 0.2764 | 2.7016 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.8913 | 2.7394 | 2.7050 | 2.7191 | **Q3** [0.1647, 0.2004] |
| `num_leaves` | 2.8080 | 2.7553 | 2.7171 | 2.7735 | **Q3** [88.0, 97.0] |
| `max_depth` | — | — | 2.7211 | 2.8585 | **Q3** [3.0, 4.0] |
| `min_data_in_leaf` | 2.7595 | 2.7150 | 2.7161 | 2.8668 | **Q2** [12.0, 16.0] |
| `lambda_l1` | 2.8272 | 2.7260 | 2.7374 | 2.7642 | **Q2** [0.0, 0.0003] |
| `lambda_l2` | 2.7599 | 2.7436 | 2.7289 | 2.8225 | **Q3** [0.0, 0.0002] |
| `feature_fraction` | 2.7704 | 2.7048 | 2.7763 | 2.8033 | **Q2** [0.6959, 0.7244] |
| `bagging_fraction` | 2.7301 | 2.7381 | 2.7229 | 2.8637 | **Q3** [0.6232, 0.6579] |

#### E. Slice plot

![vix@s43/naive-lightgbm](slice_vix@s43_naive-lightgbm.png)


### moe

- **holdout RMSE: 1.72840** (winner retrained in 0.28s, cv score of winner: 2.5539)
- cv best RMSE: 2.5539, median: 2.6418, p10: 2.5864
- train: median 0.208s/fold, mean 0.385s, p90 0.506s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `num_leaves` | 0.379 |
| `learning_rate` | 0.233 |
| `mixture_hard_m_step` | 0.129 |
| `mixture_diversity_lambda` | 0.077 |
| `max_depth` | 0.057 |
| `mixture_balance_factor` | 0.024 |
| `mixture_warmup_iters` | 0.022 |
| `min_data_in_leaf` | 0.022 |
| `feature_fraction` | 0.014 |
| `extra_trees` | 0.011 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 256 | 2.8613 | 0.9423 | 2.5539 |
| gbdt | 26 | 2.9954 | 0.9048 | 2.6127 |
| leaf_reuse | 18 | 3.0606 | 0.5741 | 2.6037 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 279 | 2.8581 | 0.9222 | 2.5539 |
| token_choice | 21 | 3.2410 | 0.8560 | 2.6758 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 190 | 2.7078 | 0.3720 | 2.5539 |
| loss_only | 47 | 3.0882 | 1.0097 | 2.6127 |
| em | 63 | 3.2673 | 1.6144 | 2.5864 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 141 | 2.7895 | 0.7303 | 2.5539 |
| tree_hierarchical | 71 | 2.8820 | 0.7706 | 2.5583 |
| random | 21 | 2.9795 | 0.8121 | 2.5717 |
| gmm | 67 | 3.0590 | 1.3428 | 2.5864 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 132 | 2.8060 | 0.4215 | 2.5583 |
| none | 115 | 2.8536 | 0.8523 | 2.5539 |
| markov | 53 | 3.1492 | 1.6478 | 2.5730 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 276 | 2.8018 | 0.8406 | 2.5539 |
| True | 24 | 3.8402 | 1.2365 | 2.8073 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 280 | 2.8627 | 0.9445 | 2.5539 |
| True | 20 | 3.1958 | 0.4273 | 2.8540 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 3.5484 | — | — | 2.8596 | **Q4** [3.0, ∞) |
| `mixture_diversity_lambda` | 2.8214 | 2.6499 | 2.7191 | 3.3492 | **Q2** [0.165, 0.1983] |
| `mixture_warmup_iters` | 3.0207 | 2.7764 | 2.7868 | 2.9775 | **Q2** [11.0, 16.0] |
| `mixture_balance_factor` | 2.9763 | — | 2.7883 | 3.0671 | **Q3** [4.0, 7.0] |
| `learning_rate` | 3.1494 | 2.8086 | 2.6777 | 2.9039 | **Q3** [0.0572, 0.0676] |
| `num_leaves` | 2.7506 | 2.6884 | 2.8862 | 3.2076 | **Q2** [23.0, 30.0] |
| `max_depth` | 2.8504 | 2.8250 | — | 2.9165 | **Q2** [6.0, 8.0] |
| `min_data_in_leaf` | 2.8904 | 2.7282 | 2.9542 | 2.9555 | **Q2** [39.0, 44.0] |

#### E. Slice plot

![vix@s43/moe](slice_vix@s43_moe.png)


---

## vix@s44  (search X=[3011, 13], holdout n=752)


### naive-lightgbm

- **holdout RMSE: 1.73480** (winner retrained in 0.06s, cv score of winner: 2.6572)
- cv best RMSE: 2.6572, median: 2.7332, p10: 2.6860
- train: median 0.043s/fold, mean 0.047s, p90 0.070s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.652 |
| `learning_rate` | 0.261 |
| `max_depth` | 0.026 |
| `bagging_freq` | 0.017 |
| `num_leaves` | 0.014 |
| `bagging_fraction` | 0.011 |
| `extra_trees` | 0.010 |
| `feature_fraction` | 0.008 |
| `lambda_l2` | 0.001 |
| `lambda_l1` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 280 | 2.7684 | 0.1413 | 2.6572 |
| True | 20 | 3.0293 | 0.3142 | 2.7179 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.8491 | 2.7598 | 2.7368 | 2.7976 | **Q3** [0.064, 0.0859] |
| `num_leaves` | 2.7942 | 2.7757 | 2.7561 | 2.8183 | **Q3** [66.0, 74.0] |
| `max_depth` | 2.7874 | 2.8004 | 2.7434 | 2.8354 | **Q3** [5.0, 5.25] |
| `min_data_in_leaf` | 2.7875 | 2.7352 | 2.7481 | 2.8533 | **Q2** [18.75, 25.0] |
| `lambda_l1` | 2.7688 | 2.8312 | 2.7583 | 2.7851 | **Q3** [0.0061, 0.0448] |
| `lambda_l2` | 2.7600 | 2.7385 | 2.7683 | 2.8766 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 2.8228 | 2.7617 | 2.7847 | 2.7742 | **Q2** [0.7432, 0.7862] |
| `bagging_fraction` | 2.8646 | 2.7674 | 2.7518 | 2.7594 | **Q3** [0.8516, 0.8933] |

#### E. Slice plot

![vix@s44/naive-lightgbm](slice_vix@s44_naive-lightgbm.png)


### moe

- **holdout RMSE: 2.10129** (winner retrained in 1.94s, cv score of winner: 2.4764)
- cv best RMSE: 2.4764, median: 2.5709, p10: 2.5095
- train: median 0.196s/fold, mean 0.404s, p90 1.052s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.306 |
| `lambda_l2` | 0.212 |
| `mixture_init` | 0.121 |
| `lambda_l1` | 0.053 |
| `bagging_freq` | 0.048 |
| `mixture_r_smoothing` | 0.039 |
| `mixture_warmup_iters` | 0.028 |
| `feature_fraction` | 0.028 |
| `num_leaves` | 0.027 |
| `min_data_in_leaf` | 0.025 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 116 | 2.7948 | 0.7756 | 2.4764 |
| none | 156 | 2.9808 | 1.5066 | 2.4910 |
| leaf_reuse | 28 | 2.9821 | 0.3557 | 2.5693 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 269 | 2.8774 | 1.1930 | 2.4764 |
| token_choice | 31 | 3.1833 | 1.1973 | 2.5352 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| em | 186 | 2.7720 | 0.9052 | 2.4910 |
| gate_only | 78 | 3.0676 | 1.6068 | 2.4764 |
| loss_only | 36 | 3.2732 | 1.3526 | 2.5156 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 244 | 2.7307 | 0.9073 | 2.4764 |
| tree_hierarchical | 14 | 3.2093 | 0.6778 | 2.7208 |
| random | 28 | 3.2839 | 1.2193 | 2.5316 |
| gmm | 14 | 4.9662 | 2.7869 | 2.7395 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 112 | 2.7501 | 0.5338 | 2.4764 |
| none | 160 | 2.9911 | 1.5297 | 2.4910 |
| ema | 28 | 3.0754 | 0.8209 | 2.5147 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 264 | 2.8395 | 1.2017 | 2.4764 |
| True | 36 | 3.4189 | 1.0274 | 2.6220 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 279 | 2.8867 | 1.2248 | 2.4764 |
| True | 21 | 3.2049 | 0.6676 | 2.7368 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 2.9280 | — | — | 2.9077 | **Q4** [3.0, ∞) |
| `mixture_diversity_lambda` | 3.0911 | 2.8717 | 2.7907 | 2.8825 | **Q3** [0.1471, 0.1935] |
| `mixture_warmup_iters` | 3.2059 | 3.0074 | 2.7036 | 2.7348 | **Q3** [42.0, 47.0] |
| `mixture_balance_factor` | — | 2.7410 | 2.9937 | 3.0844 | **Q2** [2.0, 3.0] |
| `learning_rate` | 3.1472 | 2.6381 | 3.0879 | 2.7629 | **Q2** [0.0474, 0.0847] |
| `num_leaves` | 3.1835 | 2.8463 | 2.7443 | 2.8723 | **Q3** [90.0, 97.0] |
| `max_depth` | 2.8998 | 3.1751 | 2.9037 | 2.7082 | **Q4** [11.0, ∞) |
| `min_data_in_leaf` | 2.8441 | 2.8499 | 2.7422 | 3.1560 | **Q3** [10.0, 15.0] |

#### E. Slice plot

![vix@s44/moe](slice_vix@s44_moe.png)


---

## Overall recommendations

(no categorical settings were universally significant — see per-dataset breakdown)

