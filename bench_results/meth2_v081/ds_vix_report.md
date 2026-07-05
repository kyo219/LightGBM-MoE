# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 300

- **Datasets**: ['vix'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `d25c06cf3b86`, lib sha256 `a3c30fcc7fd3…`, package `/home/mokumoku/Develop/LightGBM-MoE/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| vix | naive-lightgbm | **1.79451** ± 0.06340 | 2.64148 | 0.0% | 0.07 |
| vix | naive-ensemble | **1.75169** ± 0.04369 | 2.64838 | 0.0% | 0.19 |
| vix | moe | **1.85384** ± 0.17309 | 2.45968 | 0.0% | 0.40 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| vix@s42 | naive-lightgbm | 2.6702 | 2.7291 | 0.069 | 110 |
| vix@s42 | naive-ensemble | 2.6694 | 2.7246 | 0.072 | 156 |
| vix@s42 | moe | 2.4575 | 2.6208 | 0.231 | 486 |
| vix@s43 | naive-lightgbm | 2.5971 | 2.7157 | 0.035 | 70 |
| vix@s43 | naive-ensemble | 2.6426 | 2.7076 | 0.169 | 305 |
| vix@s43 | moe | 2.4029 | 2.5871 | 0.237 | 776 |
| vix@s44 | naive-lightgbm | 2.6572 | 2.7332 | 0.042 | 74 |
| vix@s44 | naive-ensemble | 2.6331 | 2.7123 | 0.206 | 361 |
| vix@s44 | moe | 2.5186 | 2.6282 | 0.388 | 654 |



---

## vix@s42  (search X=[3011, 13], holdout n=752)


### naive-lightgbm

- **holdout RMSE: 1.76644** (winner retrained in 0.08s, cv score of winner: 2.6702)
- cv best RMSE: 2.6702, median: 2.7291, p10: 2.6885
- train: median 0.069s/fold, mean 0.069s, p90 0.092s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.742 |
| `min_data_in_leaf` | 0.111 |
| `bagging_fraction` | 0.074 |
| `extra_trees` | 0.046 |
| `feature_fraction` | 0.009 |
| `bagging_freq` | 0.008 |
| `num_leaves` | 0.005 |
| `max_depth` | 0.004 |
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


### naive-ensemble

- **holdout RMSE: 1.73520** (winner retrained in 0.10s, cv score of winner: 2.6694)
- cv best RMSE: 2.6694, median: 2.7246, p10: 2.6883
- train: median 0.072s/fold, mean 0.100s, p90 0.184s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.557 |
| `min_data_in_leaf` | 0.341 |
| `bagging_fraction` | 0.034 |
| `feature_fraction` | 0.022 |
| `bagging_freq` | 0.018 |
| `extra_trees` | 0.016 |
| `num_leaves` | 0.006 |
| `max_depth` | 0.003 |
| `n_models` | 0.003 |
| `lambda_l2` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 280 | 2.7538 | 0.1593 | 2.6694 |
| True | 20 | 3.1217 | 0.4447 | 2.7360 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.9236 | 2.7169 | 2.7156 | 2.7571 | **Q3** [0.0958, 0.1072] |
| `num_leaves` | 2.8675 | 2.7371 | 2.7487 | 2.7604 | **Q2** [69.0, 86.0] |
| `max_depth` | — | — | 2.7340 | 2.8670 | **Q3** [3.0, 4.0] |
| `min_data_in_leaf` | 2.7496 | 2.7137 | 2.7324 | 2.9106 | **Q2** [24.0, 27.0] |
| `lambda_l1` | 2.7855 | 2.7187 | 2.8151 | 2.7939 | **Q2** [0.0, 0.0] |
| `lambda_l2` | 2.7695 | 2.7474 | 2.7828 | 2.8135 | **Q2** [0.0, 0.0001] |
| `feature_fraction` | 2.8079 | 2.7473 | 2.7531 | 2.8050 | **Q2** [0.6806, 0.753] |
| `bagging_fraction` | 2.8593 | 2.7812 | 2.7394 | 2.7334 | **Q4** [0.9751, ∞) |

#### E. Slice plot

![vix@s42/naive-ensemble](slice_vix@s42_naive-ensemble.png)


### moe

- **holdout RMSE: 2.01304** (winner retrained in 0.25s, cv score of winner: 2.4575)
- cv best RMSE: 2.4575, median: 2.6208, p10: 2.5246
- train: median 0.231s/fold, mean 0.316s, p90 0.434s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.419 |
| `bagging_fraction` | 0.346 |
| `mixture_num_experts` | 0.058 |
| `mixture_diversity_lambda` | 0.041 |
| `num_leaves` | 0.040 |
| `mixture_init` | 0.031 |
| `lambda_l2` | 0.016 |
| `feature_fraction` | 0.010 |
| `extra_trees` | 0.009 |
| `mixture_r_smoothing` | 0.008 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 221 | 2.7402 | 0.6849 | 2.4575 |
| leaf_reuse | 63 | 2.8416 | 0.4593 | 2.5581 |
| none | 16 | 3.9583 | 3.1082 | 2.6405 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 117 | 2.7606 | 0.6468 | 2.4575 |
| token_choice | 183 | 2.8686 | 1.1545 | 2.4708 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 251 | 2.7709 | 0.9509 | 2.4575 |
| em | 27 | 3.0857 | 1.1074 | 2.4802 |
| gate_only | 22 | 3.1427 | 1.1444 | 2.5192 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 253 | 2.7182 | 0.6058 | 2.4575 |
| gmm | 14 | 3.0269 | 0.6746 | 2.6045 |
| tree_hierarchical | 16 | 3.1767 | 0.6285 | 2.5599 |
| random | 17 | 3.9431 | 3.0820 | 2.5448 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 252 | 2.6810 | 0.4116 | 2.4575 |
| ema | 18 | 3.5669 | 1.5202 | 2.5696 |
| markov | 30 | 3.6039 | 2.4226 | 2.5643 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 279 | 2.7712 | 0.9263 | 2.4575 |
| True | 21 | 3.5610 | 1.4162 | 2.7173 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 279 | 2.7401 | 0.5807 | 2.4575 |
| True | 21 | 3.9737 | 2.8443 | 2.8299 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | — | — | 2.8265 | **Q4** [2.0, ∞) |
| `mixture_diversity_lambda` | 2.9217 | 2.7282 | 2.8943 | 2.7617 | **Q2** [0.3686, 0.3944] |
| `mixture_warmup_iters` | 2.7492 | 2.6492 | 3.0390 | 2.8881 | **Q2** [10.0, 14.0] |
| `mixture_balance_factor` | 3.2308 | 2.7611 | — | 2.7280 | **Q4** [8.0, ∞) |
| `learning_rate` | 3.2176 | 2.7487 | 2.6106 | 2.7290 | **Q3** [0.0747, 0.0863] |
| `num_leaves` | 2.9507 | 2.7333 | 2.6570 | 2.9734 | **Q3** [92.0, 101.0] |
| `max_depth` | 2.7307 | 2.6184 | 3.0726 | 2.9495 | **Q2** [4.0, 5.0] |
| `min_data_in_leaf` | 2.7110 | 2.6714 | 2.9147 | 2.9968 | **Q2** [9.0, 14.0] |

#### E. Slice plot

![vix@s42/moe](slice_vix@s42_moe.png)


---

## vix@s43  (search X=[3011, 13], holdout n=752)


### naive-lightgbm

- **holdout RMSE: 1.88229** (winner retrained in 0.05s, cv score of winner: 2.5971)
- cv best RMSE: 2.5971, median: 2.7157, p10: 2.6491
- train: median 0.035s/fold, mean 0.043s, p90 0.064s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.587 |
| `min_data_in_leaf` | 0.366 |
| `feature_fraction` | 0.014 |
| `bagging_freq` | 0.011 |
| `max_depth` | 0.006 |
| `bagging_fraction` | 0.005 |
| `num_leaves` | 0.005 |
| `extra_trees` | 0.004 |
| `lambda_l1` | 0.002 |
| `lambda_l2` | 0.000 |

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


### naive-ensemble

- **holdout RMSE: 1.70837** (winner retrained in 0.14s, cv score of winner: 2.6426)
- cv best RMSE: 2.6426, median: 2.7076, p10: 2.6683
- train: median 0.169s/fold, mean 0.200s, p90 0.345s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.476 |
| `learning_rate` | 0.451 |
| `feature_fraction` | 0.035 |
| `extra_trees` | 0.016 |
| `bagging_fraction` | 0.008 |
| `num_leaves` | 0.007 |
| `bagging_freq` | 0.004 |
| `max_depth` | 0.001 |
| `n_models` | 0.000 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 277 | 2.7332 | 0.1245 | 2.6426 |
| True | 23 | 3.0953 | 0.4163 | 2.7385 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.8567 | 2.7055 | 2.7182 | 2.7636 | **Q2** [0.0614, 0.0702] |
| `num_leaves` | 2.7965 | 2.7310 | 2.7236 | 2.7928 | **Q3** [52.5, 62.0] |
| `max_depth` | 2.7532 | — | 2.7309 | 2.8161 | **Q3** [5.0, 6.0] |
| `min_data_in_leaf` | 2.7660 | 2.6915 | 2.7024 | 2.8790 | **Q2** [18.0, 21.0] |
| `lambda_l1` | 2.7301 | 2.7230 | 2.7178 | 2.8730 | **Q3** [0.0, 0.0] |
| `lambda_l2` | 2.8236 | 2.7333 | 2.7388 | 2.7483 | **Q2** [0.001, 0.0285] |
| `feature_fraction` | 2.8331 | 2.7301 | 2.7250 | 2.7559 | **Q3** [0.8291, 0.8721] |
| `bagging_fraction` | 2.8113 | 2.7453 | 2.7042 | 2.7831 | **Q3** [0.7712, 0.7968] |

#### E. Slice plot

![vix@s43/naive-ensemble](slice_vix@s43_naive-ensemble.png)


### moe

- **holdout RMSE: 1.61321** (winner retrained in 0.27s, cv score of winner: 2.4029)
- cv best RMSE: 2.4029, median: 2.5871, p10: 2.4611
- train: median 0.237s/fold, mean 0.510s, p90 1.043s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.326 |
| `mixture_init` | 0.129 |
| `bagging_fraction` | 0.110 |
| `min_data_in_leaf` | 0.109 |
| `mixture_diversity_lambda` | 0.101 |
| `max_depth` | 0.051 |
| `mixture_balance_factor` | 0.027 |
| `lambda_l1` | 0.027 |
| `mixture_gate_type` | 0.026 |
| `bagging_freq` | 0.022 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 234 | 2.7080 | 0.6543 | 2.4029 |
| gbdt | 43 | 3.0336 | 1.3336 | 2.4866 |
| none | 23 | 3.4993 | 2.1521 | 2.4900 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 249 | 2.7458 | 0.9001 | 2.4029 |
| expert_choice | 51 | 3.1549 | 1.3272 | 2.4866 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 233 | 2.6729 | 0.5688 | 2.4029 |
| loss_only | 22 | 3.2524 | 1.2024 | 2.5050 |
| em | 45 | 3.3394 | 1.9448 | 2.4866 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| random | 253 | 2.7227 | 0.8249 | 2.4029 |
| uniform | 17 | 2.9253 | 0.5368 | 2.5272 |
| tree_hierarchical | 15 | 3.1823 | 0.9390 | 2.5637 |
| gmm | 15 | 3.8869 | 2.4049 | 2.6354 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 225 | 2.7115 | 0.8723 | 2.4029 |
| ema | 27 | 2.8634 | 0.4480 | 2.4913 |
| none | 48 | 3.2750 | 1.5097 | 2.4550 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 269 | 2.7637 | 0.9724 | 2.4029 |
| True | 31 | 3.2640 | 1.0978 | 2.5404 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 280 | 2.7842 | 0.9952 | 2.4029 |
| True | 20 | 3.2516 | 0.9291 | 2.7912 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 3.1514 | — | — | 2.7766 | **Q4** [4.0, ∞) |
| `mixture_diversity_lambda` | 2.6702 | 2.7124 | 2.6751 | 3.2038 | **Q1** [None, 0.1509] |
| `mixture_warmup_iters` | 2.7979 | 2.8240 | 2.5700 | 2.9791 | **Q3** [30.0, 33.0] |
| `mixture_balance_factor` | — | 2.7926 | 2.7485 | 2.8885 | **Q3** [3.0, 4.0] |
| `learning_rate` | 3.1630 | 2.6069 | 2.5905 | 2.9010 | **Q3** [0.1074, 0.128] |
| `num_leaves` | 2.6005 | 2.6549 | 2.7516 | 3.2283 | **Q1** [None, 11.0] |
| `max_depth` | 3.0660 | — | — | 2.7348 | **Q4** [9.0, ∞) |
| `min_data_in_leaf` | 2.6570 | 2.6554 | 2.8269 | 3.0834 | **Q2** [12.0, 16.0] |

#### E. Slice plot

![vix@s43/moe](slice_vix@s43_moe.png)


---

## vix@s44  (search X=[3011, 13], holdout n=752)


### naive-lightgbm

- **holdout RMSE: 1.73480** (winner retrained in 0.07s, cv score of winner: 2.6572)
- cv best RMSE: 2.6572, median: 2.7332, p10: 2.6860
- train: median 0.042s/fold, mean 0.047s, p90 0.065s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.687 |
| `learning_rate` | 0.239 |
| `max_depth` | 0.018 |
| `bagging_freq` | 0.012 |
| `bagging_fraction` | 0.011 |
| `num_leaves` | 0.011 |
| `extra_trees` | 0.009 |
| `lambda_l2` | 0.007 |
| `feature_fraction` | 0.005 |
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


### naive-ensemble

- **holdout RMSE: 1.81150** (winner retrained in 0.33s, cv score of winner: 2.6331)
- cv best RMSE: 2.6331, median: 2.7123, p10: 2.6698
- train: median 0.206s/fold, mean 0.237s, p90 0.372s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.841 |
| `min_data_in_leaf` | 0.122 |
| `num_leaves` | 0.014 |
| `bagging_fraction` | 0.006 |
| `max_depth` | 0.004 |
| `extra_trees` | 0.004 |
| `feature_fraction` | 0.003 |
| `bagging_freq` | 0.003 |
| `lambda_l1` | 0.002 |
| `n_models` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 279 | 2.7486 | 0.1730 | 2.6331 |
| False | 21 | 3.0170 | 0.3273 | 2.7407 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.9011 | 2.7086 | 2.7175 | 2.7424 | **Q2** [0.1508, 0.1777] |
| `num_leaves` | 2.8192 | 2.7442 | 2.7413 | 2.7680 | **Q3** [67.0, 76.0] |
| `max_depth` | 2.8581 | 2.7532 | — | 2.7383 | **Q4** [11.0, ∞) |
| `min_data_in_leaf` | 2.7620 | 2.7153 | 2.7455 | 2.8439 | **Q2** [9.0, 13.0] |
| `lambda_l1` | 2.8518 | 2.7571 | 2.7160 | 2.7449 | **Q3** [2.0781, 4.0546] |
| `lambda_l2` | 2.7236 | 2.7407 | 2.7551 | 2.8503 | **Q1** [None, 0.0] |
| `feature_fraction` | 2.8711 | 2.7343 | 2.7093 | 2.7550 | **Q3** [0.96, 0.9797] |
| `bagging_fraction` | 2.8710 | 2.7310 | 2.7503 | 2.7174 | **Q4** [0.9764, ∞) |

#### E. Slice plot

![vix@s44/naive-ensemble](slice_vix@s44_naive-ensemble.png)


### moe

- **holdout RMSE: 1.93526** (winner retrained in 0.68s, cv score of winner: 2.5186)
- cv best RMSE: 2.5186, median: 2.6282, p10: 2.5371
- train: median 0.388s/fold, mean 0.426s, p90 0.494s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_gate_type` | 0.342 |
| `bagging_freq` | 0.160 |
| `learning_rate` | 0.155 |
| `feature_fraction` | 0.099 |
| `mixture_e_step_mode` | 0.063 |
| `min_data_in_leaf` | 0.051 |
| `mixture_warmup_iters` | 0.036 |
| `num_leaves` | 0.023 |
| `mixture_init` | 0.015 |
| `max_depth` | 0.014 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **leaf_reuse** | 2.7669 (n=264) | gbdt | Δ +0.5501 | p=7.45e-04 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 264 | 2.7669 | 0.4650 | 2.5186 |
| gbdt | 19 | 3.3170 | 0.5725 | 2.6026 |
| none | 17 | 4.5876 | 1.5506 | 3.0500 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 274 | 2.8811 | 0.7164 | 2.5186 |
| expert_choice | 26 | 3.1561 | 0.8361 | 2.6287 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 257 | 2.8192 | 0.5720 | 2.5186 |
| em | 27 | 3.0624 | 0.6272 | 2.6132 |
| loss_only | 16 | 4.0156 | 1.6507 | 2.9388 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| tree_hierarchical | 247 | 2.7841 | 0.4635 | 2.5186 |
| random | 24 | 3.0216 | 0.8987 | 2.6132 |
| uniform | 15 | 3.6802 | 1.1894 | 2.8304 |
| gmm | 14 | 4.0061 | 1.6254 | 2.7851 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 242 | 2.8395 | 0.5604 | 2.5195 |
| none | 33 | 3.0264 | 1.1362 | 2.5186 |
| ema | 25 | 3.3775 | 1.1813 | 2.5274 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 278 | 2.8656 | 0.6054 | 2.5186 |
| False | 22 | 3.4015 | 1.5497 | 2.5771 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 280 | 2.8827 | 0.7362 | 2.5186 |
| True | 20 | 3.2157 | 0.5805 | 2.6522 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 2.7937 | — | — | 2.9120 | **Q1** [None, 3.0] |
| `mixture_diversity_lambda` | 3.1626 | 2.8165 | 2.7584 | 2.8821 | **Q3** [0.2126, 0.2541] |
| `mixture_warmup_iters` | 2.9818 | 2.7749 | 2.8441 | 3.0293 | **Q2** [20.0, 22.0] |
| `mixture_balance_factor` | 2.8761 | 3.2873 | 2.8182 | 2.7786 | **Q4** [10.0, ∞) |
| `learning_rate` | 3.0810 | 2.8310 | 2.7171 | 2.9905 | **Q3** [0.089, 0.11] |
| `num_leaves` | 3.0294 | 2.7675 | 2.9939 | 2.8293 | **Q2** [61.0, 68.0] |
| `max_depth` | 3.1370 | — | 2.7727 | 3.0056 | **Q3** [8.0, 9.0] |
| `min_data_in_leaf` | 2.9616 | 2.7632 | 2.7724 | 3.0858 | **Q2** [33.0, 36.0] |

#### E. Slice plot

![vix@s44/moe](slice_vix@s44_moe.png)


---

## Overall recommendations

**Categorical settings that are statistically significant winners (p<0.01):**

| dataset | param | best value | Δ vs runner-up | p |
|---|---|---|---|---|
| vix@s44 | `mixture_gate_type` | **leaf_reuse** | +0.5501 | 7.45e-04 |
