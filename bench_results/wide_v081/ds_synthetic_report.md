# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 500

- **Datasets**: ['synthetic'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `747bb4a7b5e6`, lib sha256 `a3c30fcc7fd3…`, package `/home/mokumoku/Develop/LightGBM-MoE/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| synthetic | naive-lightgbm | **4.83980** ± 0.31113 | 5.48121 | 0.0% | 0.23 |
| synthetic | moe | **3.79177** ± 0.24946 | 4.27509 | 0.0% | 1.76 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| synthetic@s42 | naive-lightgbm | 5.3096 | 5.5469 | 0.150 | 377 |
| synthetic@s42 | moe | 5.1618 | 5.4889 | 2.923 | 6756 |
| synthetic@s43 | naive-lightgbm | 5.5192 | 5.8184 | 0.099 | 249 |
| synthetic@s43 | moe | 4.1840 | 4.7100 | 0.388 | 1814 |
| synthetic@s44 | naive-lightgbm | 5.6149 | 6.0510 | 0.123 | 307 |
| synthetic@s44 | moe | 3.4794 | 4.5422 | 0.228 | 711 |



---

## synthetic@s42  (search X=[1600, 5], holdout n=400)


### naive-lightgbm

- **holdout RMSE: 4.40130** (winner retrained in 0.36s, cv score of winner: 5.3096)
- cv best RMSE: 5.3096, median: 5.5469, p10: 5.4101
- train: median 0.150s/fold, mean 0.146s, p90 0.209s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.671 |
| `learning_rate` | 0.260 |
| `feature_fraction` | 0.032 |
| `bagging_fraction` | 0.022 |
| `num_leaves` | 0.008 |
| `extra_trees` | 0.003 |
| `max_depth` | 0.001 |
| `bagging_freq` | 0.001 |
| `lambda_l1` | 0.001 |
| `lambda_l2` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 5.7104 | 0.4933 | 5.3096 |
| True | 32 | 6.8388 | 1.3862 | 5.6091 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 6.0027 | 5.6405 | 5.6945 | 5.7926 | **Q2** [0.0461, 0.0568] |
| `num_leaves` | 5.6610 | 5.6656 | 5.9632 | 5.8424 | **Q1** [None, 30.0] |
| `max_depth` | 6.1194 | 5.6517 | — | 5.7335 | **Q2** [9.0, 10.0] |
| `min_data_in_leaf` | 5.5496 | 5.5381 | 5.6075 | 6.4120 | **Q2** [7.0, 8.0] |
| `lambda_l1` | 5.9018 | 5.8799 | 5.6872 | 5.6616 | **Q4** [0.9322, ∞) |
| `lambda_l2` | 5.6751 | 5.8891 | 5.8415 | 5.7247 | **Q1** [None, 0.0] |
| `feature_fraction` | 6.1383 | 5.7015 | 5.6382 | 5.6524 | **Q3** [0.9547, 0.9788] |
| `bagging_fraction` | 5.7691 | 5.7066 | 5.6783 | 5.9765 | **Q3** [0.6881, 0.7753] |

#### E. Slice plot

![synthetic@s42/naive-lightgbm](slice_synthetic@s42_naive-lightgbm.png)


### moe

- **holdout RMSE: 4.11088** (winner retrained in 4.59s, cv score of winner: 5.1618)
- cv best RMSE: 5.1618, median: 5.4889, p10: 5.2769
- train: median 2.923s/fold, mean 2.685s, p90 4.444s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.606 |
| `mixture_gate_type` | 0.107 |
| `min_data_in_leaf` | 0.081 |
| `num_leaves` | 0.049 |
| `feature_fraction` | 0.039 |
| `mixture_init` | 0.021 |
| `mixture_expert_dropout_rate` | 0.019 |
| `mixture_warmup_iters` | 0.013 |
| `mixture_diversity_lambda` | 0.007 |
| `extra_trees` | 0.007 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **leaf_reuse** | 5.6612 (n=396) | gbdt | Δ +0.5435 | p=0.00e+00 |
| `mixture_routing_mode` | **token_choice** | 5.7144 (n=420) | expert_choice | Δ +0.7103 | p=0.00e+00 |
| `mixture_e_step_mode` | **gate_only** | 5.7048 (n=432) | loss_only | Δ +0.8237 | p=3.78e-04 |
| `mixture_r_smoothing` | **none** | 5.6960 (n=389) | markov | Δ +0.5051 | p=4.81e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 396 | 5.6612 | 0.7024 | 5.1618 |
| gbdt | 61 | 6.2047 | 0.6100 | 5.4905 |
| none | 43 | 6.8299 | 1.0100 | 5.8592 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 420 | 5.7144 | 0.7522 | 5.1618 |
| expert_choice | 80 | 6.4247 | 0.8154 | 5.5558 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 432 | 5.7048 | 0.6749 | 5.1618 |
| loss_only | 27 | 6.5285 | 1.0233 | 5.5642 |
| em | 41 | 6.6656 | 1.1169 | 5.5815 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gmm | 325 | 5.7071 | 0.8208 | 5.1618 |
| tree_hierarchical | 49 | 5.8407 | 0.5617 | 5.3535 |
| kmeans_features | 38 | 5.9068 | 0.7556 | 5.3458 |
| gmm_features | 17 | 6.0124 | 0.6026 | 5.4191 |
| random | 44 | 6.2409 | 0.8644 | 5.2556 |
| uniform | 27 | 6.3615 | 0.5714 | 5.4839 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 389 | 5.6960 | 0.7104 | 5.1618 |
| markov | 47 | 6.2011 | 1.1351 | 5.2259 |
| ema | 64 | 6.3566 | 0.7568 | 5.3113 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 5.7960 | 0.7793 | 5.1618 |
| True | 32 | 6.2967 | 1.0158 | 5.3865 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 467 | 5.7636 | 0.6904 | 5.1618 |
| True | 33 | 6.7402 | 1.4850 | 5.3635 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 6.5055 | 5.9987 | — | 5.6677 | **Q4** [6.0, ∞) |
| `mixture_diversity_lambda` | 6.0308 | 5.7026 | 5.8039 | 5.7748 | **Q2** [0.3259, 0.365] |
| `mixture_warmup_iters` | 5.8027 | 5.6146 | 5.7452 | 6.1165 | **Q2** [9.0, 13.0] |
| `mixture_balance_factor` | 5.8329 | — | — | 5.8270 | **Q4** [4.0, ∞) |
| `learning_rate` | 6.1627 | 5.6217 | 5.7668 | 5.7610 | **Q2** [0.1049, 0.142] |
| `num_leaves` | 6.2358 | 5.6597 | 5.7198 | 5.6958 | **Q2** [96.75, 108.0] |
| `max_depth` | 6.3327 | 5.7969 | 5.7470 | 5.6335 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 5.6772 | 5.6763 | 5.5703 | 6.2672 | **Q3** [8.0, 11.0] |

#### E. Slice plot

![synthetic@s42/moe](slice_synthetic@s42_moe.png)


---

## synthetic@s43  (search X=[1600, 5], holdout n=400)


### naive-lightgbm

- **holdout RMSE: 5.09057** (winner retrained in 0.18s, cv score of winner: 5.5192)
- cv best RMSE: 5.5192, median: 5.8184, p10: 5.6691
- train: median 0.099s/fold, mean 0.095s, p90 0.137s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.771 |
| `learning_rate` | 0.136 |
| `bagging_fraction` | 0.042 |
| `feature_fraction` | 0.027 |
| `bagging_freq` | 0.007 |
| `max_depth` | 0.005 |
| `lambda_l1` | 0.005 |
| `lambda_l2` | 0.004 |
| `extra_trees` | 0.002 |
| `num_leaves` | 0.002 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 446 | 5.9559 | 0.4653 | 5.5192 |
| False | 54 | 6.3077 | 0.6867 | 5.7045 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 6.2452 | 5.8588 | 5.9002 | 5.9714 | **Q2** [0.104, 0.12] |
| `num_leaves` | 6.1871 | 5.9433 | 5.9227 | 5.9358 | **Q3** [118.0, 123.0] |
| `max_depth` | 6.3585 | 5.9511 | — | 5.9551 | **Q2** [8.0, 9.0] |
| `min_data_in_leaf` | — | 5.7701 | 5.8555 | 6.4802 | **Q2** [5.0, 7.0] |
| `lambda_l1` | 6.0938 | 5.9732 | 5.9434 | 5.9653 | **Q3** [0.0017, 0.0067] |
| `lambda_l2` | 6.0009 | 5.9181 | 5.8718 | 6.1849 | **Q3** [0.0, 0.0] |
| `feature_fraction` | 6.2526 | 5.9369 | 5.8896 | 5.8966 | **Q3** [0.9539, 0.9731] |
| `bagging_fraction` | 6.0995 | 5.9145 | 5.9622 | 5.9995 | **Q2** [0.8768, 0.9024] |

#### E. Slice plot

![synthetic@s43/naive-lightgbm](slice_synthetic@s43_naive-lightgbm.png)


### moe

- **holdout RMSE: 3.76251** (winner retrained in 0.37s, cv score of winner: 4.1840)
- cv best RMSE: 4.1840, median: 4.7100, p10: 4.3729
- train: median 0.388s/fold, mean 0.713s, p90 2.584s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_gate_type` | 0.767 |
| `mixture_init` | 0.062 |
| `min_data_in_leaf` | 0.033 |
| `learning_rate` | 0.026 |
| `mixture_balance_factor` | 0.021 |
| `feature_fraction` | 0.021 |
| `mixture_warmup_iters` | 0.012 |
| `bagging_fraction` | 0.011 |
| `mixture_hard_m_step` | 0.009 |
| `num_leaves` | 0.006 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **gbdt** | 5.0671 (n=449) | leaf_reuse | Δ +2.9488 | p=0.00e+00 |
| `mixture_routing_mode` | **token_choice** | 5.2059 (n=348) | expert_choice | Δ +0.8183 | p=0.00e+00 |
| `mixture_init` | **gmm** | 5.1270 (n=298) | tree_hierarchical | Δ +0.4784 | p=3.32e-03 |
| `mixture_r_smoothing` | **none** | 5.2139 (n=304) | markov | Δ +0.4392 | p=1.59e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 449 | 5.0671 | 0.8647 | 4.1840 |
| leaf_reuse | 27 | 8.0159 | 1.2111 | 5.8063 |
| none | 24 | 9.8238 | 1.4500 | 6.3409 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 348 | 5.2059 | 1.3990 | 4.1840 |
| expert_choice | 152 | 6.0242 | 1.5738 | 4.4635 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 266 | 5.3008 | 1.4940 | 4.2535 |
| em | 170 | 5.3511 | 1.5034 | 4.1840 |
| gate_only | 64 | 6.3690 | 1.1788 | 5.0206 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gmm | 298 | 5.1270 | 1.4588 | 4.1840 |
| tree_hierarchical | 124 | 5.6054 | 1.5213 | 4.4607 |
| uniform | 18 | 6.2991 | 1.0230 | 5.5203 |
| kmeans_features | 16 | 6.3576 | 1.0954 | 5.2180 |
| random | 27 | 6.5719 | 0.9512 | 5.5542 |
| gmm_features | 17 | 6.5813 | 1.3023 | 5.2055 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 304 | 5.2139 | 1.4631 | 4.1840 |
| markov | 169 | 5.6531 | 1.4191 | 4.4453 |
| ema | 27 | 6.9227 | 1.4267 | 4.8742 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 296 | 5.3824 | 1.4486 | 4.1840 |
| True | 204 | 5.5595 | 1.5711 | 4.3306 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 453 | 5.3403 | 1.4776 | 4.1840 |
| False | 47 | 6.5569 | 1.2753 | 4.9661 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | — | — | 5.4547 | **Q4** [2.0, ∞) |
| `mixture_diversity_lambda` | 6.1246 | 5.3212 | 5.0842 | 5.2886 | **Q3** [0.34, 0.4317] |
| `mixture_warmup_iters` | 5.8503 | 5.4095 | 5.1507 | 5.3832 | **Q3** [38.0, 43.0] |
| `mixture_balance_factor` | 6.3315 | 5.4097 | — | 5.3515 | **Q4** [5.0, ∞) |
| `learning_rate` | 6.2409 | 5.1666 | 5.0114 | 5.3997 | **Q3** [0.1493, 0.1846] |
| `num_leaves` | 5.2485 | 5.3322 | 5.4161 | 5.8151 | **Q1** [None, 31.0] |
| `max_depth` | 5.7956 | 5.1868 | 5.2300 | 5.6351 | **Q2** [8.0, 9.0] |
| `min_data_in_leaf` | 5.2213 | 5.0892 | 5.3073 | 6.2278 | **Q2** [7.0, 11.0] |

#### E. Slice plot

![synthetic@s43/moe](slice_synthetic@s43_moe.png)


---

## synthetic@s44  (search X=[1600, 5], holdout n=400)


### naive-lightgbm

- **holdout RMSE: 5.02751** (winner retrained in 0.15s, cv score of winner: 5.6149)
- cv best RMSE: 5.6149, median: 6.0510, p10: 5.7619
- train: median 0.123s/fold, mean 0.119s, p90 0.179s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.811 |
| `feature_fraction` | 0.052 |
| `extra_trees` | 0.042 |
| `bagging_fraction` | 0.036 |
| `learning_rate` | 0.027 |
| `max_depth` | 0.010 |
| `num_leaves` | 0.009 |
| `lambda_l2` | 0.006 |
| `bagging_freq` | 0.003 |
| `lambda_l1` | 0.003 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 470 | 6.1813 | 0.5132 | 5.6149 |
| True | 30 | 7.0463 | 0.8441 | 6.0670 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 6.3728 | 6.0274 | 6.1293 | 6.4032 | **Q2** [0.0365, 0.0421] |
| `num_leaves` | 6.2477 | 6.2705 | 6.1595 | 6.2592 | **Q3** [84.0, 109.0] |
| `max_depth` | 6.3497 | 6.0993 | 6.2599 | 6.2055 | **Q2** [9.0, 10.0] |
| `min_data_in_leaf` | 5.9990 | 6.0231 | 6.1054 | 6.7147 | **Q1** [None, 6.0] |
| `lambda_l1` | 6.6018 | 6.0186 | 6.1360 | 6.1764 | **Q2** [0.0588, 0.3447] |
| `lambda_l2` | 6.2222 | 6.5009 | 6.0797 | 6.1300 | **Q3** [0.0021, 0.2991] |
| `feature_fraction` | 6.6824 | 6.0070 | 6.1069 | 6.1366 | **Q2** [0.9027, 0.9382] |
| `bagging_fraction` | 6.3217 | 6.3795 | 6.2683 | 5.9634 | **Q4** [0.9424, ∞) |

#### E. Slice plot

![synthetic@s44/naive-lightgbm](slice_synthetic@s44_naive-lightgbm.png)


### moe

- **holdout RMSE: 3.50192** (winner retrained in 0.31s, cv score of winner: 3.4794)
- cv best RMSE: 3.4794, median: 4.5422, p10: 3.8170
- train: median 0.228s/fold, mean 0.272s, p90 0.364s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_gate_type` | 0.727 |
| `mixture_init` | 0.092 |
| `min_data_in_leaf` | 0.059 |
| `learning_rate` | 0.033 |
| `mixture_hard_m_step` | 0.022 |
| `bagging_freq` | 0.013 |
| `mixture_load_balance_alpha` | 0.009 |
| `mixture_expert_dropout_rate` | 0.007 |
| `mixture_refit_leaves` | 0.005 |
| `mixture_num_experts` | 0.005 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **gbdt** | 4.7813 (n=446) | leaf_reuse | Δ +3.2045 | p=0.00e+00 |
| `mixture_routing_mode` | **expert_choice** | 5.1057 (n=440) | token_choice | Δ +1.3804 | p=1.00e-06 |
| `mixture_init` | **gmm** | 4.9002 (n=382) | tree_hierarchical | Δ +1.0038 | p=5.62e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 446 | 4.7813 | 1.1260 | 3.4794 |
| leaf_reuse | 25 | 7.9858 | 1.0044 | 6.2720 |
| none | 29 | 10.4687 | 1.9232 | 6.3885 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 440 | 5.1057 | 1.8254 | 3.4794 |
| token_choice | 60 | 6.4861 | 1.8522 | 3.7297 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 395 | 5.1302 | 1.8390 | 3.4794 |
| em | 75 | 5.7473 | 1.9437 | 3.6513 |
| loss_only | 30 | 5.9406 | 1.9588 | 3.8284 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gmm | 382 | 4.9002 | 1.8680 | 3.4794 |
| tree_hierarchical | 33 | 5.9040 | 1.8558 | 3.8861 |
| uniform | 18 | 6.2323 | 0.6778 | 5.3635 |
| gmm_features | 26 | 6.6015 | 0.8693 | 5.0048 |
| random | 19 | 6.9308 | 1.2982 | 5.9999 |
| kmeans_features | 22 | 6.9760 | 0.9869 | 5.8698 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 275 | 5.0895 | 1.8767 | 3.4794 |
| none | 198 | 5.2939 | 1.7886 | 3.7299 |
| ema | 27 | 6.9582 | 1.7699 | 4.4371 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 453 | 5.1166 | 1.8324 | 3.4794 |
| False | 47 | 6.7630 | 1.7023 | 3.8285 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 465 | 5.1733 | 1.8546 | 3.4794 |
| False | 35 | 6.5750 | 1.7663 | 4.6399 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | — | — | 5.2714 | **Q4** [2.0, ∞) |
| `mixture_diversity_lambda` | 5.6791 | 5.1389 | 5.0730 | 5.1946 | **Q3** [0.418, 0.4508] |
| `mixture_warmup_iters` | 5.3580 | 4.9184 | 5.2650 | 5.5947 | **Q2** [18.0, 21.0] |
| `mixture_balance_factor` | — | — | 5.2464 | 5.3389 | **Q3** [2.0, 8.0] |
| `learning_rate` | 6.0869 | 5.0747 | 4.9892 | 4.9348 | **Q4** [0.2714, ∞) |
| `num_leaves` | 5.6035 | 5.1656 | 4.9372 | 5.4045 | **Q3** [72.0, 99.25] |
| `max_depth` | — | — | 5.1988 | 5.4845 | **Q3** [3.0, 8.0] |
| `min_data_in_leaf` | 4.8853 | 4.9747 | 4.9725 | 6.1816 | **Q1** [None, 7.0] |

#### E. Slice plot

![synthetic@s44/moe](slice_synthetic@s44_moe.png)


---

## Overall recommendations

**Categorical settings that are statistically significant winners (p<0.01):**

| dataset | param | best value | Δ vs runner-up | p |
|---|---|---|---|---|
| synthetic@s42 | `mixture_gate_type` | **leaf_reuse** | +0.5435 | 0.00e+00 |
| synthetic@s42 | `mixture_routing_mode` | **token_choice** | +0.7103 | 0.00e+00 |
| synthetic@s42 | `mixture_e_step_mode` | **gate_only** | +0.8237 | 3.78e-04 |
| synthetic@s42 | `mixture_r_smoothing` | **none** | +0.5051 | 4.81e-03 |
| synthetic@s43 | `mixture_gate_type` | **gbdt** | +2.9488 | 0.00e+00 |
| synthetic@s43 | `mixture_routing_mode` | **token_choice** | +0.8183 | 0.00e+00 |
| synthetic@s43 | `mixture_init` | **gmm** | +0.4784 | 3.32e-03 |
| synthetic@s43 | `mixture_r_smoothing` | **none** | +0.4392 | 1.59e-03 |
| synthetic@s44 | `mixture_gate_type` | **gbdt** | +3.2045 | 0.00e+00 |
| synthetic@s44 | `mixture_routing_mode` | **expert_choice** | +1.3804 | 1.00e-06 |
| synthetic@s44 | `mixture_init` | **gmm** | +1.0038 | 5.62e-03 |
