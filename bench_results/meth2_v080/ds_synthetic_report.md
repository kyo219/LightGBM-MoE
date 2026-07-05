# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 300

- **Datasets**: ['synthetic'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `0cf3634c544c`, lib sha256 `5cec0a0bd5ab…`, package `/tmp/lgbm-moe-v080/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| synthetic | naive-lightgbm | **4.81200** ± 0.33985 | 5.55933 | 0.0% | 0.21 |
| synthetic | moe | **3.91713** ± 0.17942 | 4.44177 | 0.0% | 0.93 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| synthetic@s42 | naive-lightgbm | 5.3381 | 5.5618 | 0.137 | 196 |
| synthetic@s42 | moe | 4.7737 | 5.5156 | 0.245 | 610 |
| synthetic@s43 | naive-lightgbm | 5.5380 | 5.8312 | 0.099 | 148 |
| synthetic@s43 | moe | 4.2523 | 4.7785 | 0.637 | 1584 |
| synthetic@s44 | naive-lightgbm | 5.8019 | 6.1616 | 0.104 | 150 |
| synthetic@s44 | moe | 4.2994 | 4.9003 | 0.268 | 680 |



---

## synthetic@s42  (search X=[1600, 5], holdout n=400)


### naive-lightgbm

- **holdout RMSE: 4.34766** (winner retrained in 0.19s, cv score of winner: 5.3381)
- cv best RMSE: 5.3381, median: 5.5618, p10: 5.4179
- train: median 0.137s/fold, mean 0.126s, p90 0.169s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.570 |
| `learning_rate` | 0.318 |
| `feature_fraction` | 0.064 |
| `extra_trees` | 0.018 |
| `bagging_freq` | 0.011 |
| `bagging_fraction` | 0.010 |
| `max_depth` | 0.005 |
| `num_leaves` | 0.003 |
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
| `num_leaves` | 5.6674 | 5.7401 | 5.6392 | 6.3012 | **Q3** [33.0, 50.25] |
| `max_depth` | 6.5140 | 5.7722 | — | 5.7153 | **Q4** [9.0, ∞) |
| `min_data_in_leaf` | 5.5472 | 5.5529 | 5.6385 | 6.5512 | **Q1** [None, 6.0] |
| `lambda_l1` | 6.3199 | 5.7260 | 5.6546 | 5.6443 | **Q4** [2.3125, ∞) |
| `lambda_l2` | 5.7133 | 5.6388 | 5.9086 | 6.0841 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 6.3681 | 5.6851 | 5.6903 | 5.6013 | **Q4** [0.9868, ∞) |
| `bagging_fraction` | 5.7752 | 5.6650 | 5.8987 | 6.0059 | **Q2** [0.6994, 0.7301] |

#### E. Slice plot

![synthetic@s42/naive-lightgbm](slice_synthetic@s42_naive-lightgbm.png)


### moe

- **holdout RMSE: 3.88413** (winner retrained in 0.35s, cv score of winner: 4.7737)
- cv best RMSE: 4.7737, median: 5.5156, p10: 5.0740
- train: median 0.245s/fold, mean 0.398s, p90 0.489s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_gate_type` | 0.190 |
| `learning_rate` | 0.187 |
| `min_data_in_leaf` | 0.104 |
| `feature_fraction` | 0.088 |
| `num_leaves` | 0.086 |
| `mixture_hard_m_step` | 0.074 |
| `max_depth` | 0.047 |
| `mixture_warmup_iters` | 0.035 |
| `mixture_diversity_lambda` | 0.034 |
| `bagging_freq` | 0.033 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **gbdt** | 5.4934 (n=202) | leaf_reuse | Δ +0.6374 | p=0.00e+00 |
| `mixture_routing_mode` | **token_choice** | 5.6582 (n=260) | expert_choice | Δ +1.2065 | p=0.00e+00 |
| `mixture_e_step_mode` | **loss_only** | 5.5455 (n=135) | gate_only | Δ +0.3284 | p=4.76e-03 |
| `mixture_init` | **tree_hierarchical** | 5.5444 (n=203) | uniform | Δ +0.4319 | p=1.21e-04 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 202 | 5.4934 | 0.7390 | 4.7737 |
| leaf_reuse | 68 | 6.1308 | 0.8089 | 5.4802 |
| none | 30 | 7.3053 | 1.8121 | 5.9024 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 260 | 5.6582 | 0.9597 | 4.7737 |
| expert_choice | 40 | 6.8647 | 1.2010 | 5.7083 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 135 | 5.5455 | 1.1367 | 4.7817 |
| gate_only | 122 | 5.8739 | 0.6614 | 5.3656 |
| em | 43 | 6.5223 | 1.4315 | 4.7737 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| tree_hierarchical | 203 | 5.5444 | 0.9617 | 4.7737 |
| uniform | 43 | 5.9763 | 0.5468 | 5.5791 |
| gmm | 20 | 6.5660 | 1.3702 | 5.5208 |
| random | 34 | 6.8206 | 1.1859 | 5.7083 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 156 | 5.5997 | 1.2307 | 4.7737 |
| ema | 104 | 5.8665 | 0.6522 | 5.3669 |
| markov | 40 | 6.5508 | 0.9740 | 5.5807 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 147 | 5.6300 | 1.2569 | 4.7737 |
| False | 153 | 6.0007 | 0.8289 | 5.2491 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 203 | 5.6941 | 0.7402 | 4.9784 |
| True | 97 | 6.0805 | 1.5285 | 4.7737 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | — | — | 5.8190 | **Q4** [2.0, ∞) |
| `mixture_diversity_lambda` | 6.3100 | 6.0757 | 5.4736 | 5.4169 | **Q4** [0.4642, ∞) |
| `mixture_warmup_iters` | 5.9982 | 6.0584 | 5.6153 | 5.6075 | **Q4** [43.0, ∞) |
| `mixture_balance_factor` | 6.5738 | 5.7364 | — | 5.7011 | **Q4** [7.0, ∞) |
| `learning_rate` | 6.1802 | 5.8306 | 5.6124 | 5.6529 | **Q3** [0.1356, 0.1978] |
| `num_leaves` | 6.3095 | 5.8174 | 5.8043 | 5.3762 | **Q4** [119.0, ∞) |
| `max_depth` | 6.1976 | — | 5.4463 | 6.0925 | **Q3** [8.0, 9.0] |
| `min_data_in_leaf` | 5.6602 | 5.5857 | 5.7656 | 6.1979 | **Q2** [10.0, 16.0] |

#### E. Slice plot

![synthetic@s42/moe](slice_synthetic@s42_moe.png)


---

## synthetic@s43  (search X=[1600, 5], holdout n=400)


### naive-lightgbm

- **holdout RMSE: 5.15155** (winner retrained in 0.24s, cv score of winner: 5.5380)
- cv best RMSE: 5.5380, median: 5.8312, p10: 5.6698
- train: median 0.099s/fold, mean 0.095s, p90 0.135s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.671 |
| `learning_rate` | 0.149 |
| `feature_fraction` | 0.102 |
| `bagging_fraction` | 0.037 |
| `bagging_freq` | 0.018 |
| `max_depth` | 0.008 |
| `lambda_l2` | 0.005 |
| `lambda_l1` | 0.005 |
| `extra_trees` | 0.003 |
| `num_leaves` | 0.002 |

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


### moe

- **holdout RMSE: 3.71575** (winner retrained in 2.14s, cv score of winner: 4.2523)
- cv best RMSE: 4.2523, median: 4.7785, p10: 4.3963
- train: median 0.637s/fold, mean 1.048s, p90 1.859s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_gate_type` | 0.525 |
| `mixture_init` | 0.068 |
| `learning_rate` | 0.066 |
| `feature_fraction` | 0.062 |
| `mixture_e_step_mode` | 0.048 |
| `mixture_diversity_lambda` | 0.041 |
| `bagging_fraction` | 0.035 |
| `mixture_hard_m_step` | 0.032 |
| `num_leaves` | 0.024 |
| `mixture_r_smoothing` | 0.024 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **gbdt** | 5.1310 (n=266) | leaf_reuse | Δ +2.4720 | p=0.00e+00 |
| `mixture_routing_mode` | **token_choice** | 5.3706 (n=264) | expert_choice | Δ +1.0233 | p=5.00e-05 |
| `mixture_e_step_mode` | **em** | 5.2194 (n=238) | loss_only | Δ +1.1151 | p=2.00e-06 |
| `mixture_init` | **tree_hierarchical** | 5.2260 (n=208) | gmm | Δ +0.7928 | p=8.30e-05 |
| `mixture_r_smoothing` | **none** | 5.2524 (n=200) | markov | Δ +0.6288 | p=8.11e-04 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 266 | 5.1310 | 1.0485 | 4.2523 |
| leaf_reuse | 18 | 7.6030 | 1.1073 | 5.9746 |
| none | 16 | 9.1449 | 1.9282 | 5.9136 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 264 | 5.3706 | 1.5242 | 4.2523 |
| expert_choice | 36 | 6.3939 | 1.2458 | 4.9919 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| em | 238 | 5.2194 | 1.4699 | 4.2523 |
| loss_only | 45 | 6.3345 | 1.2535 | 4.6909 |
| gate_only | 17 | 7.1038 | 1.1790 | 5.9746 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| tree_hierarchical | 208 | 5.2260 | 1.5817 | 4.2523 |
| gmm | 64 | 6.0188 | 1.2779 | 4.4616 |
| uniform | 14 | 6.2757 | 0.7266 | 5.0953 |
| random | 14 | 6.2825 | 1.2034 | 4.7805 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 200 | 5.2524 | 1.5932 | 4.2523 |
| markov | 83 | 5.8812 | 1.3225 | 4.4616 |
| ema | 17 | 6.4357 | 0.7792 | 5.6523 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 175 | 5.4841 | 1.2931 | 4.3517 |
| True | 125 | 5.5064 | 1.8105 | 4.2523 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 248 | 5.2876 | 1.5034 | 4.2523 |
| False | 52 | 6.4750 | 1.2497 | 4.9655 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | — | 5.2526 | 6.1909 | **Q3** [2.0, 3.0] |
| `mixture_diversity_lambda` | 5.6943 | 5.2881 | 5.2949 | 5.6964 | **Q2** [0.1189, 0.1818] |
| `mixture_warmup_iters` | 5.3903 | 5.2059 | 5.2202 | 6.1591 | **Q2** [8.0, 18.0] |
| `mixture_balance_factor` | — | — | 5.3048 | 5.9865 | **Q3** [2.0, 7.0] |
| `learning_rate` | 5.7240 | 5.1158 | 5.3578 | 5.7760 | **Q2** [0.0638, 0.0766] |
| `num_leaves` | 6.1580 | 5.0508 | 5.2212 | 5.5831 | **Q2** [61.0, 67.5] |
| `max_depth` | 5.3931 | 5.2488 | 5.1931 | 6.1402 | **Q3** [8.0, 9.0] |
| `min_data_in_leaf` | 5.3889 | 5.0408 | 5.3662 | 6.2829 | **Q2** [7.0, 11.0] |

#### E. Slice plot

![synthetic@s43/moe](slice_synthetic@s43_moe.png)


---

## synthetic@s44  (search X=[1600, 5], holdout n=400)


### naive-lightgbm

- **holdout RMSE: 4.93679** (winner retrained in 0.21s, cv score of winner: 5.8019)
- cv best RMSE: 5.8019, median: 6.1616, p10: 5.9403
- train: median 0.104s/fold, mean 0.097s, p90 0.150s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.876 |
| `learning_rate` | 0.073 |
| `feature_fraction` | 0.015 |
| `extra_trees` | 0.010 |
| `bagging_fraction` | 0.009 |
| `max_depth` | 0.007 |
| `lambda_l2` | 0.003 |
| `num_leaves` | 0.003 |
| `bagging_freq` | 0.003 |
| `lambda_l1` | 0.000 |

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


### moe

- **holdout RMSE: 4.15151** (winner retrained in 0.29s, cv score of winner: 4.2994)
- cv best RMSE: 4.2994, median: 4.9003, p10: 4.4679
- train: median 0.268s/fold, mean 0.447s, p90 0.504s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_gate_type` | 0.213 |
| `bagging_freq` | 0.156 |
| `mixture_diversity_lambda` | 0.118 |
| `feature_fraction` | 0.097 |
| `bagging_fraction` | 0.077 |
| `num_leaves` | 0.058 |
| `learning_rate` | 0.046 |
| `min_data_in_leaf` | 0.043 |
| `mixture_init` | 0.038 |
| `mixture_r_smoothing` | 0.033 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **gbdt** | 5.3180 (n=262) | leaf_reuse | Δ +1.7961 | p=0.00e+00 |
| `mixture_routing_mode` | **expert_choice** | 5.1367 (n=186) | token_choice | Δ +1.2994 | p=0.00e+00 |
| `mixture_e_step_mode` | **em** | 5.0897 (n=199) | loss_only | Δ +1.6041 | p=1.00e-06 |
| `mixture_init` | **gmm** | 5.0239 (n=183) | tree_hierarchical | Δ +1.4198 | p=0.00e+00 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 262 | 5.3180 | 1.1816 | 4.2994 |
| leaf_reuse | 22 | 7.1141 | 0.9927 | 6.2259 |
| none | 16 | 8.7069 | 1.7853 | 6.3379 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 186 | 5.1367 | 1.0762 | 4.2994 |
| token_choice | 114 | 6.4361 | 1.6992 | 4.3910 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| em | 199 | 5.0897 | 0.9727 | 4.2994 |
| loss_only | 33 | 6.6938 | 1.4543 | 4.3837 |
| gate_only | 68 | 6.6970 | 1.8509 | 4.8632 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gmm | 183 | 5.0239 | 1.0265 | 4.2994 |
| tree_hierarchical | 85 | 6.4437 | 1.7861 | 4.7319 |
| random | 18 | 6.8949 | 0.8795 | 6.1000 |
| uniform | 14 | 6.9955 | 0.7366 | 6.0316 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 117 | 5.2322 | 1.3448 | 4.2994 |
| none | 156 | 5.6482 | 1.3747 | 4.3910 |
| markov | 27 | 7.2541 | 1.5832 | 5.0885 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 207 | 5.1117 | 0.9957 | 4.2994 |
| True | 93 | 6.7852 | 1.7318 | 4.8632 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 213 | 5.1365 | 1.0840 | 4.2994 |
| False | 87 | 6.8398 | 1.6414 | 5.1454 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | 5.0482 | — | 5.9924 | **Q2** [2.0, 3.0] |
| `mixture_diversity_lambda` | 4.9775 | 4.9707 | 5.6058 | 6.9679 | **Q2** [0.0173, 0.0388] |
| `mixture_warmup_iters` | 6.9080 | 5.7789 | 5.0792 | 4.9526 | **Q4** [49.0, ∞) |
| `mixture_balance_factor` | 6.9899 | 5.5101 | 4.9258 | 5.1898 | **Q3** [8.5, 10.0] |
| `learning_rate` | 6.8058 | 5.4919 | 4.9571 | 5.2670 | **Q3** [0.1155, 0.1387] |
| `num_leaves` | 4.8938 | 5.0128 | 5.7023 | 6.8623 | **Q1** [None, 21.0] |
| `max_depth` | 5.1094 | 5.1518 | 5.2200 | 6.9531 | **Q1** [None, 4.0] |
| `min_data_in_leaf` | 4.9579 | 5.2095 | 5.3734 | 6.7676 | **Q1** [None, 7.0] |

#### E. Slice plot

![synthetic@s44/moe](slice_synthetic@s44_moe.png)


---

## Overall recommendations

**Categorical settings that are statistically significant winners (p<0.01):**

| dataset | param | best value | Δ vs runner-up | p |
|---|---|---|---|---|
| synthetic@s42 | `mixture_gate_type` | **gbdt** | +0.6374 | 0.00e+00 |
| synthetic@s42 | `mixture_routing_mode` | **token_choice** | +1.2065 | 0.00e+00 |
| synthetic@s42 | `mixture_e_step_mode` | **loss_only** | +0.3284 | 4.76e-03 |
| synthetic@s42 | `mixture_init` | **tree_hierarchical** | +0.4319 | 1.21e-04 |
| synthetic@s43 | `mixture_gate_type` | **gbdt** | +2.4720 | 0.00e+00 |
| synthetic@s43 | `mixture_routing_mode` | **token_choice** | +1.0233 | 5.00e-05 |
| synthetic@s43 | `mixture_e_step_mode` | **em** | +1.1151 | 2.00e-06 |
| synthetic@s43 | `mixture_init` | **tree_hierarchical** | +0.7928 | 8.30e-05 |
| synthetic@s43 | `mixture_r_smoothing` | **none** | +0.6288 | 8.11e-04 |
| synthetic@s44 | `mixture_gate_type` | **gbdt** | +1.7961 | 0.00e+00 |
| synthetic@s44 | `mixture_routing_mode` | **expert_choice** | +1.2994 | 0.00e+00 |
| synthetic@s44 | `mixture_e_step_mode` | **em** | +1.6041 | 1.00e-06 |
| synthetic@s44 | `mixture_init` | **gmm** | +1.4198 | 0.00e+00 |
