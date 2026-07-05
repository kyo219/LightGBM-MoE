# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 300

- **Datasets**: ['hmm'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `0cf3634c544c`, lib sha256 `5cec0a0bd5ab…`, package `/tmp/lgbm-moe-v080/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| hmm | naive-lightgbm | **2.21802** ± 0.24377 | 2.27009 | 0.0% | 0.04 |
| hmm | moe | **2.23422** ± 0.25317 | 2.25054 | 0.0% | 0.19 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| hmm@s42 | naive-lightgbm | 2.2992 | 2.3152 | 0.033 | 57 |
| hmm@s42 | moe | 2.2931 | 2.3208 | 1.989 | 2541 |
| hmm@s43 | naive-lightgbm | 2.3800 | 2.3935 | 0.031 | 52 |
| hmm@s43 | moe | 2.3351 | 2.3692 | 0.076 | 249 |
| hmm@s44 | naive-lightgbm | 2.1311 | 2.1423 | 0.027 | 50 |
| hmm@s44 | moe | 2.1234 | 2.1395 | 0.125 | 196 |



---

## hmm@s42  (search X=[1600, 5], holdout n=400)


### naive-lightgbm

- **holdout RMSE: 1.88834** (winner retrained in 0.05s, cv score of winner: 2.2992)
- cv best RMSE: 2.2992, median: 2.3152, p10: 2.3078
- train: median 0.033s/fold, mean 0.034s, p90 0.041s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.401 |
| `learning_rate` | 0.293 |
| `min_data_in_leaf` | 0.112 |
| `bagging_fraction` | 0.089 |
| `num_leaves` | 0.028 |
| `max_depth` | 0.026 |
| `feature_fraction` | 0.024 |
| `bagging_freq` | 0.014 |
| `lambda_l2` | 0.010 |
| `lambda_l1` | 0.002 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 280 | 2.3177 | 0.0143 | 2.2992 |
| False | 20 | 2.3549 | 0.0208 | 2.3261 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.3258 | 2.3142 | 2.3149 | 2.3260 | **Q2** [0.0611, 0.0714] |
| `num_leaves` | 2.3210 | 2.3206 | 2.3198 | 2.3196 | **Q4** [120.0, ∞) |
| `max_depth` | — | — | 2.3163 | 2.3274 | **Q3** [3.0, 5.0] |
| `min_data_in_leaf` | 2.3232 | 2.3162 | 2.3158 | 2.3262 | **Q3** [31.0, 36.0] |
| `lambda_l1` | 2.3321 | 2.3174 | 2.3162 | 2.3152 | **Q4** [2.9866, ∞) |
| `lambda_l2` | 2.3254 | 2.3201 | 2.3165 | 2.3189 | **Q3** [0.0494, 0.2783] |
| `feature_fraction` | 2.3295 | 2.3164 | 2.3155 | 2.3195 | **Q3** [0.8977, 0.9296] |
| `bagging_fraction` | 2.3310 | 2.3194 | 2.3147 | 2.3158 | **Q3** [0.934, 0.9603] |

#### E. Slice plot

![hmm@s42/naive-lightgbm](slice_hmm@s42_naive-lightgbm.png)


### moe

- **holdout RMSE: 1.89585** (winner retrained in 0.31s, cv score of winner: 2.2931)
- cv best RMSE: 2.2931, median: 2.3208, p10: 2.3021
- train: median 1.989s/fold, mean 1.686s, p90 2.913s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_gate_type` | 0.249 |
| `learning_rate` | 0.194 |
| `num_leaves` | 0.153 |
| `min_data_in_leaf` | 0.106 |
| `mixture_refit_leaves` | 0.076 |
| `extra_trees` | 0.042 |
| `mixture_init` | 0.042 |
| `mixture_balance_factor` | 0.032 |
| `bagging_freq` | 0.020 |
| `mixture_num_experts` | 0.017 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **gbdt** | 2.3288 (n=231) | none | Δ +0.0324 | p=2.87e-04 |
| `mixture_routing_mode` | **expert_choice** | 2.3360 (n=277) | token_choice | Δ +0.0619 | p=4.81e-03 |
| `mixture_init` | **gmm** | 2.3333 (n=254) | uniform | Δ +0.0340 | p=5.36e-04 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 231 | 2.3288 | 0.0353 | 2.2931 |
| none | 51 | 2.3612 | 0.0571 | 2.3047 |
| leaf_reuse | 18 | 2.4350 | 0.0932 | 2.3765 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 277 | 2.3360 | 0.0445 | 2.2931 |
| token_choice | 23 | 2.3979 | 0.0922 | 2.3056 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 103 | 2.3321 | 0.0423 | 2.2957 |
| em | 135 | 2.3443 | 0.0573 | 2.2931 |
| loss_only | 62 | 2.3472 | 0.0546 | 2.2968 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gmm | 254 | 2.3333 | 0.0443 | 2.2931 |
| uniform | 14 | 2.3673 | 0.0270 | 2.3280 |
| random | 16 | 2.3704 | 0.0238 | 2.3304 |
| tree_hierarchical | 16 | 2.4060 | 0.1120 | 2.3188 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 263 | 2.3371 | 0.0517 | 2.2931 |
| markov | 17 | 2.3613 | 0.0475 | 2.3022 |
| none | 20 | 2.3713 | 0.0525 | 2.3067 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 279 | 2.3368 | 0.0500 | 2.2931 |
| True | 21 | 2.3923 | 0.0569 | 2.3068 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 279 | 2.3350 | 0.0486 | 2.2931 |
| False | 21 | 2.4173 | 0.0403 | 2.3534 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | — | — | 2.3407 | **Q4** [2.0, ∞) |
| `mixture_diversity_lambda` | 2.3405 | 2.3307 | 2.3446 | 2.3471 | **Q2** [0.2422, 0.2802] |
| `mixture_warmup_iters` | 2.3678 | 2.3338 | 2.3315 | 2.3334 | **Q3** [47.0, 49.0] |
| `mixture_balance_factor` | 2.3550 | — | 2.3300 | 2.3562 | **Q3** [3.0, 4.0] |
| `learning_rate` | 2.3696 | 2.3267 | 2.3270 | 2.3396 | **Q2** [0.1568, 0.1899] |
| `num_leaves` | 2.3649 | 2.3374 | 2.3276 | 2.3336 | **Q3** [116.0, 120.0] |
| `max_depth` | — | — | 2.3314 | 2.3553 | **Q3** [3.0, 4.0] |
| `min_data_in_leaf` | 2.3493 | 2.3320 | 2.3341 | 2.3477 | **Q2** [27.0, 33.0] |

#### E. Slice plot

![hmm@s42/moe](slice_hmm@s42_moe.png)


---

## hmm@s43  (search X=[1600, 5], holdout n=400)


### naive-lightgbm

- **holdout RMSE: 2.47018** (winner retrained in 0.05s, cv score of winner: 2.3800)
- cv best RMSE: 2.3800, median: 2.3935, p10: 2.3844
- train: median 0.031s/fold, mean 0.032s, p90 0.042s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.526 |
| `extra_trees` | 0.249 |
| `learning_rate` | 0.079 |
| `bagging_fraction` | 0.051 |
| `lambda_l1` | 0.038 |
| `feature_fraction` | 0.023 |
| `max_depth` | 0.015 |
| `num_leaves` | 0.009 |
| `lambda_l2` | 0.006 |
| `bagging_freq` | 0.002 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 279 | 2.3973 | 0.0165 | 2.3800 |
| False | 21 | 2.4486 | 0.0327 | 2.4035 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.4073 | 2.3941 | 2.3955 | 2.4069 | **Q2** [0.0531, 0.0605] |
| `num_leaves` | 2.3941 | 2.4000 | 2.4058 | 2.4035 | **Q1** [None, 15.75] |
| `max_depth` | 2.4084 | 2.3978 | 2.3967 | 2.4019 | **Q3** [9.0, 10.0] |
| `min_data_in_leaf` | 2.3942 | 2.3915 | 2.3982 | 2.4191 | **Q2** [7.0, 10.0] |
| `lambda_l1` | 2.4157 | 2.3956 | 2.3973 | 2.3950 | **Q4** [4.0999, ∞) |
| `lambda_l2` | 2.4147 | 2.3962 | 2.3976 | 2.3951 | **Q4** [1.2586, ∞) |
| `feature_fraction` | 2.4142 | 2.4004 | 2.3950 | 2.3941 | **Q4** [0.9768, ∞) |
| `bagging_fraction` | 2.4139 | 2.3994 | 2.3933 | 2.3972 | **Q3** [0.892, 0.921] |

#### E. Slice plot

![hmm@s43/naive-lightgbm](slice_hmm@s43_naive-lightgbm.png)


### moe

- **holdout RMSE: 2.50475** (winner retrained in 0.07s, cv score of winner: 2.3351)
- cv best RMSE: 2.3351, median: 2.3692, p10: 2.3499
- train: median 0.076s/fold, mean 0.160s, p90 0.269s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_init` | 0.247 |
| `mixture_gate_type` | 0.145 |
| `num_leaves` | 0.104 |
| `bagging_fraction` | 0.101 |
| `feature_fraction` | 0.092 |
| `bagging_freq` | 0.069 |
| `mixture_hard_m_step` | 0.061 |
| `max_depth` | 0.043 |
| `min_data_in_leaf` | 0.042 |
| `mixture_diversity_lambda` | 0.022 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_routing_mode` | **token_choice** | 2.3747 (n=248) | expert_choice | Δ +0.0400 | p=1.03e-04 |
| `mixture_init` | **uniform** | 2.3667 (n=185) | tree_hierarchical | Δ +0.0241 | p=1.39e-04 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 216 | 2.3752 | 0.0383 | 2.3351 |
| gbdt | 59 | 2.3841 | 0.0311 | 2.3375 |
| leaf_reuse | 25 | 2.4314 | 0.0551 | 2.3571 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 248 | 2.3747 | 0.0298 | 2.3351 |
| expert_choice | 52 | 2.4147 | 0.0668 | 2.3375 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 201 | 2.3769 | 0.0404 | 2.3351 |
| gate_only | 71 | 2.3861 | 0.0427 | 2.3519 |
| em | 28 | 2.4042 | 0.0396 | 2.3610 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 185 | 2.3667 | 0.0203 | 2.3351 |
| tree_hierarchical | 55 | 2.3908 | 0.0422 | 2.3480 |
| random | 42 | 2.3997 | 0.0378 | 2.3610 |
| gmm | 18 | 2.4654 | 0.0749 | 2.3654 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 178 | 2.3738 | 0.0369 | 2.3376 |
| markov | 45 | 2.3815 | 0.0437 | 2.3351 |
| ema | 77 | 2.3998 | 0.0450 | 2.3584 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 267 | 2.3749 | 0.0342 | 2.3351 |
| True | 33 | 2.4363 | 0.0544 | 2.3654 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 280 | 2.3784 | 0.0384 | 2.3351 |
| True | 20 | 2.4273 | 0.0567 | 2.3759 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | — | — | 2.3816 | **Q4** [2.0, ∞) |
| `mixture_diversity_lambda` | 2.3806 | 2.3744 | 2.3967 | 2.3748 | **Q2** [0.0464, 0.0939] |
| `mixture_warmup_iters` | 2.3863 | 2.3793 | 2.3706 | 2.3904 | **Q3** [35.0, 41.0] |
| `mixture_balance_factor` | — | 2.3838 | 2.3873 | 2.3729 | **Q4** [8.0, ∞) |
| `learning_rate` | 2.3933 | 2.3825 | 2.3635 | 2.3872 | **Q3** [0.0688, 0.0795] |
| `num_leaves` | 2.3754 | 2.3700 | 2.3685 | 2.4102 | **Q3** [26.0, 35.0] |
| `max_depth` | — | — | 2.3721 | 2.4074 | **Q3** [3.0, 5.0] |
| `min_data_in_leaf` | 2.3990 | 2.3666 | 2.3693 | 2.3954 | **Q2** [38.0, 42.0] |

#### E. Slice plot

![hmm@s43/moe](slice_hmm@s43_moe.png)


---

## hmm@s44  (search X=[1600, 5], holdout n=400)


### naive-lightgbm

- **holdout RMSE: 2.29553** (winner retrained in 0.03s, cv score of winner: 2.1311)
- cv best RMSE: 2.1311, median: 2.1423, p10: 2.1347
- train: median 0.027s/fold, mean 0.030s, p90 0.039s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.481 |
| `max_depth` | 0.155 |
| `bagging_fraction` | 0.123 |
| `learning_rate` | 0.090 |
| `min_data_in_leaf` | 0.083 |
| `feature_fraction` | 0.041 |
| `num_leaves` | 0.019 |
| `bagging_freq` | 0.005 |
| `lambda_l2` | 0.002 |
| `lambda_l1` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 278 | 2.1441 | 0.0129 | 2.1311 |
| False | 22 | 2.1867 | 0.0243 | 2.1570 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.1514 | 2.1411 | 2.1454 | 2.1511 | **Q2** [0.0406, 0.0468] |
| `num_leaves` | 2.1552 | 2.1441 | 2.1438 | 2.1462 | **Q3** [102.0, 112.0] |
| `max_depth` | — | — | 2.1430 | 2.1581 | **Q3** [3.0, 4.0] |
| `min_data_in_leaf` | 2.1393 | 2.1438 | 2.1472 | 2.1582 | **Q1** [None, 6.0] |
| `lambda_l1` | 2.1468 | 2.1476 | 2.1447 | 2.1498 | **Q3** [0.0, 0.0] |
| `lambda_l2` | 2.1443 | 2.1449 | 2.1449 | 2.1548 | **Q1** [None, 0.0] |
| `feature_fraction` | 2.1508 | 2.1424 | 2.1434 | 2.1522 | **Q2** [0.7138, 0.7421] |
| `bagging_fraction` | 2.1590 | 2.1422 | 2.1434 | 2.1443 | **Q2** [0.8818, 0.9284] |

#### E. Slice plot

![hmm@s44/naive-lightgbm](slice_hmm@s44_naive-lightgbm.png)


### moe

- **holdout RMSE: 2.30206** (winner retrained in 0.18s, cv score of winner: 2.1234)
- cv best RMSE: 2.1234, median: 2.1395, p10: 2.1295
- train: median 0.125s/fold, mean 0.124s, p90 0.155s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_init` | 0.709 |
| `feature_fraction` | 0.088 |
| `learning_rate` | 0.061 |
| `mixture_gate_type` | 0.021 |
| `mixture_warmup_iters` | 0.021 |
| `min_data_in_leaf` | 0.017 |
| `mixture_num_experts` | 0.015 |
| `extra_trees` | 0.013 |
| `mixture_diversity_lambda` | 0.012 |
| `num_leaves` | 0.008 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_routing_mode` | **expert_choice** | 2.1591 (n=258) | token_choice | Δ +0.0262 | p=9.36e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 244 | 2.1575 | 0.0543 | 2.1234 |
| gbdt | 16 | 2.1736 | 0.0436 | 2.1316 |
| none | 40 | 2.1904 | 0.0611 | 2.1349 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 258 | 2.1591 | 0.0547 | 2.1234 |
| token_choice | 42 | 2.1853 | 0.0582 | 2.1349 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 250 | 2.1586 | 0.0536 | 2.1234 |
| em | 27 | 2.1775 | 0.0487 | 2.1348 |
| loss_only | 23 | 2.1908 | 0.0744 | 2.1335 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 253 | 2.1474 | 0.0291 | 2.1234 |
| random | 18 | 2.1668 | 0.0279 | 2.1440 |
| tree_hierarchical | 15 | 2.2893 | 0.0679 | 2.1589 |
| gmm | 14 | 2.2986 | 0.0767 | 2.1679 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 205 | 2.1603 | 0.0556 | 2.1234 |
| markov | 69 | 2.1651 | 0.0584 | 2.1297 |
| ema | 26 | 2.1760 | 0.0492 | 2.1303 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 262 | 2.1600 | 0.0544 | 2.1234 |
| False | 38 | 2.1820 | 0.0621 | 2.1382 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 279 | 2.1589 | 0.0537 | 2.1234 |
| False | 21 | 2.2146 | 0.0591 | 2.1570 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | — | — | 2.1628 | **Q4** [2.0, ∞) |
| `mixture_diversity_lambda` | 2.1668 | 2.1581 | 2.1608 | 2.1653 | **Q2** [0.201, 0.2876] |
| `mixture_warmup_iters` | 2.1738 | 2.1631 | 2.1596 | 2.1561 | **Q4** [48.0, ∞) |
| `mixture_balance_factor` | 2.1896 | 2.1671 | — | 2.1554 | **Q4** [8.0, ∞) |
| `learning_rate` | 2.1913 | 2.1463 | 2.1476 | 2.1658 | **Q2** [0.1338, 0.1582] |
| `num_leaves` | 2.1604 | 2.1579 | 2.1514 | 2.1812 | **Q3** [36.0, 50.0] |
| `max_depth` | 2.1745 | 2.1558 | 2.1910 | 2.1584 | **Q2** [7.0, 11.0] |
| `min_data_in_leaf` | 2.1776 | 2.1494 | 2.1571 | 2.1668 | **Q2** [31.75, 39.0] |

#### E. Slice plot

![hmm@s44/moe](slice_hmm@s44_moe.png)


---

## Overall recommendations

**Categorical settings that are statistically significant winners (p<0.01):**

| dataset | param | best value | Δ vs runner-up | p |
|---|---|---|---|---|
| hmm@s42 | `mixture_gate_type` | **gbdt** | +0.0324 | 2.87e-04 |
| hmm@s42 | `mixture_routing_mode` | **expert_choice** | +0.0619 | 4.81e-03 |
| hmm@s42 | `mixture_init` | **gmm** | +0.0340 | 5.36e-04 |
| hmm@s43 | `mixture_routing_mode` | **token_choice** | +0.0400 | 1.03e-04 |
| hmm@s43 | `mixture_init` | **uniform** | +0.0241 | 1.39e-04 |
| hmm@s44 | `mixture_routing_mode` | **expert_choice** | +0.0262 | 9.36e-03 |
