# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 500

- **Datasets**: ['hmm'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `747bb4a7b5e6`, lib sha256 `a3c30fcc7fd3…`, package `/home/mokumoku/Develop/LightGBM-MoE/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| hmm | naive-lightgbm | **2.21586** ± 0.24444 | 2.26814 | 0.0% | 0.06 |
| hmm | moe | **2.25508** ± 0.22943 | 2.24145 | 0.0% | 2.39 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| hmm@s42 | naive-lightgbm | 2.2992 | 2.3144 | 0.035 | 101 |
| hmm@s42 | moe | 2.2708 | 2.3161 | 0.243 | 764 |
| hmm@s43 | naive-lightgbm | 2.3781 | 2.3928 | 0.052 | 150 |
| hmm@s43 | moe | 2.3271 | 2.3816 | 0.360 | 4384 |
| hmm@s44 | naive-lightgbm | 2.1272 | 2.1405 | 0.044 | 134 |
| hmm@s44 | moe | 2.1265 | 2.1455 | 0.493 | 1429 |



---

## hmm@s42  (search X=[1600, 5], holdout n=400)


### naive-lightgbm

- **holdout RMSE: 1.88834** (winner retrained in 0.05s, cv score of winner: 2.2992)
- cv best RMSE: 2.2992, median: 2.3144, p10: 2.3078
- train: median 0.035s/fold, mean 0.036s, p90 0.042s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.442 |
| `learning_rate` | 0.305 |
| `min_data_in_leaf` | 0.122 |
| `max_depth` | 0.048 |
| `feature_fraction` | 0.028 |
| `bagging_fraction` | 0.023 |
| `num_leaves` | 0.021 |
| `lambda_l1` | 0.007 |
| `bagging_freq` | 0.003 |
| `lambda_l2` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 470 | 2.3166 | 0.0122 | 2.2992 |
| False | 30 | 2.3478 | 0.0201 | 2.3261 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.3224 | 2.3143 | 2.3155 | 2.3215 | **Q2** [0.0617, 0.0706] |
| `num_leaves` | 2.3190 | 2.3188 | 2.3181 | 2.3179 | **Q4** [120.0, ∞) |
| `max_depth` | — | — | 2.3160 | 2.3256 | **Q3** [3.0, 5.0] |
| `min_data_in_leaf` | 2.3202 | 2.3156 | 2.3152 | 2.3225 | **Q3** [30.0, 34.0] |
| `lambda_l1` | 2.3260 | 2.3163 | 2.3157 | 2.3159 | **Q3** [1.6462, 3.6754] |
| `lambda_l2` | 2.3202 | 2.3191 | 2.3168 | 2.3177 | **Q3** [0.0277, 0.2475] |
| `feature_fraction` | 2.3243 | 2.3177 | 2.3141 | 2.3177 | **Q3** [0.9027, 0.9273] |
| `bagging_fraction` | 2.3256 | 2.3181 | 2.3146 | 2.3154 | **Q3** [0.9369, 0.9666] |

#### E. Slice plot

![hmm@s42/naive-lightgbm](slice_hmm@s42_naive-lightgbm.png)


### moe

- **holdout RMSE: 1.95174** (winner retrained in 0.43s, cv score of winner: 2.2708)
- cv best RMSE: 2.2708, median: 2.3161, p10: 2.2847
- train: median 0.243s/fold, mean 0.292s, p90 0.384s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_gate_type` | 0.504 |
| `mixture_init` | 0.138 |
| `mixture_expert_dropout_rate` | 0.041 |
| `learning_rate` | 0.041 |
| `mixture_hard_m_step` | 0.031 |
| `mixture_diversity_lambda` | 0.029 |
| `mixture_load_balance_alpha` | 0.029 |
| `feature_fraction` | 0.027 |
| `min_data_in_leaf` | 0.023 |
| `bagging_fraction` | 0.021 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **gbdt** | 2.3184 (n=421) | leaf_reuse | Δ +0.0526 | p=0.00e+00 |
| `mixture_init` | **tree_hierarchical** | 2.3161 (n=384) | gmm_features | Δ +0.0384 | p=4.00e-05 |
| `mixture_r_smoothing` | **markov** | 2.3222 (n=375) | none | Δ +0.0180 | p=1.45e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 421 | 2.3184 | 0.0335 | 2.2708 |
| leaf_reuse | 39 | 2.3710 | 0.0332 | 2.3240 |
| none | 40 | 2.3930 | 0.0516 | 2.3315 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 328 | 2.3271 | 0.0449 | 2.2708 |
| token_choice | 172 | 2.3310 | 0.0373 | 2.2797 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 341 | 2.3227 | 0.0424 | 2.2708 |
| gate_only | 94 | 2.3298 | 0.0372 | 2.2797 |
| em | 65 | 2.3570 | 0.0376 | 2.2836 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| tree_hierarchical | 384 | 2.3161 | 0.0379 | 2.2708 |
| gmm_features | 19 | 2.3545 | 0.0303 | 2.3185 |
| uniform | 37 | 2.3641 | 0.0279 | 2.3315 |
| gmm | 21 | 2.3741 | 0.0289 | 2.3330 |
| random | 21 | 2.3758 | 0.0290 | 2.3411 |
| kmeans_features | 18 | 2.3835 | 0.0132 | 2.3546 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 375 | 2.3222 | 0.0415 | 2.2708 |
| none | 55 | 2.3402 | 0.0369 | 2.2910 |
| ema | 70 | 2.3529 | 0.0408 | 2.2789 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 436 | 2.3234 | 0.0404 | 2.2708 |
| True | 64 | 2.3630 | 0.0398 | 2.2783 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 464 | 2.3251 | 0.0414 | 2.2708 |
| False | 36 | 2.3721 | 0.0287 | 2.3311 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 2.3666 | — | — | 2.3264 | **Q4** [4.0, ∞) |
| `mixture_diversity_lambda` | 2.3290 | 2.3159 | 2.3270 | 2.3420 | **Q2** [0.115, 0.1415] |
| `mixture_warmup_iters` | 2.3528 | 2.3244 | 2.3192 | 2.3206 | **Q3** [48.0, 49.0] |
| `mixture_balance_factor` | — | — | 2.3212 | 2.3495 | **Q3** [2.0, 5.0] |
| `learning_rate` | 2.3318 | 2.3261 | 2.3254 | 2.3305 | **Q3** [0.1106, 0.1314] |
| `num_leaves` | 2.3448 | 2.3230 | 2.3222 | 2.3241 | **Q3** [118.0, 124.0] |
| `max_depth` | 2.3430 | — | 2.3246 | 2.3274 | **Q3** [6.0, 7.0] |
| `min_data_in_leaf` | 2.3383 | 2.3267 | 2.3190 | 2.3312 | **Q3** [41.0, 47.0] |

#### E. Slice plot

![hmm@s42/moe](slice_hmm@s42_moe.png)


---

## hmm@s43  (search X=[1600, 5], holdout n=400)


### naive-lightgbm

- **holdout RMSE: 2.47543** (winner retrained in 0.08s, cv score of winner: 2.3781)
- cv best RMSE: 2.3781, median: 2.3928, p10: 2.3835
- train: median 0.052s/fold, mean 0.055s, p90 0.074s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.504 |
| `extra_trees` | 0.350 |
| `learning_rate` | 0.032 |
| `lambda_l1` | 0.030 |
| `bagging_fraction` | 0.027 |
| `num_leaves` | 0.025 |
| `feature_fraction` | 0.018 |
| `max_depth` | 0.006 |
| `bagging_freq` | 0.005 |
| `lambda_l2` | 0.002 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 468 | 2.3959 | 0.0146 | 2.3781 |
| False | 32 | 2.4482 | 0.0290 | 2.4035 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.4027 | 2.3943 | 2.3958 | 2.4040 | **Q2** [0.0537, 0.0613] |
| `num_leaves` | 2.3947 | 2.3997 | 2.4003 | 2.4020 | **Q1** [None, 16.0] |
| `max_depth` | 2.4045 | 2.3962 | 2.3971 | 2.4015 | **Q2** [8.0, 8.5] |
| `min_data_in_leaf` | 2.3981 | 2.3936 | 2.3944 | 2.4122 | **Q2** [7.0, 11.0] |
| `lambda_l1` | 2.4082 | 2.3971 | 2.3950 | 2.3965 | **Q3** [2.6671, 4.7197] |
| `lambda_l2` | 2.4061 | 2.3972 | 2.3967 | 2.3968 | **Q3** [0.3291, 1.0941] |
| `feature_fraction` | 2.4091 | 2.3965 | 2.3965 | 2.3947 | **Q4** [0.9791, ∞) |
| `bagging_fraction` | 2.4064 | 2.3977 | 2.3948 | 2.3979 | **Q3** [0.8822, 0.9203] |

#### E. Slice plot

![hmm@s43/naive-lightgbm](slice_hmm@s43_naive-lightgbm.png)


### moe

- **holdout RMSE: 2.50647** (winner retrained in 6.30s, cv score of winner: 2.3271)
- cv best RMSE: 2.3271, median: 2.3816, p10: 2.3567
- train: median 0.360s/fold, mean 1.742s, p90 4.397s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_init` | 0.294 |
| `feature_fraction` | 0.197 |
| `learning_rate` | 0.078 |
| `min_data_in_leaf` | 0.072 |
| `mixture_balance_factor` | 0.067 |
| `mixture_warmup_iters` | 0.056 |
| `mixture_expert_dropout_rate` | 0.043 |
| `extra_trees` | 0.037 |
| `bagging_freq` | 0.024 |
| `mixture_diversity_lambda` | 0.024 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **gbdt** | 2.3811 (n=189) | none | Δ +0.0142 | p=1.80e-05 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 189 | 2.3811 | 0.0319 | 2.3271 |
| none | 288 | 2.3953 | 0.0393 | 2.3449 |
| leaf_reuse | 23 | 2.4369 | 0.0763 | 2.3716 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 358 | 2.3918 | 0.0342 | 2.3271 |
| expert_choice | 142 | 2.3919 | 0.0547 | 2.3318 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 188 | 2.3864 | 0.0455 | 2.3271 |
| gate_only | 29 | 2.3943 | 0.0382 | 2.3354 |
| em | 283 | 2.3952 | 0.0377 | 2.3487 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gmm_features | 277 | 2.3827 | 0.0360 | 2.3271 |
| uniform | 152 | 2.3880 | 0.0196 | 2.3607 |
| random | 18 | 2.4000 | 0.0224 | 2.3716 |
| kmeans_features | 17 | 2.4069 | 0.0349 | 2.3710 |
| gmm | 18 | 2.4635 | 0.0582 | 2.3950 |
| tree_hierarchical | 18 | 2.4715 | 0.0738 | 2.3676 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 92 | 2.3895 | 0.0473 | 2.3354 |
| markov | 372 | 2.3904 | 0.0370 | 2.3271 |
| ema | 36 | 2.4124 | 0.0561 | 2.3727 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 166 | 2.3897 | 0.0422 | 2.3318 |
| False | 334 | 2.3929 | 0.0405 | 2.3271 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 460 | 2.3879 | 0.0349 | 2.3271 |
| True | 40 | 2.4375 | 0.0694 | 2.3462 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 2.4339 | — | — | 2.3870 | **Q4** [6.0, ∞) |
| `mixture_diversity_lambda` | 2.3773 | 2.3946 | 2.3926 | 2.4028 | **Q1** [None, 0.1223] |
| `mixture_warmup_iters` | 2.3784 | 2.3980 | 2.3908 | 2.3981 | **Q1** [None, 13.0] |
| `mixture_balance_factor` | 2.4092 | 2.3822 | 2.3840 | 2.3971 | **Q2** [5.0, 6.0] |
| `learning_rate` | 2.4011 | 2.3933 | 2.3849 | 2.3880 | **Q3** [0.0661, 0.1146] |
| `num_leaves` | 2.3901 | 2.3816 | 2.3912 | 2.4041 | **Q2** [16.0, 25.0] |
| `max_depth` | 2.3942 | 2.4148 | 2.3814 | 2.3898 | **Q3** [9.0, 11.0] |
| `min_data_in_leaf` | 2.3938 | 2.3835 | 2.3818 | 2.4076 | **Q3** [40.0, 45.0] |

#### E. Slice plot

![hmm@s43/moe](slice_hmm@s43_moe.png)


---

## hmm@s44  (search X=[1600, 5], holdout n=400)


### naive-lightgbm

- **holdout RMSE: 2.28381** (winner retrained in 0.05s, cv score of winner: 2.1272)
- cv best RMSE: 2.1272, median: 2.1405, p10: 2.1334
- train: median 0.044s/fold, mean 0.049s, p90 0.062s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.570 |
| `max_depth` | 0.137 |
| `learning_rate` | 0.090 |
| `min_data_in_leaf` | 0.088 |
| `bagging_fraction` | 0.054 |
| `feature_fraction` | 0.041 |
| `num_leaves` | 0.018 |
| `bagging_freq` | 0.002 |
| `lambda_l2` | 0.000 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 468 | 2.1429 | 0.0124 | 2.1272 |
| False | 32 | 2.1808 | 0.0224 | 2.1570 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.1483 | 2.1404 | 2.1433 | 2.1493 | **Q2** [0.0373, 0.044] |
| `num_leaves` | 2.1502 | 2.1440 | 2.1426 | 2.1448 | **Q3** [105.0, 113.0] |
| `max_depth` | — | — | — | 2.1453 | **Q4** [3.0, ∞) |
| `min_data_in_leaf` | 2.1381 | 2.1425 | 2.1448 | 2.1551 | **Q1** [None, 6.0] |
| `lambda_l1` | 2.1476 | 2.1440 | 2.1446 | 2.1450 | **Q2** [0.0, 0.0] |
| `lambda_l2` | 2.1432 | 2.1439 | 2.1436 | 2.1506 | **Q1** [None, 0.0] |
| `feature_fraction` | 2.1478 | 2.1417 | 2.1470 | 2.1447 | **Q2** [0.7242, 0.753] |
| `bagging_fraction` | 2.1528 | 2.1417 | 2.1425 | 2.1442 | **Q2** [0.8804, 0.9196] |

#### E. Slice plot

![hmm@s44/naive-lightgbm](slice_hmm@s44_naive-lightgbm.png)


### moe

- **holdout RMSE: 2.30703** (winner retrained in 0.45s, cv score of winner: 2.1265)
- cv best RMSE: 2.1265, median: 2.1455, p10: 2.1365
- train: median 0.493s/fold, mean 0.556s, p90 0.606s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_init` | 0.383 |
| `learning_rate` | 0.197 |
| `bagging_fraction` | 0.079 |
| `mixture_expert_dropout_rate` | 0.049 |
| `num_leaves` | 0.048 |
| `mixture_load_balance_alpha` | 0.048 |
| `mixture_diversity_lambda` | 0.034 |
| `mixture_warmup_iters` | 0.034 |
| `bagging_freq` | 0.029 |
| `min_data_in_leaf` | 0.024 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_routing_mode` | **expert_choice** | 2.1579 (n=468) | token_choice | Δ +0.0264 | p=8.45e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 435 | 2.1565 | 0.0352 | 2.1265 |
| leaf_reuse | 40 | 2.1683 | 0.0432 | 2.1352 |
| none | 25 | 2.1999 | 0.0673 | 2.1428 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 468 | 2.1579 | 0.0378 | 2.1265 |
| token_choice | 32 | 2.1843 | 0.0517 | 2.1406 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 153 | 2.1559 | 0.0381 | 2.1265 |
| em | 288 | 2.1598 | 0.0375 | 2.1321 |
| loss_only | 59 | 2.1681 | 0.0486 | 2.1379 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 82 | 2.1494 | 0.0279 | 2.1265 |
| random | 349 | 2.1528 | 0.0280 | 2.1315 |
| kmeans_features | 18 | 2.1766 | 0.0403 | 2.1455 |
| gmm_features | 17 | 2.1883 | 0.0466 | 2.1455 |
| gmm | 17 | 2.2276 | 0.0749 | 2.1496 |
| tree_hierarchical | 17 | 2.2329 | 0.0600 | 2.1465 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 278 | 2.1572 | 0.0335 | 2.1321 |
| none | 194 | 2.1593 | 0.0425 | 2.1265 |
| ema | 28 | 2.1860 | 0.0567 | 2.1403 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 170 | 2.1556 | 0.0351 | 2.1265 |
| False | 330 | 2.1617 | 0.0412 | 2.1354 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 469 | 2.1568 | 0.0378 | 2.1265 |
| False | 31 | 2.2021 | 0.0367 | 2.1623 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 2.1848 | — | — | 2.1575 | **Q4** [5.0, ∞) |
| `mixture_diversity_lambda` | 2.1544 | 2.1588 | 2.1658 | 2.1594 | **Q1** [None, 0.0499] |
| `mixture_warmup_iters` | 2.1533 | 2.1623 | 2.1551 | 2.1646 | **Q1** [None, 11.0] |
| `mixture_balance_factor` | 2.1753 | 2.1562 | — | 2.1565 | **Q2** [6.0, 7.0] |
| `learning_rate` | 2.1760 | 2.1534 | 2.1581 | 2.1510 | **Q4** [0.2021, ∞) |
| `num_leaves` | 2.1589 | 2.1548 | 2.1522 | 2.1730 | **Q3** [50.0, 72.25] |
| `max_depth` | 2.1688 | 2.1512 | 2.1640 | 2.1589 | **Q2** [7.0, 8.0] |
| `min_data_in_leaf` | 2.1598 | 2.1575 | 2.1562 | 2.1641 | **Q3** [63.0, 68.0] |

#### E. Slice plot

![hmm@s44/moe](slice_hmm@s44_moe.png)


---

## Overall recommendations

**Categorical settings that are statistically significant winners (p<0.01):**

| dataset | param | best value | Δ vs runner-up | p |
|---|---|---|---|---|
| hmm@s42 | `mixture_gate_type` | **gbdt** | +0.0526 | 0.00e+00 |
| hmm@s42 | `mixture_init` | **tree_hierarchical** | +0.0384 | 4.00e-05 |
| hmm@s42 | `mixture_r_smoothing` | **markov** | +0.0180 | 1.45e-03 |
| hmm@s43 | `mixture_gate_type` | **gbdt** | +0.0142 | 1.80e-05 |
| hmm@s44 | `mixture_routing_mode` | **expert_choice** | +0.0264 | 8.45e-03 |
