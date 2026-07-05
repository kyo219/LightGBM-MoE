# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 500

- **Datasets**: ['vix'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `747bb4a7b5e6`, lib sha256 `a3c30fcc7fd3…`, package `/home/mokumoku/Develop/LightGBM-MoE/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| vix | naive-lightgbm | **1.76822** ± 0.09140 | 2.61915 | 0.0% | 0.07 |
| vix | moe | **1.65253** ± 0.03007 | 2.38125 | 0.0% | 1.98 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| vix@s42 | naive-lightgbm | 2.6235 | 2.7139 | 0.078 | 216 |
| vix@s42 | moe | 2.3922 | 2.6010 | 0.472 | 1431 |
| vix@s43 | naive-lightgbm | 2.5849 | 2.6917 | 0.053 | 168 |
| vix@s43 | moe | 2.3760 | 2.5739 | 0.376 | 7444 |
| vix@s44 | naive-lightgbm | 2.6491 | 2.7244 | 0.066 | 186 |
| vix@s44 | moe | 2.3756 | 2.5804 | 0.622 | 3192 |



---

## vix@s42  (search X=[3011, 13], holdout n=752)


### naive-lightgbm

- **holdout RMSE: 1.67713** (winner retrained in 0.05s, cv score of winner: 2.6235)
- cv best RMSE: 2.6235, median: 2.7139, p10: 2.6728
- train: median 0.078s/fold, mean 0.082s, p90 0.116s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.699 |
| `min_data_in_leaf` | 0.180 |
| `extra_trees` | 0.087 |
| `feature_fraction` | 0.015 |
| `max_depth` | 0.009 |
| `num_leaves` | 0.004 |
| `bagging_freq` | 0.003 |
| `bagging_fraction` | 0.003 |
| `lambda_l2` | 0.000 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 2.7351 | 0.1102 | 2.6235 |
| True | 32 | 3.1497 | 0.4330 | 2.7210 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.8230 | 2.7158 | 2.7369 | 2.7708 | **Q2** [0.0357, 0.0412] |
| `num_leaves` | 2.7549 | 2.7697 | 2.7813 | 2.7432 | **Q4** [122.0, ∞) |
| `max_depth` | — | 2.7358 | 2.7560 | 2.8102 | **Q2** [3.0, 5.0] |
| `min_data_in_leaf` | 2.7283 | 2.7534 | 2.7232 | 2.8354 | **Q3** [25.0, 30.0] |
| `lambda_l1` | 2.7318 | 2.7470 | 2.7421 | 2.8255 | **Q1** [None, 0.0] |
| `lambda_l2` | 2.7783 | 2.7428 | 2.7554 | 2.7700 | **Q2** [0.0, 0.002] |
| `feature_fraction` | 2.8182 | 2.7281 | 2.7358 | 2.7642 | **Q2** [0.7079, 0.7368] |
| `bagging_fraction` | 2.7370 | 2.7555 | 2.7556 | 2.7984 | **Q1** [None, 0.6106] |

#### E. Slice plot

![vix@s42/naive-lightgbm](slice_vix@s42_naive-lightgbm.png)


### moe

- **holdout RMSE: 1.68475** (winner retrained in 0.21s, cv score of winner: 2.3922)
- cv best RMSE: 2.3922, median: 2.6010, p10: 2.4761
- train: median 0.472s/fold, mean 0.558s, p90 0.927s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l2` | 0.257 |
| `learning_rate` | 0.228 |
| `mixture_diversity_lambda` | 0.223 |
| `mixture_init` | 0.086 |
| `mixture_expert_dropout_rate` | 0.057 |
| `mixture_gate_type` | 0.032 |
| `bagging_fraction` | 0.031 |
| `max_depth` | 0.017 |
| `mixture_num_experts` | 0.012 |
| `feature_fraction` | 0.012 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 105 | 2.7362 | 0.5407 | 2.3922 |
| leaf_reuse | 351 | 2.7944 | 0.7703 | 2.4078 |
| gbdt | 44 | 3.1943 | 2.0600 | 2.6051 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 453 | 2.7581 | 0.7788 | 2.3922 |
| token_choice | 47 | 3.3881 | 1.7331 | 2.5232 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 286 | 2.7819 | 0.6365 | 2.4592 |
| em | 170 | 2.8088 | 1.0544 | 2.3922 |
| loss_only | 44 | 3.0809 | 1.6823 | 2.6051 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| random | 151 | 2.6059 | 0.5691 | 2.3922 |
| uniform | 266 | 2.7512 | 0.7589 | 2.5166 |
| kmeans_features | 35 | 3.0356 | 1.7968 | 2.6051 |
| tree_hierarchical | 15 | 3.1467 | 0.6490 | 2.6185 |
| gmm_features | 16 | 3.8342 | 1.2837 | 2.7617 |
| gmm | 17 | 4.0340 | 1.3746 | 2.6817 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 238 | 2.7406 | 0.5238 | 2.5347 |
| ema | 204 | 2.7947 | 1.0837 | 2.3922 |
| none | 58 | 3.2119 | 1.4229 | 2.5598 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 470 | 2.7828 | 0.8167 | 2.3922 |
| True | 30 | 3.3586 | 1.9145 | 2.6000 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 431 | 2.7227 | 0.7560 | 2.3922 |
| True | 69 | 3.4086 | 1.5156 | 2.6729 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | 2.7364 | — | 2.8920 | **Q2** [2.0, 3.0] |
| `mixture_diversity_lambda` | 2.7452 | 2.6973 | 2.8150 | 3.0120 | **Q2** [0.2239, 0.3045] |
| `mixture_warmup_iters` | 2.8629 | 2.7033 | 2.7300 | 2.9706 | **Q2** [22.0, 25.0] |
| `mixture_balance_factor` | 2.6914 | 2.8342 | 3.1705 | 2.8058 | **Q1** [None, 4.0] |
| `learning_rate` | 3.0307 | 2.7520 | 2.6736 | 2.8132 | **Q3** [0.0754, 0.0951] |
| `num_leaves` | 2.7378 | 2.9685 | 2.7124 | 2.8486 | **Q3** [94.0, 108.0] |
| `max_depth` | — | 2.7774 | 2.8727 | 2.8464 | **Q2** [3.0, 8.0] |
| `min_data_in_leaf` | 2.7109 | 2.8340 | 2.7945 | 2.9264 | **Q1** [None, 12.0] |

#### E. Slice plot

![vix@s42/moe](slice_vix@s42_moe.png)


---

## vix@s43  (search X=[3011, 13], holdout n=752)


### naive-lightgbm

- **holdout RMSE: 1.89318** (winner retrained in 0.06s, cv score of winner: 2.5849)
- cv best RMSE: 2.5849, median: 2.6917, p10: 2.6212
- train: median 0.053s/fold, mean 0.062s, p90 0.087s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.667 |
| `min_data_in_leaf` | 0.289 |
| `feature_fraction` | 0.012 |
| `bagging_fraction` | 0.012 |
| `max_depth` | 0.008 |
| `num_leaves` | 0.006 |
| `lambda_l1` | 0.004 |
| `bagging_freq` | 0.002 |
| `extra_trees` | 0.001 |
| `lambda_l2` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 449 | 2.7273 | 0.1776 | 2.5849 |
| False | 51 | 2.8425 | 0.2538 | 2.6431 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.8429 | 2.7242 | 2.6824 | 2.7067 | **Q3** [0.1774, 0.2016] |
| `num_leaves` | 2.7385 | 2.7214 | 2.7456 | 2.7507 | **Q2** [42.0, 61.0] |
| `max_depth` | — | — | 2.7044 | 2.8420 | **Q3** [3.0, 4.0] |
| `min_data_in_leaf` | 2.7062 | 2.7117 | 2.7122 | 2.8150 | **Q1** [None, 8.0] |
| `lambda_l1` | 2.7896 | 2.7304 | 2.7068 | 2.7295 | **Q3** [0.0416, 0.1672] |
| `lambda_l2` | 2.6942 | 2.7393 | 2.7290 | 2.7938 | **Q1** [None, 0.0] |
| `feature_fraction` | 2.7439 | 2.7254 | 2.7169 | 2.7701 | **Q3** [0.7498, 0.7833] |
| `bagging_fraction` | 2.7425 | 2.7329 | 2.7788 | 2.7021 | **Q4** [0.9348, ∞) |

#### E. Slice plot

![vix@s43/naive-lightgbm](slice_vix@s43_naive-lightgbm.png)


### moe

- **holdout RMSE: 1.61238** (winner retrained in 0.25s, cv score of winner: 2.3760)
- cv best RMSE: 2.3760, median: 2.5739, p10: 2.4197
- train: median 0.376s/fold, mean 2.964s, p90 8.992s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.488 |
| `mixture_diversity_lambda` | 0.168 |
| `lambda_l1` | 0.051 |
| `mixture_gate_type` | 0.047 |
| `mixture_init` | 0.036 |
| `num_leaves` | 0.031 |
| `mixture_balance_factor` | 0.027 |
| `mixture_warmup_iters` | 0.024 |
| `mixture_refit_leaves` | 0.023 |
| `mixture_load_balance_alpha` | 0.016 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **leaf_reuse** | 2.6639 (n=307) | gbdt | Δ +0.5453 | p=3.10e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 307 | 2.6639 | 0.6333 | 2.3760 |
| gbdt | 170 | 3.2092 | 2.3174 | 2.4002 |
| none | 23 | 3.7623 | 1.4888 | 2.4571 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 457 | 2.8184 | 1.2728 | 2.3760 |
| expert_choice | 43 | 3.7652 | 2.9013 | 2.3826 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 445 | 2.8346 | 1.3947 | 2.3760 |
| em | 25 | 2.9303 | 0.8025 | 2.4206 |
| loss_only | 30 | 3.8422 | 2.7528 | 2.4344 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| random | 402 | 2.7825 | 1.3325 | 2.3760 |
| uniform | 20 | 2.8916 | 0.4949 | 2.5328 |
| tree_hierarchical | 19 | 2.9529 | 0.5744 | 2.5835 |
| kmeans_features | 19 | 3.1616 | 1.1931 | 2.6313 |
| gmm | 20 | 3.6724 | 2.5959 | 2.5732 |
| gmm_features | 20 | 4.1952 | 3.1497 | 2.6524 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 414 | 2.8016 | 1.3344 | 2.3760 |
| markov | 55 | 3.0908 | 1.9192 | 2.3953 |
| none | 31 | 3.8727 | 2.2891 | 2.4451 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 2.8453 | 1.4008 | 2.3760 |
| True | 32 | 3.6983 | 2.4836 | 2.5771 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 469 | 2.8232 | 1.3070 | 2.3760 |
| True | 31 | 4.0589 | 3.0690 | 2.7129 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 3.8366 | — | — | 2.8379 | **Q4** [4.0, ∞) |
| `mixture_diversity_lambda` | 2.7451 | 2.7374 | 2.6120 | 3.5050 | **Q3** [0.237, 0.279] |
| `mixture_warmup_iters` | 2.8472 | 2.7200 | 2.8294 | 3.1732 | **Q2** [36.0, 39.0] |
| `mixture_balance_factor` | 3.2406 | 3.0562 | — | 2.7257 | **Q4** [7.0, ∞) |
| `learning_rate` | 3.5317 | 2.8098 | 2.5615 | 2.6964 | **Q3** [0.0818, 0.1011] |
| `num_leaves` | 2.6542 | 2.7849 | 2.8689 | 3.2772 | **Q1** [None, 38.0] |
| `max_depth` | — | — | 2.7284 | 3.3635 | **Q3** [3.0, 4.0] |
| `min_data_in_leaf` | 2.6373 | 2.7124 | 2.8311 | 3.3668 | **Q1** [None, 7.0] |

#### E. Slice plot

![vix@s43/moe](slice_vix@s43_moe.png)


---

## vix@s44  (search X=[3011, 13], holdout n=752)


### naive-lightgbm

- **holdout RMSE: 1.73436** (winner retrained in 0.08s, cv score of winner: 2.6491)
- cv best RMSE: 2.6491, median: 2.7244, p10: 2.6847
- train: median 0.066s/fold, mean 0.070s, p90 0.103s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.480 |
| `learning_rate` | 0.363 |
| `extra_trees` | 0.036 |
| `num_leaves` | 0.035 |
| `max_depth` | 0.035 |
| `bagging_fraction` | 0.020 |
| `bagging_freq` | 0.014 |
| `feature_fraction` | 0.012 |
| `lambda_l2` | 0.005 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 470 | 2.7555 | 0.1315 | 2.6491 |
| True | 30 | 2.9885 | 0.2639 | 2.7179 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.8129 | 2.7511 | 2.7380 | 2.7760 | **Q3** [0.0489, 0.0709] |
| `num_leaves` | 2.7794 | 2.7623 | 2.7505 | 2.7881 | **Q3** [68.0, 73.25] |
| `max_depth` | 2.7848 | 2.7764 | — | 2.7647 | **Q4** [5.0, ∞) |
| `min_data_in_leaf` | 2.7802 | 2.7276 | 2.7353 | 2.8236 | **Q2** [20.0, 24.0] |
| `lambda_l1` | 2.7709 | 2.7526 | 2.7812 | 2.7733 | **Q2** [0.0, 0.0001] |
| `lambda_l2` | 2.7519 | 2.7682 | 2.7522 | 2.8058 | **Q1** [None, 0.0] |
| `feature_fraction` | 2.7956 | 2.7418 | 2.7627 | 2.7779 | **Q2** [0.7376, 0.766] |
| `bagging_fraction` | 2.8315 | 2.7451 | 2.7405 | 2.7611 | **Q3** [0.8748, 0.8987] |

#### E. Slice plot

![vix@s44/naive-lightgbm](slice_vix@s44_naive-lightgbm.png)


### moe

- **holdout RMSE: 1.66045** (winner retrained in 5.47s, cv score of winner: 2.3756)
- cv best RMSE: 2.3756, median: 2.5804, p10: 2.4598
- train: median 0.622s/fold, mean 1.264s, p90 4.185s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_warmup_iters` | 0.428 |
| `mixture_init` | 0.205 |
| `mixture_gate_type` | 0.073 |
| `feature_fraction` | 0.064 |
| `learning_rate` | 0.059 |
| `num_leaves` | 0.045 |
| `lambda_l1` | 0.018 |
| `mixture_diversity_lambda` | 0.015 |
| `bagging_freq` | 0.015 |
| `mixture_hard_m_step` | 0.012 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 110 | 2.9130 | 1.0253 | 2.3756 |
| leaf_reuse | 342 | 3.0246 | 1.5127 | 2.4058 |
| none | 48 | 3.4747 | 2.2942 | 2.4747 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 420 | 3.0297 | 1.5477 | 2.3756 |
| expert_choice | 80 | 3.1146 | 1.3933 | 2.5366 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| em | 163 | 2.8566 | 1.0045 | 2.4399 |
| loss_only | 282 | 3.0784 | 1.6036 | 2.3756 |
| gate_only | 55 | 3.4167 | 2.1677 | 2.5226 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| random | 371 | 2.7217 | 0.9154 | 2.3756 |
| uniform | 48 | 2.8709 | 0.9621 | 2.5366 |
| tree_hierarchical | 20 | 3.1917 | 0.5085 | 2.6735 |
| kmeans_features | 23 | 3.9873 | 2.1175 | 2.7784 |
| gmm_features | 18 | 5.7448 | 3.0133 | 2.8328 |
| gmm | 20 | 5.7562 | 3.1260 | 2.8129 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 386 | 2.9424 | 1.3422 | 2.3756 |
| ema | 71 | 3.3742 | 2.0379 | 2.4787 |
| markov | 43 | 3.4028 | 1.8944 | 2.4218 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 2.9871 | 1.4394 | 2.3756 |
| True | 32 | 3.8642 | 2.2992 | 2.7441 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 3.0187 | 1.5359 | 2.3756 |
| True | 32 | 3.4022 | 1.2915 | 2.7902 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 3.1982 | — | — | 3.0253 | **Q4** [5.0, ∞) |
| `mixture_diversity_lambda` | 2.8140 | 3.0862 | 3.2380 | 3.0349 | **Q1** [None, 0.2017] |
| `mixture_warmup_iters` | 3.2073 | 2.9164 | 3.2332 | 2.8747 | **Q4** [41.0, ∞) |
| `mixture_balance_factor` | 2.8302 | — | 3.0892 | 3.0235 | **Q1** [None, 3.0] |
| `learning_rate` | 3.4137 | 2.9281 | 3.1677 | 2.6636 | **Q4** [0.2269, ∞) |
| `num_leaves` | 2.9488 | 3.1454 | 3.0754 | 2.9992 | **Q1** [None, 13.0] |
| `max_depth` | 3.1466 | 3.0045 | 3.4204 | 2.9922 | **Q4** [9.0, ∞) |
| `min_data_in_leaf` | 2.9464 | 3.2144 | 2.7946 | 3.2176 | **Q3** [23.0, 27.0] |

#### E. Slice plot

![vix@s44/moe](slice_vix@s44_moe.png)


---

## Overall recommendations

**Categorical settings that are statistically significant winners (p<0.01):**

| dataset | param | best value | Δ vs runner-up | p |
|---|---|---|---|---|
| vix@s43 | `mixture_gate_type` | **leaf_reuse** | +0.5453 | 3.10e-03 |
