# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 300

- **Datasets**: ['wine_quality'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `5232455cdc8d`, lib sha256 `a3c30fcc7fd3…`, package `/home/mokumoku/Develop/LightGBM-MoE/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| wine_quality | naive-lightgbm | **0.61547** ± 0.01639 | 0.66867 | 0.0% | 0.57 |
| wine_quality | moe | **0.61920** ± 0.01025 | 0.67219 | 0.0% | 5.99 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| wine_quality@s42 | naive-lightgbm | 0.6678 | 0.6775 | 0.402 | 583 |
| wine_quality@s42 | moe | 0.6823 | 0.6921 | 0.874 | 2286 |
| wine_quality@s43 | naive-lightgbm | 0.6630 | 0.6686 | 0.446 | 618 |
| wine_quality@s43 | moe | 0.6597 | 0.6709 | 2.255 | 4044 |
| wine_quality@s44 | naive-lightgbm | 0.6752 | 0.6836 | 0.260 | 352 |
| wine_quality@s44 | moe | 0.6747 | 0.6855 | 1.519 | 3474 |



---

## wine_quality@s42  (search X=[5198, 11], holdout n=1299)


### naive-lightgbm

- **holdout RMSE: 0.60747** (winner retrained in 0.55s, cv score of winner: 0.6678)
- cv best RMSE: 0.6678, median: 0.6775, p10: 0.6709
- train: median 0.402s/fold, mean 0.384s, p90 0.597s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.427 |
| `extra_trees` | 0.294 |
| `min_data_in_leaf` | 0.180 |
| `max_depth` | 0.043 |
| `bagging_fraction` | 0.023 |
| `feature_fraction` | 0.018 |
| `num_leaves` | 0.007 |
| `bagging_freq` | 0.005 |
| `lambda_l2` | 0.002 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 278 | 0.6799 | 0.0108 | 0.6678 |
| True | 22 | 0.7226 | 0.0318 | 0.6907 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.6896 | 0.6759 | 0.6794 | 0.6871 | **Q2** [0.0424, 0.0489] |
| `num_leaves` | 0.6918 | 0.6845 | 0.6780 | 0.6778 | **Q4** [120.25, ∞) |
| `max_depth` | 0.6997 | 0.6796 | — | 0.6782 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 0.6774 | 0.6765 | 0.6788 | 0.6981 | **Q2** [7.0, 9.0] |
| `lambda_l1` | 0.6817 | 0.6793 | 0.6787 | 0.6923 | **Q3** [0.0, 0.0] |
| `lambda_l2` | 0.6775 | 0.6869 | 0.6872 | 0.6804 | **Q1** [None, 0.0] |
| `feature_fraction` | 0.6831 | 0.6797 | 0.6814 | 0.6878 | **Q2** [0.623, 0.6574] |
| `bagging_fraction` | 0.6905 | 0.6807 | 0.6772 | 0.6836 | **Q3** [0.8827, 0.9014] |

#### E. Slice plot

![wine_quality@s42/naive-lightgbm](slice_wine_quality@s42_naive-lightgbm.png)


### moe

- **holdout RMSE: 0.60869** (winner retrained in 12.85s, cv score of winner: 0.6823)
- cv best RMSE: 0.6823, median: 0.6921, p10: 0.6853
- train: median 0.874s/fold, mean 1.507s, p90 2.930s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l2` | 0.461 |
| `lambda_l1` | 0.233 |
| `mixture_diversity_lambda` | 0.053 |
| `learning_rate` | 0.048 |
| `mixture_gate_type` | 0.030 |
| `bagging_fraction` | 0.027 |
| `extra_trees` | 0.023 |
| `feature_fraction` | 0.018 |
| `bagging_freq` | 0.018 |
| `mixture_e_step_mode` | 0.017 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 206 | 0.8358 | 0.5626 | 0.6828 |
| gbdt | 59 | 0.8736 | 0.7408 | 0.6823 |
| none | 35 | 0.9949 | 0.8207 | 0.6911 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 88 | 0.8610 | 0.5737 | 0.6823 |
| token_choice | 212 | 0.8622 | 0.6622 | 0.6828 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 43 | 0.8371 | 0.4076 | 0.6826 |
| gate_only | 223 | 0.8551 | 0.6470 | 0.6823 |
| em | 34 | 0.9375 | 0.7898 | 0.6823 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gmm | 176 | 0.7662 | 0.3917 | 0.6823 |
| tree_hierarchical | 14 | 0.8962 | 0.2927 | 0.6874 |
| random | 80 | 0.9998 | 0.8487 | 0.6865 |
| uniform | 30 | 1.0390 | 1.0327 | 0.6911 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 166 | 0.7813 | 0.4797 | 0.6823 |
| markov | 33 | 0.8718 | 0.4389 | 0.6911 |
| none | 101 | 0.9910 | 0.8595 | 0.6856 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 197 | 0.7764 | 0.4010 | 0.6823 |
| True | 103 | 1.0251 | 0.9141 | 0.6892 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 270 | 0.8159 | 0.5104 | 0.6823 |
| True | 30 | 1.2747 | 1.2368 | 0.6998 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.9636 | 0.9981 | — | 0.7699 | **Q4** [4.0, ∞) |
| `mixture_diversity_lambda` | 0.7007 | 0.7429 | 1.0417 | 0.9619 | **Q1** [None, 0.0128] |
| `mixture_warmup_iters` | 0.7916 | 0.7564 | 0.9439 | 0.9462 | **Q2** [24.0, 28.0] |
| `mixture_balance_factor` | — | — | 0.7848 | 1.0889 | **Q3** [2.0, 5.0] |
| `learning_rate` | 1.0842 | 0.6937 | 0.7682 | 0.9011 | **Q2** [0.1184, 0.1425] |
| `num_leaves` | 0.7363 | 0.8116 | 1.1113 | 0.7776 | **Q1** [None, 33.0] |
| `max_depth` | 1.1286 | — | — | 0.7895 | **Q4** [11.0, ∞) |
| `min_data_in_leaf` | 0.8454 | 0.7881 | 0.8132 | 0.9935 | **Q2** [48.0, 53.0] |

#### E. Slice plot

![wine_quality@s42/moe](slice_wine_quality@s42_moe.png)


---

## wine_quality@s43  (search X=[5198, 11], holdout n=1299)


### naive-lightgbm

- **holdout RMSE: 0.63831** (winner retrained in 0.73s, cv score of winner: 0.6630)
- cv best RMSE: 0.6630, median: 0.6686, p10: 0.6653
- train: median 0.446s/fold, mean 0.406s, p90 0.620s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.431 |
| `extra_trees` | 0.408 |
| `min_data_in_leaf` | 0.061 |
| `bagging_freq` | 0.032 |
| `bagging_fraction` | 0.023 |
| `max_depth` | 0.019 |
| `num_leaves` | 0.011 |
| `feature_fraction` | 0.010 |
| `lambda_l1` | 0.003 |
| `lambda_l2` | 0.002 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 279 | 0.6728 | 0.0112 | 0.6630 |
| True | 21 | 0.7180 | 0.0213 | 0.6886 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.6847 | 0.6711 | 0.6711 | 0.6770 | **Q2** [0.0516, 0.0598] |
| `num_leaves` | 0.6877 | 0.6725 | 0.6726 | 0.6714 | **Q4** [123.0, ∞) |
| `max_depth` | 0.6871 | — | 0.6741 | 0.6718 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 0.6692 | 0.6715 | 0.6725 | 0.6894 | **Q1** [None, 6.0] |
| `lambda_l1` | 0.6831 | 0.6735 | 0.6698 | 0.6776 | **Q3** [0.0733, 0.1665] |
| `lambda_l2` | 0.6754 | 0.6693 | 0.6759 | 0.6834 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 0.6737 | 0.6716 | 0.6720 | 0.6867 | **Q2** [0.5368, 0.5666] |
| `bagging_fraction` | 0.6761 | 0.6727 | 0.6702 | 0.6851 | **Q3** [0.6535, 0.6781] |

#### E. Slice plot

![wine_quality@s43/naive-lightgbm](slice_wine_quality@s43_naive-lightgbm.png)


### moe

- **holdout RMSE: 0.63309** (winner retrained in 2.94s, cv score of winner: 0.6597)
- cv best RMSE: 0.6597, median: 0.6709, p10: 0.6628
- train: median 2.255s/fold, mean 2.670s, p90 4.948s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.304 |
| `bagging_fraction` | 0.113 |
| `feature_fraction` | 0.087 |
| `max_depth` | 0.069 |
| `learning_rate` | 0.068 |
| `mixture_r_smoothing` | 0.059 |
| `min_data_in_leaf` | 0.051 |
| `num_leaves` | 0.043 |
| `mixture_gate_type` | 0.041 |
| `mixture_init` | 0.036 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 263 | 0.7349 | 0.3441 | 0.6597 |
| none | 21 | 0.9407 | 0.6219 | 0.6678 |
| leaf_reuse | 16 | 1.2272 | 1.1813 | 0.6895 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 273 | 0.7431 | 0.3476 | 0.6597 |
| token_choice | 27 | 1.1034 | 1.0488 | 0.6700 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| em | 116 | 0.7507 | 0.3122 | 0.6609 |
| loss_only | 167 | 0.7539 | 0.4124 | 0.6597 |
| gate_only | 17 | 1.1570 | 1.1767 | 0.6670 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gmm | 250 | 0.7378 | 0.3501 | 0.6597 |
| uniform | 18 | 0.7442 | 0.1743 | 0.6663 |
| tree_hierarchical | 15 | 0.8774 | 0.3601 | 0.6708 |
| random | 17 | 1.2738 | 1.2829 | 0.6677 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 261 | 0.7342 | 0.3442 | 0.6597 |
| markov | 19 | 1.0443 | 1.1163 | 0.6688 |
| none | 20 | 1.0597 | 0.6263 | 0.6779 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 276 | 0.7332 | 0.2852 | 0.6597 |
| True | 24 | 1.2621 | 1.2457 | 0.6691 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 20 | 0.7670 | 0.1875 | 0.6822 |
| False | 280 | 0.7761 | 0.4824 | 0.6597 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.8473 | — | — | 0.7709 | **Q4** [3.0, ∞) |
| `mixture_diversity_lambda` | 0.7244 | 0.8071 | 0.6892 | 0.8814 | **Q3** [0.0956, 0.1205] |
| `mixture_warmup_iters` | 0.8107 | 0.7237 | 0.7840 | 0.7779 | **Q2** [23.0, 26.0] |
| `mixture_balance_factor` | 1.1795 | 0.7311 | — | 0.7479 | **Q2** [6.0, 8.0] |
| `learning_rate` | 0.8472 | 0.7250 | 0.7222 | 0.8077 | **Q3** [0.0667, 0.0904] |
| `num_leaves` | 0.8544 | 0.7413 | 0.7067 | 0.8106 | **Q3** [74.0, 84.25] |
| `max_depth` | 0.9125 | 0.7428 | 0.7866 | 0.7318 | **Q4** [11.0, ∞) |
| `min_data_in_leaf` | 0.7297 | 0.7564 | 0.7965 | 0.8117 | **Q1** [None, 10.0] |

#### E. Slice plot

![wine_quality@s43/moe](slice_wine_quality@s43_moe.png)


---

## wine_quality@s44  (search X=[5198, 11], holdout n=1299)


### naive-lightgbm

- **holdout RMSE: 0.60061** (winner retrained in 0.44s, cv score of winner: 0.6752)
- cv best RMSE: 0.6752, median: 0.6836, p10: 0.6790
- train: median 0.260s/fold, mean 0.230s, p90 0.339s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.498 |
| `learning_rate` | 0.184 |
| `min_data_in_leaf` | 0.092 |
| `max_depth` | 0.059 |
| `bagging_fraction` | 0.057 |
| `num_leaves` | 0.049 |
| `bagging_freq` | 0.028 |
| `lambda_l2` | 0.015 |
| `feature_fraction` | 0.014 |
| `lambda_l1` | 0.004 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 277 | 0.6869 | 0.0103 | 0.6752 |
| True | 23 | 0.7177 | 0.0155 | 0.6983 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.6919 | 0.6861 | 0.6853 | 0.6937 | **Q3** [0.0494, 0.0663] |
| `num_leaves` | 0.6991 | 0.6865 | 0.6863 | 0.6856 | **Q4** [121.0, ∞) |
| `max_depth` | 0.6980 | — | 0.6845 | 0.6934 | **Q3** [9.0, 10.0] |
| `min_data_in_leaf` | — | 0.6835 | 0.6883 | 0.7004 | **Q2** [5.0, 8.0] |
| `lambda_l1` | 0.6947 | 0.6862 | 0.6849 | 0.6913 | **Q3** [0.027, 0.0871] |
| `lambda_l2` | 0.6903 | 0.6838 | 0.6869 | 0.6962 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 0.6873 | 0.6854 | 0.6856 | 0.6989 | **Q2** [0.5384, 0.5618] |
| `bagging_fraction` | 0.7004 | 0.6865 | 0.6836 | 0.6866 | **Q3** [0.9024, 0.9231] |

#### E. Slice plot

![wine_quality@s44/naive-lightgbm](slice_wine_quality@s44_naive-lightgbm.png)


### moe

- **holdout RMSE: 0.61581** (winner retrained in 2.19s, cv score of winner: 0.6747)
- cv best RMSE: 0.6747, median: 0.6855, p10: 0.6779
- train: median 1.519s/fold, mean 2.299s, p90 5.477s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l2` | 0.396 |
| `lambda_l1` | 0.337 |
| `bagging_fraction` | 0.038 |
| `feature_fraction` | 0.036 |
| `mixture_e_step_mode` | 0.036 |
| `mixture_hard_m_step` | 0.034 |
| `mixture_gate_type` | 0.023 |
| `learning_rate` | 0.022 |
| `mixture_warmup_iters` | 0.019 |
| `mixture_r_smoothing` | 0.018 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 261 | 0.7639 | 0.4972 | 0.6747 |
| leaf_reuse | 21 | 1.0345 | 0.6712 | 0.6911 |
| none | 18 | 1.3239 | 1.1068 | 0.6917 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 250 | 0.8090 | 0.5930 | 0.6747 |
| token_choice | 50 | 0.8537 | 0.5364 | 0.6756 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| em | 31 | 0.7871 | 0.2774 | 0.6778 |
| loss_only | 241 | 0.8016 | 0.5802 | 0.6747 |
| gate_only | 28 | 0.9761 | 0.8036 | 0.6773 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 16 | 0.7108 | 0.0317 | 0.6841 |
| tree_hierarchical | 19 | 0.7887 | 0.2049 | 0.6896 |
| gmm | 219 | 0.7896 | 0.5295 | 0.6747 |
| random | 46 | 0.9923 | 0.9138 | 0.6870 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 141 | 0.7693 | 0.5133 | 0.6747 |
| ema | 101 | 0.7785 | 0.4059 | 0.6758 |
| none | 58 | 0.9970 | 0.8931 | 0.6799 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 279 | 0.7757 | 0.5007 | 0.6747 |
| False | 21 | 1.3575 | 1.1093 | 0.6823 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 30 | 0.7614 | 0.1672 | 0.7013 |
| False | 270 | 0.8225 | 0.6130 | 0.6747 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.8038 | 0.8428 | — | 0.8077 | **Q1** [None, 3.0] |
| `mixture_diversity_lambda` | 0.8835 | 0.8844 | 0.7337 | 0.7642 | **Q3** [0.3829, 0.417] |
| `mixture_warmup_iters` | 0.9755 | 0.7210 | 0.7932 | 0.7878 | **Q2** [43.0, 47.5] |
| `mixture_balance_factor` | 1.2272 | 0.7947 | — | 0.7672 | **Q4** [5.0, ∞) |
| `learning_rate` | 0.8516 | 0.7445 | 0.8151 | 0.8546 | **Q2** [0.0563, 0.0714] |
| `num_leaves` | 0.7957 | 0.8429 | 0.7414 | 0.8835 | **Q3** [73.0, 85.0] |
| `max_depth` | 1.0047 | — | 0.7301 | 0.8450 | **Q3** [10.0, 11.0] |
| `min_data_in_leaf` | 0.8283 | 0.7450 | 0.7715 | 0.9123 | **Q2** [7.0, 9.0] |

#### E. Slice plot

![wine_quality@s44/moe](slice_wine_quality@s44_moe.png)


---

## Overall recommendations

(no categorical settings were universally significant — see per-dataset breakdown)
