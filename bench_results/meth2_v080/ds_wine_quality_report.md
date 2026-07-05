# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 300

- **Datasets**: ['wine_quality'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `5232455cdc8d`, lib sha256 `5cec0a0bd5ab…`, package `/tmp/lgbm-moe-v080/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| wine_quality | naive-lightgbm | **0.61547** ± 0.01639 | 0.66867 | 0.0% | 0.61 |
| wine_quality | moe | **0.61573** ± 0.01362 | 0.66666 | 0.0% | 2.68 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| wine_quality@s42 | naive-lightgbm | 0.6678 | 0.6775 | 0.407 | 584 |
| wine_quality@s42 | moe | 0.6654 | 0.6812 | 1.745 | 2877 |
| wine_quality@s43 | naive-lightgbm | 0.6630 | 0.6686 | 0.445 | 618 |
| wine_quality@s43 | moe | 0.6585 | 0.6651 | 2.030 | 3275 |
| wine_quality@s44 | naive-lightgbm | 0.6752 | 0.6836 | 0.261 | 352 |
| wine_quality@s44 | moe | 0.6761 | 0.6882 | 1.582 | 2299 |



---

## wine_quality@s42  (search X=[5198, 11], holdout n=1299)


### naive-lightgbm

- **holdout RMSE: 0.60747** (winner retrained in 0.62s, cv score of winner: 0.6678)
- cv best RMSE: 0.6678, median: 0.6775, p10: 0.6709
- train: median 0.407s/fold, mean 0.384s, p90 0.589s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.406 |
| `extra_trees` | 0.304 |
| `min_data_in_leaf` | 0.155 |
| `feature_fraction` | 0.060 |
| `bagging_fraction` | 0.027 |
| `max_depth` | 0.027 |
| `bagging_freq` | 0.017 |
| `num_leaves` | 0.002 |
| `lambda_l2` | 0.001 |
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

- **holdout RMSE: 0.60151** (winner retrained in 3.85s, cv score of winner: 0.6654)
- cv best RMSE: 0.6654, median: 0.6812, p10: 0.6698
- train: median 1.745s/fold, mean 1.900s, p90 3.080s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.471 |
| `lambda_l2` | 0.242 |
| `bagging_fraction` | 0.123 |
| `learning_rate` | 0.047 |
| `feature_fraction` | 0.024 |
| `extra_trees` | 0.021 |
| `mixture_routing_mode` | 0.021 |
| `mixture_gate_type` | 0.021 |
| `mixture_diversity_lambda` | 0.012 |
| `mixture_e_step_mode` | 0.006 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 220 | 0.7246 | 0.3894 | 0.6654 |
| leaf_reuse | 61 | 0.8088 | 0.4331 | 0.6807 |
| none | 19 | 0.8315 | 0.3322 | 0.6773 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 225 | 0.7075 | 0.1929 | 0.6654 |
| token_choice | 75 | 0.8715 | 0.7070 | 0.6793 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| em | 192 | 0.6910 | 0.0725 | 0.6654 |
| gate_only | 73 | 0.8381 | 0.6762 | 0.6778 |
| loss_only | 35 | 0.8773 | 0.5655 | 0.6671 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gmm | 181 | 0.7007 | 0.1548 | 0.6654 |
| random | 57 | 0.7876 | 0.4161 | 0.6778 |
| tree_hierarchical | 22 | 0.8105 | 0.3192 | 0.6716 |
| uniform | 40 | 0.8751 | 0.8618 | 0.6749 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 219 | 0.7166 | 0.3665 | 0.6654 |
| markov | 17 | 0.8173 | 0.3133 | 0.6772 |
| none | 64 | 0.8394 | 0.4916 | 0.6793 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 280 | 0.7290 | 0.2703 | 0.6654 |
| True | 20 | 1.0219 | 1.1252 | 0.6819 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 265 | 0.7243 | 0.2518 | 0.6654 |
| True | 35 | 0.9315 | 0.9142 | 0.6908 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.7978 | 0.9018 | — | 0.7152 | **Q4** [4.0, ∞) |
| `mixture_diversity_lambda` | 0.6949 | 0.8650 | 0.7316 | 0.7026 | **Q1** [None, 0.0542] |
| `mixture_warmup_iters` | 0.7674 | 0.7498 | 0.6979 | 0.8025 | **Q3** [8.0, 12.0] |
| `mixture_balance_factor` | 0.7826 | 0.8806 | — | 0.7003 | **Q4** [10.0, ∞) |
| `learning_rate` | 0.7994 | 0.7987 | 0.6822 | 0.7137 | **Q3** [0.0858, 0.1055] |
| `num_leaves` | 0.8830 | 0.7299 | 0.6849 | 0.6990 | **Q3** [110.0, 116.0] |
| `max_depth` | 0.7715 | 0.6844 | — | 0.7660 | **Q2** [8.0, 9.0] |
| `min_data_in_leaf` | 0.6827 | 0.7017 | 0.6889 | 0.9111 | **Q1** [None, 6.0] |

#### E. Slice plot

![wine_quality@s42/moe](slice_wine_quality@s42_moe.png)


---

## wine_quality@s43  (search X=[5198, 11], holdout n=1299)


### naive-lightgbm

- **holdout RMSE: 0.63831** (winner retrained in 0.76s, cv score of winner: 0.6630)
- cv best RMSE: 0.6630, median: 0.6686, p10: 0.6653
- train: median 0.445s/fold, mean 0.407s, p90 0.623s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.422 |
| `learning_rate` | 0.413 |
| `min_data_in_leaf` | 0.079 |
| `bagging_freq` | 0.036 |
| `max_depth` | 0.018 |
| `feature_fraction` | 0.013 |
| `bagging_fraction` | 0.009 |
| `num_leaves` | 0.006 |
| `lambda_l1` | 0.002 |
| `lambda_l2` | 0.001 |

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

- **holdout RMSE: 0.63410** (winner retrained in 1.75s, cv score of winner: 0.6585)
- cv best RMSE: 0.6585, median: 0.6651, p10: 0.6614
- train: median 2.030s/fold, mean 2.166s, p90 3.098s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.191 |
| `lambda_l2` | 0.089 |
| `mixture_balance_factor` | 0.076 |
| `feature_fraction` | 0.075 |
| `bagging_freq` | 0.074 |
| `num_leaves` | 0.072 |
| `mixture_e_step_mode` | 0.059 |
| `mixture_r_smoothing` | 0.040 |
| `bagging_fraction` | 0.038 |
| `min_data_in_leaf` | 0.038 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 184 | 0.7325 | 0.3607 | 0.6604 |
| leaf_reuse | 102 | 0.7753 | 0.5394 | 0.6585 |
| none | 14 | 1.1653 | 0.9351 | 0.6649 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 191 | 0.7651 | 0.4175 | 0.6604 |
| token_choice | 109 | 0.7710 | 0.5668 | 0.6585 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| em | 121 | 0.7249 | 0.3105 | 0.6585 |
| gate_only | 117 | 0.7823 | 0.6366 | 0.6604 |
| loss_only | 62 | 0.8214 | 0.3769 | 0.6618 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 222 | 0.7168 | 0.3431 | 0.6585 |
| gmm | 50 | 0.7977 | 0.3599 | 0.6627 |
| tree_hierarchical | 14 | 0.9286 | 0.4012 | 0.6644 |
| random | 14 | 1.2976 | 1.4274 | 0.6625 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 266 | 0.7259 | 0.3370 | 0.6585 |
| markov | 16 | 1.0867 | 1.1816 | 0.6628 |
| none | 18 | 1.0941 | 0.8079 | 0.6645 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 276 | 0.7226 | 0.3148 | 0.6585 |
| True | 24 | 1.2810 | 1.1914 | 0.6741 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 280 | 0.7668 | 0.4912 | 0.6585 |
| True | 20 | 0.7731 | 0.1928 | 0.6922 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.9648 | — | 0.7582 | 0.7638 | **Q3** [3.0, 4.0] |
| `mixture_diversity_lambda` | 0.7064 | 0.7033 | 0.6959 | 0.9634 | **Q3** [0.0463, 0.1134] |
| `mixture_warmup_iters` | 0.9696 | 0.6924 | 0.7426 | 0.7101 | **Q2** [37.0, 41.0] |
| `mixture_balance_factor` | 0.8540 | 0.7686 | — | 0.7319 | **Q4** [5.0, ∞) |
| `learning_rate` | 0.8116 | 0.6758 | 0.7286 | 0.8529 | **Q2** [0.0547, 0.0635] |
| `num_leaves` | 0.9249 | 0.6775 | 0.7161 | 0.7434 | **Q2** [104.75, 113.0] |
| `max_depth` | 0.9853 | 0.7345 | — | 0.7022 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 0.6911 | 0.6941 | 0.7334 | 0.9308 | **Q1** [None, 6.0] |

#### E. Slice plot

![wine_quality@s43/moe](slice_wine_quality@s43_moe.png)


---

## wine_quality@s44  (search X=[5198, 11], holdout n=1299)


### naive-lightgbm

- **holdout RMSE: 0.60061** (winner retrained in 0.44s, cv score of winner: 0.6752)
- cv best RMSE: 0.6752, median: 0.6836, p10: 0.6790
- train: median 0.261s/fold, mean 0.230s, p90 0.333s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.450 |
| `learning_rate` | 0.192 |
| `min_data_in_leaf` | 0.093 |
| `bagging_fraction` | 0.072 |
| `max_depth` | 0.071 |
| `lambda_l2` | 0.033 |
| `num_leaves` | 0.033 |
| `bagging_freq` | 0.023 |
| `feature_fraction` | 0.022 |
| `lambda_l1` | 0.010 |

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

- **holdout RMSE: 0.61159** (winner retrained in 2.44s, cv score of winner: 0.6761)
- cv best RMSE: 0.6761, median: 0.6882, p10: 0.6788
- train: median 1.582s/fold, mean 1.524s, p90 2.768s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l2` | 0.578 |
| `lambda_l1` | 0.091 |
| `num_leaves` | 0.064 |
| `feature_fraction` | 0.051 |
| `learning_rate` | 0.050 |
| `bagging_fraction` | 0.036 |
| `extra_trees` | 0.026 |
| `mixture_balance_factor` | 0.021 |
| `mixture_init` | 0.017 |
| `mixture_warmup_iters` | 0.012 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 234 | 0.8029 | 0.5126 | 0.6761 |
| gbdt | 40 | 0.9809 | 0.7424 | 0.6926 |
| leaf_reuse | 26 | 1.1821 | 1.1228 | 0.6954 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 270 | 0.8362 | 0.5617 | 0.6761 |
| token_choice | 30 | 1.0692 | 1.0596 | 0.6935 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| em | 173 | 0.8048 | 0.5713 | 0.6761 |
| loss_only | 38 | 0.9031 | 0.6772 | 0.6817 |
| gate_only | 89 | 0.9472 | 0.7125 | 0.6778 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gmm | 150 | 0.7832 | 0.5027 | 0.6761 |
| random | 90 | 0.8608 | 0.5720 | 0.6778 |
| uniform | 46 | 0.9478 | 0.7159 | 0.6874 |
| tree_hierarchical | 14 | 1.3785 | 1.3204 | 0.6822 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 233 | 0.8001 | 0.4703 | 0.6761 |
| none | 16 | 0.9509 | 0.8024 | 0.7029 |
| ema | 51 | 1.1023 | 1.0342 | 0.6926 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 273 | 0.8276 | 0.5651 | 0.6761 |
| False | 27 | 1.1816 | 1.0554 | 0.6850 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 278 | 0.8424 | 0.6235 | 0.6761 |
| True | 22 | 1.0759 | 0.7125 | 0.7026 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | — | — | 0.8595 | **Q4** [2.0, ∞) |
| `mixture_diversity_lambda` | 1.0737 | 0.8117 | 0.7294 | 0.8232 | **Q3** [0.426, 0.4508] |
| `mixture_warmup_iters` | 0.8646 | 0.9662 | 0.8149 | 0.7923 | **Q4** [43.0, ∞) |
| `mixture_balance_factor` | 1.1061 | — | 0.8189 | 0.8174 | **Q4** [6.0, ∞) |
| `learning_rate` | 0.8688 | 0.8084 | 0.9412 | 0.8196 | **Q2** [0.0713, 0.0858] |
| `num_leaves` | 0.9926 | 0.8432 | 0.8403 | 0.7629 | **Q4** [120.0, ∞) |
| `max_depth` | 1.0863 | 0.8232 | — | 0.7737 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 0.7693 | 0.8086 | 0.8491 | 1.0034 | **Q1** [None, 7.0] |

#### E. Slice plot

![wine_quality@s44/moe](slice_wine_quality@s44_moe.png)


---

## Overall recommendations

(no categorical settings were universally significant — see per-dataset breakdown)
