# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 300

- **Datasets**: ['wine_quality'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `747bb4a7b5e6`, lib sha256 `a3c30fcc7fd3…`, package `/home/mokumoku/Develop/LightGBM-MoE/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| wine_quality | naive-ensemble | **0.60819** ± 0.01215 | 0.66418 | 0.0% | 1.36 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| wine_quality@s42 | naive-ensemble | 0.6651 | 0.6707 | 1.123 | 1503 |
| wine_quality@s43 | naive-ensemble | 0.6574 | 0.6623 | 0.820 | 1142 |
| wine_quality@s44 | naive-ensemble | 0.6701 | 0.6780 | 0.630 | 840 |



---

## wine_quality@s42  (search X=[5198, 11], holdout n=1299)


### naive-ensemble

- **holdout RMSE: 0.60706** (winner retrained in 1.55s, cv score of winner: 0.6651)
- cv best RMSE: 0.6651, median: 0.6707, p10: 0.6667
- train: median 1.123s/fold, mean 0.999s, p90 1.503s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.467 |
| `learning_rate` | 0.310 |
| `min_data_in_leaf` | 0.128 |
| `max_depth` | 0.029 |
| `feature_fraction` | 0.024 |
| `n_models` | 0.022 |
| `lambda_l1` | 0.007 |
| `num_leaves` | 0.006 |
| `bagging_fraction` | 0.005 |
| `bagging_freq` | 0.002 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 280 | 0.6750 | 0.0122 | 0.6651 |
| True | 20 | 0.7165 | 0.0252 | 0.6842 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.6869 | 0.6736 | 0.6719 | 0.6787 | **Q3** [0.0565, 0.0658] |
| `num_leaves` | 0.6832 | 0.6768 | 0.6756 | 0.6757 | **Q3** [112.0, 121.0] |
| `max_depth` | 0.6939 | 0.6808 | — | 0.6719 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | — | 0.6709 | 0.6758 | 0.6915 | **Q2** [5.0, 8.0] |
| `lambda_l1` | 0.6856 | 0.6736 | 0.6739 | 0.6780 | **Q2** [0.0135, 0.0949] |
| `lambda_l2` | 0.6804 | 0.6748 | 0.6803 | 0.6756 | **Q2** [0.0, 0.0002] |
| `feature_fraction` | 0.6848 | 0.6726 | 0.6743 | 0.6794 | **Q2** [0.6852, 0.7443] |
| `bagging_fraction` | 0.6777 | 0.6728 | 0.6729 | 0.6877 | **Q2** [0.6758, 0.7057] |

#### E. Slice plot

![wine_quality@s42/naive-ensemble](slice_wine_quality@s42_naive-ensemble.png)


---

## wine_quality@s43  (search X=[5198, 11], holdout n=1299)


### naive-ensemble

- **holdout RMSE: 0.62359** (winner retrained in 1.32s, cv score of winner: 0.6574)
- cv best RMSE: 0.6574, median: 0.6623, p10: 0.6594
- train: median 0.820s/fold, mean 0.759s, p90 1.108s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.333 |
| `learning_rate` | 0.321 |
| `min_data_in_leaf` | 0.200 |
| `max_depth` | 0.088 |
| `bagging_fraction` | 0.021 |
| `bagging_freq` | 0.012 |
| `n_models` | 0.008 |
| `feature_fraction` | 0.007 |
| `num_leaves` | 0.007 |
| `lambda_l2` | 0.002 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 278 | 0.6660 | 0.0104 | 0.6574 |
| True | 22 | 0.7032 | 0.0223 | 0.6770 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.6781 | 0.6635 | 0.6628 | 0.6706 | **Q3** [0.1006, 0.1155] |
| `num_leaves` | 0.6791 | 0.6659 | 0.6653 | 0.6648 | **Q4** [124.25, ∞) |
| `max_depth` | 0.6862 | 0.6663 | — | 0.6649 | **Q4** [11.0, ∞) |
| `min_data_in_leaf` | — | 0.6631 | 0.6652 | 0.6831 | **Q2** [5.0, 8.0] |
| `lambda_l1` | 0.6730 | 0.6683 | 0.6635 | 0.6701 | **Q3** [0.0003, 0.0016] |
| `lambda_l2` | 0.6673 | 0.6649 | 0.6653 | 0.6775 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 0.6689 | 0.6632 | 0.6655 | 0.6774 | **Q2** [0.6241, 0.6459] |
| `bagging_fraction` | 0.6782 | 0.6641 | 0.6652 | 0.6674 | **Q2** [0.8801, 0.902] |

#### E. Slice plot

![wine_quality@s43/naive-ensemble](slice_wine_quality@s43_naive-ensemble.png)


---

## wine_quality@s44  (search X=[5198, 11], holdout n=1299)


### naive-ensemble

- **holdout RMSE: 0.59391** (winner retrained in 1.20s, cv score of winner: 0.6701)
- cv best RMSE: 0.6701, median: 0.6780, p10: 0.6734
- train: median 0.630s/fold, mean 0.557s, p90 0.797s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.454 |
| `learning_rate` | 0.380 |
| `min_data_in_leaf` | 0.088 |
| `max_depth` | 0.019 |
| `feature_fraction` | 0.017 |
| `bagging_freq` | 0.015 |
| `n_models` | 0.012 |
| `num_leaves` | 0.006 |
| `lambda_l1` | 0.005 |
| `bagging_fraction` | 0.003 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 279 | 0.6813 | 0.0106 | 0.6701 |
| True | 21 | 0.7198 | 0.0259 | 0.6945 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.6902 | 0.6795 | 0.6794 | 0.6869 | **Q3** [0.0638, 0.0783] |
| `num_leaves` | 0.6900 | 0.6833 | 0.6836 | 0.6794 | **Q4** [124.0, ∞) |
| `max_depth` | 0.6981 | 0.6817 | — | 0.6818 | **Q2** [9.0, 10.0] |
| `min_data_in_leaf` | — | 0.6777 | 0.6812 | 0.6986 | **Q2** [5.0, 8.0] |
| `lambda_l1` | 0.6908 | 0.6790 | 0.6791 | 0.6872 | **Q2** [0.009, 0.0359] |
| `lambda_l2` | 0.6887 | 0.6833 | 0.6800 | 0.6841 | **Q3** [0.0032, 0.0092] |
| `feature_fraction` | 0.6804 | 0.6792 | 0.6854 | 0.6912 | **Q2** [0.5295, 0.565] |
| `bagging_fraction` | 0.6933 | 0.6829 | 0.6794 | 0.6805 | **Q3** [0.8318, 0.8602] |

#### E. Slice plot

![wine_quality@s44/naive-ensemble](slice_wine_quality@s44_naive-ensemble.png)


---

## Overall recommendations

(no categorical settings were universally significant — see per-dataset breakdown)
