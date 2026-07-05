# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 500

- **Datasets**: ['wine_quality'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `4e208d83bace`, lib sha256 `a3c30fcc7fd3…`, package `/home/mokumoku/Develop/LightGBM-MoE/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| wine_quality | naive-ensemble | **0.60653** ± 0.01261 | 0.66314 | 0.0% | 1.45 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| wine_quality@s42 | naive-ensemble | 0.6624 | 0.6697 | 1.094 | 2500 |
| wine_quality@s43 | naive-ensemble | 0.6569 | 0.6621 | 0.867 | 1988 |
| wine_quality@s44 | naive-ensemble | 0.6701 | 0.6770 | 0.643 | 1477 |



---

## wine_quality@s42  (search X=[5198, 11], holdout n=1299)


### naive-ensemble

- **holdout RMSE: 0.60192** (winner retrained in 1.74s, cv score of winner: 0.6624)
- cv best RMSE: 0.6624, median: 0.6697, p10: 0.6661
- train: median 1.094s/fold, mean 0.997s, p90 1.467s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.403 |
| `learning_rate` | 0.370 |
| `min_data_in_leaf` | 0.141 |
| `feature_fraction` | 0.036 |
| `max_depth` | 0.029 |
| `n_models` | 0.010 |
| `bagging_fraction` | 0.004 |
| `lambda_l1` | 0.003 |
| `num_leaves` | 0.003 |
| `bagging_freq` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 470 | 0.6737 | 0.0114 | 0.6624 |
| True | 30 | 0.7097 | 0.0227 | 0.6842 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.6823 | 0.6722 | 0.6739 | 0.6752 | **Q2** [0.0532, 0.0644] |
| `num_leaves` | 0.6804 | 0.6748 | 0.6735 | 0.6750 | **Q3** [117.0, 123.0] |
| `max_depth` | 0.6889 | — | 0.6738 | 0.6719 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | — | 0.6700 | 0.6721 | 0.6887 | **Q2** [5.0, 7.0] |
| `lambda_l1` | 0.6811 | 0.6733 | 0.6727 | 0.6764 | **Q3** [0.0485, 0.1366] |
| `lambda_l2` | 0.6784 | 0.6782 | 0.6734 | 0.6737 | **Q3** [0.008, 0.0742] |
| `feature_fraction` | 0.6805 | 0.6722 | 0.6734 | 0.6775 | **Q2** [0.683, 0.7201] |
| `bagging_fraction` | 0.6764 | 0.6724 | 0.6729 | 0.6819 | **Q2** [0.6716, 0.6964] |

#### E. Slice plot

![wine_quality@s42/naive-ensemble](slice_wine_quality@s42_naive-ensemble.png)


---

## wine_quality@s43  (search X=[5198, 11], holdout n=1299)


### naive-ensemble

- **holdout RMSE: 0.62376** (winner retrained in 1.42s, cv score of winner: 0.6569)
- cv best RMSE: 0.6569, median: 0.6621, p10: 0.6594
- train: median 0.867s/fold, mean 0.792s, p90 1.125s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.466 |
| `learning_rate` | 0.245 |
| `min_data_in_leaf` | 0.197 |
| `max_depth` | 0.054 |
| `bagging_fraction` | 0.016 |
| `num_leaves` | 0.006 |
| `n_models` | 0.006 |
| `bagging_freq` | 0.006 |
| `feature_fraction` | 0.004 |
| `lambda_l1` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 0.6651 | 0.0093 | 0.6569 |
| True | 32 | 0.6987 | 0.0202 | 0.6770 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.6740 | 0.6632 | 0.6637 | 0.6682 | **Q2** [0.0858, 0.0975] |
| `num_leaves` | 0.6745 | 0.6642 | 0.6656 | 0.6653 | **Q2** [112.0, 119.0] |
| `max_depth` | 0.6754 | — | — | 0.6649 | **Q4** [11.0, ∞) |
| `min_data_in_leaf` | — | 0.6626 | 0.6642 | 0.6774 | **Q2** [5.0, 7.0] |
| `lambda_l1` | 0.6710 | 0.6656 | 0.6645 | 0.6681 | **Q3** [0.0008, 0.0027] |
| `lambda_l2` | 0.6664 | 0.6649 | 0.6669 | 0.6710 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 0.6664 | 0.6656 | 0.6641 | 0.6729 | **Q3** [0.6391, 0.6632] |
| `bagging_fraction` | 0.6732 | 0.6649 | 0.6639 | 0.6671 | **Q3** [0.898, 0.9211] |

#### E. Slice plot

![wine_quality@s43/naive-ensemble](slice_wine_quality@s43_naive-ensemble.png)


---

## wine_quality@s44  (search X=[5198, 11], holdout n=1299)


### naive-ensemble

- **holdout RMSE: 0.59391** (winner retrained in 1.18s, cv score of winner: 0.6701)
- cv best RMSE: 0.6701, median: 0.6770, p10: 0.6730
- train: median 0.643s/fold, mean 0.588s, p90 0.831s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.452 |
| `learning_rate` | 0.309 |
| `min_data_in_leaf` | 0.123 |
| `max_depth` | 0.030 |
| `feature_fraction` | 0.028 |
| `lambda_l1` | 0.017 |
| `bagging_freq` | 0.015 |
| `n_models` | 0.014 |
| `bagging_fraction` | 0.007 |
| `num_leaves` | 0.004 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 469 | 0.6799 | 0.0095 | 0.6701 |
| True | 31 | 0.7134 | 0.0234 | 0.6919 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.6870 | 0.6784 | 0.6795 | 0.6830 | **Q2** [0.0556, 0.0661] |
| `num_leaves` | 0.6875 | 0.6818 | 0.6798 | 0.6791 | **Q4** [124.0, ∞) |
| `max_depth` | 0.6885 | — | 0.6792 | 0.6817 | **Q3** [10.0, 11.0] |
| `min_data_in_leaf` | — | 0.6769 | 0.6783 | 0.6935 | **Q2** [5.0, 7.0] |
| `lambda_l1` | 0.6846 | 0.6800 | 0.6794 | 0.6838 | **Q3** [0.0257, 0.1005] |
| `lambda_l2` | 0.6871 | 0.6797 | 0.6803 | 0.6806 | **Q2** [0.0016, 0.006] |
| `feature_fraction` | 0.6785 | 0.6794 | 0.6806 | 0.6894 | **Q1** [None, 0.5214] |
| `bagging_fraction` | 0.6879 | 0.6803 | 0.6796 | 0.6800 | **Q3** [0.8354, 0.8616] |

#### E. Slice plot

![wine_quality@s44/naive-ensemble](slice_wine_quality@s44_naive-ensemble.png)


---

## Overall recommendations

(no categorical settings were universally significant — see per-dataset breakdown)

