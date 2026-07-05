# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 500

- **Datasets**: ['cpu_act'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `4e208d83bace`, lib sha256 `a3c30fcc7fd3…`, package `/home/mokumoku/Develop/LightGBM-MoE/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| cpu_act | naive-ensemble | **2.44848** ± 0.33583 | 2.41525 | 0.0% | 0.41 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| cpu_act@s42 | naive-ensemble | 2.4889 | 2.5387 | 0.431 | 1168 |
| cpu_act@s43 | naive-ensemble | 2.3056 | 2.3412 | 0.422 | 1084 |
| cpu_act@s44 | naive-ensemble | 2.4512 | 2.4980 | 0.305 | 819 |



---

## cpu_act@s42  (search X=[6554, 21], holdout n=1638)


### naive-ensemble

- **holdout RMSE: 2.17111** (winner retrained in 0.35s, cv score of winner: 2.4889)
- cv best RMSE: 2.4889, median: 2.5387, p10: 2.5076
- train: median 0.431s/fold, mean 0.464s, p90 0.644s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.450 |
| `min_data_in_leaf` | 0.367 |
| `extra_trees` | 0.115 |
| `feature_fraction` | 0.035 |
| `num_leaves` | 0.014 |
| `bagging_fraction` | 0.006 |
| `bagging_freq` | 0.005 |
| `n_models` | 0.005 |
| `max_depth` | 0.002 |
| `lambda_l2` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 469 | 2.6868 | 0.5755 | 2.4889 |
| True | 31 | 3.9209 | 1.7827 | 2.9891 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 3.1156 | 2.6255 | 2.6588 | 2.6534 | **Q2** [0.1296, 0.1597] |
| `num_leaves` | 2.6483 | 2.6765 | 2.6492 | 3.0698 | **Q1** [None, 17.0] |
| `max_depth` | 3.0887 | 2.8398 | 2.6972 | 2.6731 | **Q4** [11.0, ∞) |
| `min_data_in_leaf` | — | 2.6012 | 2.6379 | 3.1572 | **Q2** [5.0, 7.0] |
| `lambda_l1` | 2.8631 | 2.6670 | 2.6997 | 2.8233 | **Q2** [0.0004, 0.0011] |
| `lambda_l2` | 3.0140 | 2.6876 | 2.6504 | 2.7012 | **Q3** [0.0158, 0.0433] |
| `feature_fraction` | 2.7862 | 2.6790 | 2.6354 | 2.9526 | **Q3** [0.652, 0.6791] |
| `bagging_fraction` | 2.6772 | 2.7591 | 2.7279 | 2.8890 | **Q1** [None, 0.5312] |

#### E. Slice plot

![cpu_act@s42/naive-ensemble](slice_cpu_act@s42_naive-ensemble.png)


---

## cpu_act@s43  (search X=[6554, 21], holdout n=1638)


### naive-ensemble

- **holdout RMSE: 2.92104** (winner retrained in 0.50s, cv score of winner: 2.3056)
- cv best RMSE: 2.3056, median: 2.3412, p10: 2.3170
- train: median 0.422s/fold, mean 0.430s, p90 0.576s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.610 |
| `learning_rate` | 0.286 |
| `extra_trees` | 0.035 |
| `bagging_fraction` | 0.021 |
| `num_leaves` | 0.018 |
| `bagging_freq` | 0.016 |
| `feature_fraction` | 0.009 |
| `max_depth` | 0.002 |
| `lambda_l2` | 0.002 |
| `n_models` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 2.4838 | 0.5328 | 2.3056 |
| True | 32 | 3.6754 | 1.6598 | 2.6196 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.8752 | 2.4080 | 2.4155 | 2.5418 | **Q2** [0.1506, 0.1733] |
| `num_leaves` | 2.5635 | 2.5013 | 2.4581 | 2.7206 | **Q3** [40.0, 48.0] |
| `max_depth` | 2.6555 | — | — | 2.5473 | **Q4** [5.0, ∞) |
| `min_data_in_leaf` | — | 2.3887 | 2.4588 | 2.9370 | **Q2** [5.0, 8.0] |
| `lambda_l1` | 2.4677 | 2.5515 | 2.6585 | 2.5627 | **Q1** [None, 0.0] |
| `lambda_l2` | 2.5512 | 2.5146 | 2.4890 | 2.6856 | **Q3** [0.0, 0.0021] |
| `feature_fraction` | 2.9020 | 2.4195 | 2.4396 | 2.4793 | **Q2** [0.9381, 0.9634] |
| `bagging_fraction` | 2.7924 | 2.5251 | 2.4210 | 2.5019 | **Q3** [0.886, 0.909] |

#### E. Slice plot

![cpu_act@s43/naive-ensemble](slice_cpu_act@s43_naive-ensemble.png)


---

## cpu_act@s44  (search X=[6554, 21], holdout n=1638)


### naive-ensemble

- **holdout RMSE: 2.25330** (winner retrained in 0.37s, cv score of winner: 2.4512)
- cv best RMSE: 2.4512, median: 2.4980, p10: 2.4684
- train: median 0.305s/fold, mean 0.324s, p90 0.459s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.385 |
| `learning_rate` | 0.379 |
| `bagging_freq` | 0.091 |
| `extra_trees` | 0.073 |
| `max_depth` | 0.026 |
| `num_leaves` | 0.019 |
| `feature_fraction` | 0.011 |
| `bagging_fraction` | 0.009 |
| `lambda_l2` | 0.004 |
| `n_models` | 0.002 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 455 | 2.6740 | 0.6567 | 2.4512 |
| True | 45 | 3.5669 | 1.1725 | 2.8613 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 3.3075 | 2.5685 | 2.5339 | 2.6076 | **Q3** [0.2038, 0.2343] |
| `num_leaves` | 2.6290 | 2.6242 | 2.6454 | 3.1069 | **Q2** [15.75, 20.0] |
| `max_depth` | 3.1408 | 2.6535 | 2.6529 | 2.6485 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 2.6126 | 2.5798 | 2.5548 | 3.2649 | **Q3** [8.0, 12.0] |
| `lambda_l1` | 2.9587 | 2.7398 | 2.7198 | 2.5990 | **Q4** [0.9857, ∞) |
| `lambda_l2` | 2.7626 | 2.8383 | 2.6597 | 2.7568 | **Q3** [0.0024, 0.0185] |
| `feature_fraction` | 2.7468 | 2.6178 | 2.7558 | 2.8970 | **Q2** [0.6719, 0.7013] |
| `bagging_fraction` | 2.7468 | 2.6598 | 2.6411 | 2.9698 | **Q3** [0.7346, 0.7607] |

#### E. Slice plot

![cpu_act@s44/naive-ensemble](slice_cpu_act@s44_naive-ensemble.png)


---

## Overall recommendations

(no categorical settings were universally significant — see per-dataset breakdown)

