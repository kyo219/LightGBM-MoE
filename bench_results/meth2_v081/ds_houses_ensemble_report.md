# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 300

- **Datasets**: ['houses'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `747bb4a7b5e6`, lib sha256 `a3c30fcc7fd3…`, package `/home/mokumoku/Develop/LightGBM-MoE/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| houses | naive-ensemble | **48923.76265** ± 850.01220 | 50476.70623 | 0.0% | 0.82 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| houses@s42 | naive-ensemble | 50578.9081 | 50981.6644 | 0.679 | 940 |
| houses@s43 | naive-ensemble | 50415.3430 | 50739.7084 | 0.734 | 1044 |
| houses@s44 | naive-ensemble | 50435.8676 | 51099.2628 | 0.659 | 861 |



---

## houses@s42  (search X=[8000, 8], holdout n=2000)


### naive-ensemble

- **holdout RMSE: 50074.59416** (winner retrained in 0.97s, cv score of winner: 50578.9081)
- cv best RMSE: 50578.9081, median: 50981.6644, p10: 50734.1221
- train: median 0.679s/fold, mean 0.623s, p90 0.805s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.536 |
| `extra_trees` | 0.388 |
| `min_data_in_leaf` | 0.038 |
| `feature_fraction` | 0.014 |
| `num_leaves` | 0.011 |
| `max_depth` | 0.009 |
| `bagging_freq` | 0.002 |
| `bagging_fraction` | 0.001 |
| `n_models` | 0.000 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 280 | 51748.6664 | 3014.1058 | 50578.9081 |
| True | 20 | 64637.5571 | 7290.6405 | 55924.7269 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 55201.3691 | 51575.4940 | 51266.5691 | 52388.2708 | **Q3** [0.0848, 0.0982] |
| `num_leaves` | 53904.5898 | 51916.7492 | 51301.5756 | 53288.6499 | **Q3** [76.0, 87.0] |
| `max_depth` | 56780.4814 | 52525.5377 | — | 51618.5296 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 51907.8658 | 51624.5413 | 52007.1930 | 54763.9636 | **Q2** [14.0, 18.5] |
| `lambda_l1` | 51361.2993 | 51987.7430 | 52142.8813 | 54939.7795 | **Q1** [None, 0.0] |
| `lambda_l2` | 52272.7789 | 53378.5415 | 52007.9284 | 52772.4543 | **Q3** [0.0, 0.0] |
| `feature_fraction` | 52483.4408 | 51419.9953 | 51532.1766 | 54996.0904 | **Q2** [0.6921, 0.7093] |
| `bagging_fraction` | 53876.8157 | 51333.9294 | 52120.6982 | 53100.2597 | **Q2** [0.8437, 0.8759] |

#### E. Slice plot

![houses@s42/naive-ensemble](slice_houses@s42_naive-ensemble.png)


---

## houses@s43  (search X=[8000, 8], holdout n=2000)


### naive-ensemble

- **holdout RMSE: 48649.13130** (winner retrained in 0.90s, cv score of winner: 50415.3430)
- cv best RMSE: 50415.3430, median: 50739.7084, p10: 50517.3258
- train: median 0.734s/fold, mean 0.693s, p90 0.980s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.564 |
| `learning_rate` | 0.347 |
| `max_depth` | 0.052 |
| `min_data_in_leaf` | 0.010 |
| `bagging_fraction` | 0.009 |
| `num_leaves` | 0.007 |
| `feature_fraction` | 0.006 |
| `bagging_freq` | 0.003 |
| `n_models` | 0.001 |
| `lambda_l1` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 278 | 51618.1532 | 2572.2672 | 50415.3430 |
| True | 22 | 63943.2345 | 5724.1284 | 55828.6091 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 54972.8666 | 51116.4459 | 51552.3147 | 52446.3430 | **Q2** [0.0665, 0.0764] |
| `num_leaves` | 53020.2347 | 51361.9096 | 51955.5208 | 53766.3171 | **Q2** [60.0, 70.5] |
| `max_depth` | 55668.8406 | 51845.0431 | — | 51891.6972 | **Q2** [10.0, 11.0] |
| `min_data_in_leaf` | 51587.5479 | 51413.4106 | 51860.5632 | 55073.7576 | **Q2** [10.0, 13.0] |
| `lambda_l1` | 52281.4770 | 52297.3752 | 51852.3108 | 53656.8071 | **Q3** [0.0, 0.0002] |
| `lambda_l2` | 53156.2125 | 51790.0639 | 51600.2993 | 53541.3944 | **Q3** [0.0, 0.0] |
| `feature_fraction` | 52495.8225 | 51417.6715 | 51701.2794 | 54473.1968 | **Q2** [0.6052, 0.6371] |
| `bagging_fraction` | 53971.5358 | 51764.9898 | 51378.1243 | 52973.3202 | **Q3** [0.8349, 0.8602] |

#### E. Slice plot

![houses@s43/naive-ensemble](slice_houses@s43_naive-ensemble.png)


---

## houses@s44  (search X=[8000, 8], holdout n=2000)


### naive-ensemble

- **holdout RMSE: 48047.56250** (winner retrained in 0.60s, cv score of winner: 50435.8676)
- cv best RMSE: 50435.8676, median: 51099.2628, p10: 50815.4997
- train: median 0.659s/fold, mean 0.571s, p90 0.764s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.566 |
| `extra_trees` | 0.400 |
| `feature_fraction` | 0.011 |
| `n_models` | 0.005 |
| `max_depth` | 0.005 |
| `num_leaves` | 0.004 |
| `min_data_in_leaf` | 0.004 |
| `bagging_fraction` | 0.003 |
| `bagging_freq` | 0.001 |
| `lambda_l2` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 280 | 51895.2195 | 3020.6784 | 50435.8676 |
| True | 20 | 62912.3473 | 6320.9872 | 57054.2326 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 54939.3252 | 51662.7108 | 51329.0514 | 52587.6911 | **Q3** [0.1114, 0.1294] |
| `num_leaves` | 52726.4566 | 51899.1814 | 52106.5290 | 53727.2554 | **Q2** [59.0, 64.0] |
| `max_depth` | 55750.8352 | 52347.7495 | — | 51792.4584 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 51594.7326 | 52217.0060 | 52249.4160 | 54272.8034 | **Q1** [None, 11.0] |
| `lambda_l1` | 51866.3206 | 52379.4503 | 51749.6752 | 54523.3324 | **Q3** [0.0, 0.0] |
| `lambda_l2` | 53868.6318 | 52993.0084 | 51778.9643 | 51878.1741 | **Q3** [0.5249, 1.6767] |
| `feature_fraction` | 54017.0045 | 51846.4190 | 51465.8423 | 53189.5128 | **Q3** [0.776, 0.7944] |
| `bagging_fraction` | 52382.4300 | 51944.4746 | 52277.0778 | 53914.7962 | **Q2** [0.7278, 0.7511] |

#### E. Slice plot

![houses@s44/naive-ensemble](slice_houses@s44_naive-ensemble.png)


---

## Overall recommendations

(no categorical settings were universally significant — see per-dataset breakdown)

