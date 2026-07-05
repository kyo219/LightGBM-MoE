# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 500

- **Datasets**: ['elevators'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `4e208d83bace`, lib sha256 `a3c30fcc7fd3…`, package `/home/mokumoku/Develop/LightGBM-MoE/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| elevators | naive-ensemble | **0.00230** ± 0.00006 | 0.00252 | 0.0% | 0.36 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| elevators@s42 | naive-ensemble | 0.0025 | 0.0026 | 0.224 | 690 |
| elevators@s43 | naive-ensemble | 0.0025 | 0.0026 | 0.273 | 846 |
| elevators@s44 | naive-ensemble | 0.0025 | 0.0025 | 0.310 | 848 |



---

## elevators@s42  (search X=[8000, 18], holdout n=2000)


### naive-ensemble

- **holdout RMSE: 0.00238** (winner retrained in 0.48s, cv score of winner: 0.0025)
- cv best RMSE: 0.0025, median: 0.0026, p10: 0.0026
- train: median 0.224s/fold, mean 0.272s, p90 0.394s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.708 |
| `learning_rate` | 0.111 |
| `extra_trees` | 0.053 |
| `min_data_in_leaf` | 0.036 |
| `n_models` | 0.029 |
| `feature_fraction` | 0.019 |
| `max_depth` | 0.011 |
| `bagging_freq` | 0.011 |
| `num_leaves` | 0.010 |
| `bagging_fraction` | 0.010 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 0.0027 | 0.0005 | 0.0025 |
| True | 32 | 0.0032 | 0.0007 | 0.0026 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0031 | 0.0027 | 0.0026 | 0.0027 | **Q3** [0.2517, 0.2785] |
| `num_leaves` | 0.0027 | 0.0028 | 0.0027 | 0.0029 | **Q1** [None, 22.75] |
| `max_depth` | 0.0028 | — | 0.0028 | 0.0028 | **Q1** [None, 4.0] |
| `min_data_in_leaf` | 0.0027 | 0.0027 | 0.0027 | 0.0030 | **Q1** [None, 10.0] |
| `lambda_l1` | 0.0028 | 0.0027 | 0.0027 | 0.0029 | **Q2** [0.0, 0.0001] |
| `lambda_l2` | 0.0027 | 0.0028 | 0.0027 | 0.0029 | **Q1** [None, 0.0] |
| `feature_fraction` | 0.0028 | 0.0028 | 0.0027 | 0.0028 | **Q3** [0.9601, 0.9779] |
| `bagging_fraction` | 0.0028 | 0.0028 | 0.0028 | 0.0027 | **Q4** [0.9212, ∞) |

#### E. Slice plot

![elevators@s42/naive-ensemble](slice_elevators@s42_naive-ensemble.png)


---

## elevators@s43  (search X=[8000, 18], holdout n=2000)


### naive-ensemble

- **holdout RMSE: 0.00226** (winner retrained in 0.30s, cv score of winner: 0.0025)
- cv best RMSE: 0.0025, median: 0.0026, p10: 0.0026
- train: median 0.273s/fold, mean 0.335s, p90 0.563s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.819 |
| `bagging_freq` | 0.061 |
| `learning_rate` | 0.030 |
| `num_leaves` | 0.020 |
| `bagging_fraction` | 0.019 |
| `max_depth` | 0.019 |
| `feature_fraction` | 0.019 |
| `extra_trees` | 0.008 |
| `min_data_in_leaf` | 0.002 |
| `n_models` | 0.002 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 0.0027 | 0.0005 | 0.0025 |
| True | 32 | 0.0033 | 0.0008 | 0.0026 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0031 | 0.0027 | 0.0026 | 0.0027 | **Q3** [0.2375, 0.266] |
| `num_leaves` | 0.0028 | 0.0027 | 0.0027 | 0.0030 | **Q2** [12.0, 15.0] |
| `max_depth` | 0.0031 | 0.0028 | — | 0.0027 | **Q4** [11.0, ∞) |
| `min_data_in_leaf` | 0.0027 | 0.0027 | 0.0027 | 0.0030 | **Q1** [None, 7.0] |
| `lambda_l1` | 0.0027 | 0.0027 | 0.0026 | 0.0030 | **Q3** [0.0, 0.0] |
| `lambda_l2` | 0.0028 | 0.0027 | 0.0028 | 0.0028 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 0.0030 | 0.0027 | 0.0027 | 0.0027 | **Q2** [0.929, 0.9593] |
| `bagging_fraction` | 0.0028 | 0.0027 | 0.0027 | 0.0029 | **Q2** [0.7398, 0.7637] |

#### E. Slice plot

![elevators@s43/naive-ensemble](slice_elevators@s43_naive-ensemble.png)


---

## elevators@s44  (search X=[8000, 18], holdout n=2000)


### naive-ensemble

- **holdout RMSE: 0.00225** (winner retrained in 0.30s, cv score of winner: 0.0025)
- cv best RMSE: 0.0025, median: 0.0025, p10: 0.0025
- train: median 0.310s/fold, mean 0.335s, p90 0.432s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.602 |
| `learning_rate` | 0.221 |
| `bagging_fraction` | 0.069 |
| `num_leaves` | 0.030 |
| `feature_fraction` | 0.024 |
| `min_data_in_leaf` | 0.018 |
| `max_depth` | 0.014 |
| `lambda_l2` | 0.011 |
| `extra_trees` | 0.009 |
| `bagging_freq` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 470 | 0.0027 | 0.0005 | 0.0025 |
| True | 30 | 0.0033 | 0.0008 | 0.0026 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0031 | 0.0026 | 0.0026 | 0.0026 | **Q2** [0.232, 0.2637] |
| `num_leaves` | 0.0026 | 0.0027 | 0.0027 | 0.0030 | **Q1** [None, 12.0] |
| `max_depth` | 0.0029 | 0.0027 | 0.0026 | 0.0027 | **Q3** [9.0, 10.0] |
| `min_data_in_leaf` | 0.0026 | 0.0027 | 0.0027 | 0.0029 | **Q1** [None, 7.0] |
| `lambda_l1` | 0.0026 | 0.0026 | 0.0027 | 0.0030 | **Q1** [None, 0.0] |
| `lambda_l2` | 0.0029 | 0.0027 | 0.0026 | 0.0027 | **Q3** [1.2242, 3.9295] |
| `feature_fraction` | 0.0029 | 0.0026 | 0.0027 | 0.0027 | **Q2** [0.8848, 0.9175] |
| `bagging_fraction` | 0.0027 | 0.0027 | 0.0027 | 0.0028 | **Q1** [None, 0.7029] |

#### E. Slice plot

![elevators@s44/naive-ensemble](slice_elevators@s44_naive-ensemble.png)


---

## Overall recommendations

(no categorical settings were universally significant — see per-dataset breakdown)

