# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 300

- **Datasets**: ['elevators'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `747bb4a7b5e6`, lib sha256 `a3c30fcc7fd3…`, package `/home/mokumoku/Develop/LightGBM-MoE/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| elevators | naive-ensemble | **0.00230** ± 0.00009 | 0.00252 | 0.0% | 0.28 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| elevators@s42 | naive-ensemble | 0.0026 | 0.0026 | 0.206 | 388 |
| elevators@s43 | naive-ensemble | 0.0025 | 0.0026 | 0.264 | 518 |
| elevators@s44 | naive-ensemble | 0.0025 | 0.0026 | 0.283 | 444 |



---

## elevators@s42  (search X=[8000, 18], holdout n=2000)


### naive-ensemble

- **holdout RMSE: 0.00242** (winner retrained in 0.24s, cv score of winner: 0.0026)
- cv best RMSE: 0.0026, median: 0.0026, p10: 0.0026
- train: median 0.206s/fold, mean 0.256s, p90 0.395s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.639 |
| `learning_rate` | 0.139 |
| `extra_trees` | 0.060 |
| `feature_fraction` | 0.041 |
| `min_data_in_leaf` | 0.035 |
| `max_depth` | 0.023 |
| `bagging_fraction` | 0.022 |
| `n_models` | 0.019 |
| `bagging_freq` | 0.015 |
| `lambda_l2` | 0.004 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 278 | 0.0028 | 0.0005 | 0.0026 |
| True | 22 | 0.0034 | 0.0008 | 0.0026 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0032 | 0.0027 | 0.0027 | 0.0026 | **Q4** [0.2822, ∞) |
| `num_leaves` | 0.0028 | 0.0027 | 0.0030 | 0.0028 | **Q2** [33.0, 54.0] |
| `max_depth` | 0.0028 | — | 0.0027 | 0.0030 | **Q3** [4.0, 5.0] |
| `min_data_in_leaf` | 0.0027 | 0.0027 | 0.0027 | 0.0031 | **Q1** [None, 12.0] |
| `lambda_l1` | 0.0028 | 0.0027 | 0.0028 | 0.0030 | **Q2** [0.0, 0.0] |
| `lambda_l2` | 0.0027 | 0.0028 | 0.0028 | 0.0029 | **Q1** [None, 0.0] |
| `feature_fraction` | 0.0029 | 0.0028 | 0.0028 | 0.0028 | **Q2** [0.9137, 0.9504] |
| `bagging_fraction` | 0.0030 | 0.0028 | 0.0027 | 0.0027 | **Q3** [0.9136, 0.9337] |

#### E. Slice plot

![elevators@s42/naive-ensemble](slice_elevators@s42_naive-ensemble.png)


---

## elevators@s43  (search X=[8000, 18], holdout n=2000)


### naive-ensemble

- **holdout RMSE: 0.00223** (winner retrained in 0.29s, cv score of winner: 0.0025)
- cv best RMSE: 0.0025, median: 0.0026, p10: 0.0026
- train: median 0.264s/fold, mean 0.342s, p90 0.643s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.390 |
| `bagging_freq` | 0.262 |
| `learning_rate` | 0.088 |
| `num_leaves` | 0.055 |
| `n_models` | 0.050 |
| `feature_fraction` | 0.040 |
| `bagging_fraction` | 0.038 |
| `min_data_in_leaf` | 0.037 |
| `max_depth` | 0.026 |
| `extra_trees` | 0.012 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 278 | 0.0027 | 0.0005 | 0.0025 |
| True | 22 | 0.0035 | 0.0009 | 0.0026 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0032 | 0.0027 | 0.0027 | 0.0026 | **Q4** [0.2642, ∞) |
| `num_leaves` | 0.0027 | 0.0027 | 0.0028 | 0.0031 | **Q1** [None, 12.0] |
| `max_depth` | 0.0031 | — | — | 0.0027 | **Q4** [11.0, ∞) |
| `min_data_in_leaf` | 0.0028 | 0.0026 | 0.0027 | 0.0031 | **Q2** [7.0, 10.0] |
| `lambda_l1` | 0.0027 | 0.0027 | 0.0026 | 0.0032 | **Q3** [0.0, 0.0] |
| `lambda_l2` | 0.0029 | 0.0028 | 0.0027 | 0.0028 | **Q3** [0.0001, 0.0004] |
| `feature_fraction` | 0.0031 | 0.0028 | 0.0026 | 0.0027 | **Q3** [0.9516, 0.9744] |
| `bagging_fraction` | 0.0029 | 0.0027 | 0.0027 | 0.0029 | **Q2** [0.7346, 0.7667] |

#### E. Slice plot

![elevators@s43/naive-ensemble](slice_elevators@s43_naive-ensemble.png)


---

## elevators@s44  (search X=[8000, 18], holdout n=2000)


### naive-ensemble

- **holdout RMSE: 0.00225** (winner retrained in 0.31s, cv score of winner: 0.0025)
- cv best RMSE: 0.0025, median: 0.0026, p10: 0.0025
- train: median 0.283s/fold, mean 0.293s, p90 0.398s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.505 |
| `learning_rate` | 0.334 |
| `bagging_fraction` | 0.072 |
| `num_leaves` | 0.040 |
| `min_data_in_leaf` | 0.018 |
| `extra_trees` | 0.014 |
| `max_depth` | 0.006 |
| `n_models` | 0.006 |
| `feature_fraction` | 0.002 |
| `bagging_freq` | 0.002 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 280 | 0.0027 | 0.0005 | 0.0025 |
| True | 20 | 0.0035 | 0.0008 | 0.0026 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0032 | 0.0027 | 0.0026 | 0.0026 | **Q3** [0.255, 0.2789] |
| `num_leaves` | 0.0026 | 0.0026 | 0.0028 | 0.0030 | **Q1** [None, 12.0] |
| `max_depth` | 0.0029 | 0.0028 | 0.0027 | 0.0027 | **Q3** [7.0, 9.0] |
| `min_data_in_leaf` | 0.0026 | 0.0027 | 0.0028 | 0.0029 | **Q1** [None, 7.0] |
| `lambda_l1` | 0.0027 | 0.0026 | 0.0026 | 0.0031 | **Q2** [0.0, 0.0] |
| `lambda_l2` | 0.0029 | 0.0028 | 0.0027 | 0.0026 | **Q4** [1.8467, ∞) |
| `feature_fraction` | 0.0030 | 0.0026 | 0.0027 | 0.0027 | **Q2** [0.873, 0.9218] |
| `bagging_fraction` | 0.0028 | 0.0027 | 0.0027 | 0.0028 | **Q2** [0.7538, 0.7975] |

#### E. Slice plot

![elevators@s44/naive-ensemble](slice_elevators@s44_naive-ensemble.png)


---

## Overall recommendations

(no categorical settings were universally significant — see per-dataset breakdown)
