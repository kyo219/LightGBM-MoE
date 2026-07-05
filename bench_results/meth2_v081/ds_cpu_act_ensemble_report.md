# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 300

- **Datasets**: ['cpu_act'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `747bb4a7b5e6`, lib sha256 `a3c30fcc7fd3…`, package `/home/mokumoku/Develop/LightGBM-MoE/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| cpu_act | naive-ensemble | **2.41869** ± 0.28385 | 2.41602 | 0.0% | 0.38 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| cpu_act@s42 | naive-ensemble | 2.4889 | 2.5477 | 0.435 | 702 |
| cpu_act@s43 | naive-ensemble | 2.3073 | 2.3422 | 0.426 | 653 |
| cpu_act@s44 | naive-ensemble | 2.4518 | 2.5092 | 0.308 | 482 |



---

## cpu_act@s42  (search X=[6554, 21], holdout n=1638)


### naive-ensemble

- **holdout RMSE: 2.17111** (winner retrained in 0.31s, cv score of winner: 2.4889)
- cv best RMSE: 2.4889, median: 2.5477, p10: 2.5078
- train: median 0.435s/fold, mean 0.465s, p90 0.669s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.521 |
| `min_data_in_leaf` | 0.344 |
| `extra_trees` | 0.072 |
| `feature_fraction` | 0.019 |
| `num_leaves` | 0.018 |
| `n_models` | 0.018 |
| `bagging_fraction` | 0.004 |
| `max_depth` | 0.003 |
| `bagging_freq` | 0.001 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 280 | 2.7304 | 0.7005 | 2.4889 |
| True | 20 | 4.3699 | 2.0869 | 3.0027 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 3.3423 | 2.6228 | 2.7259 | 2.6678 | **Q2** [0.1142, 0.1418] |
| `num_leaves` | 2.6300 | 2.7092 | 2.7932 | 3.2001 | **Q1** [None, 18.0] |
| `max_depth` | 3.1646 | 2.9631 | 2.7677 | 2.6904 | **Q4** [11.0, ∞) |
| `min_data_in_leaf` | — | 2.5932 | 2.7208 | 3.3845 | **Q2** [5.0, 8.0] |
| `lambda_l1` | 3.0472 | 2.6804 | 2.6948 | 2.9363 | **Q2** [0.0003, 0.0009] |
| `lambda_l2` | 3.2172 | 2.7150 | 2.7123 | 2.7144 | **Q3** [0.0196, 0.0608] |
| `feature_fraction` | 2.8681 | 2.6691 | 2.6433 | 3.1783 | **Q3** [0.6534, 0.6944] |
| `bagging_fraction` | 2.6581 | 2.8491 | 2.7085 | 3.1431 | **Q1** [None, 0.5274] |

#### E. Slice plot

![cpu_act@s42/naive-ensemble](slice_cpu_act@s42_naive-ensemble.png)


---

## cpu_act@s43  (search X=[6554, 21], holdout n=1638)


### naive-ensemble

- **holdout RMSE: 2.81612** (winner retrained in 0.46s, cv score of winner: 2.3073)
- cv best RMSE: 2.3073, median: 2.3422, p10: 2.3200
- train: median 0.426s/fold, mean 0.432s, p90 0.597s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.647 |
| `learning_rate` | 0.245 |
| `num_leaves` | 0.034 |
| `extra_trees` | 0.024 |
| `n_models` | 0.017 |
| `bagging_fraction` | 0.010 |
| `feature_fraction` | 0.010 |
| `bagging_freq` | 0.007 |
| `max_depth` | 0.005 |
| `lambda_l2` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 278 | 2.4998 | 0.5775 | 2.3073 |
| True | 22 | 4.0691 | 1.8553 | 2.6196 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 3.0647 | 2.3818 | 2.4427 | 2.5704 | **Q2** [0.1464, 0.1718] |
| `num_leaves` | 2.5938 | 2.4571 | 2.4348 | 2.9509 | **Q3** [41.0, 47.0] |
| `max_depth` | 2.8272 | — | 2.4980 | 2.7838 | **Q3** [5.0, 6.0] |
| `min_data_in_leaf` | 2.3959 | 2.3641 | 2.4158 | 3.2625 | **Q2** [6.0, 8.0] |
| `lambda_l1` | 2.4488 | 2.4472 | 2.7096 | 2.8539 | **Q2** [0.0, 0.0] |
| `lambda_l2` | 2.6984 | 2.4899 | 2.5132 | 2.7581 | **Q2** [0.0, 0.0004] |
| `feature_fraction` | 3.1242 | 2.4031 | 2.4788 | 2.4535 | **Q2** [0.9266, 0.9606] |
| `bagging_fraction` | 3.0158 | 2.4907 | 2.4218 | 2.5313 | **Q3** [0.8868, 0.9196] |

#### E. Slice plot

![cpu_act@s43/naive-ensemble](slice_cpu_act@s43_naive-ensemble.png)


---

## cpu_act@s44  (search X=[6554, 21], holdout n=1638)


### naive-ensemble

- **holdout RMSE: 2.26882** (winner retrained in 0.36s, cv score of winner: 2.4518)
- cv best RMSE: 2.4518, median: 2.5092, p10: 2.4700
- train: median 0.308s/fold, mean 0.318s, p90 0.455s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.439 |
| `min_data_in_leaf` | 0.352 |
| `bagging_freq` | 0.086 |
| `num_leaves` | 0.051 |
| `extra_trees` | 0.026 |
| `bagging_fraction` | 0.019 |
| `feature_fraction` | 0.012 |
| `max_depth` | 0.006 |
| `lambda_l2` | 0.004 |
| `n_models` | 0.003 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 265 | 2.7034 | 0.6832 | 2.4518 |
| True | 35 | 3.6304 | 1.2635 | 2.8613 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 3.4669 | 2.6311 | 2.5181 | 2.6301 | **Q3** [0.2026, 0.2344] |
| `num_leaves` | 2.6019 | 2.5750 | 2.7753 | 3.2725 | **Q2** [16.0, 21.0] |
| `max_depth` | 3.3737 | 2.7165 | 2.6298 | 2.6565 | **Q3** [11.0, 12.0] |
| `min_data_in_leaf` | — | 2.5512 | 2.5989 | 3.5000 | **Q2** [5.0, 8.0] |
| `lambda_l1` | 3.2911 | 2.6336 | 2.7083 | 2.6131 | **Q4** [1.3908, ∞) |
| `lambda_l2` | 2.9047 | 2.7957 | 2.7870 | 2.7589 | **Q4** [0.0954, ∞) |
| `feature_fraction` | 2.7702 | 2.7246 | 2.8473 | 2.9041 | **Q2** [0.6905, 0.7326] |
| `bagging_fraction` | 2.7722 | 2.6339 | 2.6351 | 3.2050 | **Q2** [0.7154, 0.7443] |

#### E. Slice plot

![cpu_act@s44/naive-ensemble](slice_cpu_act@s44_naive-ensemble.png)


---

## Overall recommendations

(no categorical settings were universally significant — see per-dataset breakdown)

