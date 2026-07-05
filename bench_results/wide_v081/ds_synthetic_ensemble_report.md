# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 500

- **Datasets**: ['synthetic'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `747bb4a7b5e6`, lib sha256 `a3c30fcc7fd3…`, package `/home/mokumoku/Develop/LightGBM-MoE/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| synthetic | naive-ensemble | **4.72332** ± 0.32678 | 5.42496 | 0.0% | 0.43 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| synthetic@s42 | naive-ensemble | 5.3159 | 5.6074 | 0.173 | 437 |
| synthetic@s43 | naive-ensemble | 5.4566 | 5.7179 | 0.429 | 980 |
| synthetic@s44 | naive-ensemble | 5.5024 | 5.7684 | 0.280 | 672 |



---

## synthetic@s42  (search X=[1600, 5], holdout n=400)


### naive-ensemble

- **holdout RMSE: 4.97808** (winner retrained in 0.17s, cv score of winner: 5.3159)
- cv best RMSE: 5.3159, median: 5.6074, p10: 5.4192
- train: median 0.173s/fold, mean 0.172s, p90 0.254s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.696 |
| `learning_rate` | 0.253 |
| `bagging_fraction` | 0.017 |
| `bagging_freq` | 0.014 |
| `feature_fraction` | 0.010 |
| `num_leaves` | 0.007 |
| `max_depth` | 0.002 |
| `lambda_l1` | 0.001 |
| `extra_trees` | 0.001 |
| `n_models` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 437 | 5.8014 | 0.6251 | 5.3159 |
| False | 63 | 6.0527 | 0.5984 | 5.3736 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 6.1770 | 5.6431 | 5.6511 | 5.8612 | **Q2** [0.1028, 0.118] |
| `num_leaves` | 5.7659 | 5.6967 | 5.8270 | 6.0304 | **Q2** [27.0, 32.0] |
| `max_depth` | 6.1574 | 5.8847 | 5.6745 | 5.9240 | **Q3** [9.0, 10.0] |
| `min_data_in_leaf` | — | 5.5233 | 5.6605 | 6.4879 | **Q2** [5.0, 7.0] |
| `lambda_l1` | 5.9329 | 5.9424 | 5.7070 | 5.7500 | **Q3** [0.0083, 0.0226] |
| `lambda_l2` | 5.8499 | 5.6781 | 5.7458 | 6.0584 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 6.0968 | 5.8607 | 5.7405 | 5.6343 | **Q4** [0.9811, ∞) |
| `bagging_fraction` | 6.0281 | 5.6927 | 5.6123 | 5.9992 | **Q3** [0.7759, 0.7967] |

#### E. Slice plot

![synthetic@s42/naive-ensemble](slice_synthetic@s42_naive-ensemble.png)


---

## synthetic@s43  (search X=[1600, 5], holdout n=400)


### naive-ensemble

- **holdout RMSE: 4.26202** (winner retrained in 0.53s, cv score of winner: 5.4566)
- cv best RMSE: 5.4566, median: 5.7179, p10: 5.5596
- train: median 0.429s/fold, mean 0.389s, p90 0.553s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.833 |
| `feature_fraction` | 0.055 |
| `learning_rate` | 0.029 |
| `bagging_fraction` | 0.027 |
| `num_leaves` | 0.021 |
| `max_depth` | 0.016 |
| `bagging_freq` | 0.015 |
| `extra_trees` | 0.004 |
| `n_models` | 0.001 |
| `lambda_l1` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 5.8785 | 0.4542 | 5.4566 |
| True | 32 | 6.5740 | 0.9689 | 5.6650 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 6.0646 | 5.8554 | 5.8343 | 5.9378 | **Q3** [0.0442, 0.0551] |
| `num_leaves` | 5.9562 | 5.7972 | 5.8913 | 6.0475 | **Q2** [46.0, 53.0] |
| `max_depth` | 6.5456 | 5.9124 | 5.8932 | 5.8209 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | — | 5.7131 | 5.7993 | 6.3985 | **Q2** [5.0, 7.0] |
| `lambda_l1` | 5.9598 | 5.8026 | 5.9442 | 5.9855 | **Q2** [0.0014, 0.0047] |
| `lambda_l2` | 5.9276 | 5.8550 | 5.8819 | 6.0276 | **Q2** [0.0, 0.0002] |
| `feature_fraction` | 6.2281 | 5.7835 | 5.7656 | 5.9150 | **Q3** [0.9281, 0.9523] |
| `bagging_fraction` | 6.0150 | 5.8623 | 5.8120 | 6.0029 | **Q3** [0.8396, 0.8803] |

#### E. Slice plot

![synthetic@s43/naive-ensemble](slice_synthetic@s43_naive-ensemble.png)


---

## synthetic@s44  (search X=[1600, 5], holdout n=400)


### naive-ensemble

- **holdout RMSE: 4.92985** (winner retrained in 0.59s, cv score of winner: 5.5024)
- cv best RMSE: 5.5024, median: 5.7684, p10: 5.5967
- train: median 0.280s/fold, mean 0.266s, p90 0.397s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.626 |
| `learning_rate` | 0.206 |
| `feature_fraction` | 0.088 |
| `n_models` | 0.027 |
| `bagging_fraction` | 0.016 |
| `max_depth` | 0.015 |
| `num_leaves` | 0.012 |
| `extra_trees` | 0.005 |
| `bagging_freq` | 0.002 |
| `lambda_l2` | 0.002 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 413 | 5.8941 | 0.4930 | 5.5024 |
| True | 87 | 6.4892 | 0.6728 | 5.8148 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 6.0287 | 5.9279 | 5.9387 | 6.0953 | **Q2** [0.041, 0.0703] |
| `num_leaves` | 6.3999 | 5.9567 | 5.8209 | 5.8324 | **Q3** [105.0, 117.0] |
| `max_depth` | 6.5329 | 5.8996 | 5.9648 | 5.9191 | **Q2** [8.0, 9.0] |
| `min_data_in_leaf` | — | 5.7539 | 5.8150 | 6.5983 | **Q2** [5.0, 7.0] |
| `lambda_l1` | 6.1981 | 5.9757 | 5.9435 | 5.8734 | **Q4** [0.7269, ∞) |
| `lambda_l2` | 5.9886 | 5.8508 | 5.9147 | 6.2366 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 6.4315 | 5.7970 | 5.8457 | 5.9164 | **Q2** [0.9078, 0.9304] |
| `bagging_fraction` | 5.9992 | 6.0241 | 5.9536 | 6.0137 | **Q3** [0.8284, 0.8945] |

#### E. Slice plot

![synthetic@s44/naive-ensemble](slice_synthetic@s44_naive-ensemble.png)


---

## Overall recommendations

(no categorical settings were universally significant — see per-dataset breakdown)
