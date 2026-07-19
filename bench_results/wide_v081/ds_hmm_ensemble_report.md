# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 500

- **Datasets**: ['hmm'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `747bb4a7b5e6`, lib sha256 `a3c30fcc7fd3…`, package `/home/mokumoku/Develop/LightGBM-MoE/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| hmm | naive-ensemble | **2.20801** ± 0.24193 | 2.26486 | 0.0% | 0.11 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| hmm@s42 | naive-ensemble | 2.2955 | 2.3133 | 0.071 | 191 |
| hmm@s43 | naive-ensemble | 2.3714 | 2.3870 | 0.085 | 243 |
| hmm@s44 | naive-ensemble | 2.1277 | 2.1354 | 0.079 | 213 |



---

## hmm@s42  (search X=[1600, 5], holdout n=400)


### naive-ensemble

- **holdout RMSE: 1.88558** (winner retrained in 0.10s, cv score of winner: 2.2955)
- cv best RMSE: 2.2955, median: 2.3133, p10: 2.3049
- train: median 0.071s/fold, mean 0.073s, p90 0.097s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.466 |
| `learning_rate` | 0.269 |
| `min_data_in_leaf` | 0.090 |
| `feature_fraction` | 0.060 |
| `bagging_fraction` | 0.042 |
| `max_depth` | 0.037 |
| `num_leaves` | 0.024 |
| `bagging_freq` | 0.006 |
| `lambda_l1` | 0.005 |
| `lambda_l2` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 466 | 2.3151 | 0.0133 | 2.2955 |
| False | 34 | 2.3503 | 0.0160 | 2.3146 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.3271 | 2.3147 | 2.3138 | 2.3143 | **Q3** [0.2563, 0.2789] |
| `num_leaves` | 2.3211 | 2.3142 | 2.3143 | 2.3197 | **Q2** [94.75, 100.0] |
| `max_depth` | 2.3271 | 2.3208 | — | 2.3152 | **Q4** [11.0, ∞) |
| `min_data_in_leaf` | 2.3206 | 2.3162 | 2.3130 | 2.3210 | **Q3** [30.0, 34.0] |
| `lambda_l1` | 2.3128 | 2.3165 | 2.3201 | 2.3205 | **Q1** [None, 0.0] |
| `lambda_l2` | 2.3194 | 2.3195 | 2.3161 | 2.3150 | **Q4** [0.0624, ∞) |
| `feature_fraction` | 2.3227 | 2.3144 | 2.3155 | 2.3173 | **Q2** [0.7238, 0.8281] |
| `bagging_fraction` | 2.3225 | 2.3153 | 2.3139 | 2.3182 | **Q3** [0.836, 0.8524] |

#### E. Slice plot

![hmm@s42/naive-ensemble](slice_hmm@s42_naive-ensemble.png)


---

## hmm@s43  (search X=[1600, 5], holdout n=400)


### naive-ensemble

- **holdout RMSE: 2.46832** (winner retrained in 0.12s, cv score of winner: 2.3714)
- cv best RMSE: 2.3714, median: 2.3870, p10: 2.3786
- train: median 0.085s/fold, mean 0.094s, p90 0.150s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.624 |
| `learning_rate` | 0.183 |
| `min_data_in_leaf` | 0.147 |
| `bagging_fraction` | 0.011 |
| `num_leaves` | 0.009 |
| `lambda_l1` | 0.008 |
| `max_depth` | 0.006 |
| `bagging_freq` | 0.005 |
| `feature_fraction` | 0.004 |
| `n_models` | 0.002 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 469 | 2.3904 | 0.0168 | 2.3714 |
| False | 31 | 2.4579 | 0.0271 | 2.4190 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.4056 | 2.3905 | 2.3905 | 2.3918 | **Q2** [0.2092, 0.248] |
| `num_leaves` | 2.3930 | 2.3936 | 2.3916 | 2.3994 | **Q3** [24.0, 29.0] |
| `max_depth` | 2.4069 | 2.3950 | — | 2.3914 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 2.3983 | 2.3886 | 2.3903 | 2.4035 | **Q2** [7.0, 9.0] |
| `lambda_l1` | 2.4100 | 2.3921 | 2.3872 | 2.3891 | **Q3** [3.7629, 5.7976] |
| `lambda_l2` | 2.3949 | 2.3903 | 2.3933 | 2.4000 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 2.4005 | 2.3949 | 2.3915 | 2.3916 | **Q3** [0.9684, 0.9836] |
| `bagging_fraction` | 2.3939 | 2.3973 | 2.3905 | 2.3968 | **Q3** [0.7394, 0.7662] |

#### E. Slice plot

![hmm@s43/naive-ensemble](slice_hmm@s43_naive-ensemble.png)


---

## hmm@s44  (search X=[1600, 5], holdout n=400)


### naive-ensemble

- **holdout RMSE: 2.27014** (winner retrained in 0.10s, cv score of winner: 2.1277)
- cv best RMSE: 2.1277, median: 2.1354, p10: 2.1308
- train: median 0.079s/fold, mean 0.082s, p90 0.103s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.441 |
| `learning_rate` | 0.259 |
| `min_data_in_leaf` | 0.207 |
| `lambda_l1` | 0.028 |
| `num_leaves` | 0.016 |
| `feature_fraction` | 0.015 |
| `bagging_fraction` | 0.014 |
| `max_depth` | 0.010 |
| `bagging_freq` | 0.010 |
| `n_models` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 468 | 2.1389 | 0.0134 | 2.1277 |
| False | 32 | 2.1809 | 0.0195 | 2.1591 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.1526 | 2.1387 | 2.1361 | 2.1391 | **Q3** [0.1463, 0.168] |
| `num_leaves` | 2.1407 | 2.1408 | 2.1390 | 2.1457 | **Q3** [62.0, 69.0] |
| `max_depth` | — | — | 2.1380 | 2.1479 | **Q3** [3.0, 4.0] |
| `min_data_in_leaf` | 2.1375 | 2.1358 | 2.1384 | 2.1525 | **Q2** [7.0, 9.0] |
| `lambda_l1` | 2.1494 | 2.1398 | 2.1386 | 2.1386 | **Q3** [3.38, 5.9232] |
| `lambda_l2` | 2.1398 | 2.1376 | 2.1419 | 2.1471 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 2.1460 | 2.1413 | 2.1396 | 2.1395 | **Q4** [0.8134, ∞) |
| `bagging_fraction` | 2.1461 | 2.1394 | 2.1385 | 2.1425 | **Q3** [0.7137, 0.7534] |

#### E. Slice plot

![hmm@s44/naive-ensemble](slice_hmm@s44_naive-ensemble.png)


---

## Overall recommendations

(no categorical settings were universally significant — see per-dataset breakdown)
