# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 500

- **Datasets**: ['vix'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `747bb4a7b5e6`, lib sha256 `a3c30fcc7fd3…`, package `/home/mokumoku/Develop/LightGBM-MoE/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| vix | naive-ensemble | **1.73943** ± 0.05112 | 2.64216 | 0.0% | 0.12 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| vix@s42 | naive-ensemble | 2.6508 | 2.7183 | 0.067 | 224 |
| vix@s43 | naive-ensemble | 2.6426 | 2.7052 | 0.123 | 383 |
| vix@s44 | naive-ensemble | 2.6331 | 2.7077 | 0.181 | 478 |



---

## vix@s42  (search X=[3011, 13], holdout n=752)


### naive-ensemble

- **holdout RMSE: 1.69842** (winner retrained in 0.07s, cv score of winner: 2.6508)
- cv best RMSE: 2.6508, median: 2.7183, p10: 2.6885
- train: median 0.067s/fold, mean 0.086s, p90 0.147s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.553 |
| `min_data_in_leaf` | 0.340 |
| `bagging_fraction` | 0.036 |
| `feature_fraction` | 0.022 |
| `extra_trees` | 0.016 |
| `max_depth` | 0.016 |
| `bagging_freq` | 0.013 |
| `num_leaves` | 0.004 |
| `n_models` | 0.001 |
| `lambda_l2` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 469 | 2.7479 | 0.1438 | 2.6508 |
| True | 31 | 2.9925 | 0.3982 | 2.7016 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.8636 | 2.7229 | 2.7255 | 2.7404 | **Q2** [0.0868, 0.1006] |
| `num_leaves` | 2.8205 | 2.7400 | 2.7437 | 2.7505 | **Q2** [80.0, 89.0] |
| `max_depth` | — | — | 2.7368 | 2.8333 | **Q3** [3.0, 4.0] |
| `min_data_in_leaf` | 2.7352 | 2.7226 | 2.7294 | 2.8555 | **Q2** [25.0, 27.0] |
| `lambda_l1` | 2.7623 | 2.7261 | 2.7874 | 2.7765 | **Q2** [0.0, 0.0] |
| `lambda_l2` | 2.7617 | 2.7361 | 2.7734 | 2.7811 | **Q2** [0.0, 0.0001] |
| `feature_fraction` | 2.7847 | 2.7610 | 2.7279 | 2.7786 | **Q3** [0.7743, 0.8111] |
| `bagging_fraction` | 2.8401 | 2.7338 | 2.7358 | 2.7426 | **Q2** [0.9019, 0.9562] |

#### E. Slice plot

![vix@s42/naive-ensemble](slice_vix@s42_naive-ensemble.png)


---

## vix@s43  (search X=[3011, 13], holdout n=752)


### naive-ensemble

- **holdout RMSE: 1.70837** (winner retrained in 0.13s, cv score of winner: 2.6426)
- cv best RMSE: 2.6426, median: 2.7052, p10: 2.6666
- train: median 0.123s/fold, mean 0.150s, p90 0.246s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.495 |
| `learning_rate` | 0.467 |
| `extra_trees` | 0.011 |
| `bagging_fraction` | 0.009 |
| `num_leaves` | 0.007 |
| `feature_fraction` | 0.004 |
| `bagging_freq` | 0.002 |
| `lambda_l2` | 0.002 |
| `max_depth` | 0.002 |
| `n_models` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 467 | 2.7302 | 0.1234 | 2.6426 |
| True | 33 | 3.0072 | 0.3737 | 2.6878 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.8245 | 2.7048 | 2.7138 | 2.7509 | **Q2** [0.0624, 0.0699] |
| `num_leaves` | 2.7716 | 2.7405 | 2.7202 | 2.7653 | **Q3** [53.0, 64.0] |
| `max_depth` | 2.7492 | — | — | 2.7483 | **Q4** [5.0, ∞) |
| `min_data_in_leaf` | 2.7481 | 2.6963 | 2.7083 | 2.8322 | **Q2** [18.0, 21.0] |
| `lambda_l1` | 2.7320 | 2.7319 | 2.7170 | 2.8131 | **Q3** [0.0, 0.0] |
| `lambda_l2` | 2.7910 | 2.7296 | 2.7289 | 2.7445 | **Q3** [0.0359, 0.0975] |
| `feature_fraction` | 2.7846 | 2.7343 | 2.7328 | 2.7422 | **Q3** [0.8128, 0.852] |
| `bagging_fraction` | 2.7907 | 2.7340 | 2.7161 | 2.7531 | **Q3** [0.7776, 0.7961] |

#### E. Slice plot

![vix@s43/naive-ensemble](slice_vix@s43_naive-ensemble.png)


---

## vix@s44  (search X=[3011, 13], holdout n=752)


### naive-ensemble

- **holdout RMSE: 1.81150** (winner retrained in 0.17s, cv score of winner: 2.6331)
- cv best RMSE: 2.6331, median: 2.7077, p10: 2.6688
- train: median 0.181s/fold, mean 0.188s, p90 0.250s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.859 |
| `min_data_in_leaf` | 0.111 |
| `bagging_fraction` | 0.013 |
| `num_leaves` | 0.007 |
| `extra_trees` | 0.005 |
| `max_depth` | 0.001 |
| `n_models` | 0.001 |
| `feature_fraction` | 0.001 |
| `lambda_l1` | 0.001 |
| `bagging_freq` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 468 | 2.7414 | 0.1586 | 2.6331 |
| False | 32 | 2.9614 | 0.2773 | 2.7407 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.8571 | 2.7134 | 2.7307 | 2.7208 | **Q2** [0.1588, 0.1929] |
| `num_leaves` | 2.7925 | 2.7255 | 2.7544 | 2.7501 | **Q2** [62.0, 69.0] |
| `max_depth` | 2.8207 | 2.7326 | — | 2.7434 | **Q2** [10.0, 11.0] |
| `min_data_in_leaf` | 2.7576 | 2.7098 | 2.7413 | 2.8087 | **Q2** [10.0, 14.0] |
| `lambda_l1` | 2.8110 | 2.7343 | 2.7366 | 2.7401 | **Q2** [0.8139, 1.8887] |
| `lambda_l2` | 2.7285 | 2.7435 | 2.7323 | 2.8178 | **Q1** [None, 0.0] |
| `feature_fraction` | 2.8208 | 2.7300 | 2.7250 | 2.7462 | **Q3** [0.9634, 0.9799] |
| `bagging_fraction` | 2.8314 | 2.7256 | 2.7409 | 2.7242 | **Q4** [0.9836, ∞) |

#### E. Slice plot

![vix@s44/naive-ensemble](slice_vix@s44_naive-ensemble.png)


---

## Overall recommendations

(no categorical settings were universally significant — see per-dataset breakdown)
