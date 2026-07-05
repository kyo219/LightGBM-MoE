# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 500

- **Datasets**: ['fred_gdp'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `747bb4a7b5e6`, lib sha256 `a3c30fcc7fd3…`, package `/home/mokumoku/Develop/LightGBM-MoE/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| fred_gdp | naive-ensemble | **1.53029** ± 0.00814 | 0.80924 | 0.0% | 0.04 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| fred_gdp@s42 | naive-ensemble | 0.8126 | 0.8301 | 0.020 | 63 |
| fred_gdp@s43 | naive-ensemble | 0.8044 | 0.8218 | 0.017 | 52 |
| fred_gdp@s44 | naive-ensemble | 0.8107 | 0.8243 | 0.019 | 57 |



---

## fred_gdp@s42  (search X=[249, 12], holdout n=62)


### naive-ensemble

- **holdout RMSE: 1.52662** (winner retrained in 0.05s, cv score of winner: 0.8126)
- cv best RMSE: 0.8126, median: 0.8301, p10: 0.8206
- train: median 0.020s/fold, mean 0.022s, p90 0.030s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.974 |
| `extra_trees` | 0.005 |
| `num_leaves` | 0.004 |
| `bagging_fraction` | 0.004 |
| `learning_rate` | 0.004 |
| `max_depth` | 0.003 |
| `bagging_freq` | 0.003 |
| `feature_fraction` | 0.002 |
| `n_models` | 0.001 |
| `lambda_l2` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 467 | 0.8304 | 0.0100 | 0.8126 |
| True | 33 | 0.8416 | 0.0107 | 0.8235 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.8305 | 0.8287 | 0.8310 | 0.8345 | **Q2** [0.0218, 0.0259] |
| `num_leaves` | 0.8332 | 0.8296 | 0.8292 | 0.8328 | **Q3** [75.0, 82.0] |
| `max_depth` | 0.8354 | 0.8309 | — | 0.8302 | **Q4** [10.0, ∞) |
| `min_data_in_leaf` | 0.8354 | 0.8274 | 0.8275 | 0.8350 | **Q2** [23.0, 26.5] |
| `lambda_l1` | 0.8346 | 0.8304 | 0.8288 | 0.8309 | **Q3** [0.0006, 0.0042] |
| `lambda_l2` | 0.8351 | 0.8301 | 0.8292 | 0.8303 | **Q3** [0.09, 0.4816] |
| `feature_fraction` | 0.8330 | 0.8305 | 0.8291 | 0.8322 | **Q3** [0.8576, 0.8882] |
| `bagging_fraction` | 0.8349 | 0.8333 | 0.8289 | 0.8275 | **Q4** [0.9907, ∞) |

#### E. Slice plot

![fred_gdp@s42/naive-ensemble](slice_fred_gdp@s42_naive-ensemble.png)


---

## fred_gdp@s43  (search X=[249, 12], holdout n=62)


### naive-ensemble

- **holdout RMSE: 1.54157** (winner retrained in 0.03s, cv score of winner: 0.8044)
- cv best RMSE: 0.8044, median: 0.8218, p10: 0.8117
- train: median 0.017s/fold, mean 0.018s, p90 0.024s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.975 |
| `bagging_fraction` | 0.009 |
| `learning_rate` | 0.007 |
| `feature_fraction` | 0.002 |
| `extra_trees` | 0.002 |
| `num_leaves` | 0.002 |
| `max_depth` | 0.002 |
| `bagging_freq` | 0.001 |
| `n_models` | 0.000 |
| `lambda_l2` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 454 | 0.8236 | 0.0124 | 0.8044 |
| True | 46 | 0.8348 | 0.0133 | 0.8095 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.8345 | 0.8236 | 0.8208 | 0.8196 | **Q4** [0.2793, ∞) |
| `num_leaves` | 0.8313 | 0.8234 | 0.8211 | 0.8230 | **Q3** [79.0, 88.0] |
| `max_depth` | — | — | 0.8217 | 0.8311 | **Q3** [3.0, 4.0] |
| `min_data_in_leaf` | 0.8315 | 0.8200 | 0.8186 | 0.8300 | **Q3** [24.0, 28.0] |
| `lambda_l1` | 0.8323 | 0.8237 | 0.8199 | 0.8227 | **Q3** [0.1003, 0.2288] |
| `lambda_l2` | 0.8262 | 0.8224 | 0.8205 | 0.8295 | **Q3** [0.0, 0.0] |
| `feature_fraction` | 0.8303 | 0.8207 | 0.8210 | 0.8267 | **Q2** [0.8615, 0.8826] |
| `bagging_fraction` | 0.8318 | 0.8215 | 0.8221 | 0.8231 | **Q2** [0.9007, 0.9421] |

#### E. Slice plot

![fred_gdp@s43/naive-ensemble](slice_fred_gdp@s43_naive-ensemble.png)


---

## fred_gdp@s44  (search X=[249, 12], holdout n=62)


### naive-ensemble

- **holdout RMSE: 1.52267** (winner retrained in 0.04s, cv score of winner: 0.8107)
- cv best RMSE: 0.8107, median: 0.8243, p10: 0.8145
- train: median 0.019s/fold, mean 0.019s, p90 0.024s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.943 |
| `learning_rate` | 0.016 |
| `extra_trees` | 0.014 |
| `bagging_fraction` | 0.007 |
| `num_leaves` | 0.006 |
| `feature_fraction` | 0.004 |
| `max_depth` | 0.003 |
| `bagging_freq` | 0.003 |
| `lambda_l2` | 0.003 |
| `n_models` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 451 | 0.8245 | 0.0095 | 0.8107 |
| True | 49 | 0.8367 | 0.0122 | 0.8166 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.8272 | 0.8225 | 0.8255 | 0.8276 | **Q2** [0.0522, 0.0624] |
| `num_leaves` | 0.8256 | 0.8248 | 0.8270 | 0.8255 | **Q2** [34.0, 62.0] |
| `max_depth` | 0.8333 | 0.8258 | 0.8232 | 0.8254 | **Q3** [10.0, 11.0] |
| `min_data_in_leaf` | 0.8303 | 0.8209 | 0.8201 | 0.8309 | **Q3** [26.0, 29.0] |
| `lambda_l1` | 0.8285 | 0.8219 | 0.8251 | 0.8274 | **Q2** [0.0, 0.0] |
| `lambda_l2` | 0.8297 | 0.8240 | 0.8226 | 0.8267 | **Q3** [0.2789, 0.7212] |
| `feature_fraction` | 0.8308 | 0.8229 | 0.8228 | 0.8264 | **Q3** [0.7157, 0.7414] |
| `bagging_fraction` | 0.8252 | 0.8236 | 0.8260 | 0.8281 | **Q2** [0.714, 0.7585] |

#### E. Slice plot

![fred_gdp@s44/naive-ensemble](slice_fred_gdp@s44_naive-ensemble.png)


---

## Overall recommendations

(no categorical settings were universally significant — see per-dataset breakdown)
