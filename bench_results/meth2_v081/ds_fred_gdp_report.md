# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 300

- **Datasets**: ['fred_gdp'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `0cf3634c544c`, lib sha256 `a3c30fcc7fd3…`, package `/home/mokumoku/Develop/LightGBM-MoE/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| fred_gdp | naive-lightgbm | **1.52767** ± 0.01959 | 0.80518 | 0.0% | 0.02 |
| fred_gdp | naive-ensemble | **1.54257** ± 0.00696 | 0.80904 | 0.0% | 0.06 |
| fred_gdp | moe | **1.48863** ± 0.01429 | 0.81671 | 0.0% | 0.95 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| fred_gdp@s42 | naive-lightgbm | 0.8046 | 0.8281 | 0.013 | 25 |
| fred_gdp@s42 | naive-ensemble | 0.8127 | 0.8286 | 0.041 | 67 |
| fred_gdp@s42 | moe | 0.8247 | 0.9069 | 0.150 | 282 |
| fred_gdp@s43 | naive-lightgbm | 0.7992 | 0.8220 | 0.016 | 30 |
| fred_gdp@s43 | naive-ensemble | 0.8044 | 0.8233 | 0.025 | 47 |
| fred_gdp@s43 | moe | 0.8110 | 0.8575 | 0.129 | 218 |
| fred_gdp@s44 | naive-lightgbm | 0.8117 | 0.8302 | 0.015 | 30 |
| fred_gdp@s44 | naive-ensemble | 0.8101 | 0.8252 | 0.028 | 50 |
| fred_gdp@s44 | moe | 0.8144 | 0.8929 | 0.120 | 987 |



---

## fred_gdp@s42  (search X=[249, 12], holdout n=62)


### naive-lightgbm

- **holdout RMSE: 1.50612** (winner retrained in 0.02s, cv score of winner: 0.8046)
- cv best RMSE: 0.8046, median: 0.8281, p10: 0.8113
- train: median 0.013s/fold, mean 0.013s, p90 0.017s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.868 |
| `learning_rate` | 0.041 |
| `max_depth` | 0.022 |
| `feature_fraction` | 0.017 |
| `bagging_fraction` | 0.017 |
| `bagging_freq` | 0.016 |
| `extra_trees` | 0.010 |
| `num_leaves` | 0.008 |
| `lambda_l2` | 0.000 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 278 | 0.8276 | 0.0146 | 0.8046 |
| True | 22 | 0.8415 | 0.0103 | 0.8237 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.8370 | 0.8285 | 0.8227 | 0.8264 | **Q3** [0.2513, 0.2747] |
| `num_leaves` | 0.8229 | 0.8247 | 0.8300 | 0.8367 | **Q1** [None, 12.0] |
| `max_depth` | 0.8319 | 0.8270 | 0.8239 | 0.8327 | **Q3** [7.0, 8.0] |
| `min_data_in_leaf` | 0.8392 | 0.8207 | 0.8220 | 0.8337 | **Q2** [20.0, 23.0] |
| `lambda_l1` | 0.8320 | 0.8254 | 0.8293 | 0.8279 | **Q2** [0.0, 0.0001] |
| `lambda_l2` | 0.8262 | 0.8260 | 0.8294 | 0.8330 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 0.8354 | 0.8244 | 0.8248 | 0.8301 | **Q2** [0.8639, 0.892] |
| `bagging_fraction` | 0.8319 | 0.8227 | 0.8258 | 0.8343 | **Q2** [0.7971, 0.8357] |

#### E. Slice plot

![fred_gdp@s42/naive-lightgbm](slice_fred_gdp@s42_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 1.55155** (winner retrained in 0.08s, cv score of winner: 0.8127)
- cv best RMSE: 0.8127, median: 0.8286, p10: 0.8164
- train: median 0.041s/fold, mean 0.041s, p90 0.052s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.904 |
| `num_leaves` | 0.027 |
| `bagging_fraction` | 0.023 |
| `bagging_freq` | 0.020 |
| `extra_trees` | 0.008 |
| `learning_rate` | 0.005 |
| `max_depth` | 0.004 |
| `feature_fraction` | 0.004 |
| `lambda_l2` | 0.004 |
| `n_models` | 0.002 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 279 | 0.8284 | 0.0114 | 0.8127 |
| True | 21 | 0.8417 | 0.0145 | 0.8194 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.8336 | 0.8293 | 0.8236 | 0.8309 | **Q3** [0.1523, 0.188] |
| `num_leaves` | 0.8360 | 0.8348 | 0.8256 | 0.8215 | **Q4** [125.0, ∞) |
| `max_depth` | 0.8344 | 0.8354 | 0.8338 | 0.8232 | **Q4** [11.0, ∞) |
| `min_data_in_leaf` | 0.8355 | 0.8253 | 0.8231 | 0.8340 | **Q3** [24.0, 28.0] |
| `lambda_l1` | 0.8325 | 0.8353 | 0.8252 | 0.8243 | **Q4** [0.0286, ∞) |
| `lambda_l2` | 0.8344 | 0.8308 | 0.8223 | 0.8298 | **Q3** [0.07, 0.386] |
| `feature_fraction` | 0.8344 | 0.8309 | 0.8248 | 0.8274 | **Q3** [0.8997, 0.9372] |
| `bagging_fraction` | 0.8364 | 0.8335 | 0.8226 | 0.8249 | **Q3** [0.8828, 0.949] |

#### E. Slice plot

![fred_gdp@s42/naive-ensemble](slice_fred_gdp@s42_naive-ensemble.png)


### moe

- **holdout RMSE: 1.48024** (winner retrained in 0.37s, cv score of winner: 0.8247)
- cv best RMSE: 0.8247, median: 0.9069, p10: 0.8512
- train: median 0.150s/fold, mean 0.179s, p90 0.256s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.922 |
| `lambda_l1` | 0.022 |
| `mixture_warmup_iters` | 0.015 |
| `mixture_balance_factor` | 0.010 |
| `mixture_r_smoothing` | 0.005 |
| `mixture_init` | 0.005 |
| `mixture_diversity_lambda` | 0.003 |
| `mixture_hard_m_step` | 0.003 |
| `bagging_freq` | 0.002 |
| `extra_trees` | 0.002 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 207 | 0.9094 | 0.0644 | 0.8391 |
| none | 71 | 0.9238 | 0.0747 | 0.8247 |
| gbdt | 22 | 0.9539 | 0.0743 | 0.8596 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 253 | 0.9124 | 0.0699 | 0.8247 |
| expert_choice | 47 | 0.9356 | 0.0592 | 0.8509 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 227 | 0.9083 | 0.0689 | 0.8247 |
| em | 42 | 0.9298 | 0.0536 | 0.8509 |
| loss_only | 31 | 0.9544 | 0.0706 | 0.8558 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 208 | 0.9058 | 0.0608 | 0.8391 |
| gmm | 45 | 0.9197 | 0.0762 | 0.8247 |
| random | 30 | 0.9431 | 0.0742 | 0.8596 |
| tree_hierarchical | 17 | 0.9842 | 0.0773 | 0.8633 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 150 | 0.9079 | 0.0657 | 0.8247 |
| none | 131 | 0.9162 | 0.0618 | 0.8399 |
| ema | 19 | 0.9795 | 0.0992 | 0.8555 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 197 | 0.9094 | 0.0695 | 0.8391 |
| False | 103 | 0.9288 | 0.0657 | 0.8247 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 263 | 0.9070 | 0.0639 | 0.8247 |
| False | 37 | 0.9806 | 0.0678 | 0.8945 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.9731 | — | — | 0.9129 | **Q4** [3.0, ∞) |
| `mixture_diversity_lambda` | 0.9455 | 0.9038 | 0.9111 | 0.9039 | **Q2** [0.3032, 0.3921] |
| `mixture_warmup_iters` | 0.9316 | 0.8985 | 0.9177 | 0.9184 | **Q2** [26.0, 28.5] |
| `mixture_balance_factor` | 0.9202 | — | 0.8969 | 0.9393 | **Q3** [5.0, 6.0] |
| `learning_rate` | 0.9325 | 0.8995 | 0.9025 | 0.9298 | **Q2** [0.0718, 0.0894] |
| `num_leaves` | 0.9118 | 0.9357 | 0.9036 | 0.9126 | **Q3** [95.0, 111.0] |
| `max_depth` | 0.9240 | 0.9167 | 0.9007 | 0.9267 | **Q3** [7.0, 8.0] |
| `min_data_in_leaf` | 0.8848 | 0.8697 | 0.9137 | 0.9865 | **Q2** [9.0, 11.0] |

#### E. Slice plot

![fred_gdp@s42/moe](slice_fred_gdp@s42_moe.png)


---

## fred_gdp@s43  (search X=[249, 12], holdout n=62)


### naive-lightgbm

- **holdout RMSE: 1.55352** (winner retrained in 0.03s, cv score of winner: 0.7992)
- cv best RMSE: 0.7992, median: 0.8220, p10: 0.8085
- train: median 0.016s/fold, mean 0.016s, p90 0.021s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.906 |
| `bagging_fraction` | 0.059 |
| `bagging_freq` | 0.008 |
| `lambda_l1` | 0.007 |
| `learning_rate` | 0.007 |
| `feature_fraction` | 0.006 |
| `num_leaves` | 0.004 |
| `max_depth` | 0.002 |
| `extra_trees` | 0.000 |
| `lambda_l2` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 278 | 0.8294 | 0.0263 | 0.7992 |
| True | 22 | 0.8461 | 0.0274 | 0.8170 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.8366 | 0.8283 | 0.8300 | 0.8273 | **Q4** [0.2758, ∞) |
| `num_leaves` | 0.8261 | 0.8267 | 0.8299 | 0.8396 | **Q1** [None, 15.0] |
| `max_depth` | 0.8351 | 0.8372 | 0.8280 | 0.8230 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 0.8565 | 0.8175 | 0.8182 | 0.8338 | **Q2** [14.0, 17.0] |
| `lambda_l1` | 0.8284 | 0.8235 | 0.8347 | 0.8357 | **Q2** [0.0001, 0.0004] |
| `lambda_l2` | 0.8363 | 0.8285 | 0.8232 | 0.8343 | **Q3** [0.0001, 0.0003] |
| `feature_fraction` | 0.8343 | 0.8259 | 0.8245 | 0.8376 | **Q3** [0.6627, 0.6925] |
| `bagging_fraction` | 0.8400 | 0.8243 | 0.8248 | 0.8332 | **Q2** [0.8401, 0.8623] |

#### E. Slice plot

![fred_gdp@s43/naive-lightgbm](slice_fred_gdp@s43_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 1.54157** (winner retrained in 0.04s, cv score of winner: 0.8044)
- cv best RMSE: 0.8044, median: 0.8233, p10: 0.8116
- train: median 0.025s/fold, mean 0.027s, p90 0.039s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.968 |
| `bagging_fraction` | 0.010 |
| `learning_rate` | 0.009 |
| `extra_trees` | 0.004 |
| `feature_fraction` | 0.004 |
| `num_leaves` | 0.002 |
| `max_depth` | 0.001 |
| `bagging_freq` | 0.001 |
| `n_models` | 0.001 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 271 | 0.8246 | 0.0138 | 0.8044 |
| True | 29 | 0.8374 | 0.0148 | 0.8095 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.8365 | 0.8277 | 0.8204 | 0.8187 | **Q4** [0.2708, ∞) |
| `num_leaves` | 0.8344 | 0.8249 | 0.8207 | 0.8237 | **Q3** [77.0, 86.0] |
| `max_depth` | — | — | 0.8220 | 0.8368 | **Q3** [3.0, 5.0] |
| `min_data_in_leaf` | 0.8329 | 0.8181 | 0.8189 | 0.8324 | **Q2** [22.0, 25.0] |
| `lambda_l1` | 0.8341 | 0.8295 | 0.8200 | 0.8196 | **Q4** [0.1685, ∞) |
| `lambda_l2` | 0.8283 | 0.8204 | 0.8208 | 0.8337 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 0.8337 | 0.8199 | 0.8220 | 0.8276 | **Q2** [0.858, 0.8859] |
| `bagging_fraction` | 0.8328 | 0.8277 | 0.8205 | 0.8223 | **Q3** [0.9404, 0.9672] |

#### E. Slice plot

![fred_gdp@s43/naive-ensemble](slice_fred_gdp@s43_naive-ensemble.png)


### moe

- **holdout RMSE: 1.50874** (winner retrained in 0.10s, cv score of winner: 0.8110)
- cv best RMSE: 0.8110, median: 0.8575, p10: 0.8264
- train: median 0.129s/fold, mean 0.136s, p90 0.197s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.950 |
| `learning_rate` | 0.006 |
| `mixture_diversity_lambda` | 0.005 |
| `lambda_l1` | 0.005 |
| `mixture_r_smoothing` | 0.005 |
| `bagging_freq` | 0.004 |
| `mixture_init` | 0.004 |
| `mixture_num_experts` | 0.004 |
| `mixture_gate_type` | 0.004 |
| `mixture_balance_factor` | 0.003 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **leaf_reuse** | 0.8863 (n=264) | gbdt | Δ +0.0561 | p=7.00e-04 |
| `mixture_routing_mode` | **expert_choice** | 0.8851 (n=270) | token_choice | Δ +0.0861 | p=1.10e-05 |
| `mixture_init` | **random** | 0.8757 (n=248) | uniform | Δ +0.0920 | p=6.80e-05 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 264 | 0.8863 | 0.0779 | 0.8110 |
| gbdt | 19 | 0.9424 | 0.0573 | 0.8439 |
| none | 17 | 0.9547 | 0.0818 | 0.8431 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 270 | 0.8851 | 0.0739 | 0.8110 |
| token_choice | 30 | 0.9712 | 0.0867 | 0.8315 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 252 | 0.8856 | 0.0755 | 0.8110 |
| em | 29 | 0.9279 | 0.0864 | 0.8252 |
| loss_only | 19 | 0.9485 | 0.0861 | 0.8351 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| random | 248 | 0.8757 | 0.0670 | 0.8110 |
| uniform | 14 | 0.9677 | 0.0587 | 0.9234 |
| gmm | 14 | 0.9713 | 0.0810 | 0.8805 |
| tree_hierarchical | 24 | 0.9910 | 0.0867 | 0.8802 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 227 | 0.8885 | 0.0772 | 0.8110 |
| none | 55 | 0.8985 | 0.0823 | 0.8160 |
| ema | 18 | 0.9451 | 0.0819 | 0.8351 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 146 | 0.8828 | 0.0788 | 0.8110 |
| False | 154 | 0.9040 | 0.0790 | 0.8128 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 277 | 0.8874 | 0.0776 | 0.8110 |
| False | 23 | 0.9698 | 0.0625 | 0.8640 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.9928 | — | — | 0.8903 | **Q4** [3.0, ∞) |
| `mixture_diversity_lambda` | 0.9146 | 0.8600 | 0.8724 | 0.9277 | **Q2** [0.1808, 0.201] |
| `mixture_warmup_iters` | 0.9455 | 0.8860 | 0.8743 | 0.8746 | **Q3** [48.0, 49.0] |
| `mixture_balance_factor` | 0.9184 | — | — | 0.8869 | **Q4** [6.0, ∞) |
| `learning_rate` | 0.9064 | 0.8630 | 0.8918 | 0.9135 | **Q2** [0.0491, 0.0555] |
| `num_leaves` | 0.8931 | 0.8678 | 0.8921 | 0.9227 | **Q2** [39.0, 47.5] |
| `max_depth` | — | — | 0.8804 | 0.9081 | **Q3** [3.0, 4.0] |
| `min_data_in_leaf` | 0.8569 | 0.8455 | 0.8729 | 0.9814 | **Q2** [7.0, 9.0] |

#### E. Slice plot

![fred_gdp@s43/moe](slice_fred_gdp@s43_moe.png)


---

## fred_gdp@s44  (search X=[249, 12], holdout n=62)


### naive-lightgbm

- **holdout RMSE: 1.52337** (winner retrained in 0.02s, cv score of winner: 0.8117)
- cv best RMSE: 0.8117, median: 0.8302, p10: 0.8197
- train: median 0.015s/fold, mean 0.016s, p90 0.024s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.740 |
| `learning_rate` | 0.153 |
| `bagging_fraction` | 0.054 |
| `num_leaves` | 0.015 |
| `bagging_freq` | 0.014 |
| `extra_trees` | 0.011 |
| `max_depth` | 0.009 |
| `feature_fraction` | 0.004 |
| `lambda_l2` | 0.000 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 273 | 0.8323 | 0.0129 | 0.8117 |
| True | 27 | 0.8364 | 0.0097 | 0.8218 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.8337 | 0.8346 | 0.8311 | 0.8314 | **Q3** [0.0474, 0.0804] |
| `num_leaves` | 0.8316 | 0.8285 | 0.8311 | 0.8390 | **Q2** [20.0, 29.0] |
| `max_depth` | 0.8363 | 0.8330 | 0.8297 | 0.8354 | **Q3** [8.0, 9.0] |
| `min_data_in_leaf` | 0.8391 | 0.8294 | 0.8254 | 0.8376 | **Q3** [16.0, 20.0] |
| `lambda_l1` | 0.8321 | 0.8339 | 0.8289 | 0.8359 | **Q3** [0.0, 0.0] |
| `lambda_l2` | 0.8372 | 0.8316 | 0.8311 | 0.8309 | **Q4** [0.2105, ∞) |
| `feature_fraction` | 0.8376 | 0.8345 | 0.8291 | 0.8297 | **Q3** [0.748, 0.8971] |
| `bagging_fraction` | 0.8316 | 0.8278 | 0.8316 | 0.8398 | **Q2** [0.5409, 0.5594] |

#### E. Slice plot

![fred_gdp@s44/naive-lightgbm](slice_fred_gdp@s44_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 1.53460** (winner retrained in 0.05s, cv score of winner: 0.8101)
- cv best RMSE: 0.8101, median: 0.8252, p10: 0.8139
- train: median 0.028s/fold, mean 0.029s, p90 0.038s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.920 |
| `learning_rate` | 0.030 |
| `bagging_fraction` | 0.011 |
| `num_leaves` | 0.010 |
| `bagging_freq` | 0.009 |
| `extra_trees` | 0.008 |
| `feature_fraction` | 0.006 |
| `max_depth` | 0.003 |
| `lambda_l2` | 0.002 |
| `n_models` | 0.002 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 260 | 0.8255 | 0.0107 | 0.8101 |
| True | 40 | 0.8375 | 0.0134 | 0.8162 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.8293 | 0.8223 | 0.8294 | 0.8274 | **Q2** [0.0498, 0.0854] |
| `num_leaves` | 0.8260 | 0.8283 | 0.8293 | 0.8250 | **Q4** [116.0, ∞) |
| `max_depth` | 0.8279 | 0.8305 | — | 0.8255 | **Q4** [10.0, ∞) |
| `min_data_in_leaf` | 0.8336 | 0.8205 | 0.8206 | 0.8342 | **Q2** [22.0, 25.0] |
| `lambda_l1` | 0.8314 | 0.8214 | 0.8256 | 0.8301 | **Q2** [0.0, 0.0] |
| `lambda_l2` | 0.8303 | 0.8270 | 0.8229 | 0.8283 | **Q3** [0.186, 0.9973] |
| `feature_fraction` | 0.8331 | 0.8271 | 0.8217 | 0.8265 | **Q3** [0.7364, 0.7612] |
| `bagging_fraction` | 0.8256 | 0.8257 | 0.8263 | 0.8309 | **Q1** [None, 0.7274] |

#### E. Slice plot

![fred_gdp@s44/naive-ensemble](slice_fred_gdp@s44_naive-ensemble.png)


### moe

- **holdout RMSE: 1.47690** (winner retrained in 2.37s, cv score of winner: 0.8144)
- cv best RMSE: 0.8144, median: 0.8929, p10: 0.8341
- train: median 0.120s/fold, mean 0.651s, p90 2.584s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.973 |
| `learning_rate` | 0.006 |
| `extra_trees` | 0.005 |
| `num_leaves` | 0.004 |
| `mixture_diversity_lambda` | 0.002 |
| `mixture_gate_type` | 0.002 |
| `mixture_warmup_iters` | 0.001 |
| `feature_fraction` | 0.001 |
| `mixture_balance_factor` | 0.001 |
| `bagging_fraction` | 0.001 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_routing_mode` | **token_choice** | 0.8915 (n=259) | expert_choice | Δ +0.0768 | p=0.00e+00 |
| `mixture_r_smoothing` | **none** | 0.8897 (n=212) | ema | Δ +0.0391 | p=1.78e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 162 | 0.8907 | 0.0592 | 0.8256 |
| none | 72 | 0.9094 | 0.0890 | 0.8144 |
| leaf_reuse | 66 | 0.9217 | 0.0647 | 0.8360 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 259 | 0.8915 | 0.0636 | 0.8144 |
| expert_choice | 41 | 0.9683 | 0.0715 | 0.8406 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 165 | 0.8914 | 0.0623 | 0.8256 |
| loss_only | 69 | 0.9076 | 0.0817 | 0.8144 |
| em | 66 | 0.9228 | 0.0688 | 0.8360 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| random | 121 | 0.8842 | 0.0712 | 0.8144 |
| uniform | 142 | 0.9013 | 0.0618 | 0.8256 |
| tree_hierarchical | 13 | 0.9515 | 0.0417 | 0.8979 |
| gmm | 24 | 0.9695 | 0.0670 | 0.8761 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 212 | 0.8897 | 0.0651 | 0.8144 |
| ema | 42 | 0.9288 | 0.0707 | 0.8357 |
| markov | 46 | 0.9342 | 0.0733 | 0.8158 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 174 | 0.8988 | 0.0746 | 0.8144 |
| True | 126 | 0.9064 | 0.0624 | 0.8308 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 279 | 0.8956 | 0.0636 | 0.8144 |
| False | 21 | 0.9872 | 0.0904 | 0.8564 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.9486 | — | — | 0.8865 | **Q4** [4.0, ∞) |
| `mixture_diversity_lambda` | 0.9032 | 0.8832 | 0.8999 | 0.9219 | **Q2** [0.0405, 0.065] |
| `mixture_warmup_iters` | 0.9052 | 0.9238 | 0.8881 | 0.8910 | **Q3** [29.0, 34.0] |
| `mixture_balance_factor` | 0.9381 | 0.8828 | 0.9040 | 0.9018 | **Q2** [7.0, 8.0] |
| `learning_rate` | 0.9000 | 0.8881 | 0.9039 | 0.9161 | **Q2** [0.0432, 0.0658] |
| `num_leaves` | 0.9232 | 0.8965 | 0.9059 | 0.8824 | **Q4** [90.5, ∞) |
| `max_depth` | 0.9249 | 0.8789 | 0.9109 | 0.9158 | **Q2** [6.0, 7.0] |
| `min_data_in_leaf` | 0.8756 | 0.8530 | 0.8943 | 0.9794 | **Q2** [8.0, 11.0] |

#### E. Slice plot

![fred_gdp@s44/moe](slice_fred_gdp@s44_moe.png)


---

## Overall recommendations

**Categorical settings that are statistically significant winners (p<0.01):**

| dataset | param | best value | Δ vs runner-up | p |
|---|---|---|---|---|
| fred_gdp@s43 | `mixture_gate_type` | **leaf_reuse** | +0.0561 | 7.00e-04 |
| fred_gdp@s43 | `mixture_routing_mode` | **expert_choice** | +0.0861 | 1.10e-05 |
| fred_gdp@s43 | `mixture_init` | **random** | +0.0920 | 6.80e-05 |
| fred_gdp@s44 | `mixture_routing_mode` | **token_choice** | +0.0768 | 0.00e+00 |
| fred_gdp@s44 | `mixture_r_smoothing` | **none** | +0.0391 | 1.78e-03 |
