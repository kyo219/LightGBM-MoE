# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 300

- **Datasets**: ['fred_gdp'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `0cf3634c544c`, lib sha256 `5cec0a0bd5ab…`, package `/tmp/lgbm-moe-v080/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| fred_gdp | naive-lightgbm | **1.52767** ± 0.01959 | 0.80518 | 0.0% | 0.02 |
| fred_gdp | moe | **1.49375** ± 0.00592 | 0.82530 | 0.0% | 1.30 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| fred_gdp@s42 | naive-lightgbm | 0.8046 | 0.8281 | 0.013 | 25 |
| fred_gdp@s42 | moe | 0.8384 | 0.9038 | 0.136 | 240 |
| fred_gdp@s43 | naive-lightgbm | 0.7992 | 0.8220 | 0.017 | 31 |
| fred_gdp@s43 | moe | 0.8231 | 0.9203 | 0.150 | 663 |
| fred_gdp@s44 | naive-lightgbm | 0.8117 | 0.8302 | 0.010 | 22 |
| fred_gdp@s44 | moe | 0.8144 | 0.8788 | 0.233 | 1157 |



---

## fred_gdp@s42  (search X=[249, 12], holdout n=62)


### naive-lightgbm

- **holdout RMSE: 1.50612** (winner retrained in 0.02s, cv score of winner: 0.8046)
- cv best RMSE: 0.8046, median: 0.8281, p10: 0.8113
- train: median 0.013s/fold, mean 0.013s, p90 0.016s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.877 |
| `max_depth` | 0.031 |
| `learning_rate` | 0.027 |
| `bagging_fraction` | 0.017 |
| `num_leaves` | 0.015 |
| `feature_fraction` | 0.012 |
| `bagging_freq` | 0.012 |
| `extra_trees` | 0.007 |
| `lambda_l2` | 0.002 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 278 | 0.8276 | 0.0146 | 0.8046 |
| True | 22 | 0.8417 | 0.0101 | 0.8252 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.8370 | 0.8285 | 0.8227 | 0.8264 | **Q3** [0.2513, 0.2747] |
| `num_leaves` | 0.8229 | 0.8247 | 0.8300 | 0.8367 | **Q1** [None, 12.0] |
| `max_depth` | 0.8320 | 0.8270 | 0.8239 | 0.8327 | **Q3** [7.0, 8.0] |
| `min_data_in_leaf` | 0.8392 | 0.8207 | 0.8220 | 0.8337 | **Q2** [20.0, 23.0] |
| `lambda_l1` | 0.8320 | 0.8254 | 0.8293 | 0.8279 | **Q2** [0.0, 0.0001] |
| `lambda_l2` | 0.8262 | 0.8260 | 0.8294 | 0.8330 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 0.8354 | 0.8243 | 0.8249 | 0.8301 | **Q2** [0.8639, 0.892] |
| `bagging_fraction` | 0.8319 | 0.8226 | 0.8258 | 0.8343 | **Q2** [0.7971, 0.8357] |

#### E. Slice plot

![fred_gdp@s42/naive-lightgbm](slice_fred_gdp@s42_naive-lightgbm.png)


### moe

- **holdout RMSE: 1.49863** (winner retrained in 0.22s, cv score of winner: 0.8384)
- cv best RMSE: 0.8384, median: 0.9038, p10: 0.8586
- train: median 0.136s/fold, mean 0.150s, p90 0.261s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.918 |
| `mixture_warmup_iters` | 0.018 |
| `lambda_l1` | 0.015 |
| `mixture_init` | 0.006 |
| `num_leaves` | 0.006 |
| `mixture_diversity_lambda` | 0.006 |
| `learning_rate` | 0.005 |
| `feature_fraction` | 0.004 |
| `mixture_balance_factor` | 0.004 |
| `mixture_num_experts` | 0.003 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **leaf_reuse** | 0.9125 (n=211) | none | Δ +0.0277 | p=9.06e-03 |
| `mixture_routing_mode` | **expert_choice** | 0.9102 (n=246) | token_choice | Δ +0.0643 | p=1.00e-06 |
| `mixture_init` | **uniform** | 0.9071 (n=239) | random | Δ +0.0610 | p=3.14e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 211 | 0.9125 | 0.0680 | 0.8384 |
| none | 62 | 0.9402 | 0.0725 | 0.8452 |
| gbdt | 27 | 0.9521 | 0.0783 | 0.8611 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 246 | 0.9102 | 0.0630 | 0.8384 |
| token_choice | 54 | 0.9745 | 0.0829 | 0.8437 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 213 | 0.9145 | 0.0691 | 0.8437 |
| em | 60 | 0.9281 | 0.0722 | 0.8384 |
| loss_only | 27 | 0.9659 | 0.0707 | 0.8699 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 239 | 0.9071 | 0.0586 | 0.8384 |
| random | 26 | 0.9681 | 0.0921 | 0.8453 |
| gmm | 15 | 0.9740 | 0.0671 | 0.8748 |
| tree_hierarchical | 20 | 0.9980 | 0.0887 | 0.8612 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 115 | 0.9113 | 0.0714 | 0.8475 |
| none | 152 | 0.9229 | 0.0679 | 0.8384 |
| ema | 33 | 0.9538 | 0.0772 | 0.8453 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 279 | 0.9167 | 0.0669 | 0.8384 |
| True | 21 | 0.9895 | 0.0919 | 0.8786 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 250 | 0.9121 | 0.0667 | 0.8384 |
| False | 50 | 0.9702 | 0.0742 | 0.8453 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.9656 | — | — | 0.9116 | **Q4** [4.0, ∞) |
| `mixture_diversity_lambda` | 0.9442 | 0.9180 | 0.9010 | 0.9241 | **Q3** [0.355, 0.3799] |
| `mixture_warmup_iters` | 0.9243 | 0.9160 | 0.9040 | 0.9446 | **Q3** [18.0, 25.0] |
| `mixture_balance_factor` | 0.9519 | 0.9111 | 0.9286 | 0.9014 | **Q4** [10.0, ∞) |
| `learning_rate` | 0.9681 | 0.9063 | 0.8944 | 0.9185 | **Q3** [0.1419, 0.1652] |
| `num_leaves` | 0.9424 | 0.9026 | 0.9229 | 0.9196 | **Q2** [65.75, 75.5] |
| `max_depth` | 0.9116 | 0.9580 | 0.9039 | 0.9156 | **Q3** [8.0, 9.0] |
| `min_data_in_leaf` | 0.8947 | 0.8900 | 0.9038 | 0.9925 | **Q2** [7.0, 9.0] |

#### E. Slice plot

![fred_gdp@s42/moe](slice_fred_gdp@s42_moe.png)


---

## fred_gdp@s43  (search X=[249, 12], holdout n=62)


### naive-lightgbm

- **holdout RMSE: 1.55352** (winner retrained in 0.03s, cv score of winner: 0.7992)
- cv best RMSE: 0.7992, median: 0.8220, p10: 0.8085
- train: median 0.017s/fold, mean 0.017s, p90 0.022s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.897 |
| `bagging_fraction` | 0.060 |
| `feature_fraction` | 0.012 |
| `bagging_freq` | 0.008 |
| `lambda_l1` | 0.006 |
| `learning_rate` | 0.006 |
| `num_leaves` | 0.006 |
| `max_depth` | 0.002 |
| `lambda_l2` | 0.002 |
| `extra_trees` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 278 | 0.8294 | 0.0263 | 0.7992 |
| True | 22 | 0.8461 | 0.0275 | 0.8170 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.8367 | 0.8283 | 0.8300 | 0.8273 | **Q4** [0.2758, ∞) |
| `num_leaves` | 0.8261 | 0.8267 | 0.8299 | 0.8396 | **Q1** [None, 15.0] |
| `max_depth` | 0.8351 | 0.8372 | 0.8280 | 0.8230 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 0.8565 | 0.8175 | 0.8182 | 0.8338 | **Q2** [14.0, 17.0] |
| `lambda_l1` | 0.8285 | 0.8235 | 0.8346 | 0.8358 | **Q2** [0.0001, 0.0004] |
| `lambda_l2` | 0.8363 | 0.8285 | 0.8232 | 0.8343 | **Q3** [0.0001, 0.0003] |
| `feature_fraction` | 0.8343 | 0.8259 | 0.8245 | 0.8376 | **Q3** [0.6627, 0.6925] |
| `bagging_fraction` | 0.8400 | 0.8243 | 0.8248 | 0.8332 | **Q2** [0.8401, 0.8623] |

#### E. Slice plot

![fred_gdp@s43/naive-lightgbm](slice_fred_gdp@s43_naive-lightgbm.png)


### moe

- **holdout RMSE: 1.48542** (winner retrained in 0.23s, cv score of winner: 0.8231)
- cv best RMSE: 0.8231, median: 0.9203, p10: 0.8351
- train: median 0.150s/fold, mean 0.434s, p90 0.250s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.988 |
| `mixture_init` | 0.003 |
| `learning_rate` | 0.002 |
| `mixture_warmup_iters` | 0.001 |
| `max_depth` | 0.001 |
| `mixture_r_smoothing` | 0.001 |
| `bagging_fraction` | 0.001 |
| `feature_fraction` | 0.001 |
| `mixture_diversity_lambda` | 0.001 |
| `num_leaves` | 0.001 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_routing_mode` | **expert_choice** | 0.9004 (n=195) | token_choice | Δ +0.0361 | p=1.40e-05 |
| `mixture_init` | **uniform** | 0.8972 (n=170) | random | Δ +0.0284 | p=2.15e-03 |
| `mixture_r_smoothing` | **none** | 0.9040 (n=241) | ema | Δ +0.0316 | p=1.11e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 171 | 0.9020 | 0.0622 | 0.8231 |
| none | 66 | 0.9231 | 0.0704 | 0.8274 |
| gbdt | 63 | 0.9326 | 0.0554 | 0.8472 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 195 | 0.9004 | 0.0561 | 0.8231 |
| token_choice | 105 | 0.9365 | 0.0710 | 0.8274 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| em | 190 | 0.9051 | 0.0623 | 0.8231 |
| gate_only | 90 | 0.9211 | 0.0645 | 0.8274 |
| loss_only | 20 | 0.9526 | 0.0595 | 0.8662 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 170 | 0.8972 | 0.0573 | 0.8231 |
| random | 75 | 0.9256 | 0.0683 | 0.8274 |
| tree_hierarchical | 33 | 0.9432 | 0.0747 | 0.8410 |
| gmm | 22 | 0.9477 | 0.0363 | 0.8537 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 241 | 0.9040 | 0.0613 | 0.8231 |
| ema | 30 | 0.9356 | 0.0439 | 0.8472 |
| markov | 29 | 0.9653 | 0.0730 | 0.8662 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 230 | 0.9053 | 0.0632 | 0.8231 |
| True | 70 | 0.9384 | 0.0604 | 0.8472 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 231 | 0.9053 | 0.0629 | 0.8231 |
| False | 69 | 0.9391 | 0.0610 | 0.8472 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.9630 | — | — | 0.9058 | **Q4** [4.0, ∞) |
| `mixture_diversity_lambda` | 0.9208 | 0.9261 | 0.8939 | 0.9114 | **Q3** [0.32, 0.355] |
| `mixture_warmup_iters` | 0.9329 | 0.9199 | 0.9014 | 0.9012 | **Q4** [43.0, ∞) |
| `mixture_balance_factor` | 0.9375 | 0.9300 | — | 0.8960 | **Q4** [10.0, ∞) |
| `learning_rate` | 0.9387 | 0.9102 | 0.9004 | 0.9030 | **Q3** [0.1488, 0.1936] |
| `num_leaves` | 0.8924 | 0.8962 | 0.9360 | 0.9275 | **Q1** [None, 17.0] |
| `max_depth` | 0.9414 | 0.9283 | — | 0.9047 | **Q4** [9.0, ∞) |
| `min_data_in_leaf` | 0.8867 | 0.8740 | 0.8912 | 0.9757 | **Q2** [10.0, 11.0] |

#### E. Slice plot

![fred_gdp@s43/moe](slice_fred_gdp@s43_moe.png)


---

## fred_gdp@s44  (search X=[249, 12], holdout n=62)


### naive-lightgbm

- **holdout RMSE: 1.52337** (winner retrained in 0.02s, cv score of winner: 0.8117)
- cv best RMSE: 0.8117, median: 0.8302, p10: 0.8197
- train: median 0.010s/fold, mean 0.011s, p90 0.016s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.755 |
| `learning_rate` | 0.159 |
| `bagging_fraction` | 0.029 |
| `extra_trees` | 0.021 |
| `bagging_freq` | 0.011 |
| `max_depth` | 0.011 |
| `num_leaves` | 0.009 |
| `feature_fraction` | 0.004 |
| `lambda_l2` | 0.001 |
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
| `num_leaves` | 0.8316 | 0.8286 | 0.8311 | 0.8390 | **Q2** [20.0, 29.0] |
| `max_depth` | 0.8363 | 0.8330 | 0.8297 | 0.8354 | **Q3** [8.0, 9.0] |
| `min_data_in_leaf` | 0.8391 | 0.8294 | 0.8254 | 0.8376 | **Q3** [16.0, 20.0] |
| `lambda_l1` | 0.8321 | 0.8339 | 0.8289 | 0.8359 | **Q3** [0.0, 0.0] |
| `lambda_l2` | 0.8372 | 0.8316 | 0.8311 | 0.8309 | **Q4** [0.2105, ∞) |
| `feature_fraction` | 0.8376 | 0.8345 | 0.8291 | 0.8297 | **Q3** [0.748, 0.8971] |
| `bagging_fraction` | 0.8316 | 0.8278 | 0.8316 | 0.8398 | **Q2** [0.5409, 0.5594] |

#### E. Slice plot

![fred_gdp@s44/naive-lightgbm](slice_fred_gdp@s44_naive-lightgbm.png)


### moe

- **holdout RMSE: 1.49721** (winner retrained in 3.45s, cv score of winner: 0.8144)
- cv best RMSE: 0.8144, median: 0.8788, p10: 0.8301
- train: median 0.233s/fold, mean 0.763s, p90 2.278s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.984 |
| `learning_rate` | 0.003 |
| `mixture_r_smoothing` | 0.002 |
| `bagging_freq` | 0.002 |
| `mixture_diversity_lambda` | 0.001 |
| `num_leaves` | 0.001 |
| `mixture_gate_type` | 0.001 |
| `mixture_init` | 0.001 |
| `bagging_fraction` | 0.001 |
| `lambda_l1` | 0.001 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **gbdt** | 0.8950 (n=256) | leaf_reuse | Δ +0.0607 | p=1.59e-04 |
| `mixture_routing_mode` | **expert_choice** | 0.8972 (n=238) | token_choice | Δ +0.0330 | p=2.91e-04 |
| `mixture_e_step_mode` | **loss_only** | 0.8786 (n=155) | gate_only | Δ +0.0480 | p=0.00e+00 |
| `mixture_init` | **gmm** | 0.8759 (n=146) | tree_hierarchical | Δ +0.0431 | p=1.28e-04 |
| `mixture_r_smoothing` | **markov** | 0.8786 (n=153) | ema | Δ +0.0489 | p=3.90e-05 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 256 | 0.8950 | 0.0747 | 0.8144 |
| leaf_reuse | 23 | 0.9557 | 0.0614 | 0.8615 |
| none | 21 | 0.9569 | 0.0697 | 0.8692 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 238 | 0.8972 | 0.0796 | 0.8144 |
| token_choice | 62 | 0.9302 | 0.0564 | 0.8425 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 155 | 0.8786 | 0.0720 | 0.8144 |
| gate_only | 126 | 0.9266 | 0.0702 | 0.8388 |
| em | 19 | 0.9615 | 0.0752 | 0.8394 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gmm | 146 | 0.8759 | 0.0714 | 0.8144 |
| tree_hierarchical | 68 | 0.9190 | 0.0748 | 0.8340 |
| uniform | 74 | 0.9341 | 0.0648 | 0.8433 |
| random | 12 | 0.9750 | 0.0734 | 0.8536 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 153 | 0.8786 | 0.0686 | 0.8144 |
| ema | 70 | 0.9275 | 0.0827 | 0.8269 |
| none | 77 | 0.9331 | 0.0681 | 0.8425 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 269 | 0.8979 | 0.0744 | 0.8144 |
| False | 31 | 0.9571 | 0.0742 | 0.8444 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 248 | 0.8952 | 0.0752 | 0.8144 |
| False | 52 | 0.9460 | 0.0687 | 0.8606 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.9679 | — | 0.8788 | 0.9276 | **Q3** [3.0, 4.0] |
| `mixture_diversity_lambda` | 0.9051 | 0.8831 | 0.8936 | 0.9342 | **Q2** [0.1988, 0.2331] |
| `mixture_warmup_iters` | 0.8954 | 0.8817 | 0.9349 | 0.9073 | **Q2** [25.0, 37.0] |
| `mixture_balance_factor` | 0.8828 | 0.9149 | 0.9377 | 0.8828 | **Q1** [None, 4.0] |
| `learning_rate` | 0.9237 | 0.8754 | 0.8773 | 0.9397 | **Q2** [0.0397, 0.0748] |
| `num_leaves` | 0.9581 | 0.8969 | 0.8887 | 0.8768 | **Q4** [125.0, ∞) |
| `max_depth` | 0.9267 | 0.8957 | 0.8711 | 0.9446 | **Q3** [5.0, 6.0] |
| `min_data_in_leaf` | 0.8628 | 0.8619 | 0.8948 | 0.9858 | **Q2** [7.0, 10.0] |

#### E. Slice plot

![fred_gdp@s44/moe](slice_fred_gdp@s44_moe.png)


---

## Overall recommendations

**Categorical settings that are statistically significant winners (p<0.01):**

| dataset | param | best value | Δ vs runner-up | p |
|---|---|---|---|---|
| fred_gdp@s42 | `mixture_gate_type` | **leaf_reuse** | +0.0277 | 9.06e-03 |
| fred_gdp@s42 | `mixture_routing_mode` | **expert_choice** | +0.0643 | 1.00e-06 |
| fred_gdp@s42 | `mixture_init` | **uniform** | +0.0610 | 3.14e-03 |
| fred_gdp@s43 | `mixture_routing_mode` | **expert_choice** | +0.0361 | 1.40e-05 |
| fred_gdp@s43 | `mixture_init` | **uniform** | +0.0284 | 2.15e-03 |
| fred_gdp@s43 | `mixture_r_smoothing` | **none** | +0.0316 | 1.11e-03 |
| fred_gdp@s44 | `mixture_gate_type` | **gbdt** | +0.0607 | 1.59e-04 |
| fred_gdp@s44 | `mixture_routing_mode` | **expert_choice** | +0.0330 | 2.91e-04 |
| fred_gdp@s44 | `mixture_e_step_mode` | **loss_only** | +0.0480 | 0.00e+00 |
| fred_gdp@s44 | `mixture_init` | **gmm** | +0.0431 | 1.28e-04 |
| fred_gdp@s44 | `mixture_r_smoothing` | **markov** | +0.0489 | 3.90e-05 |
