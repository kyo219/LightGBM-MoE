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
| fred_gdp | naive-lightgbm | **1.54205** ± 0.02596 | 0.80281 | 0.0% | 0.03 |
| fred_gdp | moe | **1.49429** ± 0.00982 | 0.81894 | 0.0% | 2.17 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| fred_gdp@s42 | naive-lightgbm | 0.8046 | 0.8281 | 0.014 | 47 |
| fred_gdp@s42 | moe | 0.8089 | 0.8703 | 1.079 | 2385 |
| fred_gdp@s43 | naive-lightgbm | 0.7992 | 0.8210 | 0.018 | 57 |
| fred_gdp@s43 | moe | 0.8206 | 0.8563 | 0.160 | 1284 |
| fred_gdp@s44 | naive-lightgbm | 0.8046 | 0.8285 | 0.015 | 53 |
| fred_gdp@s44 | moe | 0.8274 | 0.9190 | 0.383 | 1299 |



---

## fred_gdp@s42  (search X=[249, 12], holdout n=62)


### naive-lightgbm

- **holdout RMSE: 1.50612** (winner retrained in 0.02s, cv score of winner: 0.8046)
- cv best RMSE: 0.8046, median: 0.8281, p10: 0.8114
- train: median 0.014s/fold, mean 0.014s, p90 0.018s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.937 |
| `bagging_freq` | 0.016 |
| `bagging_fraction` | 0.011 |
| `feature_fraction` | 0.011 |
| `learning_rate` | 0.011 |
| `max_depth` | 0.008 |
| `num_leaves` | 0.006 |
| `extra_trees` | 0.001 |
| `lambda_l1` | 0.000 |
| `lambda_l2` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 0.8283 | 0.0163 | 0.8046 |
| True | 32 | 0.8383 | 0.0102 | 0.8200 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.8345 | 0.8271 | 0.8245 | 0.8296 | **Q3** [0.2547, 0.2776] |
| `num_leaves` | 0.8253 | 0.8257 | 0.8297 | 0.8342 | **Q1** [None, 11.0] |
| `max_depth` | 0.8307 | 0.8262 | 0.8318 | 0.8299 | **Q2** [5.0, 6.0] |
| `min_data_in_leaf` | 0.8402 | 0.8213 | 0.8230 | 0.8311 | **Q2** [19.0, 22.0] |
| `lambda_l1` | 0.8320 | 0.8271 | 0.8272 | 0.8293 | **Q2** [0.0, 0.0001] |
| `lambda_l2` | 0.8279 | 0.8277 | 0.8265 | 0.8336 | **Q3** [0.0, 0.0] |
| `feature_fraction` | 0.8353 | 0.8268 | 0.8256 | 0.8280 | **Q3** [0.8907, 0.916] |
| `bagging_fraction` | 0.8302 | 0.8278 | 0.8254 | 0.8322 | **Q3** [0.8248, 0.8635] |

#### E. Slice plot

![fred_gdp@s42/naive-lightgbm](slice_fred_gdp@s42_naive-lightgbm.png)


### moe

- **holdout RMSE: 1.49056** (winner retrained in 1.73s, cv score of winner: 0.8089)
- cv best RMSE: 0.8089, median: 0.8703, p10: 0.8302
- train: median 1.079s/fold, mean 0.943s, p90 1.748s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.962 |
| `mixture_balance_factor` | 0.009 |
| `extra_trees` | 0.007 |
| `mixture_load_balance_alpha` | 0.004 |
| `mixture_e_step_mode` | 0.004 |
| `lambda_l2` | 0.003 |
| `mixture_init` | 0.002 |
| `max_depth` | 0.001 |
| `bagging_fraction` | 0.001 |
| `feature_fraction` | 0.001 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **none** | 0.8830 (n=372) | leaf_reuse | Δ +0.0500 | p=0.00e+00 |
| `mixture_routing_mode` | **token_choice** | 0.8858 (n=330) | expert_choice | Δ +0.0326 | p=2.00e-06 |
| `mixture_e_step_mode` | **gate_only** | 0.8867 (n=381) | loss_only | Δ +0.0407 | p=6.90e-05 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 372 | 0.8830 | 0.0645 | 0.8089 |
| leaf_reuse | 95 | 0.9330 | 0.0841 | 0.8385 |
| gbdt | 33 | 0.9490 | 0.0635 | 0.8531 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 330 | 0.8858 | 0.0704 | 0.8089 |
| expert_choice | 170 | 0.9184 | 0.0720 | 0.8270 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 381 | 0.8867 | 0.0681 | 0.8089 |
| loss_only | 62 | 0.9274 | 0.0705 | 0.8241 |
| em | 57 | 0.9315 | 0.0835 | 0.8322 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gmm | 286 | 0.8817 | 0.0666 | 0.8089 |
| gmm_features | 52 | 0.9016 | 0.0751 | 0.8160 |
| uniform | 101 | 0.9082 | 0.0699 | 0.8270 |
| random | 18 | 0.9357 | 0.0745 | 0.8415 |
| tree_hierarchical | 18 | 0.9486 | 0.0840 | 0.8358 |
| kmeans_features | 25 | 0.9487 | 0.0725 | 0.8455 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 300 | 0.8860 | 0.0688 | 0.8089 |
| none | 142 | 0.8999 | 0.0706 | 0.8270 |
| ema | 58 | 0.9457 | 0.0760 | 0.8375 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 422 | 0.8901 | 0.0715 | 0.8089 |
| True | 78 | 0.9333 | 0.0678 | 0.8445 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 459 | 0.8909 | 0.0703 | 0.8089 |
| False | 41 | 0.9631 | 0.0649 | 0.8457 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.9437 | — | 0.8909 | 0.9060 | **Q3** [3.0, 5.0] |
| `mixture_diversity_lambda` | 0.8725 | 0.8809 | 0.9028 | 0.9312 | **Q1** [None, 0.0224] |
| `mixture_warmup_iters` | 0.9149 | 0.8815 | 0.8898 | 0.9015 | **Q2** [37.0, 40.0] |
| `mixture_balance_factor` | 0.8973 | 0.8905 | 0.8901 | 0.9077 | **Q3** [5.0, 8.0] |
| `learning_rate` | 0.9169 | 0.8878 | 0.8881 | 0.8946 | **Q2** [0.0581, 0.1033] |
| `num_leaves` | 0.8985 | 0.8900 | 0.8795 | 0.9171 | **Q3** [37.0, 43.0] |
| `max_depth` | 0.8971 | 0.8976 | 0.8758 | 0.9258 | **Q3** [6.0, 7.0] |
| `min_data_in_leaf` | 0.8586 | 0.8522 | 0.8890 | 0.9843 | **Q2** [7.0, 9.5] |

#### E. Slice plot

![fred_gdp@s42/moe](slice_fred_gdp@s42_moe.png)


---

## fred_gdp@s43  (search X=[249, 12], holdout n=62)


### naive-lightgbm

- **holdout RMSE: 1.55352** (winner retrained in 0.04s, cv score of winner: 0.7992)
- cv best RMSE: 0.7992, median: 0.8210, p10: 0.8083
- train: median 0.018s/fold, mean 0.018s, p90 0.024s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.891 |
| `bagging_fraction` | 0.078 |
| `learning_rate` | 0.010 |
| `bagging_freq` | 0.006 |
| `feature_fraction` | 0.005 |
| `num_leaves` | 0.005 |
| `max_depth` | 0.002 |
| `lambda_l1` | 0.002 |
| `extra_trees` | 0.001 |
| `lambda_l2` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 0.8272 | 0.0245 | 0.7992 |
| True | 32 | 0.8436 | 0.0255 | 0.8129 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.8323 | 0.8261 | 0.8259 | 0.8286 | **Q3** [0.2488, 0.274] |
| `num_leaves` | 0.8262 | 0.8299 | 0.8285 | 0.8281 | **Q1** [None, 16.0] |
| `max_depth` | 0.8344 | 0.8316 | 0.8301 | 0.8229 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 0.8551 | 0.8172 | 0.8191 | 0.8289 | **Q2** [14.0, 18.0] |
| `lambda_l1` | 0.8282 | 0.8229 | 0.8276 | 0.8342 | **Q2** [0.0001, 0.0005] |
| `lambda_l2` | 0.8345 | 0.8257 | 0.8249 | 0.8278 | **Q3** [0.0001, 0.0004] |
| `feature_fraction` | 0.8314 | 0.8236 | 0.8256 | 0.8323 | **Q2** [0.6343, 0.6578] |
| `bagging_fraction` | 0.8373 | 0.8242 | 0.8225 | 0.8288 | **Q3** [0.8706, 0.8901] |

#### E. Slice plot

![fred_gdp@s43/naive-lightgbm](slice_fred_gdp@s43_naive-lightgbm.png)


### moe

- **holdout RMSE: 1.50774** (winner retrained in 3.83s, cv score of winner: 0.8206)
- cv best RMSE: 0.8206, median: 0.8563, p10: 0.8322
- train: median 0.160s/fold, mean 0.502s, p90 1.863s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.922 |
| `extra_trees` | 0.022 |
| `lambda_l1` | 0.009 |
| `mixture_balance_factor` | 0.006 |
| `lambda_l2` | 0.005 |
| `mixture_warmup_iters` | 0.005 |
| `mixture_init` | 0.005 |
| `mixture_load_balance_alpha` | 0.005 |
| `num_leaves` | 0.004 |
| `mixture_e_step_mode` | 0.003 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **none** | 0.8801 (n=363) | gbdt | Δ +0.0243 | p=4.98e-03 |
| `mixture_routing_mode` | **expert_choice** | 0.8808 (n=418) | token_choice | Δ +0.0552 | p=0.00e+00 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 363 | 0.8801 | 0.0645 | 0.8228 |
| gbdt | 114 | 0.9044 | 0.0830 | 0.8206 |
| leaf_reuse | 23 | 0.9722 | 0.0950 | 0.8567 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 418 | 0.8808 | 0.0682 | 0.8206 |
| token_choice | 82 | 0.9360 | 0.0829 | 0.8345 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 274 | 0.8837 | 0.0697 | 0.8228 |
| em | 162 | 0.8859 | 0.0729 | 0.8206 |
| gate_only | 64 | 0.9263 | 0.0818 | 0.8413 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gmm_features | 290 | 0.8774 | 0.0651 | 0.8228 |
| kmeans_features | 81 | 0.8846 | 0.0563 | 0.8302 |
| gmm | 42 | 0.9040 | 0.0923 | 0.8206 |
| uniform | 17 | 0.9100 | 0.0598 | 0.8526 |
| random | 53 | 0.9281 | 0.0879 | 0.8377 |
| tree_hierarchical | 17 | 0.9543 | 0.1065 | 0.8402 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 360 | 0.8806 | 0.0662 | 0.8228 |
| markov | 75 | 0.9014 | 0.0842 | 0.8206 |
| none | 65 | 0.9279 | 0.0851 | 0.8413 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 423 | 0.8804 | 0.0672 | 0.8206 |
| False | 77 | 0.9417 | 0.0854 | 0.8372 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 469 | 0.8857 | 0.0707 | 0.8206 |
| False | 31 | 0.9535 | 0.0880 | 0.8539 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.9082 | — | — | 0.8854 | **Q4** [3.0, ∞) |
| `mixture_diversity_lambda` | 0.8935 | 0.8875 | 0.9020 | 0.8764 | **Q4** [0.4764, ∞) |
| `mixture_warmup_iters` | 0.9143 | 0.8864 | 0.8845 | 0.8801 | **Q4** [39.0, ∞) |
| `mixture_balance_factor` | 0.8797 | 0.8829 | 0.8788 | 0.9192 | **Q3** [6.0, 7.0] |
| `learning_rate` | 0.9088 | 0.8786 | 0.8751 | 0.8971 | **Q3** [0.0582, 0.075] |
| `num_leaves` | 0.9079 | 0.8708 | 0.8880 | 0.8928 | **Q2** [38.0, 47.0] |
| `max_depth` | 0.9212 | 0.8908 | — | 0.8777 | **Q4** [10.0, ∞) |
| `min_data_in_leaf` | 0.8672 | 0.8437 | 0.8551 | 0.9803 | **Q2** [7.0, 9.0] |

#### E. Slice plot

![fred_gdp@s43/moe](slice_fred_gdp@s43_moe.png)


---

## fred_gdp@s44  (search X=[249, 12], holdout n=62)


### naive-lightgbm

- **holdout RMSE: 1.56652** (winner retrained in 0.02s, cv score of winner: 0.8046)
- cv best RMSE: 0.8046, median: 0.8285, p10: 0.8183
- train: median 0.015s/fold, mean 0.017s, p90 0.024s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.868 |
| `learning_rate` | 0.056 |
| `bagging_fraction` | 0.036 |
| `lambda_l2` | 0.009 |
| `num_leaves` | 0.006 |
| `max_depth` | 0.006 |
| `bagging_freq` | 0.006 |
| `extra_trees` | 0.006 |
| `feature_fraction` | 0.005 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 463 | 0.8302 | 0.0120 | 0.8046 |
| True | 37 | 0.8362 | 0.0092 | 0.8218 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.8333 | 0.8306 | 0.8288 | 0.8297 | **Q3** [0.0718, 0.1145] |
| `num_leaves` | 0.8291 | 0.8288 | 0.8320 | 0.8322 | **Q2** [20.0, 29.0] |
| `max_depth` | 0.8292 | 0.8298 | — | 0.8315 | **Q1** [None, 6.0] |
| `min_data_in_leaf` | 0.8362 | 0.8257 | 0.8258 | 0.8357 | **Q2** [14.0, 18.0] |
| `lambda_l1` | 0.8329 | 0.8299 | 0.8287 | 0.8309 | **Q3** [0.0, 0.0] |
| `lambda_l2` | 0.8348 | 0.8312 | 0.8293 | 0.8271 | **Q4** [0.8764, ∞) |
| `feature_fraction` | 0.8371 | 0.8291 | 0.8287 | 0.8275 | **Q4** [0.9381, ∞) |
| `bagging_fraction` | 0.8313 | 0.8281 | 0.8316 | 0.8315 | **Q2** [0.5466, 0.5716] |

#### E. Slice plot

![fred_gdp@s44/naive-lightgbm](slice_fred_gdp@s44_naive-lightgbm.png)


### moe

- **holdout RMSE: 1.48456** (winner retrained in 0.95s, cv score of winner: 0.8274)
- cv best RMSE: 0.8274, median: 0.9190, p10: 0.8459
- train: median 0.383s/fold, mean 0.507s, p90 0.629s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.902 |
| `lambda_l1` | 0.047 |
| `extra_trees` | 0.015 |
| `mixture_hard_m_step` | 0.012 |
| `mixture_expert_dropout_rate` | 0.004 |
| `mixture_init` | 0.004 |
| `max_depth` | 0.003 |
| `mixture_diversity_lambda` | 0.003 |
| `mixture_e_step_mode` | 0.002 |
| `mixture_balance_factor` | 0.002 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_e_step_mode` | **loss_only** | 0.9014 (n=358) | em | Δ +0.0285 | p=1.01e-04 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 448 | 0.9074 | 0.0596 | 0.8274 |
| leaf_reuse | 26 | 0.9291 | 0.0780 | 0.8405 |
| none | 26 | 0.9517 | 0.0965 | 0.8417 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 426 | 0.9101 | 0.0628 | 0.8274 |
| expert_choice | 74 | 0.9153 | 0.0706 | 0.8342 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 358 | 0.9014 | 0.0569 | 0.8274 |
| em | 114 | 0.9299 | 0.0689 | 0.8309 |
| gate_only | 28 | 0.9543 | 0.0891 | 0.8493 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 364 | 0.9022 | 0.0580 | 0.8274 |
| gmm | 24 | 0.9142 | 0.0400 | 0.8510 |
| tree_hierarchical | 33 | 0.9248 | 0.0778 | 0.8315 |
| gmm_features | 26 | 0.9296 | 0.0644 | 0.8382 |
| random | 20 | 0.9432 | 0.0845 | 0.8513 |
| kmeans_features | 33 | 0.9559 | 0.0787 | 0.8415 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 440 | 0.9068 | 0.0606 | 0.8274 |
| ema | 35 | 0.9374 | 0.0793 | 0.8405 |
| markov | 25 | 0.9444 | 0.0783 | 0.8432 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 465 | 0.9064 | 0.0615 | 0.8274 |
| False | 35 | 0.9703 | 0.0671 | 0.8519 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 469 | 0.9063 | 0.0620 | 0.8274 |
| False | 31 | 0.9800 | 0.0541 | 0.9018 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.9255 | — | 0.9068 | 0.9118 | **Q3** [4.0, 5.0] |
| `mixture_diversity_lambda` | 0.9174 | 0.9029 | 0.9051 | 0.9180 | **Q2** [0.1945, 0.2441] |
| `mixture_warmup_iters` | 0.9095 | 0.9033 | 0.9207 | 0.9117 | **Q2** [23.0, 28.0] |
| `mixture_balance_factor` | 0.9188 | — | 0.9062 | 0.9151 | **Q3** [4.0, 6.0] |
| `learning_rate` | 0.9197 | 0.9012 | 0.9169 | 0.9056 | **Q2** [0.0422, 0.0516] |
| `num_leaves` | 0.9156 | 0.8977 | 0.9061 | 0.9235 | **Q2** [69.0, 83.0] |
| `max_depth` | 0.9046 | 0.9052 | — | 0.9164 | **Q1** [None, 7.0] |
| `min_data_in_leaf` | 0.8806 | 0.8576 | 0.9357 | 0.9711 | **Q2** [9.0, 12.0] |

#### E. Slice plot

![fred_gdp@s44/moe](slice_fred_gdp@s44_moe.png)


---

## Overall recommendations

**Categorical settings that are statistically significant winners (p<0.01):**

| dataset | param | best value | Δ vs runner-up | p |
|---|---|---|---|---|
| fred_gdp@s42 | `mixture_gate_type` | **none** | +0.0500 | 0.00e+00 |
| fred_gdp@s42 | `mixture_routing_mode` | **token_choice** | +0.0326 | 2.00e-06 |
| fred_gdp@s42 | `mixture_e_step_mode` | **gate_only** | +0.0407 | 6.90e-05 |
| fred_gdp@s43 | `mixture_gate_type` | **none** | +0.0243 | 4.98e-03 |
| fred_gdp@s43 | `mixture_routing_mode` | **expert_choice** | +0.0552 | 0.00e+00 |
| fred_gdp@s44 | `mixture_e_step_mode` | **loss_only** | +0.0285 | 1.01e-04 |
