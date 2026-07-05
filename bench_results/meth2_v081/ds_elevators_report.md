# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 300

- **Datasets**: ['elevators'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `5232455cdc8d`, lib sha256 `a3c30fcc7fd3…`, package `/home/mokumoku/Develop/LightGBM-MoE/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| elevators | naive-lightgbm | **0.00238** ± 0.00005 | 0.00264 | 0.0% | 0.23 |
| elevators | moe | **0.00230** ± 0.00010 | 0.00251 | 0.0% | 2.04 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| elevators@s42 | naive-lightgbm | 0.0026 | 0.0027 | 0.124 | 206 |
| elevators@s42 | moe | 0.0025 | 0.0026 | 1.690 | 2307 |
| elevators@s43 | naive-lightgbm | 0.0027 | 0.0027 | 0.149 | 244 |
| elevators@s43 | moe | 0.0026 | 0.0028 | 0.763 | 1337 |
| elevators@s44 | naive-lightgbm | 0.0026 | 0.0027 | 0.227 | 342 |
| elevators@s44 | moe | 0.0025 | 0.0026 | 2.152 | 3529 |



---

## elevators@s42  (search X=[8000, 18], holdout n=2000)


### naive-lightgbm

- **holdout RMSE: 0.00245** (winner retrained in 0.14s, cv score of winner: 0.0026)
- cv best RMSE: 0.0026, median: 0.0027, p10: 0.0027
- train: median 0.124s/fold, mean 0.132s, p90 0.181s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.733 |
| `learning_rate` | 0.188 |
| `extra_trees` | 0.032 |
| `bagging_freq` | 0.019 |
| `feature_fraction` | 0.017 |
| `num_leaves` | 0.004 |
| `bagging_fraction` | 0.003 |
| `max_depth` | 0.002 |
| `min_data_in_leaf` | 0.002 |
| `lambda_l2` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 278 | 0.0028 | 0.0004 | 0.0026 |
| True | 22 | 0.0036 | 0.0009 | 0.0028 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0033 | 0.0027 | 0.0027 | 0.0028 | **Q2** [0.1755, 0.2043] |
| `num_leaves` | 0.0029 | 0.0027 | 0.0028 | 0.0031 | **Q2** [14.0, 18.0] |
| `max_depth` | 0.0031 | — | — | 0.0028 | **Q4** [6.0, ∞) |
| `min_data_in_leaf` | 0.0028 | 0.0028 | 0.0029 | 0.0031 | **Q1** [None, 11.0] |
| `lambda_l1` | 0.0029 | 0.0029 | 0.0028 | 0.0030 | **Q3** [0.0003, 0.0015] |
| `lambda_l2` | 0.0031 | 0.0028 | 0.0028 | 0.0029 | **Q2** [0.0001, 0.0003] |
| `feature_fraction` | 0.0031 | 0.0028 | 0.0028 | 0.0028 | **Q2** [0.9043, 0.9313] |
| `bagging_fraction` | 0.0031 | 0.0028 | 0.0028 | 0.0029 | **Q2** [0.9313, 0.9559] |

#### E. Slice plot

![elevators@s42/naive-lightgbm](slice_elevators@s42_naive-lightgbm.png)


### moe

- **holdout RMSE: 0.00232** (winner retrained in 2.20s, cv score of winner: 0.0025)
- cv best RMSE: 0.0025, median: 0.0026, p10: 0.0025
- train: median 1.690s/fold, mean 1.512s, p90 2.060s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.346 |
| `mixture_diversity_lambda` | 0.154 |
| `mixture_r_smoothing` | 0.097 |
| `lambda_l1` | 0.076 |
| `num_leaves` | 0.058 |
| `max_depth` | 0.053 |
| `mixture_warmup_iters` | 0.049 |
| `mixture_e_step_mode` | 0.028 |
| `mixture_init` | 0.027 |
| `mixture_gate_type` | 0.027 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 255 | 0.0038 | 0.0055 | 0.0025 |
| gbdt | 24 | 0.0046 | 0.0030 | 0.0028 |
| none | 21 | 0.0446 | 0.1652 | 0.0031 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 40 | 0.0058 | 0.0062 | 0.0028 |
| token_choice | 260 | 0.0068 | 0.0485 | 0.0025 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 32 | 0.0042 | 0.0028 | 0.0028 |
| gate_only | 247 | 0.0067 | 0.0496 | 0.0025 |
| em | 21 | 0.0106 | 0.0155 | 0.0031 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 237 | 0.0034 | 0.0037 | 0.0025 |
| random | 33 | 0.0039 | 0.0022 | 0.0025 |
| gmm | 15 | 0.0103 | 0.0178 | 0.0029 |
| tree_hierarchical | 15 | 0.0615 | 0.1928 | 0.0038 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 90 | 0.0037 | 0.0036 | 0.0025 |
| none | 36 | 0.0047 | 0.0037 | 0.0026 |
| ema | 174 | 0.0087 | 0.0592 | 0.0025 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 57 | 0.0053 | 0.0055 | 0.0027 |
| True | 243 | 0.0070 | 0.0502 | 0.0025 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 35 | 0.0055 | 0.0046 | 0.0030 |
| False | 265 | 0.0069 | 0.0481 | 0.0025 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.0060 | — | 0.0094 | 0.0030 | **Q4** [4.0, ∞) |
| `mixture_diversity_lambda` | 0.0052 | 0.0031 | 0.0132 | 0.0052 | **Q2** [0.3096, 0.3476] |
| `mixture_warmup_iters` | 0.0039 | 0.0031 | 0.0049 | 0.0141 | **Q2** [12.0, 14.5] |
| `mixture_balance_factor` | 0.0044 | 0.0153 | — | 0.0032 | **Q4** [10.0, ∞) |
| `learning_rate` | 0.0060 | 0.0044 | 0.0032 | 0.0132 | **Q3** [0.2608, 0.2821] |
| `num_leaves` | 0.0151 | 0.0049 | 0.0032 | 0.0040 | **Q3** [86.0, 101.0] |
| `max_depth` | 0.0037 | — | 0.0079 | 0.0052 | **Q1** [None, 5.0] |
| `min_data_in_leaf` | 0.0029 | 0.0037 | 0.0034 | 0.0163 | **Q1** [None, 9.0] |

#### E. Slice plot

![elevators@s42/moe](slice_elevators@s42_moe.png)


---

## elevators@s43  (search X=[8000, 18], holdout n=2000)


### naive-lightgbm

- **holdout RMSE: 0.00236** (winner retrained in 0.15s, cv score of winner: 0.0027)
- cv best RMSE: 0.0027, median: 0.0027, p10: 0.0027
- train: median 0.149s/fold, mean 0.157s, p90 0.230s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.610 |
| `learning_rate` | 0.147 |
| `feature_fraction` | 0.056 |
| `num_leaves` | 0.055 |
| `min_data_in_leaf` | 0.043 |
| `max_depth` | 0.039 |
| `extra_trees` | 0.030 |
| `bagging_fraction` | 0.015 |
| `lambda_l2` | 0.004 |
| `bagging_freq` | 0.002 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 279 | 0.0028 | 0.0005 | 0.0027 |
| True | 21 | 0.0037 | 0.0007 | 0.0029 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0033 | 0.0027 | 0.0028 | 0.0027 | **Q2** [0.1548, 0.1835] |
| `num_leaves` | 0.0028 | 0.0028 | 0.0029 | 0.0031 | **Q1** [None, 17.0] |
| `max_depth` | 0.0034 | 0.0029 | — | 0.0029 | **Q2** [5.0, 8.0] |
| `min_data_in_leaf` | 0.0028 | 0.0028 | 0.0027 | 0.0032 | **Q3** [20.5, 24.0] |
| `lambda_l1` | 0.0029 | 0.0028 | 0.0028 | 0.0031 | **Q2** [0.0, 0.0004] |
| `lambda_l2` | 0.0029 | 0.0029 | 0.0029 | 0.0030 | **Q1** [None, 0.0001] |
| `feature_fraction` | 0.0032 | 0.0029 | 0.0028 | 0.0027 | **Q4** [0.9712, ∞) |
| `bagging_fraction` | 0.0031 | 0.0029 | 0.0029 | 0.0028 | **Q4** [0.9656, ∞) |

#### E. Slice plot

![elevators@s43/naive-lightgbm](slice_elevators@s43_naive-lightgbm.png)


### moe

- **holdout RMSE: 0.00240** (winner retrained in 0.54s, cv score of winner: 0.0026)
- cv best RMSE: 0.0026, median: 0.0028, p10: 0.0026
- train: median 0.763s/fold, mean 0.874s, p90 1.305s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.224 |
| `mixture_hard_m_step` | 0.185 |
| `mixture_balance_factor` | 0.170 |
| `max_depth` | 0.083 |
| `learning_rate` | 0.081 |
| `mixture_routing_mode` | 0.060 |
| `bagging_freq` | 0.056 |
| `bagging_fraction` | 0.040 |
| `num_leaves` | 0.034 |
| `mixture_gate_type` | 0.023 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 268 | 0.0038 | 0.0097 | 0.0026 |
| none | 16 | 0.0062 | 0.0046 | 0.0032 |
| leaf_reuse | 16 | 0.0078 | 0.0063 | 0.0028 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 40 | 0.0041 | 0.0022 | 0.0028 |
| token_choice | 260 | 0.0042 | 0.0100 | 0.0026 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 117 | 0.0037 | 0.0025 | 0.0027 |
| em | 168 | 0.0043 | 0.0122 | 0.0026 |
| gate_only | 15 | 0.0072 | 0.0056 | 0.0028 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gmm | 29 | 0.0041 | 0.0033 | 0.0027 |
| tree_hierarchical | 233 | 0.0041 | 0.0104 | 0.0026 |
| uniform | 24 | 0.0042 | 0.0033 | 0.0028 |
| random | 14 | 0.0061 | 0.0068 | 0.0026 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 24 | 0.0040 | 0.0021 | 0.0028 |
| markov | 253 | 0.0041 | 0.0101 | 0.0026 |
| none | 23 | 0.0049 | 0.0041 | 0.0027 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 281 | 0.0034 | 0.0025 | 0.0026 |
| False | 19 | 0.0150 | 0.0341 | 0.0027 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 172 | 0.0039 | 0.0033 | 0.0027 |
| True | 128 | 0.0045 | 0.0138 | 0.0026 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | — | — | 0.0042 | **Q4** [2.0, ∞) |
| `mixture_diversity_lambda` | 0.0035 | 0.0054 | 0.0037 | 0.0042 | **Q1** [None, 0.1484] |
| `mixture_warmup_iters` | 0.0034 | 0.0049 | 0.0045 | 0.0038 | **Q1** [None, 12.0] |
| `mixture_balance_factor` | 0.0066 | 0.0043 | — | 0.0032 | **Q4** [10.0, ∞) |
| `learning_rate` | 0.0052 | 0.0034 | 0.0031 | 0.0050 | **Q3** [0.2295, 0.2635] |
| `num_leaves` | 0.0031 | 0.0039 | 0.0061 | 0.0035 | **Q1** [None, 15.0] |
| `max_depth` | 0.0085 | 0.0034 | 0.0034 | 0.0044 | **Q2** [4.0, 7.0] |
| `min_data_in_leaf` | 0.0061 | 0.0034 | 0.0032 | 0.0043 | **Q3** [20.0, 24.25] |

#### E. Slice plot

![elevators@s43/moe](slice_elevators@s43_moe.png)


---

## elevators@s44  (search X=[8000, 18], holdout n=2000)


### naive-lightgbm

- **holdout RMSE: 0.00234** (winner retrained in 0.41s, cv score of winner: 0.0026)
- cv best RMSE: 0.0026, median: 0.0027, p10: 0.0027
- train: median 0.227s/fold, mean 0.222s, p90 0.323s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.631 |
| `feature_fraction` | 0.092 |
| `learning_rate` | 0.090 |
| `bagging_fraction` | 0.051 |
| `bagging_freq` | 0.050 |
| `max_depth` | 0.038 |
| `num_leaves` | 0.027 |
| `min_data_in_leaf` | 0.016 |
| `extra_trees` | 0.002 |
| `lambda_l2` | 0.002 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 279 | 0.0028 | 0.0005 | 0.0026 |
| True | 21 | 0.0035 | 0.0008 | 0.0029 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0032 | 0.0027 | 0.0027 | 0.0029 | **Q2** [0.1153, 0.1319] |
| `num_leaves` | 0.0031 | 0.0029 | 0.0028 | 0.0027 | **Q4** [124.0, ∞) |
| `max_depth` | 0.0031 | 0.0028 | 0.0028 | 0.0028 | **Q2** [6.0, 7.0] |
| `min_data_in_leaf` | 0.0029 | 0.0027 | 0.0028 | 0.0031 | **Q2** [15.0, 19.0] |
| `lambda_l1` | 0.0030 | 0.0027 | 0.0028 | 0.0031 | **Q2** [0.0, 0.0002] |
| `lambda_l2` | 0.0028 | 0.0028 | 0.0028 | 0.0030 | **Q1** [None, 0.0003] |
| `feature_fraction` | 0.0032 | 0.0028 | 0.0028 | 0.0027 | **Q4** [0.9776, ∞) |
| `bagging_fraction` | 0.0031 | 0.0029 | 0.0027 | 0.0027 | **Q3** [0.9628, 0.9815] |

#### E. Slice plot

![elevators@s44/naive-lightgbm](slice_elevators@s44_naive-lightgbm.png)


### moe

- **holdout RMSE: 0.00217** (winner retrained in 3.39s, cv score of winner: 0.0025)
- cv best RMSE: 0.0025, median: 0.0026, p10: 0.0025
- train: median 2.152s/fold, mean 2.322s, p90 4.357s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.675 |
| `lambda_l2` | 0.087 |
| `extra_trees` | 0.062 |
| `mixture_diversity_lambda` | 0.043 |
| `mixture_r_smoothing` | 0.030 |
| `mixture_gate_type` | 0.016 |
| `mixture_init` | 0.016 |
| `num_leaves` | 0.015 |
| `mixture_warmup_iters` | 0.015 |
| `mixture_refit_leaves` | 0.014 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 239 | 0.0029 | 0.0020 | 0.0025 |
| gbdt | 45 | 0.0040 | 0.0032 | 0.0026 |
| none | 16 | 0.0055 | 0.0039 | 0.0029 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 280 | 0.0030 | 0.0018 | 0.0025 |
| expert_choice | 20 | 0.0060 | 0.0059 | 0.0027 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 267 | 0.0030 | 0.0019 | 0.0025 |
| loss_only | 16 | 0.0051 | 0.0041 | 0.0027 |
| em | 17 | 0.0054 | 0.0047 | 0.0028 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 166 | 0.0029 | 0.0018 | 0.0025 |
| random | 82 | 0.0030 | 0.0011 | 0.0025 |
| tree_hierarchical | 40 | 0.0045 | 0.0044 | 0.0026 |
| gmm | 12 | 0.0050 | 0.0046 | 0.0026 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 211 | 0.0031 | 0.0024 | 0.0025 |
| none | 38 | 0.0035 | 0.0028 | 0.0025 |
| ema | 51 | 0.0038 | 0.0020 | 0.0026 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 279 | 0.0031 | 0.0023 | 0.0025 |
| False | 21 | 0.0044 | 0.0036 | 0.0026 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 259 | 0.0031 | 0.0022 | 0.0025 |
| True | 41 | 0.0042 | 0.0032 | 0.0027 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | 0.0033 | 0.0046 | 0.0029 | **Q4** [4.0, ∞) |
| `mixture_diversity_lambda` | 0.0028 | 0.0029 | 0.0035 | 0.0036 | **Q1** [None, 0.0284] |
| `mixture_warmup_iters` | 0.0028 | 0.0033 | 0.0030 | 0.0038 | **Q1** [None, 8.0] |
| `mixture_balance_factor` | 0.0042 | 0.0033 | — | 0.0030 | **Q4** [9.0, ∞) |
| `learning_rate` | 0.0041 | 0.0027 | 0.0029 | 0.0032 | **Q2** [0.1764, 0.2192] |
| `num_leaves` | 0.0043 | 0.0027 | 0.0030 | 0.0029 | **Q2** [90.75, 99.5] |
| `max_depth` | 0.0038 | 0.0032 | — | 0.0030 | **Q4** [8.0, ∞) |
| `min_data_in_leaf` | 0.0035 | 0.0028 | 0.0029 | 0.0036 | **Q2** [17.0, 21.0] |

#### E. Slice plot

![elevators@s44/moe](slice_elevators@s44_moe.png)


---

## Overall recommendations

(no categorical settings were universally significant — see per-dataset breakdown)
