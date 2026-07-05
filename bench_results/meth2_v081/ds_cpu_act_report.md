# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 300

- **Datasets**: ['cpu_act'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `5232455cdc8d`, lib sha256 `a3c30fcc7fd3…`, package `/home/mokumoku/Develop/LightGBM-MoE/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| cpu_act | naive-lightgbm | **2.51321** ± 0.31795 | 2.45180 | 0.0% | 0.22 |
| cpu_act | moe | **2.42513** ± 0.28908 | 2.40656 | 0.0% | 2.28 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| cpu_act@s42 | naive-lightgbm | 2.5168 | 2.5887 | 0.161 | 264 |
| cpu_act@s42 | moe | 2.4623 | 2.6068 | 0.906 | 1824 |
| cpu_act@s43 | naive-lightgbm | 2.3606 | 2.4179 | 0.210 | 326 |
| cpu_act@s43 | moe | 2.3140 | 2.4026 | 0.790 | 1257 |
| cpu_act@s44 | naive-lightgbm | 2.4780 | 2.5416 | 0.165 | 258 |
| cpu_act@s44 | moe | 2.4434 | 2.5061 | 2.418 | 3314 |



---

## cpu_act@s42  (search X=[6554, 21], holdout n=1638)


### naive-lightgbm

- **holdout RMSE: 2.30096** (winner retrained in 0.13s, cv score of winner: 2.5168)
- cv best RMSE: 2.5168, median: 2.5887, p10: 2.5383
- train: median 0.161s/fold, mean 0.171s, p90 0.222s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.541 |
| `min_data_in_leaf` | 0.225 |
| `extra_trees` | 0.131 |
| `feature_fraction` | 0.063 |
| `max_depth` | 0.016 |
| `num_leaves` | 0.013 |
| `bagging_freq` | 0.005 |
| `bagging_fraction` | 0.004 |
| `lambda_l1` | 0.001 |
| `lambda_l2` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 278 | 2.7237 | 0.5702 | 2.5168 |
| True | 22 | 5.1620 | 2.8843 | 2.9974 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 3.5242 | 2.5987 | 2.6638 | 2.8232 | **Q2** [0.0878, 0.1015] |
| `num_leaves` | 2.7932 | 2.6426 | 2.7366 | 3.4317 | **Q2** [17.0, 22.0] |
| `max_depth` | 3.5982 | 2.7181 | — | 2.6996 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 2.8011 | 2.5874 | 2.6193 | 3.6265 | **Q2** [7.0, 9.0] |
| `lambda_l1` | 3.1783 | 3.0481 | 2.5990 | 2.7845 | **Q3** [0.101, 0.2619] |
| `lambda_l2` | 2.7125 | 2.6434 | 2.9249 | 3.3292 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 3.3781 | 2.7630 | 2.6453 | 2.8235 | **Q3** [0.86, 0.8808] |
| `bagging_fraction` | 2.8270 | 2.6803 | 3.2550 | 2.8477 | **Q2** [0.5773, 0.6504] |

#### E. Slice plot

![cpu_act@s42/naive-lightgbm](slice_cpu_act@s42_naive-lightgbm.png)


### moe

- **holdout RMSE: 2.23920** (winner retrained in 2.64s, cv score of winner: 2.4623)
- cv best RMSE: 2.4623, median: 2.6068, p10: 2.4914
- train: median 0.906s/fold, mean 1.198s, p90 2.208s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.515 |
| `extra_trees` | 0.135 |
| `mixture_diversity_lambda` | 0.121 |
| `min_data_in_leaf` | 0.072 |
| `bagging_fraction` | 0.047 |
| `lambda_l2` | 0.039 |
| `bagging_freq` | 0.020 |
| `mixture_warmup_iters` | 0.012 |
| `mixture_balance_factor` | 0.010 |
| `feature_fraction` | 0.007 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 244 | 3.3111 | 3.4651 | 2.4623 |
| gbdt | 30 | 4.9927 | 6.1027 | 2.5034 |
| none | 26 | 5.4331 | 5.2738 | 2.5866 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 245 | 3.4384 | 3.8126 | 2.4623 |
| token_choice | 55 | 4.6642 | 4.8731 | 2.5857 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| em | 170 | 3.4655 | 3.8109 | 2.4623 |
| gate_only | 89 | 3.8502 | 4.6786 | 2.5653 |
| loss_only | 41 | 4.0769 | 3.4958 | 2.4695 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 222 | 3.2678 | 3.7413 | 2.4623 |
| tree_hierarchical | 40 | 4.0216 | 3.0899 | 2.6258 |
| random | 16 | 4.6633 | 4.1256 | 2.7618 |
| gmm | 22 | 6.2741 | 6.6315 | 3.1499 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 220 | 3.3916 | 3.8205 | 2.4623 |
| markov | 63 | 4.0205 | 3.7580 | 2.5857 |
| none | 17 | 5.8528 | 6.5449 | 2.5945 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 240 | 3.4968 | 3.9023 | 2.4623 |
| True | 60 | 4.3285 | 4.5588 | 2.5857 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 265 | 3.2792 | 3.3358 | 2.4623 |
| True | 35 | 6.5701 | 6.8687 | 3.0286 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | 4.1098 | — | 3.3485 | **Q4** [4.0, ∞) |
| `mixture_diversity_lambda` | 4.3804 | 2.7317 | 3.4674 | 4.0732 | **Q2** [0.1238, 0.1661] |
| `mixture_warmup_iters` | 3.5492 | 3.5419 | 4.1139 | 3.4743 | **Q4** [41.0, ∞) |
| `mixture_balance_factor` | — | 3.2420 | 3.6335 | 4.1217 | **Q2** [2.0, 3.0] |
| `learning_rate` | 6.0818 | 2.7104 | 2.7357 | 3.1248 | **Q2** [0.1558, 0.1887] |
| `num_leaves` | 3.8267 | 4.0646 | 3.6768 | 3.0865 | **Q4** [92.25, ∞) |
| `max_depth` | 3.6810 | 3.1338 | — | 3.7775 | **Q2** [5.0, 6.0] |
| `min_data_in_leaf` | 3.5808 | 2.9696 | 2.7862 | 5.5393 | **Q3** [11.0, 19.0] |

#### E. Slice plot

![cpu_act@s42/moe](slice_cpu_act@s42_moe.png)


---

## cpu_act@s43  (search X=[6554, 21], holdout n=1638)


### naive-lightgbm

- **holdout RMSE: 2.96263** (winner retrained in 0.35s, cv score of winner: 2.3606)
- cv best RMSE: 2.3606, median: 2.4179, p10: 2.3767
- train: median 0.210s/fold, mean 0.212s, p90 0.290s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.545 |
| `learning_rate` | 0.327 |
| `extra_trees` | 0.047 |
| `num_leaves` | 0.030 |
| `bagging_fraction` | 0.025 |
| `feature_fraction` | 0.017 |
| `bagging_freq` | 0.004 |
| `max_depth` | 0.003 |
| `lambda_l1` | 0.002 |
| `lambda_l2` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 279 | 2.6156 | 0.7424 | 2.3606 |
| True | 21 | 3.9526 | 1.4165 | 2.8650 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 3.0578 | 2.5427 | 2.6238 | 2.6125 | **Q2** [0.0635, 0.0793] |
| `num_leaves` | 2.6429 | 2.5224 | 2.5676 | 3.0855 | **Q2** [20.0, 26.0] |
| `max_depth` | 3.0309 | 2.5906 | 2.5758 | 2.6500 | **Q3** [10.0, 11.0] |
| `min_data_in_leaf` | 2.4133 | 2.5333 | 2.4333 | 3.4038 | **Q1** [None, 6.0] |
| `lambda_l1` | 2.5531 | 2.6273 | 2.9238 | 2.7327 | **Q1** [None, 0.0] |
| `lambda_l2` | 2.7548 | 2.5349 | 2.6795 | 2.8676 | **Q2** [0.0001, 0.009] |
| `feature_fraction` | 2.8955 | 2.6856 | 2.7271 | 2.5286 | **Q4** [0.9543, ∞) |
| `bagging_fraction` | 2.7772 | 2.7105 | 2.6495 | 2.6996 | **Q3** [0.7722, 0.8744] |

#### E. Slice plot

![cpu_act@s43/naive-lightgbm](slice_cpu_act@s43_naive-lightgbm.png)


### moe

- **holdout RMSE: 2.83340** (winner retrained in 1.05s, cv score of winner: 2.3140)
- cv best RMSE: 2.3140, median: 2.4026, p10: 2.3413
- train: median 0.790s/fold, mean 0.824s, p90 1.181s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_gate_type` | 0.468 |
| `learning_rate` | 0.261 |
| `mixture_hard_m_step` | 0.228 |
| `mixture_diversity_lambda` | 0.008 |
| `num_leaves` | 0.008 |
| `mixture_refit_leaves` | 0.006 |
| `bagging_freq` | 0.005 |
| `mixture_e_step_mode` | 0.004 |
| `bagging_fraction` | 0.004 |
| `mixture_init` | 0.003 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 266 | 3.1136 | 3.0107 | 2.3140 |
| gbdt | 18 | 4.0626 | 2.1326 | 2.3721 |
| none | 16 | 15.4928 | 13.0750 | 4.7629 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 280 | 3.6877 | 4.9211 | 2.3140 |
| expert_choice | 20 | 5.8327 | 5.8111 | 2.5741 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 265 | 3.4971 | 4.6327 | 2.3140 |
| em | 17 | 6.2459 | 7.4159 | 2.5857 |
| loss_only | 18 | 6.4621 | 6.0853 | 2.5913 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gmm | 231 | 3.5989 | 4.8006 | 2.3140 |
| uniform | 38 | 3.7280 | 2.8564 | 2.3278 |
| tree_hierarchical | 16 | 5.2546 | 6.4304 | 2.4897 |
| random | 15 | 6.1423 | 8.6471 | 2.4511 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 79 | 3.5600 | 2.7358 | 2.3140 |
| none | 206 | 3.9046 | 5.7009 | 2.3206 |
| ema | 15 | 4.2423 | 4.0513 | 2.3676 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 274 | 3.5558 | 3.9941 | 2.3140 |
| False | 26 | 6.7286 | 10.6191 | 2.3278 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 278 | 3.7673 | 5.1621 | 2.3140 |
| True | 22 | 4.6324 | 2.3235 | 2.8686 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | — | — | 3.8307 | **Q4** [2.0, ∞) |
| `mixture_diversity_lambda` | 3.3907 | 2.9584 | 3.8226 | 5.1513 | **Q2** [0.152, 0.1788] |
| `mixture_warmup_iters` | 4.0415 | 3.2335 | 3.9730 | 4.0709 | **Q2** [24.75, 28.5] |
| `mixture_balance_factor` | 5.8107 | 3.3543 | — | 3.3847 | **Q2** [5.0, 6.0] |
| `learning_rate` | 5.6853 | 3.3653 | 2.7831 | 3.4892 | **Q3** [0.116, 0.1365] |
| `num_leaves` | 3.5355 | 3.6885 | 3.3865 | 4.7285 | **Q3** [32.0, 40.25] |
| `max_depth` | 3.9140 | — | — | 3.8049 | **Q4** [8.0, ∞) |
| `min_data_in_leaf` | 2.7071 | 3.3078 | 3.8189 | 5.2091 | **Q1** [None, 7.0] |

#### E. Slice plot

![cpu_act@s43/moe](slice_cpu_act@s43_moe.png)


---

## cpu_act@s44  (search X=[6554, 21], holdout n=1638)


### naive-lightgbm

- **holdout RMSE: 2.27606** (winner retrained in 0.17s, cv score of winner: 2.4780)
- cv best RMSE: 2.4780, median: 2.5416, p10: 2.5063
- train: median 0.165s/fold, mean 0.167s, p90 0.220s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.525 |
| `learning_rate` | 0.251 |
| `extra_trees` | 0.116 |
| `feature_fraction` | 0.052 |
| `num_leaves` | 0.030 |
| `bagging_freq` | 0.008 |
| `max_depth` | 0.008 |
| `bagging_fraction` | 0.007 |
| `lambda_l2` | 0.003 |
| `lambda_l1` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 280 | 2.7136 | 0.6089 | 2.4780 |
| True | 20 | 3.8921 | 1.1427 | 2.8796 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 3.1451 | 2.5864 | 2.6362 | 2.8009 | **Q2** [0.0864, 0.0971] |
| `num_leaves` | 2.6861 | 2.6349 | 2.6778 | 3.1672 | **Q2** [19.0, 23.0] |
| `max_depth` | 2.9955 | 2.8130 | — | 2.7386 | **Q4** [7.0, ∞) |
| `min_data_in_leaf` | 2.5623 | 2.5740 | 2.5874 | 3.4190 | **Q1** [None, 6.0] |
| `lambda_l1` | 2.6384 | 3.0447 | 2.6391 | 2.8463 | **Q1** [None, 0.0] |
| `lambda_l2` | 2.6267 | 2.8511 | 2.6385 | 3.0522 | **Q1** [None, 0.0] |
| `feature_fraction` | 3.2942 | 2.6950 | 2.5976 | 2.5817 | **Q4** [0.9903, ∞) |
| `bagging_fraction` | 3.1352 | 2.7428 | 2.6153 | 2.6753 | **Q3** [0.9574, 0.9762] |

#### E. Slice plot

![cpu_act@s44/naive-lightgbm](slice_cpu_act@s44_naive-lightgbm.png)


### moe

- **holdout RMSE: 2.20279** (winner retrained in 3.13s, cv score of winner: 2.4434)
- cv best RMSE: 2.4434, median: 2.5061, p10: 2.4615
- train: median 2.418s/fold, mean 2.187s, p90 2.894s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.424 |
| `mixture_init` | 0.241 |
| `min_data_in_leaf` | 0.067 |
| `feature_fraction` | 0.064 |
| `bagging_fraction` | 0.062 |
| `max_depth` | 0.040 |
| `num_leaves` | 0.023 |
| `mixture_e_step_mode` | 0.021 |
| `mixture_gate_type` | 0.017 |
| `bagging_freq` | 0.010 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_init` | **uniform** | 2.9670 (n=244) | tree_hierarchical | Δ +0.7020 | p=2.60e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 239 | 3.2016 | 3.3744 | 2.4434 |
| gbdt | 34 | 5.2745 | 7.9049 | 2.4466 |
| none | 27 | 7.3371 | 10.4888 | 2.4777 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 216 | 3.3970 | 3.9114 | 2.4434 |
| token_choice | 84 | 4.8672 | 7.6207 | 2.4466 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| em | 145 | 3.2747 | 3.2065 | 2.4465 |
| loss_only | 120 | 3.7675 | 5.6128 | 2.4434 |
| gate_only | 35 | 6.1620 | 8.9553 | 2.4733 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 244 | 2.9670 | 2.8978 | 2.4434 |
| tree_hierarchical | 25 | 3.6690 | 0.6571 | 2.7851 |
| random | 15 | 7.5860 | 8.0410 | 2.6231 |
| gmm | 16 | 13.3211 | 14.6819 | 3.0624 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 170 | 3.1766 | 2.8735 | 2.4465 |
| ema | 101 | 4.0842 | 5.9329 | 2.4434 |
| none | 29 | 6.5543 | 10.2732 | 2.5073 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 264 | 3.4606 | 4.5371 | 2.4434 |
| True | 36 | 6.3616 | 8.5186 | 2.6902 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 264 | 3.5380 | 4.7516 | 2.4434 |
| True | 36 | 5.7940 | 7.8028 | 3.0339 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 6.4299 | — | — | 3.2970 | **Q4** [4.0, ∞) |
| `mixture_diversity_lambda` | 5.5338 | 3.9483 | 2.8018 | 2.9509 | **Q3** [0.4261, 0.4588] |
| `mixture_warmup_iters` | 6.4075 | 3.4812 | 2.8855 | 2.8087 | **Q4** [49.0, ∞) |
| `mixture_balance_factor` | 6.6520 | 3.4831 | — | 2.9893 | **Q4** [8.0, ∞) |
| `learning_rate` | 6.8362 | 2.7500 | 2.7029 | 2.9456 | **Q3** [0.1815, 0.2191] |
| `num_leaves` | 4.5199 | 3.1052 | 3.6820 | 3.9498 | **Q2** [48.0, 62.0] |
| `max_depth` | 6.6434 | 3.4673 | — | 2.9945 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 3.9329 | 2.7633 | 2.7539 | 6.0361 | **Q3** [9.0, 14.25] |

#### E. Slice plot

![cpu_act@s44/moe](slice_cpu_act@s44_moe.png)


---

## Overall recommendations

**Categorical settings that are statistically significant winners (p<0.01):**

| dataset | param | best value | Δ vs runner-up | p |
|---|---|---|---|---|
| cpu_act@s44 | `mixture_init` | **uniform** | +0.7020 | 2.60e-03 |
