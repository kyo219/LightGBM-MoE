# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 300

- **Datasets**: ['houses'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `5232455cdc8d`, lib sha256 `5cec0a0bd5ab…`, package `/tmp/lgbm-moe-v080/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| houses | naive-lightgbm | **49805.38271** ± 962.89058 | 50811.46230 | 0.0% | 0.49 |
| houses | moe | **49077.47792** ± 338.55158 | 50401.78870 | 0.0% | 3.46 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| houses@s42 | naive-lightgbm | 50683.0901 | 51224.1336 | 0.198 | 301 |
| houses@s42 | moe | 50407.5337 | 51276.1226 | 1.138 | 2413 |
| houses@s43 | naive-lightgbm | 51076.3747 | 51723.6494 | 0.342 | 497 |
| houses@s43 | moe | 50286.4728 | 51305.1561 | 0.769 | 1489 |
| houses@s44 | naive-lightgbm | 50674.9222 | 51222.7143 | 0.430 | 571 |
| houses@s44 | moe | 50511.3595 | 51281.7718 | 0.996 | 2730 |



---

## houses@s42  (search X=[8000, 8], holdout n=2000)


### naive-lightgbm

- **holdout RMSE: 50760.44294** (winner retrained in 0.20s, cv score of winner: 50683.0901)
- cv best RMSE: 50683.0901, median: 51224.1336, p10: 50895.5274
- train: median 0.198s/fold, mean 0.195s, p90 0.244s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.675 |
| `extra_trees` | 0.286 |
| `min_data_in_leaf` | 0.010 |
| `feature_fraction` | 0.009 |
| `bagging_fraction` | 0.007 |
| `max_depth` | 0.006 |
| `num_leaves` | 0.004 |
| `bagging_freq` | 0.002 |
| `lambda_l2` | 0.000 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 278 | 51888.9305 | 3228.7653 | 50683.0901 |
| True | 22 | 65625.9156 | 10037.1753 | 57820.1777 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 55845.0459 | 51310.6123 | 51856.6872 | 52572.8922 | **Q2** [0.0884, 0.1021] |
| `num_leaves` | 53047.3600 | 51953.3869 | 51484.0810 | 55174.9507 | **Q3** [36.0, 44.0] |
| `max_depth` | 56986.4618 | 52400.8256 | — | 51756.6192 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 52753.3685 | 51803.1950 | 51974.3697 | 54945.4953 | **Q2** [17.0, 21.0] |
| `lambda_l1` | 53364.3411 | 52532.0795 | 53811.8016 | 51877.0154 | **Q4** [0.1035, ∞) |
| `lambda_l2` | 55110.6574 | 51893.7303 | 51780.1980 | 52800.6519 | **Q3** [0.1017, 0.379] |
| `feature_fraction` | 55369.3077 | 51491.1940 | 51682.8047 | 53041.9312 | **Q2** [0.7235, 0.7507] |
| `bagging_fraction` | 53204.3996 | 52442.3521 | 53142.7276 | 52795.7583 | **Q2** [0.6939, 0.7489] |

#### E. Slice plot

![houses@s42/naive-lightgbm](slice_houses@s42_naive-lightgbm.png)


### moe

- **holdout RMSE: 49530.87620** (winner retrained in 8.95s, cv score of winner: 50407.5337)
- cv best RMSE: 50407.5337, median: 51276.1226, p10: 50724.3187
- train: median 1.138s/fold, mean 1.591s, p90 4.071s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_hard_m_step` | 0.389 |
| `extra_trees` | 0.254 |
| `learning_rate` | 0.220 |
| `feature_fraction` | 0.034 |
| `bagging_fraction` | 0.023 |
| `mixture_gate_type` | 0.015 |
| `num_leaves` | 0.015 |
| `min_data_in_leaf` | 0.006 |
| `mixture_diversity_lambda` | 0.006 |
| `mixture_num_experts` | 0.006 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_routing_mode` | **expert_choice** | 53388.0952 (n=275) | token_choice | Δ +12290.6428 | p=5.83e-04 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 195 | 53310.1990 | 6867.1164 | 50559.7323 |
| none | 49 | 55523.8300 | 10168.5848 | 50407.5337 |
| gbdt | 56 | 57277.4673 | 12995.4211 | 51074.0547 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 275 | 53388.0952 | 7455.6643 | 50407.5337 |
| token_choice | 25 | 65678.7380 | 15137.3998 | 51158.6269 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 177 | 52936.9608 | 5894.2778 | 50407.5337 |
| gate_only | 68 | 56376.4533 | 10080.6978 | 50829.7399 |
| em | 55 | 56731.8865 | 13799.5880 | 50526.9366 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| random | 86 | 53377.9221 | 6767.5864 | 50407.5337 |
| uniform | 187 | 53792.6868 | 8015.7289 | 50702.9195 |
| gmm | 12 | 58525.4573 | 8490.6653 | 51718.7702 |
| tree_hierarchical | 15 | 64777.0275 | 19458.4270 | 51097.0453 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 171 | 53565.3070 | 7523.4264 | 50559.7323 |
| markov | 61 | 54220.2270 | 7626.0443 | 50407.5337 |
| none | 68 | 56714.6073 | 12582.0059 | 51008.1911 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 280 | 53135.2819 | 7227.0901 | 50407.5337 |
| True | 20 | 72290.7855 | 12252.8821 | 54084.4628 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 279 | 53248.3850 | 7353.6555 | 50407.5337 |
| True | 21 | 69875.9627 | 13783.8080 | 57075.3232 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | 54753.4678 | — | 54105.7102 | **Q4** [4.0, ∞) |
| `mixture_diversity_lambda` | 51747.1781 | 54390.0201 | 53146.2985 | 58365.7652 | **Q1** [None, 0.041] |
| `mixture_warmup_iters` | 52022.6404 | 53644.2177 | 52869.0906 | 57823.9983 | **Q1** [None, 6.0] |
| `mixture_balance_factor` | 58595.2707 | 56340.4032 | — | 52788.1832 | **Q4** [9.0, ∞) |
| `learning_rate` | 59624.4941 | 51869.8134 | 52353.7975 | 53801.1569 | **Q2** [0.1152, 0.1363] |
| `num_leaves` | 57338.1101 | 52234.5327 | 55604.5945 | 52695.2379 | **Q2** [71.0, 86.5] |
| `max_depth` | 57919.9116 | — | — | 53461.1029 | **Q4** [10.0, ∞) |
| `min_data_in_leaf` | 53450.9262 | 52759.8236 | 54572.3092 | 56860.4658 | **Q2** [15.0, 22.0] |

#### E. Slice plot

![houses@s42/moe](slice_houses@s42_moe.png)


---

## houses@s43  (search X=[8000, 8], holdout n=2000)


### naive-lightgbm

- **holdout RMSE: 50168.46687** (winner retrained in 0.56s, cv score of winner: 51076.3747)
- cv best RMSE: 51076.3747, median: 51723.6494, p10: 51396.0980
- train: median 0.342s/fold, mean 0.325s, p90 0.439s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.557 |
| `learning_rate` | 0.273 |
| `feature_fraction` | 0.067 |
| `max_depth` | 0.051 |
| `min_data_in_leaf` | 0.028 |
| `bagging_freq` | 0.013 |
| `bagging_fraction` | 0.006 |
| `num_leaves` | 0.002 |
| `lambda_l2` | 0.001 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 279 | 52383.8445 | 2614.0622 | 51076.3747 |
| True | 21 | 64191.2960 | 5869.1887 | 57331.5582 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 54846.3206 | 52554.2688 | 52209.4324 | 53231.4426 | **Q3** [0.0676, 0.0786] |
| `num_leaves` | 54728.3649 | 52898.7171 | 52641.8450 | 52584.7191 | **Q4** [117.0, ∞) |
| `max_depth` | 55898.8596 | 52305.1409 | 53184.8689 | 53192.0540 | **Q2** [9.0, 10.0] |
| `min_data_in_leaf` | 52738.2128 | 52435.7065 | 52297.1280 | 55178.8607 | **Q3** [17.0, 24.0] |
| `lambda_l1` | 53621.7121 | 52673.3426 | 52469.2696 | 54077.1402 | **Q3** [0.0, 0.0001] |
| `lambda_l2` | 53797.3334 | 53224.9688 | 52603.0649 | 53216.0974 | **Q3** [0.222, 0.8473] |
| `feature_fraction` | 54145.7746 | 52498.8939 | 52392.1659 | 53804.6301 | **Q3** [0.7851, 0.8138] |
| `bagging_fraction` | 53940.5034 | 52654.5802 | 53087.9937 | 53158.3872 | **Q2** [0.782, 0.8405] |

#### E. Slice plot

![houses@s43/naive-lightgbm](slice_houses@s43_naive-lightgbm.png)


### moe

- **holdout RMSE: 48984.00134** (winner retrained in 0.90s, cv score of winner: 50286.4728)
- cv best RMSE: 50286.4728, median: 51305.1561, p10: 50601.9288
- train: median 0.769s/fold, mean 0.978s, p90 1.279s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.495 |
| `mixture_hard_m_step` | 0.202 |
| `extra_trees` | 0.100 |
| `mixture_balance_factor` | 0.033 |
| `mixture_init` | 0.031 |
| `feature_fraction` | 0.030 |
| `num_leaves` | 0.020 |
| `mixture_e_step_mode` | 0.015 |
| `mixture_diversity_lambda` | 0.014 |
| `mixture_num_experts` | 0.010 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_r_smoothing` | **ema** | 52782.9066 (n=256) | markov | Δ +6136.9992 | p=1.90e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 178 | 53113.2660 | 6812.2550 | 50286.4728 |
| gbdt | 97 | 54818.0059 | 8653.2220 | 50917.9606 |
| leaf_reuse | 25 | 56687.1698 | 8241.7661 | 51311.1350 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 213 | 53547.5762 | 7918.8790 | 50286.4728 |
| expert_choice | 87 | 54977.6257 | 6884.4908 | 51138.4745 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 177 | 52945.4514 | 5721.2863 | 50286.4728 |
| loss_only | 98 | 54430.1107 | 7917.3156 | 50600.9454 |
| em | 25 | 59327.6571 | 13777.9729 | 52121.3896 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| random | 110 | 53701.2203 | 8169.1478 | 50600.9454 |
| tree_hierarchical | 141 | 53781.7069 | 7724.4142 | 50286.4728 |
| gmm | 23 | 54722.5486 | 3387.2029 | 51733.6953 |
| uniform | 26 | 55373.6023 | 7625.4666 | 51884.4858 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 256 | 52782.9066 | 5744.1865 | 50286.4728 |
| markov | 27 | 58919.9058 | 8950.4682 | 52121.3896 |
| none | 17 | 63848.5659 | 15932.9593 | 51755.7322 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 277 | 52668.3577 | 5140.4971 | 50286.4728 |
| True | 23 | 69545.7428 | 13574.1233 | 52536.4317 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 280 | 53227.4708 | 6913.9159 | 50286.4728 |
| True | 20 | 64249.7681 | 9885.0296 | 56498.4203 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 55720.4137 | — | — | 53683.9777 | **Q4** [3.0, ∞) |
| `mixture_diversity_lambda` | 53900.2291 | 52477.6863 | 52934.5061 | 56536.7409 | **Q2** [0.0173, 0.0362] |
| `mixture_warmup_iters` | 52606.4120 | 52039.5121 | 55977.3914 | 55034.2302 | **Q2** [8.0, 12.0] |
| `mixture_balance_factor` | — | — | 52709.2397 | 56584.6549 | **Q3** [2.0, 3.0] |
| `learning_rate` | 58100.8467 | 52819.2616 | 52767.6855 | 52161.3687 | **Q4** [0.2488, ∞) |
| `num_leaves` | 53434.8033 | 51903.3966 | 53952.1940 | 56605.9507 | **Q2** [29.0, 34.0] |
| `max_depth` | 54917.4674 | — | 52868.4079 | 54899.0230 | **Q3** [9.0, 10.0] |
| `min_data_in_leaf` | 52737.3513 | 53553.1347 | 53868.2030 | 55562.3937 | **Q1** [None, 11.0] |

#### E. Slice plot

![houses@s43/moe](slice_houses@s43_moe.png)


---

## houses@s44  (search X=[8000, 8], holdout n=2000)


### naive-lightgbm

- **holdout RMSE: 48487.23831** (winner retrained in 0.71s, cv score of winner: 50674.9222)
- cv best RMSE: 50674.9222, median: 51222.7143, p10: 50855.8811
- train: median 0.430s/fold, mean 0.374s, p90 0.572s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.668 |
| `learning_rate` | 0.247 |
| `min_data_in_leaf` | 0.029 |
| `max_depth` | 0.026 |
| `feature_fraction` | 0.015 |
| `bagging_fraction` | 0.006 |
| `num_leaves` | 0.006 |
| `bagging_freq` | 0.003 |
| `lambda_l1` | 0.000 |
| `lambda_l2` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 280 | 52023.0161 | 2849.5999 | 50674.9222 |
| True | 20 | 65091.4977 | 4234.1415 | 58652.9668 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 55034.4959 | 51555.5742 | 52070.6520 | 52916.2707 | **Q2** [0.0547, 0.0629] |
| `num_leaves` | 54040.3712 | 53077.1628 | 52163.2740 | 52344.9255 | **Q3** [110.0, 116.25] |
| `max_depth` | 54770.7357 | 53077.9487 | — | 52205.8020 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 51813.2123 | 51580.5544 | 52392.5069 | 55575.1304 | **Q2** [12.0, 15.0] |
| `lambda_l1` | 54635.0590 | 52656.6604 | 51978.5144 | 52306.7590 | **Q3** [0.1121, 0.6478] |
| `lambda_l2` | 54169.3509 | 52512.7667 | 51859.7302 | 53035.1450 | **Q3** [0.0008, 0.0033] |
| `feature_fraction` | 54831.5402 | 52020.4412 | 51598.4895 | 53126.5219 | **Q3** [0.7333, 0.7547] |
| `bagging_fraction` | 53409.2827 | 52100.9552 | 53134.6432 | 52932.1117 | **Q2** [0.8281, 0.8641] |

#### E. Slice plot

![houses@s44/naive-lightgbm](slice_houses@s44_naive-lightgbm.png)


### moe

- **holdout RMSE: 48717.55622** (winner retrained in 0.51s, cv score of winner: 50511.3595)
- cv best RMSE: 50511.3595, median: 51281.7718, p10: 50680.9809
- train: median 0.996s/fold, mean 1.802s, p90 3.177s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.548 |
| `extra_trees` | 0.211 |
| `mixture_hard_m_step` | 0.085 |
| `feature_fraction` | 0.045 |
| `bagging_freq` | 0.018 |
| `mixture_e_step_mode` | 0.017 |
| `mixture_diversity_lambda` | 0.016 |
| `max_depth` | 0.016 |
| `mixture_warmup_iters` | 0.010 |
| `min_data_in_leaf` | 0.007 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_routing_mode` | **token_choice** | 53563.8494 (n=262) | expert_choice | Δ +7291.1001 | p=4.37e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 216 | 53239.4740 | 6174.8385 | 50614.2885 |
| none | 65 | 55820.7813 | 9502.5677 | 50511.3595 |
| leaf_reuse | 19 | 64112.6030 | 13611.0465 | 52329.1432 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 262 | 53563.8494 | 6215.6286 | 50511.3595 |
| expert_choice | 38 | 60854.9495 | 14465.0411 | 50925.9556 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 215 | 53473.4440 | 7229.1807 | 50614.2885 |
| em | 65 | 54803.0058 | 6661.4683 | 50511.3595 |
| loss_only | 20 | 64361.5392 | 13302.9058 | 50763.5080 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| random | 216 | 53032.8246 | 6577.1688 | 50511.3595 |
| uniform | 15 | 57115.9751 | 7937.7804 | 51003.8473 |
| tree_hierarchical | 45 | 58454.3180 | 12184.9311 | 51217.1212 |
| gmm | 24 | 58497.6066 | 7107.0044 | 52184.7640 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 152 | 53234.0732 | 6288.3526 | 50614.2885 |
| markov | 69 | 55572.5474 | 7563.6556 | 51193.0772 |
| ema | 79 | 55951.0345 | 10902.3798 | 50511.3595 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 277 | 53528.9046 | 6754.4656 | 50511.3595 |
| True | 23 | 66030.8717 | 12989.2217 | 54112.3439 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 280 | 53731.5198 | 7679.2356 | 50511.3595 |
| True | 20 | 65069.5533 | 6804.2062 | 58478.2963 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 61405.1427 | — | — | 54248.8455 | **Q4** [3.0, ∞) |
| `mixture_diversity_lambda` | 55427.6934 | 52319.9615 | 53967.8612 | 56234.0387 | **Q2** [0.1636, 0.21] |
| `mixture_warmup_iters` | 53779.3357 | 54850.5701 | 54004.8640 | 55264.3893 | **Q1** [None, 15.0] |
| `mixture_balance_factor` | 55924.3990 | 54692.5417 | 53927.6106 | 54292.9993 | **Q3** [6.0, 7.0] |
| `learning_rate` | 57907.5149 | 52202.0028 | 52870.7757 | 54969.2615 | **Q2** [0.0859, 0.1039] |
| `num_leaves` | 55757.3978 | 53911.4339 | 53394.3440 | 54886.3792 | **Q3** [59.5, 72.25] |
| `max_depth` | 61058.6041 | 53829.9318 | — | 52512.8950 | **Q4** [11.0, ∞) |
| `min_data_in_leaf` | 53959.0225 | 51878.8925 | 53234.6944 | 59297.7016 | **Q2** [12.0, 16.5] |

#### E. Slice plot

![houses@s44/moe](slice_houses@s44_moe.png)


---

## Overall recommendations

**Categorical settings that are statistically significant winners (p<0.01):**

| dataset | param | best value | Δ vs runner-up | p |
|---|---|---|---|---|
| houses@s42 | `mixture_routing_mode` | **expert_choice** | +12290.6428 | 5.83e-04 |
| houses@s43 | `mixture_r_smoothing` | **ema** | +6136.9992 | 1.90e-03 |
| houses@s44 | `mixture_routing_mode` | **token_choice** | +7291.1001 | 4.37e-03 |
