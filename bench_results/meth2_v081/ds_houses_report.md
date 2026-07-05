# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 300

- **Datasets**: ['houses'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `5232455cdc8d`, lib sha256 `a3c30fcc7fd3…`, package `/home/mokumoku/Develop/LightGBM-MoE/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| houses | naive-lightgbm | **49805.38271** ± 962.89058 | 50811.46230 | 0.0% | 0.48 |
| houses | moe | **48930.56383** ± 669.41513 | 50541.22167 | 0.0% | 2.06 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| houses@s42 | naive-lightgbm | 50683.0901 | 51224.1336 | 0.197 | 300 |
| houses@s42 | moe | 50328.3599 | 51141.4085 | 1.113 | 2013 |
| houses@s43 | naive-lightgbm | 51076.3747 | 51723.6494 | 0.345 | 496 |
| houses@s43 | moe | 50705.3778 | 51454.1465 | 1.032 | 1790 |
| houses@s44 | naive-lightgbm | 50674.9222 | 51222.7143 | 0.424 | 568 |
| houses@s44 | moe | 50589.9273 | 51070.8269 | 1.555 | 2121 |



---

## houses@s42  (search X=[8000, 8], holdout n=2000)


### naive-lightgbm

- **holdout RMSE: 50760.44294** (winner retrained in 0.22s, cv score of winner: 50683.0901)
- cv best RMSE: 50683.0901, median: 51224.1336, p10: 50895.5274
- train: median 0.197s/fold, mean 0.195s, p90 0.241s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.667 |
| `extra_trees` | 0.289 |
| `min_data_in_leaf` | 0.016 |
| `feature_fraction` | 0.010 |
| `max_depth` | 0.008 |
| `bagging_fraction` | 0.004 |
| `num_leaves` | 0.003 |
| `bagging_freq` | 0.002 |
| `lambda_l2` | 0.001 |
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

- **holdout RMSE: 49606.48685** (winner retrained in 1.46s, cv score of winner: 50328.3599)
- cv best RMSE: 50328.3599, median: 51141.4085, p10: 50554.8589
- train: median 1.113s/fold, mean 1.320s, p90 1.740s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.475 |
| `extra_trees` | 0.201 |
| `mixture_gate_type` | 0.084 |
| `min_data_in_leaf` | 0.077 |
| `mixture_hard_m_step` | 0.039 |
| `mixture_routing_mode` | 0.027 |
| `mixture_refit_leaves` | 0.019 |
| `mixture_warmup_iters` | 0.018 |
| `mixture_init` | 0.018 |
| `num_leaves` | 0.012 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 227 | 52561.0029 | 6368.4483 | 50328.3599 |
| gbdt | 55 | 56240.5273 | 11281.4232 | 51145.6924 |
| none | 18 | 63717.1137 | 13748.0140 | 51062.7292 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 220 | 53468.6453 | 7969.6502 | 50361.2373 |
| token_choice | 80 | 55104.7843 | 9950.5723 | 50328.3599 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| em | 158 | 53171.5982 | 6642.3209 | 50457.1591 |
| gate_only | 106 | 53364.3401 | 7771.4590 | 50328.3599 |
| loss_only | 36 | 58715.3375 | 14653.0101 | 51411.9706 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| random | 202 | 52550.3983 | 6077.8880 | 50328.3599 |
| uniform | 44 | 55715.1854 | 11224.4957 | 51411.9706 |
| gmm | 29 | 55971.0784 | 11265.8340 | 50337.1123 |
| tree_hierarchical | 25 | 59266.9930 | 12661.6279 | 50394.0203 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 207 | 52495.6730 | 4660.7289 | 50361.2373 |
| markov | 64 | 56575.7168 | 14466.6525 | 50328.3599 |
| none | 29 | 58070.1560 | 9808.6356 | 51633.6428 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 245 | 53320.6019 | 7898.1635 | 50361.2373 |
| True | 55 | 56507.9501 | 10712.6703 | 50328.3599 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 279 | 53035.9593 | 7887.1092 | 50328.3599 |
| True | 21 | 65450.0990 | 8960.5847 | 56860.8700 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 53654.6265 | — | 53838.5214 | 54089.7996 | **Q1** [None, 3.0] |
| `mixture_diversity_lambda` | 52374.4155 | 52238.5013 | 52833.9575 | 58172.9220 | **Q2** [0.032, 0.0776] |
| `mixture_warmup_iters` | 52406.5517 | 56073.3724 | 52841.3306 | 54469.9122 | **Q1** [None, 8.75] |
| `mixture_balance_factor` | 56881.4091 | 61967.6384 | — | 52203.9551 | **Q4** [9.0, ∞) |
| `learning_rate` | 57435.0749 | 52550.6951 | 52185.2741 | 53448.7521 | **Q3** [0.1347, 0.1636] |
| `num_leaves` | 54996.4036 | 52287.4793 | 55858.3200 | 52811.6703 | **Q2** [46.0, 57.5] |
| `max_depth` | 57901.3487 | 53461.3992 | — | 52599.2521 | **Q4** [11.0, ∞) |
| `min_data_in_leaf` | 53200.6035 | 53404.0227 | 52813.2276 | 56055.9596 | **Q3** [32.5, 41.0] |

#### E. Slice plot

![houses@s42/moe](slice_houses@s42_moe.png)


---

## houses@s43  (search X=[8000, 8], holdout n=2000)


### naive-lightgbm

- **holdout RMSE: 50168.46687** (winner retrained in 0.55s, cv score of winner: 51076.3747)
- cv best RMSE: 51076.3747, median: 51723.6494, p10: 51396.0980
- train: median 0.345s/fold, mean 0.324s, p90 0.440s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.574 |
| `learning_rate` | 0.271 |
| `feature_fraction` | 0.057 |
| `min_data_in_leaf` | 0.037 |
| `max_depth` | 0.034 |
| `bagging_freq` | 0.013 |
| `bagging_fraction` | 0.011 |
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

- **holdout RMSE: 49166.64150** (winner retrained in 2.51s, cv score of winner: 50705.3778)
- cv best RMSE: 50705.3778, median: 51454.1465, p10: 50923.4819
- train: median 1.032s/fold, mean 1.170s, p90 1.719s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.516 |
| `max_depth` | 0.111 |
| `extra_trees` | 0.088 |
| `bagging_freq` | 0.069 |
| `mixture_refit_leaves` | 0.055 |
| `min_data_in_leaf` | 0.053 |
| `mixture_diversity_lambda` | 0.031 |
| `mixture_gate_type` | 0.021 |
| `mixture_warmup_iters` | 0.019 |
| `mixture_balance_factor` | 0.012 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 179 | 52866.5474 | 6220.6660 | 50705.3778 |
| leaf_reuse | 100 | 53685.4918 | 6373.4063 | 50796.7843 |
| none | 21 | 65120.0933 | 22218.9926 | 51294.3960 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 204 | 53506.3314 | 8840.3014 | 50705.3778 |
| expert_choice | 96 | 55040.5365 | 9183.4818 | 51077.4343 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| em | 189 | 53111.9626 | 7324.2664 | 50736.4124 |
| gate_only | 83 | 53848.7232 | 6533.9606 | 51077.4343 |
| loss_only | 28 | 60413.5055 | 18134.1152 | 50705.3778 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| random | 245 | 53078.8314 | 7167.8098 | 50705.3778 |
| gmm | 16 | 53912.5258 | 1899.4457 | 51655.0006 |
| uniform | 24 | 58768.9707 | 15595.3853 | 51155.2892 |
| tree_hierarchical | 15 | 61454.2483 | 16579.1002 | 51191.7155 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 245 | 52588.9081 | 5704.8175 | 50705.3778 |
| markov | 37 | 57923.3044 | 12378.8113 | 51615.3225 |
| none | 18 | 65096.5767 | 20104.9514 | 51538.8151 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 274 | 52918.0363 | 6769.5434 | 50705.3778 |
| True | 26 | 65370.8143 | 17490.4114 | 51384.6113 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 280 | 53420.1525 | 8856.2560 | 50705.3778 |
| True | 20 | 62077.0209 | 6451.7889 | 56970.8863 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 58698.8078 | — | — | 53697.1794 | **Q4** [3.0, ∞) |
| `mixture_diversity_lambda` | 53070.2117 | 52584.8410 | 52521.7762 | 57812.2793 | **Q3** [0.1181, 0.1421] |
| `mixture_warmup_iters` | 52904.4205 | 52822.8717 | 53658.9723 | 56188.9507 | **Q2** [12.0, 14.0] |
| `mixture_balance_factor` | 59625.8324 | 52480.4251 | — | 52769.0678 | **Q2** [9.0, 10.0] |
| `learning_rate` | 58862.5035 | 51947.7161 | 51888.1531 | 53290.7355 | **Q3** [0.1313, 0.1694] |
| `num_leaves` | 53584.9353 | 52029.8004 | 52594.1891 | 57723.1666 | **Q2** [27.0, 32.0] |
| `max_depth` | 56743.4058 | 52483.7026 | — | 53560.6274 | **Q2** [7.0, 8.0] |
| `min_data_in_leaf` | 52719.5208 | 53415.7267 | 52202.8965 | 57565.4635 | **Q3** [22.0, 27.0] |

#### E. Slice plot

![houses@s43/moe](slice_houses@s43_moe.png)


---

## houses@s44  (search X=[8000, 8], holdout n=2000)


### naive-lightgbm

- **holdout RMSE: 48487.23831** (winner retrained in 0.68s, cv score of winner: 50674.9222)
- cv best RMSE: 50674.9222, median: 51222.7143, p10: 50855.8811
- train: median 0.424s/fold, mean 0.372s, p90 0.567s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.692 |
| `learning_rate` | 0.219 |
| `min_data_in_leaf` | 0.041 |
| `max_depth` | 0.024 |
| `feature_fraction` | 0.011 |
| `bagging_fraction` | 0.006 |
| `bagging_freq` | 0.003 |
| `num_leaves` | 0.002 |
| `lambda_l1` | 0.001 |
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

- **holdout RMSE: 48018.56314** (winner retrained in 2.20s, cv score of winner: 50589.9273)
- cv best RMSE: 50589.9273, median: 51070.8269, p10: 50742.9081
- train: median 1.555s/fold, mean 1.395s, p90 1.998s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.390 |
| `extra_trees` | 0.207 |
| `min_data_in_leaf` | 0.081 |
| `mixture_init` | 0.073 |
| `max_depth` | 0.061 |
| `mixture_hard_m_step` | 0.046 |
| `num_leaves` | 0.042 |
| `mixture_r_smoothing` | 0.021 |
| `feature_fraction` | 0.019 |
| `bagging_freq` | 0.016 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_routing_mode` | **token_choice** | 52913.8376 (n=258) | expert_choice | Δ +9376.2789 | p=6.72e-03 |
| `mixture_init` | **uniform** | 52756.1869 (n=242) | tree_hierarchical | Δ +3942.9164 | p=3.73e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 167 | 53463.2453 | 7286.9204 | 50589.9273 |
| none | 98 | 54828.5029 | 13304.8481 | 50654.0919 |
| gbdt | 35 | 56182.8498 | 11838.6863 | 50863.9007 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 258 | 52913.8376 | 6177.5775 | 50589.9273 |
| expert_choice | 42 | 62290.1165 | 20911.4932 | 50764.1484 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| em | 167 | 53457.6306 | 7280.0732 | 50589.9273 |
| loss_only | 97 | 54455.4010 | 12248.8014 | 50654.0919 |
| gate_only | 36 | 57176.5774 | 14497.4678 | 50863.9007 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 242 | 52756.1869 | 6947.7786 | 50589.9273 |
| tree_hierarchical | 16 | 56699.1033 | 4324.0192 | 51802.8678 |
| random | 28 | 58706.8823 | 12195.6176 | 51284.8351 |
| gmm | 14 | 67855.6715 | 28608.9037 | 52030.5229 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 246 | 52710.0895 | 5988.0027 | 50589.9273 |
| none | 25 | 59012.3863 | 14535.8784 | 51013.1889 |
| markov | 29 | 62964.2524 | 21918.5678 | 50965.0651 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 265 | 53100.9391 | 7798.4288 | 50589.9273 |
| True | 35 | 62748.7465 | 18817.7422 | 53485.5830 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 280 | 53220.1237 | 9282.0985 | 50589.9273 |
| True | 20 | 68316.0177 | 12267.1583 | 57306.1212 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 54681.2728 | 70944.6474 | — | 52920.2660 | **Q4** [4.0, ∞) |
| `mixture_diversity_lambda` | 54564.6107 | 52005.6751 | 53029.8492 | 57305.9316 | **Q2** [0.1355, 0.1658] |
| `mixture_warmup_iters` | 57036.4245 | 52273.1550 | 54688.2600 | 53148.9989 | **Q2** [31.0, 36.0] |
| `mixture_balance_factor` | 54444.2875 | 57889.6490 | 53627.9676 | 53154.0900 | **Q4** [9.0, ∞) |
| `learning_rate` | 59137.9441 | 52262.9935 | 52171.0060 | 53334.1230 | **Q3** [0.1004, 0.1229] |
| `num_leaves` | 58963.9617 | 52539.5857 | 53414.8579 | 52077.2859 | **Q4** [118.25, ∞) |
| `max_depth` | 62658.8768 | 55333.5254 | — | 52173.4793 | **Q4** [11.0, ∞) |
| `min_data_in_leaf` | 53911.4772 | 51964.4502 | 52823.0953 | 57963.8021 | **Q2** [16.0, 19.0] |

#### E. Slice plot

![houses@s44/moe](slice_houses@s44_moe.png)


---

## Overall recommendations

**Categorical settings that are statistically significant winners (p<0.01):**

| dataset | param | best value | Δ vs runner-up | p |
|---|---|---|---|---|
| houses@s44 | `mixture_routing_mode` | **token_choice** | +9376.2789 | 6.72e-03 |
| houses@s44 | `mixture_init` | **uniform** | +3942.9164 | 3.73e-03 |
