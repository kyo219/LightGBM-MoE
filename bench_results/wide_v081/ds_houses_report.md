# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 500

- **Datasets**: ['houses'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `747bb4a7b5e6`, lib sha256 `a3c30fcc7fd3…`, package `/home/mokumoku/Develop/LightGBM-MoE/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| houses | naive-lightgbm | **50108.73806** ± 901.62518 | 50708.02892 | 0.0% | 0.26 |
| houses | moe | **48622.59264** ± 683.43205 | 50148.18266 | 0.0% | 16.15 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| houses@s42 | naive-lightgbm | 50493.3397 | 51154.2425 | 0.191 | 490 |
| houses@s42 | moe | 50063.7533 | 51081.3391 | 1.184 | 3597 |
| houses@s43 | naive-lightgbm | 50976.4089 | 51629.0269 | 0.356 | 863 |
| houses@s43 | moe | 50032.1886 | 50957.3259 | 4.853 | 17889 |
| houses@s44 | naive-lightgbm | 50654.3382 | 51120.2765 | 0.206 | 463 |
| houses@s44 | moe | 50348.6061 | 51067.4774 | 1.472 | 4169 |



---

## houses@s42  (search X=[8000, 8], holdout n=2000)


### naive-lightgbm

- **holdout RMSE: 50837.85512** (winner retrained in 0.17s, cv score of winner: 50493.3397)
- cv best RMSE: 50493.3397, median: 51154.2425, p10: 50883.3160
- train: median 0.191s/fold, mean 0.190s, p90 0.236s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.622 |
| `extra_trees` | 0.325 |
| `min_data_in_leaf` | 0.015 |
| `feature_fraction` | 0.013 |
| `max_depth` | 0.010 |
| `num_leaves` | 0.009 |
| `bagging_fraction` | 0.005 |
| `bagging_freq` | 0.001 |
| `lambda_l2` | 0.001 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 51746.4828 | 2793.8770 | 50493.3397 |
| True | 32 | 64290.3048 | 8625.1273 | 57820.1777 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 54781.7420 | 51578.3220 | 51574.0783 | 52263.0074 | **Q3** [0.0995, 0.1121] |
| `num_leaves` | 52730.6402 | 52014.5142 | 51693.7554 | 53692.0510 | **Q3** [35.0, 40.0] |
| `max_depth` | 55249.4566 | 52409.8563 | — | 51745.7003 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 52514.0039 | 51974.6581 | 52062.3832 | 53639.1470 | **Q2** [16.0, 20.0] |
| `lambda_l1` | 52689.4904 | 52130.2930 | 52972.4713 | 52404.8950 | **Q2** [0.0, 0.0004] |
| `lambda_l2` | 53832.3713 | 52038.4769 | 51838.7513 | 52487.5502 | **Q3** [0.1009, 0.3701] |
| `feature_fraction` | 54015.5799 | 51563.4755 | 51891.5653 | 52726.5290 | **Q2** [0.7264, 0.7497] |
| `bagging_fraction` | 52586.4592 | 52324.8414 | 52858.4815 | 52427.3677 | **Q2** [0.6783, 0.7446] |

#### E. Slice plot

![houses@s42/naive-lightgbm](slice_houses@s42_naive-lightgbm.png)


### moe

- **holdout RMSE: 49448.98956** (winner retrained in 1.92s, cv score of winner: 50063.7533)
- cv best RMSE: 50063.7533, median: 51081.3391, p10: 50425.2539
- train: median 1.184s/fold, mean 1.415s, p90 2.104s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.642 |
| `extra_trees` | 0.155 |
| `mixture_gate_type` | 0.093 |
| `mixture_init` | 0.028 |
| `bagging_fraction` | 0.013 |
| `bagging_freq` | 0.012 |
| `mixture_diversity_lambda` | 0.009 |
| `min_data_in_leaf` | 0.009 |
| `mixture_e_step_mode` | 0.007 |
| `lambda_l1` | 0.007 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_routing_mode` | **token_choice** | 54160.4088 (n=453) | expert_choice | Δ +6772.1273 | p=1.16e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 321 | 53851.6558 | 7583.0029 | 50063.7533 |
| leaf_reuse | 157 | 54854.0059 | 13746.7022 | 50387.7777 |
| none | 22 | 68183.3605 | 15483.9601 | 51929.4044 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 453 | 54160.4088 | 10271.8968 | 50063.7533 |
| expert_choice | 47 | 60932.5361 | 12949.3021 | 51533.7082 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 436 | 53765.2295 | 8640.4830 | 50063.7533 |
| loss_only | 39 | 57792.0083 | 10857.5253 | 51533.7082 |
| em | 25 | 68118.6401 | 24683.5026 | 51089.0199 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| random | 190 | 53027.1533 | 10068.3199 | 50387.7777 |
| uniform | 209 | 53891.4833 | 6013.5774 | 50063.7533 |
| gmm_features | 37 | 54025.7028 | 8156.2326 | 50729.6527 |
| kmeans_features | 34 | 55256.2911 | 6831.3861 | 51076.9097 |
| tree_hierarchical | 13 | 66717.7100 | 15218.0083 | 50559.7076 |
| gmm | 17 | 77354.1243 | 27073.4725 | 51137.9576 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 70 | 53908.2782 | 8869.8399 | 50594.2988 |
| none | 365 | 54772.8536 | 11171.0938 | 50063.7533 |
| markov | 65 | 55889.5897 | 9954.5724 | 50881.7753 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 259 | 54330.6249 | 12550.7596 | 50387.7777 |
| True | 241 | 55298.1848 | 8328.3174 | 50063.7533 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 53428.7546 | 7725.0691 | 50063.7533 |
| True | 32 | 74807.4134 | 22369.5943 | 56293.7129 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 68459.8475 | — | — | 54197.9908 | **Q4** [4.0, ∞) |
| `mixture_diversity_lambda` | 53740.4996 | 53843.9551 | 56915.7114 | 54687.7889 | **Q1** [None, 0.1405] |
| `mixture_warmup_iters` | 55403.6587 | 53403.5448 | 56295.8581 | 55149.2912 | **Q2** [6.0, 8.0] |
| `mixture_balance_factor` | 54573.7064 | 53951.4524 | 55778.0034 | 53914.5693 | **Q4** [9.0, ∞) |
| `learning_rate` | 59187.7545 | 52203.5091 | 53139.8697 | 54656.8218 | **Q2** [0.1216, 0.1661] |
| `num_leaves` | 57640.0757 | 53551.3946 | 54159.7571 | 53858.1199 | **Q2** [97.75, 118.0] |
| `max_depth` | 60222.5688 | 54595.0013 | 55259.8675 | 53109.3432 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 54203.4623 | 54103.5091 | 52816.7240 | 57483.1833 | **Q3** [20.0, 26.0] |

#### E. Slice plot

![houses@s42/moe](slice_houses@s42_moe.png)


---

## houses@s43  (search X=[8000, 8], holdout n=2000)


### naive-lightgbm

- **holdout RMSE: 50650.09521** (winner retrained in 0.33s, cv score of winner: 50976.4089)
- cv best RMSE: 50976.4089, median: 51629.0269, p10: 51303.1013
- train: median 0.356s/fold, mean 0.338s, p90 0.449s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.585 |
| `learning_rate` | 0.305 |
| `min_data_in_leaf` | 0.063 |
| `feature_fraction` | 0.025 |
| `max_depth` | 0.017 |
| `bagging_fraction` | 0.003 |
| `num_leaves` | 0.001 |
| `bagging_freq` | 0.001 |
| `lambda_l2` | 0.000 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 470 | 52168.2304 | 2334.6902 | 50976.4089 |
| True | 30 | 63746.3169 | 5075.7170 | 57331.5582 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 53935.3332 | 52579.0710 | 52023.8399 | 52913.4185 | **Q3** [0.0668, 0.0788] |
| `num_leaves` | 52920.8235 | 53441.8723 | 52517.0176 | 52569.0361 | **Q3** [101.5, 116.0] |
| `max_depth` | 55668.9373 | 52305.1409 | 52743.1669 | 52680.7233 | **Q2** [9.0, 10.0] |
| `min_data_in_leaf` | 51997.5348 | 52801.3171 | 52270.6233 | 54208.1829 | **Q1** [None, 9.0] |
| `lambda_l1` | 53341.0887 | 52435.9029 | 52987.9780 | 52686.6929 | **Q2** [0.0, 0.0] |
| `lambda_l2` | 53545.8936 | 52884.3822 | 52504.6073 | 52516.7794 | **Q3** [0.5555, 2.3927] |
| `feature_fraction` | 53422.5595 | 52289.6307 | 52469.9102 | 53269.5620 | **Q2** [0.741, 0.782] |
| `bagging_fraction` | 53144.9239 | 52669.6318 | 52291.9041 | 53345.2027 | **Q3** [0.7877, 0.8594] |

#### E. Slice plot

![houses@s43/naive-lightgbm](slice_houses@s43_naive-lightgbm.png)


### moe

- **holdout RMSE: 48643.46809** (winner retrained in 44.24s, cv score of winner: 50032.1886)
- cv best RMSE: 50032.1886, median: 50957.3259, p10: 50426.0756
- train: median 4.853s/fold, mean 7.128s, p90 17.693s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.317 |
| `mixture_diversity_lambda` | 0.226 |
| `lambda_l2` | 0.141 |
| `learning_rate` | 0.097 |
| `bagging_fraction` | 0.076 |
| `mixture_routing_mode` | 0.025 |
| `mixture_warmup_iters` | 0.022 |
| `feature_fraction` | 0.021 |
| `mixture_init` | 0.016 |
| `extra_trees` | 0.013 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_routing_mode` | **expert_choice** | 53630.6562 (n=449) | token_choice | Δ +13124.3606 | p=2.72e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 143 | 54071.6466 | 10898.4952 | 50286.1129 |
| none | 134 | 54650.9092 | 9836.9510 | 50319.5920 |
| gbdt | 223 | 55736.3371 | 19329.9573 | 50032.1886 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 449 | 53630.6562 | 11762.1476 | 50032.1886 |
| token_choice | 51 | 66755.0168 | 29214.5955 | 50038.4021 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 69 | 54704.3690 | 8815.5175 | 50032.1886 |
| em | 286 | 54847.0487 | 15842.0633 | 50038.4021 |
| loss_only | 145 | 55336.6420 | 15836.7701 | 50286.1129 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| tree_hierarchical | 156 | 52488.3863 | 6821.7910 | 50187.0748 |
| random | 103 | 53228.3811 | 7335.0664 | 50032.1886 |
| kmeans_features | 68 | 54996.2863 | 10949.6795 | 50444.1114 |
| gmm | 142 | 56955.0442 | 22295.2548 | 50195.5568 |
| uniform | 17 | 62534.3639 | 26069.0675 | 50808.4965 |
| gmm_features | 14 | 65965.0731 | 20606.0131 | 50738.5145 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 268 | 53909.9816 | 10818.9119 | 50187.0748 |
| ema | 210 | 55297.8656 | 17696.1112 | 50032.1886 |
| none | 22 | 64738.3478 | 25209.1516 | 50576.5624 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 54335.8159 | 14598.3506 | 50032.1886 |
| True | 32 | 64234.6452 | 18425.3868 | 51092.1033 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 470 | 54424.3998 | 15231.1019 | 50032.1886 |
| True | 30 | 63506.7524 | 8515.1764 | 57253.8543 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 59377.3346 | — | — | 54001.7326 | **Q4** [6.0, ∞) |
| `mixture_diversity_lambda` | 55180.4023 | 53362.9569 | 53092.0466 | 58241.9581 | **Q3** [0.1073, 0.157] |
| `mixture_warmup_iters` | 55839.6956 | 55686.0805 | 53201.8775 | 55108.9572 | **Q3** [17.0, 24.0] |
| `mixture_balance_factor` | 59404.2610 | 55584.5889 | 52957.5405 | 53655.9899 | **Q3** [8.0, 10.0] |
| `learning_rate` | 59809.8159 | 54342.5300 | 53438.2420 | 52286.7758 | **Q4** [0.1136, ∞) |
| `num_leaves` | 54170.7331 | 53408.3415 | 55795.5767 | 56494.4281 | **Q2** [38.0, 48.0] |
| `max_depth` | 58813.4305 | 53116.3478 | 55641.0040 | 53874.1591 | **Q2** [9.0, 10.0] |
| `min_data_in_leaf` | 53163.3115 | 54246.5006 | 54323.1530 | 57974.7094 | **Q1** [None, 8.0] |

#### E. Slice plot

![houses@s43/moe](slice_houses@s43_moe.png)


---

## houses@s44  (search X=[8000, 8], holdout n=2000)


### naive-lightgbm

- **holdout RMSE: 48838.26386** (winner retrained in 0.29s, cv score of winner: 50654.3382)
- cv best RMSE: 50654.3382, median: 51120.2765, p10: 50847.9634
- train: median 0.206s/fold, mean 0.181s, p90 0.261s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.667 |
| `learning_rate` | 0.228 |
| `min_data_in_leaf` | 0.063 |
| `max_depth` | 0.026 |
| `feature_fraction` | 0.008 |
| `bagging_fraction` | 0.005 |
| `num_leaves` | 0.002 |
| `bagging_freq` | 0.001 |
| `lambda_l1` | 0.000 |
| `lambda_l2` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 470 | 51846.3373 | 2450.6921 | 50654.3382 |
| True | 30 | 64350.9585 | 3681.9798 | 58652.9668 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 54142.3994 | 51567.6162 | 52081.2127 | 52595.2301 | **Q2** [0.0548, 0.0624] |
| `num_leaves` | 53888.4843 | 51873.9383 | 52670.7577 | 51910.8426 | **Q2** [96.0, 113.0] |
| `max_depth` | 54200.7503 | 52468.5722 | — | 52134.2667 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 51943.2776 | 51918.9861 | 52091.0049 | 54292.1137 | **Q2** [12.0, 15.0] |
| `lambda_l1` | 53980.9400 | 51692.0805 | 52463.0322 | 52250.4058 | **Q2** [0.0138, 0.3111] |
| `lambda_l2` | 53238.8650 | 52128.8312 | 52314.3403 | 52704.4219 | **Q2** [0.0002, 0.0007] |
| `feature_fraction` | 53775.7243 | 52081.5784 | 51757.3688 | 52771.7871 | **Q3** [0.7446, 0.7641] |
| `bagging_fraction` | 52748.7338 | 52226.9187 | 52445.8032 | 52965.0027 | **Q2** [0.8165, 0.8376] |

#### E. Slice plot

![houses@s44/naive-lightgbm](slice_houses@s44_naive-lightgbm.png)


### moe

- **holdout RMSE: 47775.32027** (winner retrained in 2.30s, cv score of winner: 50348.6061)
- cv best RMSE: 50348.6061, median: 51067.4774, p10: 50562.9446
- train: median 1.472s/fold, mean 1.635s, p90 2.787s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.693 |
| `extra_trees` | 0.223 |
| `mixture_expert_dropout_rate` | 0.020 |
| `num_leaves` | 0.014 |
| `min_data_in_leaf` | 0.011 |
| `mixture_gate_type` | 0.007 |
| `feature_fraction` | 0.005 |
| `mixture_warmup_iters` | 0.005 |
| `mixture_e_step_mode` | 0.004 |
| `bagging_freq` | 0.003 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_routing_mode` | **expert_choice** | 52859.5617 (n=372) | token_choice | Δ +3198.5537 | p=4.12e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 408 | 52833.8534 | 7092.3304 | 50348.6061 |
| gbdt | 63 | 54676.3713 | 8222.0732 | 50613.5339 |
| none | 29 | 63392.1433 | 17061.2321 | 50608.3009 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 372 | 52859.5617 | 6919.7726 | 50348.6061 |
| token_choice | 128 | 56058.1154 | 11701.5448 | 50431.9257 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| em | 293 | 52865.1352 | 5831.4007 | 50348.6061 |
| loss_only | 153 | 53238.7388 | 7158.7776 | 50503.6169 |
| gate_only | 54 | 59336.7427 | 17510.1263 | 50435.4329 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 137 | 52337.2726 | 5577.6317 | 50348.6061 |
| kmeans_features | 114 | 52881.7566 | 8170.5323 | 50378.3946 |
| random | 129 | 53692.4163 | 9104.7337 | 50503.6169 |
| gmm | 32 | 54553.5168 | 8353.9899 | 51073.1369 |
| tree_hierarchical | 70 | 56268.4030 | 11726.2868 | 50360.3696 |
| gmm_features | 18 | 57202.5936 | 6881.1089 | 50730.8716 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 117 | 52855.9268 | 3765.1454 | 50503.6169 |
| markov | 259 | 53154.3453 | 7102.7703 | 50348.6061 |
| ema | 124 | 55549.0070 | 13014.9506 | 50360.3696 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 192 | 53456.4146 | 7314.2267 | 50378.3946 |
| False | 308 | 53816.7667 | 9192.4684 | 50348.6061 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 469 | 52792.7314 | 7324.3767 | 50348.6061 |
| True | 31 | 67077.5712 | 12975.9660 | 56323.5633 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 59718.5784 | — | — | 52711.4010 | **Q4** [6.0, ∞) |
| `mixture_diversity_lambda` | 52596.2334 | 52710.1620 | 54410.6510 | 54996.5195 | **Q1** [None, 0.1392] |
| `mixture_warmup_iters` | 52903.8325 | 52519.7261 | 53501.4449 | 55565.3661 | **Q2** [6.0, 8.0] |
| `mixture_balance_factor` | — | 52619.1678 | 55505.3252 | 54072.2816 | **Q2** [2.0, 3.0] |
| `learning_rate` | 57922.2611 | 51976.9329 | 52322.3763 | 52491.9956 | **Q2** [0.0929, 0.1076] |
| `num_leaves` | 54825.5820 | 53191.1202 | 52793.2573 | 53834.9690 | **Q3** [76.0, 84.0] |
| `max_depth` | 59553.7531 | 53142.6459 | — | 52479.1975 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 52530.9065 | 52469.1748 | 52682.2717 | 56666.7004 | **Q2** [13.0, 17.0] |

#### E. Slice plot

![houses@s44/moe](slice_houses@s44_moe.png)


---

## Overall recommendations

**Categorical settings that are statistically significant winners (p<0.01):**

| dataset | param | best value | Δ vs runner-up | p |
|---|---|---|---|---|
| houses@s42 | `mixture_routing_mode` | **token_choice** | +6772.1273 | 1.16e-03 |
| houses@s43 | `mixture_routing_mode` | **expert_choice** | +13124.3606 | 2.72e-03 |
| houses@s44 | `mixture_routing_mode` | **expert_choice** | +3198.5537 | 4.12e-03 |
