# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 500

- **Datasets**: ['wine_quality'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `747bb4a7b5e6`, lib sha256 `a3c30fcc7fd3…`, package `/home/mokumoku/Develop/LightGBM-MoE/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| wine_quality | naive-lightgbm | **0.61457** ± 0.01410 | 0.66815 | 0.0% | 0.49 |
| wine_quality | moe | **0.61195** ± 0.01642 | 0.66367 | 0.0% | 2.75 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| wine_quality@s42 | naive-lightgbm | 0.6678 | 0.6752 | 0.444 | 998 |
| wine_quality@s42 | moe | 0.6642 | 0.6770 | 2.022 | 5340 |
| wine_quality@s43 | naive-lightgbm | 0.6616 | 0.6684 | 0.364 | 839 |
| wine_quality@s43 | moe | 0.6579 | 0.6694 | 1.700 | 6838 |
| wine_quality@s44 | naive-lightgbm | 0.6750 | 0.6832 | 0.253 | 589 |
| wine_quality@s44 | moe | 0.6689 | 0.6816 | 0.692 | 6366 |



---

## wine_quality@s42  (search X=[5198, 11], holdout n=1299)


### naive-lightgbm

- **holdout RMSE: 0.60747** (winner retrained in 0.58s, cv score of winner: 0.6678)
- cv best RMSE: 0.6678, median: 0.6752, p10: 0.6708
- train: median 0.444s/fold, mean 0.393s, p90 0.576s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.431 |
| `extra_trees` | 0.406 |
| `min_data_in_leaf` | 0.106 |
| `max_depth` | 0.026 |
| `bagging_fraction` | 0.017 |
| `feature_fraction` | 0.008 |
| `num_leaves` | 0.003 |
| `bagging_freq` | 0.002 |
| `lambda_l2` | 0.001 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 0.6785 | 0.0102 | 0.6678 |
| True | 32 | 0.7184 | 0.0293 | 0.6904 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.6867 | 0.6761 | 0.6781 | 0.6835 | **Q2** [0.0428, 0.0486] |
| `num_leaves` | 0.6892 | 0.6794 | 0.6772 | 0.6786 | **Q3** [117.0, 123.0] |
| `max_depth` | 0.6960 | 0.6798 | — | 0.6779 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 0.6756 | 0.6764 | 0.6774 | 0.6941 | **Q1** [None, 7.0] |
| `lambda_l1` | 0.6792 | 0.6795 | 0.6780 | 0.6876 | **Q3** [0.0, 0.0] |
| `lambda_l2` | 0.6783 | 0.6881 | 0.6777 | 0.6803 | **Q3** [0.0012, 0.0165] |
| `feature_fraction` | 0.6831 | 0.6777 | 0.6779 | 0.6856 | **Q2** [0.6164, 0.6434] |
| `bagging_fraction` | 0.6877 | 0.6786 | 0.6760 | 0.6821 | **Q3** [0.8882, 0.9047] |

#### E. Slice plot

![wine_quality@s42/naive-lightgbm](slice_wine_quality@s42_naive-lightgbm.png)


### moe

- **holdout RMSE: 0.60354** (winner retrained in 3.89s, cv score of winner: 0.6642)
- cv best RMSE: 0.6642, median: 0.6770, p10: 0.6690
- train: median 2.022s/fold, mean 2.118s, p90 3.403s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l2` | 0.512 |
| `mixture_r_smoothing` | 0.114 |
| `learning_rate` | 0.064 |
| `feature_fraction` | 0.047 |
| `extra_trees` | 0.040 |
| `lambda_l1` | 0.029 |
| `min_data_in_leaf` | 0.029 |
| `mixture_num_experts` | 0.024 |
| `mixture_init` | 0.024 |
| `mixture_e_step_mode` | 0.020 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 314 | 0.7158 | 0.2388 | 0.6642 |
| gbdt | 127 | 0.8242 | 0.6551 | 0.6706 |
| leaf_reuse | 59 | 1.1825 | 1.0711 | 0.6849 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 269 | 0.7449 | 0.3284 | 0.6687 |
| token_choice | 231 | 0.8608 | 0.7218 | 0.6642 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 352 | 0.7206 | 0.2408 | 0.6642 |
| gate_only | 97 | 0.9746 | 0.9734 | 0.6661 |
| em | 51 | 1.0001 | 0.7860 | 0.6750 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| kmeans_features | 145 | 0.7227 | 0.2405 | 0.6687 |
| uniform | 109 | 0.7520 | 0.3656 | 0.6709 |
| random | 188 | 0.7843 | 0.5624 | 0.6642 |
| gmm_features | 17 | 1.0460 | 1.1231 | 0.6720 |
| gmm | 28 | 1.1425 | 1.0025 | 0.6728 |
| tree_hierarchical | 13 | 1.1704 | 0.9513 | 0.6726 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 318 | 0.7474 | 0.4348 | 0.6642 |
| none | 125 | 0.8154 | 0.5179 | 0.6706 |
| markov | 57 | 1.0460 | 0.9646 | 0.6874 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 450 | 0.7664 | 0.4504 | 0.6642 |
| True | 50 | 1.0864 | 1.0500 | 0.6768 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 0.7753 | 0.5041 | 0.6642 |
| True | 32 | 1.1360 | 0.9389 | 0.6800 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | 0.7246 | — | 0.8493 | **Q2** [2.0, 3.0] |
| `mixture_diversity_lambda` | 0.7935 | 0.7483 | 0.7960 | 0.8559 | **Q2** [0.0303, 0.0764] |
| `mixture_warmup_iters` | 0.8344 | 0.8656 | 0.7253 | 0.7672 | **Q3** [33.0, 40.0] |
| `mixture_balance_factor` | 0.8672 | 0.7097 | 0.8310 | 0.7222 | **Q2** [6.0, 7.0] |
| `learning_rate` | 0.8382 | 0.7447 | 0.7690 | 0.8417 | **Q2** [0.0746, 0.1015] |
| `num_leaves` | 1.0571 | 0.7266 | 0.7259 | 0.6934 | **Q4** [113.0, ∞) |
| `max_depth` | 1.0099 | — | 0.7640 | 0.7276 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 0.7058 | 0.7442 | 0.7004 | 1.0149 | **Q3** [10.0, 16.0] |

#### E. Slice plot

![wine_quality@s42/moe](slice_wine_quality@s42_moe.png)


---

## wine_quality@s43  (search X=[5198, 11], holdout n=1299)


### naive-lightgbm

- **holdout RMSE: 0.63426** (winner retrained in 0.46s, cv score of winner: 0.6616)
- cv best RMSE: 0.6616, median: 0.6684, p10: 0.6653
- train: median 0.364s/fold, mean 0.330s, p90 0.484s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.633 |
| `learning_rate` | 0.198 |
| `min_data_in_leaf` | 0.121 |
| `max_depth` | 0.018 |
| `bagging_fraction` | 0.013 |
| `feature_fraction` | 0.007 |
| `bagging_freq` | 0.006 |
| `num_leaves` | 0.003 |
| `lambda_l1` | 0.001 |
| `lambda_l2` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 469 | 0.6719 | 0.0100 | 0.6616 |
| True | 31 | 0.7142 | 0.0202 | 0.6886 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.6807 | 0.6707 | 0.6711 | 0.6756 | **Q2** [0.0519, 0.0594] |
| `num_leaves` | 0.6827 | 0.6744 | 0.6711 | 0.6703 | **Q4** [124.25, ∞) |
| `max_depth` | 0.6861 | 0.6733 | — | 0.6718 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 0.6691 | 0.6714 | 0.6705 | 0.6843 | **Q1** [None, 6.0] |
| `lambda_l1` | 0.6789 | 0.6731 | 0.6705 | 0.6755 | **Q3** [0.0707, 0.1534] |
| `lambda_l2` | 0.6737 | 0.6702 | 0.6742 | 0.6800 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 0.6736 | 0.6709 | 0.6721 | 0.6813 | **Q2** [0.5505, 0.5732] |
| `bagging_fraction` | 0.6747 | 0.6730 | 0.6701 | 0.6802 | **Q3** [0.659, 0.6804] |

#### E. Slice plot

![wine_quality@s43/naive-lightgbm](slice_wine_quality@s43_naive-lightgbm.png)


### moe

- **holdout RMSE: 0.63490** (winner retrained in 3.50s, cv score of winner: 0.6579)
- cv best RMSE: 0.6579, median: 0.6694, p10: 0.6630
- train: median 1.700s/fold, mean 2.719s, p90 6.880s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l2` | 0.811 |
| `lambda_l1` | 0.059 |
| `learning_rate` | 0.020 |
| `mixture_expert_dropout_rate` | 0.019 |
| `mixture_init` | 0.014 |
| `num_leaves` | 0.011 |
| `mixture_load_balance_alpha` | 0.009 |
| `mixture_hard_m_step` | 0.008 |
| `min_data_in_leaf` | 0.007 |
| `mixture_warmup_iters` | 0.007 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_routing_mode` | **token_choice** | 0.7241 (n=406) | expert_choice | Δ +0.3154 | p=1.60e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 278 | 0.7160 | 0.2509 | 0.6579 |
| gbdt | 173 | 0.8270 | 0.6402 | 0.6635 |
| leaf_reuse | 49 | 1.0118 | 0.8401 | 0.6605 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 406 | 0.7241 | 0.3103 | 0.6579 |
| expert_choice | 94 | 1.0395 | 0.9252 | 0.6683 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| em | 230 | 0.7384 | 0.3207 | 0.6579 |
| loss_only | 194 | 0.7516 | 0.4347 | 0.6620 |
| gate_only | 76 | 1.0006 | 0.9074 | 0.6683 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 111 | 0.7075 | 0.1496 | 0.6600 |
| gmm | 174 | 0.7309 | 0.3219 | 0.6579 |
| random | 64 | 0.7663 | 0.4044 | 0.6683 |
| gmm_features | 114 | 0.8219 | 0.7006 | 0.6635 |
| kmeans_features | 22 | 1.1245 | 0.9604 | 0.6639 |
| tree_hierarchical | 15 | 1.2339 | 0.9308 | 0.6674 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 352 | 0.7190 | 0.3272 | 0.6579 |
| none | 71 | 0.8429 | 0.4949 | 0.6603 |
| markov | 77 | 1.0228 | 0.9254 | 0.6656 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 0.7455 | 0.3718 | 0.6579 |
| True | 32 | 1.3381 | 1.2738 | 0.6851 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 470 | 0.7768 | 0.4951 | 0.6579 |
| True | 30 | 0.8872 | 0.6211 | 0.6789 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.8241 | — | — | 0.7819 | **Q4** [3.0, ∞) |
| `mixture_diversity_lambda` | 0.7048 | 0.8208 | 0.7510 | 0.8570 | **Q1** [None, 0.0265] |
| `mixture_warmup_iters` | 0.7388 | 0.7005 | 0.7460 | 0.9480 | **Q2** [10.0, 16.0] |
| `mixture_balance_factor` | 0.9055 | 0.9678 | 0.6968 | 0.7379 | **Q3** [8.0, 10.0] |
| `learning_rate` | 0.8304 | 0.7668 | 0.8348 | 0.7016 | **Q4** [0.162, ∞) |
| `num_leaves` | 0.7780 | 0.7051 | 0.7524 | 0.8997 | **Q2** [70.0, 80.0] |
| `max_depth` | 1.0306 | 0.7380 | 0.7879 | 0.7078 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 0.6939 | 0.7180 | 0.7326 | 0.9799 | **Q1** [None, 7.0] |

#### E. Slice plot

![wine_quality@s43/moe](slice_wine_quality@s43_moe.png)


---

## wine_quality@s44  (search X=[5198, 11], holdout n=1299)


### naive-lightgbm

- **holdout RMSE: 0.60197** (winner retrained in 0.42s, cv score of winner: 0.6750)
- cv best RMSE: 0.6750, median: 0.6832, p10: 0.6787
- train: median 0.253s/fold, mean 0.231s, p90 0.331s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.463 |
| `learning_rate` | 0.193 |
| `min_data_in_leaf` | 0.121 |
| `max_depth` | 0.071 |
| `bagging_fraction` | 0.055 |
| `lambda_l2` | 0.037 |
| `feature_fraction` | 0.024 |
| `bagging_freq` | 0.014 |
| `num_leaves` | 0.011 |
| `lambda_l1` | 0.010 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 467 | 0.6860 | 0.0095 | 0.6750 |
| True | 33 | 0.7144 | 0.0143 | 0.6983 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.6899 | 0.6850 | 0.6855 | 0.6911 | **Q2** [0.0429, 0.0497] |
| `num_leaves` | 0.6954 | 0.6864 | 0.6855 | 0.6852 | **Q4** [121.0, ∞) |
| `max_depth` | 0.6974 | — | 0.6848 | 0.6877 | **Q3** [9.0, 10.0] |
| `min_data_in_leaf` | — | 0.6834 | 0.6849 | 0.6972 | **Q2** [5.0, 7.0] |
| `lambda_l1` | 0.6902 | 0.6870 | 0.6857 | 0.6885 | **Q3** [0.0105, 0.0557] |
| `lambda_l2` | 0.6878 | 0.6853 | 0.6860 | 0.6924 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 0.6864 | 0.6860 | 0.6851 | 0.6940 | **Q3** [0.5659, 0.5959] |
| `bagging_fraction` | 0.6946 | 0.6863 | 0.6842 | 0.6863 | **Q3** [0.9006, 0.9204] |

#### E. Slice plot

![wine_quality@s44/naive-lightgbm](slice_wine_quality@s44_naive-lightgbm.png)


### moe

- **holdout RMSE: 0.59740** (winner retrained in 0.85s, cv score of winner: 0.6689)
- cv best RMSE: 0.6689, median: 0.6816, p10: 0.6737
- train: median 0.692s/fold, mean 2.536s, p90 6.826s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_warmup_iters` | 0.234 |
| `mixture_r_smoothing` | 0.145 |
| `learning_rate` | 0.107 |
| `mixture_balance_factor` | 0.081 |
| `lambda_l1` | 0.077 |
| `bagging_fraction` | 0.074 |
| `mixture_hard_m_step` | 0.031 |
| `mixture_init` | 0.028 |
| `num_leaves` | 0.028 |
| `lambda_l2` | 0.027 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_r_smoothing` | **ema** | 0.8104 (n=432) | none | Δ +0.6892 | p=1.56e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 423 | 0.8988 | 0.7927 | 0.6689 |
| leaf_reuse | 24 | 0.9515 | 0.6315 | 0.6800 |
| gbdt | 53 | 1.1668 | 1.0338 | 0.6757 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 455 | 0.8978 | 0.7610 | 0.6689 |
| expert_choice | 45 | 1.2524 | 1.2178 | 0.6806 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| em | 138 | 0.7634 | 0.3504 | 0.6736 |
| loss_only | 100 | 0.9642 | 0.8916 | 0.6760 |
| gate_only | 262 | 1.0042 | 0.9442 | 0.6689 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| random | 84 | 0.7883 | 0.5910 | 0.6739 |
| gmm | 34 | 0.8832 | 0.5735 | 0.6749 |
| uniform | 40 | 0.9049 | 0.9399 | 0.6826 |
| kmeans_features | 290 | 0.9611 | 0.8639 | 0.6689 |
| tree_hierarchical | 16 | 1.0100 | 0.5253 | 0.6765 |
| gmm_features | 36 | 1.0430 | 1.0009 | 0.6729 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 432 | 0.8104 | 0.6317 | 0.6689 |
| none | 41 | 1.4996 | 1.2732 | 0.6962 |
| markov | 27 | 1.9739 | 1.3279 | 0.6921 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 421 | 0.8702 | 0.6830 | 0.6689 |
| True | 79 | 1.2468 | 1.2807 | 0.6906 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 469 | 0.8862 | 0.7185 | 0.6689 |
| True | 31 | 1.5877 | 1.5960 | 0.6864 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.9595 | — | 1.0485 | 0.8193 | **Q4** [5.0, ∞) |
| `mixture_diversity_lambda` | 0.7734 | 0.9252 | 0.9724 | 1.0479 | **Q1** [None, 0.0812] |
| `mixture_warmup_iters` | 1.0311 | 0.9783 | 0.8875 | 0.8655 | **Q4** [34.0, ∞) |
| `mixture_balance_factor` | 0.9830 | — | 0.8380 | 1.0811 | **Q3** [7.0, 8.0] |
| `learning_rate` | 1.0319 | 0.7830 | 1.0300 | 0.8741 | **Q2** [0.1691, 0.2274] |
| `num_leaves` | 0.9024 | 0.7842 | 0.9735 | 1.0511 | **Q2** [48.75, 73.0] |
| `max_depth` | 1.1368 | 0.6967 | — | 0.9673 | **Q2** [10.0, 12.0] |
| `min_data_in_leaf` | 0.8617 | 0.8523 | 0.9054 | 1.0863 | **Q2** [7.0, 10.0] |

#### E. Slice plot

![wine_quality@s44/moe](slice_wine_quality@s44_moe.png)


---

## Overall recommendations

**Categorical settings that are statistically significant winners (p<0.01):**

| dataset | param | best value | Δ vs runner-up | p |
|---|---|---|---|---|
| wine_quality@s43 | `mixture_routing_mode` | **token_choice** | +0.3154 | 1.60e-03 |
| wine_quality@s44 | `mixture_r_smoothing` | **ema** | +0.6892 | 1.56e-03 |
