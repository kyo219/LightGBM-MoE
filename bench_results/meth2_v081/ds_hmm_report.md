# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 300

- **Datasets**: ['hmm'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `0cf3634c544c`, lib sha256 `a3c30fcc7fd3…`, package `/home/mokumoku/Develop/LightGBM-MoE/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| hmm | naive-lightgbm | **2.21802** ± 0.24377 | 2.27009 | 0.0% | 0.06 |
| hmm | naive-ensemble | **2.20636** ± 0.24187 | 2.26524 | 0.0% | 0.13 |
| hmm | moe | **2.21037** ± 0.24330 | 2.24682 | 0.0% | 0.22 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| hmm@s42 | naive-lightgbm | 2.2992 | 2.3152 | 0.033 | 58 |
| hmm@s42 | naive-ensemble | 2.2955 | 2.3144 | 0.105 | 163 |
| hmm@s42 | moe | 2.2914 | 2.3413 | 0.165 | 783 |
| hmm@s43 | naive-lightgbm | 2.3800 | 2.3935 | 0.033 | 57 |
| hmm@s43 | naive-ensemble | 2.3741 | 2.3876 | 0.089 | 152 |
| hmm@s43 | moe | 2.3213 | 2.3586 | 0.240 | 414 |
| hmm@s44 | naive-lightgbm | 2.1311 | 2.1423 | 0.028 | 54 |
| hmm@s44 | naive-ensemble | 2.1261 | 2.1368 | 0.086 | 138 |
| hmm@s44 | moe | 2.1278 | 2.1430 | 0.125 | 347 |



---

## hmm@s42  (search X=[1600, 5], holdout n=400)


### naive-lightgbm

- **holdout RMSE: 1.88834** (winner retrained in 0.06s, cv score of winner: 2.2992)
- cv best RMSE: 2.2992, median: 2.3152, p10: 2.3078
- train: median 0.033s/fold, mean 0.034s, p90 0.041s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.429 |
| `learning_rate` | 0.262 |
| `min_data_in_leaf` | 0.101 |
| `bagging_fraction` | 0.057 |
| `num_leaves` | 0.056 |
| `max_depth` | 0.048 |
| `feature_fraction` | 0.027 |
| `bagging_freq` | 0.012 |
| `lambda_l2` | 0.005 |
| `lambda_l1` | 0.004 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 280 | 2.3177 | 0.0143 | 2.2992 |
| False | 20 | 2.3549 | 0.0208 | 2.3261 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.3258 | 2.3142 | 2.3149 | 2.3260 | **Q2** [0.0611, 0.0714] |
| `num_leaves` | 2.3210 | 2.3205 | 2.3198 | 2.3196 | **Q4** [120.0, ∞) |
| `max_depth` | — | — | 2.3163 | 2.3274 | **Q3** [3.0, 5.0] |
| `min_data_in_leaf` | 2.3232 | 2.3162 | 2.3158 | 2.3262 | **Q3** [31.0, 36.0] |
| `lambda_l1` | 2.3321 | 2.3174 | 2.3162 | 2.3152 | **Q4** [2.9866, ∞) |
| `lambda_l2` | 2.3254 | 2.3201 | 2.3165 | 2.3189 | **Q3** [0.0494, 0.2783] |
| `feature_fraction` | 2.3295 | 2.3163 | 2.3155 | 2.3195 | **Q3** [0.8977, 0.9296] |
| `bagging_fraction` | 2.3310 | 2.3194 | 2.3147 | 2.3158 | **Q3** [0.934, 0.9603] |

#### E. Slice plot

![hmm@s42/naive-lightgbm](slice_hmm@s42_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 1.88558** (winner retrained in 0.17s, cv score of winner: 2.2955)
- cv best RMSE: 2.2955, median: 2.3144, p10: 2.3046
- train: median 0.105s/fold, mean 0.104s, p90 0.141s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.457 |
| `learning_rate` | 0.307 |
| `min_data_in_leaf` | 0.063 |
| `bagging_fraction` | 0.050 |
| `feature_fraction` | 0.043 |
| `max_depth` | 0.039 |
| `num_leaves` | 0.030 |
| `lambda_l1` | 0.006 |
| `bagging_freq` | 0.004 |
| `lambda_l2` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 277 | 2.3167 | 0.0156 | 2.2955 |
| False | 23 | 2.3513 | 0.0184 | 2.3146 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.3338 | 2.3168 | 2.3126 | 2.3140 | **Q3** [0.252, 0.2766] |
| `num_leaves` | 2.3246 | 2.3162 | 2.3134 | 2.3230 | **Q3** [100.0, 105.0] |
| `max_depth` | 2.3336 | 2.3238 | — | 2.3157 | **Q4** [11.0, ∞) |
| `min_data_in_leaf` | 2.3218 | 2.3199 | 2.3131 | 2.3230 | **Q3** [30.0, 34.0] |
| `lambda_l1` | 2.3133 | 2.3188 | 2.3269 | 2.3182 | **Q1** [None, 0.0] |
| `lambda_l2` | 2.3234 | 2.3201 | 2.3163 | 2.3174 | **Q3** [0.003, 0.0226] |
| `feature_fraction` | 2.3209 | 2.3217 | 2.3152 | 2.3195 | **Q3** [0.7787, 0.8628] |
| `bagging_fraction` | 2.3241 | 2.3179 | 2.3153 | 2.3199 | **Q3** [0.825, 0.8466] |

#### E. Slice plot

![hmm@s42/naive-ensemble](slice_hmm@s42_naive-ensemble.png)


### moe

- **holdout RMSE: 1.89065** (winner retrained in 0.26s, cv score of winner: 2.2914)
- cv best RMSE: 2.2914, median: 2.3413, p10: 2.3129
- train: median 0.165s/fold, mean 0.514s, p90 1.957s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_gate_type` | 0.203 |
| `learning_rate` | 0.168 |
| `mixture_diversity_lambda` | 0.155 |
| `bagging_fraction` | 0.088 |
| `mixture_init` | 0.082 |
| `mixture_warmup_iters` | 0.050 |
| `min_data_in_leaf` | 0.034 |
| `mixture_num_experts` | 0.033 |
| `num_leaves` | 0.029 |
| `mixture_routing_mode` | 0.028 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **gbdt** | 2.3336 (n=201) | none | Δ +0.0473 | p=0.00e+00 |
| `mixture_routing_mode` | **expert_choice** | 2.3460 (n=277) | token_choice | Δ +0.0555 | p=3.07e-03 |
| `mixture_e_step_mode` | **em** | 2.3379 (n=208) | gate_only | Δ +0.0376 | p=0.00e+00 |
| `mixture_init` | **tree_hierarchical** | 2.3366 (n=201) | uniform | Δ +0.0367 | p=0.00e+00 |
| `mixture_r_smoothing` | **none** | 2.3391 (n=230) | markov | Δ +0.0395 | p=7.55e-04 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 201 | 2.3336 | 0.0243 | 2.2914 |
| none | 77 | 2.3809 | 0.0353 | 2.3122 |
| leaf_reuse | 22 | 2.3945 | 0.0812 | 2.3273 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 277 | 2.3460 | 0.0343 | 2.2914 |
| token_choice | 23 | 2.4015 | 0.0780 | 2.3443 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| em | 208 | 2.3379 | 0.0377 | 2.2914 |
| gate_only | 47 | 2.3755 | 0.0416 | 2.3051 |
| loss_only | 45 | 2.3809 | 0.0332 | 2.3122 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| tree_hierarchical | 201 | 2.3366 | 0.0403 | 2.2914 |
| uniform | 40 | 2.3733 | 0.0282 | 2.3124 |
| random | 45 | 2.3787 | 0.0273 | 2.3122 |
| gmm | 14 | 2.3885 | 0.0416 | 2.3348 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 230 | 2.3391 | 0.0284 | 2.2914 |
| markov | 17 | 2.3786 | 0.0380 | 2.3298 |
| ema | 53 | 2.3896 | 0.0610 | 2.3122 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 225 | 2.3409 | 0.0408 | 2.2914 |
| False | 75 | 2.3782 | 0.0323 | 2.3122 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 279 | 2.3470 | 0.0407 | 2.2914 |
| False | 21 | 2.3936 | 0.0349 | 2.3409 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 2.3748 | — | — | 2.3423 | **Q4** [3.0, ∞) |
| `mixture_diversity_lambda` | 2.3511 | 2.3314 | 2.3416 | 2.3768 | **Q2** [0.2232, 0.2582] |
| `mixture_warmup_iters` | 2.3288 | 2.3312 | 2.3539 | 2.3821 | **Q1** [None, 7.0] |
| `mixture_balance_factor` | 2.3823 | 2.3417 | — | 2.3398 | **Q4** [7.0, ∞) |
| `learning_rate` | 2.3525 | 2.3272 | 2.3595 | 2.3618 | **Q2** [0.0664, 0.0853] |
| `num_leaves` | 2.3773 | 2.3506 | 2.3361 | 2.3388 | **Q3** [111.0, 119.0] |
| `max_depth` | — | 2.3334 | 2.3480 | 2.3807 | **Q2** [3.0, 4.0] |
| `min_data_in_leaf` | 2.3442 | 2.3266 | 2.3496 | 2.3780 | **Q2** [28.0, 32.0] |

#### E. Slice plot

![hmm@s42/moe](slice_hmm@s42_moe.png)


---

## hmm@s43  (search X=[1600, 5], holdout n=400)


### naive-lightgbm

- **holdout RMSE: 2.47018** (winner retrained in 0.07s, cv score of winner: 2.3800)
- cv best RMSE: 2.3800, median: 2.3935, p10: 2.3844
- train: median 0.033s/fold, mean 0.034s, p90 0.046s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.532 |
| `extra_trees` | 0.222 |
| `learning_rate` | 0.098 |
| `bagging_fraction` | 0.056 |
| `lambda_l1` | 0.035 |
| `feature_fraction` | 0.033 |
| `max_depth` | 0.013 |
| `num_leaves` | 0.005 |
| `bagging_freq` | 0.003 |
| `lambda_l2` | 0.003 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 279 | 2.3973 | 0.0165 | 2.3800 |
| False | 21 | 2.4486 | 0.0327 | 2.4035 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.4073 | 2.3941 | 2.3955 | 2.4069 | **Q2** [0.0531, 0.0605] |
| `num_leaves` | 2.3941 | 2.4000 | 2.4058 | 2.4035 | **Q1** [None, 15.75] |
| `max_depth` | 2.4084 | 2.3978 | 2.3967 | 2.4019 | **Q3** [9.0, 10.0] |
| `min_data_in_leaf` | 2.3942 | 2.3915 | 2.3982 | 2.4191 | **Q2** [7.0, 10.0] |
| `lambda_l1` | 2.4157 | 2.3956 | 2.3973 | 2.3950 | **Q4** [4.0999, ∞) |
| `lambda_l2` | 2.4147 | 2.3962 | 2.3976 | 2.3951 | **Q4** [1.2586, ∞) |
| `feature_fraction` | 2.4142 | 2.4004 | 2.3950 | 2.3941 | **Q4** [0.9768, ∞) |
| `bagging_fraction` | 2.4139 | 2.3994 | 2.3933 | 2.3972 | **Q3** [0.892, 0.921] |

#### E. Slice plot

![hmm@s43/naive-lightgbm](slice_hmm@s43_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 2.46958** (winner retrained in 0.12s, cv score of winner: 2.3741)
- cv best RMSE: 2.3741, median: 2.3876, p10: 2.3785
- train: median 0.089s/fold, mean 0.098s, p90 0.151s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.523 |
| `min_data_in_leaf` | 0.222 |
| `learning_rate` | 0.215 |
| `lambda_l1` | 0.013 |
| `bagging_fraction` | 0.006 |
| `bagging_freq` | 0.006 |
| `feature_fraction` | 0.004 |
| `num_leaves` | 0.004 |
| `max_depth` | 0.003 |
| `n_models` | 0.003 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 280 | 2.3910 | 0.0183 | 2.3741 |
| False | 20 | 2.4489 | 0.0188 | 2.4261 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.4087 | 2.3928 | 2.3905 | 2.3875 | **Q4** [0.2655, ∞) |
| `num_leaves` | 2.3946 | 2.3916 | 2.3948 | 2.3990 | **Q2** [20.0, 25.5] |
| `max_depth` | 2.4131 | 2.3953 | — | 2.3897 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 2.3886 | 2.3913 | 2.3893 | 2.4099 | **Q1** [None, 7.75] |
| `lambda_l1` | 2.4131 | 2.3930 | 2.3869 | 2.3865 | **Q4** [5.8577, ∞) |
| `lambda_l2` | 2.3935 | 2.3898 | 2.3918 | 2.4044 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 2.4070 | 2.3939 | 2.3901 | 2.3885 | **Q4** [0.9816, ∞) |
| `bagging_fraction` | 2.3984 | 2.3924 | 2.3896 | 2.3992 | **Q3** [0.7428, 0.7681] |

#### E. Slice plot

![hmm@s43/naive-ensemble](slice_hmm@s43_naive-ensemble.png)


### moe

- **holdout RMSE: 2.48033** (winner retrained in 0.27s, cv score of winner: 2.3213)
- cv best RMSE: 2.3213, median: 2.3586, p10: 2.3380
- train: median 0.240s/fold, mean 0.267s, p90 0.412s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_e_step_mode` | 0.344 |
| `mixture_routing_mode` | 0.180 |
| `mixture_gate_type` | 0.101 |
| `mixture_r_smoothing` | 0.071 |
| `bagging_freq` | 0.055 |
| `mixture_diversity_lambda` | 0.051 |
| `num_leaves` | 0.043 |
| `learning_rate` | 0.029 |
| `min_data_in_leaf` | 0.024 |
| `feature_fraction` | 0.022 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **leaf_reuse** | 2.3690 (n=266) | gbdt | Δ +0.0451 | p=2.42e-03 |
| `mixture_routing_mode` | **token_choice** | 2.3710 (n=280) | expert_choice | Δ +0.0916 | p=1.30e-05 |
| `mixture_e_step_mode` | **gate_only** | 2.3659 (n=265) | loss_only | Δ +0.0647 | p=6.69e-04 |
| `mixture_r_smoothing` | **ema** | 2.3624 (n=163) | markov | Δ +0.0265 | p=2.60e-05 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 266 | 2.3690 | 0.0436 | 2.3213 |
| gbdt | 18 | 2.4141 | 0.0518 | 2.3617 |
| none | 16 | 2.4707 | 0.0784 | 2.3799 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 280 | 2.3710 | 0.0456 | 2.3213 |
| expert_choice | 20 | 2.4626 | 0.0687 | 2.3749 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 265 | 2.3659 | 0.0366 | 2.3213 |
| loss_only | 18 | 2.4306 | 0.0642 | 2.3725 |
| em | 17 | 2.4955 | 0.0689 | 2.3819 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| tree_hierarchical | 255 | 2.3720 | 0.0514 | 2.3213 |
| random | 16 | 2.3897 | 0.0304 | 2.3666 |
| uniform | 14 | 2.3928 | 0.0250 | 2.3607 |
| gmm | 15 | 2.4363 | 0.0694 | 2.3727 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 163 | 2.3624 | 0.0326 | 2.3213 |
| markov | 121 | 2.3889 | 0.0610 | 2.3361 |
| none | 16 | 2.4386 | 0.0778 | 2.3393 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 211 | 2.3635 | 0.0372 | 2.3213 |
| True | 89 | 2.4093 | 0.0679 | 2.3448 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 277 | 2.3741 | 0.0522 | 2.3213 |
| False | 23 | 2.4129 | 0.0455 | 2.3500 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 2.4735 | — | — | 2.3738 | **Q4** [3.0, ∞) |
| `mixture_diversity_lambda` | 2.3750 | 2.3537 | 2.3814 | 2.3985 | **Q2** [0.3094, 0.3349] |
| `mixture_warmup_iters` | 2.3628 | 2.3648 | 2.3793 | 2.3969 | **Q1** [None, 8.0] |
| `mixture_balance_factor` | 2.3838 | — | — | 2.3755 | **Q4** [6.0, ∞) |
| `learning_rate` | 2.3999 | 2.3754 | 2.3649 | 2.3683 | **Q3** [0.1507, 0.1813] |
| `num_leaves` | 2.3929 | 2.3568 | 2.3691 | 2.3890 | **Q2** [80.75, 90.0] |
| `max_depth` | 2.3970 | 2.4049 | — | 2.3611 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 2.3914 | 2.3647 | 2.3659 | 2.3883 | **Q2** [18.0, 24.0] |

#### E. Slice plot

![hmm@s43/moe](slice_hmm@s43_moe.png)


---

## hmm@s44  (search X=[1600, 5], holdout n=400)


### naive-lightgbm

- **holdout RMSE: 2.29553** (winner retrained in 0.04s, cv score of winner: 2.1311)
- cv best RMSE: 2.1311, median: 2.1423, p10: 2.1347
- train: median 0.028s/fold, mean 0.032s, p90 0.043s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.495 |
| `max_depth` | 0.155 |
| `bagging_fraction` | 0.121 |
| `min_data_in_leaf` | 0.092 |
| `learning_rate` | 0.075 |
| `feature_fraction` | 0.040 |
| `num_leaves` | 0.017 |
| `bagging_freq` | 0.003 |
| `lambda_l1` | 0.001 |
| `lambda_l2` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 278 | 2.1441 | 0.0129 | 2.1311 |
| False | 22 | 2.1867 | 0.0243 | 2.1570 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.1514 | 2.1411 | 2.1454 | 2.1511 | **Q2** [0.0406, 0.0468] |
| `num_leaves` | 2.1552 | 2.1441 | 2.1438 | 2.1462 | **Q3** [102.0, 112.0] |
| `max_depth` | — | — | 2.1430 | 2.1581 | **Q3** [3.0, 4.0] |
| `min_data_in_leaf` | 2.1393 | 2.1438 | 2.1472 | 2.1582 | **Q1** [None, 6.0] |
| `lambda_l1` | 2.1468 | 2.1476 | 2.1447 | 2.1498 | **Q3** [0.0, 0.0] |
| `lambda_l2` | 2.1443 | 2.1449 | 2.1449 | 2.1548 | **Q1** [None, 0.0] |
| `feature_fraction` | 2.1508 | 2.1424 | 2.1434 | 2.1522 | **Q2** [0.7138, 0.7421] |
| `bagging_fraction` | 2.1590 | 2.1422 | 2.1434 | 2.1443 | **Q2** [0.8818, 0.9284] |

#### E. Slice plot

![hmm@s44/naive-lightgbm](slice_hmm@s44_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 2.26391** (winner retrained in 0.09s, cv score of winner: 2.1261)
- cv best RMSE: 2.1261, median: 2.1368, p10: 2.1309
- train: median 0.086s/fold, mean 0.089s, p90 0.111s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.354 |
| `extra_trees` | 0.326 |
| `min_data_in_leaf` | 0.192 |
| `feature_fraction` | 0.051 |
| `bagging_fraction` | 0.025 |
| `num_leaves` | 0.019 |
| `bagging_freq` | 0.012 |
| `lambda_l1` | 0.011 |
| `max_depth` | 0.006 |
| `n_models` | 0.004 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 279 | 2.1405 | 0.0146 | 2.1261 |
| False | 21 | 2.1840 | 0.0223 | 2.1586 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.1595 | 2.1392 | 2.1359 | 2.1394 | **Q3** [0.1555, 0.1872] |
| `num_leaves` | 2.1464 | 2.1367 | 2.1387 | 2.1521 | **Q2** [52.0, 58.0] |
| `max_depth` | — | — | 2.1398 | 2.1535 | **Q3** [3.0, 8.0] |
| `min_data_in_leaf` | 2.1376 | 2.1386 | 2.1416 | 2.1553 | **Q1** [None, 8.0] |
| `lambda_l1` | 2.1547 | 2.1422 | 2.1396 | 2.1375 | **Q4** [5.8191, ∞) |
| `lambda_l2` | 2.1390 | 2.1381 | 2.1434 | 2.1535 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 2.1481 | 2.1408 | 2.1442 | 2.1409 | **Q2** [0.6187, 0.6597] |
| `bagging_fraction` | 2.1497 | 2.1378 | 2.1386 | 2.1479 | **Q2** [0.6736, 0.735] |

#### E. Slice plot

![hmm@s44/naive-ensemble](slice_hmm@s44_naive-ensemble.png)


### moe

- **holdout RMSE: 2.26012** (winner retrained in 0.15s, cv score of winner: 2.1278)
- cv best RMSE: 2.1278, median: 2.1430, p10: 2.1345
- train: median 0.125s/fold, mean 0.225s, p90 0.291s
- finite trials: 300 / 300

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_init` | 0.625 |
| `feature_fraction` | 0.099 |
| `mixture_routing_mode` | 0.093 |
| `learning_rate` | 0.027 |
| `mixture_r_smoothing` | 0.025 |
| `bagging_fraction` | 0.019 |
| `mixture_balance_factor` | 0.016 |
| `mixture_warmup_iters` | 0.015 |
| `mixture_gate_type` | 0.014 |
| `min_data_in_leaf` | 0.013 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 248 | 2.1613 | 0.0589 | 2.1278 |
| leaf_reuse | 36 | 2.1708 | 0.0534 | 2.1336 |
| gbdt | 16 | 2.2004 | 0.0696 | 2.1390 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 253 | 2.1617 | 0.0595 | 2.1278 |
| token_choice | 47 | 2.1799 | 0.0573 | 2.1373 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 155 | 2.1586 | 0.0550 | 2.1293 |
| em | 108 | 2.1670 | 0.0627 | 2.1278 |
| gate_only | 37 | 2.1819 | 0.0642 | 2.1373 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| random | 249 | 2.1506 | 0.0338 | 2.1278 |
| uniform | 21 | 2.1630 | 0.0353 | 2.1383 |
| gmm | 16 | 2.2358 | 0.0906 | 2.1404 |
| tree_hierarchical | 14 | 2.3333 | 0.0819 | 2.1612 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 257 | 2.1590 | 0.0567 | 2.1278 |
| ema | 27 | 2.1859 | 0.0540 | 2.1397 |
| markov | 16 | 2.2173 | 0.0772 | 2.1366 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 272 | 2.1615 | 0.0568 | 2.1278 |
| True | 28 | 2.1940 | 0.0755 | 2.1313 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 279 | 2.1623 | 0.0599 | 2.1278 |
| False | 21 | 2.1939 | 0.0451 | 2.1552 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 2.1877 | — | — | 2.1616 | **Q4** [4.0, ∞) |
| `mixture_diversity_lambda` | 2.1540 | 2.1519 | 2.1671 | 2.1851 | **Q2** [0.0221, 0.0423] |
| `mixture_warmup_iters` | 2.1656 | 2.1639 | 2.1556 | 2.1699 | **Q3** [21.5, 24.0] |
| `mixture_balance_factor` | 2.1606 | — | 2.1621 | 2.1731 | **Q1** [None, 5.0] |
| `learning_rate` | 2.1796 | 2.1568 | 2.1491 | 2.1726 | **Q3** [0.0761, 0.0932] |
| `num_leaves` | 2.1557 | 2.1741 | 2.1560 | 2.1720 | **Q1** [None, 39.0] |
| `max_depth` | — | — | 2.1577 | 2.1812 | **Q3** [3.0, 4.0] |
| `min_data_in_leaf` | 2.1556 | 2.1572 | 2.1669 | 2.1774 | **Q1** [None, 8.0] |

#### E. Slice plot

![hmm@s44/moe](slice_hmm@s44_moe.png)


---

## Overall recommendations

**Categorical settings that are statistically significant winners (p<0.01):**

| dataset | param | best value | Δ vs runner-up | p |
|---|---|---|---|---|
| hmm@s42 | `mixture_gate_type` | **gbdt** | +0.0473 | 0.00e+00 |
| hmm@s42 | `mixture_routing_mode` | **expert_choice** | +0.0555 | 3.07e-03 |
| hmm@s42 | `mixture_e_step_mode` | **em** | +0.0376 | 0.00e+00 |
| hmm@s42 | `mixture_init` | **tree_hierarchical** | +0.0367 | 0.00e+00 |
| hmm@s42 | `mixture_r_smoothing` | **none** | +0.0395 | 7.55e-04 |
| hmm@s43 | `mixture_gate_type` | **leaf_reuse** | +0.0451 | 2.42e-03 |
| hmm@s43 | `mixture_routing_mode` | **token_choice** | +0.0916 | 1.30e-05 |
| hmm@s43 | `mixture_e_step_mode` | **gate_only** | +0.0647 | 6.69e-04 |
| hmm@s43 | `mixture_r_smoothing` | **ema** | +0.0265 | 2.60e-05 |
