# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 500

- **Datasets**: ['synthetic', 'fred_gdp', 'sp500_basic', 'sp500', 'vix', 'hmm', 'houses', 'cpu_act', 'elevators', 'wine_quality'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `c0608d3ca705`, lib sha256 `3f08b4dc3421…`, package `/home/mokumoku/Develop/LightGBM-MoE/python-package/lightgbm_moe`


---

## README tables (generated — paste verbatim)

**Regime datasets**

| Dataset | Shape | naive | ensemble | MoE | MoE vs naive | **MoE vs ensemble** |
|---|---|---|---|---|---|---|
| `synthetic` | 2000 × 5 | 4.840 ± 0.311 | 4.723 ± 0.327 | **3.770 ± 0.222** | −22.1 % | **−20.2 %** |
| `vix` | 3763 × 13 | 1.768 ± 0.091 | **1.739 ± 0.051** | 1.776 ± 0.219 | +0.4 % | **+2.1 %** |
| `fred_gdp` | 311 × 12 | 1.553 ± 0.011 | 1.540 ± 0.017 | **1.504 ± 0.003** | −3.1 % | **−2.3 %** |
| `hmm` | 2000 × 5 | 2.216 ± 0.244 | **2.208 ± 0.242** | 2.258 ± 0.308 | +1.9 % | **+2.3 %** |

**Non-regime control datasets**

| Dataset | Shape | naive | ensemble | MoE | MoE vs naive | **MoE vs ensemble** |
|---|---|---|---|---|---|---|
| `houses` | 10000 × 8 | 50109 ± 902 | 48816 ± 1005 | **48668 ± 533** | −2.9 % | **−0.3 %** |
| `cpu_act` | 8192 × 21 | 2.498 ± 0.297 | 2.448 ± 0.336 | **2.416 ± 0.332** | −3.3 % | **−1.3 %** |
| `elevators` | 10000 × 18 | 0.002357 ± 0.000080 | **0.002296 ± 0.000060** | 0.002332 ± 0.000099 | −1.1 % | **+1.6 %** |
| `wine_quality` | 6497 × 11 | 0.6146 ± 0.0141 | 0.6065 ± 0.0126 | **0.6034 ± 0.0120** | −1.8 % | **−0.5 %** |

**Noise-floor datasets** (next-day returns)

| Dataset | Shape | naive | ensemble | MoE | MoE vs naive | **MoE vs ensemble** |
|---|---|---|---|---|---|---|
| `sp500_basic` | 3762 × 13 | **0.01102 ± 0.00001** | 0.01103 ± 0.00002 | 0.01103 ± 0.00001 | +0.1 % | **+0.0 %** |
| `sp500` | 3711 × 28 | 0.01109 ± 0.00003 | **0.01104 ± 0.00001** | 0.01105 ± 0.00001 | −0.4 % | **+0.0 %** |

**Compute cost** (median train s/fold across trials, mean over seeds)

| Dataset | naive | ensemble | MoE | MoE / naive | MoE / ensemble |
|---|---|---|---|---|---|
| `synthetic` | 0.041 | 0.210 | 0.311 | 7.5 × | 1.5 × |
| `fred_gdp` | 0.013 | 0.108 | 0.098 | 7.3 × | 0.9 × |
| `sp500_basic` | 0.018 | 0.086 | 0.109 | 6.0 × | 1.3 × |
| `sp500` | 0.026 | 0.189 | 0.200 | 7.5 × | 1.1 × |
| `vix` | 0.038 | 0.146 | 0.406 | 10.7 × | 2.8 × |
| `hmm` | 0.032 | 0.184 | 0.163 | 5.0 × | 0.9 × |
| `houses` | 0.136 | 0.512 | 4.189 | 30.8 × | 8.2 × |
| `cpu_act` | 0.074 | 0.309 | 0.638 | 8.6 × | 2.1 × |
| `elevators` | 0.064 | 0.203 | 0.845 | 13.1 × | 4.2 × |
| `wine_quality` | 0.149 | 0.598 | 0.505 | 3.4 × | 0.8 × |

**Winning MoE configs** (30 = datasets × seeds)

| Knob | Winners |
|---|---|
| `mixture_diversity_lambda` > 0.02 | 28 / 30 |
| `mixture_expert_dropout_rate` > 0.02 | 26 / 30 |
| `mixture_load_balance_alpha` > 0.02 | 29 / 30 |
| `mixture_gate_entropy_lambda` > 0.02 (gated winners) | 21 / 25 |
| gate `T_init` > 1.5 (gated winners) | 12 / 25 |
| `mixture_refit_leaves` ON (effective, i.e. not `leaf_reuse`) | 11 / 30 |
| `mixture_regrow_oldest_trees` ON (effective) | 6 / 30 |
| `mixture_hard_m_step` soft (False) | 15 / 30 |
| `mixture_init` | random 9, gmm 6, gmm_features 5, uniform 5, tree_hierarchical 3, kmeans_features 2 |
| `mixture_gate_type` | gbdt 14, leaf_reuse 11, none 5 |
| `mixture_routing_mode` | token_choice 19, expert_choice 11 |
| `mixture_num_experts` | K=5 8, K=6 8, K=2 7, K=3 4, K=4 3 |


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| synthetic | naive-lightgbm | **4.83980** ± 0.31113 | 5.48121 | 0.0% | 0.08 |
| synthetic | naive-ensemble | **4.72332** ± 0.32678 | 5.42496 | 0.0% | 0.44 |
| synthetic | moe | **3.76981** ± 0.22177 | 4.27284 | 0.0% | 1.04 |
| fred_gdp | naive-lightgbm | **1.55306** ± 0.01118 | 0.80372 | 0.0% | 0.02 |
| fred_gdp | naive-ensemble | **1.53993** ± 0.01741 | 0.80549 | 0.0% | 0.10 |
| fred_gdp | moe | **1.50443** ± 0.00268 | 0.82107 | 0.0% | 0.51 |
| sp500_basic | naive-lightgbm | **0.01102** ± 0.00001 | 0.00969 | 0.0% | 0.02 |
| sp500_basic | naive-ensemble | **0.01103** ± 0.00002 | 0.00969 | 0.0% | 0.10 |
| sp500_basic | moe | **0.01103** ± 0.00001 | 0.00970 | 0.0% | 0.26 |
| sp500 | naive-lightgbm | **0.01109** ± 0.00003 | 0.00975 | 0.0% | 0.04 |
| sp500 | naive-ensemble | **0.01104** ± 0.00001 | 0.00974 | 0.0% | 0.23 |
| sp500 | moe | **0.01105** ± 0.00001 | 0.00975 | 0.0% | 0.30 |
| vix | naive-lightgbm | **1.76822** ± 0.09140 | 2.61915 | 0.0% | 0.05 |
| vix | naive-ensemble | **1.73943** ± 0.05112 | 2.64216 | 0.0% | 0.15 |
| vix | moe | **1.77618** ± 0.21950 | 2.41161 | 0.0% | 0.68 |
| hmm | naive-lightgbm | **2.21586** ± 0.24444 | 2.26814 | 0.0% | 0.05 |
| hmm | naive-ensemble | **2.20801** ± 0.24193 | 2.26482 | 0.0% | 0.16 |
| hmm | moe | **2.25844** ± 0.30799 | 2.26137 | 0.0% | 0.33 |
| houses | naive-lightgbm | **50108.73806** ± 901.62518 | 50708.02892 | 0.0% | 0.16 |
| houses | naive-ensemble | **48816.10629** ± 1005.30253 | 50343.50687 | 0.0% | 0.67 |
| houses | moe | **48668.43377** ± 533.30348 | 50076.27509 | 0.0% | 4.45 |
| cpu_act | naive-lightgbm | **2.49817** ± 0.29668 | 2.44834 | 0.0% | 0.13 |
| cpu_act | naive-ensemble | **2.44848** ± 0.33583 | 2.41525 | 0.0% | 0.31 |
| cpu_act | moe | **2.41578** ± 0.33226 | 2.36944 | 0.0% | 1.61 |
| elevators | naive-lightgbm | **0.00236** ± 0.00008 | 0.00263 | 0.0% | 0.11 |
| elevators | naive-ensemble | **0.00230** ± 0.00006 | 0.00252 | 0.0% | 0.28 |
| elevators | moe | **0.00233** ± 0.00010 | 0.00257 | 0.0% | 1.59 |
| wine_quality | naive-lightgbm | **0.61457** ± 0.01410 | 0.66815 | 0.0% | 0.21 |
| wine_quality | naive-ensemble | **0.60653** ± 0.01261 | 0.66314 | 0.0% | 0.88 |
| wine_quality | moe | **0.60341** ± 0.01196 | 0.66242 | 0.0% | 1.35 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| synthetic@s42 | naive-lightgbm | 5.3096 | 5.5469 | 0.040 | 153 |
| synthetic@s42 | naive-ensemble | 5.3159 | 5.6074 | 0.116 | 315 |
| synthetic@s42 | moe | 5.1551 | 5.4318 | 0.610 | 1651 |
| synthetic@s43 | naive-lightgbm | 5.5192 | 5.8184 | 0.040 | 163 |
| synthetic@s43 | naive-ensemble | 5.4566 | 5.7179 | 0.260 | 621 |
| synthetic@s43 | moe | 4.1840 | 4.7100 | 0.171 | 962 |
| synthetic@s44 | naive-lightgbm | 5.6149 | 6.0510 | 0.043 | 162 |
| synthetic@s44 | naive-ensemble | 5.5024 | 5.7684 | 0.254 | 704 |
| synthetic@s44 | moe | 3.4794 | 4.5422 | 0.152 | 561 |
| fred_gdp@s42 | naive-lightgbm | 0.8073 | 0.8248 | 0.012 | 100 |
| fred_gdp@s42 | naive-ensemble | 0.8098 | 0.8243 | 0.155 | 386 |
| fred_gdp@s42 | moe | 0.8196 | 0.8625 | 0.187 | 578 |
| fred_gdp@s43 | naive-lightgbm | 0.7992 | 0.8210 | 0.015 | 112 |
| fred_gdp@s43 | naive-ensemble | 0.8044 | 0.8218 | 0.083 | 232 |
| fred_gdp@s43 | moe | 0.8206 | 0.8563 | 0.066 | 437 |
| fred_gdp@s44 | naive-lightgbm | 0.8046 | 0.8286 | 0.013 | 102 |
| fred_gdp@s44 | naive-ensemble | 0.8023 | 0.8227 | 0.086 | 215 |
| fred_gdp@s44 | moe | 0.8231 | 0.8675 | 0.040 | 266 |
| sp500_basic@s42 | naive-lightgbm | 0.0097 | 0.0097 | 0.022 | 119 |
| sp500_basic@s42 | naive-ensemble | 0.0097 | 0.0097 | 0.078 | 221 |
| sp500_basic@s42 | moe | 0.0097 | 0.0097 | 0.068 | 290 |
| sp500_basic@s43 | naive-lightgbm | 0.0097 | 0.0097 | 0.015 | 98 |
| sp500_basic@s43 | naive-ensemble | 0.0097 | 0.0097 | 0.107 | 282 |
| sp500_basic@s43 | moe | 0.0097 | 0.0097 | 0.195 | 2046 |
| sp500_basic@s44 | naive-lightgbm | 0.0097 | 0.0097 | 0.017 | 103 |
| sp500_basic@s44 | naive-ensemble | 0.0097 | 0.0097 | 0.074 | 216 |
| sp500_basic@s44 | moe | 0.0097 | 0.0097 | 0.063 | 293 |
| sp500@s42 | naive-lightgbm | 0.0098 | 0.0098 | 0.027 | 136 |
| sp500@s42 | naive-ensemble | 0.0097 | 0.0098 | 0.288 | 747 |
| sp500@s42 | moe | 0.0098 | 0.0098 | 0.144 | 477 |
| sp500@s43 | naive-lightgbm | 0.0097 | 0.0098 | 0.031 | 150 |
| sp500@s43 | naive-ensemble | 0.0098 | 0.0098 | 0.147 | 399 |
| sp500@s43 | moe | 0.0097 | 0.0098 | 0.250 | 2223 |
| sp500@s44 | naive-lightgbm | 0.0097 | 0.0098 | 0.021 | 110 |
| sp500@s44 | naive-ensemble | 0.0097 | 0.0098 | 0.133 | 377 |
| sp500@s44 | moe | 0.0097 | 0.0098 | 0.204 | 848 |
| vix@s42 | naive-lightgbm | 2.6235 | 2.7139 | 0.031 | 139 |
| vix@s42 | naive-ensemble | 2.6508 | 2.7183 | 0.113 | 318 |
| vix@s42 | moe | 2.3993 | 2.5454 | 0.181 | 836 |
| vix@s43 | naive-lightgbm | 2.5849 | 2.6917 | 0.032 | 154 |
| vix@s43 | naive-ensemble | 2.6426 | 2.7052 | 0.139 | 443 |
| vix@s43 | moe | 2.3536 | 2.5795 | 0.157 | 2080 |
| vix@s44 | naive-lightgbm | 2.6491 | 2.7244 | 0.050 | 198 |
| vix@s44 | naive-ensemble | 2.6331 | 2.7077 | 0.188 | 489 |
| vix@s44 | moe | 2.4820 | 2.6453 | 0.881 | 2260 |
| hmm@s42 | naive-lightgbm | 2.2992 | 2.3144 | 0.019 | 113 |
| hmm@s42 | naive-ensemble | 2.2955 | 2.3133 | 0.162 | 403 |
| hmm@s42 | moe | 2.3100 | 2.3539 | 0.069 | 557 |
| hmm@s43 | naive-lightgbm | 2.3781 | 2.3928 | 0.038 | 174 |
| hmm@s43 | naive-ensemble | 2.3714 | 2.3870 | 0.195 | 508 |
| hmm@s43 | moe | 2.3420 | 2.3817 | 0.141 | 667 |
| hmm@s44 | naive-lightgbm | 2.1272 | 2.1405 | 0.039 | 178 |
| hmm@s44 | naive-ensemble | 2.1276 | 2.1354 | 0.195 | 497 |
| hmm@s44 | moe | 2.1321 | 2.1469 | 0.278 | 946 |
| houses@s42 | naive-lightgbm | 50493.3397 | 51154.2425 | 0.090 | 296 |
| houses@s42 | naive-ensemble | 50578.9081 | 50951.9468 | 0.501 | 1209 |
| houses@s42 | moe | 49880.1370 | 51081.5655 | 6.387 | 16085 |
| houses@s43 | naive-lightgbm | 50976.4089 | 51629.0269 | 0.149 | 430 |
| houses@s43 | naive-ensemble | 50306.1130 | 50702.0156 | 0.566 | 1374 |
| houses@s43 | moe | 49975.9778 | 50906.9051 | 0.730 | 3721 |
| houses@s44 | naive-lightgbm | 50654.3382 | 51120.2765 | 0.169 | 472 |
| houses@s44 | naive-ensemble | 50145.4995 | 50954.3367 | 0.468 | 1257 |
| houses@s44 | moe | 50372.7105 | 51559.1744 | 5.448 | 15672 |
| cpu_act@s42 | naive-lightgbm | 2.5168 | 2.5811 | 0.063 | 217 |
| cpu_act@s42 | naive-ensemble | 2.4889 | 2.5387 | 0.308 | 795 |
| cpu_act@s42 | moe | 2.4241 | 2.5144 | 0.741 | 2032 |
| cpu_act@s43 | naive-lightgbm | 2.3502 | 2.3963 | 0.098 | 299 |
| cpu_act@s43 | naive-ensemble | 2.3056 | 2.3412 | 0.294 | 764 |
| cpu_act@s43 | moe | 2.2881 | 2.3740 | 0.411 | 1828 |
| cpu_act@s44 | naive-lightgbm | 2.4780 | 2.5377 | 0.062 | 211 |
| cpu_act@s44 | naive-ensemble | 2.4512 | 2.4980 | 0.325 | 862 |
| cpu_act@s44 | moe | 2.3960 | 2.5175 | 0.762 | 2087 |
| elevators@s42 | naive-lightgbm | 0.0026 | 0.0027 | 0.044 | 167 |
| elevators@s42 | naive-ensemble | 0.0025 | 0.0026 | 0.198 | 549 |
| elevators@s42 | moe | 0.0026 | 0.0028 | 0.268 | 2050 |
| elevators@s43 | naive-lightgbm | 0.0026 | 0.0027 | 0.070 | 241 |
| elevators@s43 | naive-ensemble | 0.0025 | 0.0026 | 0.198 | 575 |
| elevators@s43 | moe | 0.0025 | 0.0026 | 0.880 | 5645 |
| elevators@s44 | naive-lightgbm | 0.0026 | 0.0027 | 0.079 | 255 |
| elevators@s44 | naive-ensemble | 0.0025 | 0.0025 | 0.212 | 551 |
| elevators@s44 | moe | 0.0026 | 0.0029 | 1.387 | 3874 |
| wine_quality@s42 | naive-lightgbm | 0.6678 | 0.6752 | 0.157 | 444 |
| wine_quality@s42 | naive-ensemble | 0.6624 | 0.6697 | 0.625 | 1520 |
| wine_quality@s42 | moe | 0.6668 | 0.6759 | 0.423 | 1462 |
| wine_quality@s43 | naive-lightgbm | 0.6616 | 0.6684 | 0.153 | 437 |
| wine_quality@s43 | naive-ensemble | 0.6569 | 0.6621 | 0.619 | 1482 |
| wine_quality@s43 | moe | 0.6534 | 0.6655 | 0.582 | 2165 |
| wine_quality@s44 | naive-lightgbm | 0.6750 | 0.6832 | 0.136 | 400 |
| wine_quality@s44 | naive-ensemble | 0.6701 | 0.6770 | 0.549 | 1317 |
| wine_quality@s44 | moe | 0.6670 | 0.6808 | 0.511 | 1379 |



---

## synthetic@s42  (search X=[1600, 5], holdout n=400)


### naive-lightgbm

- **holdout RMSE: 4.40130** (winner retrained in 0.10s, cv score of winner: 5.3096)
- cv best RMSE: 5.3096, median: 5.5469, p10: 5.4101
- train: median 0.040s/fold, mean 0.040s, p90 0.058s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.676 |
| `learning_rate` | 0.243 |
| `bagging_fraction` | 0.033 |
| `feature_fraction` | 0.032 |
| `num_leaves` | 0.008 |
| `extra_trees` | 0.006 |
| `max_depth` | 0.001 |
| `bagging_freq` | 0.001 |
| `lambda_l1` | 0.001 |
| `lambda_l2` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 5.7104 | 0.4933 | 5.3096 |
| True | 32 | 6.8388 | 1.3862 | 5.6091 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 6.0027 | 5.6405 | 5.6945 | 5.7926 | **Q2** [0.0461, 0.0568] |
| `num_leaves` | 5.6610 | 5.6656 | 5.9632 | 5.8424 | **Q1** [None, 30.0] |
| `max_depth` | 6.1194 | 5.6517 | — | 5.7335 | **Q2** [9.0, 10.0] |
| `min_data_in_leaf` | 5.5496 | 5.5381 | 5.6075 | 6.4120 | **Q2** [7.0, 8.0] |
| `lambda_l1` | 5.9018 | 5.8799 | 5.6872 | 5.6616 | **Q4** [0.9322, ∞) |
| `lambda_l2` | 5.6751 | 5.8891 | 5.8415 | 5.7247 | **Q1** [None, 0.0] |
| `feature_fraction` | 6.1383 | 5.7015 | 5.6382 | 5.6524 | **Q3** [0.9547, 0.9788] |
| `bagging_fraction` | 5.7691 | 5.7066 | 5.6783 | 5.9765 | **Q3** [0.6881, 0.7753] |

#### E. Slice plot

![synthetic@s42/naive-lightgbm](parts/slice_synthetic@s42_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 4.97808** (winner retrained in 0.14s, cv score of winner: 5.3159)
- cv best RMSE: 5.3159, median: 5.6074, p10: 5.4192
- train: median 0.116s/fold, mean 0.121s, p90 0.163s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.672 |
| `learning_rate` | 0.263 |
| `bagging_fraction` | 0.024 |
| `bagging_freq` | 0.013 |
| `feature_fraction` | 0.012 |
| `num_leaves` | 0.007 |
| `extra_trees` | 0.005 |
| `max_depth` | 0.003 |
| `lambda_l1` | 0.001 |
| `n_models` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 437 | 5.8014 | 0.6251 | 5.3159 |
| False | 63 | 6.0527 | 0.5984 | 5.3736 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 6.1770 | 5.6431 | 5.6511 | 5.8612 | **Q2** [0.1028, 0.118] |
| `num_leaves` | 5.7659 | 5.6967 | 5.8270 | 6.0304 | **Q2** [27.0, 32.0] |
| `max_depth` | 6.1574 | 5.8847 | 5.6745 | 5.9240 | **Q3** [9.0, 10.0] |
| `min_data_in_leaf` | — | 5.5233 | 5.6605 | 6.4879 | **Q2** [5.0, 7.0] |
| `lambda_l1` | 5.9329 | 5.9424 | 5.7070 | 5.7500 | **Q3** [0.0083, 0.0226] |
| `lambda_l2` | 5.8499 | 5.6781 | 5.7458 | 6.0584 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 6.0968 | 5.8607 | 5.7405 | 5.6343 | **Q4** [0.9811, ∞) |
| `bagging_fraction` | 6.0281 | 5.6927 | 5.6123 | 5.9992 | **Q3** [0.7759, 0.7967] |

#### E. Slice plot

![synthetic@s42/naive-ensemble](parts/slice_synthetic@s42_naive-ensemble.png)


### moe

- **holdout RMSE: 4.04501** (winner retrained in 2.46s, cv score of winner: 5.1551)
- cv best RMSE: 5.1551, median: 5.4318, p10: 5.2654
- train: median 0.610s/fold, mean 0.628s, p90 0.999s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.587 |
| `mixture_e_step_mode` | 0.080 |
| `min_data_in_leaf` | 0.073 |
| `num_leaves` | 0.040 |
| `mixture_gate_type` | 0.031 |
| `bagging_freq` | 0.028 |
| `mixture_expert_dropout_rate` | 0.028 |
| `mixture_warmup_iters` | 0.027 |
| `mixture_diversity_lambda` | 0.022 |
| `feature_fraction` | 0.017 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **leaf_reuse** | 5.6828 (n=396) | gbdt | Δ +0.5112 | p=0.00e+00 |
| `mixture_routing_mode` | **token_choice** | 5.7161 (n=420) | expert_choice | Δ +0.7651 | p=0.00e+00 |
| `mixture_e_step_mode` | **gate_only** | 5.6869 (n=432) | loss_only | Δ +0.9726 | p=3.40e-05 |
| `mixture_r_smoothing` | **none** | 5.6992 (n=408) | ema | Δ +0.6574 | p=0.00e+00 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 396 | 5.6828 | 0.7577 | 5.1551 |
| gbdt | 61 | 6.1940 | 0.6106 | 5.4358 |
| none | 43 | 6.7687 | 1.1121 | 5.8592 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 420 | 5.7161 | 0.7720 | 5.1551 |
| expert_choice | 80 | 6.4812 | 0.9227 | 5.5815 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 432 | 5.6869 | 0.6868 | 5.1551 |
| loss_only | 27 | 6.6595 | 0.9884 | 5.6529 |
| em | 41 | 6.8954 | 1.1443 | 5.7788 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| tree_hierarchical | 112 | 5.7096 | 0.6183 | 5.2595 |
| gmm | 285 | 5.7427 | 0.9092 | 5.1551 |
| kmeans_features | 17 | 6.0431 | 0.7365 | 5.3689 |
| gmm_features | 17 | 6.1230 | 0.6898 | 5.3181 |
| uniform | 26 | 6.3137 | 0.5639 | 5.5385 |
| random | 43 | 6.3287 | 0.8497 | 5.3100 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 408 | 5.6992 | 0.7659 | 5.1551 |
| ema | 64 | 6.3566 | 0.7588 | 5.3489 |
| markov | 28 | 6.6846 | 1.1414 | 5.3100 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 5.8019 | 0.8065 | 5.1551 |
| True | 32 | 6.3745 | 1.1662 | 5.3448 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 466 | 5.7651 | 0.7272 | 5.1551 |
| True | 34 | 6.8447 | 1.4792 | 5.4729 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 6.5232 | — | — | 5.6841 | **Q4** [6.0, ∞) |
| `mixture_diversity_lambda` | 5.9855 | 5.7495 | 5.7029 | 5.9163 | **Q3** [0.3627, 0.3825] |
| `mixture_warmup_iters` | 5.7536 | 5.6613 | 5.7436 | 6.1796 | **Q2** [7.0, 9.0] |
| `mixture_balance_factor` | 5.8679 | 5.9208 | 5.7344 | 5.8740 | **Q3** [6.0, 7.0] |
| `learning_rate` | 6.1494 | 5.8048 | 5.6794 | 5.7206 | **Q3** [0.1579, 0.1837] |
| `num_leaves` | 6.2913 | 5.6819 | 5.6548 | 5.7273 | **Q3** [111.0, 116.0] |
| `max_depth` | 6.2721 | 5.7269 | 5.7668 | 5.7103 | **Q4** [11.0, ∞) |
| `min_data_in_leaf` | — | 5.5755 | 5.6694 | 6.3741 | **Q2** [5.0, 7.0] |

#### E. Slice plot

![synthetic@s42/moe](parts/slice_synthetic@s42_moe.png)


---

## synthetic@s43  (search X=[1600, 5], holdout n=400)


### naive-lightgbm

- **holdout RMSE: 5.09057** (winner retrained in 0.07s, cv score of winner: 5.5192)
- cv best RMSE: 5.5192, median: 5.8184, p10: 5.6691
- train: median 0.040s/fold, mean 0.042s, p90 0.062s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.724 |
| `learning_rate` | 0.158 |
| `bagging_fraction` | 0.047 |
| `feature_fraction` | 0.036 |
| `bagging_freq` | 0.015 |
| `max_depth` | 0.010 |
| `lambda_l2` | 0.005 |
| `extra_trees` | 0.002 |
| `lambda_l1` | 0.002 |
| `num_leaves` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 446 | 5.9559 | 0.4653 | 5.5192 |
| False | 54 | 6.3077 | 0.6867 | 5.7045 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 6.2452 | 5.8588 | 5.9002 | 5.9714 | **Q2** [0.104, 0.12] |
| `num_leaves` | 6.1871 | 5.9433 | 5.9227 | 5.9358 | **Q3** [118.0, 123.0] |
| `max_depth` | 6.3585 | 5.9511 | — | 5.9551 | **Q2** [8.0, 9.0] |
| `min_data_in_leaf` | — | 5.7701 | 5.8555 | 6.4802 | **Q2** [5.0, 7.0] |
| `lambda_l1` | 6.0938 | 5.9732 | 5.9434 | 5.9653 | **Q3** [0.0017, 0.0067] |
| `lambda_l2` | 6.0009 | 5.9181 | 5.8718 | 6.1849 | **Q3** [0.0, 0.0] |
| `feature_fraction` | 6.2526 | 5.9369 | 5.8896 | 5.8966 | **Q3** [0.9539, 0.9731] |
| `bagging_fraction` | 6.0995 | 5.9145 | 5.9622 | 5.9995 | **Q2** [0.8768, 0.9024] |

#### E. Slice plot

![synthetic@s43/naive-lightgbm](parts/slice_synthetic@s43_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 4.26202** (winner retrained in 0.33s, cv score of winner: 5.4566)
- cv best RMSE: 5.4566, median: 5.7179, p10: 5.5596
- train: median 0.260s/fold, mean 0.244s, p90 0.323s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.810 |
| `feature_fraction` | 0.059 |
| `bagging_fraction` | 0.044 |
| `max_depth` | 0.036 |
| `learning_rate` | 0.019 |
| `bagging_freq` | 0.015 |
| `num_leaves` | 0.010 |
| `extra_trees` | 0.005 |
| `n_models` | 0.001 |
| `lambda_l2` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 5.8785 | 0.4542 | 5.4566 |
| True | 32 | 6.5740 | 0.9689 | 5.6650 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 6.0646 | 5.8554 | 5.8343 | 5.9378 | **Q3** [0.0442, 0.0551] |
| `num_leaves` | 5.9562 | 5.7972 | 5.8913 | 6.0475 | **Q2** [46.0, 53.0] |
| `max_depth` | 6.5456 | 5.9124 | 5.8932 | 5.8209 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | — | 5.7131 | 5.7993 | 6.3985 | **Q2** [5.0, 7.0] |
| `lambda_l1` | 5.9598 | 5.8026 | 5.9442 | 5.9855 | **Q2** [0.0014, 0.0047] |
| `lambda_l2` | 5.9276 | 5.8550 | 5.8819 | 6.0276 | **Q2** [0.0, 0.0002] |
| `feature_fraction` | 6.2281 | 5.7835 | 5.7656 | 5.9150 | **Q3** [0.9281, 0.9523] |
| `bagging_fraction` | 6.0150 | 5.8623 | 5.8120 | 6.0029 | **Q3** [0.8396, 0.8803] |

#### E. Slice plot

![synthetic@s43/naive-ensemble](parts/slice_synthetic@s43_naive-ensemble.png)


### moe

- **holdout RMSE: 3.76251** (winner retrained in 0.37s, cv score of winner: 4.1840)
- cv best RMSE: 4.1840, median: 4.7100, p10: 4.3729
- train: median 0.171s/fold, mean 0.353s, p90 1.306s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_gate_type` | 0.753 |
| `mixture_init` | 0.071 |
| `learning_rate` | 0.036 |
| `min_data_in_leaf` | 0.021 |
| `feature_fraction` | 0.021 |
| `mixture_e_step_mode` | 0.019 |
| `mixture_balance_factor` | 0.018 |
| `mixture_warmup_iters` | 0.012 |
| `num_leaves` | 0.009 |
| `mixture_hard_m_step` | 0.008 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **gbdt** | 5.0670 (n=449) | leaf_reuse | Δ +2.9519 | p=0.00e+00 |
| `mixture_routing_mode` | **token_choice** | 5.2066 (n=348) | expert_choice | Δ +0.8162 | p=0.00e+00 |
| `mixture_init` | **gmm** | 5.1291 (n=298) | tree_hierarchical | Δ +0.4719 | p=3.68e-03 |
| `mixture_r_smoothing` | **none** | 5.2137 (n=304) | markov | Δ +0.4419 | p=1.53e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 449 | 5.0670 | 0.8645 | 4.1840 |
| leaf_reuse | 27 | 8.0189 | 1.2282 | 5.8231 |
| none | 24 | 9.8238 | 1.4500 | 6.3409 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 348 | 5.2066 | 1.4015 | 4.1840 |
| expert_choice | 152 | 6.0228 | 1.5727 | 4.4635 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 266 | 5.3008 | 1.4934 | 4.2535 |
| em | 170 | 5.3529 | 1.5087 | 4.1840 |
| gate_only | 64 | 6.3652 | 1.1769 | 5.0206 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gmm | 298 | 5.1291 | 1.4649 | 4.1840 |
| tree_hierarchical | 124 | 5.6010 | 1.5133 | 4.4607 |
| uniform | 18 | 6.2989 | 1.0223 | 5.5203 |
| kmeans_features | 16 | 6.3581 | 1.0933 | 5.2180 |
| random | 27 | 6.5715 | 0.9518 | 5.5542 |
| gmm_features | 17 | 6.5786 | 1.3007 | 5.2055 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 304 | 5.2137 | 1.4633 | 4.1840 |
| markov | 169 | 5.6556 | 1.4244 | 4.4453 |
| ema | 27 | 6.9117 | 1.4183 | 4.8721 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 296 | 5.3833 | 1.4515 | 4.1840 |
| True | 204 | 5.5584 | 1.5697 | 4.3306 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 453 | 5.3412 | 1.4800 | 4.1840 |
| False | 47 | 6.5493 | 1.2678 | 4.9661 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | — | — | 5.4548 | **Q4** [2.0, ∞) |
| `mixture_diversity_lambda` | 6.1257 | 5.3205 | 5.0850 | 5.2878 | **Q3** [0.34, 0.4317] |
| `mixture_warmup_iters` | 5.8474 | 5.4173 | 5.1511 | 5.3800 | **Q3** [38.0, 43.0] |
| `mixture_balance_factor` | 6.3317 | 5.4131 | — | 5.3498 | **Q4** [5.0, ∞) |
| `learning_rate` | 6.2403 | 5.1635 | 5.0156 | 5.3997 | **Q3** [0.1493, 0.1846] |
| `num_leaves` | 5.2537 | 5.3301 | 5.4136 | 5.8153 | **Q1** [None, 31.0] |
| `max_depth` | 5.7934 | 5.1873 | 5.2273 | 5.6399 | **Q2** [8.0, 9.0] |
| `min_data_in_leaf` | 5.2215 | 5.0889 | 5.3082 | 6.2274 | **Q2** [7.0, 11.0] |

#### E. Slice plot

![synthetic@s43/moe](parts/slice_synthetic@s43_moe.png)


---

## synthetic@s44  (search X=[1600, 5], holdout n=400)


### naive-lightgbm

- **holdout RMSE: 5.02751** (winner retrained in 0.06s, cv score of winner: 5.6149)
- cv best RMSE: 5.6149, median: 6.0510, p10: 5.7619
- train: median 0.043s/fold, mean 0.043s, p90 0.065s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.822 |
| `feature_fraction` | 0.047 |
| `bagging_fraction` | 0.034 |
| `extra_trees` | 0.029 |
| `learning_rate` | 0.026 |
| `num_leaves` | 0.021 |
| `max_depth` | 0.010 |
| `bagging_freq` | 0.007 |
| `lambda_l2` | 0.003 |
| `lambda_l1` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 470 | 6.1813 | 0.5132 | 5.6149 |
| True | 30 | 7.0463 | 0.8441 | 6.0670 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 6.3728 | 6.0274 | 6.1293 | 6.4032 | **Q2** [0.0365, 0.0421] |
| `num_leaves` | 6.2477 | 6.2705 | 6.1595 | 6.2592 | **Q3** [84.0, 109.0] |
| `max_depth` | 6.3497 | 6.0993 | 6.2599 | 6.2055 | **Q2** [9.0, 10.0] |
| `min_data_in_leaf` | 5.9990 | 6.0231 | 6.1054 | 6.7147 | **Q1** [None, 6.0] |
| `lambda_l1` | 6.6018 | 6.0186 | 6.1360 | 6.1764 | **Q2** [0.0588, 0.3447] |
| `lambda_l2` | 6.2222 | 6.5009 | 6.0797 | 6.1300 | **Q3** [0.0021, 0.2991] |
| `feature_fraction` | 6.6824 | 6.0070 | 6.1069 | 6.1366 | **Q2** [0.9027, 0.9382] |
| `bagging_fraction` | 6.3217 | 6.3795 | 6.2683 | 5.9634 | **Q4** [0.9424, ∞) |

#### E. Slice plot

![synthetic@s44/naive-lightgbm](parts/slice_synthetic@s44_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 4.92985** (winner retrained in 0.86s, cv score of winner: 5.5024)
- cv best RMSE: 5.5024, median: 5.7684, p10: 5.5967
- train: median 0.254s/fold, mean 0.277s, p90 0.455s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.645 |
| `learning_rate` | 0.195 |
| `feature_fraction` | 0.094 |
| `bagging_fraction` | 0.020 |
| `n_models` | 0.014 |
| `num_leaves` | 0.011 |
| `max_depth` | 0.010 |
| `extra_trees` | 0.009 |
| `bagging_freq` | 0.003 |
| `lambda_l2` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 413 | 5.8941 | 0.4930 | 5.5024 |
| True | 87 | 6.4892 | 0.6728 | 5.8148 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 6.0287 | 5.9279 | 5.9387 | 6.0953 | **Q2** [0.041, 0.0703] |
| `num_leaves` | 6.3999 | 5.9567 | 5.8209 | 5.8324 | **Q3** [105.0, 117.0] |
| `max_depth` | 6.5329 | 5.8996 | 5.9648 | 5.9191 | **Q2** [8.0, 9.0] |
| `min_data_in_leaf` | — | 5.7539 | 5.8150 | 6.5983 | **Q2** [5.0, 7.0] |
| `lambda_l1` | 6.1981 | 5.9757 | 5.9435 | 5.8734 | **Q4** [0.7269, ∞) |
| `lambda_l2` | 5.9886 | 5.8508 | 5.9147 | 6.2366 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 6.4315 | 5.7970 | 5.8457 | 5.9164 | **Q2** [0.9078, 0.9304] |
| `bagging_fraction` | 5.9992 | 6.0241 | 5.9536 | 6.0137 | **Q3** [0.8284, 0.8945] |

#### E. Slice plot

![synthetic@s44/naive-ensemble](parts/slice_synthetic@s44_naive-ensemble.png)


### moe

- **holdout RMSE: 3.50192** (winner retrained in 0.29s, cv score of winner: 3.4794)
- cv best RMSE: 3.4794, median: 4.5422, p10: 3.8170
- train: median 0.152s/fold, mean 0.189s, p90 0.252s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_gate_type` | 0.663 |
| `mixture_init` | 0.122 |
| `min_data_in_leaf` | 0.083 |
| `learning_rate` | 0.033 |
| `mixture_load_balance_alpha` | 0.019 |
| `bagging_freq` | 0.015 |
| `mixture_hard_m_step` | 0.014 |
| `mixture_diversity_lambda` | 0.009 |
| `mixture_refit_leaves` | 0.006 |
| `mixture_num_experts` | 0.005 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **gbdt** | 4.7813 (n=446) | leaf_reuse | Δ +3.1646 | p=0.00e+00 |
| `mixture_routing_mode` | **expert_choice** | 5.1032 (n=440) | token_choice | Δ +1.3851 | p=1.00e-06 |
| `mixture_init` | **gmm** | 4.8980 (n=382) | tree_hierarchical | Δ +1.0053 | p=5.50e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 446 | 4.7813 | 1.1261 | 3.4794 |
| leaf_reuse | 25 | 7.9459 | 0.9733 | 6.2785 |
| none | 29 | 10.4686 | 1.9231 | 6.3888 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 440 | 5.1032 | 1.8204 | 3.4794 |
| token_choice | 60 | 6.4883 | 1.8545 | 3.7297 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 395 | 5.1272 | 1.8332 | 3.4794 |
| em | 75 | 5.7494 | 1.9463 | 3.6513 |
| loss_only | 30 | 5.9425 | 1.9613 | 3.8284 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gmm | 382 | 4.8980 | 1.8636 | 3.4794 |
| tree_hierarchical | 33 | 5.9033 | 1.8534 | 3.8861 |
| uniform | 18 | 6.2322 | 0.6776 | 5.3635 |
| gmm_features | 26 | 6.6005 | 0.8692 | 5.0048 |
| random | 19 | 6.9310 | 1.2975 | 5.9999 |
| kmeans_features | 22 | 6.9711 | 0.9770 | 5.8698 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 275 | 5.0865 | 1.8714 | 3.4794 |
| none | 198 | 5.2929 | 1.7858 | 3.7299 |
| ema | 27 | 6.9605 | 1.7727 | 4.4371 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 453 | 5.1143 | 1.8279 | 3.4794 |
| False | 47 | 6.7641 | 1.7036 | 3.8285 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 465 | 5.1714 | 1.8512 | 3.4794 |
| False | 35 | 6.5719 | 1.7595 | 4.6399 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | — | — | 5.2694 | **Q4** [2.0, ∞) |
| `mixture_diversity_lambda` | 5.6785 | 5.1341 | 5.0695 | 5.1956 | **Q3** [0.418, 0.4508] |
| `mixture_warmup_iters` | 5.3595 | 4.9130 | 5.2638 | 5.5932 | **Q2** [18.0, 21.0] |
| `mixture_balance_factor` | — | — | 5.2439 | 5.3385 | **Q3** [2.0, 8.0] |
| `learning_rate` | 6.0872 | 5.0731 | 4.9886 | 4.9288 | **Q4** [0.2714, ∞) |
| `num_leaves` | 5.6011 | 5.1631 | 4.9360 | 5.4027 | **Q3** [72.0, 99.25] |
| `max_depth` | — | — | 5.1964 | 5.4837 | **Q3** [3.0, 8.0] |
| `min_data_in_leaf` | 4.8798 | 4.9743 | 4.9689 | 6.1826 | **Q1** [None, 7.0] |

#### E. Slice plot

![synthetic@s44/moe](parts/slice_synthetic@s44_moe.png)


---

## fred_gdp@s42  (search X=[249, 12], holdout n=62)


### naive-lightgbm

- **holdout RMSE: 1.53914** (winner retrained in 0.02s, cv score of winner: 0.8073)
- cv best RMSE: 0.8073, median: 0.8248, p10: 0.8129
- train: median 0.012s/fold, mean 0.013s, p90 0.020s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.883 |
| `bagging_fraction` | 0.035 |
| `feature_fraction` | 0.032 |
| `learning_rate` | 0.017 |
| `num_leaves` | 0.011 |
| `bagging_freq` | 0.009 |
| `extra_trees` | 0.008 |
| `lambda_l2` | 0.003 |
| `max_depth` | 0.002 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 449 | 0.8253 | 0.0137 | 0.8073 |
| True | 51 | 0.8426 | 0.0152 | 0.8218 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.8257 | 0.8244 | 0.8231 | 0.8351 | **Q3** [0.0914, 0.1089] |
| `num_leaves` | 0.8245 | 0.8247 | 0.8256 | 0.8335 | **Q1** [None, 22.0] |
| `max_depth` | 0.8365 | 0.8249 | — | 0.8257 | **Q2** [7.0, 8.0] |
| `min_data_in_leaf` | 0.8362 | 0.8211 | 0.8193 | 0.8311 | **Q3** [25.0, 29.0] |
| `lambda_l1` | 0.8371 | 0.8238 | 0.8228 | 0.8246 | **Q3** [0.0003, 0.0016] |
| `lambda_l2` | 0.8355 | 0.8241 | 0.8237 | 0.8250 | **Q3** [1.1508, 2.7358] |
| `feature_fraction` | 0.8339 | 0.8255 | 0.8241 | 0.8248 | **Q3** [0.9051, 0.9318] |
| `bagging_fraction` | 0.8336 | 0.8238 | 0.8226 | 0.8282 | **Q3** [0.7548, 0.7816] |

#### E. Slice plot

![fred_gdp@s42/naive-lightgbm](parts/slice_fred_gdp@s42_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 1.51783** (winner retrained in 0.12s, cv score of winner: 0.8098)
- cv best RMSE: 0.8098, median: 0.8243, p10: 0.8154
- train: median 0.155s/fold, mean 0.150s, p90 0.175s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.960 |
| `bagging_fraction` | 0.017 |
| `learning_rate` | 0.007 |
| `feature_fraction` | 0.006 |
| `num_leaves` | 0.004 |
| `extra_trees` | 0.002 |
| `bagging_freq` | 0.001 |
| `n_models` | 0.001 |
| `max_depth` | 0.001 |
| `lambda_l2` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 469 | 0.8255 | 0.0105 | 0.8098 |
| True | 31 | 0.8365 | 0.0133 | 0.8205 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.8309 | 0.8260 | 0.8247 | 0.8230 | **Q4** [0.2597, ∞) |
| `num_leaves` | 0.8280 | 0.8233 | 0.8231 | 0.8305 | **Q3** [70.0, 78.0] |
| `max_depth` | 0.8307 | 0.8289 | 0.8240 | 0.8239 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 0.8321 | 0.8214 | 0.8213 | 0.8314 | **Q3** [24.0, 28.0] |
| `lambda_l1` | 0.8313 | 0.8263 | 0.8230 | 0.8241 | **Q3** [0.0472, 0.2917] |
| `lambda_l2` | 0.8271 | 0.8253 | 0.8238 | 0.8285 | **Q3** [0.0, 0.0] |
| `feature_fraction` | 0.8260 | 0.8229 | 0.8289 | 0.8268 | **Q2** [0.6964, 0.7264] |
| `bagging_fraction` | 0.8330 | 0.8260 | 0.8241 | 0.8215 | **Q4** [0.9851, ∞) |

#### E. Slice plot

![fred_gdp@s42/naive-ensemble](parts/slice_fred_gdp@s42_naive-ensemble.png)


### moe

- **holdout RMSE: 1.50117** (winner retrained in 0.10s, cv score of winner: 0.8196)
- cv best RMSE: 0.8196, median: 0.8625, p10: 0.8370
- train: median 0.187s/fold, mean 0.203s, p90 0.359s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.978 |
| `mixture_init` | 0.005 |
| `feature_fraction` | 0.003 |
| `lambda_l1` | 0.002 |
| `max_depth` | 0.002 |
| `mixture_diversity_lambda` | 0.001 |
| `mixture_e_step_mode` | 0.001 |
| `mixture_expert_dropout_rate` | 0.001 |
| `bagging_freq` | 0.001 |
| `mixture_load_balance_alpha` | 0.001 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **none** | 0.8856 (n=380) | leaf_reuse | Δ +0.0442 | p=5.00e-06 |
| `mixture_e_step_mode` | **gate_only** | 0.8854 (n=350) | loss_only | Δ +0.0259 | p=2.42e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 380 | 0.8856 | 0.0667 | 0.8196 |
| leaf_reuse | 86 | 0.9298 | 0.0791 | 0.8340 |
| gbdt | 34 | 0.9347 | 0.0667 | 0.8350 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 390 | 0.8937 | 0.0677 | 0.8294 |
| token_choice | 110 | 0.9068 | 0.0836 | 0.8196 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 350 | 0.8854 | 0.0631 | 0.8196 |
| loss_only | 104 | 0.9113 | 0.0780 | 0.8212 |
| em | 46 | 0.9486 | 0.0873 | 0.8359 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| random | 329 | 0.8834 | 0.0638 | 0.8196 |
| tree_hierarchical | 64 | 0.8910 | 0.0736 | 0.8212 |
| gmm_features | 17 | 0.9293 | 0.0850 | 0.8384 |
| uniform | 44 | 0.9313 | 0.0725 | 0.8515 |
| kmeans_features | 25 | 0.9461 | 0.0646 | 0.8499 |
| gmm | 21 | 0.9612 | 0.0834 | 0.8524 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 348 | 0.8851 | 0.0637 | 0.8196 |
| none | 83 | 0.9063 | 0.0817 | 0.8212 |
| ema | 69 | 0.9427 | 0.0764 | 0.8342 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 462 | 0.8911 | 0.0667 | 0.8196 |
| False | 38 | 0.9625 | 0.0938 | 0.8374 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 459 | 0.8907 | 0.0699 | 0.8196 |
| False | 41 | 0.9620 | 0.0578 | 0.8458 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.9186 | — | — | 0.8908 | **Q4** [5.0, ∞) |
| `mixture_diversity_lambda` | 0.9197 | 0.8795 | 0.8932 | 0.8939 | **Q2** [0.3421, 0.3755] |
| `mixture_warmup_iters` | 0.9077 | 0.9086 | 0.8934 | 0.8803 | **Q4** [49.0, ∞) |
| `mixture_balance_factor` | 0.9144 | — | 0.8781 | 0.9107 | **Q3** [6.0, 7.0] |
| `learning_rate` | 0.9165 | 0.8855 | 0.8766 | 0.9077 | **Q3** [0.0791, 0.0903] |
| `num_leaves` | 0.8999 | 0.8825 | 0.8998 | 0.9039 | **Q2** [62.0, 69.0] |
| `max_depth` | 0.9250 | 0.9021 | — | 0.8860 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 0.8615 | 0.8541 | 0.8725 | 0.9913 | **Q2** [7.0, 9.0] |

#### E. Slice plot

![fred_gdp@s42/moe](parts/slice_fred_gdp@s42_moe.png)


---

## fred_gdp@s43  (search X=[249, 12], holdout n=62)


### naive-lightgbm

- **holdout RMSE: 1.55352** (winner retrained in 0.03s, cv score of winner: 0.7992)
- cv best RMSE: 0.7992, median: 0.8210, p10: 0.8083
- train: median 0.015s/fold, mean 0.016s, p90 0.024s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.903 |
| `bagging_fraction` | 0.061 |
| `bagging_freq` | 0.011 |
| `learning_rate` | 0.008 |
| `num_leaves` | 0.007 |
| `feature_fraction` | 0.007 |
| `max_depth` | 0.001 |
| `extra_trees` | 0.001 |
| `lambda_l1` | 0.001 |
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
| `max_depth` | 0.8344 | 0.8316 | 0.8300 | 0.8229 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 0.8551 | 0.8172 | 0.8191 | 0.8289 | **Q2** [14.0, 18.0] |
| `lambda_l1` | 0.8282 | 0.8229 | 0.8276 | 0.8342 | **Q2** [0.0001, 0.0005] |
| `lambda_l2` | 0.8345 | 0.8257 | 0.8249 | 0.8277 | **Q3** [0.0001, 0.0004] |
| `feature_fraction` | 0.8313 | 0.8236 | 0.8256 | 0.8323 | **Q2** [0.6343, 0.6578] |
| `bagging_fraction` | 0.8373 | 0.8242 | 0.8225 | 0.8289 | **Q3** [0.8706, 0.8901] |

#### E. Slice plot

![fred_gdp@s43/naive-lightgbm](parts/slice_fred_gdp@s43_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 1.54157** (winner retrained in 0.11s, cv score of winner: 0.8044)
- cv best RMSE: 0.8044, median: 0.8218, p10: 0.8117
- train: median 0.083s/fold, mean 0.088s, p90 0.109s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.969 |
| `learning_rate` | 0.013 |
| `bagging_fraction` | 0.008 |
| `num_leaves` | 0.002 |
| `feature_fraction` | 0.002 |
| `extra_trees` | 0.002 |
| `max_depth` | 0.002 |
| `bagging_freq` | 0.001 |
| `n_models` | 0.000 |
| `lambda_l2` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 454 | 0.8236 | 0.0124 | 0.8044 |
| True | 46 | 0.8348 | 0.0133 | 0.8095 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.8345 | 0.8236 | 0.8208 | 0.8197 | **Q4** [0.2793, ∞) |
| `num_leaves` | 0.8313 | 0.8234 | 0.8211 | 0.8231 | **Q3** [79.0, 88.0] |
| `max_depth` | — | — | 0.8218 | 0.8311 | **Q3** [3.0, 4.0] |
| `min_data_in_leaf` | 0.8315 | 0.8201 | 0.8186 | 0.8300 | **Q3** [24.0, 28.0] |
| `lambda_l1` | 0.8323 | 0.8237 | 0.8199 | 0.8228 | **Q3** [0.1003, 0.2288] |
| `lambda_l2` | 0.8262 | 0.8224 | 0.8205 | 0.8295 | **Q3** [0.0, 0.0] |
| `feature_fraction` | 0.8303 | 0.8207 | 0.8210 | 0.8267 | **Q2** [0.8615, 0.8826] |
| `bagging_fraction` | 0.8319 | 0.8215 | 0.8221 | 0.8231 | **Q2** [0.9007, 0.9421] |

#### E. Slice plot

![fred_gdp@s43/naive-ensemble](parts/slice_fred_gdp@s43_naive-ensemble.png)


### moe

- **holdout RMSE: 1.50774** (winner retrained in 1.01s, cv score of winner: 0.8206)
- cv best RMSE: 0.8206, median: 0.8563, p10: 0.8321
- train: median 0.066s/fold, mean 0.140s, p90 0.390s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.913 |
| `extra_trees` | 0.020 |
| `lambda_l1` | 0.019 |
| `mixture_warmup_iters` | 0.008 |
| `mixture_load_balance_alpha` | 0.008 |
| `mixture_init` | 0.006 |
| `mixture_balance_factor` | 0.004 |
| `mixture_e_step_mode` | 0.004 |
| `mixture_expert_dropout_rate` | 0.003 |
| `feature_fraction` | 0.003 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **none** | 0.8801 (n=363) | gbdt | Δ +0.0242 | p=5.00e-03 |
| `mixture_routing_mode` | **expert_choice** | 0.8810 (n=418) | token_choice | Δ +0.0550 | p=0.00e+00 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 363 | 0.8801 | 0.0645 | 0.8228 |
| gbdt | 114 | 0.9043 | 0.0830 | 0.8206 |
| leaf_reuse | 23 | 0.9745 | 0.0939 | 0.8570 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 418 | 0.8810 | 0.0683 | 0.8206 |
| token_choice | 82 | 0.9360 | 0.0829 | 0.8317 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 274 | 0.8838 | 0.0698 | 0.8228 |
| em | 162 | 0.8859 | 0.0730 | 0.8206 |
| gate_only | 64 | 0.9267 | 0.0816 | 0.8413 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gmm_features | 290 | 0.8776 | 0.0653 | 0.8228 |
| kmeans_features | 81 | 0.8846 | 0.0563 | 0.8302 |
| gmm | 42 | 0.9040 | 0.0923 | 0.8206 |
| uniform | 17 | 0.9096 | 0.0599 | 0.8526 |
| random | 53 | 0.9281 | 0.0879 | 0.8377 |
| tree_hierarchical | 17 | 0.9538 | 0.1058 | 0.8402 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 360 | 0.8807 | 0.0664 | 0.8228 |
| markov | 75 | 0.9013 | 0.0842 | 0.8206 |
| none | 65 | 0.9283 | 0.0849 | 0.8413 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 423 | 0.8805 | 0.0673 | 0.8206 |
| False | 77 | 0.9420 | 0.0851 | 0.8372 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 469 | 0.8858 | 0.0708 | 0.8206 |
| False | 31 | 0.9535 | 0.0878 | 0.8539 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.9084 | — | — | 0.8855 | **Q4** [3.0, ∞) |
| `mixture_diversity_lambda` | 0.8934 | 0.8874 | 0.9026 | 0.8765 | **Q4** [0.4764, ∞) |
| `mixture_warmup_iters` | 0.9148 | 0.8864 | 0.8845 | 0.8800 | **Q4** [39.0, ∞) |
| `mixture_balance_factor` | 0.8800 | 0.8827 | 0.8788 | 0.9194 | **Q3** [6.0, 7.0] |
| `learning_rate` | 0.9088 | 0.8785 | 0.8753 | 0.8973 | **Q3** [0.0582, 0.075] |
| `num_leaves` | 0.9079 | 0.8708 | 0.8883 | 0.8929 | **Q2** [38.0, 47.0] |
| `max_depth` | 0.9213 | 0.8910 | — | 0.8777 | **Q4** [10.0, ∞) |
| `min_data_in_leaf` | 0.8673 | 0.8437 | 0.8553 | 0.9804 | **Q2** [7.0, 9.0] |

#### E. Slice plot

![fred_gdp@s43/moe](parts/slice_fred_gdp@s43_moe.png)


---

## fred_gdp@s44  (search X=[249, 12], holdout n=62)


### naive-lightgbm

- **holdout RMSE: 1.56652** (winner retrained in 0.03s, cv score of winner: 0.8046)
- cv best RMSE: 0.8046, median: 0.8286, p10: 0.8183
- train: median 0.013s/fold, mean 0.014s, p90 0.019s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.839 |
| `bagging_fraction` | 0.061 |
| `learning_rate` | 0.046 |
| `lambda_l2` | 0.014 |
| `feature_fraction` | 0.009 |
| `extra_trees` | 0.009 |
| `bagging_freq` | 0.008 |
| `max_depth` | 0.007 |
| `num_leaves` | 0.005 |
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
| `learning_rate` | 0.8333 | 0.8307 | 0.8288 | 0.8297 | **Q3** [0.0718, 0.1145] |
| `num_leaves` | 0.8291 | 0.8288 | 0.8320 | 0.8322 | **Q2** [20.0, 29.0] |
| `max_depth` | 0.8292 | 0.8298 | — | 0.8315 | **Q1** [None, 6.0] |
| `min_data_in_leaf` | 0.8362 | 0.8257 | 0.8258 | 0.8357 | **Q2** [14.0, 18.0] |
| `lambda_l1` | 0.8329 | 0.8299 | 0.8288 | 0.8309 | **Q3** [0.0, 0.0] |
| `lambda_l2` | 0.8348 | 0.8312 | 0.8294 | 0.8271 | **Q4** [0.8764, ∞) |
| `feature_fraction` | 0.8371 | 0.8291 | 0.8287 | 0.8276 | **Q4** [0.9381, ∞) |
| `bagging_fraction` | 0.8313 | 0.8281 | 0.8315 | 0.8315 | **Q2** [0.5466, 0.5716] |

#### E. Slice plot

![fred_gdp@s44/naive-lightgbm](parts/slice_fred_gdp@s44_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 1.56039** (winner retrained in 0.06s, cv score of winner: 0.8023)
- cv best RMSE: 0.8023, median: 0.8227, p10: 0.8120
- train: median 0.086s/fold, mean 0.081s, p90 0.107s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.975 |
| `extra_trees` | 0.005 |
| `feature_fraction` | 0.005 |
| `learning_rate` | 0.004 |
| `bagging_fraction` | 0.003 |
| `num_leaves` | 0.003 |
| `max_depth` | 0.003 |
| `bagging_freq` | 0.001 |
| `n_models` | 0.001 |
| `lambda_l2` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 439 | 0.8239 | 0.0136 | 0.8023 |
| True | 61 | 0.8317 | 0.0113 | 0.8094 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.8296 | 0.8254 | 0.8224 | 0.8218 | **Q4** [0.2771, ∞) |
| `num_leaves` | 0.8279 | 0.8242 | 0.8221 | 0.8250 | **Q3** [86.0, 96.0] |
| `max_depth` | 0.8248 | 0.8253 | 0.8247 | 0.8244 | **Q4** [8.0, ∞) |
| `min_data_in_leaf` | 0.8297 | 0.8213 | 0.8188 | 0.8309 | **Q3** [20.0, 24.0] |
| `lambda_l1` | 0.8228 | 0.8219 | 0.8254 | 0.8291 | **Q2** [0.0, 0.0] |
| `lambda_l2` | 0.8267 | 0.8257 | 0.8227 | 0.8242 | **Q3** [0.2033, 1.5595] |
| `feature_fraction` | 0.8307 | 0.8226 | 0.8222 | 0.8236 | **Q3** [0.9135, 0.9372] |
| `bagging_fraction` | 0.8267 | 0.8226 | 0.8210 | 0.8289 | **Q3** [0.6569, 0.6834] |

#### E. Slice plot

![fred_gdp@s44/naive-ensemble](parts/slice_fred_gdp@s44_naive-ensemble.png)


### moe

- **holdout RMSE: 1.50439** (winner retrained in 0.42s, cv score of winner: 0.8231)
- cv best RMSE: 0.8231, median: 0.8675, p10: 0.8356
- train: median 0.040s/fold, mean 0.084s, p90 0.168s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.915 |
| `lambda_l1` | 0.049 |
| `extra_trees` | 0.010 |
| `feature_fraction` | 0.006 |
| `num_leaves` | 0.004 |
| `mixture_diversity_lambda` | 0.003 |
| `learning_rate` | 0.003 |
| `mixture_num_experts` | 0.002 |
| `mixture_init` | 0.001 |
| `mixture_balance_factor` | 0.001 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **none** | 0.8877 (n=288) | gbdt | Δ +0.0206 | p=1.75e-03 |
| `mixture_routing_mode` | **expert_choice** | 0.8882 (n=364) | token_choice | Δ +0.0328 | p=2.00e-06 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 288 | 0.8877 | 0.0672 | 0.8245 |
| gbdt | 177 | 0.9083 | 0.0686 | 0.8231 |
| leaf_reuse | 35 | 0.9174 | 0.0623 | 0.8425 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 364 | 0.8882 | 0.0662 | 0.8231 |
| token_choice | 136 | 0.9210 | 0.0681 | 0.8309 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 271 | 0.8923 | 0.0667 | 0.8245 |
| loss_only | 68 | 0.8984 | 0.0729 | 0.8231 |
| em | 161 | 0.9046 | 0.0682 | 0.8279 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gmm | 280 | 0.8834 | 0.0642 | 0.8231 |
| tree_hierarchical | 39 | 0.9078 | 0.0706 | 0.8471 |
| random | 62 | 0.9084 | 0.0607 | 0.8251 |
| uniform | 77 | 0.9163 | 0.0644 | 0.8309 |
| gmm_features | 17 | 0.9212 | 0.0766 | 0.8342 |
| kmeans_features | 25 | 0.9303 | 0.0904 | 0.8333 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 332 | 0.8871 | 0.0655 | 0.8231 |
| markov | 45 | 0.9165 | 0.0716 | 0.8319 |
| none | 123 | 0.9169 | 0.0685 | 0.8309 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 409 | 0.8935 | 0.0669 | 0.8245 |
| False | 91 | 0.9131 | 0.0719 | 0.8231 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 469 | 0.8934 | 0.0656 | 0.8231 |
| False | 31 | 0.9532 | 0.0822 | 0.8489 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.9387 | — | — | 0.8937 | **Q4** [4.0, ∞) |
| `mixture_diversity_lambda` | 0.8844 | 0.8828 | 0.9026 | 0.9186 | **Q2** [0.0373, 0.0716] |
| `mixture_warmup_iters` | 0.9057 | 0.8827 | 0.8845 | 0.9172 | **Q2** [22.0, 25.0] |
| `mixture_balance_factor` | 0.9228 | 0.9078 | — | 0.8861 | **Q4** [8.0, ∞) |
| `learning_rate` | 0.9093 | 0.8880 | 0.8910 | 0.9001 | **Q2** [0.0523, 0.0683] |
| `num_leaves` | 0.8905 | 0.8875 | 0.8961 | 0.9144 | **Q2** [39.0, 50.0] |
| `max_depth` | 0.9186 | — | — | 0.8904 | **Q4** [9.0, ∞) |
| `min_data_in_leaf` | 0.8562 | 0.8535 | 0.8982 | 0.9741 | **Q2** [8.0, 11.0] |

#### E. Slice plot

![fred_gdp@s44/moe](parts/slice_fred_gdp@s44_moe.png)


---

## sp500_basic@s42  (search X=[3010, 13], holdout n=752)


### naive-lightgbm

- **holdout RMSE: 0.01103** (winner retrained in 0.02s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0097, p10: 0.0097
- train: median 0.022s/fold, mean 0.022s, p90 0.030s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.661 |
| `learning_rate` | 0.155 |
| `extra_trees` | 0.115 |
| `feature_fraction` | 0.029 |
| `num_leaves` | 0.014 |
| `bagging_freq` | 0.009 |
| `bagging_fraction` | 0.007 |
| `lambda_l2` | 0.005 |
| `max_depth` | 0.005 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 469 | 0.0097 | 0.0000 | 0.0097 |
| False | 31 | 0.0097 | 0.0000 | 0.0097 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0558] |
| `num_leaves` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 18.75] |
| `max_depth` | 0.0097 | — | 0.0097 | 0.0097 | **Q1** [None, 4.0] |
| `min_data_in_leaf` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 51.0] |
| `lambda_l1` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0001] |
| `lambda_l2` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0] |
| `feature_fraction` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.7453] |
| `bagging_fraction` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.622] |

#### E. Slice plot

![sp500_basic@s42/naive-lightgbm](parts/slice_sp500_basic@s42_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 0.01104** (winner retrained in 0.07s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0097, p10: 0.0097
- train: median 0.078s/fold, mean 0.084s, p90 0.109s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.330 |
| `min_data_in_leaf` | 0.260 |
| `extra_trees` | 0.256 |
| `num_leaves` | 0.046 |
| `bagging_fraction` | 0.035 |
| `feature_fraction` | 0.032 |
| `bagging_freq` | 0.018 |
| `n_models` | 0.011 |
| `max_depth` | 0.006 |
| `lambda_l2` | 0.003 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 31 | 0.0097 | 0.0000 | 0.0097 |
| True | 469 | 0.0097 | 0.0000 | 0.0097 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0432] |
| `num_leaves` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 27.0] |
| `max_depth` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 4.0] |
| `min_data_in_leaf` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 73.0] |
| `lambda_l1` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0] |
| `lambda_l2` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0024] |
| `feature_fraction` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.6788] |
| `bagging_fraction` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.7604] |

#### E. Slice plot

![sp500_basic@s42/naive-ensemble](parts/slice_sp500_basic@s42_naive-ensemble.png)


### moe

- **holdout RMSE: 0.01103** (winner retrained in 0.12s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0097, p10: 0.0097
- train: median 0.068s/fold, mean 0.088s, p90 0.108s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.191 |
| `mixture_init` | 0.138 |
| `mixture_balance_factor` | 0.079 |
| `learning_rate` | 0.076 |
| `mixture_expert_dropout_rate` | 0.072 |
| `mixture_load_balance_alpha` | 0.065 |
| `num_leaves` | 0.057 |
| `feature_fraction` | 0.051 |
| `mixture_e_step_mode` | 0.051 |
| `extra_trees` | 0.048 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_routing_mode` | **token_choice** | 0.0097 (n=468) | expert_choice | Δ +0.0000 | p=2.35e-03 |
| `mixture_e_step_mode` | **gate_only** | 0.0097 (n=338) | loss_only | Δ +0.0000 | p=9.01e-03 |
| `mixture_r_smoothing` | **ema** | 0.0097 (n=28) | markov | Δ +0.0000 | p=1.44e-04 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 56 | 0.0097 | 0.0000 | 0.0097 |
| none | 31 | 0.0097 | 0.0000 | 0.0097 |
| gbdt | 413 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 468 | 0.0097 | 0.0000 | 0.0097 |
| expert_choice | 32 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 338 | 0.0097 | 0.0000 | 0.0097 |
| loss_only | 130 | 0.0097 | 0.0000 | 0.0097 |
| em | 32 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 31 | 0.0097 | 0.0000 | 0.0097 |
| gmm | 21 | 0.0097 | 0.0000 | 0.0097 |
| random | 40 | 0.0097 | 0.0000 | 0.0097 |
| gmm_features | 331 | 0.0097 | 0.0000 | 0.0097 |
| tree_hierarchical | 60 | 0.0097 | 0.0000 | 0.0097 |
| kmeans_features | 17 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 28 | 0.0097 | 0.0000 | 0.0097 |
| markov | 347 | 0.0097 | 0.0000 | 0.0097 |
| none | 125 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 33 | 0.0097 | 0.0000 | 0.0097 |
| True | 467 | 0.0097 | 0.0000 | 0.0097 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 470 | 0.0097 | 0.0000 | 0.0097 |
| False | 30 | 0.0097 | 0.0000 | 0.0097 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | — | — | 0.0097 | **Q4** [2.0, ∞) |
| `mixture_diversity_lambda` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.3411] |
| `mixture_warmup_iters` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 9.0] |
| `mixture_balance_factor` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 4.0] |
| `learning_rate` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0747] |
| `num_leaves` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 49.0] |
| `max_depth` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 8.0] |
| `min_data_in_leaf` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 54.0] |

#### E. Slice plot

![sp500_basic@s42/moe](parts/slice_sp500_basic@s42_moe.png)


---

## sp500_basic@s43  (search X=[3010, 13], holdout n=752)


### naive-lightgbm

- **holdout RMSE: 0.01101** (winner retrained in 0.01s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0097, p10: 0.0097
- train: median 0.015s/fold, mean 0.016s, p90 0.022s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.422 |
| `min_data_in_leaf` | 0.277 |
| `learning_rate` | 0.220 |
| `num_leaves` | 0.028 |
| `bagging_fraction` | 0.017 |
| `feature_fraction` | 0.013 |
| `max_depth` | 0.010 |
| `bagging_freq` | 0.007 |
| `lambda_l2` | 0.005 |
| `lambda_l1` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 469 | 0.0097 | 0.0000 | 0.0097 |
| False | 31 | 0.0097 | 0.0000 | 0.0097 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0618] |
| `num_leaves` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 52.0] |
| `max_depth` | 0.0097 | 0.0097 | — | 0.0097 | **Q1** [None, 11.0] |
| `min_data_in_leaf` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 60.0] |
| `lambda_l1` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0] |
| `lambda_l2` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0] |
| `feature_fraction` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.548] |
| `bagging_fraction` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.5154] |

#### E. Slice plot

![sp500_basic@s43/naive-lightgbm](parts/slice_sp500_basic@s43_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 0.01105** (winner retrained in 0.09s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0097, p10: 0.0097
- train: median 0.107s/fold, mean 0.108s, p90 0.147s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.432 |
| `min_data_in_leaf` | 0.211 |
| `bagging_fraction` | 0.122 |
| `feature_fraction` | 0.097 |
| `learning_rate` | 0.043 |
| `max_depth` | 0.030 |
| `n_models` | 0.025 |
| `bagging_freq` | 0.018 |
| `num_leaves` | 0.017 |
| `lambda_l2` | 0.004 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 31 | 0.0097 | 0.0000 | 0.0097 |
| True | 469 | 0.0097 | 0.0000 | 0.0097 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.1163] |
| `num_leaves` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 26.0] |
| `max_depth` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 6.0] |
| `min_data_in_leaf` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 53.0] |
| `lambda_l1` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0004] |
| `lambda_l2` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0] |
| `feature_fraction` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.5273] |
| `bagging_fraction` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.5501] |

#### E. Slice plot

![sp500_basic@s43/naive-ensemble](parts/slice_sp500_basic@s43_naive-ensemble.png)


### moe

- **holdout RMSE: 0.01104** (winner retrained in 0.52s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0097, p10: 0.0097
- train: median 0.195s/fold, mean 0.786s, p90 2.573s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_num_experts` | 0.242 |
| `mixture_warmup_iters` | 0.202 |
| `mixture_diversity_lambda` | 0.135 |
| `mixture_init` | 0.115 |
| `learning_rate` | 0.052 |
| `mixture_expert_dropout_rate` | 0.048 |
| `bagging_fraction` | 0.037 |
| `feature_fraction` | 0.029 |
| `max_depth` | 0.025 |
| `mixture_load_balance_alpha` | 0.025 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **gbdt** | 0.0097 (n=424) | leaf_reuse | Δ +0.0000 | p=7.10e-05 |
| `mixture_e_step_mode` | **loss_only** | 0.0097 (n=68) | gate_only | Δ +0.0000 | p=2.01e-04 |
| `mixture_r_smoothing` | **ema** | 0.0097 (n=46) | none | Δ +0.0000 | p=6.46e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 424 | 0.0097 | 0.0000 | 0.0097 |
| leaf_reuse | 51 | 0.0097 | 0.0000 | 0.0097 |
| none | 25 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 320 | 0.0097 | 0.0000 | 0.0097 |
| token_choice | 180 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 68 | 0.0097 | 0.0000 | 0.0097 |
| gate_only | 406 | 0.0097 | 0.0000 | 0.0097 |
| em | 26 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gmm | 71 | 0.0097 | 0.0000 | 0.0097 |
| tree_hierarchical | 61 | 0.0097 | 0.0000 | 0.0097 |
| uniform | 18 | 0.0097 | 0.0000 | 0.0097 |
| kmeans_features | 285 | 0.0097 | 0.0000 | 0.0097 |
| gmm_features | 45 | 0.0097 | 0.0000 | 0.0097 |
| random | 20 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 46 | 0.0097 | 0.0000 | 0.0097 |
| none | 273 | 0.0097 | 0.0000 | 0.0097 |
| markov | 181 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 71 | 0.0097 | 0.0000 | 0.0097 |
| True | 429 | 0.0097 | 0.0000 | 0.0097 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 31 | 0.0097 | 0.0000 | 0.0097 |
| True | 469 | 0.0097 | 0.0000 | 0.0097 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.0097 | — | — | 0.0097 | **Q1** [None, 5.0] |
| `mixture_diversity_lambda` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0561] |
| `mixture_warmup_iters` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 33.75] |
| `mixture_balance_factor` | 0.0097 | 0.0097 | — | 0.0097 | **Q1** [None, 6.0] |
| `learning_rate` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0389] |
| `num_leaves` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 52.0] |
| `max_depth` | — | 0.0097 | 0.0097 | 0.0097 | **Q2** [3.0, 6.0] |
| `min_data_in_leaf` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 60.0] |

#### E. Slice plot

![sp500_basic@s43/moe](parts/slice_sp500_basic@s43_moe.png)


---

## sp500_basic@s44  (search X=[3010, 13], holdout n=752)


### naive-lightgbm

- **holdout RMSE: 0.01102** (winner retrained in 0.04s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0097, p10: 0.0097
- train: median 0.017s/fold, mean 0.018s, p90 0.023s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.345 |
| `min_data_in_leaf` | 0.308 |
| `feature_fraction` | 0.155 |
| `extra_trees` | 0.065 |
| `bagging_fraction` | 0.054 |
| `num_leaves` | 0.034 |
| `max_depth` | 0.020 |
| `bagging_freq` | 0.018 |
| `lambda_l2` | 0.001 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 46 | 0.0097 | 0.0000 | 0.0097 |
| True | 454 | 0.0097 | 0.0000 | 0.0097 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0148] |
| `num_leaves` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 49.0] |
| `max_depth` | — | — | 0.0097 | 0.0097 | **Q3** [3.0, 4.0] |
| `min_data_in_leaf` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 91.0] |
| `lambda_l1` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0] |
| `lambda_l2` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0] |
| `feature_fraction` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.5183] |
| `bagging_fraction` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.8992] |

#### E. Slice plot

![sp500_basic@s44/naive-lightgbm](parts/slice_sp500_basic@s44_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 0.01101** (winner retrained in 0.14s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0097, p10: 0.0097
- train: median 0.074s/fold, mean 0.082s, p90 0.112s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.669 |
| `extra_trees` | 0.079 |
| `num_leaves` | 0.070 |
| `learning_rate` | 0.056 |
| `bagging_fraction` | 0.046 |
| `lambda_l2` | 0.035 |
| `feature_fraction` | 0.031 |
| `bagging_freq` | 0.006 |
| `max_depth` | 0.005 |
| `n_models` | 0.002 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 468 | 0.0097 | 0.0000 | 0.0097 |
| False | 32 | 0.0097 | 0.0000 | 0.0097 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0635] |
| `num_leaves` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 66.0] |
| `max_depth` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 10.0] |
| `min_data_in_leaf` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 63.0] |
| `lambda_l1` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0] |
| `lambda_l2` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0006] |
| `feature_fraction` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.5149] |
| `bagging_fraction` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.5988] |

#### E. Slice plot

![sp500_basic@s44/naive-ensemble](parts/slice_sp500_basic@s44_naive-ensemble.png)


### moe

- **holdout RMSE: 0.01103** (winner retrained in 0.15s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0097, p10: 0.0097
- train: median 0.063s/fold, mean 0.086s, p90 0.146s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_refit_leaves` | 0.229 |
| `min_data_in_leaf` | 0.106 |
| `mixture_init` | 0.105 |
| `extra_trees` | 0.095 |
| `bagging_fraction` | 0.091 |
| `mixture_load_balance_alpha` | 0.072 |
| `learning_rate` | 0.051 |
| `mixture_diversity_lambda` | 0.041 |
| `mixture_warmup_iters` | 0.027 |
| `num_leaves` | 0.026 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_routing_mode` | **token_choice** | 0.0097 (n=470) | expert_choice | Δ +0.0000 | p=8.50e-05 |
| `mixture_e_step_mode` | **em** | 0.0097 (n=40) | gate_only | Δ +0.0000 | p=2.34e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 33 | 0.0097 | 0.0000 | 0.0097 |
| none | 442 | 0.0097 | 0.0000 | 0.0097 |
| leaf_reuse | 25 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 470 | 0.0097 | 0.0000 | 0.0097 |
| expert_choice | 30 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| em | 40 | 0.0097 | 0.0000 | 0.0097 |
| gate_only | 411 | 0.0097 | 0.0000 | 0.0097 |
| loss_only | 49 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| kmeans_features | 20 | 0.0097 | 0.0000 | 0.0097 |
| tree_hierarchical | 37 | 0.0097 | 0.0000 | 0.0097 |
| random | 42 | 0.0097 | 0.0000 | 0.0097 |
| gmm | 33 | 0.0097 | 0.0000 | 0.0097 |
| gmm_features | 350 | 0.0097 | 0.0000 | 0.0097 |
| uniform | 18 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 52 | 0.0097 | 0.0000 | 0.0097 |
| ema | 130 | 0.0097 | 0.0000 | 0.0097 |
| markov | 318 | 0.0097 | 0.0000 | 0.0097 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 32 | 0.0097 | 0.0000 | 0.0097 |
| True | 468 | 0.0097 | 0.0000 | 0.0097 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 469 | 0.0097 | 0.0000 | 0.0097 |
| False | 31 | 0.0097 | 0.0000 | 0.0097 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | — | — | 0.0097 | **Q4** [2.0, ∞) |
| `mixture_diversity_lambda` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.1119] |
| `mixture_warmup_iters` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 25.0] |
| `mixture_balance_factor` | 0.0097 | 0.0097 | — | 0.0097 | **Q1** [None, 9.0] |
| `learning_rate` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 0.0535] |
| `num_leaves` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 27.0] |
| `max_depth` | 0.0097 | — | 0.0097 | 0.0097 | **Q1** [None, 7.0] |
| `min_data_in_leaf` | 0.0097 | 0.0097 | 0.0097 | 0.0097 | **Q1** [None, 66.0] |

#### E. Slice plot

![sp500_basic@s44/moe](parts/slice_sp500_basic@s44_moe.png)


---

## sp500@s42  (search X=[2969, 28], holdout n=742)


### naive-lightgbm

- **holdout RMSE: 0.01112** (winner retrained in 0.04s, cv score of winner: 0.0098)
- cv best RMSE: 0.0098, median: 0.0098, p10: 0.0098
- train: median 0.027s/fold, mean 0.028s, p90 0.040s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `num_leaves` | 0.395 |
| `learning_rate` | 0.346 |
| `min_data_in_leaf` | 0.147 |
| `extra_trees` | 0.045 |
| `bagging_fraction` | 0.031 |
| `feature_fraction` | 0.022 |
| `max_depth` | 0.007 |
| `bagging_freq` | 0.006 |
| `lambda_l2` | 0.000 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 470 | 0.0098 | 0.0000 | 0.0098 |
| False | 30 | 0.0098 | 0.0000 | 0.0098 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0523] |
| `num_leaves` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 19.0] |
| `max_depth` | 0.0098 | 0.0098 | — | 0.0098 | **Q1** [None, 6.0] |
| `min_data_in_leaf` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 81.0] |
| `lambda_l1` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0] |
| `lambda_l2` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0] |
| `feature_fraction` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.6683] |
| `bagging_fraction` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.6068] |

#### E. Slice plot

![sp500@s42/naive-lightgbm](parts/slice_sp500@s42_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 0.01105** (winner retrained in 0.29s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0098, p10: 0.0098
- train: median 0.288s/fold, mean 0.294s, p90 0.400s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.787 |
| `feature_fraction` | 0.051 |
| `min_data_in_leaf` | 0.043 |
| `lambda_l2` | 0.034 |
| `bagging_fraction` | 0.024 |
| `max_depth` | 0.017 |
| `num_leaves` | 0.016 |
| `bagging_freq` | 0.013 |
| `n_models` | 0.011 |
| `lambda_l1` | 0.003 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 469 | 0.0098 | 0.0000 | 0.0097 |
| True | 31 | 0.0098 | 0.0000 | 0.0098 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0109] |
| `num_leaves` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 79.0] |
| `max_depth` | 0.0098 | — | 0.0098 | 0.0098 | **Q1** [None, 8.0] |
| `min_data_in_leaf` | — | 0.0098 | 0.0098 | 0.0098 | **Q2** [5.0, 8.0] |
| `lambda_l1` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0045] |
| `lambda_l2` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0] |
| `feature_fraction` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.7035] |
| `bagging_fraction` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.9476] |

#### E. Slice plot

![sp500@s42/naive-ensemble](parts/slice_sp500@s42_naive-ensemble.png)


### moe

- **holdout RMSE: 0.01103** (winner retrained in 0.61s, cv score of winner: 0.0098)
- cv best RMSE: 0.0098, median: 0.0098, p10: 0.0098
- train: median 0.144s/fold, mean 0.162s, p90 0.230s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.292 |
| `mixture_init` | 0.268 |
| `learning_rate` | 0.156 |
| `bagging_freq` | 0.063 |
| `num_leaves` | 0.032 |
| `bagging_fraction` | 0.027 |
| `mixture_num_experts` | 0.022 |
| `mixture_hard_m_step` | 0.019 |
| `mixture_expert_dropout_rate` | 0.018 |
| `mixture_load_balance_alpha` | 0.018 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_routing_mode` | **token_choice** | 0.0098 (n=357) | expert_choice | Δ +0.0000 | p=0.00e+00 |
| `mixture_e_step_mode` | **gate_only** | 0.0098 (n=414) | loss_only | Δ +0.0000 | p=1.00e-06 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 50 | 0.0098 | 0.0000 | 0.0098 |
| none | 25 | 0.0098 | 0.0000 | 0.0098 |
| gbdt | 425 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 357 | 0.0098 | 0.0000 | 0.0098 |
| expert_choice | 143 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 414 | 0.0098 | 0.0000 | 0.0098 |
| loss_only | 60 | 0.0098 | 0.0000 | 0.0098 |
| em | 26 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 26 | 0.0098 | 0.0000 | 0.0098 |
| gmm | 29 | 0.0098 | 0.0000 | 0.0098 |
| random | 21 | 0.0098 | 0.0000 | 0.0098 |
| gmm_features | 382 | 0.0098 | 0.0000 | 0.0098 |
| kmeans_features | 24 | 0.0098 | 0.0000 | 0.0098 |
| tree_hierarchical | 18 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 24 | 0.0098 | 0.0000 | 0.0098 |
| markov | 132 | 0.0098 | 0.0000 | 0.0098 |
| none | 344 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 40 | 0.0098 | 0.0000 | 0.0098 |
| True | 460 | 0.0098 | 0.0000 | 0.0098 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 32 | 0.0098 | 0.0000 | 0.0098 |
| False | 468 | 0.0098 | 0.0000 | 0.0098 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | — | 0.0098 | 0.0098 | **Q3** [2.0, 3.0] |
| `mixture_diversity_lambda` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.2481] |
| `mixture_warmup_iters` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 37.0] |
| `mixture_balance_factor` | 0.0098 | 0.0098 | — | 0.0098 | **Q1** [None, 9.0] |
| `learning_rate` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0113] |
| `num_leaves` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 42.0] |
| `max_depth` | 0.0098 | 0.0098 | — | 0.0098 | **Q1** [None, 8.0] |
| `min_data_in_leaf` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 44.0] |

#### E. Slice plot

![sp500@s42/moe](parts/slice_sp500@s42_moe.png)


---

## sp500@s43  (search X=[2969, 28], holdout n=742)


### naive-lightgbm

- **holdout RMSE: 0.01105** (winner retrained in 0.02s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0098, p10: 0.0098
- train: median 0.031s/fold, mean 0.033s, p90 0.047s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.388 |
| `bagging_fraction` | 0.308 |
| `max_depth` | 0.091 |
| `min_data_in_leaf` | 0.079 |
| `bagging_freq` | 0.056 |
| `feature_fraction` | 0.041 |
| `num_leaves` | 0.019 |
| `lambda_l2` | 0.015 |
| `extra_trees` | 0.002 |
| `lambda_l1` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 31 | 0.0098 | 0.0000 | 0.0098 |
| False | 469 | 0.0098 | 0.0000 | 0.0097 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.1715] |
| `num_leaves` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 13.0] |
| `max_depth` | — | — | 0.0098 | 0.0098 | **Q3** [3.0, 4.0] |
| `min_data_in_leaf` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 38.75] |
| `lambda_l1` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0002] |
| `lambda_l2` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0] |
| `feature_fraction` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.9436] |
| `bagging_fraction` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.8923] |

#### E. Slice plot

![sp500@s43/naive-lightgbm](parts/slice_sp500@s43_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 0.01103** (winner retrained in 0.20s, cv score of winner: 0.0098)
- cv best RMSE: 0.0098, median: 0.0098, p10: 0.0098
- train: median 0.147s/fold, mean 0.155s, p90 0.202s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.506 |
| `learning_rate` | 0.196 |
| `feature_fraction` | 0.101 |
| `bagging_freq` | 0.044 |
| `num_leaves` | 0.043 |
| `bagging_fraction` | 0.041 |
| `max_depth` | 0.039 |
| `extra_trees` | 0.019 |
| `n_models` | 0.011 |
| `lambda_l2` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 30 | 0.0098 | 0.0000 | 0.0098 |
| True | 470 | 0.0098 | 0.0000 | 0.0098 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.069] |
| `num_leaves` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 56.0] |
| `max_depth` | 0.0098 | — | 0.0098 | 0.0098 | **Q1** [None, 4.0] |
| `min_data_in_leaf` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 92.0] |
| `lambda_l1` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0] |
| `lambda_l2` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0] |
| `feature_fraction` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.5246] |
| `bagging_fraction` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.6773] |

#### E. Slice plot

![sp500@s43/naive-ensemble](parts/slice_sp500@s43_naive-ensemble.png)


### moe

- **holdout RMSE: 0.01106** (winner retrained in 0.17s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0098, p10: 0.0098
- train: median 0.250s/fold, mean 0.854s, p90 1.411s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `max_depth` | 0.143 |
| `learning_rate` | 0.142 |
| `mixture_expert_dropout_rate` | 0.131 |
| `bagging_fraction` | 0.110 |
| `mixture_load_balance_alpha` | 0.085 |
| `mixture_diversity_lambda` | 0.074 |
| `mixture_init` | 0.069 |
| `mixture_refit_leaves` | 0.049 |
| `mixture_hard_m_step` | 0.033 |
| `feature_fraction` | 0.029 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **gbdt** | 0.0098 (n=112) | leaf_reuse | Δ +0.0000 | p=0.00e+00 |
| `mixture_routing_mode` | **expert_choice** | 0.0098 (n=356) | token_choice | Δ +0.0000 | p=0.00e+00 |
| `mixture_e_step_mode` | **loss_only** | 0.0098 (n=163) | gate_only | Δ +0.0000 | p=7.80e-05 |
| `mixture_init` | **gmm** | 0.0098 (n=46) | tree_hierarchical | Δ +0.0000 | p=7.00e-05 |
| `mixture_r_smoothing` | **ema** | 0.0098 (n=165) | none | Δ +0.0000 | p=9.00e-06 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 112 | 0.0098 | 0.0000 | 0.0098 |
| leaf_reuse | 346 | 0.0098 | 0.0000 | 0.0097 |
| none | 42 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 356 | 0.0098 | 0.0000 | 0.0097 |
| token_choice | 144 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 163 | 0.0098 | 0.0000 | 0.0098 |
| gate_only | 65 | 0.0098 | 0.0000 | 0.0098 |
| em | 272 | 0.0098 | 0.0000 | 0.0097 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gmm | 46 | 0.0098 | 0.0000 | 0.0098 |
| tree_hierarchical | 78 | 0.0098 | 0.0000 | 0.0098 |
| uniform | 20 | 0.0098 | 0.0000 | 0.0098 |
| kmeans_features | 47 | 0.0098 | 0.0000 | 0.0098 |
| gmm_features | 278 | 0.0098 | 0.0000 | 0.0097 |
| random | 31 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 165 | 0.0098 | 0.0000 | 0.0098 |
| none | 65 | 0.0098 | 0.0000 | 0.0098 |
| markov | 270 | 0.0098 | 0.0000 | 0.0097 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 368 | 0.0098 | 0.0000 | 0.0097 |
| True | 132 | 0.0098 | 0.0000 | 0.0098 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 429 | 0.0098 | 0.0000 | 0.0097 |
| True | 71 | 0.0098 | 0.0000 | 0.0098 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.0098 | — | 0.0098 | 0.0098 | **Q1** [None, 3.0] |
| `mixture_diversity_lambda` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.3355] |
| `mixture_warmup_iters` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 19.0] |
| `mixture_balance_factor` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 5.0] |
| `learning_rate` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.014] |
| `num_leaves` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 63.0] |
| `max_depth` | 0.0098 | — | 0.0098 | 0.0098 | **Q1** [None, 4.0] |
| `min_data_in_leaf` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 7.0] |

#### E. Slice plot

![sp500@s43/moe](parts/slice_sp500@s43_moe.png)


---

## sp500@s44  (search X=[2969, 28], holdout n=742)


### naive-lightgbm

- **holdout RMSE: 0.01110** (winner retrained in 0.06s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0098, p10: 0.0097
- train: median 0.021s/fold, mean 0.022s, p90 0.028s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.630 |
| `min_data_in_leaf` | 0.178 |
| `num_leaves` | 0.094 |
| `bagging_fraction` | 0.059 |
| `max_depth` | 0.022 |
| `feature_fraction` | 0.011 |
| `bagging_freq` | 0.003 |
| `extra_trees` | 0.002 |
| `lambda_l2` | 0.001 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 470 | 0.0098 | 0.0000 | 0.0097 |
| True | 30 | 0.0098 | 0.0000 | 0.0098 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0187] |
| `num_leaves` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 37.0] |
| `max_depth` | 0.0098 | — | 0.0098 | 0.0098 | **Q1** [None, 4.0] |
| `min_data_in_leaf` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 7.0] |
| `lambda_l1` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0001] |
| `lambda_l2` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0] |
| `feature_fraction` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.6546] |
| `bagging_fraction` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.5488] |

#### E. Slice plot

![sp500@s44/naive-lightgbm](parts/slice_sp500@s44_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 0.01105** (winner retrained in 0.21s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0098, p10: 0.0097
- train: median 0.133s/fold, mean 0.146s, p90 0.228s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.413 |
| `min_data_in_leaf` | 0.156 |
| `bagging_fraction` | 0.130 |
| `bagging_freq` | 0.106 |
| `lambda_l2` | 0.080 |
| `num_leaves` | 0.044 |
| `max_depth` | 0.034 |
| `feature_fraction` | 0.025 |
| `extra_trees` | 0.006 |
| `n_models` | 0.006 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 31 | 0.0098 | 0.0000 | 0.0098 |
| False | 469 | 0.0098 | 0.0000 | 0.0097 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0194] |
| `num_leaves` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 54.0] |
| `max_depth` | 0.0098 | — | 0.0098 | 0.0098 | **Q1** [None, 7.0] |
| `min_data_in_leaf` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 7.0] |
| `lambda_l1` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0] |
| `lambda_l2` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 1.4199] |
| `feature_fraction` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.7416] |
| `bagging_fraction` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.6501] |

#### E. Slice plot

![sp500@s44/naive-ensemble](parts/slice_sp500@s44_naive-ensemble.png)


### moe

- **holdout RMSE: 0.01106** (winner retrained in 0.13s, cv score of winner: 0.0097)
- cv best RMSE: 0.0097, median: 0.0098, p10: 0.0098
- train: median 0.204s/fold, mean 0.307s, p90 0.456s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_init` | 0.341 |
| `learning_rate` | 0.213 |
| `mixture_load_balance_alpha` | 0.065 |
| `mixture_expert_dropout_rate` | 0.056 |
| `bagging_fraction` | 0.053 |
| `num_leaves` | 0.045 |
| `mixture_warmup_iters` | 0.041 |
| `min_data_in_leaf` | 0.033 |
| `mixture_hard_m_step` | 0.033 |
| `feature_fraction` | 0.028 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **gbdt** | 0.0098 (n=57) | none | Δ +0.0000 | p=7.10e-05 |
| `mixture_e_step_mode` | **em** | 0.0098 (n=221) | gate_only | Δ +0.0000 | p=2.20e-05 |
| `mixture_r_smoothing` | **none** | 0.0098 (n=143) | ema | Δ +0.0000 | p=2.20e-05 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 57 | 0.0098 | 0.0000 | 0.0098 |
| none | 280 | 0.0098 | 0.0000 | 0.0097 |
| leaf_reuse | 163 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 279 | 0.0098 | 0.0000 | 0.0098 |
| expert_choice | 221 | 0.0098 | 0.0000 | 0.0097 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| em | 221 | 0.0098 | 0.0000 | 0.0097 |
| gate_only | 80 | 0.0098 | 0.0000 | 0.0098 |
| loss_only | 199 | 0.0098 | 0.0000 | 0.0098 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| kmeans_features | 21 | 0.0098 | 0.0000 | 0.0098 |
| tree_hierarchical | 29 | 0.0098 | 0.0000 | 0.0098 |
| random | 21 | 0.0098 | 0.0000 | 0.0098 |
| gmm | 21 | 0.0098 | 0.0000 | 0.0098 |
| gmm_features | 18 | 0.0098 | 0.0000 | 0.0098 |
| uniform | 390 | 0.0098 | 0.0000 | 0.0097 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 143 | 0.0098 | 0.0000 | 0.0097 |
| ema | 42 | 0.0098 | 0.0000 | 0.0098 |
| markov | 315 | 0.0098 | 0.0000 | 0.0097 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 467 | 0.0098 | 0.0000 | 0.0097 |
| True | 33 | 0.0098 | 0.0000 | 0.0098 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 33 | 0.0098 | 0.0000 | 0.0098 |
| False | 467 | 0.0098 | 0.0000 | 0.0097 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | 0.0098 | — | 0.0098 | **Q2** [2.0, 3.0] |
| `mixture_diversity_lambda` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.2175] |
| `mixture_warmup_iters` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 32.0] |
| `mixture_balance_factor` | 0.0098 | 0.0098 | — | 0.0098 | **Q1** [None, 4.0] |
| `learning_rate` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 0.0126] |
| `num_leaves` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 22.0] |
| `max_depth` | 0.0098 | 0.0098 | — | 0.0098 | **Q1** [None, 9.0] |
| `min_data_in_leaf` | 0.0098 | 0.0098 | 0.0098 | 0.0098 | **Q1** [None, 7.0] |

#### E. Slice plot

![sp500@s44/moe](parts/slice_sp500@s44_moe.png)


---

## vix@s42  (search X=[3011, 13], holdout n=752)


### naive-lightgbm

- **holdout RMSE: 1.67713** (winner retrained in 0.03s, cv score of winner: 2.6235)
- cv best RMSE: 2.6235, median: 2.7139, p10: 2.6728
- train: median 0.031s/fold, mean 0.033s, p90 0.045s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.613 |
| `min_data_in_leaf` | 0.222 |
| `extra_trees` | 0.133 |
| `feature_fraction` | 0.009 |
| `bagging_fraction` | 0.008 |
| `num_leaves` | 0.007 |
| `max_depth` | 0.005 |
| `bagging_freq` | 0.003 |
| `lambda_l2` | 0.000 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 2.7351 | 0.1102 | 2.6235 |
| True | 32 | 3.1497 | 0.4330 | 2.7210 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.8230 | 2.7158 | 2.7369 | 2.7708 | **Q2** [0.0357, 0.0412] |
| `num_leaves` | 2.7549 | 2.7697 | 2.7813 | 2.7432 | **Q4** [122.0, ∞) |
| `max_depth` | — | 2.7358 | 2.7560 | 2.8102 | **Q2** [3.0, 5.0] |
| `min_data_in_leaf` | 2.7283 | 2.7534 | 2.7232 | 2.8354 | **Q3** [25.0, 30.0] |
| `lambda_l1` | 2.7318 | 2.7470 | 2.7421 | 2.8255 | **Q1** [None, 0.0] |
| `lambda_l2` | 2.7783 | 2.7428 | 2.7554 | 2.7700 | **Q2** [0.0, 0.002] |
| `feature_fraction` | 2.8182 | 2.7281 | 2.7358 | 2.7642 | **Q2** [0.7079, 0.7368] |
| `bagging_fraction` | 2.7370 | 2.7555 | 2.7556 | 2.7984 | **Q1** [None, 0.6106] |

#### E. Slice plot

![vix@s42/naive-lightgbm](parts/slice_vix@s42_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 1.69842** (winner retrained in 0.18s, cv score of winner: 2.6508)
- cv best RMSE: 2.6508, median: 2.7183, p10: 2.6885
- train: median 0.113s/fold, mean 0.122s, p90 0.164s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.577 |
| `min_data_in_leaf` | 0.327 |
| `bagging_fraction` | 0.031 |
| `extra_trees` | 0.021 |
| `feature_fraction` | 0.013 |
| `max_depth` | 0.011 |
| `num_leaves` | 0.009 |
| `bagging_freq` | 0.008 |
| `n_models` | 0.002 |
| `lambda_l2` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 469 | 2.7479 | 0.1438 | 2.6508 |
| True | 31 | 2.9925 | 0.3982 | 2.7016 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.8636 | 2.7229 | 2.7255 | 2.7404 | **Q2** [0.0868, 0.1006] |
| `num_leaves` | 2.8205 | 2.7400 | 2.7437 | 2.7505 | **Q2** [80.0, 89.0] |
| `max_depth` | — | — | 2.7368 | 2.8333 | **Q3** [3.0, 4.0] |
| `min_data_in_leaf` | 2.7352 | 2.7226 | 2.7294 | 2.8555 | **Q2** [25.0, 27.0] |
| `lambda_l1` | 2.7623 | 2.7261 | 2.7874 | 2.7765 | **Q2** [0.0, 0.0] |
| `lambda_l2` | 2.7617 | 2.7361 | 2.7734 | 2.7811 | **Q2** [0.0, 0.0001] |
| `feature_fraction` | 2.7847 | 2.7610 | 2.7279 | 2.7786 | **Q3** [0.7743, 0.8111] |
| `bagging_fraction` | 2.8401 | 2.7338 | 2.7358 | 2.7426 | **Q2** [0.9019, 0.9562] |

#### E. Slice plot

![vix@s42/naive-ensemble](parts/slice_vix@s42_naive-ensemble.png)


### moe

- **holdout RMSE: 1.58697** (winner retrained in 0.29s, cv score of winner: 2.3993)
- cv best RMSE: 2.3993, median: 2.5454, p10: 2.4550
- train: median 0.181s/fold, mean 0.305s, p90 0.572s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l2` | 0.571 |
| `learning_rate` | 0.203 |
| `mixture_init` | 0.036 |
| `bagging_fraction` | 0.034 |
| `mixture_hard_m_step` | 0.020 |
| `mixture_expert_dropout_rate` | 0.016 |
| `num_leaves` | 0.014 |
| `bagging_freq` | 0.013 |
| `lambda_l1` | 0.012 |
| `mixture_e_step_mode` | 0.011 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_routing_mode` | **expert_choice** | 2.7466 (n=412) | token_choice | Δ +0.5847 | p=1.04e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 372 | 2.7556 | 1.0352 | 2.3993 |
| leaf_reuse | 71 | 2.9845 | 1.1613 | 2.4690 |
| none | 57 | 3.2944 | 1.1025 | 2.4522 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 412 | 2.7466 | 0.9093 | 2.3993 |
| token_choice | 88 | 3.3313 | 1.5596 | 2.4303 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 306 | 2.6860 | 0.6059 | 2.3993 |
| loss_only | 137 | 2.9195 | 1.4120 | 2.4260 |
| em | 57 | 3.5590 | 1.6569 | 2.5221 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| random | 291 | 2.6577 | 0.6757 | 2.3993 |
| uniform | 133 | 2.8242 | 1.0357 | 2.4538 |
| tree_hierarchical | 16 | 2.9315 | 0.4469 | 2.5018 |
| gmm | 19 | 3.5660 | 1.4312 | 2.5171 |
| gmm_features | 25 | 3.7565 | 1.4794 | 2.7036 |
| kmeans_features | 16 | 4.1984 | 2.9241 | 2.5212 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 393 | 2.7089 | 0.7291 | 2.3993 |
| none | 48 | 3.2718 | 1.6115 | 2.4303 |
| ema | 59 | 3.4429 | 1.8814 | 2.4505 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 470 | 2.7957 | 0.9652 | 2.3993 |
| True | 30 | 3.6925 | 1.9910 | 2.6817 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 442 | 2.7553 | 0.9411 | 2.3993 |
| True | 58 | 3.5673 | 1.6308 | 2.7540 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 3.6886 | — | — | 2.7286 | **Q4** [6.0, ∞) |
| `mixture_diversity_lambda` | 2.7365 | 2.6088 | 2.7707 | 3.2821 | **Q2** [0.0349, 0.0504] |
| `mixture_warmup_iters` | 2.8185 | 2.6954 | 2.6356 | 3.2686 | **Q3** [9.0, 12.0] |
| `mixture_balance_factor` | 3.1536 | 2.8102 | — | 2.7443 | **Q4** [10.0, ∞) |
| `learning_rate` | 3.4992 | 2.6862 | 2.5888 | 2.6239 | **Q3** [0.2331, 0.2576] |
| `num_leaves` | 2.8692 | 2.6152 | 2.8623 | 3.0551 | **Q2** [44.0, 50.0] |
| `max_depth` | 3.3824 | 2.6895 | — | 2.7630 | **Q2** [9.0, 10.0] |
| `min_data_in_leaf` | 2.8472 | 2.7891 | 2.7274 | 3.0175 | **Q3** [15.0, 19.0] |

#### E. Slice plot

![vix@s42/moe](parts/slice_vix@s42_moe.png)


---

## vix@s43  (search X=[3011, 13], holdout n=752)


### naive-lightgbm

- **holdout RMSE: 1.89318** (winner retrained in 0.04s, cv score of winner: 2.5849)
- cv best RMSE: 2.5849, median: 2.6917, p10: 2.6212
- train: median 0.032s/fold, mean 0.035s, p90 0.049s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.690 |
| `min_data_in_leaf` | 0.278 |
| `bagging_fraction` | 0.013 |
| `feature_fraction` | 0.009 |
| `max_depth` | 0.004 |
| `num_leaves` | 0.003 |
| `lambda_l1` | 0.002 |
| `bagging_freq` | 0.001 |
| `lambda_l2` | 0.001 |
| `extra_trees` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 449 | 2.7273 | 0.1776 | 2.5849 |
| False | 51 | 2.8425 | 0.2538 | 2.6431 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.8429 | 2.7242 | 2.6824 | 2.7067 | **Q3** [0.1774, 0.2016] |
| `num_leaves` | 2.7385 | 2.7214 | 2.7456 | 2.7507 | **Q2** [42.0, 61.0] |
| `max_depth` | — | — | 2.7044 | 2.8420 | **Q3** [3.0, 4.0] |
| `min_data_in_leaf` | 2.7062 | 2.7117 | 2.7122 | 2.8150 | **Q1** [None, 8.0] |
| `lambda_l1` | 2.7896 | 2.7304 | 2.7068 | 2.7295 | **Q3** [0.0416, 0.1672] |
| `lambda_l2` | 2.6942 | 2.7393 | 2.7290 | 2.7938 | **Q1** [None, 0.0] |
| `feature_fraction` | 2.7439 | 2.7254 | 2.7169 | 2.7701 | **Q3** [0.7498, 0.7833] |
| `bagging_fraction` | 2.7425 | 2.7329 | 2.7788 | 2.7021 | **Q4** [0.9348, ∞) |

#### E. Slice plot

![vix@s43/naive-lightgbm](parts/slice_vix@s43_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 1.70837** (winner retrained in 0.15s, cv score of winner: 2.6426)
- cv best RMSE: 2.6426, median: 2.7052, p10: 2.6666
- train: median 0.139s/fold, mean 0.172s, p90 0.283s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.510 |
| `learning_rate` | 0.444 |
| `extra_trees` | 0.014 |
| `num_leaves` | 0.011 |
| `bagging_fraction` | 0.009 |
| `bagging_freq` | 0.007 |
| `feature_fraction` | 0.003 |
| `max_depth` | 0.001 |
| `lambda_l1` | 0.000 |
| `n_models` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 467 | 2.7302 | 0.1234 | 2.6426 |
| True | 33 | 3.0072 | 0.3737 | 2.6878 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.8245 | 2.7048 | 2.7138 | 2.7509 | **Q2** [0.0624, 0.0699] |
| `num_leaves` | 2.7716 | 2.7405 | 2.7202 | 2.7653 | **Q3** [53.0, 64.0] |
| `max_depth` | 2.7492 | — | — | 2.7483 | **Q4** [5.0, ∞) |
| `min_data_in_leaf` | 2.7481 | 2.6963 | 2.7083 | 2.8322 | **Q2** [18.0, 21.0] |
| `lambda_l1` | 2.7320 | 2.7319 | 2.7170 | 2.8131 | **Q3** [0.0, 0.0] |
| `lambda_l2` | 2.7910 | 2.7296 | 2.7289 | 2.7445 | **Q3** [0.0359, 0.0975] |
| `feature_fraction` | 2.7846 | 2.7343 | 2.7328 | 2.7422 | **Q3** [0.8128, 0.852] |
| `bagging_fraction` | 2.7907 | 2.7340 | 2.7161 | 2.7531 | **Q3** [0.7776, 0.7961] |

#### E. Slice plot

![vix@s43/naive-ensemble](parts/slice_vix@s43_naive-ensemble.png)


### moe

- **holdout RMSE: 1.65765** (winner retrained in 0.21s, cv score of winner: 2.3536)
- cv best RMSE: 2.3536, median: 2.5795, p10: 2.3964
- train: median 0.157s/fold, mean 0.802s, p90 0.774s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_gate_type` | 0.376 |
| `learning_rate` | 0.235 |
| `mixture_e_step_mode` | 0.138 |
| `mixture_init` | 0.097 |
| `lambda_l1` | 0.048 |
| `mixture_num_experts` | 0.025 |
| `mixture_load_balance_alpha` | 0.017 |
| `feature_fraction` | 0.014 |
| `mixture_diversity_lambda` | 0.008 |
| `mixture_warmup_iters` | 0.008 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_e_step_mode` | **gate_only** | 2.7600 (n=424) | em | Δ +1.0357 | p=2.16e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 409 | 2.7665 | 0.9767 | 2.3536 |
| gbdt | 67 | 3.2738 | 2.0827 | 2.4806 |
| none | 24 | 4.7772 | 1.6712 | 2.5382 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 468 | 2.8319 | 0.9934 | 2.3536 |
| expert_choice | 32 | 4.3804 | 3.1372 | 2.4472 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 424 | 2.7600 | 0.8610 | 2.3536 |
| em | 48 | 3.7957 | 2.1732 | 2.3760 |
| loss_only | 28 | 4.0382 | 2.8345 | 2.5381 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| random | 401 | 2.7511 | 0.9413 | 2.3536 |
| uniform | 20 | 3.0624 | 0.9737 | 2.5350 |
| tree_hierarchical | 20 | 3.1645 | 0.7995 | 2.5787 |
| kmeans_features | 19 | 3.4037 | 1.6025 | 2.6284 |
| gmm | 20 | 3.9500 | 2.1393 | 2.6301 |
| gmm_features | 20 | 4.7042 | 3.3087 | 2.7299 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 394 | 2.8401 | 1.1373 | 2.3536 |
| ema | 76 | 3.0465 | 1.1549 | 2.4806 |
| markov | 30 | 3.8319 | 2.6395 | 2.4228 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 2.8723 | 1.1555 | 2.3536 |
| True | 32 | 3.7898 | 2.4909 | 2.5598 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 469 | 2.8719 | 1.1293 | 2.3536 |
| True | 31 | 3.8243 | 2.6886 | 2.7441 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 3.4680 | — | — | 2.8893 | **Q4** [5.0, ∞) |
| `mixture_diversity_lambda` | 2.8933 | 2.8643 | 2.6744 | 3.2919 | **Q3** [0.2852, 0.3056] |
| `mixture_warmup_iters` | 2.8462 | 2.8019 | 2.8233 | 3.2596 | **Q2** [23.0, 25.0] |
| `mixture_balance_factor` | 3.2634 | — | — | 2.8383 | **Q4** [8.0, ∞) |
| `learning_rate` | 3.3299 | 2.7591 | 2.8565 | 2.7785 | **Q2** [0.0707, 0.0813] |
| `num_leaves` | 3.1913 | 2.9224 | 2.6751 | 2.9223 | **Q3** [123.0, 126.0] |
| `max_depth` | — | — | 2.7905 | 3.2816 | **Q3** [3.0, 4.0] |
| `min_data_in_leaf` | 2.7615 | 2.7364 | 2.8463 | 3.3179 | **Q2** [8.0, 10.0] |

#### E. Slice plot

![vix@s43/moe](parts/slice_vix@s43_moe.png)


---

## vix@s44  (search X=[3011, 13], holdout n=752)


### naive-lightgbm

- **holdout RMSE: 1.73436** (winner retrained in 0.08s, cv score of winner: 2.6491)
- cv best RMSE: 2.6491, median: 2.7244, p10: 2.6847
- train: median 0.050s/fold, mean 0.053s, p90 0.076s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.504 |
| `learning_rate` | 0.367 |
| `num_leaves` | 0.040 |
| `extra_trees` | 0.033 |
| `bagging_fraction` | 0.020 |
| `feature_fraction` | 0.013 |
| `max_depth` | 0.011 |
| `bagging_freq` | 0.009 |
| `lambda_l2` | 0.002 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 470 | 2.7555 | 0.1315 | 2.6491 |
| True | 30 | 2.9885 | 0.2639 | 2.7179 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.8129 | 2.7511 | 2.7380 | 2.7760 | **Q3** [0.0489, 0.0709] |
| `num_leaves` | 2.7794 | 2.7623 | 2.7505 | 2.7881 | **Q3** [68.0, 73.25] |
| `max_depth` | 2.7848 | 2.7764 | — | 2.7647 | **Q4** [5.0, ∞) |
| `min_data_in_leaf` | 2.7802 | 2.7276 | 2.7353 | 2.8236 | **Q2** [20.0, 24.0] |
| `lambda_l1` | 2.7709 | 2.7526 | 2.7812 | 2.7733 | **Q2** [0.0, 0.0001] |
| `lambda_l2` | 2.7519 | 2.7682 | 2.7522 | 2.8058 | **Q1** [None, 0.0] |
| `feature_fraction` | 2.7956 | 2.7418 | 2.7627 | 2.7779 | **Q2** [0.7376, 0.766] |
| `bagging_fraction` | 2.8315 | 2.7451 | 2.7405 | 2.7611 | **Q3** [0.8748, 0.8987] |

#### E. Slice plot

![vix@s44/naive-lightgbm](parts/slice_vix@s44_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 1.81150** (winner retrained in 0.12s, cv score of winner: 2.6331)
- cv best RMSE: 2.6331, median: 2.7077, p10: 2.6688
- train: median 0.188s/fold, mean 0.191s, p90 0.248s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.825 |
| `min_data_in_leaf` | 0.136 |
| `bagging_fraction` | 0.010 |
| `num_leaves` | 0.009 |
| `extra_trees` | 0.008 |
| `feature_fraction` | 0.004 |
| `bagging_freq` | 0.004 |
| `lambda_l1` | 0.001 |
| `max_depth` | 0.001 |
| `n_models` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 468 | 2.7414 | 0.1586 | 2.6331 |
| False | 32 | 2.9614 | 0.2773 | 2.7407 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.8571 | 2.7134 | 2.7307 | 2.7208 | **Q2** [0.1588, 0.1929] |
| `num_leaves` | 2.7925 | 2.7255 | 2.7544 | 2.7501 | **Q2** [62.0, 69.0] |
| `max_depth` | 2.8207 | 2.7326 | — | 2.7434 | **Q2** [10.0, 11.0] |
| `min_data_in_leaf` | 2.7576 | 2.7098 | 2.7413 | 2.8087 | **Q2** [10.0, 14.0] |
| `lambda_l1` | 2.8110 | 2.7343 | 2.7366 | 2.7401 | **Q2** [0.8139, 1.8887] |
| `lambda_l2` | 2.7285 | 2.7435 | 2.7323 | 2.8178 | **Q1** [None, 0.0] |
| `feature_fraction` | 2.8208 | 2.7300 | 2.7250 | 2.7462 | **Q3** [0.9634, 0.9799] |
| `bagging_fraction` | 2.8314 | 2.7256 | 2.7409 | 2.7241 | **Q4** [0.9836, ∞) |

#### E. Slice plot

![vix@s44/naive-ensemble](parts/slice_vix@s44_naive-ensemble.png)


### moe

- **holdout RMSE: 2.08390** (winner retrained in 1.55s, cv score of winner: 2.4820)
- cv best RMSE: 2.4820, median: 2.6453, p10: 2.5620
- train: median 0.881s/fold, mean 0.869s, p90 1.300s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_init` | 0.223 |
| `mixture_gate_type` | 0.197 |
| `bagging_fraction` | 0.135 |
| `mixture_diversity_lambda` | 0.130 |
| `learning_rate` | 0.102 |
| `lambda_l2` | 0.076 |
| `num_leaves` | 0.055 |
| `min_data_in_leaf` | 0.039 |
| `mixture_r_smoothing` | 0.012 |
| `mixture_load_balance_alpha` | 0.009 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_init` | **uniform** | 2.7576 (n=382) | random | Δ +0.6521 | p=6.20e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 42 | 2.9339 | 0.5145 | 2.6368 |
| leaf_reuse | 433 | 3.1062 | 1.2930 | 2.4820 |
| none | 25 | 4.4608 | 2.9443 | 2.6868 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 55 | 3.1008 | 1.3702 | 2.6064 |
| token_choice | 445 | 3.1667 | 1.4173 | 2.4820 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| em | 421 | 3.1328 | 1.3921 | 2.4820 |
| gate_only | 46 | 3.2803 | 1.6652 | 2.6064 |
| loss_only | 33 | 3.3319 | 1.2556 | 2.6069 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 382 | 2.7576 | 0.7235 | 2.4820 |
| random | 19 | 3.4097 | 0.8862 | 2.6368 |
| tree_hierarchical | 18 | 3.9383 | 0.6741 | 2.8698 |
| kmeans_features | 33 | 4.3899 | 2.3712 | 2.6677 |
| gmm | 30 | 4.4674 | 2.2628 | 2.6941 |
| gmm_features | 18 | 6.2090 | 2.2696 | 2.9675 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 90 | 2.9784 | 0.8895 | 2.5257 |
| ema | 59 | 3.1986 | 1.5183 | 2.6031 |
| none | 351 | 3.1993 | 1.4970 | 2.4820 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 3.1499 | 1.4448 | 2.4820 |
| True | 32 | 3.2992 | 0.7882 | 2.8305 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 467 | 3.1448 | 1.4455 | 2.4820 |
| True | 33 | 3.3674 | 0.7808 | 2.8465 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 3.1950 | — | — | 3.1533 | **Q4** [6.0, ∞) |
| `mixture_diversity_lambda` | 3.0741 | 3.0850 | 3.1142 | 3.3647 | **Q1** [None, 0.3774] |
| `mixture_warmup_iters` | 3.2286 | 3.0909 | 3.1662 | 3.1632 | **Q2** [35.0, 38.0] |
| `mixture_balance_factor` | 3.0653 | 3.1476 | — | 3.2023 | **Q1** [None, 8.0] |
| `learning_rate` | 3.4635 | 2.9039 | 3.2076 | 3.0629 | **Q2** [0.0584, 0.0655] |
| `num_leaves` | 3.1530 | 3.0492 | 2.9799 | 3.4385 | **Q3** [99.0, 104.0] |
| `max_depth` | 3.2134 | 3.3719 | 2.9688 | 3.0652 | **Q3** [10.0, 11.0] |
| `min_data_in_leaf` | 3.1524 | 3.0088 | 3.2658 | 3.2294 | **Q2** [22.0, 26.0] |

#### E. Slice plot

![vix@s44/moe](parts/slice_vix@s44_moe.png)


---

## hmm@s42  (search X=[1600, 5], holdout n=400)


### naive-lightgbm

- **holdout RMSE: 1.88834** (winner retrained in 0.04s, cv score of winner: 2.2992)
- cv best RMSE: 2.2992, median: 2.3144, p10: 2.3078
- train: median 0.019s/fold, mean 0.020s, p90 0.027s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.529 |
| `learning_rate` | 0.269 |
| `min_data_in_leaf` | 0.086 |
| `max_depth` | 0.047 |
| `bagging_fraction` | 0.025 |
| `feature_fraction` | 0.020 |
| `num_leaves` | 0.017 |
| `bagging_freq` | 0.004 |
| `lambda_l1` | 0.002 |
| `lambda_l2` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 470 | 2.3166 | 0.0122 | 2.2992 |
| False | 30 | 2.3478 | 0.0201 | 2.3261 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.3224 | 2.3143 | 2.3155 | 2.3215 | **Q2** [0.0617, 0.0706] |
| `num_leaves` | 2.3190 | 2.3188 | 2.3181 | 2.3179 | **Q4** [120.0, ∞) |
| `max_depth` | — | — | 2.3160 | 2.3256 | **Q3** [3.0, 5.0] |
| `min_data_in_leaf` | 2.3202 | 2.3156 | 2.3152 | 2.3225 | **Q3** [30.0, 34.0] |
| `lambda_l1` | 2.3260 | 2.3163 | 2.3157 | 2.3159 | **Q3** [1.6462, 3.6754] |
| `lambda_l2` | 2.3202 | 2.3191 | 2.3168 | 2.3177 | **Q3** [0.0277, 0.2475] |
| `feature_fraction` | 2.3243 | 2.3177 | 2.3141 | 2.3177 | **Q3** [0.9027, 0.9273] |
| `bagging_fraction` | 2.3256 | 2.3181 | 2.3146 | 2.3154 | **Q3** [0.9369, 0.9666] |

#### E. Slice plot

![hmm@s42/naive-lightgbm](parts/slice_hmm@s42_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 1.88558** (winner retrained in 0.18s, cv score of winner: 2.2955)
- cv best RMSE: 2.2955, median: 2.3133, p10: 2.3049
- train: median 0.162s/fold, mean 0.156s, p90 0.198s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.434 |
| `learning_rate` | 0.285 |
| `min_data_in_leaf` | 0.112 |
| `feature_fraction` | 0.067 |
| `max_depth` | 0.036 |
| `bagging_fraction` | 0.027 |
| `num_leaves` | 0.026 |
| `lambda_l1` | 0.009 |
| `bagging_freq` | 0.003 |
| `lambda_l2` | 0.002 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 466 | 2.3151 | 0.0133 | 2.2955 |
| False | 34 | 2.3503 | 0.0160 | 2.3146 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.3271 | 2.3147 | 2.3138 | 2.3143 | **Q3** [0.2563, 0.2789] |
| `num_leaves` | 2.3211 | 2.3142 | 2.3143 | 2.3197 | **Q2** [94.75, 100.0] |
| `max_depth` | 2.3271 | 2.3208 | — | 2.3152 | **Q4** [11.0, ∞) |
| `min_data_in_leaf` | 2.3206 | 2.3162 | 2.3130 | 2.3210 | **Q3** [30.0, 34.0] |
| `lambda_l1` | 2.3128 | 2.3165 | 2.3201 | 2.3205 | **Q1** [None, 0.0] |
| `lambda_l2` | 2.3194 | 2.3195 | 2.3161 | 2.3150 | **Q4** [0.0624, ∞) |
| `feature_fraction` | 2.3227 | 2.3144 | 2.3155 | 2.3173 | **Q2** [0.7238, 0.8281] |
| `bagging_fraction` | 2.3225 | 2.3153 | 2.3139 | 2.3182 | **Q3** [0.836, 0.8524] |

#### E. Slice plot

![hmm@s42/naive-ensemble](parts/slice_hmm@s42_naive-ensemble.png)


### moe

- **holdout RMSE: 1.88597** (winner retrained in 0.13s, cv score of winner: 2.3100)
- cv best RMSE: 2.3100, median: 2.3539, p10: 2.3250
- train: median 0.069s/fold, mean 0.189s, p90 0.346s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_init` | 0.357 |
| `learning_rate` | 0.266 |
| `feature_fraction` | 0.062 |
| `min_data_in_leaf` | 0.055 |
| `mixture_expert_dropout_rate` | 0.047 |
| `bagging_fraction` | 0.038 |
| `mixture_load_balance_alpha` | 0.036 |
| `mixture_diversity_lambda` | 0.030 |
| `bagging_freq` | 0.022 |
| `mixture_num_experts` | 0.022 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_routing_mode` | **expert_choice** | 2.3576 (n=403) | token_choice | Δ +0.0164 | p=2.50e-05 |
| `mixture_e_step_mode` | **gate_only** | 2.3535 (n=244) | em | Δ +0.0109 | p=5.12e-04 |
| `mixture_init` | **random** | 2.3513 (n=378) | uniform | Δ +0.0255 | p=5.00e-06 |
| `mixture_r_smoothing` | **none** | 2.3539 (n=287) | markov | Δ +0.0101 | p=4.32e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 431 | 2.3581 | 0.0327 | 2.3100 |
| gbdt | 24 | 2.3729 | 0.0281 | 2.3339 |
| none | 45 | 2.3801 | 0.0348 | 2.3326 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 403 | 2.3576 | 0.0327 | 2.3100 |
| token_choice | 97 | 2.3740 | 0.0333 | 2.3114 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 244 | 2.3535 | 0.0328 | 2.3100 |
| em | 206 | 2.3644 | 0.0326 | 2.3114 |
| loss_only | 50 | 2.3813 | 0.0287 | 2.3386 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| random | 378 | 2.3513 | 0.0277 | 2.3100 |
| uniform | 34 | 2.3768 | 0.0266 | 2.3154 |
| gmm_features | 19 | 2.3771 | 0.0314 | 2.3363 |
| tree_hierarchical | 29 | 2.3894 | 0.0366 | 2.3401 |
| gmm | 22 | 2.4046 | 0.0280 | 2.3577 |
| kmeans_features | 18 | 2.4128 | 0.0195 | 2.3758 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 287 | 2.3539 | 0.0336 | 2.3100 |
| markov | 107 | 2.3640 | 0.0295 | 2.3114 |
| ema | 106 | 2.3761 | 0.0308 | 2.3260 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 463 | 2.3597 | 0.0333 | 2.3100 |
| False | 37 | 2.3751 | 0.0319 | 2.3316 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 443 | 2.3580 | 0.0332 | 2.3100 |
| False | 57 | 2.3824 | 0.0264 | 2.3386 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | — | 2.3555 | 2.3764 | **Q3** [2.0, 5.0] |
| `mixture_diversity_lambda` | 2.3770 | 2.3591 | 2.3549 | 2.3521 | **Q4** [0.4468, ∞) |
| `mixture_warmup_iters` | 2.3736 | 2.3569 | 2.3516 | 2.3621 | **Q3** [26.0, 29.0] |
| `mixture_balance_factor` | 2.3659 | — | 2.3546 | 2.3641 | **Q3** [8.0, 9.0] |
| `learning_rate` | 2.3836 | 2.3556 | 2.3520 | 2.3519 | **Q4** [0.2478, ∞) |
| `num_leaves` | 2.3642 | 2.3528 | 2.3540 | 2.3726 | **Q2** [74.0, 82.0] |
| `max_depth` | 2.3730 | 2.3702 | 2.3557 | 2.3535 | **Q4** [8.0, ∞) |
| `min_data_in_leaf` | 2.3630 | 2.3456 | 2.3543 | 2.3792 | **Q2** [16.0, 19.0] |

#### E. Slice plot

![hmm@s42/moe](parts/slice_hmm@s42_moe.png)


---

## hmm@s43  (search X=[1600, 5], holdout n=400)


### naive-lightgbm

- **holdout RMSE: 2.47543** (winner retrained in 0.05s, cv score of winner: 2.3781)
- cv best RMSE: 2.3781, median: 2.3928, p10: 2.3835
- train: median 0.038s/fold, mean 0.040s, p90 0.058s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.485 |
| `extra_trees` | 0.335 |
| `feature_fraction` | 0.050 |
| `learning_rate` | 0.036 |
| `num_leaves` | 0.029 |
| `lambda_l1` | 0.028 |
| `bagging_fraction` | 0.024 |
| `bagging_freq` | 0.006 |
| `max_depth` | 0.004 |
| `lambda_l2` | 0.003 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 468 | 2.3959 | 0.0146 | 2.3781 |
| False | 32 | 2.4482 | 0.0290 | 2.4035 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.4027 | 2.3943 | 2.3958 | 2.4040 | **Q2** [0.0537, 0.0613] |
| `num_leaves` | 2.3947 | 2.3997 | 2.4003 | 2.4020 | **Q1** [None, 16.0] |
| `max_depth` | 2.4045 | 2.3962 | 2.3971 | 2.4015 | **Q2** [8.0, 8.5] |
| `min_data_in_leaf` | 2.3981 | 2.3936 | 2.3944 | 2.4122 | **Q2** [7.0, 11.0] |
| `lambda_l1` | 2.4082 | 2.3971 | 2.3950 | 2.3965 | **Q3** [2.6671, 4.7197] |
| `lambda_l2` | 2.4061 | 2.3972 | 2.3967 | 2.3968 | **Q3** [0.3291, 1.0941] |
| `feature_fraction` | 2.4091 | 2.3965 | 2.3965 | 2.3947 | **Q4** [0.9791, ∞) |
| `bagging_fraction` | 2.4064 | 2.3977 | 2.3948 | 2.3979 | **Q3** [0.8822, 0.9203] |

#### E. Slice plot

![hmm@s43/naive-lightgbm](parts/slice_hmm@s43_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 2.46832** (winner retrained in 0.16s, cv score of winner: 2.3714)
- cv best RMSE: 2.3714, median: 2.3870, p10: 2.3787
- train: median 0.195s/fold, mean 0.198s, p90 0.248s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.609 |
| `learning_rate` | 0.199 |
| `min_data_in_leaf` | 0.144 |
| `num_leaves` | 0.017 |
| `bagging_fraction` | 0.008 |
| `lambda_l1` | 0.007 |
| `max_depth` | 0.006 |
| `bagging_freq` | 0.004 |
| `feature_fraction` | 0.004 |
| `n_models` | 0.002 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 469 | 2.3904 | 0.0168 | 2.3714 |
| False | 31 | 2.4579 | 0.0271 | 2.4190 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.4056 | 2.3905 | 2.3905 | 2.3918 | **Q2** [0.2092, 0.248] |
| `num_leaves` | 2.3930 | 2.3936 | 2.3916 | 2.3994 | **Q3** [24.0, 29.0] |
| `max_depth` | 2.4069 | 2.3950 | — | 2.3914 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 2.3983 | 2.3886 | 2.3903 | 2.4035 | **Q2** [7.0, 9.0] |
| `lambda_l1` | 2.4100 | 2.3921 | 2.3872 | 2.3891 | **Q3** [3.7629, 5.7976] |
| `lambda_l2` | 2.3949 | 2.3903 | 2.3933 | 2.4000 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 2.4005 | 2.3949 | 2.3915 | 2.3916 | **Q3** [0.9684, 0.9836] |
| `bagging_fraction` | 2.3939 | 2.3973 | 2.3905 | 2.3968 | **Q3** [0.7394, 0.7662] |

#### E. Slice plot

![hmm@s43/naive-ensemble](parts/slice_hmm@s43_naive-ensemble.png)


### moe

- **holdout RMSE: 2.64021** (winner retrained in 0.23s, cv score of winner: 2.3420)
- cv best RMSE: 2.3420, median: 2.3817, p10: 2.3679
- train: median 0.141s/fold, mean 0.232s, p90 0.522s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_init` | 0.420 |
| `learning_rate` | 0.245 |
| `feature_fraction` | 0.163 |
| `min_data_in_leaf` | 0.024 |
| `mixture_gate_type` | 0.016 |
| `bagging_freq` | 0.015 |
| `mixture_expert_dropout_rate` | 0.015 |
| `mixture_diversity_lambda` | 0.015 |
| `mixture_warmup_iters` | 0.013 |
| `mixture_load_balance_alpha` | 0.012 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_init` | **random** | 2.3791 (n=191) | uniform | Δ +0.0088 | p=1.80e-05 |
| `mixture_r_smoothing` | **none** | 2.3873 (n=182) | markov | Δ +0.0110 | p=5.71e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 173 | 2.3894 | 0.0464 | 2.3420 |
| none | 288 | 2.3978 | 0.0422 | 2.3607 |
| gbdt | 39 | 2.4039 | 0.0356 | 2.3564 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 188 | 2.3946 | 0.0513 | 2.3420 |
| token_choice | 312 | 2.3958 | 0.0381 | 2.3564 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 177 | 2.3927 | 0.0486 | 2.3420 |
| em | 303 | 2.3949 | 0.0378 | 2.3564 |
| gate_only | 20 | 2.4249 | 0.0625 | 2.3727 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| random | 191 | 2.3791 | 0.0169 | 2.3420 |
| uniform | 152 | 2.3879 | 0.0197 | 2.3607 |
| gmm_features | 108 | 2.4018 | 0.0505 | 2.3625 |
| kmeans_features | 16 | 2.4218 | 0.0316 | 2.3868 |
| tree_hierarchical | 16 | 2.4889 | 0.0812 | 2.4040 |
| gmm | 17 | 2.4897 | 0.0647 | 2.3961 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 182 | 2.3873 | 0.0411 | 2.3420 |
| markov | 283 | 2.3983 | 0.0421 | 2.3564 |
| ema | 35 | 2.4129 | 0.0570 | 2.3727 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 2.3926 | 0.0413 | 2.3420 |
| True | 32 | 2.4359 | 0.0543 | 2.3728 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 470 | 2.3919 | 0.0384 | 2.3420 |
| True | 30 | 2.4484 | 0.0742 | 2.3776 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 2.4275 | — | — | 2.3910 | **Q4** [6.0, ∞) |
| `mixture_diversity_lambda` | 2.3980 | 2.3914 | 2.3920 | 2.3999 | **Q2** [0.3102, 0.3427] |
| `mixture_warmup_iters` | 2.4067 | 2.3907 | 2.3981 | 2.3882 | **Q4** [48.0, ∞) |
| `mixture_balance_factor` | 2.4088 | — | 2.3941 | 2.3897 | **Q4** [9.0, ∞) |
| `learning_rate` | 2.4009 | 2.3868 | 2.3863 | 2.4073 | **Q3** [0.0579, 0.0755] |
| `num_leaves` | 2.3902 | 2.3875 | 2.3969 | 2.4069 | **Q2** [12.0, 18.0] |
| `max_depth` | 2.3941 | 2.4124 | 2.3900 | 2.3936 | **Q3** [9.0, 11.0] |
| `min_data_in_leaf` | 2.3926 | 2.3871 | 2.3898 | 2.4121 | **Q2** [30.0, 35.0] |

#### E. Slice plot

![hmm@s43/moe](parts/slice_hmm@s43_moe.png)


---

## hmm@s44  (search X=[1600, 5], holdout n=400)


### naive-lightgbm

- **holdout RMSE: 2.28381** (winner retrained in 0.06s, cv score of winner: 2.1272)
- cv best RMSE: 2.1272, median: 2.1405, p10: 2.1334
- train: median 0.039s/fold, mean 0.041s, p90 0.059s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.549 |
| `min_data_in_leaf` | 0.134 |
| `max_depth` | 0.119 |
| `learning_rate` | 0.098 |
| `bagging_fraction` | 0.054 |
| `feature_fraction` | 0.028 |
| `num_leaves` | 0.016 |
| `bagging_freq` | 0.001 |
| `lambda_l1` | 0.001 |
| `lambda_l2` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 468 | 2.1429 | 0.0124 | 2.1272 |
| False | 32 | 2.1808 | 0.0224 | 2.1570 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.1483 | 2.1404 | 2.1433 | 2.1493 | **Q2** [0.0373, 0.044] |
| `num_leaves` | 2.1502 | 2.1440 | 2.1426 | 2.1448 | **Q3** [105.0, 113.0] |
| `max_depth` | — | — | — | 2.1453 | **Q4** [3.0, ∞) |
| `min_data_in_leaf` | 2.1381 | 2.1425 | 2.1448 | 2.1551 | **Q1** [None, 6.0] |
| `lambda_l1` | 2.1476 | 2.1440 | 2.1446 | 2.1450 | **Q2** [0.0, 0.0] |
| `lambda_l2` | 2.1432 | 2.1439 | 2.1436 | 2.1506 | **Q1** [None, 0.0] |
| `feature_fraction` | 2.1478 | 2.1417 | 2.1470 | 2.1447 | **Q2** [0.7242, 0.753] |
| `bagging_fraction` | 2.1528 | 2.1417 | 2.1425 | 2.1442 | **Q2** [0.8804, 0.9196] |

#### E. Slice plot

![hmm@s44/naive-lightgbm](parts/slice_hmm@s44_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 2.27014** (winner retrained in 0.14s, cv score of winner: 2.1276)
- cv best RMSE: 2.1276, median: 2.1354, p10: 2.1308
- train: median 0.195s/fold, mean 0.194s, p90 0.234s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.465 |
| `learning_rate` | 0.245 |
| `min_data_in_leaf` | 0.205 |
| `lambda_l1` | 0.029 |
| `bagging_fraction` | 0.015 |
| `feature_fraction` | 0.014 |
| `num_leaves` | 0.009 |
| `lambda_l2` | 0.007 |
| `bagging_freq` | 0.005 |
| `max_depth` | 0.004 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 468 | 2.1389 | 0.0134 | 2.1276 |
| False | 32 | 2.1809 | 0.0195 | 2.1591 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.1526 | 2.1387 | 2.1361 | 2.1391 | **Q3** [0.1463, 0.168] |
| `num_leaves` | 2.1407 | 2.1408 | 2.1390 | 2.1457 | **Q3** [62.0, 69.0] |
| `max_depth` | — | — | 2.1380 | 2.1479 | **Q3** [3.0, 4.0] |
| `min_data_in_leaf` | 2.1375 | 2.1358 | 2.1384 | 2.1525 | **Q2** [7.0, 9.0] |
| `lambda_l1` | 2.1494 | 2.1398 | 2.1386 | 2.1386 | **Q3** [3.38, 5.9232] |
| `lambda_l2` | 2.1398 | 2.1376 | 2.1419 | 2.1471 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 2.1460 | 2.1413 | 2.1396 | 2.1395 | **Q4** [0.8134, ∞) |
| `bagging_fraction` | 2.1461 | 2.1394 | 2.1385 | 2.1425 | **Q3** [0.7137, 0.7534] |

#### E. Slice plot

![hmm@s44/naive-ensemble](parts/slice_hmm@s44_naive-ensemble.png)


### moe

- **holdout RMSE: 2.24914** (winner retrained in 0.63s, cv score of winner: 2.1321)
- cv best RMSE: 2.1321, median: 2.1469, p10: 2.1404
- train: median 0.278s/fold, mean 0.341s, p90 0.455s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_init` | 0.505 |
| `mixture_gate_type` | 0.152 |
| `learning_rate` | 0.135 |
| `mixture_diversity_lambda` | 0.034 |
| `bagging_freq` | 0.026 |
| `mixture_warmup_iters` | 0.021 |
| `num_leaves` | 0.019 |
| `mixture_expert_dropout_rate` | 0.017 |
| `bagging_fraction` | 0.016 |
| `extra_trees` | 0.015 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 449 | 2.1614 | 0.0386 | 2.1321 |
| leaf_reuse | 25 | 2.1733 | 0.0438 | 2.1400 |
| none | 26 | 2.2176 | 0.0832 | 2.1428 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 468 | 2.1642 | 0.0439 | 2.1321 |
| token_choice | 32 | 2.1760 | 0.0478 | 2.1406 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| em | 384 | 2.1632 | 0.0424 | 2.1321 |
| loss_only | 71 | 2.1649 | 0.0448 | 2.1379 |
| gate_only | 45 | 2.1797 | 0.0546 | 2.1360 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| random | 392 | 2.1558 | 0.0327 | 2.1321 |
| uniform | 19 | 2.1633 | 0.0290 | 2.1434 |
| kmeans_features | 31 | 2.1781 | 0.0414 | 2.1360 |
| gmm_features | 19 | 2.1869 | 0.0450 | 2.1455 |
| gmm | 19 | 2.2326 | 0.0884 | 2.1496 |
| tree_hierarchical | 20 | 2.2386 | 0.0544 | 2.1465 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 372 | 2.1607 | 0.0407 | 2.1321 |
| none | 95 | 2.1734 | 0.0496 | 2.1360 |
| ema | 33 | 2.1884 | 0.0540 | 2.1403 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 362 | 2.1628 | 0.0416 | 2.1354 |
| True | 138 | 2.1706 | 0.0499 | 2.1321 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 469 | 2.1623 | 0.0435 | 2.1321 |
| False | 31 | 2.2044 | 0.0361 | 2.1623 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 2.1885 | — | — | 2.1630 | **Q4** [5.0, ∞) |
| `mixture_diversity_lambda` | 2.1597 | 2.1601 | 2.1687 | 2.1711 | **Q1** [None, 0.0483] |
| `mixture_warmup_iters` | 2.1747 | 2.1632 | 2.1554 | 2.1674 | **Q3** [25.0, 28.0] |
| `mixture_balance_factor` | 2.1763 | — | 2.1578 | 2.1646 | **Q3** [6.0, 7.0] |
| `learning_rate` | 2.1820 | 2.1525 | 2.1557 | 2.1695 | **Q2** [0.1264, 0.1462] |
| `num_leaves` | 2.1614 | 2.1554 | 2.1648 | 2.1784 | **Q2** [33.0, 41.0] |
| `max_depth` | 2.1687 | 2.1525 | — | 2.1700 | **Q2** [7.0, 8.0] |
| `min_data_in_leaf` | 2.1723 | 2.1664 | 2.1561 | 2.1635 | **Q3** [64.0, 68.0] |

#### E. Slice plot

![hmm@s44/moe](parts/slice_hmm@s44_moe.png)


---

## houses@s42  (search X=[8000, 8], holdout n=2000)


### naive-lightgbm

- **holdout RMSE: 50837.85512** (winner retrained in 0.10s, cv score of winner: 50493.3397)
- cv best RMSE: 50493.3397, median: 51154.2425, p10: 50883.3160
- train: median 0.090s/fold, mean 0.092s, p90 0.120s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.638 |
| `extra_trees` | 0.321 |
| `feature_fraction` | 0.016 |
| `max_depth` | 0.009 |
| `min_data_in_leaf` | 0.007 |
| `num_leaves` | 0.004 |
| `bagging_freq` | 0.002 |
| `bagging_fraction` | 0.001 |
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

![houses@s42/naive-lightgbm](parts/slice_houses@s42_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 50074.59416** (winner retrained in 0.80s, cv score of winner: 50578.9081)
- cv best RMSE: 50578.9081, median: 50951.9468, p10: 50717.1793
- train: median 0.501s/fold, mean 0.479s, p90 0.598s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.480 |
| `learning_rate` | 0.409 |
| `min_data_in_leaf` | 0.066 |
| `feature_fraction` | 0.018 |
| `max_depth` | 0.015 |
| `num_leaves` | 0.008 |
| `bagging_freq` | 0.004 |
| `bagging_fraction` | 0.001 |
| `n_models` | 0.000 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 470 | 51594.7709 | 2643.8710 | 50578.9081 |
| True | 30 | 63778.4860 | 6109.2892 | 55924.7269 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 54204.7761 | 51750.1474 | 51377.5294 | 51970.7223 | **Q3** [0.0831, 0.0948] |
| `num_leaves` | 53042.0465 | 51689.8978 | 51846.7040 | 52657.1079 | **Q2** [71.0, 83.0] |
| `max_depth` | 54288.1637 | — | — | 51685.5511 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 52261.9589 | 51830.1195 | 51485.8547 | 53754.5170 | **Q3** [19.0, 23.0] |
| `lambda_l1` | 51672.6462 | 51891.9411 | 52053.4803 | 53685.1076 | **Q1** [None, 0.0] |
| `lambda_l2` | 52646.7672 | 52202.5099 | 52081.3806 | 52372.5175 | **Q3** [0.0, 0.0002] |
| `feature_fraction` | 52387.0316 | 51485.4577 | 51785.7763 | 53644.9096 | **Q2** [0.6942, 0.7128] |
| `bagging_fraction` | 53170.8605 | 51442.8143 | 52158.2332 | 52531.2672 | **Q2** [0.8469, 0.8744] |

#### E. Slice plot

![houses@s42/naive-ensemble](parts/slice_houses@s42_naive-ensemble.png)


### moe

- **holdout RMSE: 49402.90831** (winner retrained in 6.03s, cv score of winner: 49880.1370)
- cv best RMSE: 49880.1370, median: 51081.5655, p10: 50263.1629
- train: median 6.387s/fold, mean 6.400s, p90 11.725s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.363 |
| `extra_trees` | 0.181 |
| `mixture_balance_factor` | 0.128 |
| `mixture_init` | 0.058 |
| `mixture_gate_type` | 0.055 |
| `mixture_warmup_iters` | 0.054 |
| `num_leaves` | 0.033 |
| `min_data_in_leaf` | 0.027 |
| `mixture_e_step_mode` | 0.025 |
| `feature_fraction` | 0.024 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 443 | 54405.1038 | 12896.9158 | 49880.1370 |
| leaf_reuse | 32 | 64505.7865 | 23496.3577 | 50525.6492 |
| none | 25 | 65594.6445 | 15561.9122 | 50179.6738 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 453 | 55196.0516 | 14147.3835 | 49880.1370 |
| expert_choice | 47 | 59610.6580 | 15693.8095 | 50457.2260 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 410 | 54625.9128 | 13351.9468 | 49880.1370 |
| loss_only | 65 | 55276.2957 | 9166.0798 | 50025.6185 |
| em | 25 | 72637.1520 | 25974.5897 | 50389.9226 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| kmeans_features | 211 | 53857.2468 | 9153.9923 | 49932.0607 |
| tree_hierarchical | 210 | 54748.0339 | 13756.5625 | 49880.1370 |
| gmm_features | 20 | 56918.2094 | 11332.0722 | 50507.8010 |
| uniform | 19 | 58585.2958 | 16979.8333 | 50237.1184 |
| random | 19 | 63812.9020 | 28244.4995 | 50180.1935 |
| gmm | 21 | 70505.5306 | 27170.5271 | 51012.6208 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 329 | 54392.1842 | 10361.0988 | 49880.1370 |
| ema | 139 | 56728.3504 | 18506.3726 | 49924.8082 |
| none | 32 | 63288.8426 | 23452.0195 | 50394.6600 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 450 | 55317.0735 | 14486.9688 | 49880.1370 |
| True | 50 | 58256.5841 | 12838.7427 | 50360.5497 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 54053.4596 | 11708.1748 | 49880.1370 |
| True | 32 | 78390.4117 | 25726.1751 | 56907.9819 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 68985.5821 | — | — | 55024.6661 | **Q4** [4.0, ∞) |
| `mixture_diversity_lambda` | 55384.0080 | 54105.4569 | 56067.6808 | 56886.9526 | **Q2** [0.0259, 0.0559] |
| `mixture_warmup_iters` | 57419.9274 | 55564.7040 | 55726.5771 | 54333.0221 | **Q4** [39.0, ∞) |
| `mixture_balance_factor` | 54955.1046 | 54486.2701 | 59519.5639 | 55011.3782 | **Q2** [4.0, 6.0] |
| `learning_rate` | 62707.6790 | 54313.0161 | 52386.8694 | 53036.5338 | **Q3** [0.1352, 0.162] |
| `num_leaves` | 56666.9427 | 53211.7294 | 56332.5562 | 56068.6124 | **Q2** [40.0, 52.0] |
| `max_depth` | 59080.6709 | 56201.2731 | — | 54251.2854 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 54201.7305 | 54902.1949 | 54692.8968 | 58171.1396 | **Q1** [None, 18.0] |

#### E. Slice plot

![houses@s42/moe](parts/slice_houses@s42_moe.png)


---

## houses@s43  (search X=[8000, 8], holdout n=2000)


### naive-lightgbm

- **holdout RMSE: 50650.09521** (winner retrained in 0.06s, cv score of winner: 50976.4089)
- cv best RMSE: 50976.4089, median: 51629.0269, p10: 51303.1013
- train: median 0.149s/fold, mean 0.145s, p90 0.194s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.579 |
| `learning_rate` | 0.280 |
| `min_data_in_leaf` | 0.061 |
| `max_depth` | 0.045 |
| `feature_fraction` | 0.026 |
| `bagging_fraction` | 0.003 |
| `lambda_l2` | 0.002 |
| `bagging_freq` | 0.001 |
| `lambda_l1` | 0.001 |
| `num_leaves` | 0.001 |

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

![houses@s43/naive-lightgbm](parts/slice_houses@s43_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 48759.66773** (winner retrained in 0.72s, cv score of winner: 50306.1130)
- cv best RMSE: 50306.1130, median: 50702.0156, p10: 50474.1811
- train: median 0.566s/fold, mean 0.545s, p90 0.705s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.573 |
| `learning_rate` | 0.366 |
| `max_depth` | 0.019 |
| `feature_fraction` | 0.012 |
| `min_data_in_leaf` | 0.011 |
| `bagging_fraction` | 0.010 |
| `num_leaves` | 0.005 |
| `bagging_freq` | 0.002 |
| `n_models` | 0.001 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 51471.3325 | 2532.4147 | 50306.1130 |
| True | 32 | 63284.2766 | 5030.1460 | 55828.6091 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 54094.8771 | 51331.9991 | 51329.5088 | 52153.0586 | **Q3** [0.0825, 0.0939] |
| `num_leaves` | 52380.7887 | 51717.7594 | 51660.8736 | 53155.7956 | **Q3** [64.0, 78.0] |
| `max_depth` | 55227.3532 | 51843.9037 | — | 51773.4144 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 51854.8172 | 51725.1498 | 51475.5275 | 53789.6855 | **Q3** [14.0, 19.0] |
| `lambda_l1` | 52051.3419 | 51796.3011 | 52216.9901 | 52844.8105 | **Q2** [0.0, 0.0] |
| `lambda_l2` | 52793.0268 | 51433.9140 | 51719.5379 | 52962.9649 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 52335.7167 | 51606.0057 | 51354.5583 | 53613.1630 | **Q3** [0.606, 0.6534] |
| `bagging_fraction` | 52982.7074 | 51918.4886 | 51625.3643 | 52382.8833 | **Q3** [0.8237, 0.8477] |

#### E. Slice plot

![houses@s43/naive-ensemble](parts/slice_houses@s43_naive-ensemble.png)


### moe

- **holdout RMSE: 48449.61906** (winner retrained in 3.35s, cv score of winner: 49975.9778)
- cv best RMSE: 49975.9778, median: 50906.9051, p10: 50450.7793
- train: median 0.730s/fold, mean 1.453s, p90 3.552s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.468 |
| `min_data_in_leaf` | 0.189 |
| `mixture_balance_factor` | 0.087 |
| `mixture_routing_mode` | 0.064 |
| `extra_trees` | 0.044 |
| `mixture_init` | 0.033 |
| `mixture_diversity_lambda` | 0.030 |
| `lambda_l2` | 0.029 |
| `mixture_expert_dropout_rate` | 0.011 |
| `max_depth` | 0.008 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_routing_mode` | **expert_choice** | 52872.0549 (n=467) | token_choice | Δ +18092.3153 | p=5.63e-04 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 255 | 53534.8135 | 10016.5612 | 50196.6435 |
| gbdt | 111 | 54581.4028 | 14139.1473 | 49975.9778 |
| none | 134 | 54650.4529 | 9836.9754 | 50319.5920 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 467 | 52872.0549 | 7647.5960 | 49975.9778 |
| token_choice | 33 | 70964.3702 | 26678.3076 | 50778.9222 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 339 | 53660.2202 | 11392.0772 | 49975.9778 |
| em | 98 | 54541.0117 | 9756.2201 | 50319.5920 |
| gate_only | 63 | 55511.7468 | 10805.7618 | 50645.5187 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| tree_hierarchical | 294 | 52753.1358 | 7692.6246 | 50161.3164 |
| random | 82 | 53282.4000 | 7723.7400 | 49975.9778 |
| kmeans_features | 68 | 54986.4339 | 10892.1055 | 50444.1114 |
| uniform | 20 | 55738.2130 | 8049.2885 | 50670.9132 |
| gmm_features | 18 | 61529.4254 | 18396.4699 | 50656.4290 |
| gmm | 18 | 66284.6498 | 32303.9290 | 50606.6751 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 348 | 53264.8405 | 8604.0999 | 49975.9778 |
| ema | 104 | 55564.4684 | 15760.0954 | 50161.3164 |
| none | 48 | 56629.2637 | 13324.9001 | 50251.9786 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 53367.6896 | 9955.5073 | 49975.9778 |
| True | 32 | 64281.0974 | 18487.4821 | 50663.9224 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 470 | 53249.6120 | 10067.3138 | 49975.9778 |
| True | 30 | 66858.5408 | 16361.9615 | 57061.4352 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 56756.7916 | 55798.1103 | — | 53086.9909 | **Q4** [6.0, ∞) |
| `mixture_diversity_lambda` | 53014.3545 | 53757.8086 | 52077.1914 | 57415.2363 | **Q3** [0.1012, 0.1458] |
| `mixture_warmup_iters` | 54761.4527 | 52391.5449 | 53478.1052 | 55329.7589 | **Q2** [20.0, 22.0] |
| `mixture_balance_factor` | 60530.7601 | 52987.2505 | — | 53375.4368 | **Q2** [8.0, 10.0] |
| `learning_rate` | 59938.1879 | 52040.4090 | 52173.4943 | 52112.4997 | **Q2** [0.1086, 0.1595] |
| `num_leaves` | 54214.7726 | 53984.0301 | 53108.9578 | 55044.9123 | **Q3** [50.0, 57.25] |
| `max_depth` | 57337.9191 | 52870.1781 | — | 53530.4031 | **Q2** [10.0, 11.0] |
| `min_data_in_leaf` | 52703.4519 | 53808.0901 | 52611.3530 | 57240.3862 | **Q3** [20.0, 25.25] |

#### E. Slice plot

![houses@s43/moe](parts/slice_houses@s43_moe.png)


---

## houses@s44  (search X=[8000, 8], holdout n=2000)


### naive-lightgbm

- **holdout RMSE: 48838.26386** (winner retrained in 0.31s, cv score of winner: 50654.3382)
- cv best RMSE: 50654.3382, median: 51120.2765, p10: 50847.9634
- train: median 0.169s/fold, mean 0.161s, p90 0.231s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.693 |
| `learning_rate` | 0.214 |
| `min_data_in_leaf` | 0.037 |
| `max_depth` | 0.030 |
| `feature_fraction` | 0.017 |
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

![houses@s44/naive-lightgbm](parts/slice_houses@s44_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 47614.05700** (winner retrained in 0.50s, cv score of winner: 50145.4995)
- cv best RMSE: 50145.4995, median: 50954.3367, p10: 50372.8465
- train: median 0.468s/fold, mean 0.498s, p90 0.696s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.560 |
| `learning_rate` | 0.406 |
| `lambda_l2` | 0.008 |
| `max_depth` | 0.005 |
| `n_models` | 0.005 |
| `min_data_in_leaf` | 0.004 |
| `bagging_freq` | 0.004 |
| `num_leaves` | 0.003 |
| `bagging_fraction` | 0.003 |
| `feature_fraction` | 0.002 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 470 | 51497.8930 | 2778.7443 | 50145.4995 |
| True | 30 | 62522.4075 | 5194.3681 | 57054.2326 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 53617.9055 | 51671.2877 | 51267.9287 | 52080.3335 | **Q3** [0.1024, 0.1189] |
| `num_leaves` | 51926.1866 | 51834.7773 | 51795.8666 | 53068.0959 | **Q3** [57.0, 66.0] |
| `max_depth` | 54427.0738 | 51902.7651 | 51787.3592 | 51792.4584 | **Q3** [11.0, 12.0] |
| `min_data_in_leaf` | 51325.1800 | 51261.6039 | 52526.0846 | 53182.8759 | **Q2** [10.0, 13.0] |
| `lambda_l1` | 51969.8261 | 52222.2471 | 51744.5557 | 52700.8266 | **Q3** [0.0, 0.0095] |
| `lambda_l2` | 53318.9367 | 51687.9985 | 51705.6903 | 51924.8300 | **Q2** [0.0013, 0.0252] |
| `feature_fraction` | 51587.4868 | 52393.7908 | 52109.2860 | 52546.8918 | **Q1** [None, 0.5411] |
| `bagging_fraction` | 52385.4380 | 52059.5029 | 51978.2825 | 52214.2320 | **Q3** [0.8248, 0.9275] |

#### E. Slice plot

![houses@s44/naive-ensemble](parts/slice_houses@s44_naive-ensemble.png)


### moe

- **holdout RMSE: 48152.77394** (winner retrained in 3.98s, cv score of winner: 50372.7105)
- cv best RMSE: 50372.7105, median: 51559.1744, p10: 50772.9541
- train: median 5.448s/fold, mean 6.236s, p90 12.452s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.629 |
| `feature_fraction` | 0.175 |
| `bagging_fraction` | 0.046 |
| `max_depth` | 0.024 |
| `num_leaves` | 0.021 |
| `min_data_in_leaf` | 0.019 |
| `mixture_r_smoothing` | 0.015 |
| `mixture_load_balance_alpha` | 0.012 |
| `mixture_num_experts` | 0.011 |
| `mixture_expert_dropout_rate` | 0.009 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 368 | 54871.5833 | 11597.4514 | 50470.7170 |
| leaf_reuse | 24 | 57136.8130 | 8551.8388 | 51031.5228 |
| none | 108 | 59461.1081 | 17586.5134 | 50372.7105 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| expert_choice | 36 | 55766.5153 | 12231.4914 | 50867.6643 |
| token_choice | 464 | 55987.5674 | 13217.3276 | 50372.7105 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 355 | 55415.7145 | 12906.1437 | 50372.7105 |
| em | 108 | 55747.7434 | 9645.9636 | 50793.5435 |
| gate_only | 37 | 61959.2132 | 20681.0125 | 50729.3216 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 21 | 53176.5805 | 2162.7855 | 50997.5414 |
| gmm_features | 95 | 54336.9216 | 7023.1130 | 50793.5435 |
| random | 294 | 55434.3056 | 13261.7781 | 50372.7105 |
| gmm | 19 | 56341.1796 | 8463.9778 | 50723.8021 |
| tree_hierarchical | 46 | 58100.3770 | 14339.5547 | 50548.6526 |
| kmeans_features | 25 | 66652.9800 | 24956.1222 | 50735.0752 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 450 | 55086.1131 | 12422.2592 | 50372.7105 |
| none | 25 | 62958.9915 | 15898.8310 | 52367.5187 |
| markov | 25 | 64924.0069 | 16864.5190 | 52062.7290 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 55386.6165 | 12603.3790 | 50372.7105 |
| True | 32 | 64527.7907 | 17324.9901 | 53374.9684 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 469 | 55223.1221 | 12425.4433 | 50372.7105 |
| True | 31 | 67296.1801 | 17779.3592 | 56787.4846 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 57502.4928 | — | 54887.4479 | 57563.2916 | **Q3** [3.0, 4.0] |
| `mixture_diversity_lambda` | 53271.9278 | 55846.0829 | 55877.6106 | 58890.9854 | **Q1** [None, 0.0489] |
| `mixture_warmup_iters` | 56019.1871 | 54013.3589 | 55514.9002 | 58133.9660 | **Q2** [6.0, 7.0] |
| `mixture_balance_factor` | 54816.1277 | — | — | 56340.5656 | **Q1** [None, 3.0] |
| `learning_rate` | 63552.4167 | 54429.7450 | 52833.9938 | 53070.4512 | **Q3** [0.2333, 0.2645] |
| `num_leaves` | 59396.5584 | 53693.9331 | 57105.4348 | 53841.1611 | **Q2** [104.0, 118.0] |
| `max_depth` | 58156.3068 | — | — | 55517.8344 | **Q4** [7.0, ∞) |
| `min_data_in_leaf` | 56359.1328 | 54493.6744 | 54085.7719 | 58892.2448 | **Q3** [28.0, 32.0] |

#### E. Slice plot

![houses@s44/moe](parts/slice_houses@s44_moe.png)


---

## cpu_act@s42  (search X=[6554, 21], holdout n=1638)


### naive-lightgbm

- **holdout RMSE: 2.30096** (winner retrained in 0.07s, cv score of winner: 2.5168)
- cv best RMSE: 2.5168, median: 2.5811, p10: 2.5383
- train: median 0.063s/fold, mean 0.066s, p90 0.084s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.553 |
| `min_data_in_leaf` | 0.288 |
| `extra_trees` | 0.077 |
| `feature_fraction` | 0.031 |
| `bagging_freq` | 0.021 |
| `num_leaves` | 0.015 |
| `bagging_fraction` | 0.010 |
| `max_depth` | 0.005 |
| `lambda_l1` | 0.001 |
| `lambda_l2` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 2.7109 | 0.5267 | 2.5168 |
| True | 32 | 4.6123 | 2.5358 | 2.9974 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 3.2347 | 2.6648 | 2.6706 | 2.7602 | **Q2** [0.0888, 0.1003] |
| `num_leaves` | 2.7448 | 2.6924 | 2.7699 | 3.0788 | **Q2** [18.0, 22.0] |
| `max_depth` | 3.4628 | 2.7405 | — | 2.7123 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 2.7492 | 2.5831 | 2.6221 | 3.3756 | **Q2** [7.0, 9.0] |
| `lambda_l1` | 3.0699 | 2.8129 | 2.6887 | 2.7588 | **Q3** [0.1361, 0.2856] |
| `lambda_l2` | 2.7000 | 2.6290 | 2.7748 | 3.2265 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 3.1650 | 2.7474 | 2.6873 | 2.7306 | **Q3** [0.8599, 0.8784] |
| `bagging_fraction` | 2.8267 | 2.6647 | 2.9527 | 2.8862 | **Q2** [0.5678, 0.5935] |

#### E. Slice plot

![cpu_act@s42/naive-lightgbm](parts/slice_cpu_act@s42_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 2.17111** (winner retrained in 0.31s, cv score of winner: 2.4889)
- cv best RMSE: 2.4889, median: 2.5387, p10: 2.5076
- train: median 0.308s/fold, mean 0.313s, p90 0.392s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.451 |
| `min_data_in_leaf` | 0.367 |
| `extra_trees` | 0.142 |
| `bagging_freq` | 0.013 |
| `feature_fraction` | 0.012 |
| `num_leaves` | 0.009 |
| `n_models` | 0.002 |
| `bagging_fraction` | 0.002 |
| `max_depth` | 0.001 |
| `lambda_l1` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 469 | 2.6868 | 0.5755 | 2.4889 |
| True | 31 | 3.9209 | 1.7827 | 2.9891 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 3.1156 | 2.6255 | 2.6588 | 2.6534 | **Q2** [0.1296, 0.1597] |
| `num_leaves` | 2.6483 | 2.6765 | 2.6492 | 3.0698 | **Q1** [None, 17.0] |
| `max_depth` | 3.0887 | 2.8398 | 2.6972 | 2.6731 | **Q4** [11.0, ∞) |
| `min_data_in_leaf` | — | 2.6012 | 2.6379 | 3.1572 | **Q2** [5.0, 7.0] |
| `lambda_l1` | 2.8631 | 2.6670 | 2.6997 | 2.8233 | **Q2** [0.0004, 0.0011] |
| `lambda_l2` | 3.0140 | 2.6876 | 2.6504 | 2.7012 | **Q3** [0.0158, 0.0433] |
| `feature_fraction` | 2.7862 | 2.6790 | 2.6354 | 2.9526 | **Q3** [0.652, 0.6791] |
| `bagging_fraction` | 2.6772 | 2.7591 | 2.7279 | 2.8890 | **Q1** [None, 0.5312] |

#### E. Slice plot

![cpu_act@s42/naive-ensemble](parts/slice_cpu_act@s42_naive-ensemble.png)


### moe

- **holdout RMSE: 2.14410** (winner retrained in 1.69s, cv score of winner: 2.4241)
- cv best RMSE: 2.4241, median: 2.5144, p10: 2.4556
- train: median 0.741s/fold, mean 0.778s, p90 1.185s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.724 |
| `mixture_e_step_mode` | 0.144 |
| `mixture_warmup_iters` | 0.023 |
| `feature_fraction` | 0.020 |
| `mixture_load_balance_alpha` | 0.015 |
| `mixture_init` | 0.013 |
| `mixture_expert_dropout_rate` | 0.011 |
| `num_leaves` | 0.009 |
| `mixture_r_smoothing` | 0.007 |
| `mixture_diversity_lambda` | 0.007 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_e_step_mode` | **gate_only** | 3.3748 (n=448) | loss_only | Δ +4.5039 | p=6.96e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 449 | 3.7224 | 4.4839 | 2.4241 |
| gbdt | 26 | 4.6795 | 2.9400 | 2.4557 |
| none | 25 | 6.8173 | 7.9486 | 2.7065 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 468 | 3.7283 | 4.4673 | 2.4241 |
| expert_choice | 32 | 6.8319 | 6.7215 | 2.7230 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 448 | 3.3748 | 3.4264 | 2.4241 |
| loss_only | 25 | 7.8787 | 7.4482 | 2.5138 |
| em | 27 | 9.4285 | 10.6860 | 2.5396 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 207 | 3.2964 | 3.4557 | 2.4241 |
| gmm_features | 64 | 3.4418 | 3.7532 | 2.4431 |
| tree_hierarchical | 22 | 3.8447 | 3.4426 | 2.4733 |
| kmeans_features | 105 | 3.9993 | 4.0340 | 2.4255 |
| gmm | 88 | 5.2239 | 6.3567 | 2.4316 |
| random | 14 | 6.9009 | 11.3552 | 2.4720 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 404 | 3.4402 | 3.3936 | 2.4241 |
| markov | 71 | 4.9496 | 5.6094 | 2.4874 |
| none | 25 | 8.8881 | 11.6650 | 2.4748 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 468 | 3.6373 | 3.7351 | 2.4241 |
| False | 32 | 8.1636 | 11.0840 | 2.4701 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 3.5613 | 3.5565 | 2.4241 |
| True | 32 | 9.2739 | 11.4249 | 2.8913 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 4.9061 | — | 3.0261 | 4.1593 | **Q3** [4.0, 5.0] |
| `mixture_diversity_lambda` | 3.5514 | 3.4041 | 3.8696 | 4.8827 | **Q2** [0.0554, 0.1] |
| `mixture_warmup_iters` | 3.5822 | 3.2353 | 3.8166 | 4.8241 | **Q2** [6.0, 7.0] |
| `mixture_balance_factor` | 3.6895 | — | 3.3575 | 5.5397 | **Q3** [3.0, 4.25] |
| `learning_rate` | 5.9737 | 3.0792 | 3.1760 | 3.4788 | **Q2** [0.1102, 0.1395] |
| `num_leaves` | 5.4093 | 3.9813 | 3.4398 | 2.9115 | **Q4** [115.0, ∞) |
| `max_depth` | 5.8302 | 3.5176 | 4.1795 | 3.5434 | **Q2** [9.0, 11.0] |
| `min_data_in_leaf` | 3.6416 | 3.6138 | 3.0947 | 5.4249 | **Q3** [8.0, 12.0] |

#### E. Slice plot

![cpu_act@s42/moe](parts/slice_cpu_act@s42_moe.png)


---

## cpu_act@s43  (search X=[6554, 21], holdout n=1638)


### naive-lightgbm

- **holdout RMSE: 2.91750** (winner retrained in 0.17s, cv score of winner: 2.3502)
- cv best RMSE: 2.3502, median: 2.3963, p10: 2.3683
- train: median 0.098s/fold, mean 0.097s, p90 0.130s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.544 |
| `learning_rate` | 0.313 |
| `extra_trees` | 0.068 |
| `feature_fraction` | 0.023 |
| `num_leaves` | 0.020 |
| `bagging_fraction` | 0.013 |
| `bagging_freq` | 0.009 |
| `max_depth` | 0.007 |
| `lambda_l1` | 0.003 |
| `lambda_l2` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 469 | 2.5500 | 0.6068 | 2.3502 |
| True | 31 | 3.6338 | 1.2558 | 2.8568 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.8419 | 2.5106 | 2.5571 | 2.5593 | **Q2** [0.0693, 0.0844] |
| `num_leaves` | 2.6072 | 2.5236 | 2.5118 | 2.8061 | **Q3** [28.0, 33.0] |
| `max_depth` | 2.8834 | 2.4895 | 2.5351 | 2.5840 | **Q2** [9.0, 10.0] |
| `min_data_in_leaf` | 2.4213 | 2.4891 | 2.4110 | 3.0877 | **Q3** [9.0, 12.0] |
| `lambda_l1` | 2.5797 | 2.7601 | 2.4987 | 2.6305 | **Q3** [0.1564, 1.2329] |
| `lambda_l2` | 2.6582 | 2.5230 | 2.5067 | 2.7811 | **Q3** [0.0026, 0.0157] |
| `feature_fraction` | 2.8475 | 2.5626 | 2.5383 | 2.5205 | **Q4** [0.9563, ∞) |
| `bagging_fraction` | 2.7173 | 2.5851 | 2.5059 | 2.6607 | **Q3** [0.782, 0.8282] |

#### E. Slice plot

![cpu_act@s43/naive-lightgbm](parts/slice_cpu_act@s43_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 2.92104** (winner retrained in 0.36s, cv score of winner: 2.3056)
- cv best RMSE: 2.3056, median: 2.3412, p10: 2.3170
- train: median 0.294s/fold, mean 0.301s, p90 0.388s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.622 |
| `learning_rate` | 0.279 |
| `extra_trees` | 0.030 |
| `feature_fraction` | 0.020 |
| `num_leaves` | 0.015 |
| `bagging_fraction` | 0.014 |
| `bagging_freq` | 0.010 |
| `n_models` | 0.004 |
| `lambda_l2` | 0.003 |
| `max_depth` | 0.003 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 2.4838 | 0.5328 | 2.3056 |
| True | 32 | 3.6754 | 1.6598 | 2.6196 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.8752 | 2.4080 | 2.4155 | 2.5418 | **Q2** [0.1506, 0.1733] |
| `num_leaves` | 2.5635 | 2.5013 | 2.4581 | 2.7206 | **Q3** [40.0, 48.0] |
| `max_depth` | 2.6555 | — | — | 2.5473 | **Q4** [5.0, ∞) |
| `min_data_in_leaf` | — | 2.3887 | 2.4588 | 2.9370 | **Q2** [5.0, 8.0] |
| `lambda_l1` | 2.4677 | 2.5515 | 2.6585 | 2.5627 | **Q1** [None, 0.0] |
| `lambda_l2` | 2.5512 | 2.5146 | 2.4890 | 2.6856 | **Q3** [0.0, 0.0021] |
| `feature_fraction` | 2.9020 | 2.4195 | 2.4396 | 2.4793 | **Q2** [0.9381, 0.9634] |
| `bagging_fraction` | 2.7924 | 2.5251 | 2.4210 | 2.5019 | **Q3** [0.886, 0.909] |

#### E. Slice plot

![cpu_act@s43/naive-ensemble](parts/slice_cpu_act@s43_naive-ensemble.png)


### moe

- **holdout RMSE: 2.88364** (winner retrained in 1.76s, cv score of winner: 2.2881)
- cv best RMSE: 2.2881, median: 2.3740, p10: 2.3081
- train: median 0.411s/fold, mean 0.699s, p90 1.602s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.338 |
| `mixture_init` | 0.154 |
| `mixture_balance_factor` | 0.125 |
| `mixture_expert_dropout_rate` | 0.101 |
| `mixture_e_step_mode` | 0.052 |
| `feature_fraction` | 0.048 |
| `min_data_in_leaf` | 0.047 |
| `lambda_l2` | 0.032 |
| `mixture_refit_leaves` | 0.029 |
| `num_leaves` | 0.029 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 153 | 3.2828 | 4.5078 | 2.2941 |
| gbdt | 200 | 3.5518 | 4.6465 | 2.2881 |
| none | 147 | 4.6058 | 8.8421 | 2.3323 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 447 | 3.6122 | 5.9580 | 2.2881 |
| expert_choice | 53 | 5.1891 | 7.6268 | 2.3454 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 202 | 3.1435 | 3.7661 | 2.2881 |
| loss_only | 244 | 3.4472 | 4.6489 | 2.3295 |
| em | 54 | 7.6588 | 13.6141 | 2.3562 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 385 | 2.8848 | 2.9567 | 2.2881 |
| tree_hierarchical | 24 | 3.4155 | 1.7215 | 2.4541 |
| random | 35 | 3.4518 | 3.7979 | 2.3433 |
| kmeans_features | 18 | 7.3839 | 9.7061 | 2.4540 |
| gmm | 19 | 7.6778 | 12.8569 | 2.4881 |
| gmm_features | 19 | 15.6558 | 18.8799 | 2.5260 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 192 | 3.1545 | 3.9082 | 2.2881 |
| ema | 260 | 3.3231 | 4.1397 | 2.3294 |
| none | 48 | 8.7500 | 14.6947 | 2.4011 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 3.4652 | 5.6234 | 2.2881 |
| True | 32 | 8.3742 | 10.5281 | 2.3881 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 470 | 3.6190 | 6.0133 | 2.2881 |
| True | 30 | 6.2918 | 7.8982 | 2.7279 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 4.5212 | — | 4.1231 | 2.6901 | **Q4** [6.0, ∞) |
| `mixture_diversity_lambda` | 4.5448 | 3.6063 | 3.9369 | 3.0295 | **Q4** [0.4304, ∞) |
| `mixture_warmup_iters` | 3.0662 | 3.1429 | 3.8544 | 5.0076 | **Q1** [None, 14.0] |
| `mixture_balance_factor` | 6.9837 | — | 3.3426 | 3.1008 | **Q4** [8.0, ∞) |
| `learning_rate` | 6.7139 | 3.1206 | 2.4879 | 2.7950 | **Q3** [0.2331, 0.2665] |
| `num_leaves` | 3.2287 | 3.2008 | 3.6359 | 4.9342 | **Q2** [21.0, 58.0] |
| `max_depth` | 2.6500 | — | 4.2202 | 3.3085 | **Q1** [None, 5.0] |
| `min_data_in_leaf` | 3.1569 | 3.0857 | 2.9274 | 5.6684 | **Q3** [12.0, 17.0] |

#### E. Slice plot

![cpu_act@s43/moe](parts/slice_cpu_act@s43_moe.png)


---

## cpu_act@s44  (search X=[6554, 21], holdout n=1638)


### naive-lightgbm

- **holdout RMSE: 2.27606** (winner retrained in 0.17s, cv score of winner: 2.4780)
- cv best RMSE: 2.4780, median: 2.5377, p10: 2.5053
- train: median 0.062s/fold, mean 0.063s, p90 0.083s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.448 |
| `learning_rate` | 0.277 |
| `extra_trees` | 0.160 |
| `feature_fraction` | 0.056 |
| `num_leaves` | 0.028 |
| `bagging_fraction` | 0.015 |
| `max_depth` | 0.014 |
| `bagging_freq` | 0.002 |
| `lambda_l1` | 0.001 |
| `lambda_l2` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 470 | 2.6744 | 0.5240 | 2.4780 |
| True | 30 | 3.6872 | 1.0227 | 2.8796 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 2.9760 | 2.6160 | 2.6132 | 2.7356 | **Q3** [0.0969, 0.1088] |
| `num_leaves` | 2.6476 | 2.6437 | 2.6442 | 2.9940 | **Q2** [18.0, 21.0] |
| `max_depth` | 2.9005 | — | — | 2.6891 | **Q4** [7.0, ∞) |
| `min_data_in_leaf` | 2.5548 | 2.5717 | 2.6144 | 3.1387 | **Q1** [None, 5.75] |
| `lambda_l1` | 2.7812 | 2.8047 | 2.5968 | 2.7580 | **Q3** [0.0071, 0.017] |
| `lambda_l2` | 2.6496 | 2.7250 | 2.6849 | 2.8814 | **Q1** [None, 0.0] |
| `feature_fraction` | 3.0691 | 2.6737 | 2.6105 | 2.5875 | **Q4** [0.9908, ∞) |
| `bagging_fraction` | 2.8852 | 2.7112 | 2.6834 | 2.6610 | **Q4** [0.969, ∞) |

#### E. Slice plot

![cpu_act@s44/naive-lightgbm](parts/slice_cpu_act@s44_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 2.25330** (winner retrained in 0.26s, cv score of winner: 2.4512)
- cv best RMSE: 2.4512, median: 2.4980, p10: 2.4684
- train: median 0.325s/fold, mean 0.340s, p90 0.463s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.397 |
| `min_data_in_leaf` | 0.383 |
| `extra_trees` | 0.107 |
| `bagging_freq` | 0.049 |
| `max_depth` | 0.029 |
| `num_leaves` | 0.015 |
| `bagging_fraction` | 0.010 |
| `feature_fraction` | 0.008 |
| `lambda_l1` | 0.001 |
| `n_models` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 455 | 2.6740 | 0.6567 | 2.4512 |
| True | 45 | 3.5669 | 1.1725 | 2.8613 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 3.3075 | 2.5685 | 2.5339 | 2.6076 | **Q3** [0.2038, 0.2343] |
| `num_leaves` | 2.6290 | 2.6242 | 2.6454 | 3.1069 | **Q2** [15.75, 20.0] |
| `max_depth` | 3.1408 | 2.6535 | 2.6529 | 2.6485 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 2.6126 | 2.5798 | 2.5548 | 3.2649 | **Q3** [8.0, 12.0] |
| `lambda_l1` | 2.9587 | 2.7398 | 2.7198 | 2.5990 | **Q4** [0.9857, ∞) |
| `lambda_l2` | 2.7626 | 2.8383 | 2.6597 | 2.7568 | **Q3** [0.0024, 0.0185] |
| `feature_fraction` | 2.7468 | 2.6178 | 2.7558 | 2.8970 | **Q2** [0.6719, 0.7013] |
| `bagging_fraction` | 2.7468 | 2.6598 | 2.6411 | 2.9698 | **Q3** [0.7346, 0.7607] |

#### E. Slice plot

![cpu_act@s44/naive-ensemble](parts/slice_cpu_act@s44_naive-ensemble.png)


### moe

- **holdout RMSE: 2.21959** (winner retrained in 1.38s, cv score of winner: 2.3960)
- cv best RMSE: 2.3960, median: 2.5175, p10: 2.4329
- train: median 0.762s/fold, mean 0.798s, p90 1.180s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.345 |
| `mixture_balance_factor` | 0.204 |
| `mixture_gate_type` | 0.143 |
| `mixture_init` | 0.048 |
| `mixture_r_smoothing` | 0.045 |
| `num_leaves` | 0.034 |
| `bagging_freq` | 0.032 |
| `lambda_l2` | 0.032 |
| `mixture_load_balance_alpha` | 0.024 |
| `mixture_e_step_mode` | 0.022 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 444 | 3.4313 | 4.0863 | 2.3960 |
| gbdt | 28 | 6.1849 | 5.4456 | 2.5067 |
| none | 28 | 10.7263 | 12.5805 | 3.1083 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 470 | 3.8255 | 5.1146 | 2.3960 |
| expert_choice | 30 | 6.6334 | 7.5335 | 2.7485 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 445 | 3.8138 | 5.3783 | 2.3960 |
| em | 31 | 4.8934 | 4.0860 | 2.4760 |
| loss_only | 24 | 6.1731 | 5.3096 | 2.6696 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 17 | 3.4107 | 1.5139 | 2.5551 |
| tree_hierarchical | 326 | 3.5169 | 4.2195 | 2.3960 |
| gmm_features | 19 | 3.6831 | 1.7193 | 2.5034 |
| kmeans_features | 63 | 4.6458 | 5.8358 | 2.4525 |
| gmm | 57 | 5.1187 | 8.2581 | 2.4445 |
| random | 18 | 7.6723 | 10.1743 | 2.6180 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 356 | 3.7180 | 4.9267 | 2.3960 |
| markov | 120 | 4.1095 | 4.6971 | 2.4525 |
| none | 24 | 7.5109 | 10.4199 | 2.4321 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 467 | 3.8178 | 4.9845 | 2.3960 |
| False | 33 | 6.4881 | 8.5229 | 2.4607 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 469 | 3.7771 | 5.0887 | 2.3960 |
| True | 31 | 7.2752 | 7.4475 | 2.8021 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 5.5826 | — | — | 3.7049 | **Q4** [5.0, ∞) |
| `mixture_diversity_lambda` | 3.2341 | 3.9950 | 3.1262 | 5.6208 | **Q3** [0.0268, 0.0596] |
| `mixture_warmup_iters` | 3.9360 | 4.2748 | 3.2376 | 4.3601 | **Q3** [33.0, 35.0] |
| `mixture_balance_factor` | 5.2957 | 4.4376 | — | 3.3571 | **Q4** [9.0, ∞) |
| `learning_rate` | 7.1769 | 2.9354 | 2.9294 | 2.9344 | **Q3** [0.2095, 0.2388] |
| `num_leaves` | 3.3174 | 3.9511 | 3.2313 | 5.5705 | **Q3** [24.0, 36.25] |
| `max_depth` | 4.2582 | — | 3.4710 | 4.8030 | **Q3** [8.0, 9.0] |
| `min_data_in_leaf` | 3.4670 | 3.7849 | 3.7160 | 4.9276 | **Q1** [None, 7.0] |

#### E. Slice plot

![cpu_act@s44/moe](parts/slice_cpu_act@s44_moe.png)


---

## elevators@s42  (search X=[8000, 18], holdout n=2000)


### naive-lightgbm

- **holdout RMSE: 0.00245** (winner retrained in 0.07s, cv score of winner: 0.0026)
- cv best RMSE: 0.0026, median: 0.0027, p10: 0.0027
- train: median 0.044s/fold, mean 0.045s, p90 0.060s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.707 |
| `min_data_in_leaf` | 0.100 |
| `learning_rate` | 0.083 |
| `num_leaves` | 0.043 |
| `bagging_fraction` | 0.022 |
| `feature_fraction` | 0.021 |
| `max_depth` | 0.021 |
| `bagging_freq` | 0.003 |
| `extra_trees` | 0.001 |
| `lambda_l2` | 0.000 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 0.0028 | 0.0005 | 0.0026 |
| True | 32 | 0.0034 | 0.0008 | 0.0028 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0031 | 0.0028 | 0.0027 | 0.0028 | **Q3** [0.2162, 0.2435] |
| `num_leaves` | 0.0029 | 0.0028 | 0.0028 | 0.0030 | **Q2** [13.0, 17.0] |
| `max_depth` | 0.0030 | — | — | 0.0028 | **Q4** [6.0, ∞) |
| `min_data_in_leaf` | 0.0027 | 0.0029 | 0.0028 | 0.0030 | **Q1** [None, 11.0] |
| `lambda_l1` | 0.0029 | 0.0028 | 0.0028 | 0.0030 | **Q2** [0.0001, 0.0004] |
| `lambda_l2` | 0.0030 | 0.0028 | 0.0028 | 0.0028 | **Q2** [0.0001, 0.0004] |
| `feature_fraction` | 0.0030 | 0.0028 | 0.0028 | 0.0028 | **Q2** [0.91, 0.9324] |
| `bagging_fraction` | 0.0030 | 0.0028 | 0.0028 | 0.0028 | **Q2** [0.9364, 0.9573] |

#### E. Slice plot

![elevators@s42/naive-lightgbm](parts/slice_elevators@s42_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 0.00238** (winner retrained in 0.32s, cv score of winner: 0.0025)
- cv best RMSE: 0.0025, median: 0.0026, p10: 0.0026
- train: median 0.198s/fold, mean 0.215s, p90 0.299s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.712 |
| `learning_rate` | 0.101 |
| `bagging_freq` | 0.043 |
| `feature_fraction` | 0.026 |
| `max_depth` | 0.026 |
| `min_data_in_leaf` | 0.025 |
| `n_models` | 0.024 |
| `extra_trees` | 0.019 |
| `num_leaves` | 0.014 |
| `bagging_fraction` | 0.009 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 0.0027 | 0.0005 | 0.0025 |
| True | 32 | 0.0032 | 0.0007 | 0.0026 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0031 | 0.0027 | 0.0026 | 0.0027 | **Q3** [0.2517, 0.2785] |
| `num_leaves` | 0.0027 | 0.0028 | 0.0027 | 0.0029 | **Q1** [None, 22.75] |
| `max_depth` | 0.0028 | — | 0.0028 | 0.0028 | **Q1** [None, 4.0] |
| `min_data_in_leaf` | 0.0027 | 0.0027 | 0.0027 | 0.0030 | **Q1** [None, 10.0] |
| `lambda_l1` | 0.0028 | 0.0027 | 0.0027 | 0.0029 | **Q2** [0.0, 0.0001] |
| `lambda_l2` | 0.0027 | 0.0028 | 0.0027 | 0.0029 | **Q1** [None, 0.0] |
| `feature_fraction` | 0.0028 | 0.0028 | 0.0027 | 0.0028 | **Q3** [0.9601, 0.9779] |
| `bagging_fraction` | 0.0028 | 0.0028 | 0.0028 | 0.0027 | **Q4** [0.9212, ∞) |

#### E. Slice plot

![elevators@s42/naive-ensemble](parts/slice_elevators@s42_naive-ensemble.png)


### moe

- **holdout RMSE: 0.00241** (winner retrained in 0.41s, cv score of winner: 0.0026)
- cv best RMSE: 0.0026, median: 0.0028, p10: 0.0027
- train: median 0.268s/fold, mean 0.788s, p90 2.953s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `mixture_load_balance_alpha` | 0.445 |
| `lambda_l1` | 0.132 |
| `mixture_r_smoothing` | 0.104 |
| `mixture_num_experts` | 0.082 |
| `mixture_diversity_lambda` | 0.058 |
| `num_leaves` | 0.043 |
| `mixture_warmup_iters` | 0.040 |
| `learning_rate` | 0.023 |
| `min_data_in_leaf` | 0.013 |
| `mixture_e_step_mode` | 0.011 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 447 | 0.0039 | 0.0071 | 0.0026 |
| none | 24 | 0.0050 | 0.0021 | 0.0031 |
| leaf_reuse | 29 | 0.0053 | 0.0032 | 0.0032 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 470 | 0.0039 | 0.0069 | 0.0026 |
| expert_choice | 30 | 0.0052 | 0.0044 | 0.0028 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| em | 354 | 0.0036 | 0.0040 | 0.0026 |
| gate_only | 25 | 0.0040 | 0.0019 | 0.0027 |
| loss_only | 121 | 0.0050 | 0.0120 | 0.0027 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 65 | 0.0035 | 0.0026 | 0.0027 |
| kmeans_features | 22 | 0.0035 | 0.0009 | 0.0028 |
| tree_hierarchical | 334 | 0.0040 | 0.0080 | 0.0026 |
| gmm | 37 | 0.0044 | 0.0028 | 0.0027 |
| gmm_features | 22 | 0.0045 | 0.0043 | 0.0027 |
| random | 20 | 0.0053 | 0.0049 | 0.0027 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 418 | 0.0036 | 0.0039 | 0.0026 |
| ema | 25 | 0.0047 | 0.0033 | 0.0027 |
| none | 57 | 0.0064 | 0.0169 | 0.0027 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 468 | 0.0039 | 0.0070 | 0.0026 |
| False | 32 | 0.0061 | 0.0031 | 0.0028 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 0.0038 | 0.0064 | 0.0026 |
| True | 32 | 0.0071 | 0.0110 | 0.0027 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | — | — | — | 0.0040 | **Q4** [2.0, ∞) |
| `mixture_diversity_lambda` | 0.0050 | 0.0036 | 0.0040 | 0.0034 | **Q4** [0.3914, ∞) |
| `mixture_warmup_iters` | 0.0034 | 0.0035 | 0.0036 | 0.0054 | **Q1** [None, 11.0] |
| `mixture_balance_factor` | 0.0055 | 0.0034 | — | 0.0036 | **Q2** [9.0, 10.0] |
| `learning_rate` | 0.0045 | 0.0038 | 0.0032 | 0.0045 | **Q3** [0.2421, 0.2683] |
| `num_leaves` | 0.0034 | 0.0037 | 0.0038 | 0.0050 | **Q1** [None, 13.0] |
| `max_depth` | 0.0065 | 0.0037 | — | 0.0038 | **Q2** [4.0, 5.0] |
| `min_data_in_leaf` | 0.0037 | 0.0035 | 0.0049 | 0.0040 | **Q2** [16.0, 20.5] |

#### E. Slice plot

![elevators@s42/moe](parts/slice_elevators@s42_moe.png)


---

## elevators@s43  (search X=[8000, 18], holdout n=2000)


### naive-lightgbm

- **holdout RMSE: 0.00237** (winner retrained in 0.10s, cv score of winner: 0.0026)
- cv best RMSE: 0.0026, median: 0.0027, p10: 0.0027
- train: median 0.070s/fold, mean 0.073s, p90 0.101s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.685 |
| `learning_rate` | 0.103 |
| `feature_fraction` | 0.086 |
| `bagging_fraction` | 0.041 |
| `num_leaves` | 0.024 |
| `max_depth` | 0.015 |
| `bagging_freq` | 0.014 |
| `min_data_in_leaf` | 0.014 |
| `extra_trees` | 0.013 |
| `lambda_l2` | 0.004 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 469 | 0.0028 | 0.0004 | 0.0026 |
| True | 31 | 0.0035 | 0.0006 | 0.0028 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0031 | 0.0028 | 0.0028 | 0.0028 | **Q2** [0.1346, 0.1677] |
| `num_leaves` | 0.0028 | 0.0028 | 0.0028 | 0.0030 | **Q1** [None, 19.0] |
| `max_depth` | 0.0030 | 0.0028 | 0.0028 | 0.0028 | **Q2** [7.0, 8.0] |
| `min_data_in_leaf` | 0.0027 | 0.0028 | 0.0028 | 0.0030 | **Q1** [None, 13.0] |
| `lambda_l1` | 0.0028 | 0.0028 | 0.0028 | 0.0030 | **Q1** [None, 0.0] |
| `lambda_l2` | 0.0029 | 0.0028 | 0.0028 | 0.0029 | **Q2** [0.0035, 0.0406] |
| `feature_fraction` | 0.0030 | 0.0027 | 0.0028 | 0.0027 | **Q2** [0.901, 0.9313] |
| `bagging_fraction` | 0.0029 | 0.0028 | 0.0029 | 0.0028 | **Q2** [0.8597, 0.9186] |

#### E. Slice plot

![elevators@s43/naive-lightgbm](parts/slice_elevators@s43_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 0.00226** (winner retrained in 0.21s, cv score of winner: 0.0025)
- cv best RMSE: 0.0025, median: 0.0026, p10: 0.0026
- train: median 0.198s/fold, mean 0.225s, p90 0.337s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.709 |
| `bagging_freq` | 0.091 |
| `learning_rate` | 0.059 |
| `bagging_fraction` | 0.037 |
| `max_depth` | 0.033 |
| `n_models` | 0.023 |
| `min_data_in_leaf` | 0.023 |
| `num_leaves` | 0.013 |
| `extra_trees` | 0.009 |
| `feature_fraction` | 0.002 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 0.0027 | 0.0005 | 0.0025 |
| True | 32 | 0.0033 | 0.0008 | 0.0026 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0031 | 0.0027 | 0.0026 | 0.0027 | **Q3** [0.2375, 0.266] |
| `num_leaves` | 0.0028 | 0.0027 | 0.0027 | 0.0030 | **Q2** [12.0, 15.0] |
| `max_depth` | 0.0031 | 0.0028 | — | 0.0027 | **Q4** [11.0, ∞) |
| `min_data_in_leaf` | 0.0027 | 0.0027 | 0.0027 | 0.0030 | **Q1** [None, 7.0] |
| `lambda_l1` | 0.0027 | 0.0027 | 0.0026 | 0.0030 | **Q3** [0.0, 0.0] |
| `lambda_l2` | 0.0028 | 0.0027 | 0.0028 | 0.0028 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 0.0030 | 0.0027 | 0.0027 | 0.0027 | **Q2** [0.929, 0.9593] |
| `bagging_fraction` | 0.0028 | 0.0027 | 0.0027 | 0.0029 | **Q2** [0.7398, 0.7637] |

#### E. Slice plot

![elevators@s43/naive-ensemble](parts/slice_elevators@s43_naive-ensemble.png)


### moe

- **holdout RMSE: 0.00219** (winner retrained in 0.74s, cv score of winner: 0.0025)
- cv best RMSE: 0.0025, median: 0.0026, p10: 0.0025
- train: median 0.880s/fold, mean 2.219s, p90 8.263s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `min_data_in_leaf` | 0.199 |
| `lambda_l1` | 0.135 |
| `mixture_load_balance_alpha` | 0.120 |
| `learning_rate` | 0.099 |
| `bagging_fraction` | 0.086 |
| `mixture_r_smoothing` | 0.076 |
| `mixture_e_step_mode` | 0.046 |
| `bagging_freq` | 0.037 |
| `extra_trees` | 0.033 |
| `mixture_refit_leaves` | 0.024 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_e_step_mode` | **gate_only** | 0.0030 (n=416) | loss_only | Δ +0.0027 | p=2.84e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 304 | 0.0032 | 0.0027 | 0.0025 |
| gbdt | 171 | 0.0038 | 0.0033 | 0.0025 |
| none | 25 | 0.0057 | 0.0047 | 0.0028 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 465 | 0.0034 | 0.0028 | 0.0025 |
| expert_choice | 35 | 0.0058 | 0.0053 | 0.0030 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 416 | 0.0030 | 0.0021 | 0.0025 |
| loss_only | 30 | 0.0057 | 0.0045 | 0.0029 |
| em | 54 | 0.0065 | 0.0053 | 0.0027 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 100 | 0.0031 | 0.0024 | 0.0025 |
| random | 238 | 0.0032 | 0.0025 | 0.0025 |
| tree_hierarchical | 71 | 0.0033 | 0.0028 | 0.0026 |
| gmm_features | 28 | 0.0042 | 0.0034 | 0.0025 |
| gmm | 36 | 0.0052 | 0.0048 | 0.0025 |
| kmeans_features | 27 | 0.0061 | 0.0050 | 0.0025 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 291 | 0.0032 | 0.0024 | 0.0025 |
| none | 137 | 0.0034 | 0.0030 | 0.0025 |
| ema | 72 | 0.0054 | 0.0046 | 0.0026 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 418 | 0.0031 | 0.0022 | 0.0025 |
| False | 82 | 0.0059 | 0.0052 | 0.0026 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 470 | 0.0034 | 0.0030 | 0.0025 |
| True | 30 | 0.0055 | 0.0043 | 0.0027 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.0069 | — | 0.0031 | 0.0042 | **Q3** [4.0, 5.0] |
| `mixture_diversity_lambda` | 0.0028 | 0.0035 | 0.0032 | 0.0047 | **Q1** [None, 0.0413] |
| `mixture_warmup_iters` | 0.0035 | 0.0029 | 0.0033 | 0.0045 | **Q2** [16.0, 21.0] |
| `mixture_balance_factor` | 0.0059 | 0.0033 | 0.0036 | 0.0031 | **Q4** [9.0, ∞) |
| `learning_rate` | 0.0046 | 0.0032 | 0.0034 | 0.0030 | **Q4** [0.2689, ∞) |
| `num_leaves` | 0.0034 | 0.0032 | 0.0032 | 0.0043 | **Q2** [65.0, 72.0] |
| `max_depth` | 0.0036 | 0.0028 | 0.0031 | 0.0043 | **Q2** [6.0, 7.0] |
| `min_data_in_leaf` | 0.0027 | 0.0034 | 0.0028 | 0.0051 | **Q1** [None, 13.0] |

#### E. Slice plot

![elevators@s43/moe](parts/slice_elevators@s43_moe.png)


---

## elevators@s44  (search X=[8000, 18], holdout n=2000)


### naive-lightgbm

- **holdout RMSE: 0.00225** (winner retrained in 0.15s, cv score of winner: 0.0026)
- cv best RMSE: 0.0026, median: 0.0027, p10: 0.0026
- train: median 0.079s/fold, mean 0.079s, p90 0.107s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.777 |
| `learning_rate` | 0.076 |
| `feature_fraction` | 0.047 |
| `max_depth` | 0.024 |
| `extra_trees` | 0.021 |
| `min_data_in_leaf` | 0.019 |
| `bagging_fraction` | 0.015 |
| `num_leaves` | 0.013 |
| `lambda_l2` | 0.006 |
| `bagging_freq` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 469 | 0.0028 | 0.0004 | 0.0026 |
| True | 31 | 0.0034 | 0.0007 | 0.0029 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0030 | 0.0027 | 0.0027 | 0.0028 | **Q2** [0.1225, 0.1411] |
| `num_leaves` | 0.0028 | 0.0029 | 0.0030 | 0.0027 | **Q4** [121.0, ∞) |
| `max_depth` | 0.0031 | 0.0028 | — | 0.0028 | **Q2** [6.0, 8.0] |
| `min_data_in_leaf` | 0.0029 | 0.0028 | 0.0027 | 0.0029 | **Q3** [19.0, 23.0] |
| `lambda_l1` | 0.0029 | 0.0027 | 0.0027 | 0.0030 | **Q2** [0.0, 0.0001] |
| `lambda_l2` | 0.0028 | 0.0027 | 0.0028 | 0.0030 | **Q2** [0.0, 0.0001] |
| `feature_fraction` | 0.0031 | 0.0028 | 0.0027 | 0.0027 | **Q3** [0.9642, 0.9813] |
| `bagging_fraction` | 0.0030 | 0.0028 | 0.0028 | 0.0027 | **Q4** [0.9802, ∞) |

#### E. Slice plot

![elevators@s44/naive-lightgbm](parts/slice_elevators@s44_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 0.00225** (winner retrained in 0.31s, cv score of winner: 0.0025)
- cv best RMSE: 0.0025, median: 0.0025, p10: 0.0025
- train: median 0.212s/fold, mean 0.215s, p90 0.263s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.547 |
| `learning_rate` | 0.291 |
| `bagging_fraction` | 0.060 |
| `max_depth` | 0.027 |
| `bagging_freq` | 0.027 |
| `feature_fraction` | 0.020 |
| `n_models` | 0.012 |
| `min_data_in_leaf` | 0.006 |
| `lambda_l2` | 0.006 |
| `num_leaves` | 0.003 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 470 | 0.0027 | 0.0005 | 0.0025 |
| True | 30 | 0.0033 | 0.0008 | 0.0026 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.0031 | 0.0026 | 0.0026 | 0.0026 | **Q2** [0.232, 0.2637] |
| `num_leaves` | 0.0026 | 0.0027 | 0.0027 | 0.0030 | **Q1** [None, 12.0] |
| `max_depth` | 0.0029 | 0.0027 | 0.0026 | 0.0027 | **Q3** [9.0, 10.0] |
| `min_data_in_leaf` | 0.0026 | 0.0027 | 0.0027 | 0.0029 | **Q1** [None, 7.0] |
| `lambda_l1` | 0.0026 | 0.0026 | 0.0027 | 0.0030 | **Q1** [None, 0.0] |
| `lambda_l2` | 0.0029 | 0.0027 | 0.0026 | 0.0027 | **Q3** [1.2242, 3.9295] |
| `feature_fraction` | 0.0029 | 0.0026 | 0.0027 | 0.0027 | **Q2** [0.8848, 0.9175] |
| `bagging_fraction` | 0.0027 | 0.0027 | 0.0027 | 0.0028 | **Q1** [None, 0.7029] |

#### E. Slice plot

![elevators@s44/naive-ensemble](parts/slice_elevators@s44_naive-ensemble.png)


### moe

- **holdout RMSE: 0.00239** (winner retrained in 3.62s, cv score of winner: 0.0026)
- cv best RMSE: 0.0026, median: 0.0029, p10: 0.0027
- train: median 1.387s/fold, mean 1.510s, p90 2.195s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.357 |
| `lambda_l2` | 0.243 |
| `bagging_fraction` | 0.138 |
| `mixture_gate_type` | 0.067 |
| `mixture_routing_mode` | 0.051 |
| `mixture_balance_factor` | 0.035 |
| `mixture_init` | 0.029 |
| `num_leaves` | 0.024 |
| `learning_rate` | 0.015 |
| `mixture_refit_leaves` | 0.015 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_gate_type` | **gbdt** | 0.0039 (n=449) | leaf_reuse | Δ +0.0029 | p=4.90e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gbdt | 449 | 0.0039 | 0.0035 | 0.0026 |
| leaf_reuse | 25 | 0.0068 | 0.0045 | 0.0032 |
| none | 26 | 0.0069 | 0.0039 | 0.0038 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 468 | 0.0041 | 0.0037 | 0.0026 |
| expert_choice | 32 | 0.0055 | 0.0036 | 0.0030 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 446 | 0.0040 | 0.0036 | 0.0026 |
| em | 29 | 0.0053 | 0.0034 | 0.0027 |
| gate_only | 25 | 0.0065 | 0.0046 | 0.0028 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 24 | 0.0036 | 0.0015 | 0.0028 |
| tree_hierarchical | 17 | 0.0037 | 0.0013 | 0.0028 |
| gmm | 333 | 0.0040 | 0.0038 | 0.0026 |
| gmm_features | 90 | 0.0042 | 0.0034 | 0.0027 |
| random | 18 | 0.0052 | 0.0047 | 0.0027 |
| kmeans_features | 18 | 0.0063 | 0.0043 | 0.0027 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| markov | 312 | 0.0039 | 0.0031 | 0.0026 |
| none | 165 | 0.0045 | 0.0043 | 0.0026 |
| ema | 23 | 0.0057 | 0.0047 | 0.0027 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 468 | 0.0039 | 0.0034 | 0.0026 |
| False | 32 | 0.0081 | 0.0053 | 0.0027 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 0.0041 | 0.0036 | 0.0026 |
| True | 32 | 0.0057 | 0.0041 | 0.0030 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.0046 | — | — | 0.0041 | **Q4** [4.0, ∞) |
| `mixture_diversity_lambda` | 0.0037 | 0.0045 | 0.0036 | 0.0048 | **Q3** [0.23, 0.2629] |
| `mixture_warmup_iters` | 0.0037 | 0.0037 | 0.0042 | 0.0050 | **Q1** [None, 16.0] |
| `mixture_balance_factor` | 0.0045 | 0.0040 | — | 0.0041 | **Q2** [9.0, 10.0] |
| `learning_rate` | 0.0048 | 0.0036 | 0.0039 | 0.0044 | **Q2** [0.1404, 0.1786] |
| `num_leaves` | 0.0042 | 0.0042 | 0.0038 | 0.0044 | **Q3** [78.0, 92.0] |
| `max_depth` | 0.0044 | 0.0046 | — | 0.0038 | **Q4** [10.0, ∞) |
| `min_data_in_leaf` | 0.0042 | 0.0038 | 0.0037 | 0.0049 | **Q3** [31.0, 35.0] |

#### E. Slice plot

![elevators@s44/moe](parts/slice_elevators@s44_moe.png)


---

## wine_quality@s42  (search X=[5198, 11], holdout n=1299)


### naive-lightgbm

- **holdout RMSE: 0.60747** (winner retrained in 0.19s, cv score of winner: 0.6678)
- cv best RMSE: 0.6678, median: 0.6752, p10: 0.6708
- train: median 0.157s/fold, mean 0.150s, p90 0.216s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.413 |
| `learning_rate` | 0.381 |
| `min_data_in_leaf` | 0.121 |
| `max_depth` | 0.031 |
| `bagging_fraction` | 0.028 |
| `feature_fraction` | 0.011 |
| `num_leaves` | 0.010 |
| `bagging_freq` | 0.004 |
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

![wine_quality@s42/naive-lightgbm](parts/slice_wine_quality@s42_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 0.60192** (winner retrained in 0.86s, cv score of winner: 0.6624)
- cv best RMSE: 0.6624, median: 0.6697, p10: 0.6661
- train: median 0.625s/fold, mean 0.603s, p90 0.867s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.423 |
| `learning_rate` | 0.329 |
| `min_data_in_leaf` | 0.178 |
| `max_depth` | 0.024 |
| `feature_fraction` | 0.019 |
| `n_models` | 0.009 |
| `bagging_fraction` | 0.006 |
| `num_leaves` | 0.005 |
| `lambda_l1` | 0.004 |
| `bagging_freq` | 0.002 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 470 | 0.6737 | 0.0114 | 0.6624 |
| True | 30 | 0.7097 | 0.0227 | 0.6842 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.6823 | 0.6722 | 0.6739 | 0.6752 | **Q2** [0.0532, 0.0644] |
| `num_leaves` | 0.6804 | 0.6748 | 0.6735 | 0.6750 | **Q3** [117.0, 123.0] |
| `max_depth` | 0.6889 | — | 0.6738 | 0.6719 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | — | 0.6700 | 0.6721 | 0.6887 | **Q2** [5.0, 7.0] |
| `lambda_l1` | 0.6811 | 0.6733 | 0.6727 | 0.6764 | **Q3** [0.0485, 0.1366] |
| `lambda_l2` | 0.6784 | 0.6782 | 0.6734 | 0.6737 | **Q3** [0.008, 0.0742] |
| `feature_fraction` | 0.6805 | 0.6722 | 0.6734 | 0.6775 | **Q2** [0.683, 0.7201] |
| `bagging_fraction` | 0.6764 | 0.6724 | 0.6729 | 0.6819 | **Q2** [0.6716, 0.6964] |

#### E. Slice plot

![wine_quality@s42/naive-ensemble](parts/slice_wine_quality@s42_naive-ensemble.png)


### moe

- **holdout RMSE: 0.59823** (winner retrained in 1.88s, cv score of winner: 0.6668)
- cv best RMSE: 0.6668, median: 0.6759, p10: 0.6705
- train: median 0.423s/fold, mean 0.555s, p90 0.890s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.364 |
| `mixture_diversity_lambda` | 0.124 |
| `lambda_l2` | 0.110 |
| `bagging_fraction` | 0.061 |
| `mixture_expert_dropout_rate` | 0.046 |
| `feature_fraction` | 0.038 |
| `mixture_num_experts` | 0.037 |
| `mixture_load_balance_alpha` | 0.037 |
| `mixture_r_smoothing` | 0.034 |
| `mixture_e_step_mode` | 0.026 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 295 | 0.7960 | 0.5029 | 0.6679 |
| gbdt | 96 | 0.9325 | 0.9576 | 0.6726 |
| leaf_reuse | 109 | 0.9376 | 0.7621 | 0.6668 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 470 | 0.8454 | 0.6737 | 0.6668 |
| expert_choice | 30 | 0.9735 | 0.6958 | 0.6715 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| loss_only | 56 | 0.8035 | 0.4731 | 0.6726 |
| em | 346 | 0.8185 | 0.5714 | 0.6668 |
| gate_only | 98 | 1.0034 | 1.0104 | 0.6690 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 349 | 0.7340 | 0.4065 | 0.6668 |
| kmeans_features | 33 | 0.7554 | 0.2503 | 0.6730 |
| tree_hierarchical | 26 | 0.9663 | 0.6546 | 0.6755 |
| random | 40 | 1.1539 | 1.1128 | 0.6690 |
| gmm_features | 19 | 1.4087 | 1.3034 | 0.6738 |
| gmm | 33 | 1.4362 | 1.2222 | 0.6744 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 244 | 0.7752 | 0.5418 | 0.6676 |
| markov | 195 | 0.8956 | 0.7417 | 0.6668 |
| none | 61 | 1.0286 | 0.8652 | 0.6697 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 450 | 0.8265 | 0.6150 | 0.6668 |
| True | 50 | 1.0924 | 1.0478 | 0.6728 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 0.8222 | 0.6276 | 0.6668 |
| True | 32 | 1.3041 | 1.0751 | 0.6841 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.8333 | — | 0.8188 | 0.9176 | **Q3** [3.0, 5.0] |
| `mixture_diversity_lambda` | 0.8015 | 0.9211 | 0.8929 | 0.7969 | **Q4** [0.2, ∞) |
| `mixture_warmup_iters` | 1.0962 | 0.7983 | 0.7188 | 0.8154 | **Q3** [41.0, 43.0] |
| `mixture_balance_factor` | — | 0.8151 | 0.8403 | 0.9047 | **Q2** [2.0, 4.0] |
| `learning_rate` | 0.9196 | 0.7682 | 0.7485 | 0.9760 | **Q3** [0.0787, 0.0965] |
| `num_leaves` | 1.0686 | 0.7019 | 0.8235 | 0.8214 | **Q2** [95.0, 117.0] |
| `max_depth` | 0.9575 | — | 0.7964 | 0.8398 | **Q3** [11.0, 12.0] |
| `min_data_in_leaf` | 0.8000 | 0.7611 | 0.8269 | 1.0011 | **Q2** [7.0, 9.0] |

#### E. Slice plot

![wine_quality@s42/moe](parts/slice_wine_quality@s42_moe.png)


---

## wine_quality@s43  (search X=[5198, 11], holdout n=1299)


### naive-lightgbm

- **holdout RMSE: 0.63426** (winner retrained in 0.24s, cv score of winner: 0.6616)
- cv best RMSE: 0.6616, median: 0.6684, p10: 0.6653
- train: median 0.153s/fold, mean 0.147s, p90 0.215s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.645 |
| `learning_rate` | 0.202 |
| `min_data_in_leaf` | 0.100 |
| `lambda_l1` | 0.013 |
| `max_depth` | 0.013 |
| `bagging_fraction` | 0.012 |
| `feature_fraction` | 0.007 |
| `bagging_freq` | 0.005 |
| `num_leaves` | 0.003 |
| `lambda_l2` | 0.001 |

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

![wine_quality@s43/naive-lightgbm](parts/slice_wine_quality@s43_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 0.62376** (winner retrained in 0.91s, cv score of winner: 0.6569)
- cv best RMSE: 0.6569, median: 0.6621, p10: 0.6594
- train: median 0.619s/fold, mean 0.588s, p90 0.806s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.435 |
| `learning_rate` | 0.292 |
| `min_data_in_leaf` | 0.164 |
| `max_depth` | 0.062 |
| `bagging_fraction` | 0.015 |
| `bagging_freq` | 0.015 |
| `n_models` | 0.007 |
| `num_leaves` | 0.005 |
| `feature_fraction` | 0.004 |
| `lambda_l1` | 0.001 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 0.6651 | 0.0093 | 0.6569 |
| True | 32 | 0.6987 | 0.0202 | 0.6770 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.6740 | 0.6632 | 0.6637 | 0.6682 | **Q2** [0.0858, 0.0975] |
| `num_leaves` | 0.6745 | 0.6642 | 0.6656 | 0.6653 | **Q2** [112.0, 119.0] |
| `max_depth` | 0.6754 | — | — | 0.6649 | **Q4** [11.0, ∞) |
| `min_data_in_leaf` | — | 0.6626 | 0.6642 | 0.6774 | **Q2** [5.0, 7.0] |
| `lambda_l1` | 0.6710 | 0.6656 | 0.6645 | 0.6681 | **Q3** [0.0008, 0.0027] |
| `lambda_l2` | 0.6664 | 0.6649 | 0.6669 | 0.6710 | **Q2** [0.0, 0.0] |
| `feature_fraction` | 0.6664 | 0.6656 | 0.6641 | 0.6729 | **Q3** [0.6391, 0.6632] |
| `bagging_fraction` | 0.6732 | 0.6649 | 0.6639 | 0.6671 | **Q3** [0.898, 0.9211] |

#### E. Slice plot

![wine_quality@s43/naive-ensemble](parts/slice_wine_quality@s43_naive-ensemble.png)


### moe

- **holdout RMSE: 0.61994** (winner retrained in 1.10s, cv score of winner: 0.6534)
- cv best RMSE: 0.6534, median: 0.6655, p10: 0.6575
- train: median 0.582s/fold, mean 0.835s, p90 2.207s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l1` | 0.459 |
| `bagging_fraction` | 0.118 |
| `learning_rate` | 0.091 |
| `mixture_diversity_lambda` | 0.056 |
| `lambda_l2` | 0.041 |
| `mixture_e_step_mode` | 0.037 |
| `mixture_init` | 0.035 |
| `mixture_warmup_iters` | 0.034 |
| `num_leaves` | 0.032 |
| `mixture_gate_type` | 0.024 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 360 | 0.7687 | 0.4723 | 0.6534 |
| gbdt | 112 | 0.9033 | 0.7492 | 0.6617 |
| leaf_reuse | 28 | 1.0538 | 0.7522 | 0.6581 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 387 | 0.7816 | 0.5005 | 0.6534 |
| expert_choice | 113 | 0.9286 | 0.7491 | 0.6617 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| em | 347 | 0.7745 | 0.4892 | 0.6534 |
| gate_only | 82 | 0.8952 | 0.6429 | 0.6544 |
| loss_only | 71 | 0.9189 | 0.7825 | 0.6617 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 18 | 0.7029 | 0.0752 | 0.6619 |
| kmeans_features | 317 | 0.7636 | 0.4626 | 0.6534 |
| random | 64 | 0.7756 | 0.5252 | 0.6694 |
| gmm_features | 65 | 0.8781 | 0.6628 | 0.6617 |
| tree_hierarchical | 18 | 1.1842 | 0.8590 | 0.6573 |
| gmm | 18 | 1.3709 | 1.1878 | 0.6576 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| ema | 412 | 0.7520 | 0.4554 | 0.6534 |
| markov | 64 | 0.9794 | 0.7424 | 0.6722 |
| none | 24 | 1.4547 | 1.0873 | 0.6590 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 468 | 0.7852 | 0.4929 | 0.6534 |
| True | 32 | 1.2475 | 1.1481 | 0.6712 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 469 | 0.8086 | 0.5657 | 0.6534 |
| True | 31 | 0.9091 | 0.6189 | 0.6748 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 1.0380 | 0.8513 | — | 0.7801 | **Q4** [6.0, ∞) |
| `mixture_diversity_lambda` | 0.7532 | 0.7569 | 0.7848 | 0.9643 | **Q1** [None, 0.019] |
| `mixture_warmup_iters` | 0.8312 | 0.7748 | 0.7304 | 0.8798 | **Q3** [36.0, 38.0] |
| `mixture_balance_factor` | 1.0262 | 0.8185 | — | 0.7557 | **Q4** [10.0, ∞) |
| `learning_rate` | 1.0519 | 0.7106 | 0.7531 | 0.7435 | **Q2** [0.142, 0.1736] |
| `num_leaves` | 0.9242 | 0.7592 | 0.8428 | 0.7415 | **Q4** [123.0, ∞) |
| `max_depth` | 0.9512 | 0.8776 | — | 0.7424 | **Q4** [12.0, ∞) |
| `min_data_in_leaf` | 0.7618 | 0.7859 | 0.7103 | 0.9962 | **Q3** [9.0, 14.0] |

#### E. Slice plot

![wine_quality@s43/moe](parts/slice_wine_quality@s43_moe.png)


---

## wine_quality@s44  (search X=[5198, 11], holdout n=1299)


### naive-lightgbm

- **holdout RMSE: 0.60197** (winner retrained in 0.21s, cv score of winner: 0.6750)
- cv best RMSE: 0.6750, median: 0.6832, p10: 0.6787
- train: median 0.136s/fold, mean 0.133s, p90 0.187s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.450 |
| `learning_rate` | 0.205 |
| `min_data_in_leaf` | 0.105 |
| `max_depth` | 0.068 |
| `bagging_fraction` | 0.053 |
| `lambda_l2` | 0.043 |
| `feature_fraction` | 0.039 |
| `bagging_freq` | 0.020 |
| `num_leaves` | 0.012 |
| `lambda_l1` | 0.004 |

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

![wine_quality@s44/naive-lightgbm](parts/slice_wine_quality@s44_naive-lightgbm.png)


### naive-ensemble

- **holdout RMSE: 0.59391** (winner retrained in 0.89s, cv score of winner: 0.6701)
- cv best RMSE: 0.6701, median: 0.6770, p10: 0.6730
- train: median 0.549s/fold, mean 0.522s, p90 0.713s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.471 |
| `learning_rate` | 0.343 |
| `min_data_in_leaf` | 0.114 |
| `bagging_freq` | 0.024 |
| `max_depth` | 0.022 |
| `n_models` | 0.008 |
| `feature_fraction` | 0.006 |
| `lambda_l1` | 0.006 |
| `bagging_fraction` | 0.004 |
| `num_leaves` | 0.004 |

<details><summary>All categorical breakdowns</summary>


**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 469 | 0.6799 | 0.0095 | 0.6701 |
| True | 31 | 0.7134 | 0.0234 | 0.6919 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `learning_rate` | 0.6870 | 0.6784 | 0.6795 | 0.6830 | **Q2** [0.0556, 0.0661] |
| `num_leaves` | 0.6875 | 0.6818 | 0.6798 | 0.6791 | **Q4** [124.0, ∞) |
| `max_depth` | 0.6885 | — | 0.6792 | 0.6817 | **Q3** [10.0, 11.0] |
| `min_data_in_leaf` | — | 0.6769 | 0.6783 | 0.6935 | **Q2** [5.0, 7.0] |
| `lambda_l1` | 0.6846 | 0.6800 | 0.6794 | 0.6838 | **Q3** [0.0257, 0.1005] |
| `lambda_l2` | 0.6871 | 0.6797 | 0.6803 | 0.6806 | **Q2** [0.0016, 0.006] |
| `feature_fraction` | 0.6785 | 0.6794 | 0.6806 | 0.6894 | **Q1** [None, 0.5214] |
| `bagging_fraction` | 0.6879 | 0.6803 | 0.6796 | 0.6800 | **Q3** [0.8354, 0.8616] |

#### E. Slice plot

![wine_quality@s44/naive-ensemble](parts/slice_wine_quality@s44_naive-ensemble.png)


### moe

- **holdout RMSE: 0.59206** (winner retrained in 1.06s, cv score of winner: 0.6670)
- cv best RMSE: 0.6670, median: 0.6808, p10: 0.6716
- train: median 0.511s/fold, mean 0.518s, p90 0.783s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `lambda_l2` | 0.512 |
| `lambda_l1` | 0.205 |
| `mixture_load_balance_alpha` | 0.065 |
| `mixture_gate_type` | 0.029 |
| `feature_fraction` | 0.026 |
| `bagging_fraction` | 0.024 |
| `num_leaves` | 0.021 |
| `mixture_diversity_lambda` | 0.019 |
| `mixture_e_step_mode` | 0.015 |
| `learning_rate` | 0.015 |

#### B. Categorical: clearly best values (p<0.01)

| param | best | mean RMSE | runner-up | Δ | p |
|---|---|---|---|---|---|
| `mixture_init` | **uniform** | 0.6948 (n=18) | gmm_features | Δ +0.0985 | p=2.10e-03 |

<details><summary>All categorical breakdowns</summary>


**`mixture_gate_type`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| leaf_reuse | 444 | 0.8219 | 0.6468 | 0.6670 |
| none | 28 | 0.8770 | 0.4444 | 0.6937 |
| gbdt | 28 | 1.1110 | 1.1038 | 0.6766 |

**`mixture_routing_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| token_choice | 470 | 0.8349 | 0.6722 | 0.6670 |
| expert_choice | 30 | 0.9386 | 0.7053 | 0.6834 |

**`mixture_e_step_mode`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| gate_only | 444 | 0.8167 | 0.6384 | 0.6670 |
| loss_only | 25 | 0.9971 | 0.7088 | 0.6945 |
| em | 31 | 1.0659 | 1.0102 | 0.7016 |

**`mixture_init`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| uniform | 18 | 0.6948 | 0.0306 | 0.6729 |
| gmm_features | 327 | 0.7933 | 0.5579 | 0.6670 |
| gmm | 56 | 0.8885 | 0.8222 | 0.6714 |
| random | 29 | 0.9727 | 0.9490 | 0.6707 |
| kmeans_features | 50 | 0.9886 | 0.7777 | 0.6728 |
| tree_hierarchical | 20 | 1.0630 | 1.1780 | 0.6755 |

**`mixture_r_smoothing`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| none | 309 | 0.7773 | 0.4980 | 0.6682 |
| markov | 93 | 0.9130 | 0.7663 | 0.6725 |
| ema | 98 | 0.9743 | 0.9737 | 0.6670 |

**`mixture_hard_m_step`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| True | 467 | 0.8184 | 0.6407 | 0.6670 |
| False | 33 | 1.1636 | 0.9880 | 0.6887 |

**`extra_trees`**
| value | n | mean RMSE | std | min |
|---|---|---|---|---|
| False | 469 | 0.8264 | 0.6484 | 0.6670 |
| True | 31 | 1.0638 | 0.9640 | 0.6864 |

</details>


#### D. Numeric: quartile mean RMSE (sweet spot)

| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |
|---|---|---|---|---|---|
| `mixture_num_experts` | 0.9387 | 0.8023 | — | 0.8132 | **Q2** [5.0, 6.0] |
| `mixture_diversity_lambda` | 0.8799 | 0.8212 | 0.7624 | 0.9012 | **Q3** [0.2278, 0.2536] |
| `mixture_warmup_iters` | 1.0245 | 0.7620 | 0.7834 | 0.7978 | **Q2** [39.0, 42.0] |
| `mixture_balance_factor` | 1.0421 | 0.8869 | — | 0.8156 | **Q4** [4.0, ∞) |
| `learning_rate` | 1.0033 | 0.8165 | 0.7657 | 0.7791 | **Q3** [0.1601, 0.193] |
| `num_leaves` | 0.9789 | 0.7863 | 0.8575 | 0.7585 | **Q4** [123.0, ∞) |
| `max_depth` | 0.9178 | — | — | 0.8215 | **Q4** [11.0, ∞) |
| `min_data_in_leaf` | 0.8362 | 0.8453 | 0.7715 | 0.9059 | **Q3** [9.0, 12.0] |

#### E. Slice plot

![wine_quality@s44/moe](parts/slice_wine_quality@s44_moe.png)


---

## Overall recommendations

**Categorical settings that are statistically significant winners (p<0.01):**

| dataset | param | best value | Δ vs runner-up | p |
|---|---|---|---|---|
| synthetic@s42 | `mixture_gate_type` | **leaf_reuse** | +0.5112 | 0.00e+00 |
| synthetic@s42 | `mixture_routing_mode` | **token_choice** | +0.7651 | 0.00e+00 |
| synthetic@s42 | `mixture_e_step_mode` | **gate_only** | +0.9726 | 3.40e-05 |
| synthetic@s42 | `mixture_r_smoothing` | **none** | +0.6574 | 0.00e+00 |
| synthetic@s43 | `mixture_gate_type` | **gbdt** | +2.9519 | 0.00e+00 |
| synthetic@s43 | `mixture_routing_mode` | **token_choice** | +0.8162 | 0.00e+00 |
| synthetic@s43 | `mixture_init` | **gmm** | +0.4719 | 3.68e-03 |
| synthetic@s43 | `mixture_r_smoothing` | **none** | +0.4419 | 1.53e-03 |
| synthetic@s44 | `mixture_gate_type` | **gbdt** | +3.1646 | 0.00e+00 |
| synthetic@s44 | `mixture_routing_mode` | **expert_choice** | +1.3851 | 1.00e-06 |
| synthetic@s44 | `mixture_init` | **gmm** | +1.0053 | 5.50e-03 |
| fred_gdp@s42 | `mixture_gate_type` | **none** | +0.0442 | 5.00e-06 |
| fred_gdp@s42 | `mixture_e_step_mode` | **gate_only** | +0.0259 | 2.42e-03 |
| fred_gdp@s43 | `mixture_gate_type` | **none** | +0.0242 | 5.00e-03 |
| fred_gdp@s43 | `mixture_routing_mode` | **expert_choice** | +0.0550 | 0.00e+00 |
| fred_gdp@s44 | `mixture_gate_type` | **none** | +0.0206 | 1.75e-03 |
| fred_gdp@s44 | `mixture_routing_mode` | **expert_choice** | +0.0328 | 2.00e-06 |
| sp500_basic@s42 | `mixture_routing_mode` | **token_choice** | +0.0000 | 2.35e-03 |
| sp500_basic@s42 | `mixture_e_step_mode` | **gate_only** | +0.0000 | 9.01e-03 |
| sp500_basic@s42 | `mixture_r_smoothing` | **ema** | +0.0000 | 1.44e-04 |
| sp500_basic@s43 | `mixture_gate_type` | **gbdt** | +0.0000 | 7.10e-05 |
| sp500_basic@s43 | `mixture_e_step_mode` | **loss_only** | +0.0000 | 2.01e-04 |
| sp500_basic@s43 | `mixture_r_smoothing` | **ema** | +0.0000 | 6.46e-03 |
| sp500_basic@s44 | `mixture_routing_mode` | **token_choice** | +0.0000 | 8.50e-05 |
| sp500_basic@s44 | `mixture_e_step_mode` | **em** | +0.0000 | 2.34e-03 |
| sp500@s42 | `mixture_routing_mode` | **token_choice** | +0.0000 | 0.00e+00 |
| sp500@s42 | `mixture_e_step_mode` | **gate_only** | +0.0000 | 1.00e-06 |
| sp500@s43 | `mixture_gate_type` | **gbdt** | +0.0000 | 0.00e+00 |
| sp500@s43 | `mixture_routing_mode` | **expert_choice** | +0.0000 | 0.00e+00 |
| sp500@s43 | `mixture_e_step_mode` | **loss_only** | +0.0000 | 7.80e-05 |
| sp500@s43 | `mixture_init` | **gmm** | +0.0000 | 7.00e-05 |
| sp500@s43 | `mixture_r_smoothing` | **ema** | +0.0000 | 9.00e-06 |
| sp500@s44 | `mixture_gate_type` | **gbdt** | +0.0000 | 7.10e-05 |
| sp500@s44 | `mixture_e_step_mode` | **em** | +0.0000 | 2.20e-05 |
| sp500@s44 | `mixture_r_smoothing` | **none** | +0.0000 | 2.20e-05 |
| vix@s42 | `mixture_routing_mode` | **expert_choice** | +0.5847 | 1.04e-03 |
| vix@s43 | `mixture_e_step_mode` | **gate_only** | +1.0357 | 2.16e-03 |
| vix@s44 | `mixture_init` | **uniform** | +0.6521 | 6.20e-03 |
| hmm@s42 | `mixture_routing_mode` | **expert_choice** | +0.0164 | 2.50e-05 |
| hmm@s42 | `mixture_e_step_mode` | **gate_only** | +0.0109 | 5.12e-04 |
| hmm@s42 | `mixture_init` | **random** | +0.0255 | 5.00e-06 |
| hmm@s42 | `mixture_r_smoothing` | **none** | +0.0101 | 4.32e-03 |
| hmm@s43 | `mixture_init` | **random** | +0.0088 | 1.80e-05 |
| hmm@s43 | `mixture_r_smoothing` | **none** | +0.0110 | 5.71e-03 |
| houses@s43 | `mixture_routing_mode` | **expert_choice** | +18092.3153 | 5.63e-04 |
| cpu_act@s42 | `mixture_e_step_mode` | **gate_only** | +4.5039 | 6.96e-03 |
| elevators@s43 | `mixture_e_step_mode` | **gate_only** | +0.0027 | 2.84e-03 |
| elevators@s44 | `mixture_gate_type` | **gbdt** | +0.0029 | 4.90e-03 |
| wine_quality@s44 | `mixture_init` | **uniform** | +0.0985 | 2.10e-03 |
