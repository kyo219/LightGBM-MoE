# Comparative Study Report — naive vs naive-ensemble vs MoE

- **Trials per (variant × dataset × seed)**: 500

- **Datasets**: ['houses'], **seeds**: [42, 43, 44]

- **n_splits**: 5 (gap=1), **rounds**: 100, **holdout**: final 20% (never seen by Optuna), **ES**: chronological tail 15% of each train window

- **Build**: commit `4e208d83bace`, lib sha256 `a3c30fcc7fd3…`, package `/home/mokumoku/Develop/LightGBM-MoE/python-package/lightgbm_moe`


---

## Headline: holdout RMSE (chronological tail, evaluated once per seed)

Selection happened on CV inside the search region; this table is the unbiased comparison. `cv_best` is the (optimistic) selection metric, shown for reference.

| Dataset | Variant | holdout RMSE (mean ± std) | cv_best (mean) | crash rate | retrain s |
|---|---|---|---|---|---|
| houses | naive-ensemble | **48816.10629** ± 1005.30253 | 50343.50687 | 0.0% | 0.82 |


## Selection metric per run (CV over the search region)

| Run | Variant | cv best | cv median | median train s/fold | wall s |
|---|---|---|---|---|---|
| houses@s42 | naive-ensemble | 50578.9081 | 50951.9468 | 0.698 | 1632 |
| houses@s43 | naive-ensemble | 50306.1130 | 50702.0156 | 0.728 | 1746 |
| houses@s44 | naive-ensemble | 50145.4995 | 50954.3367 | 0.474 | 1289 |



---

## houses@s42  (search X=[8000, 8], holdout n=2000)


### naive-ensemble

- **holdout RMSE: 50074.59416** (winner retrained in 1.06s, cv score of winner: 50578.9081)
- cv best RMSE: 50578.9081, median: 50951.9468, p10: 50717.1793
- train: median 0.698s/fold, mean 0.649s, p90 0.834s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `learning_rate` | 0.508 |
| `extra_trees` | 0.424 |
| `min_data_in_leaf` | 0.032 |
| `feature_fraction` | 0.019 |
| `num_leaves` | 0.009 |
| `max_depth` | 0.005 |
| `bagging_freq` | 0.002 |
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

![houses@s42/naive-ensemble](slice_houses@s42_naive-ensemble.png)


---

## houses@s43  (search X=[8000, 8], holdout n=2000)


### naive-ensemble

- **holdout RMSE: 48759.66773** (winner retrained in 0.87s, cv score of winner: 50306.1130)
- cv best RMSE: 50306.1130, median: 50702.0156, p10: 50474.1811
- train: median 0.728s/fold, mean 0.695s, p90 0.982s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.592 |
| `learning_rate` | 0.336 |
| `max_depth` | 0.030 |
| `num_leaves` | 0.015 |
| `min_data_in_leaf` | 0.014 |
| `feature_fraction` | 0.007 |
| `bagging_fraction` | 0.006 |
| `bagging_freq` | 0.001 |
| `lambda_l1` | 0.001 |
| `n_models` | 0.000 |

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

![houses@s43/naive-ensemble](slice_houses@s43_naive-ensemble.png)


---

## houses@s44  (search X=[8000, 8], holdout n=2000)


### naive-ensemble

- **holdout RMSE: 47614.05700** (winner retrained in 0.55s, cv score of winner: 50145.4995)
- cv best RMSE: 50145.4995, median: 50954.3367, p10: 50372.8465
- train: median 0.474s/fold, mean 0.513s, p90 0.746s
- finite trials: 500 / 500

#### A. fANOVA importance (top 10)

| param | importance |
|---|---|
| `extra_trees` | 0.585 |
| `learning_rate` | 0.366 |
| `max_depth` | 0.016 |
| `lambda_l2` | 0.008 |
| `feature_fraction` | 0.006 |
| `bagging_fraction` | 0.006 |
| `min_data_in_leaf` | 0.004 |
| `num_leaves` | 0.003 |
| `n_models` | 0.003 |
| `bagging_freq` | 0.002 |

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

![houses@s44/naive-ensemble](slice_houses@s44_naive-ensemble.png)


---

## Overall recommendations

(no categorical settings were universally significant — see per-dataset breakdown)
