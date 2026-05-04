# v0.6 vs v0.8 study comparison

- v0.6 source: `bench_results/study_500_3way_20260502_200635.json`
- v0.8 source: `bench_results/study_v08_500_20260503_014626.json`
- Datasets compared: fred_gdp, hmm, sp500, sp500_basic, synthetic, vix

## 1. RMSE: v0.6 best vs v0.8 best

| Dataset | v0.6 MoE best | v0.8 MoE best | Δ (abs) | Δ% | verdict |
|---|---|---|---|---|---|
| `fred_gdp` | 0.9381 | 0.9387 | +0.0006 | +0.07% | v0.8 **loss** |
| `hmm` | 2.1465 | 2.1560 | +0.0096 | +0.45% | v0.8 **loss** |
| `sp500` | 0.0100 | 0.0099 | -0.0001 | -1.12% | v0.8 **strict win** |
| `sp500_basic` | 0.0100 | 0.0100 | +0.0000 | +0.29% | v0.8 **loss** |
| `synthetic` | 3.6779 | 3.9447 | +0.2668 | +7.25% | v0.8 **loss** |
| `vix` | 2.6745 | 2.6983 | +0.0237 | +0.89% | v0.8 **loss** |

**Summary**: 1/6 strict win, 0/6 tie, 5/6 loss.

(*Strict win* = v0.8 best is more than 1 std (per-fold) below v0.6 best.)

## 2. Did the v0.8 winning config use the new knobs?

Per dataset, value of each v0.8-specific param in the v0.8 `best_params`. `—` = the param was not in best_params (Optuna didn't sample it for the winning trial, e.g. because `mixture_refit_leaves=False` short-circuited the conditional sub-tree).

| Dataset | refit_leaves | refit_trigger | regrow | regrow_per_fire | regrow_mode | init |
|---|---|---|---|---|---|---|
| `fred_gdp` | False | — | — | — | — | uniform |
| `hmm` | False | — | — | — | — | random |
| `sp500` | False | — | — | — | — | random |
| `sp500_basic` | True | every_n | False | — | — | tree_hierarchical |
| `synthetic` | False | — | — | — | — | gmm |
| `vix` | False | — | — | — | — | random |

## 3. Search-space utilization across v0.8 trials

From the per-categorical stats already computed by `comparative_study.py`. Shows how often each value appeared in trials and the mean RMSE conditional on that value.

### `fred_gdp`

- `mixture_init`: 
  - `random` (n=48): mean RMSE = 0.9978, std = 0.057, min = 0.9474
  - `tree_hierarchical` (n=22): mean RMSE = 1.0429, std = 0.0388, min = 0.9749
  - `uniform` (n=399): mean RMSE = 0.9672, std = 0.0423, min = 0.9387
  - `gmm` (n=31): mean RMSE = 1.026, std = 0.0449, min = 0.9754

### `hmm`

- `mixture_init`: 
  - `random` (n=370): mean RMSE = 2.1983, std = 0.0267, min = 2.156
  - `tree_hierarchical` (n=23): mean RMSE = 2.2944, std = 0.042, min = 2.2251
  - `uniform` (n=76): mean RMSE = 2.2181, std = 0.0171, min = 2.1877
  - `gmm` (n=31): mean RMSE = 2.2543, std = 0.0522, min = 2.1585

### `sp500`

- `mixture_init`: 
  - `random` (n=377): mean RMSE = 0.0101, std = 0.0004, min = 0.0099
  - `tree_hierarchical` (n=49): mean RMSE = 0.0101, std = 0.0, min = 0.01
  - `uniform` (n=22): mean RMSE = 0.0101, std = 0.0, min = 0.0101
  - `gmm` (n=52): mean RMSE = 0.0101, std = 0.0, min = 0.01

### `sp500_basic`

- `mixture_init`: 
  - `random` (n=23): mean RMSE = 0.0101, std = 0.0, min = 0.0101
  - `tree_hierarchical` (n=413): mean RMSE = 0.0101, std = 0.0, min = 0.01
  - `uniform` (n=34): mean RMSE = 0.0101, std = 0.0, min = 0.0101
  - `gmm` (n=30): mean RMSE = 0.0101, std = 0.0, min = 0.0101

### `synthetic`

- `mixture_init`: 
  - `random` (n=24): mean RMSE = 6.4586, std = 1.3138, min = 5.2773
  - `tree_hierarchical` (n=29): mean RMSE = 6.2597, std = 2.175, min = 4.4936
  - `uniform` (n=22): mean RMSE = 6.014, std = 0.5338, min = 5.3572
  - `gmm` (n=425): mean RMSE = 5.1731, std = 1.5386, min = 3.9447

### `vix`

- `mixture_init`: 
  - `random` (n=420): mean RMSE = 2.8595, std = 0.6123, min = 2.6983
  - `tree_hierarchical` (n=25): mean RMSE = 4.2655, std = 2.5761, min = 2.7678
  - `uniform` (n=33): mean RMSE = 3.1497, std = 1.0635, min = 2.755
  - `gmm` (n=22): mean RMSE = 5.7972, std = 3.4722, min = 2.8522

## 4. fANOVA importance of v0.8 params

Higher value = the param explains more of the RMSE variance across trials.

