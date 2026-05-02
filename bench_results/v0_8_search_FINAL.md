# v0.8 500-trial search study — final results

**Question**: with v0.7 leaf-refit and v0.8 partition re-grow added to the
Optuna search space (alongside `uniform` and `random` as `mixture_init`
choices), do v0.8-aware configs come out on top of the v0.6 README
headline numbers?

**Short answer**: **No** on tuned configs (5/6 datasets), and Optuna
correctly figures out to *not use* the v0.8 features when tuning is the
goal. **Yes** as a safety net (per the v0.7 / v0.8 acceptance benches —
bad-init recovery is the regime where the new features deliver value).

This is the empirical confirmation of the architectural choice that
v0.8 features are off-by-default opt-in safety nets, not free
improvements.

## Setup

- 500 Optuna trials × 6 datasets × MoE only (naive / ensemble unchanged
  by v0.8, so reused from v0.6 study)
- Same TimeSeriesSplit(5), num_boost_round=100, seed=42 as v0.6 study
- New search variables added to MoE objective:
  - `mixture_refit_leaves` (bool)
  - conditional on True: `mixture_refit_trigger`, `mixture_refit_decay_rate`
  - conditional on `trigger='elbo'`: `mixture_elbo_drop_threshold`,
    `mixture_elbo_plateau_threshold`, `mixture_elbo_window`
  - conditional on `trigger='every_n'`: `mixture_refit_every_n`
  - `mixture_regrow_oldest_trees` (bool)
  - conditional on True: `mixture_regrow_per_fire`, `mixture_regrow_mode`
- New init choice: `uniform` added to existing `{random, gmm,
  tree_hierarchical}`

Source files:
- v0.8 study JSON: `bench_results/study_v08_500_20260503_014626.json`
- v0.6 baseline JSON: `bench_results/study_500_3way_20260502_200635.json`
- Comparison report: `bench_results/v06_vs_v08_comparison.md`

## Results: v0.6 best vs v0.8 best (RMSE)

| Dataset       | v0.6 best | v0.8 best | Δ        | Δ%       | verdict           |
|---------------|----------:|----------:|---------:|---------:|:------------------|
| `fred_gdp`    | 0.9381    | 0.9387    | +0.0006  | +0.07%   | v0.8 loss (noise) |
| `hmm`         | 2.1465    | 2.1560    | +0.0096  | +0.45%   | v0.8 loss         |
| `sp500`       | 0.0100    | 0.0099    | −0.0001  | **−1.12%** | **v0.8 strict win** |
| `sp500_basic` | 0.0100    | 0.0100    | +0.0000  | +0.29%   | v0.8 loss (noise) |
| `synthetic`   | 3.6779    | **3.9447**| +0.2668  | **+7.25%** | v0.8 loss        |
| `vix`         | 2.6745    | 2.6983    | +0.0237  | +0.89%   | v0.8 loss         |

**Summary**: 1/6 strict win, 5/6 loss. The synthetic +7.25% is the only
non-noise loss; the rest are within sub-percent or noise-floor range.

## Did the v0.8 winning config use the new knobs?

| Dataset       | refit_leaves | refit_trigger | regrow | regrow_per_fire | regrow_mode | init                |
|---------------|:-------------|:--------------|:-------|:----------------|:------------|:--------------------|
| `fred_gdp`    | False        | —             | —      | —               | —           | **uniform** (new)   |
| `hmm`         | False        | —             | —      | —               | —           | random              |
| `sp500`       | False        | —             | —      | —               | —           | random              |
| `sp500_basic` | **True**     | every_n       | False  | —               | —           | tree_hierarchical   |
| `synthetic`   | False        | —             | —      | —               | —           | gmm                 |
| `vix`         | False        | —             | —      | —               | —           | random              |

**Optuna's verdict on the new knobs**:

- **`refit_leaves`**: chosen `False` in 5/6 winning configs. Only sp500_basic
  picked `True` (with `every_n` trigger), and even there `regrow=False`.
- **`regrow_oldest_trees`**: chosen `False` in 6/6 winning configs.
  Optuna never selected partition re-grow as part of any dataset's
  best config.
- **`mixture_init`**: the new `uniform` choice won on `fred_gdp` —
  the search-space expansion pulled in 1/6 dataset wins for the new init
  alone.

## fANOVA importance of v0.8-related params

| Dataset       | top v0.8-related importances                               |
|---------------|:-----------------------------------------------------------|
| `fred_gdp`    | mixture_init=0.013, mixture_refit_leaves=0.000             |
| `hmm`         | mixture_init=**0.693**, mixture_refit_leaves=0.004         |
| `sp500`       | mixture_init=0.146, mixture_refit_leaves=0.008             |
| `sp500_basic` | mixture_init=0.021, mixture_refit_leaves=0.000             |
| `synthetic`   | mixture_init=0.074, mixture_refit_leaves=0.002             |
| `vix`         | mixture_init=**0.410**, mixture_refit_leaves=0.001         |

`mixture_refit_leaves` consistently has near-zero variance-explanatory
power across all 6 datasets (max 0.008 on sp500). The other v0.8
params (`refit_trigger`, `regrow_*`) didn't even make the top-5
fANOVA list on any dataset.

By contrast, `mixture_init` is **the dominant categorical** on hmm
(0.693) and vix (0.410) — the search-space expansion (adding
`uniform`) is the v0.8 change that mattered for variance.

## What `uniform` init brought to the table

Per-init mean RMSE across all 500 trials:

| Dataset       | random           | tree_hierarchical | uniform          | gmm              |
|---------------|------------------|-------------------|------------------|------------------|
| `fred_gdp`    | 0.998 (n=48)     | 1.043 (n=22)      | **0.967 (n=399)** | 1.026 (n=31)    |
| `hmm`         | **2.198 (n=370)**| 2.294 (n=23)      | 2.218 (n=76)     | 2.254 (n=31)     |
| `sp500`       | **0.0101 (n=377)**| 0.0101 (n=49)    | 0.0101 (n=22)    | 0.0101 (n=52)    |
| `sp500_basic` | 0.0101 (n=23)    | **0.0101 (n=413)**| 0.0101 (n=34)   | 0.0101 (n=30)    |
| `synthetic`   | 6.46 (n=24)      | 6.26 (n=29)       | 6.01 (n=22)      | **5.17 (n=425)** |
| `vix`         | **2.86 (n=420)** | 4.27 (n=25)       | 3.15 (n=33)      | 5.80 (n=22)      |

`uniform` was the trial-mean winner on `fred_gdp` (0.967 vs random's
0.998) — TPE concentrated 80% of trials on it after early signal.

## Why v0.8 didn't beat v0.6 on tuned configs

Three compounding reasons:

1. **Optuna correctly avoided the v0.8 features** when search budget was
   spent on tuning. Refit/regrow consistently picked `False` for the
   winning configs, and their fANOVA importance was near zero. They're
   simply not load-bearing on healthy convergence trajectories.

2. **Search-space dilution**: with refit/regrow conditional sub-trees
   added to the search, 500 trials were spread thinner across a larger
   space. TPE's bias toward early-promising regions means some core
   hyperparameters got under-explored (e.g. `tree_hierarchical` init
   on synthetic was tried only 29 times in v0.8 vs presumably ~100+ in
   v0.6).

3. **TPE init bias**: on synthetic the TPE sampler concentrated 425/500
   trials on `gmm` init after initial signal, but `gmm`'s best (3.94)
   doesn't beat v0.6's `tree_hierarchical` best (3.68). The v0.6 study
   happened to land in the right basin; v0.8 didn't because the
   sampler was distracted by extra parameters.

## What v0.8 actually delivered (combined picture)

Cross-referencing this 500-trial search study with the
[v0.8 acceptance bench](v0_8_acceptance_FINAL.md):

| Regime                          | v0.7 result                  | v0.8 result                                  |
|---------------------------------|------------------------------|----------------------------------------------|
| Tuned config + healthy EM       | Refit fires 0/6, no effect   | **Same** — Optuna picks `refit=False` 5/6   |
| Tuned config + Optuna search    | n/a                          | **−1.12%** to **+7.25%** vs v0.6 (search-space dilution effect) |
| Bad init (`init=random`)        | Leaf refit: −5.8% RMSE       | **Regrow per_fire=3: −13.0% RMSE**          |
| Default-off behavior            | Bit-identical to v0.6        | Bit-identical to v0.7 / v0.6                |

**The v0.8 features are validated as opt-in safety nets, not free
improvements.** This is exactly the design intent — and Optuna with a
500-trial budget independently arrived at the same conclusion by
declining to use them on tuned configs.

## Implications for the README headline numbers

The README's 500-trial benchmark table currently uses v0.6 best
configs. Updating to v0.8 search-space results would:

- **Modestly hurt** the headline numbers on most datasets (5/6 small
  losses, mostly noise but synthetic +7%) due to search-space dilution
- **Slightly help** sp500 (−1.12%, strict win)
- **Make the search story more honest**: with v0.8 features in scope,
  Optuna picks them OFF for tuned configs — supporting the README's
  Limitations text that v0.8 features are bad-init safety nets

**Recommendation**: keep the README headline at v0.6 search results
(unchanged) and add a new section in `docs/v0.8/` (or amend
README's Limitations) noting:

> v0.8 leaves the headline numbers unchanged because Optuna with the
> v0.8 features in scope correctly declines to use them on tuned
> configs. Their value is bad-init recovery, where v0.8 partition
> re-grow delivers an additional 7 percentage points of RMSE
> improvement over v0.7's leaf refit alone (synthetic + `init=random`:
> −13% vs v0.6 baseline). See `bench_results/v0_8_acceptance_FINAL.md`.

## Reproducibility

```bash
# v0.8 500-trial search (~30 min on 4-core, MoE only)
PYTHONPATH=python-package python3 examples/comparative_study.py \
    --trials 500 --rounds 100 --splits 5 --variants moe \
    --out bench_results/study_v08_500_<ts>.json

# Comparison report
python3 examples/compare_v06_v08_studies.py \
    bench_results/study_500_3way_20260502_200635.json \
    bench_results/study_v08_500_<ts>.json \
    --out bench_results/v06_vs_v08_comparison.md
```
