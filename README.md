<img src="docs/logo.png" width=300 />

LightGBM-MoE
============

**A regime-switching / Mixture-of-Experts extension of LightGBM.**

[English](README.md) | [日本語](README.ja.md)

---

## Overview

LightGBM-MoE is a fork of [Microsoft LightGBM](https://github.com/microsoft/LightGBM) that implements **Mixture-of-Experts (MoE) / Regime-Switching GBDT** natively in C++:

```
ŷ(x) = Σₖ gₖ(x) · fₖ(x)
```

- `fₖ(x)`: Expert k's prediction (K regression GBDTs)
- `gₖ(x)`: Gate's routing probability for expert k (softmax)
- `K`: Number of experts (hyperparameter)

## Requirements

- **Python**: 3.10+
- **OS**: Linux (x86_64, aarch64), macOS (Intel, Apple Silicon)
- **Build**: CMake 3.16+ and a C++ compiler (GCC, Clang, or Apple Clang) when installing from source

## Installation

```bash
# From GitHub (builds from source)
pip install git+https://github.com/kyo219/LightGBM-MoE.git

# Editable install for development
git clone https://github.com/kyo219/LightGBM-MoE.git
cd LightGBM-MoE/python-package
pip install -e .
```

## Quick Start

```python
import lightgbm_moe as lgb

params = {
    'boosting': 'mixture',           # Enable MoE mode
    'mixture_num_experts': 3,        # Number of experts (search 2-6 for your data)
    'mixture_gate_type': 'gbdt',     # Best on 4/6 datasets; search 'leaf_reuse' too
    'mixture_routing_mode': 'token_choice',  # Tied 3/3 with 'expert_choice' across datasets
    'objective': 'regression',
    'metric': 'rmse',
    'verbose': -1,
}

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

model = lgb.train(
    params, train_data,
    num_boost_round=500,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50)],
)

# Predictions
y_pred = model.predict(X_test)                      # Weighted mixture
regime = model.predict_regime(X_test)               # Regime index (argmax)
regime_proba = model.predict_regime_proba(X_test)   # Gate probabilities (N, K)
expert_preds = model.predict_expert_pred(X_test)    # Expert predictions (N, K)
```

## How regime estimation works

Each boosting round runs one **soft-EM step** (Jordan & Jacobs 1991, GBDT-ified). After a short warmup (default 5 rounds — gate fits `r_init` first so the first real E-step has a meaningful prior):

```
per round:
  Forward    →  expert preds f_k(x),  gate prior π_k(x)
  E-step     →  r_ik = softmax_k[ log π_k(x_i) + log p(y_i | f_k(x_i), σ_k²) ]
  M-step σ²  →  σ_k² = Σ_i r_ik (y_i − f_k)² / Σ_i r_ik
  M-step f   →  expert k gets +1 tree, gradient = r_ik · ∂L/∂f_k
  M-step g   →  gate gets +K trees, soft CE  ∇z = (p − r) / T
```

`r_ik` is the **posterior probability that sample i was emitted by regime k** under the current mixture, not just a routing weight — this is what makes `predict_regime_proba` well-calibrated rather than just "where the gate happens to point". Three implementation choices distinguish this from a textbook GBDT-MoE — each was a real bug at some point:

| Choice | What it fixes | PR |
|---|---|---|
| Per-expert noise scale `σ_k²` is **estimated each round** (default since #24) | Without it, a hand-tuned `mixture_e_step_alpha` is silently doing what `1/(2σ²)` should — y-scale variant, broken across datasets. With it, the responsibility softmax is the actual Bayesian posterior — y-scale invariant. | [#24](https://github.com/kyo219/LightGBM-MoE/pull/24) |
| **Bias-free prior split**: `π_k = softmax(z/T)` for the E-step, `softmax((z+b)/T)` for the actual routing | DeepSeek "loss-free LB" bias is a routing nudge, not part of the probabilistic model; mixing it into the prior makes the gate spend each iter unlearning the bias the load balancer just added. | [#25](https://github.com/kyo219/LightGBM-MoE/pull/25) |
| **Soft CE gate** with `∇z = (p − r)/T`, Hessian = `K/(K−1) · p(1−p) / T²` (Friedman) | Earlier code argmax'd `r` to hard pseudo-labels (dropped the soft routing signal); missing chain-rule mis-scaled Newton steps by T at non-unit temperatures; missing Friedman factor mis-sized leaf values by `(K−1)/K`. | [#23](https://github.com/kyo219/LightGBM-MoE/pull/23) |

Three pathology guards worth knowing about:

- **Uniform `r` is an EM fixed point** — same gradient → identical trees → `r` stays uniform forever. After init, a deterministic sin perturbation breaks the symmetry (no-op when `r` is non-uniform; only `mixture_init=uniform` actually triggers it). Reproducer: `examples/em_init_sensitivity.py` ([#36](https://github.com/kyo219/LightGBM-MoE/pull/36)).
- **Diversity regularizer sign** was inverted in earlier code (pushed experts *together*). Sign + Huber clip + Hessian damping fix in [#26](https://github.com/kyo219/LightGBM-MoE/pull/26) — the empirical "`mixture_diversity_lambda` is the only universal MoE knob" finding ([Settings that won](#settings-that-won--search-every-knob)) only became real after this.
- **ELBO is logged every 10 iters** (`Σ_i log Σ_k π_k p(y_i | f_k, σ_k²)`). Approximate M-step makes small dips normal; persistent / >5% drops have historically meant high expert dropout, adaptive-LR decoupling experts from EM, or a recent gradient/Hessian regression — flagged loudly so we don't miss it.

A v0.7 opt-in (`mixture_refit_leaves=true`, [#37](https://github.com/kyo219/LightGBM-MoE/issues/37)) makes the M-step refit every existing tree's leaves in closed form against the current `r_ik` before the next tree is appended — recovering classical EM's "free-parameter M-step" within each tree's existing partition. v0.8 ([#41](https://github.com/kyo219/LightGBM-MoE/issues/41)) goes one step further with `mixture_regrow_oldest_trees=true` — discards the oldest trees' splits entirely and rebuilds them via the LightGBM tree learner against current `r`, extending the M-step to the (split, leaf) pair. Both are opt-in; both help bad-init recovery and don't help tuned configs (Optuna with v0.8 features in scope explicitly picks them OFF on 5/6 datasets — see [Limitations § v0.7 leaf-value refit](#v07-leaf-value-refit--opt-in-safety-net-for-bad-inits) and [§ v0.8 partition re-grow](#v08-partition-re-grow--same-safety-net-story-one-level-deeper)).

Full deep dive with code refs: [docs/moe/architecture.md](docs/moe/architecture.md).

## Limitations

Two practical caveats of regime estimation in this model. The first is universal to switching models; the second is specific to this implementation.

### Init dependence (universal to switching models / MoE)

EM converges to a *local* fixed point of `r`, and which fixed point you reach depends on `r_init`. This is the same multimodality that makes GMM and HMM fits init-dependent — not unique to MoE-GBDT. Practical implications:

- If the regime is observable from X, prefer `mixture_init=gmm_features` or `kmeans_features` over the `[X, y]`-aware variants. Cluster on raw features, not on the label.
- `mixture_init` is worth searching over with Optuna — it's not interchangeable across datasets, and there is no universal best init (verified in the 500-trial study).
- **Don't read `r_ik` as "the true regime" of sample i.** It's "the regime *this* run of EM converged to under *this* init". Different runs with different inits are valid alternative explanations of the same data.

### EM updates are additive-only (model-specific)

Classical EM rewrites parameters freely between iterations — at iter t+1 the component means/variances can be radically different from t, and that is how EM escapes one basin and finds another mode. **Here the experts are gradient-boosted ensembles, so each EM round only *appends* one tree per expert** (historical trees are frozen). The expert function `f_k(x)` therefore evolves incrementally (one tree of `learning_rate`-scaled correction per round); `r_ik` shifts only as fast as the new tree's contribution to `loss(y_i, f_k(x_i))` shifts; and the gate's accumulated trees lock in routing toward whichever expert `r` concentrated on early.

Two consequences worth being honest about:

- **Large regime re-assignments don't happen mid-training.** Some refinement happens — that's why the [#26](https://github.com/kyo219/LightGBM-MoE/pull/26) diversity term and the [#36](https://github.com/kyo219/LightGBM-MoE/pull/36) symmetry breaker matter — but you don't see the "EM flipped two components" behavior of free-parameter EM. Snapshotting `model.get_responsibilities()` from a per-iter callback (see the docstring in `python-package/lightgbm_moe/basic.py:4994`) shows `r` smoothing toward its fixed point, not bouncing between modes.
- **This compounds init dependence above** — the basin set by `r_init` is sticky because the model cannot take big jumps in `r` to escape it.

### v0.7 leaf-value refit — opt-in safety net for bad inits

The root-cause fix — **leaf-value refit-on-r-update**, which restores the closed-form M-step on each tree's existing partition structure — landed in v0.7 ([#37](https://github.com/kyo219/LightGBM-MoE/issues/37) / [PR #40](https://github.com/kyo219/LightGBM-MoE/pull/40)). Enable with `mixture_refit_leaves=true` (default `false`); the M-step then rewrites every existing tree's leaves against the current `r_ik` before appending the next tree, instead of leaving them frozen at the `r` of their construction round.

```python
params = {
    "boosting": "mixture",
    ...,
    "mixture_refit_leaves": True,         # opt-in v0.7 behavior
    "mixture_refit_trigger": "elbo",      # always | elbo (default safest) | every_n
    "mixture_refit_decay_rate": 0.0,      # 0.0=full replace, 1.0=no-op
}
```

**Empirical position (verified by [`bench_results/v0_7_acceptance_FINAL.md`](bench_results/v0_7_acceptance_FINAL.md))**:

- **Refit is a safety net for bad inits, not a free improvement on Optuna-tuned configs.** On the 6-dataset 500-trial study with v0.6 winning configs held fixed (per-config ablation), refit on `decay=0.0`/`every_n=10` slightly *degrades* RMSE on 5/6 datasets (+0.4–7.6%) — those configs were tuned to work well *under* append-only EM, and refit breaks the invariants they relied on. Tiny improvement on `fred_gdp` (−0.21%) is within one std.
- **The strongest case is genuinely bad init.** On synthetic two-regime data with `mixture_init=random`, refit recovers from RMSE 1.015 → 0.634 (**−37%**) and `||r_t − r_init||_F` climbs to 0.99 (escapes the bad basin). On the same data with `mixture_init=gmm`, refit *hurts* (0.885 → 1.002, +13%) because the gmm init was already near the right mode and refit drifts away from it. See [`examples/em_refit_regime_evolution.py`](examples/em_refit_regime_evolution.py) for the four-config side-by-side regime tape plots.
- **`elbo` trigger is bit-identical to off** when EM is well-behaved — the trigger condition (>5% ELBO drop) doesn't fire for monotone-converging configs. So `trigger=elbo` is the safe opt-in: zero cost when nothing's wrong, leaf-refit kicks in when it is.
- **`gate_type="leaf_reuse"` is incompatible with refit and is auto-force-disabled at Init** with a one-time warning (refit rewrites expert leaves, but leaf_reuse's gate GBDT stays frozen, producing an asymmetric update — empirically +7% RMSE on `vix`).

Track regime evolution mid-training with `model.get_responsibilities()` from a per-iter callback — `RegimeEvolutionRecorder` in `python-package/lightgbm_moe/viz.py` does this with a 4-panel diagnostic plot.

### v0.8 partition re-grow — same safety-net story, one level deeper

v0.7's leaf refit only updates leaf *values*; the split *structures* chosen against `r_init` stay frozen. v0.8 ([#41](https://github.com/kyo219/LightGBM-MoE/issues/41) / [PR #42](https://github.com/kyo219/LightGBM-MoE/pull/42) for the ELBO trigger fix, [PR #43](https://github.com/kyo219/LightGBM-MoE/pull/43) for partition re-grow) extends the M-step to discard old splits and rebuild trees against current `r_ik` via the LightGBM tree learner — block coordinate ascent on the (split, leaf) pair. Enable on top of `mixture_refit_leaves=true`:

```python
params = {
    "boosting": "mixture",
    ...,
    "mixture_refit_leaves": True,
    "mixture_regrow_oldest_trees": True,        # opt-in v0.8 behavior
    "mixture_regrow_per_fire": 3,               # 3 is the empirical sweet spot
    "mixture_regrow_mode": "replace",           # "replace" | "delete" (ablation)
}
```

**Empirical position (verified by [`bench_results/v0_8_acceptance_FINAL.md`](bench_results/v0_8_acceptance_FINAL.md) and [`bench_results/v0_8_search_FINAL.md`](bench_results/v0_8_search_FINAL.md))** — the same "bad init helps, tuned init doesn't" pattern as v0.7, just one level deeper:

- **Bad-init recovery**: on synthetic + `mixture_init=random`, regrow with `per_fire=3` reaches RMSE 5.11, **−13.0% vs the off baseline (5.87)** — beating v0.7's leaf-refit-alone (5.53, −5.8%) by ~7 percentage points. Partition re-build does what leaf refit can't: when the early-iter splits encode `r_init`'s wrong partition, no leaf-value optimization rescues that — splits have to be re-chosen.
- **Tuned configs gain nothing**: on the 6-dataset per-config bench with v0.6 winning configs held fixed, regrow at the safe `elbo` trigger fires 0/6 (correctly inert — these trajectories don't plateau). At forced `always` trigger, regrow degrades 4/6 by +0.2-7.1%, same shape as v0.7's "refit-always-degrades-tuned-configs" finding. The mechanism is sound; the regime is wrong.
- **The 500-trial Optuna search independently confirms**: with `mixture_refit_leaves` and `mixture_regrow_oldest_trees` added to the search space (and `uniform` added to `mixture_init`), Optuna picked **`refit_leaves=False` on 5/6 datasets** and **`regrow=False` on 6/6 datasets** for the winning configs. fANOVA importance of `refit_leaves` is 0.000-0.008 across all datasets — near-zero variance-explanatory power. **Optuna with budget independently agrees: don't use refit/regrow when tuning is the goal.** Headline numbers drift slightly worse vs v0.6 baseline (5/6 small losses, search-space dilution effect — see `v0_8_search_FINAL.md` for the full breakdown).
- **`gate_type="leaf_reuse"` is auto-disabled** when `mixture_regrow_oldest_trees=true` for the same reason as v0.7 leaf refit (asymmetric update against the frozen leaf-reuse gate GBDT). One-time warning at Init.

Bottom line: v0.7 and v0.8 features are **opt-in safety nets for bad-init recovery, not improvements to dial in for tuned configs**. The 500-trial search makes this an empirical fact, not just our claim. If your problem genuinely needs mode-discovery (unsupervised regime detection where regime structure is unknown a priori), this implementation still under-delivers versus a free-parameter EM with K random restarts even with v0.8 regrow on — partition re-build helps but K restarts is structurally stronger because each restart escapes the entire init basin, not just a few oldest trees within one run.

## When to use MoE

MoE's gating mechanism wins on regime-structured data — but the **honest baseline is not single-model naive LightGBM, it's a K-way ensemble of LightGBMs** with the same total tree budget. Simply averaging K independent models with different seeds gives variance reduction "for free"; the question is whether MoE's learned routing beats that. The holdout-first study below answers it: **MoE clearly wins on `synthetic` (−18.4 % vs the ensemble) — the case where the regime is a deterministic function of `X` — and on `fred_gdp` (−3.5 %), where the holdout window contains a genuine regime shock (COVID quarters). It ties on `hmm` and the `sp500` rows; on `vix` the standard search space loses (+5.8 %) but the v0.8.1 widened space (`--moe-space wide`) recovers a −6.5 % win.** Compute trade-off: MoE costs **3–10 ×** naive single-model time and **0.8–4 ×** naive-ensemble time per trial. Net rule: try MoE when (a) you believe the regime is observable from your features or your evaluation window contains regime shifts, and (b) your wall time has 5–10 × headroom over single-model LightGBM.

## Benchmark — holdout-first study (naive vs naive-ensemble vs MoE)

> **Methodology note (v0.8.1).** This table supersedes the pre-v0.8.1 benchmark, which had
> four defects found in a methodology audit: (1) the headline was the *minimum CV score over
> 500 Optuna trials* — the selection metric itself, a winner's-curse-biased estimate that
> rewards larger search spaces; (2) early stopping validated on the scoring fold; (3) the
> `vix` features leaked the full-sample mean (lookahead for a mean-reverting series — and a
> regime proxy, precisely the signal a gate exploits); (4) `sp500`/`vix` targets were
> misaligned by one step. All four are fixed; the old report remains at
> [`bench_results/study_500_report.md`](bench_results/study_500_report.md) for archaeology.
> Two headline results changed: **the old `vix` win (−6.9 %) is retracted** (it was an
> artifact of the leak + selection bias), and **`fred_gdp` flipped from a loss to a win**
> (the old protocol never evaluated the COVID-containing tail).

Protocol: the final 20 % of every series is a **chronological holdout never seen by Optuna**.
Hyperparameters are selected by expanding-window time-series CV (5 folds, 1-step embargo,
early stopping on the chronological tail of each training window — never on the scoring
fold) over the first 80 %; the CV winner is retrained once and scored once on the holdout.
300 Optuna trials per (variant × dataset × seed), 3 seeds (synthetic/hmm redraw data,
TPE + models reseed), reported as holdout RMSE mean ± std. Deterministic per seed
(`--n-jobs 1`); build provenance (git commit, lib sha256, dataset sha256s) is recorded in
the output JSON. Full report: [`bench_results/meth2_v081/`](bench_results/meth2_v081/).

The third variant — **`naive-ensemble`** — is a K-way (K ∈ {2, 3, 4}) seed-ensemble of
standard LightGBMs that share hyperparameters but diverge per-member via the `seed`
master-seed override. Same total tree budget as MoE (K × `num_boost_round`), same Optuna
search space as `naive-lightgbm` plus K. It is the fair ablation for "is gating doing real
work, or would any K-way ensemble suffice?" — see [PR #33](https://github.com/kyo219/LightGBM-MoE/pull/33).

### Regime datasets (holdout RMSE, mean ± std over 3 seeds)

| Dataset | Shape | naive | ensemble | MoE | MoE vs naive | **MoE vs ensemble** *(the fair test)* |
|---|---|---|---|---|---|---|
| `synthetic` | 2000 × 5 | 4.812 ± 0.340 | 4.706 ± 0.372 | **3.842 ± 0.290** | −20.2 % | **−18.4 %** 🎯 |
| `fred_gdp` | 311 × 12 | 1.528 ± 0.020 | 1.543 ± 0.007 | **1.489 ± 0.014** | −2.6 % | **−3.5 %** 🎯 |
| `hmm` | 2000 × 5 | 2.218 ± 0.244 | 2.206 ± 0.242 | 2.210 ± 0.243 | −0.3 % | +0.2 % *(tie)* |
| `vix` | 3762 × 13 | 1.794 ± 0.063 | **1.752 ± 0.044** | 1.854 ± 0.173 | +3.3 % | +5.8 % *(ensemble wins)* |
| `sp500_basic` (13 feat) | 3761 × 13 | 0.01104 | 0.01104 | 0.01102 | −0.2 % | −0.1 % *(noise floor)* |
| `sp500` (28 feat) | 3711 × 28 | 0.01105 | 0.01105 | 0.01108 | +0.3 % | +0.3 % *(noise floor)* |

The pattern is sharper than the old report's: MoE wins where regime structure is
**strong and observable** (`synthetic` — regime is a deterministic function of X) or where
the **evaluation window itself contains a regime break** (`fred_gdp` holdout spans COVID).
It ties where the regime is weakly observable (`hmm` — two noisy proxy columns) or the
target is at the noise floor (`sp500`). Under this standard search space MoE loses `vix`
now that the regime-proxy leak is gone — but the widened space recovers a legitimate
`vix` win (see [Settings that won](#settings-that-won--search-every-knob)).

### Non-regime control datasets (Grinsztajn et al. 2022 regression track)

Four standard i.i.d. tabular regression datasets (OpenML: `houses` 537, `cpu_act` 197,
`elevators` 216, `wine_quality` 287; numeric features, seeded shuffle, 10k-row cap). These
have **no regime structure** — the control group: MoE should at best tie naive here, and a
win would mean the "regime" story is confounded with generic extra capacity.

| Dataset | Shape | naive | ensemble | MoE | MoE vs naive | **MoE vs ensemble** |
|---|---|---|---|---|---|---|
| `houses` | 10000 × 8 | 49805 ± 962 | 48924 ± 849 | 48931 ± 672 | −1.8 % | **+0.0 %** |
| `cpu_act` | 8192 × 21 | 2.513 ± 0.32 | 2.419 ± 0.28 | 2.425 ± 0.29 | −3.5 % | **+0.3 %** |
| `elevators` | 10000 × 18 | 0.002384 | 0.002302 | 0.002298 | −3.6 % | **−0.2 %** |
| `wine_quality` | 6497 × 11 | 0.6155 ± 0.016 | 0.6082 ± 0.012 | 0.6192 ± 0.010 | +0.6 % | **+1.8 %** |

This is the attribution result the regime table needs: on non-regime data MoE beats naive
by 2–4 % — **but so does the capacity-matched seed-ensemble, and MoE never beats the
ensemble here** (within ±0.5 % on three of four, one small loss). MoE's advantage over
naive on i.i.d. tabular data is fully explained by generic K-way capacity; learned routing
adds nothing without regime/cluster structure to route on. Combined with the regime table
(where MoE beats the *same* capacity-matched ensemble by 18.4 % on `synthetic` and 3.5 % on
`fred_gdp`), the regime-routing claim survives its control.

### Compute cost (median train s/fold across trials, v0.8.1)

| Dataset | naive | ensemble | MoE | MoE / naive | MoE / ensemble |
|---|---|---|---|---|---|
| `synthetic` | 0.110 | 0.321 | 0.327 | 3.0 × | 1.0 × |
| `fred_gdp` | 0.014 | 0.031 | 0.133 | 9.5 × | 4.3 × |
| `hmm` | 0.031 | 0.093 | 0.177 | 5.7 × | 1.9 × |
| `vix` | 0.049 | 0.149 | 0.285 | 5.8 × | 1.9 × |
| `sp500_basic` | 0.022 | 0.057 | 0.143 | 6.5 × | 2.5 × |
| `sp500` | 0.023 | 0.133 | 0.103 | 4.5 × | 0.8 × |

> **The sp500 pair is a controlled feature-engineering ablation on the same raw series**:
> only the feature set changes between rows, identical CV / Optuna budget / seed. All three
> variants are essentially tied at the noise floor under both feature sets — the
> next-day-log-return objective appears irreducible under any architecture we've tried.

### Datasets

Regime datasets span the regime-switching applicability spectrum: one MoE-ideal synthetic,
three real-world series (canonical references in the regime-switching literature), and one
HMM with known ground-truth labels. All time-series generators follow the audited
alignment convention: **row t's features use information available at time t only; the
target is the value at t + 1** (verified by `tests/python_package_test/test_benchmark_methodology.py`).

#### `synthetic` — feature-driven regime, MoE-ideal case (2000 × 5)

- **Source**: internal generator in this repo (`generate_synthetic_data` in `examples/benchmark.py`).
- **Construction**: five i.i.d. Gaussian features; the regime is a deterministic function of `X`, so the gate can route perfectly:

  ```
  regime = (0.5·X1 + 0.3·X2 − 0.2·X3 > 0)
  y | regime=0 :   5·X0 + 3·X0·X2 + 2·sin(2·X3) + 10  +  ε
  y | regime=1 :  −5·X0 − 2·X1²   + 3·cos(2·X4) − 10  +  ε     ε ~ N(0, 0.5²)
  ```

  The two regimes use *opposite-sign* coefficients on the same features — a single GBDT is forced to average them, which is exactly the failure mode MoE is built to avoid.

#### `fred_gdp` — US Real GDP, Hamilton-style MS-AR(4) (~310 × 12)

- **Source**: FRED series [`GDPC1`](https://fred.stlouisfed.org/series/GDPC1) — Real Gross Domestic Product, Chained 2017 Dollars, Quarterly, Seasonally Adjusted Annual Rate (BEA via FRED, no auth, CSV endpoint).
- **Methodology cite**: Hamilton, J. D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle.* **Econometrica** 57(2), 357-384. <https://www.jstor.org/stable/1912559>.
- **Construction**: target is the quarterly growth rate `100·Δlog(GDP)`; features are 4 lags of growth (Hamilton's MS-AR(4)) plus engineered MA / volatility / regime-proxy features. The regime (expansion / recession) is genuinely latent — there is no oracle column. The chronological holdout contains the COVID quarters, which is exactly where MoE's win comes from. Generator: `generate_fred_gdp_data`.

#### `sp500_basic` & `sp500` — S&P 500 daily log returns, two feature sets

The S&P 500 series is included in **two parallel configurations** to test how MoE's lift scales with feature richness:

- **`sp500_basic`** (~3760 × 13): minimal — lagged returns at {1, 2, 3, 5, 10} (lag-1 = today's return) plus the standard MA / rolling-vol / regime-proxy block (`generate_sp500_basic_data`).
- **`sp500`** (~3710 × 28): practitioner-grade — multi-horizon lags, momentum, realized volatility + ratio, MA crossovers, RSI(14)/RSI(30), rolling skew/kurtosis, Bollinger z-score, drawdowns, positive-day fractions (`generate_sp500_data`).

Common to both:

- **Source**: Yahoo Finance, symbol [`^GSPC`](https://finance.yahoo.com/quote/%5EGSPC/history) (default range `2010-01-01` to `2024-12-31`).
- **Index methodology**: [S&P Dow Jones Indices, S&P 500](https://www.spglobal.com/spdji/en/indices/equity/sp-500/).
- **Target**: next-day log return (deliberately hard predictive setup).
- **Regime structure**: latent (low-vol vs high-vol periods).

#### `vix` — CBOE Volatility Index, daily level (~3760 × 13)

- **Source**: Yahoo Finance, symbol [`^VIX`](https://finance.yahoo.com/quote/%5EVIX/history) (same date range as `sp500`).
- **Index methodology**: [CBOE VIX](https://www.cboe.com/tradable_products/vix/).
- **Construction**: target is next-day VIX level. Features: lagged VIX at lags {1, 2, 3, 5, 10} plus MA / rolling-volatility features computed on a **causally demeaned** series (expanding past-only mean — the pre-v0.8.1 version demeaned with the full-sample mean, which leaked "above/below the all-time mean" into the features and inflated MoE's score; that win is retracted). Generator: `generate_vix_data`.

#### `hmm` — 3-state Gaussian HMM with known regime labels (2000 × 5)

- **Source**: internal generator in this repo (`generate_hmm_data`).
- **Methodology cite**: Rabiner, L. R. (1989). *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition.* **Proceedings of the IEEE** 77(2), 257-286. <https://www.cs.ubc.ca/~murphyk/Bayes/rabiner.pdf>.
- **Construction**: hidden state evolves as a 3-state Markov chain with persistent (95 % diagonal) transitions; emissions are Gaussian with well-separated means `{−3, 0, +3}` and varying scales `{0.4, 0.7, 1.0}`. Two of the five feature columns carry a weak linear signal of the hidden state; the rest are pure noise. **Returns ground-truth regime labels** — usable for measuring regime recovery, not just RMSE.

#### Control datasets — `houses`, `cpu_act`, `elevators`, `wine_quality`

- **Source**: OpenML v1 originals (data_ids 537 / 197 / 216 / 287), from the regression-on-numerical-features track of Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). *Why do tree-based models still outperform deep learning on typical tabular data?* **NeurIPS Datasets & Benchmarks**.
- **Preprocessing** (ours, documented): numeric columns only, NaN rows dropped, seeded shuffle, 10k-row cap (tabular-benchmark "medium-sized" convention). Rows carry no temporal meaning, so the chronological splits behave like random splits here.

### Caching

Real-world fetches (FRED, yfinance, OpenML) are cached under `examples/data_cache/`
(gitignored); sha256 checksums of every cache file are recorded in each study's output
JSON, so any run can be tied to the exact bytes it saw. First run pulls from the network;
subsequent runs are offline.

### Settings that won — search every knob

Re-measured under the fixed protocol with the widened space (`--moe-space wide`, 500
trials × 3 seeds; the pre-v0.8.1 per-knob table came from the retracted methodology and is
superseded). Two headline effects of the wider space:

- **`vix` becomes a legitimate MoE win: −6.5 % vs naive and −5.0 % vs the
  capacity-matched seed-ensemble** (1.653 ± 0.03 vs 1.768 ± 0.09 / 1.739 ± 0.05, all at the
  same 500-trial budget) — the standard space had MoE *losing* vix. The winning configs are
  driven by the previously-frozen knobs (gate temperature ≈ 2.4–2.9 with annealing, expert
  dropout 0.1–0.2, soft M-step), **not** by the new init choices. So the retracted
  pre-v0.8.1 vix win is re-established under the leak-free holdout protocol — by tuning
  knobs the old study never searched. `synthetic` holds at −19.7 % vs the ensemble and
  `fred_gdp` at −2.4 % under the wide space.
- **`hmm` does not improve** (+1.8 % vs naive) — the weakly-observable-regime case remains
  MoE's honest limitation, wider search or not.

Across the 24 winning configs (8 datasets × 3 seeds):

| Knob | Finding |
|---|---|
| `mixture_diversity_lambda` | active (> 0.02) in **23 / 24** winners — still the one near-universal knob |
| `mixture_expert_dropout_rate` | active in **21 / 24** (typically 0.05–0.2) — was frozen at 0 pre-v0.8.1 |
| `mixture_load_balance_alpha` | active in **22 / 24** — was frozen at 0 |
| `mixture_gate_entropy_lambda` | active in **17 / 19** gate-ful winners — was frozen at 0 |
| gate temperature | `T_init > 1.5` in **12 / 19** — the old fixed `T = 1.0` was leaving accuracy on the table |
| `mixture_refit_leaves` | **ON in 15 / 24 winners** — updates the v0.8 finding ("Optuna declines refit on 5/6 datasets"): after the v0.8.1 stale-score fix and ELBO-trigger cooldown, leaf refit is an ordinary useful knob rather than a bad-init-only safety net |
| `mixture_init` / `mixture_gate_type` / `mixture_routing_mode` / K | no universal winner (gmm 8, uniform 6, random 5, feature-space inits 4, tree 1 / gbdt 11, leaf_reuse 8, none 5 / token 14, expert 10 / K spread 2–6) — **search them** |

```bash
# Full headline study (holdout protocol, 3 seeds; deterministic per seed)
python examples/comparative_study.py --trials 300 --seeds 42,43,44 \
    --out bench_results/study_holdout.json

# Wide MoE search space (adds gmm_features/kmeans_features inits, K up to 6,
# gate temperature annealing, entropy lambda, dropout, load-balance alpha)
python examples/comparative_study.py --trials 500 --seeds 42,43,44 \
    --variants naive-lightgbm,moe --moe-space wide \
    --out bench_results/study_wide.json

# Smoke test (~2 min)
python examples/comparative_study.py --trials 10 --seeds 42 \
    --datasets synthetic,fred_gdp --out bench_results/smoke.json

# A/B two builds of the C++ core with the same protocol
LGBM_MOE_PACKAGE_DIR=/path/to/other/python-package \
    python examples/comparative_study.py --trials 300 --seeds 42,43,44 \
    --out bench_results/study_other_build.json
```

## Documentation

| Topic | Doc |
|---|---|
| Full parameter reference (MoE core, Gate, Smoothing, Prediction APIs) | [docs/moe/parameters.md](docs/moe/parameters.md) |
| Optuna search templates | [docs/moe/optuna-recipes.md](docs/moe/optuna-recipes.md) |
| 500-trial / 5-dataset benchmark methodology & per-dataset recommendations | [docs/moe/benchmark.md](docs/moe/benchmark.md) |
| Per-expert hyperparameters & role-based recipe | [docs/moe/per-expert-hp.md](docs/moe/per-expert-hp.md) |
| Expert Choice routing | [docs/moe/advanced-routing.md](docs/moe/advanced-routing.md) |
| Progressive training (EvoMoE) & gate temperature annealing | [docs/moe/advanced-progressive.md](docs/moe/advanced-progressive.md) |
| Expert collapse prevention & `diagnose_moe` | [docs/moe/advanced-collapse.md](docs/moe/advanced-collapse.md) |
| SHAP analysis for MoE components | [docs/moe/shap.md](docs/moe/shap.md) |
| `int8` & `use_quantized_grad` compatibility (8-axis matrix) | [docs/moe/int8-compat.md](docs/moe/int8-compat.md) |
| Architecture & EM-loop deep dive (incl. v0.7 leaf-refit §2.7) | [docs/moe/architecture.md](docs/moe/architecture.md) |
| v0.8 partition re-grow design (math + impl plan) | [docs/v0.8/partition_regrow_design.md](docs/v0.8/partition_regrow_design.md) |
| v0.7 leaf-refit acceptance bench | [bench_results/v0_7_acceptance_FINAL.md](bench_results/v0_7_acceptance_FINAL.md) |
| v0.8 acceptance bench (per-config + bad-init recovery) | [bench_results/v0_8_acceptance_FINAL.md](bench_results/v0_8_acceptance_FINAL.md) |
| v0.8 500-trial search (Optuna picks refit/regrow OFF for tuned configs) | [bench_results/v0_8_search_FINAL.md](bench_results/v0_8_search_FINAL.md) |

## License

MIT license. Based on [Microsoft LightGBM](https://github.com/microsoft/LightGBM).
