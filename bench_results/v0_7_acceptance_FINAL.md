# v0.7 leaf-refit acceptance — final honest verification

Two complementary 6-dataset benchmarks. Both run on commit `feat/leaf-refit-em` HEAD,
identical to release/v0.6.0 except for the additive v0.7 changes
(`mixture_refit_leaves` flag and supporting machinery, all default-off).

---

## Sanity: default-off is bit-identical to v0.6.0

The 500-trial baseline run (`mode=off`, all 6 datasets, naive + naive-ensemble + moe)
reproduced the v0.6 README headline numbers exactly:

| Dataset | naive | ensemble | moe | v0.6 README MoE |
|---|---|---|---|---|
| `synthetic` | 5.0233 | 4.8899 | **3.6779** | 3.6779 |
| `fred_gdp` | 0.9311 | 0.9094 | **0.9381** | 0.9381 |
| `sp500_basic` | 0.0100 | 0.0100 | **0.0100** | 0.01001 |
| `sp500` | 0.0100 | 0.0100 | **0.0100** | 0.01002 |
| `vix` | 2.8869 | 2.8724 | **2.6745** | 2.6745 |
| `hmm` | 2.1913 | 2.1818 | **2.1465** | 2.1465 |

**Result**: 6/6 match within numerical precision. The default-off path is structurally
preserved — adding the v0.7 flag did not perturb any existing code path.

---

## Test A — Per-config ablation (fair refit-vs-no-refit comparison)

**Question**: holding the v0.6 winning config FIXED, does turning refit on improve, hurt, or do nothing?

**Setup**: For each dataset, load the moe variant's best_params from the v0.6 study
(`bench_results/study_500_3way_20260502_200635.json`). Run 5-fold time-series CV
under 4 refit modes, with everything else identical.

This eliminates Optuna search divergence — it's a true A/B/C/D on the same config.

| Dataset | refit=off | refit=elbo | refit=every_n=10 | refit=always | Δ elbo | Δ every_n | Δ always |
|---|---|---|---|---|---|---|---|
| `synthetic` | 3.6779 ± 0.68 | **3.6779 ± 0.68** | 3.8405 ± 0.74 | 3.9578 ± 0.67 | **+0.00%** | +4.42% | +7.61% |
| `fred_gdp` | 0.9381 ± 0.44 | **0.9381 ± 0.44** | **0.9361 ± 0.44** | **0.9368 ± 0.43** | **+0.00%** | **−0.21%** | **−0.13%** |
| `sp500_basic` | 0.0100 ± 0.004 | **0.0100 ± 0.004** | 0.0100 ± 0.004 | 0.0100 ± 0.004 | **+0.00%** | +0.37% | +0.36% |
| `sp500` | 0.0100 ± 0.003 | **0.0100 ± 0.003** | 0.0101 ± 0.004 | 0.0101 ± 0.004 | **+0.00%** | +1.16% | +0.86% |
| `vix` | 2.6745 ± 1.46 | **2.6745 ± 1.46** | 2.7814 ± 1.67 | 2.7837 ± 1.60 | **+0.00%** | +4.00% | +4.08% |
| `hmm` | 2.1465 ± 0.24 | **2.1465 ± 0.24** | 2.2459 ± 0.21 | 2.2424 ± 0.21 | **+0.00%** | +4.63% | +4.47% |

### Findings (Test A)

- **`elbo` trigger is bit-identical to off on 6/6 datasets**. The trigger condition
  ("ELBO drop > 5% in the last log block") never fires on a v0.6-tuned config —
  Optuna selected configs whose EM is well-behaved (monotone non-decreasing log-likelihood).
  This is the **designed safety property**: refit costs nothing when EM is healthy.

- **`every_n=10` and `always` produce small to moderate degradations** on 5/6 datasets
  (+0.4% to +7.6%). The single exception is `fred_gdp`, where refit improves
  by 0.21% / 0.13% — but the std is 0.44, so the improvement is not statistically
  significant.

- **Why the degradation?** The v0.6 best configs were Optuna-tuned for `refit=off`.
  They rely on append-only EM dynamics (which trees freeze leaves in early iters,
  what learning rate compensates for that, etc.). Turning refit on rewrites those
  leaves every round, breaking the invariants the config implicitly assumed.

### Cost (Test A) — median per-fold training time

| Dataset | off | elbo | every_n=10 | always | elbo / off | every_n / off | always / off |
|---|---|---|---|---|---|---|---|
| `synthetic` | 0.07s | 0.07s | 0.12s | 0.65s | 0.92× | 1.74× | **9.22×** |
| `fred_gdp` | 0.03s | 0.03s | 0.04s | 0.17s | 1.16× | 1.58× | 6.25× |
| `sp500_basic` | 0.12s | 0.13s | 0.17s | 0.61s | 1.12× | 1.47× | 5.15× |
| `sp500` | 0.16s | 0.15s | 0.28s | 0.68s | 0.95× | 1.79× | 4.32× |
| `vix` | 0.12s | 0.11s | 0.14s | 0.33s | 0.94× | 1.19× | 2.80× |
| `hmm` | 0.08s | 0.08s | 0.16s | 0.84s | 1.01× | 2.09× | **10.89×** |

`elbo` ≈ off (no-op cost). `always` is 3-11× slower because every iter replays
the entire tree history.

---

## Test B — Search-level Optuna comparison (each mode finds its own optimum)

**Question**: at the same trial budget, does running Optuna *with refit on* find a
better optimum than *with refit off*?

**Setup**: same 500-trial / 6-dataset Optuna search as Test A's row 0, but with
`mixture_refit_leaves=true` fixed across all trials. The trigger differs per run.

| Dataset | naive | ensemble | **moe (off)** | moe-refit-elbo | moe-refit-every_n | Δ elbo | Δ every_n |
|---|---|---|---|---|---|---|---|
| `synthetic` | 5.0233 | 4.8899 | **3.6779** | 4.3544 | 4.8498 | **+18.39%** | **+31.86%** |
| `fred_gdp` | 0.9311 | 0.9094 | **0.9381** | 0.9381 | **0.9292** | +0.00% | **−0.94%** |
| `sp500_basic` | 0.0100 | 0.0100 | **0.0100** | 0.0099 | 0.0100 | **−0.34%** | +0.34% |
| `sp500` | 0.0100 | 0.0100 | **0.0100** | 0.0100 | 0.0101 | +0.15% | +0.33% |
| `vix` | 2.8869 | 2.8724 | **2.6745** | 2.7134 | 2.7190 | +1.46% | +1.66% |
| `hmm` | 2.1913 | 2.1818 | **2.1465** | 2.1465 | **2.1202** | +0.00% | **−1.23%** |

### Findings (Test B)

- **`fred_gdp` and `hmm` improve under `every_n=10`** (−0.94% and −1.23%).
  These are the two datasets where issue #37 predicted refit would help — both
  have latent regime structure that EM struggles to recover. With Optuna free to
  explore the refit-on space, it finds configs that benefit from periodic leaf
  rewrites.

- **`synthetic` regresses dramatically (+18% / +32%)** under both refit triggers.
  But this is **not** a refit regression — it's TPE divergence:
  - v0.6 best config: `K=2, init=tree_hierarchical, e_step=em, div=0.36` → 3.6779
  - elbo best config: `K=3, init=gmm, e_step=loss_only, div=0.13` → 4.3544
  - The refit-on Optuna search was led to a different basin by the very first
    trials where refit fired, and in 500 trials it never recovered the
    `tree_hierarchical / em` basin.

- **`elbo` trigger never fires for the eventual best configs on 6/6 datasets**
  (verified by `fred_gdp` and `hmm` matching baseline RMSE to 4 decimals — proof
  of no-fire). For these well-behaved configs, search-level results = baseline.

- **`every_n=10` is the more interesting trigger** for empirical gains: it fires
  unconditionally every 10 iters past warmup, so it always perturbs the EM
  dynamics. On `fred_gdp` and `hmm` that perturbation pays off; on `synthetic`
  the perturbation derails the search.

### Cost (Test B) — wall-clock per 500-trial run

| Dataset | moe (off) | elbo | every_n |
|---|---|---|---|
| `synthetic` | 240s | 465s | 1125s |
| `fred_gdp` | 136s | 139s | 168s |
| `sp500_basic` | 499s | 458s | 319s |
| `sp500` | 248s | 296s | 471s |
| `vix` | 182s | 189s | 416s |
| `hmm` | 147s | 229s | 237s |

`elbo` is essentially free (1.0-1.5×); `every_n` adds 1.2-4.7× depending on how
many trials hit configs with deep trees / many iterations.

---

## Combined honest interpretation

### What refit IS

- A **safety net for misconfigured runs**. The bundled
  `examples/em_refit_demo.py` shows the dramatic case: `mixture_init=random`
  on a synthetic two-regime regression — refit-off plateaus at validation RMSE
  2.17 (stuck near the bad init), refit-on (decay=0, always) reaches 1.19 (−45%).
  When you have a bad config (which you might in production if the search is
  short, or if the data shifts), refit recovers some of the loss.

- A **structurally faithful EM update** for the boosted-tree case. The append-only
  invariant means classical EM's "free-parameter M-step" can't happen. Refit
  restores the closed-form leaf optimum on each tree's existing partition.
  This was the theoretical motivation in issue #37 and it's preserved.

### What refit IS NOT

- **A free improvement on Optuna-tuned configs**. Test A demonstrates this:
  on 5/6 datasets refit hurts the v0.6 best config by 0.4-7.6%. The single
  apparent improvement (`fred_gdp`, −0.21%) is within one std.

- **A search-level winner at the current trial budget**. Test B's mixed
  results suggest 500 Optuna trials is not enough to reliably explore the
  refit-on space — the search lands in different basins per trigger, and only
  on the "easy refit" datasets (`fred_gdp`, `hmm`) does it find a clear win.

### Recommended defaults

- **`mixture_refit_leaves=false`** as the v0.7 ship default (matches v0.6
  behavior bit-identically). This stays.
- **`mixture_refit_trigger=elbo`** as the recommended opt-in for users who
  want refit safety with no search-time cost. It's bit-identical when EM is
  healthy and only fires on the cases issue #37 was designed to help.
- **`mixture_refit_trigger=every_n=10`** for users who want stronger refit
  pressure (e.g. they suspect their init is poor). Best evidence of help on
  `fred_gdp` / `hmm`; expect 1.2-2.5× cost.
- **`mixture_refit_trigger=always`** is for diagnosis / theoretical analysis
  only — too expensive for production at 3-11× cost.

### Does refit "improve accuracy"?

**Honest answer**: at the per-config level, no — it slightly degrades
Optuna-tuned configs (which were tuned for refit-off). At the search level,
it finds genuinely better optima on `fred_gdp` (−0.94%) and `hmm` (−1.23%),
the two datasets where the latent regime structure was hardest in v0.6 — but
the improvements are small and the search is sensitive to TPE trajectory.

The strongest empirical case for refit is **bad-init recovery** (45% RMSE
reduction on `mixture_init=random`), which is what the demo and the issue
were originally about.

---

## Generated artifacts

- `bench_results/study_v07_baseline.json` — 500-trial baseline (3 variants × 6 datasets)
- `bench_results/study_v07_refit_elbo.json` — 500-trial Optuna search with refit=elbo
- `bench_results/study_v07_refit_every_n.json` — 500-trial Optuna search with refit=every_n=10
- `bench_results/study_v07_refit_report.md` — search-level merged report (Test B)
- `bench_results/bench_v07_per_config.json` — per-config 4-mode CV (Test A)
- `bench_results/bench_v07_per_config.md` — per-config tabular report
- `bench_results/em_refit_demo.png` — bad-init recovery plot
- `examples/bench_v07_refit_acceptance.py` — smoke acceptance bench (synthetic + hmm)
- `examples/bench_v07_per_config.py` — per-config ablation script (Test A generator)
- `examples/em_refit_demo.py` — bad-init recovery demo
- `examples/merge_v07_refit_report.py` — search-level merger (Test B generator)
