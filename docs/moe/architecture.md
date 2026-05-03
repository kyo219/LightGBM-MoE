# Technical Deep Dive

How MoE is implemented inside LightGBM-MoE: the architecture, the EM training loop, and the per-iteration update derivations. Code references point at `src/boosting/mixture_gbdt.cpp` (and the matching `.h`) on the current `master`; line numbers are kept consistent with the implementation. A short index of every function referenced below is in [§6 Code map](#6-code-map).

## 1. Architecture

The MoE model is **K Expert GBDTs** (regression, same objective as the mixture) plus **1 Gate GBDT** (multiclass with `num_class = K`):

```
                    ┌─────────────┐
                    │   Input X   │
                    └──────┬──────┘
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌────────────┐  ┌────────────┐  ┌────────────┐
    │  Expert 0  │  │  Expert 1  │  │    Gate    │
    │   (GBDT)   │  │   (GBDT)   │  │   (GBDT)   │
    │ Regression │  │ Regression │  │ Multiclass │
    └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
          │  f₀(x)        │  f₁(x)        │ logits z
          ▼               ▼               ▼
    ┌─────────────────────────────────────────────┐
    │           Weighted Combination              │
    │  ŷ = Σₖ softmax((z + b)/T)ₖ · fₖ(x)         │
    └─────────────────────────────────────────────┘
```

Key state held on `MixtureGBDT`:

| Component | Implementation | Field |
|---|---|---|
| Expert GBDTs | K independent regression GBDTs | `experts_` (`vector<unique_ptr<GBDT>>`) |
| Gate GBDT | one multiclass GBDT (K classes) | `gate_` |
| Responsibilities r_ik | N × K soft assignments (sample-major) | `responsibilities_` |
| Per-expert noise scale σ_k² (or b_k) | size K, M-step updated each round | `expert_variance_` |
| Load-balancing bias b_k | size K, DeepSeek-style routing nudge | `expert_bias_` |
| Bias-free prior π_k(x) = softmax(z/T) | used by E-step / ELBO | `gate_proba_no_bias_` |
| Routing distribution softmax((z+b)/T) | used for ŷ and inference | `gate_proba_` |

Gate config setup (`mixture_gbdt.cpp:134–141`):

```cpp
gate_config_->objective = "multiclass";
gate_config_->num_class = num_experts_;
gate_config_->max_depth = config_->mixture_gate_max_depth;         // default: 3
gate_config_->num_leaves = config_->mixture_gate_num_leaves;       // default: 8
gate_config_->learning_rate = config_->mixture_gate_learning_rate; // default: 0.1
gate_config_->lambda_l2 = config_->mixture_gate_lambda_l2;
```

## 2. The EM training loop

Each call to `TrainOneIter()` (`mixture_gbdt.cpp:2880`) runs one EM round:

```
Forward()                                  // f_k(x), π_k(x), and yhat
↓
if iter ≥ mixture_warmup_iters (default 5):
    EStep()                                 // r_ik = posterior under current mixture
    UpdateExpertLoad()                      // mean r_ik per k
    SmoothResponsibilities()                // optional EMA / momentum (time-axis)
    UpdateExpertBias()                      // DeepSeek loss-free LB
    UpdateExpertVariances()                 // M-step on σ_k² / b_k
↓
MStepExperts()                             // adds +1 tree per expert (K total)
↓
MStepGate() or MStepGateLeafReuse()        // adds +K trees in the gate (one per class)
↓
ComputeMarginalLogLikelihood()             // every 10 iters: ELBO monotonicity check
```

**Per-round growth**: each EM round adds K trees on the expert side (one per expert) and K trees on the gate side (one per class — multiclass softmax adds K classes per call to `gate_->TrainOneIter`). Expert iters per round is fixed at 1; gate iters per round is `mixture_gate_iters_per_round` (default 1).

**Warmup behavior** (`mixture_gbdt.cpp:2963–2989`): for the first `mixture_warmup_iters` rounds, the E-step / variance estimate / load balancing all skip — only `MStepExperts` and `MStepGate` run. This lets the gate's logits reach `≈ log(r_init)` before the first real E-step, so the prior π_k(x) carries the partition structure the init supplied (quantile / GMM / kmeans) instead of a flat softmax. Discarding `r_init` on iter 0 was empirically harmful — see the long comment in `TrainOneIter` for the trade-off rationale.

### 2.1 Forward — and why two views of the gate distribution

`Forward()` (`mixture_gbdt.cpp:1444`) computes:

- expert predictions `f_k(x_i)` from each expert GBDT (sample-major mirror in `expert_pred_sm_`);
- the gate's **routing distribution** `gate_proba_` = `softmax((z + b)/T)` — used for ŷ during training and for inference;
- the gate's **prior distribution** `gate_proba_no_bias_` = `softmax(z/T)` — used as π_k(x) in the E-step and ELBO.

The split exists because `expert_bias_` is a routing-decision nudge from DeepSeek's "Auxiliary-Loss-Free Load Balancing" (2024) — it is *not* part of the probabilistic model. If b enters the prior π_k that defines responsibilities, the gate is forced to spend each EM iter "undoing" the bias the load balancer just added. PR [#25](https://github.com/kyo219/LightGBM-MoE/pull/25) fixed this on the gradient side; the prior side stayed bias-tainted until this fix landed in the same line of work. See `gate_proba_no_bias_` doc-comment in `mixture_gbdt.h` for the full rationale.

When `mixture_gate_type` is `none` or `leaf_reuse`, no bias is applied to `gate_proba_` in those modes, so `gate_proba_no_bias_` is just a copy of `gate_proba_`.

### 2.2 E-step — `r_ik` as a Bayesian posterior

`EStep()` (`mixture_gbdt.cpp:1652`) computes per sample, per expert:

```
score_ik = log π_k(x_i)                       // (a) prior — bias-free
         + log_norm_k                          // (b) per-expert log-density normalizer
         − inv_scale_k · loss(y_i, f_k(x_i))   // (c) y_i log-likelihood exponent
         − load_penalty_k                      // (d) auxiliary load-balance, optional
r_ik = softmax_k(score_ik)
```

Three modes via `mixture_e_step_mode`:

| Mode | Score | When to use |
|---|---|---|
| `em` (default) | full prior + log-density | Standard EM; gate prior plus expert-fit posterior |
| `loss_only` | `−α · loss` only | Gate is locked into wrong initial beliefs; let expert error dominate routing |
| `gate_only` | `log π_k(x_i)` only | Diagnostics — pure gate routing, expert error ignored |

When `mixture_gate_type="none"`, the mode is forced to `loss_only` (no gate probabilities exist).

**Loss → log-density mapping** (selected by `e_step_loss_type_`, derived from the objective):

| `e_step_loss_type_` | `log_norm_k` | `inv_scale_k` | Density model |
|---|---|---|---|
| `l2` | `−½ log σ_k²` | `1 / (2 σ_k²)` | Gaussian likelihood |
| `l1` | `−log(2 b_k)` | `1 / b_k` | Laplace likelihood |
| `quantile` | `0` | `1 / scale_k` | Pinball as pseudo-energy (no proper density) |

The `0.5·log(2π)` constant in the Gaussian normalizer is dropped because it is k-independent and cancels under softmax.

**Auxiliary load-balance penalty** (`mixture_load_balance_alpha > 0`, default off):
```
load_penalty_k = α · log(load_k · K)
```
where `load_k = (1/N) Σ_i r_ik` is the previous round's expert load. Overloaded experts (load > 1/K) get a positive penalty subtracted from their score; underloaded experts get a negative penalty (i.e. a bonus). This is the classic Switch-Transformer-style auxiliary loss; the DeepSeek loss-free bias path (`UpdateExpertBias`) is the recommended alternative for regime-switching workloads where genuine imbalance (e.g. 70/30) is correct.

**Variance estimation M-step** (`UpdateExpertVariances`, `mixture_gbdt.cpp:1752`):
```
σ_k² = Σ_i r_ik (y_i − f_k(x_i))² / Σ_i r_ik       // l2: Gaussian
b_k  = Σ_i r_ik |y_i − f_k(x_i)| / Σ_i r_ik       // l1: Laplace
scale_k = Σ_i r_ik · pinball(y_i, f_k) / Σ_i r_ik // quantile: pseudo-scale
```

This is the standard Jordan-Jacobs M-step on the noise scale. With `mixture_estimate_variance=true` (default since [#24](https://github.com/kyo219/LightGBM-MoE/pull/24)), the responsibility softmax is the actual Bayesian posterior — y-scale invariant, free of the temperature hyperparameter `mixture_e_step_alpha` that legacy code held fixed. Setting `false` re-enables the legacy fixed-alpha behavior and emits a warning at Init.

The `expert_variance_[k]` floor at `kMixtureEpsilon` prevents collapse on fully de-routed experts (`Σ_i r_ik → 0`). The dimension trap fixed in this codebase: quantile users previously got `pinball / E[diff²]` because the residual term was hardcoded to `(diff·diff)` — dimensionally `O(1/diff)` and silently temperature-coupled to y-scale. Now each loss type accumulates its own residual term, matching the E-step's exponent.

### 2.3 M-step on experts

`MStepExperts()` (`mixture_gbdt.cpp:2181`) computes responsibility-weighted gradients per expert and adds one tree:

```
grad_k[i] = r_ik · ∂L(y_i, f_k(x_i)) / ∂f_k(x_i)
hess_k[i] = max(r_ik · ∂²L / ∂f_k², kMixtureEpsilon)
experts_[k]->TrainOneIter(grad_k, hess_k)
```

**Hard M-step** (`mixture_hard_m_step=true`, default): builds a per-expert sample subset from `argmax_k r_ik` and passes it to `SetBaggingData`, so each expert's tree learner only constructs histograms over its assigned samples. **The gradients themselves stay soft (r-weighted)**; the hard part is which rows are visible to each expert. This is the sparse-activation optimization from #10 / #16, ~3-5× speedup at large K. Falls back to dense activation if a partition has fewer than `max(2 · min_data_in_leaf, 16)` samples — otherwise SerialTreeLearner can pick a split with one side empty and CHECK-fail.

The earlier behavior zeroed gradients for non-winners outright; that produced expert collapse whenever bagging fell back to the full dataset (losers got *no* gradient signal at all for those samples, so the gate could not learn to route to them). Soft-gradient + hard-bagging is the post-#16 design.

**Diversity regularizer** (`mixture_diversity_lambda > 0`):
```
grad_k[i] −= λ / (K−1) · Σ_{j≠k} r_ij · clip(f_k(x_i) − f_j(x_i), ±δ)
hess_k[i] += λ / (K−1) · Σ_{j≠k} r_ij
```
with `δ = 1.0`. Pushes f_k *away* from f_j on samples j has a strong claim on (high r_ij). Three details that all matter:

1. **Sign**: original code added `+λ Σ r_ij (f_k − f_j)` — that's the gradient of *aligning* f_k with the other experts. Empirically at λ=0.05, K=3 it cut pairwise expert distance to 22% of the λ=0 baseline.
2. **Huber clip**: naively flipping the sign on the unbounded reward `−½λ Σ r_ij (f_k − f_j)²` makes predictions diverge to ±∞ within 30 iters at λ=0.001 (peak |pred| inflated 25×). The clip caps the per-pair contribution at `±λ·δ·r_ij`.
3. **Hessian damping**: the un-saturated true Hessian of a diversity reward is negative, which destabilizes Newton's method on leaf values. The `+λ Σ r_ij` term keeps the Newton step well-conditioned inside the clip region.

All three landed in [#26](https://github.com/kyo219/LightGBM-MoE/pull/26). The empirical "`mixture_diversity_lambda` is non-zero in every dataset's per-best params" finding from the 500-trial study is what this fix unlocked.

**Optional**: expert dropout (`mixture_expert_dropout_rate`), curriculum dropout schedule (`mixture_dropout_schedule`), per-expert adaptive learning rate (`mixture_adaptive_lr`). All off by default. See [parameters.md](parameters.md) for full reference.

### 2.4 M-step on gate

`MStepGate()` (`mixture_gbdt.cpp:2522`) trains the gate GBDT with **soft cross-entropy against the full responsibility distribution** (Jordan-Jacobs), not against argmax pseudo-labels. With logit `u = z/T` and `p = softmax(u)`:

```
∇L/∇z_ik = (1/T) (p_k − r_k)              // base CE gradient
∇²L/∇z_ik² = (K/(K−1)) · p(1−p) / T²       // Friedman factor on the K-redundant softmax
```

Plus optional Dirichlet shrinkage toward uniform (`mixture_gate_entropy_lambda > 0`):

```
∇reg = λ · (p_k − 1/K) / T
∇²reg = λ / T² (constant damping; the exact diagonal Hessian λ p(1−p)/T² vanishes
                at simplex corners and would explode the Newton step there)
```

The parameter is named `entropy_lambda` for back-compat but the gradient `λ(p − 1/K)` is actually the gradient of a Dirichlet shrinkage, not `d(−H)/dz`. The entropy gradient vanishes near corners and is a weak anti-collapse signal; Dirichlet shrinkage doesn't and is what collapse-prevention actually needs.

Three subtleties that all matter:

1. **Bias-free target** — gradient is computed against `softmax(z/T)`, not `softmax((z+b)/T)`. Same DeepSeek-LB rationale as the prior split in §2.1.
2. **Temperature chain rule** — `(p − r)/T` and `p(1−p)/T²`. Earlier code used `(p−r)` and `p(1−p)` directly, mis-scaling the Newton step by T at non-unit temperatures (introduced when temperature annealing landed without updating the gradient).
3. **Friedman K/(K−1) factor** — matches standard LightGBM `MulticlassSoftmax::GetGradients` in `multiclass_objective.hpp`. Without it, Newton leaf values are systematically `(K−1)/K` of standard (e.g. 2/3 at K=3, 9/10 at K=10).

The argmax-pseudolabel → soft-CE switch landed in [#23](https://github.com/kyo219/LightGBM-MoE/pull/23) and is what makes regime probabilities well-calibrated rather than just "where the gate happens to point".

**Inner gate loop**: when `mixture_gate_iters_per_round > 1`, the gradient/Hessian are recomputed inside the inner loop because each `gate_->TrainOneIter` adds K trees and changes `z` (and hence p). Reusing iter-1 grad/hess on iter 2+ would degrade Newton's method to constant-gradient subgradient descent.

### 2.5 LeafReuse gate

`MStepGateLeafReuse()` (`mixture_gbdt.cpp:2662`, selected by `mixture_gate_type="leaf_reuse"`) is an alternative gate whose routing is derived from expert-tree leaf statistics:

1. Traverse expert 0's latest tree using bin data to get each sample's leaf index.
2. For each leaf, average `r_ik` over the samples that fell in it → per-leaf routing distribution.
3. Set `gate_proba_` from the per-leaf distribution.
4. Train the gate GBDT against the same leaf-aggregated target via soft CE (same gradient form as `MStepGate` above) so out-of-sample routing remains valid.

Cheaper at training time, structurally tied to expert tree shape. Won on `fred_gdp` and `vix` in the 500-trial study (the two datasets where `gbdt`-gate lost). Step 4 was added to fix a real bug: earlier behavior trained the gate only every `mixture_gate_retrain_interval` iterations against argmax pseudo-labels, so the gate GBDT trees were decorative — Forward read `gate_proba_` from leaf stats during training, contributed nothing to in-domain routing, and was sparsely fit for `PredictRegimeProba` on unseen data.

### 2.6 ELBO monotonicity diagnostic

`ComputeMarginalLogLikelihood()` (`mixture_gbdt.cpp:1808`) computes the training-set log-marginal:

```
ELBO = Σ_i log Σ_k π_k(x_i) · p(y_i | x_i, f_k, σ_k²)
```

via logsumexp. Logged every 10 iters (and the first 5). Exact-M-step EM is monotone non-decreasing in this quantity; the GBDT M-step is approximate (one tree per round), so small dips are normal — but a >5% drop is logged as a warning. Historically this signal has caught:

- the bias-side regularizer fighting the gate (pre #25);
- the diversity sign flip (pre #26);
- the dimension mismatch in quantile scale estimation (recent fix);
- aggressive expert dropout / `mixture_adaptive_lr` decoupling experts from EM.

If you tune aggressively (high dropout, custom schedules) and see warnings, the ELBO loss is the single best signal that something is fundamentally inconsistent between E and M.

**v0.8 change**: ELBO is now *computed* every post-warmup iter (was every 10 in v0.7) — required so the sliding-window refit trigger in §2.8 can detect both drops and plateaus. The *logging* cadence is unchanged (still every 10 iters + first 5), so console output is identical to v0.7 for users who don't enable the trigger. The 5% drop warning above is independent of the trigger and stays tied to the log cadence.

### 2.7 Leaf-value refit on each E-step (v0.7, opt-in)

The "additive-only EM" limitation — boosted-tree experts can only inch their predictions per round, so `r_ik` can only inch in response, so EM is stuck near `r_init` — has its root cause not in "we only append trees" but in "leaf values are write-once at the `r` of the round each tree was built in". The tree partitions form a useful data-dependent feature representation `Φ_k(x) ∈ {0,1}^{num_leaves}`; the leaf values are linear coefficients on top of `Φ_k`. Classical EM rewrites those coefficients each round in closed form. v0.6 leaves them frozen.

`MixtureGBDT::RefitExpertsAndGate` (`mixture_gbdt.cpp:2522`, called from `TrainOneIter` between `UpdateExpertVariances` and `MStepExperts` when `mixture_refit_leaves=true`) restores the closed-form M-step over each tree's existing partition. For each expert k:

```
new_v_l = -shrinkage_k · (Σ_{i ∈ leaf_l} r_ik · g_i)
                      / (Σ_{i ∈ leaf_l} r_ik · h_i + λ_l2 + l2_reg)
final_v_l = decay · old_v_l + (1 − decay) · new_v_l
```

with `g_i, h_i` from `objective_function_->GetGradients(f_k, ...)` evaluated against the cumulative score *of trees iter 0..t-1* (NOT including tree t's own contribution — see "score reset + replay" below). For the gate (gbdt mode only — `leaf_reuse` derives routing from leaf statistics so refitting would conflict with `MStepGateLeafReuse`'s Step 4), the same machinery runs against the soft-CE gradient form `(p − r)/T` with Friedman `K/(K-1)` Hessian and Dirichlet shrinkage — identical to `MStepGate`'s gradient.

**Implementation in `GBDT::RefitLeavesByGradients` (`gbdt.cpp:266`)**: a callback-based variant of `RefitTree` that:

1. **Zeros all score updaters** (`MultiplyScore(0.0, tid)` per class, on both `train_score_updater_` and every `valid_score_updater_`). This is required because the per-iter Newton step targets the optimal absolute leaf value, which only matches the "replace" semantics of `Tree::SetLeafOutput` if the gradient is computed against `f_without_iter_t`. Computing it against the full score (which includes iter t's old leaf) lands at a fixed point that undershoots by `shrinkage / (1 + shrinkage)` — verified empirically: refit on a synthetic K=2 case drove RMSE from 0.6 → 2.6 before the score-reset fix.
2. **Replays tree-by-tree**: for each iter, `recompute_grad_hess(grad, hess)` is invoked once (matching `RefitTree`'s per-iter `Boosting()` cadence so per-iter softmax coupling for multiclass gates stays correct), then each of the `num_tree_per_iteration_` trees in that iter is refit by computing per-leaf `Σ r·g`, `Σ r·h` via `Tree::PredictLeafIndices` (a new mirror of `AddPredictionToScore` that writes leaf indices instead of accumulating leaf values). New leaf values are committed via `Tree::SetLeafOutput` and the tree's contribution is added back to all score updaters via `AddScore(tree, tree_id)`.

After all iters are replayed, the score updaters reflect the post-refit cumulative score, and `Forward()` is re-called from MixtureGBDT to refresh `expert_pred_` / `gate_proba_` so the subsequent `MStepExperts` / `MStepGate` see the new state.

**Trigger modes** (`mixture_refit_trigger`):

| Mode | When refit fires | Cost |
|---|---|---|
| `always` (default) | Every post-warmup iter | Highest; ~3-7× wall time at 60 rounds |
| `elbo` | Most recent ELBO log block (every 10 iters) showed a >5% drop | Cheap — reuses the existing every-10-iter ELBO computation; requires `mixture_estimate_variance=true` (otherwise no ELBO is computed and trigger never fires) |
| `every_n` | Every `mixture_refit_every_n` post-warmup iters | Tunable; e.g. n=10 ≈ 6× fewer fires than `always` |

**Decay semantics** (`mixture_refit_decay_rate ∈ [0, 1]`): `final_v = decay · old_v + (1 − decay) · fit_v`. `0.0` (default) replaces fully — closest to classical EM. `1.0` is an exact pass-through (the refit machinery still runs but every leaf returns to its original value). Intermediate values stabilize the fixed-point iteration when E-step / M-step are temporarily out of sync (e.g. high dropout, aggressive annealing). Same parameter shape as LightGBM core's `refit_decay_rate`, but namespaced to the mixture path so non-mixture refit users are unaffected.

**Empirical effect** (`examples/em_refit_demo.py`): on a synthetic two-regime regression with `mixture_init=random` (forced bad init), refit-off plateaus at validation RMSE 2.17 with `||r_t − r_init||_F` capped near 0.6 (stuck in the bad basin); refit-on with `decay=0.0` reaches RMSE 1.19 with `||r_t − r_init||_F` rising to 0.99 (escaped). The plot at `bench_results/em_refit_demo.png` shows the two trajectories side-by-side.

**Subsumed pathology guards**: with refit-on, the symmetry breaker (PR #36), the diversity-reg Huber clip (PR #26), and parts of the variance-estimator anti-collapse (PR #24) become structurally redundant — they exist to compensate for the same root cause refit fixes (frozen leaves can't track changing `r`). For now the guards remain (default-off refit is a strict superset of v0.6 behavior), but a future cleanup PR could simplify them.

**Incompatibility — `gate_type='leaf_reuse'`**: refit only rewrites expert leaves and (in `gbdt` gate mode) the gate's GBDT leaves. `leaf_reuse` derives gate routing from expert-tree leaf statistics and trains a separate gate GBDT for out-of-sample inference; refit would touch the experts but leave that gate GBDT frozen, producing an asymmetric update that empirically degrades performance (verified at +7% RMSE on `vix` under uniform init in `bench_results/bench_v07_per_config_uniform.md`). The Init guard auto-disables `mixture_refit_leaves` when `mixture_gate_type='leaf_reuse'` is set, with a one-time warning. Use `gate_type='gbdt'` if you want refit semantics.

### 2.8 Partition re-grow + sliding-window ELBO trigger (v0.8, opt-in)

§2.7's leaf refit only rewrites leaf *values* `v_k^(s)`; the split *structures* `S_k^(s)` chosen against `r_init` stay frozen. v0.8 ([#41](https://github.com/kyo219/LightGBM-MoE/issues/41) / [PR #42](https://github.com/kyo219/LightGBM-MoE/pull/42) ELBO trigger fix, [PR #43](https://github.com/kyo219/LightGBM-MoE/pull/43) partition re-grow) extends the M-step to the **(split, leaf) pair** — block coordinate ascent on the full tree parameter rather than just its linear-coefficient slice. Together with §2.7, this is the closest GBDT-MoE gets to a free-parameter EM M-step within each tree slot. Full derivation: [`docs/v0.8/partition_regrow_design.md`](../v0.8/partition_regrow_design.md).

`MixtureGBDT::RefitExpertsAndGate` is now a two-phase pipeline (Phase 1 conditional on `mixture_regrow_oldest_trees=true`; Phase 2 = §2.7 leaf refit, runs whenever `mixture_refit_leaves=true`):

```
Phase 1 (v0.8): per expert k (and gate when gate_type=gbdt):
  for s in [0..M-1]                          # M = mixture_regrow_per_fire,
                                              # capped by mixture_regrow_min_remaining
    if mode == "replace":
        experts_[k]->RegrowTreeAt(s, cb)     # rebuild splits + leaves
    else:  # mode == "delete"
        experts_[k]->DeleteTreeAt(s)          # ablation: ensemble shrinks

Phase 2 (v0.7): existing leaf refit on all surviving trees, retrofitting
  leaf values of [M..T-1] to the new state Phase 1 produced.
```

**Single-tree subproblem** (the math `RegrowTreeAt` solves):

```
backbone_i = f_k(x_i) − v_k^{(s)}[S_k^{(s)}(x_i)]      // f_k minus tree s's contribution
g_i        = r_ik · ∂loss(y_i, backbone_i) / ∂backbone_i
h_i        = r_ik · ∂²loss / ∂backbone_i²
(S*, v*)   = tree_learner_->Train(g, h)                 // LightGBM picks fresh splits
```

LightGBM's greedy split selector can always fall back to a constant tree (= weighted mean), so `Q(θ_new; r) ≥ Q(θ_old; r)` — each partition swap is **monotone non-decreasing in Q**. By the EM inequality `L(θ_{t+1}) − L(θ_t) ≥ Q(θ_{t+1}; r_t) − Q(θ_t; r_t)`, ELBO is non-decreasing in expectation.

**Implementation in `GBDT::RegrowTreeAt`** (`gbdt.cpp` near `RefitLeavesByGradients`): same zero-and-replay score-updater idiom as §2.7, with one tree-build inserted at slot `s_iter`:

1. Zero `train_score_updater_` and all `valid_score_updater_`, replay iters `[0, s_iter)` → score = `f^{(s_iter-1)}` (the backbone).
2. Invoke `recompute_grad_hess(g_buf, h_buf)` once. The callback reads `GetPredictAt(0, ...)`, which now returns the backbone — same callback shape as Phase 2 leaf refit, so v0.7 and v0.8 share gradient code.
3. For each `tree_id ∈ [0, num_tree_per_iteration_)`: `tree_learner_->Train(g, h, is_first_tree=false)` → fresh `Tree*`. Apply `Tree::Shrinkage(shrinkage_rate_)` (same as gbdt.cpp:581 in normal training). Swap into `models_[s_iter * K + tree_id]` in place. `AddScore(tree, tree_id)` on all updaters.
4. Replay iters `(s_iter, T)` so the score reflects the post-swap full ensemble.

**Bagging during regrow**: deliberately *not* overridden. An attempt to force full-data via `SetBaggingData(nullptr, all_idx, num_data_)` segfaulted the next M-step's `serial_tree_learner` state at iter 7 in the smoke test. The regrown tree therefore sees the previous M-step's hard partition rather than the full sample set; the gradients themselves are still r-weighted via the callback, so the structure is r-aware in spirit, just restricted to the active partition.

**Sliding-window ELBO trigger** (B, the trigger that gates Phase 1 + Phase 2 when `mixture_refit_trigger='elbo'`): the v0.7 trigger ("relative drop > 5% in the most recent log block") fired 0/6 datasets in the v0.7 acceptance bench because Optuna-tuned configs don't *drop* ELBO — they *plateau* at a sub-optimal local fixed point (E-step output stops moving → M-step contributions vanish → ELBO flatlines without dropping). v0.8 maintains a per-iter sliding window `elbo_history_` of size `mixture_elbo_window` (default 10) and fires on EITHER:

```
drop signal    = (elbo_t - elbo_{t-W}) / max(|elbo_{t-W}|, 1)
                                                < -mixture_elbo_drop_threshold     (default 0.01)
plateau signal = (max(window) - min(window)) / max(|max(window)|, 1)
                                                < mixture_elbo_plateau_threshold   (default 0.001)
                  AND moe_iter > warmup_iters + mixture_elbo_min_iter_for_plateau  (default 20)
```

Drop detection retains the v0.7 catch for gross E/M misalignment; plateau detection adds the basin-lock-in catch the v0.7 trigger missed. Cost: one O(N·K) logsumexp per post-warmup iter (~5–10% wall time). Setting `plateau_threshold=0` disables plateau detection (recovers a tightened version of v0.7 behavior).

**Trigger modes** (extended from §2.7's table):

| Mode | When it fires (v0.8) | Cost |
|---|---|---|
| `always` | Every post-warmup iter | Highest |
| `elbo` (default safest) | drop OR plateau on the sliding window (above). v0.7's 5%-drop check is recoverable via `mixture_elbo_drop_threshold=0.05, mixture_elbo_plateau_threshold=0` | Cheap (per-iter ELBO + window math) |
| `every_n` | Every `mixture_refit_every_n` post-warmup iters | Tunable |

**Empirical effect** ([`bench_results/v0_8_acceptance_FINAL.md`](../../bench_results/v0_8_acceptance_FINAL.md), [`bench_results/v0_8_search_FINAL.md`](../../bench_results/v0_8_search_FINAL.md)):

- **Bad-init recovery (the design target)**: synthetic + `mixture_init=random`, regrow with `per_fire=3` reaches RMSE 5.11, **−13.0% vs the off baseline** (5.87) — beating §2.7 leaf-refit-alone (5.53, −5.8%) by ~7 percentage points. Partition re-build does what leaf refit can't when the early-iter splits encode the wrong partition.
- **Tuned configs (v0.6 winning configs held fixed)**: regrow at the safe `elbo` trigger fires 0/6 (correctly inert — these trajectories don't plateau). At forced `always` trigger, regrow degrades 4/6 by +0.2-7.1% — same shape as §2.7's "refit-always-degrades-tuned-configs" finding. The mechanism is sound; the regime is wrong.
- **500-trial Optuna search with v0.8 features in scope** (`mixture_refit_leaves` and `mixture_regrow_oldest_trees` as search variables, `uniform` added to `mixture_init`): Optuna picked `refit_leaves=False` on 5/6 datasets and `regrow=False` on 6/6 winning configs; fANOVA importance of `refit_leaves` is 0.000-0.008 across all datasets. Optuna with budget independently agrees with the design intent: refit/regrow are bad-init safety nets, not free improvements for tuned configs.

**Subsumed pathology guards**: same point as §2.7. With Phase 1 + Phase 2 on, the symmetry breaker (PR #36), diversity-reg Huber clip (PR #26), and parts of the variance-estimator anti-collapse (PR #24) become structurally more redundant — they exist to compensate for the same root cause re-grow + refit fix (frozen splits + frozen leaves can't track changing `r`). Default-off keeps the v0.6 / v0.7 behavior bit-identical, so the guards remain as defense in depth.

**Incompatibility — `gate_type='leaf_reuse'`**: same root cause as §2.7's leaf-refit incompat. `leaf_reuse` derives gate routing from expert-tree leaf statistics and trains a separate gate GBDT for out-of-sample inference; partition re-grow rewrites expert tree splits but leaves that gate GBDT structurally frozen, producing an asymmetric update. The Init guard auto-disables `mixture_regrow_oldest_trees` when `mixture_gate_type='leaf_reuse'` is set, with a one-time warning at Init *and* at `ResetConfig` (mirrors the §2.7 guard pair).

## 3. Initializing `r_ik`

`InitResponsibilities()` (`mixture_gbdt.cpp:462`) supports 7 schemes via `mixture_init`:

| Scheme | Description | When |
|---|---|---|
| `uniform` (default) | All `r_ik = 1/K`, broken by per-expert seeds + symmetry breaker | Generic |
| `quantile` | Sort by y, assign by rank to expert `⌊rank·K/N⌋` (with soft boundaries) | y-magnitude is the regime |
| `random` | Random hard assignment per sample | Baseline / ablation |
| `balanced_kmeans` | Balanced K-Means on `[X, y]`, equal-size clusters | y-aware regime discovery |
| `kmeans_features` | Balanced K-Means on raw X only | Regime is in X-space (recommended for macro/financial) |
| `gmm` | GMM on `[X, y]`, soft probabilities | y-aware probabilistic init |
| `gmm_features` | GMM on raw X only | Probabilistic regime in X-space |
| `tree_hierarchical` | Train deep tree on y; agglomeratively cluster leaves by mean(y) into K groups | Decision-tree-friendly y-partition |

Note: `balanced_kmeans` and `gmm` include `y` as an extra dimension, biasing clusters toward y-magnitude. The `*_features` variants discover regimes in X-space alone — preferred when regimes live in features (macro indicators, market microstructure) rather than in y itself.

**Symmetry breaker** (`BreakUniformSymmetryIfNeeded`, `mixture_gbdt.cpp:1346`): runs unconditionally after the chosen init. If every `r_i` is essentially uniform (within `1e-6`), inject `r_ik += 0.05 · sin(2π · i · (k+1) / N)` and renormalize. Without this, uniform `r` is an EM fixed point — every expert sees the same gradient, builds the same tree, and `r` stays uniform forever. Empirically confirmed in `examples/em_init_sensitivity.py`: no combination of `hard_m_step` / `mixture_estimate_variance` / `mixture_diversity_lambda` can break out without this. The breaker is a no-op for non-uniform inits ([#36](https://github.com/kyo219/LightGBM-MoE/pull/36)).

## 4. Routing variants

### 4.1 Token Choice (default)

Each sample i gets a soft distribution `r_ik` over experts (the E-step output above). All experts see all samples, weighted by `r`. This is what the §2 equations describe and what `mixture_routing_mode="token_choice"` selects.

### 4.2 Expert Choice routing

`mixture_routing_mode="expert_choice"`: `EStepExpertChoice()` (`mixture_gbdt.cpp:1897`) replaces the E-step with a three-stage pipeline:

1. `ComputeAffinityScores` (`mixture_gbdt.cpp:1908`): per-sample, per-expert affinity = `log π_k − α · loss_ik` (or just one of those, controlled by `mixture_expert_choice_score`).
2. `SelectTopSamplesPerExpert` (`mixture_gbdt.cpp:1940`): each expert k picks its own top-C samples by affinity (with adaptive Gaussian noise on top — large noise during warmup to force differentiation, small for tie-breaking after).
3. `ConvertSelectionToResponsibilities` (`mixture_gbdt.cpp:2003`): selected → high `r`; non-selected → small floor (or 0 in `mixture_expert_choice_hard=true` mode).

Capacity `C = expert_capacity_`, defaults to `N · capacity_factor / K`. See [docs/moe/advanced-routing.md](advanced-routing.md) for the affinity score variants and capacity-factor recipe.

### 4.3 Time-series smoothing

`SmoothResponsibilities()` (`mixture_gbdt.cpp:2054`) — applied after the E-step, before the bias / variance update — smooths `r` along row order (assumed time order):

| `mixture_r_smoothing` | Update |
|---|---|
| `none` (default) | no-op |
| `ema` | `r[i] ← (1−λ) r[i] + λ r[i−1]` |
| `momentum` | `r[i] ← (1−λ) r[i] + λ · (r[i−1] + λ (r[i−1] − r[i−2]))` |
| `markov` | smooths `gate_proba_` and `gate_proba_no_bias_` directly inside `Forward`, same single-pass sweep — i.e. routing-prior smoothing rather than posterior smoothing |

Sized by `mixture_smoothing_lambda`. The Markov path is intended for problems where regimes have temporal persistence (financial vol regimes, business-cycle phases); it models π_k as a function of `(x_t, regime_{t−1})` via the time-axis blend rather than learning the persistence into the gate's tree splits.

A subtle correctness fix in the Markov path: previously `prev_gate_proba_` was carried across training iterations and accumulated an iteration-axis EMA on top of the time-axis shift, producing exponentially-weighted-history routing instead of a Markov prior. The corrected sweep uses only the unsmoothed value of row `i−1` from *this* iteration as sample i's prior; no state survives across training rounds.

## 5. Per-expert hyperparameters

Each Expert can have different tree **structural** configurations:

```cpp
std::vector<std::unique_ptr<Config>> expert_configs_;  // one per expert

// Per-expert structural parameters (comma-separated in config)
std::vector<int>    mixture_expert_max_depths;        // e.g. "3,5,7"
std::vector<int>    mixture_expert_num_leaves;        // e.g. "8,16,32"
std::vector<int>    mixture_expert_min_data_in_leaf;  // e.g. "50,20,5"
std::vector<double> mixture_expert_min_gain_to_split; // e.g. "0.1,0.01,0.001"
```

| Specification | Behavior |
|---|---|
| Not specified | All experts use base structural hyperparameters |
| Comma-separated list (length K) | Each expert uses its corresponding value |

See [per-expert-hp.md](per-expert-hp.md) for usage and the role-based Optuna recipe.

**Symmetry breaking via per-expert seeds** — even with shared hyperparameters, experts differentiate via:

```cpp
expert_configs_[k]->seed = config_->seed + k + 1;
```

No label-based initialization is needed for expert differentiation (and `mixture_init=quantile` can leak target information — use with care).

## 6. Code map

Quick file:line index for everything referenced above. All paths are `src/boosting/mixture_gbdt.cpp` unless noted.

| Concept | Function | File:line |
|---|---|---|
| EM loop entry | `TrainOneIter` | mixture_gbdt.cpp:2880 |
| Forward / two-view gate | `Forward` | mixture_gbdt.cpp:1444 |
| E-step (token choice) | `EStep` | mixture_gbdt.cpp:1652 |
| E-step (expert choice) | `EStepExpertChoice` | mixture_gbdt.cpp:1897 |
|   ↳ affinity scores | `ComputeAffinityScores` | mixture_gbdt.cpp:1908 |
|   ↳ top-C selection | `SelectTopSamplesPerExpert` | mixture_gbdt.cpp:1940 |
|   ↳ selection → r | `ConvertSelectionToResponsibilities` | mixture_gbdt.cpp:2003 |
| Variance M-step | `UpdateExpertVariances` | mixture_gbdt.cpp:1752 |
| ELBO diagnostic | `ComputeMarginalLogLikelihood` | mixture_gbdt.cpp:1808 |
| Smoothing (post-E) | `SmoothResponsibilities` | mixture_gbdt.cpp:2054 |
| Load-balance bias | `UpdateExpertBias` | mixture_gbdt.cpp:2120 |
| Expert M-step | `MStepExperts` | mixture_gbdt.cpp:2181 |
| Gate M-step (gbdt) | `MStepGate` | mixture_gbdt.cpp:2665 |
| Gate M-step (leaf_reuse) | `MStepGateLeafReuse` | mixture_gbdt.cpp:2805 |
| r initialization | `InitResponsibilities` | mixture_gbdt.cpp:462 |
| Symmetry breaker | `BreakUniformSymmetryIfNeeded` | mixture_gbdt.cpp:1346 |
| Pointwise loss (E-step) | `ComputePointwiseLoss` | mixture_gbdt.cpp:1314 |
| Gate config setup | `Init` | mixture_gbdt.cpp:134 |
| Two-view gate state | `gate_proba_*` doc | mixture_gbdt.h:338–355 |
| Refit + regrow dispatch (v0.7 + v0.8) | `RefitExpertsAndGate` | mixture_gbdt.cpp:2605 |
| Refit/regrow trigger gate (incl. v0.8 sliding window) | `ShouldRefit` | mixture_gbdt.cpp:2540 |
| ELBO sliding window state (v0.8) | `elbo_history_` | mixture_gbdt.h |
| Leaf-refit core (LightGBM, v0.7) | `GBDT::RefitLeavesByGradients` | gbdt.cpp:268 |
| Partition re-grow core (LightGBM, v0.8) | `GBDT::RegrowTreeAt` | gbdt.cpp |
| Partition delete (v0.8 ablation) | `GBDT::DeleteTreeAt` | gbdt.cpp |
| Leaf-index lookup (LightGBM) | `Tree::PredictLeafIndices` | src/io/tree.cpp |
| v0.8 Init guards (leaf_reuse incompat, regrow + refit) | `MixtureGBDT::Init` / `ResetConfig` | mixture_gbdt.cpp:84, 3841 |
