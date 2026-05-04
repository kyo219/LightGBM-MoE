# v0.8 Partition Re-grow — Design Document

This is the formal design for **work item A** of v0.8.0 (issue
[#41](https://github.com/kyo219/LightGBM-MoE/issues/41)). It is the
companion to **work item B** (the ELBO-trigger fix in PR
[#42](https://github.com/kyo219/LightGBM-MoE/pull/42)) — A fires from
B's trigger.

## 1. Goal

v0.7's leaf refit (`mixture_refit_leaves=true`, PR
[#40](https://github.com/kyo219/LightGBM-MoE/pull/40)) keeps each tree's
split structure `S_k^{(s)}` fixed and rewrites only its leaf values
`v_k^{(s)}`. Splits chosen against `r_init` therefore retain a permanent
bias regardless of how `r` evolves — this is the **third of the four
structural constraints** itemized in issue #41 ("tree partitions never
change").

A discards old splits entirely and rebuilds each targeted tree via
`tree_learner_->Train()` against current `r`-weighted gradients. The
old `(S_k^{(s)}, v_k^{(s)})` pair is replaced by a fresh pair chosen by
LightGBM's greedy split selector at the new `r`.

## 2. Math

### 2.1 Q function

The EM Q function (objective the M-step maximizes):

```
Q(θ; r) = Σ_{i,k} r_ik log p(y_i | f_k(x_i), σ_k²)
        + Σ_{i,k} r_ik log π_k(x_i)
        + H(r)
```

Each expert's prediction decomposes into a sum of trees:

```
f_k(x) = Σ_{s=1..T_k} v_k^{(s)}[ S_k^{(s)}(x) ]

  S_k^{(s)} : X → {1, ..., L_s}   (split structure / partition)
  v_k^{(s)} ∈ R^{L_s}              (leaf-value vector)
```

### 2.2 Block coordinate ascent

v0.7 leaf refit is block coordinate ascent on `{ v_k^{(s)} }_{s,k}`
with each `S_k^{(s)}` held fixed:

```
{v_k^{(s)}}* = argmax_{ {v_k^{(s)}} } Q(θ; r)
```

Per `(k, s, leaf)` this reduces to an independent closed-form Newton
step. Implementation: `GBDT::RefitLeavesByGradients` (`gbdt.cpp:268`).

A is block coordinate ascent on the **pair** `(S_k^{(s)}, v_k^{(s)})`:

```
(S_k^{(s)}*, v_k^{(s)}*) = argmax_{(S, v) ∈ S_trees × R^L} Q(θ'; r)
```

where `θ'` holds all other trees fixed and replaces only the
`(S_k^{(s)}, v_k^{(s)})` pair.

### 2.3 Reduction to a single-tree subproblem

For one specific `(k, s)`, define the **backbone score** (current
expert prediction with tree `s`'s contribution removed):

```
backbone_i = f_k(x_i) − v_k^{(s)}[S_k^{(s)}(x_i)]
```

Then:

```
max_{(S, v)}  Σ_i r_ik · [ −loss(y_i, backbone_i + v[S(x_i)]) ]
```

This is exactly the problem `tree_learner_->Train(g, h)` solves, with:

```
g_i = r_ik · ∂loss(y_i, backbone_i) / ∂backbone_i
h_i = r_ik · ∂²loss(y_i, backbone_i) / ∂backbone_i²
```

LightGBM's greedy split selector picks the best `(S, v)` for these
gradients.

### 2.4 Gate analog

The gate's M-step is soft cross-entropy against `r`. With logit
`u_ik = z_ik / T` and `p = softmax(u)`:

```
g^{gate}_ik = (p_k − r_k) / T
h^{gate}_ik = (K/(K−1)) · p_k(1 − p_k) / T²        (Friedman factor)
```

For partition re-grow on the gate, the backbone is `z_full − tree_s`,
producing `p` from the backbone logits. Same Newton-step gradient form
as `MStepGate` (`mixture_gbdt.cpp:2683`) and the v0.7 gate refit
(`mixture_gbdt.cpp:2640`).

### 2.5 Convergence

LightGBM's `tree_learner_->Train` is greedy best-split. It can always
fall back to a constant tree (= weighted mean), so the new tree's Q
contribution is no smaller than the old tree's:

```
Q(θ_new; r) ≥ Q(θ_old; r)
```

→ each partition swap is **monotone non-decreasing in Q**. By the EM
inequality `L(θ_{t+1}) − L(θ_t) ≥ Q(θ_{t+1}; r_t) − Q(θ_t; r_t)`,
ELBO is non-decreasing in expectation.

### 2.6 Why "oldest M" trees

Trees built during warmup or the first few post-warmup iters were
fitted against `r ≈ r_init`. Their splits encode `r_init`'s partition.
These are the trees most responsible for basin lock-in — leaf refit
can adjust their leaf values but cannot undo their splits.

Newer trees were built against more recent `r`, so their splits are
already "less stale". Targeting the oldest gives the largest expected
return per regrow.

## 3. Algorithm

### 3.1 Top-level (within `RefitExpertsAndGate`)

```
if mixture_regrow_oldest_trees:
  # Phase 1: partition re-grow
  for k in experts:
    M_k = clamp(mixture_regrow_per_fire,
                0,
                num_trees_k − mixture_regrow_min_remaining)
    for s in range(M_k):
      slot = (mode == "delete") ? 0 : s     # delete shifts each iter
      if mode == "replace":
        experts_[k]->RegrowTreeAt(slot, expert_callback(k), l2_reg)
      else:
        experts_[k]->DeleteTreeAt(slot)

  if gate_type == "gbdt":
    # gate analog (auto-disabled when leaf_reuse, see Init guard)
    ...

# Phase 2: existing v0.7 leaf refit on remaining trees
for k in experts: experts_[k]->RefitLeavesByGradients(...)
if gate_type == "gbdt": gate_->RefitLeavesByGradients(...)
```

**Phase 1 → Phase 2 ordering is mathematically required.** After
regrow, downstream trees' leaf values are stale (built assuming the
old tree `s` exists). Phase 2 retrofits them to the new state.

### 3.2 `GBDT::RegrowTreeAt(s_iter, callback, l2_reg)`

```
1. Zero train_score_updater_ and all valid_score_updater_
2. Replay trees [0, s_iter) by calling AddScore for each
3. callback(g_buf, h_buf)             # buffers sized [K * num_data_]
                                      # K = num_tree_per_iteration_
                                      # callback sees current score = backbone
4. Set bagging to full data (no expert-specific subset)
5. for tree_id in range(K):
     g = g_buf + tree_id * num_data_
     h = h_buf + tree_id * num_data_
     new_tree = tree_learner_->Train(g, h, is_first_tree=false)
     tree_learner_->RenewTreeOutput(new_tree, ...)   # if applicable
     new_tree->Shrinkage(shrinkage_rate_)             # apply learning_rate
     models_[s_iter * K + tree_id] = new_tree         # in-place replace
     train_score_updater_->AddScore(tree_learner_.get(), new_tree, tree_id)
     valid_score_updater_[v]->AddScore(new_tree, tree_id) for each
6. Replay trees (s_iter, T) by calling AddScore for each
```

The score-updater zero+replay pattern is the same idiom as
`RefitLeavesByGradients` (`gbdt.cpp:294-321`) — reused verbatim.

### 3.3 `GBDT::DeleteTreeAt(s_iter)` (ablation)

```
1. Zero score updaters
2. Replay trees [0, s_iter)
3. Erase models_[s_iter * K .. s_iter * K + K]   (K elements)
4. Replay shifted trees (s_iter .. T−1)          (now T−1 iters total)
```

Ensemble shrinks by K trees (=1 iter). Subsequent normal training rounds
will refill via `MStepExperts`.

## 4. New GBDT Public API

`gbdt.h` (near `RefitLeavesByGradients` at line ~210):

```cpp
/*! \brief v0.8 partition re-grow. Drops the iter-`s_iter` tree(s) and
 *  rebuilds them via `tree_learner_->Train()` against gradients/hessians
 *  provided by the callback. The callback is invoked exactly once per call,
 *  after score updaters have been re-accumulated to f^{(s_iter-1)}
 *  (i.e. the backbone score with iter s_iter's tree(s) removed). For
 *  multiclass models (num_tree_per_iteration_ > 1), the callback receives
 *  buffers sized [num_tree_per_iteration_ * num_data_] in class-major
 *  layout, matching RefitLeavesByGradients.
 *
 *  Score updaters are fully replayed inside this call — entry assumes the
 *  updaters reflect all trees, exit guarantees the same with the new tree
 *  swapped in at iter s_iter.
 *
 *  No-op when fewer than (s_iter+1) iters exist. */
virtual void RegrowTreeAt(
    int s_iter,
    std::function<void(score_t* grad_buf, score_t* hess_buf)> recompute_grad_hess,
    double l2_reg);

/*! \brief v0.8 partition re-grow, delete-mode variant (ablation).
 *  Removes iter-`s_iter` tree(s) entirely. Subsequent iters shift down
 *  by one. Ensemble size shrinks by num_tree_per_iteration_. */
virtual void DeleteTreeAt(int s_iter);
```

## 5. New Config Parameters

```cpp
// [doc-only] v0.8 partition re-grow
bool        mixture_regrow_oldest_trees = false;   // opt-in, off by default
int         mixture_regrow_per_fire     = 1;       // trees per refit fire per expert
int         mixture_regrow_min_remaining = 5;      // floor on num_trees after regrow
std::string mixture_regrow_mode         = "replace";  // "replace" | "delete"
```

Auto-disabled when `mixture_gate_type='leaf_reuse'` (same incompat
reason as v0.7 leaf refit — see PR
[#40](https://github.com/kyo219/LightGBM-MoE/pull/40) Init guard).

`mixture_refit_decay_rate` does **not** apply to partition swap (decay
between an old split and a new split is undefined). Phase 2 leaf refit
still uses it.

## 6. Implementation Choices

### 6.1 Bagging during regrow

`tree_learner_->Train` honors whatever bagging was set via the most
recent `SetBaggingData` call. The previous `MStepExperts` call set
this to the expert's hard partition (sparse activation). For regrow,
we want **full data** (the new tree should consider all samples
weighted by `r`).

Solution: build `std::vector<data_size_t> all_idx(num_data_)` filled
0..num_data_-1, call `SetBaggingData(nullptr, all_idx.data(), num_data_)`
before `Train`. The next `MStepExperts` will reset bagging anyway.

### 6.2 Shrinkage

After `tree_learner_->Train`, leaves hold raw Newton-step values.
`Tree::Shrinkage(shrinkage_rate_)` scales them by `learning_rate` —
identical to the regular training path (`gbdt.cpp:581`).

### 6.3 Callback design

The callback queries `experts_[k]->GetPredictAt(0, ...)` to read the
current cumulative score. After `RegrowTreeAt`'s zero+replay step 2,
this score equals the backbone (= `f_k − tree_s.contribution`). So
the callback writes:

```
backbone = current GetPredictAt(0)   # = f_k − v_k^{(s)}[S_k^{(s)}(x)]
g_i = r_ik · 2 · (backbone_i − y_i)  # for L2; objective_function for general
h_i = r_ik · 2
```

This is the same callback shape as the v0.7 expert refit
(`mixture_gbdt.cpp:2628-2622`) — refactored to share code.

### 6.4 Multiclass gate (num_tree_per_iteration_ = K)

The gate has K trees per iter. `RegrowTreeAt(s_iter)` rebuilds **all
K** of them as a unit (softmax couples them). Per-class regrow would
require recomputing the softmax target between each rebuild, which
RefitLeavesByGradients's caller already avoids by recomputing g/h
once per iter.

## 7. Acceptance Criteria

Per issue [#41](https://github.com/kyo219/LightGBM-MoE/issues/41):

### Bad-init recovery (the real test)

`synthetic` + `mixture_init=random`:
- v0.7 leaf-refit: RMSE 1.19 (current best)
- **v0.8 target: RMSE < 1.05** (close to gmm-init baseline ~0.88)
- `||r_t − r_init||_F` should reach > 0.95 (v0.7 cap: 0.67)

### Per-config (tuned-config protection)

v0.6 winning configs + v0.8 enabled, all 6 datasets:
- No worse than v0.7 baseline at the elbo trigger
- Plateau-fire rate logged per dataset (>0 fires on at least 2/6)

### Search-level (subset)

500-trial study on at least 3 datasets:
- v0.8 best ≤ v0.7 best (no regression on tuned configs)

## 8. References

- Issue [#41](https://github.com/kyo219/LightGBM-MoE/issues/41) — v0.8
  umbrella spec
- PR [#42](https://github.com/kyo219/LightGBM-MoE/pull/42) — work item
  B (ELBO trigger fix, the trigger this fires from)
- `bench_results/v0_7_acceptance_FINAL.md` — v0.7 baseline empirics
  this design is meant to improve on
- `docs/moe/architecture.md` §2.7 — v0.7 leaf refit derivation that A
  extends
- PR [#40](https://github.com/kyo219/LightGBM-MoE/pull/40) — v0.7 leaf
  refit implementation (the leaf_reuse incompat guard pattern reused
  here)
