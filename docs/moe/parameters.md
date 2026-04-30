# MoE Parameter Reference

Full parameter list for `boosting='mixture'`. For a curated quick-start config, see [the README](../../README.md). For Optuna search templates, see [optuna-recipes.md](optuna-recipes.md).

## MoE Core Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `boosting` | string | `"gbdt"` | `"gbdt"`, `"mixture"` | Set to `"mixture"` to enable MoE mode |
| `mixture_num_experts` | int | 4 | 2-10 | Number of expert models (K). Each expert is a separate GBDT. |
| `mixture_e_step_alpha` | float | 1.0 | 0.1-5.0 | Weight for loss term in E-step responsibility update. Higher = more weight on prediction accuracy vs gate probability. |
| `mixture_e_step_mode` | string | `"em"` | `"em"`, `"loss_only"`, `"gate_only"` | E-step mode. `"em"`: gate probability + loss (standard EM). `"loss_only"`: assigns to best-fitting expert. `"gate_only"`: prevents Expert Collapse. |
| `mixture_e_step_loss` | string | `"auto"` | `"auto"`, `"l2"`, `"l1"`, `"quantile"` | Loss function for E-step. `"auto"`: infer from objective (fallback to L2). |
| `mixture_warmup_iters` | int | 5 | 0-50 | Warmup iterations with uniform `1/K` responsibilities, letting experts learn before specialization. |
| `mixture_gate_iters_per_round` | int | 1 | ≥1 | Number of gate training iterations per boosting round. |
| `mixture_load_balance_alpha` | float | 0.0 | 0.0-10.0 | Auxiliary load balancing coefficient (`s_ik -= α_lb * log(load_k * K)`). Recommended: 0.1-1.0 for token-choice. |
| `mixture_balance_factor` | int | 10 | 2-20 | Load balancing aggressiveness. Minimum expert usage = `1/(factor × K)`. Lower = more aggressive. |
| `mixture_r_smoothing` | string | `"none"` | `"none"`, `"ema"`, `"markov"`, `"momentum"` | Responsibility smoothing for time-series stability. **Default `"none"`**. |
| `mixture_smoothing_lambda` | float | 0.0 | 0.0-1.0 | Smoothing strength. Only used when `mixture_r_smoothing != "none"`. |
| `mixture_gate_entropy_lambda` | float | 0.0 | 0.0-0.1 | Encourages gate to produce uncertain predictions, preventing premature expert collapse. |
| `mixture_expert_dropout_rate` | float | 0.0 | 0.0-0.3 | Randomly drops experts during training to force all experts to be useful. |
| `mixture_hard_m_step` | bool | `true` | `true`, `false` | Hard (argmax) assignment in M-step. Each sample's gradient goes only to the expert with highest responsibility. Prevents Expert Collapse. |
| `mixture_diversity_lambda` | float | 0.0 | 0.0-0.5 | Diversity regularization pushing expert predictions apart: `grad += λ * Σ_{j≠k} r_j * (f_k - f_j) / (K-1)`. |
| `mixture_init` | string | `"uniform"` | see [Initialization](#initialization-methods) | Initial responsibility scheme |

## Gate Parameters

The gate is a **multiclass classifier** (K classes = K experts). It uses shallow trees by default to prevent overfitting on routing.

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `mixture_gate_type` | string | `"gbdt"` | `"gbdt"`, `"none"`, `"leaf_reuse"` | Gate implementation. `"gbdt"`: full multiclass GBDT (most accurate). `"none"`: skip gate (fastest, no out-of-sample routing). `"leaf_reuse"`: reuse expert leaves for routing. |
| `mixture_gate_max_depth` | int | 3 | 2-10 | Max depth of gate trees. |
| `mixture_gate_num_leaves` | int | 8 | 4-64 | Number of leaves in gate trees. |
| `mixture_gate_learning_rate` | float | 0.1 | 0.01-0.5 | LR for gate. Can be higher than experts since gate trees are shallower. |
| `mixture_gate_lambda_l2` | float | 1.0 | 0.001-10.0 | L2 regularization for gate. |
| `mixture_gate_retrain_interval` | int | 10 | ≥1 | Only used with `mixture_gate_type="leaf_reuse"`. |

**Design rationale**: gate handles routing only; experts handle prediction accuracy. Shallow gate trees prevent memorizing sample→expert mappings.

## Smoothing Methods

| Method | Formula | Use Case |
|--------|---------|----------|
| `none` | `r_t = r_t` (no change) | i.i.d. data, regime determinable from X |
| `ema` | `r_t = λ·r_{t-1} + (1-λ)·r_t` | Time-series with persistent regimes |
| `markov` | `r_t ∝ r_t · (A·r_{t-1})` | Regime transitions follow Markov chain |
| `momentum` | `r_t = λ·r_{t-1} + (1-λ)·r_t + β·Δr` | Trending regime changes |

## Initialization Methods

| Mode | Description |
|------|-------------|
| `uniform` (default) | Equal `1/K` responsibility, symmetry broken by per-expert seeds |
| `random` | Random assignment of each sample to one expert |
| `quantile` | Assign by label quantiles (y-dependent — caution: target leakage) |
| `balanced_kmeans` | K-means++ on features, then balanced assignment (N/K per cluster) |
| `gmm` | Gaussian Mixture Model soft clustering (aligns with EM theory) |
| `tree_hierarchical` | Deep tree → leaf clustering → hierarchical merge into K groups |

## Prediction APIs

| Method | Output Shape | Description |
|--------|--------------|-------------|
| `predict(X)` | `(N,)` | Final prediction: weighted mixture of expert predictions |
| `predict_regime(X)` | `(N,)` int | Most likely regime index: `argmax_k(gate_proba)` |
| `predict_regime_proba(X)` | `(N, K)` | Gate probabilities for each expert (sums to 1) |
| `predict_expert_pred(X)` | `(N, K)` | Individual prediction from each expert |
| `predict_markov(X)` | `(N,)` | Prediction with Markov-smoothed regime switching |
| `predict_regime_proba_markov(X)` | `(N, K)` | Gate probabilities with Markov smoothing |
| `is_mixture()` | `bool` | Check if model is MoE |
| `num_experts()` | `int` | K |

**Prediction output mode** (`mixture_predict_output` parameter):

| Mode | Output | Description |
|------|--------|-------------|
| `"value"` (default) | `ŷ` only | Standard prediction |
| `"value_and_regime"` | `ŷ` + regime index | Prediction with argmax regime |
| `"all"` | `ŷ` + regime probabilities + expert predictions | Full diagnostic output |

## Auto-applied Settings

The library forces some settings to avoid known footguns:

| When | Forced setting | Why |
|------|----------------|-----|
| `mixture_hard_m_step=true` | per-expert `bagging_fraction=1.0`, `bagging_freq=0` | Sparse activation already restricts each expert; double-bagging produces degenerate histograms (#16). Gate-side bagging values are still respected. |
| `use_quantized_grad=true` (under MoE) | `quant_train_renew_leaf=true` on every expert + gate | Without renewal the quantized leaf-output path is biased by sparse-activation `hess≈1e-12` rows, causing 3-20× RMSE blow-up. |
| `mixture_gate_type="none"` | E-step runs in `loss_only` regardless of `mixture_e_step_mode` | No gate probabilities to weight by. |
