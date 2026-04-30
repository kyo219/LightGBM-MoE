# Expert Collapse Prevention & Regime Diagnostics

When experts converge to similar predictions or one expert dominates, the MoE has effectively collapsed into a single model. This page covers the prevention knobs and the diagnostic tool to detect it.

---

## 1. Prevention Parameters

| Parameter | When to Use | Effect |
|-----------|-------------|--------|
| `mixture_gate_entropy_lambda` | Gate assigns all samples to one expert early | Forces gate to be less confident, giving experts time to differentiate |
| `mixture_expert_dropout_rate` | One expert dominates and others stop learning | Randomly disables experts during training, forcing all to be useful |
| `mixture_diversity_lambda` | Experts predict similar values | Adds gradient penalty pushing expert outputs apart |
| `mixture_hard_m_step=true` (default) | — | Each sample's gradient goes only to argmax expert. Already enforced by default. |
| `mixture_routing_mode='expert_choice'` | One expert gets all samples | Each expert selects its top samples (perfect load balance) |

### Example: anti-collapse config

```python
params = {
    'boosting': 'mixture',
    'mixture_num_experts': 2,
    'objective': 'regression',

    # Collapse prevention
    'mixture_gate_entropy_lambda': 0.05,   # Encourage uncertain gate predictions
    'mixture_expert_dropout_rate': 0.2,    # 20% chance to drop each expert per iteration
    'mixture_diversity_lambda': 0.3,       # Push expert predictions apart

    # Other helpful settings
    'mixture_warmup_iters': 20,            # Allow experts to differentiate first
    'mixture_balance_factor': 5,           # More aggressive load balancing
}
```

### How they work

**Gate Entropy Regularization** (`mixture_gate_entropy_lambda`):
- Adds a penalty when gate is too confident: `grad += λ * (p - 1/K)`
- Pushes gate probabilities toward uniform `1/K`
- Effect decreases as experts become genuinely specialized

**Expert Dropout** (`mixture_expert_dropout_rate`):
- Each iteration, randomly drops experts (zero gradients)
- Dropped experts don't update, forcing others to cover their samples
- At least one expert is always kept
- Similar to dropout in neural networks

**Diversity Regularization** (`mixture_diversity_lambda`):
- Adds: `grad += λ * Σ_{j≠k} r_j * (f_k - f_j) / (K-1)`
- Each expert's gradient gets pushed away from the weighted average of others

---

## 2. Collapse Stopper (Optuna Callback)

Use `expert_collapse_stopper` to prune Optuna trials whose experts collapsed:

```python
from examples.benchmark import expert_collapse_stopper
import optuna

callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=False),
    expert_collapse_stopper(
        X_sample,                    # Subsample for efficiency
        corr_threshold=0.7,          # Max pairwise expert correlation
        min_expert_ratio=0.05,       # Min utilization per expert
        check_every=20,              # Check every N iterations
        min_iters=50,                # Skip early iterations (high corr is normal)
    ),
]

try:
    model = lgb.train(params, train_data, callbacks=callbacks)
except lgb.EarlyStopException:
    raise optuna.TrialPruned("Expert collapse detected")
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `corr_threshold` | 0.7 | Prune if max pairwise expert correlation > threshold |
| `min_expert_ratio` | 0.05 | Prune if any expert utilization < 5% |
| `check_every` | 20 | Check frequency (iterations) |
| `min_iters` | 50 | Skip initial iterations |

---

## 3. Post-Hoc Quality Filter

For Optuna setups that don't want a callback (or for analyzing already-trained models):

```python
import numpy as np

def compute_model_quality(model, X_val):
    """Quality metrics for an MoE model (no labels needed)."""
    gate_proba = model.predict_regime_proba(X_val)      # (N, K)
    expert_preds = model.predict_expert_pred(X_val)     # (N, K)
    K = gate_proba.shape[1]

    # 1. Expert correlation (collapse detection)
    correlations = []
    for i in range(K):
        for j in range(i + 1, K):
            corr = np.corrcoef(expert_preds[:, i], expert_preds[:, j])[0, 1]
            correlations.append(corr)
    max_corr = max(correlations) if correlations else 0.0

    # 2. Gate entropy (routing confidence)
    eps = 1e-10
    entropy = -np.sum(gate_proba * np.log(gate_proba + eps), axis=1)
    normalized_entropy = entropy / np.log(K)
    return {'expert_corr_max': max_corr, 'gate_entropy': normalized_entropy.mean()}

# Inside Optuna objective:
quality = compute_model_quality(model, X_val)
if quality['expert_corr_max'] > 0.8:
    raise optuna.TrialPruned("Expert collapse detected")
if quality['gate_entropy'] > 0.6:
    raise optuna.TrialPruned("Gate confusion detected")
```

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| `expert_corr_max` | < 0.8 (strict: < 0.7) | Experts should predict differently |
| `gate_entropy` | < 0.6 (strict: < 0.5) | Gate should route with confidence |

---

## 4. Regime Diagnostics (`diagnose_moe`)

After training, `diagnose_moe` answers "**is the model actually working as a switching model?**" without ground-truth regime labels.

### Usage

```python
import lightgbm_moe as lgb

model = lgb.train(params, train_data, num_boost_round=100)

# Print full report
result = lgb.diagnose_moe(model, X, y)

# Silent mode — returns dict only
result = lgb.diagnose_moe(model, X, y, print_report=False)
```

### Output Example

```
MoE Regime Diagnostics
======================
Model: K=2 experts

[1] Gate Entropy
    Mean entropy       : 0.412 / 0.693 (max)
    Confidence ratio   : 61.2%

[2] Expert Specialization
    Specialization rate: 72.4%
    Mean loss improvement: 18.3%

[3] Routing Gain
    MoE RMSE           : 1.2340
    Expert RMSEs       : E0=1.4512  E1=1.3801
    Routing gain       : +10.6%

[4] Expert Correlation
    Pairwise corr      : 0.72 (max)  0.72 (min)
    Collapsed           : No

[5] Expert Utilization
    E0: 48.2%   E1: 51.8%

Verdict: Effective Switching
```

### Diagnostic Metrics

**[1] Gate Entropy** — Is the gate making confident routing decisions?

| Metric | Meaning |
|--------|---------|
| `mean_entropy` | Average Shannon entropy across all samples. Lower = more decisive |
| `max_entropy` | Theoretical max `log(K)`. For K=2, this is 0.693 |
| `confidence_ratio` | Fraction of samples with `H < 0.3 × max_entropy` |

**[2] Expert Specialization** — Does the assigned expert actually predict better than the others?

| Metric | Meaning |
|--------|---------|
| `specialization_rate` | Fraction where assigned expert beats average of others. >0.6 is good |
| `mean_loss_improvement` | When the assigned expert wins, how much better on average |

**[3] Routing Gain** — Does the MoE mixture beat the best single expert?

| Metric | Meaning |
|--------|---------|
| `moe_rmse` | RMSE of weighted mixture |
| `expert_rmses` | RMSE per expert |
| `best_single_rmse` | Best individual expert RMSE |
| `routing_gain` | `(best_single - moe) / best_single * 100`. Positive = mixture is better |

**[4] Expert Correlation** — Have the experts collapsed?

| Metric | Meaning |
|--------|---------|
| `expert_corr_max` | Highest pairwise correlation. >0.99 = collapsed |
| `expert_corr_min` | Lowest pairwise correlation |
| `expert_collapsed` | `True` if `expert_corr_max > 0.99` |

**[5] Expert Utilization** — Are all experts being used?

| Metric | Meaning |
|--------|---------|
| `utilization` | Assignment ratio per expert (sums to 1.0) |
| `utilization_min` | Minimum utilization across experts |
| `any_underutilized` | `True` if any expert gets < 5% |

### Verdict

| Verdict | Condition | Interpretation |
|---------|-----------|----------------|
| **Effective Switching** | `specialization_rate > 0.6` AND `confidence_ratio > 0.5` AND `routing_gain > 1%` AND NOT `collapsed` | Working as intended |
| **Not Switching (Collapsed)** | `collapsed` OR `utilization_min < 0.01` OR `specialization_rate < 0.3` | Experts collapsed, dead expert, or random routing |
| **Weak Switching** | Everything else | Some switching but not strong. Try increasing `mixture_diversity_lambda` or adjusting gate LR |

### Return Value

`diagnose_moe` returns a dict containing all metrics above plus `K`, `entropy_per_sample`, and the `verdict` string. See the source for the full schema.
