<img src=https://github.com/microsoft/LightGBM/blob/master/docs/logo/LightGBM_logo_black_text.svg width=300 />

LightGBM-MoE
============

**A regime-switching / Mixture-of-Experts extension of LightGBM.**

[English](#english) | [Japanese (日本語)](#japanese)

---

<a name="english"></a>
## English

### Overview

LightGBM-MoE is a fork of [Microsoft LightGBM](https://github.com/microsoft/LightGBM) that implements **Mixture-of-Experts (MoE) / Regime-Switching GBDT** natively in C++.

```
ŷ(x) = Σₖ gₖ(x) · fₖ(x)
```

Where:
- `fₖ(x)`: Expert k's prediction (K regression GBDTs)
- `gₖ(x)`: Gate's routing probability for expert k (softmax output)
- `K`: Number of experts (hyperparameter)

### Requirements

- **Python**: 3.10 or later
- **OS**: Linux (x86_64, aarch64), macOS (Intel, Apple Silicon)
- **Dependencies**: numpy, scipy (automatically installed)

### Installation

**Recommended: Install directly from GitHub** (builds from source, requires CMake):

```bash
pip install git+https://github.com/kyo219/LightGBM-MoE.git
```

**Alternative: Build manually from source**:

```bash
git clone https://github.com/kyo219/LightGBM-MoE.git
cd LightGBM-MoE
pip install ./python-package
```

**For development (editable install)**:

```bash
git clone https://github.com/kyo219/LightGBM-MoE.git
cd LightGBM-MoE/python-package
pip install -e .
```

> **Note**: Building from source requires CMake (3.16+) and a C++ compiler (GCC, Clang, or Apple Clang).

### Quick Start

```python
import lightgbm_moe as lgb

params = {
    'boosting': 'mixture',           # Enable MoE mode
    'mixture_num_experts': 2,        # Number of experts
    'mixture_r_smoothing': 'ema',    # Smoothing method
    'mixture_smoothing_lambda': 0.5, # Smoothing strength
    'objective': 'regression',
}

train_data = lgb.Dataset(X_train, label=y_train)
model = lgb.train(params, train_data, num_boost_round=100)

# Predictions
y_pred = model.predict(X_test)                     # Weighted mixture
regime = model.predict_regime(X_test)              # Regime index (argmax)
regime_proba = model.predict_regime_proba(X_test)  # Gate probabilities (N, K)
expert_preds = model.predict_expert_pred(X_test)   # Expert predictions (N, K)
```

### Training with Validation & Early Stopping

```python
import lightgbm_moe as lgb

params = {
    'boosting': 'mixture',
    'mixture_num_experts': 2,
    'objective': 'regression',
    'metric': 'rmse',
    'verbose': 1,
}

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

model = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[valid_data],
    valid_names=['valid'],
    callbacks=[lgb.early_stopping(stopping_rounds=10)]
)

print(f"Best iteration: {model.best_iteration}")
```

---

## API Reference

### MoE Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `boosting` | string | `"gbdt"` | `"gbdt"`, `"mixture"` | Set to `"mixture"` to enable MoE mode |
| `mixture_num_experts` | int | 4 | 2-10 | Number of expert models (K). Each expert is a separate GBDT that specializes in different data regimes. |
| `mixture_e_step_alpha` | float | 1.0 | 0.1-5.0 | Weight for loss term in E-step responsibility update. Higher = more weight on prediction accuracy vs gate probability. |
| `mixture_e_step_mode` | string | `"em"` | `"em"`, `"loss_only"` | E-step mode. `"em"`: use gate probability + loss (standard EM). `"loss_only"`: use only loss (simpler, assigns to best-fitting expert). |
| `mixture_warmup_iters` | int | 10 | 0-50 | Number of warmup iterations. During warmup, responsibilities are uniform (1/K) to allow experts to learn before specialization. |
| `mixture_balance_factor` | int | 10 | 2-20 | Load balancing aggressiveness. Minimum expert usage = 1/(factor × K). Lower = more aggressive balancing. Recommended: 5-7. |
| `mixture_r_smoothing` | string | `"none"` | `"none"`, `"ema"`, `"markov"`, `"momentum"` | Responsibility smoothing method for time-series stability. **Recommended: `"none"`** (see note below). |
| `mixture_smoothing_lambda` | float | 0.0 | 0.0-1.0 | Smoothing strength. Only used when `mixture_r_smoothing` is not `"none"`. Higher = more smoothing (slower regime transitions). |
| `mixture_gate_entropy_lambda` | float | 0.0 | 0.0-1.0 | Gate entropy regularization. Encourages gate to produce more uncertain predictions, preventing premature expert collapse. **Recommended: 0.01-0.1**. |
| `mixture_expert_dropout_rate` | float | 0.0 | 0.0-1.0 | Expert dropout rate. Randomly drops experts during training to force all experts to be useful. **Recommended: 0.1-0.3**. |

> **Important: Use `mixture_r_smoothing="none"` (default)**
>
> Smoothing methods (`ema`, `markov`, `momentum`) can cause **expert collapse** where all experts converge to similar predictions. In benchmarks with Optuna optimization, `smoothing=none` consistently achieves good expert separation (correlation ~0.02, regime accuracy ~98%), while other smoothing methods often collapse (correlation ~0.99, regime accuracy ~50%).

### Expert Collapse Prevention (Advanced)

If experts are collapsing (all producing similar predictions), try these new parameters:

| Parameter | When to Use | Effect |
|-----------|-------------|--------|
| `mixture_gate_entropy_lambda` | Gate assigns all samples to one expert early | Forces gate to be less confident, giving experts time to differentiate |
| `mixture_expert_dropout_rate` | One expert dominates and others stop learning | Forces all experts to be useful by randomly disabling them during training |

**Example: Preventing collapse**

```python
params = {
    'boosting': 'mixture',
    'mixture_num_experts': 2,
    'objective': 'regression',

    # Collapse prevention
    'mixture_gate_entropy_lambda': 0.05,  # Encourage uncertain gate predictions
    'mixture_expert_dropout_rate': 0.2,   # 20% chance to drop each expert per iteration

    # Other recommended settings
    'mixture_warmup_iters': 20,           # Allow experts to differentiate first
    'mixture_balance_factor': 5,          # More aggressive load balancing
}
```

**How these work:**

1. **Gate Entropy Regularization** (`mixture_gate_entropy_lambda`):
   - Adds a penalty when gate is too confident: `grad += λ * (p - 1/K)`
   - Pushes gate probabilities toward uniform distribution (1/K)
   - Effect decreases as experts become genuinely specialized

2. **Expert Dropout** (`mixture_expert_dropout_rate`):
   - Each iteration, randomly drops experts (they receive zero gradients)
   - Dropped experts don't update, forcing others to cover their samples
   - At least one expert is always kept
   - Similar to dropout in neural networks

### Early Stopping

MoE supports validation-based early stopping, useful for hyperparameter tuning with Optuna.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `early_stopping_round` | int | 0 | Stop training if validation metric doesn't improve for N rounds. Set via `lgb.early_stopping()` callback. |
| `first_metric_only` | bool | False | Only use the first metric for early stopping (when multiple metrics specified). |

**Usage with callbacks:**

```python
model = lgb.train(
    params,
    train_data,
    valid_sets=[valid_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=10),  # Stop after 10 rounds without improvement
        lgb.log_evaluation(period=10),           # Log every 10 iterations
    ]
)
```

**Usage with Optuna:**

```python
def objective(trial):
    params = {
        'boosting': 'mixture',
        'mixture_num_experts': trial.suggest_int('num_experts', 2, 4),
        'objective': 'regression',
        'metric': 'rmse',
        'verbose': -1,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=20)]
    )

    return model.best_score['valid_0']['rmse']
```

### Per-Expert Hyperparameters (Advanced)

These parameters allow each expert to have different tree **structural** configurations. If not specified, all experts share the same hyperparameters.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mixture_expert_max_depths` | string | `""` | Comma-separated max_depth for each expert. Must have exactly K values if specified. |
| `mixture_expert_num_leaves` | string | `""` | Comma-separated num_leaves for each expert. Must have exactly K values if specified. |
| `mixture_expert_min_data_in_leaf` | string | `""` | Comma-separated min_data_in_leaf for each expert. Controls tree granularity. |
| `mixture_expert_min_gain_to_split` | string | `""` | Comma-separated min_gain_to_split for each expert. Controls split aggressiveness. |
| `mixture_expert_extra_trees` | string | `""` | Comma-separated 0/1 for each expert. Enables extremely randomized trees per expert. |

#### Same Hyperparameters for All Experts (Default)

When per-expert parameters are not specified, all experts share the base hyperparameters:

```python
params = {
    'boosting': 'mixture',
    'mixture_num_experts': 3,
    'max_depth': 5,           # All 3 experts use max_depth=5
    'num_leaves': 31,         # All 3 experts use num_leaves=31
    'min_data_in_leaf': 20,   # All 3 experts use min_data_in_leaf=20
}
```

#### Different Hyperparameters per Expert

Specify comma-separated values (one per expert) to customize each expert:

```python
params = {
    'boosting': 'mixture',
    'mixture_num_experts': 3,
    # Expert 0: coarse (high min_data), Expert 1: medium, Expert 2: fine (low min_data)
    'mixture_expert_max_depths': '3,5,7',
    'mixture_expert_num_leaves': '8,16,32',
    'mixture_expert_min_data_in_leaf': '50,20,5',      # coarse → fine
    'mixture_expert_min_gain_to_split': '0.1,0.01,0.001',  # conservative → aggressive
}
```

This allows experts to have different structural capacities:
- **Coarse expert** (high min_data_in_leaf): captures broad patterns, less prone to overfitting
- **Fine expert** (low min_data_in_leaf): captures detailed patterns, good for complex regimes

#### Training Behavior with Per-Expert Hyperparameters

**EM iteration structure**: Each boosting iteration adds ONE tree to each expert:

```
TrainOneIter() {
  1. Forward()      → Compute predictions from all experts
  2. EStep()        → Update responsibilities
  3. MStepExperts() → Each expert adds 1 tree (sequential)
  4. MStepGate()    → Gate adds 1 tree
}
```

**What changes / doesn't change with different hyperparameters:**

| Aspect | Deep Expert | Shallow Expert |
|--------|-------------|----------------|
| EM iterations | Same | Same |
| Number of trees | Same | Same |
| Time per tree | Longer | Shorter |
| Expressiveness per tree | Higher | Lower |

**Key insight**: `num_boost_round=100` means each expert builds 100 trees, regardless of depth settings. The per-expert hyperparameters control **expressiveness per tree**, not the number of trees.

```
num_boost_round = 100:
  Expert 0 (shallow): 100 shallow trees → simple patterns
  Expert 1 (deep):    100 deep trees    → complex patterns
                ↓
        Same 100 EM iterations
```

**Training time**: Currently experts are trained sequentially in each iteration, so the deepest/most complex expert becomes the bottleneck. Total time ≈ sum of all expert tree build times per iteration.

#### Initialization and Symmetry Breaking

By default, MoE uses **uniform initialization**: all samples start with equal responsibility `1/K` for all experts.

```
Initial state (K=2):
  Sample 0: r = [0.5, 0.5]  (equal for both experts)
  Sample 1: r = [0.5, 0.5]
  ...
```

**Symmetry breaking** is achieved through per-expert random seeds:
```cpp
expert_configs_[k]->seed = config_->seed + k + 1;  // Different seed per expert
```

This means experts naturally differentiate as training progresses, without relying on label-based initialization (which could leak target information).

Available initialization modes (`mixture_init` parameter):
| Mode | Description |
|------|-------------|
| `uniform` (default) | Equal `1/K` responsibility, symmetry broken by per-expert seeds |
| `random` | Randomly assign each sample to one expert |
| `quantile` | Assign by label quantiles (y-dependent, use with caution) |

#### Optuna Optimization Examples

**Standard MoE** (shared hyperparameters across experts):

```python
import optuna
import lightgbm_moe as lgb
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'boosting': 'mixture',
        'objective': 'regression',
        'verbose': -1,
        # Tree structure (shared by all experts)
        'num_leaves': trial.suggest_int('num_leaves', 8, 128),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        # MoE specific
        'mixture_num_experts': trial.suggest_int('mixture_num_experts', 2, 4),
        'mixture_e_step_alpha': trial.suggest_float('mixture_e_step_alpha', 0.1, 2.0),
        'mixture_warmup_iters': trial.suggest_int('mixture_warmup_iters', 5, 30),
        'mixture_balance_factor': trial.suggest_int('mixture_balance_factor', 2, 10),
        # Smoothing (optional, for time-series)
        'mixture_r_smoothing': trial.suggest_categorical(
            'mixture_r_smoothing', ['none', 'ema', 'markov']
        ),
    }
    if params['mixture_r_smoothing'] != 'none':
        params['mixture_smoothing_lambda'] = trial.suggest_float(
            'mixture_smoothing_lambda', 0.1, 0.9
        )

    # Train and evaluate
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=100)
    pred = model.predict(X_val)
    return mean_squared_error(y_val, pred)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

**Per-Expert MoE** (different tree structure per expert):

```python
def objective_per_expert(trial):
    num_experts = trial.suggest_int('mixture_num_experts', 2, 4)

    # Per-expert tree structure
    max_depths = [trial.suggest_int(f'max_depth_{k}', 3, 12) for k in range(num_experts)]
    num_leaves = [trial.suggest_int(f'num_leaves_{k}', 8, 128) for k in range(num_experts)]
    min_data = [trial.suggest_int(f'min_data_{k}', 5, 100) for k in range(num_experts)]

    params = {
        'boosting': 'mixture',
        'objective': 'regression',
        'verbose': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        # MoE specific
        'mixture_num_experts': num_experts,
        'mixture_e_step_alpha': trial.suggest_float('mixture_e_step_alpha', 0.1, 2.0),
        'mixture_warmup_iters': trial.suggest_int('mixture_warmup_iters', 5, 30),
        'mixture_balance_factor': trial.suggest_int('mixture_balance_factor', 2, 10),
        # Per-expert structure (comma-separated)
        'mixture_expert_max_depths': ','.join(map(str, max_depths)),
        'mixture_expert_num_leaves': ','.join(map(str, num_leaves)),
        'mixture_expert_min_data_in_leaf': ','.join(map(str, min_data)),
    }

    # Train and evaluate
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=100)
    pred = model.predict(X_val)
    return mean_squared_error(y_val, pred)

# Note: Per-expert adds K×3 more hyperparameters, so needs more trials (e.g., 200+)
study = optuna.create_study(direction='minimize')
study.optimize(objective_per_expert, n_trials=200)
```

**Tip**: Start with standard MoE. Per-expert requires more trials to converge due to larger search space.

#### Role-based Per-Expert (Recommended for Optuna)

The naive per-expert approach has a problem: **each expert's parameters are independent**, so you might end up with similar experts (e.g., all with similar depth and leaves). This wastes the MoE's potential for specialization.

**Solution**: Assign each expert a distinct "role" (personality) and constrain the search space accordingly.

```
             num_leaves
              Low    High
max_depth Low  E0     E1    ← E1: shallow but wide
          High E2     E3    ← E2: deep but narrow
```

This ensures diversity: E1 (shallow × many leaves) captures different patterns than E2 (deep × few leaves).

```python
def suggest_moe_expert_params(
    trial,
    num_experts: int,
    depth_range: tuple = (2, 15),
    leaves_range: tuple = (4, 128),
    min_data_range: tuple = (5, 100),
    use_extra_trees: bool = True,
):
    """
    Assign distinct "roles" to each expert while searching concrete values with Optuna.

    - Reduces search space: K×3 params → 4 params (depth_low/high, leaves_low/high)
    - Guarantees diversity: each expert has a different (depth, leaves) combination
    - Full range search: all experts can be "deep" or "shallow", but within each trial
      they are constrained to have relative differences (low < high)
    - Extra trees: shallow experts use extra_trees for diversity, deep experts don't
    """

    # Search from full range, but constrain low < high within each trial
    # This allows: Trial A (3 vs 12), Trial B (10 vs 14), Trial C (2 vs 4)
    depth_low = trial.suggest_int('depth_low', depth_range[0], depth_range[1] - 1)
    depth_high = trial.suggest_int('depth_high', depth_low + 1, depth_range[1])

    leaves_low = trial.suggest_int('leaves_low', leaves_range[0], leaves_range[1] - 1)
    leaves_high = trial.suggest_int('leaves_high', leaves_low + 1, leaves_range[1])

    # Role patterns based on K
    # (depth_level, leaves_level): 0=low, 1=high, 0.5=mid
    PATTERNS = {
        2: [(0, 0), (1, 1)],                                  # diagonal: simple vs complex
        3: [(0, 0), (0, 1), (1, 1)],                          # simple, shallow×wide, complex
        4: [(0, 0), (0, 1), (1, 0), (1, 1)],                  # all 4 quadrants
        5: [(0, 0), (0, 1), (1, 0), (1, 1), (0.5, 0.5)],      # 4 quadrants + center
        6: [(0, 0), (0, 1), (1, 0), (1, 1), (0, 0.5), (1, 0.5)],
    }

    if num_experts in PATTERNS:
        patterns = PATTERNS[num_experts]
    else:
        # K > 6: cycle through 4 quadrants + interpolations
        base = [(0, 0), (0, 1), (1, 0), (1, 1)]
        patterns = (base * ((num_experts // 4) + 1))[:num_experts]

    def interp(low, high, t):
        return round(low + t * (high - low))

    depths, leaves_list, min_datas, extra_trees = [], [], [], []
    for d_level, l_level in patterns:
        depths.append(interp(depth_low, depth_high, d_level))
        leaves_list.append(interp(leaves_low, leaves_high, l_level))
        # min_data inversely correlated with depth (deep → small min_data)
        min_datas.append(interp(min_data_range[1], min_data_range[0], d_level))
        # extra_trees for shallow experts (more randomness), off for deep (precision)
        extra_trees.append(1 if d_level < 0.5 else 0)

    result = {
        'mixture_expert_max_depths': ','.join(map(str, depths)),
        'mixture_expert_num_leaves': ','.join(map(str, leaves_list)),
        'mixture_expert_min_data_in_leaf': ','.join(map(str, min_datas)),
    }
    if use_extra_trees:
        result['mixture_expert_extra_trees'] = ','.join(map(str, extra_trees))
    return result


def objective_role_based(trial):
    num_experts = trial.suggest_int('num_experts', 2, 4)

    # Get role-based expert params (only 4 search params instead of K×3)
    expert_params = suggest_moe_expert_params(
        trial,
        num_experts=num_experts,
        depth_range=(2, 15),
        leaves_range=(4, 128),
    )

    params = {
        'boosting': 'mixture',
        'objective': 'regression',
        'verbose': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'mixture_num_experts': num_experts,
        'mixture_e_step_alpha': trial.suggest_float('mixture_e_step_alpha', 0.1, 2.0),
        'mixture_warmup_iters': trial.suggest_int('mixture_warmup_iters', 5, 30),
        **expert_params,  # role-based expert structure
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=100)
    pred = model.predict(X_val)
    return mean_squared_error(y_val, pred)

study = optuna.create_study(direction='minimize')
study.optimize(objective_role_based, n_trials=100)
```

**Example output (K=4):**
```
Search result: depth_low=3, depth_high=10, leaves_low=8, leaves_high=64

E0: depth=3,  leaves=8,  min_data=100, extra_trees=1  (shallow × few)   → fast, randomized
E1: depth=3,  leaves=64, min_data=100, extra_trees=1  (shallow × many)  → wide, randomized
E2: depth=10, leaves=8,  min_data=5,   extra_trees=0  (deep × few)      → narrow, precise
E3: depth=10, leaves=64, min_data=5,   extra_trees=0  (deep × many)     → complex, precise
```

**Benefits:**
- Search space: K×3 parameters → 4 parameters (dramatic reduction)
- Diversity guaranteed: each expert has a distinct structural "personality"
- Interpretable: you know what each expert is designed to capture

#### Model Quality Filtering (Pruning Collapsed Models)

MoE models can fail in two ways: **Expert collapse** (all experts predict the same thing) and **Gate confusion** (gate can't decide which expert to use). Filter these out during Optuna optimization:

```python
import numpy as np

def compute_model_quality(model, X_val):
    """Compute quality metrics for MoE model (no labels needed)."""
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
    mean_entropy = normalized_entropy.mean()

    return {'expert_corr_max': max_corr, 'gate_entropy': mean_entropy}

def objective_with_quality_filter(trial):
    params = {
        'boosting': 'mixture',
        'mixture_num_experts': trial.suggest_int('num_experts', 2, 4),
        # ... other params ...
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=100)

    # Quality check (no labels needed - can use on live data)
    quality = compute_model_quality(model, X_val)

    # Prune collapsed or confused models
    if quality['expert_corr_max'] > 0.8:  # Experts too similar
        raise optuna.TrialPruned("Expert collapse detected")
    if quality['gate_entropy'] > 0.6:     # Gate can't decide
        raise optuna.TrialPruned("Gate confusion detected")

    pred = model.predict(X_val)
    return mean_squared_error(y_val, pred)
```

**Quality Thresholds:**

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| `expert_corr_max` | < 0.8 (strict: < 0.7) | Experts should predict differently |
| `gate_entropy` | < 0.6 (strict: < 0.5) | Gate should route with confidence |

**Interpretation of normalized entropy (K-independent):**

| Entropy | Gate Probability (K=2) | Meaning |
|---------|------------------------|---------|
| 0.3 | [0.88, 0.12] | High confidence |
| 0.5 | [0.77, 0.23] | Moderate confidence |
| 0.6 | [0.70, 0.30] | Acceptable |
| 0.8 | [0.57, 0.43] | Low confidence |

### Smoothing Methods

| Method | Formula | Use Case |
|--------|---------|----------|
| `none` | `r_t = r_t` (no change) | i.i.d. data, regime determinable from X |
| `ema` | `r_t = λ·r_{t-1} + (1-λ)·r_t` | Time-series with persistent regimes |
| `markov` | `r_t ∝ r_t · (A·r_{t-1})` | Regime transitions follow Markov chain |
| `momentum` | `r_t = λ·r_{t-1} + (1-λ)·r_t + β·Δr` | Trending regime changes |

### Prediction APIs

| Method | Output Shape | Description |
|--------|--------------|-------------|
| `predict(X)` | `(N,)` | Final prediction: weighted mixture of expert predictions |
| `predict_regime(X)` | `(N,)` int | Most likely regime index: `argmax_k(gate_proba)` |
| `predict_regime_proba(X)` | `(N, K)` | Gate probabilities for each expert (sums to 1) |
| `predict_expert_pred(X)` | `(N, K)` | Individual prediction from each expert |

### SHAP Analysis for MoE Components

LightGBM-MoE provides APIs to extract individual component models (Gate and Experts) for SHAP analysis. This allows you to understand feature importance for each component separately.

#### Extracting Component Boosters

```python
import lightgbm_moe as lgb

# Train MoE model
model = lgb.train(params, train_data, num_boost_round=100)

# Extract individual components as standalone Boosters
gate_booster = model.get_gate_booster()           # Gate model
expert_0_booster = model.get_expert_booster(0)    # Expert 0
expert_1_booster = model.get_expert_booster(1)    # Expert 1

# Or get all components at once
boosters = model.get_all_boosters()
# Returns: {'gate': Booster, 'expert_0': Booster, 'expert_1': Booster, ...}
```

#### SHAP Analysis Example

```python
import shap
import lightgbm as standard_lgb  # Standard LightGBM required for SHAP
import tempfile

# Helper function to convert lightgbm_moe Booster to SHAP-compatible format
def to_shap_model(booster):
    """Convert lightgbm_moe Booster to standard lightgbm Booster for SHAP."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(booster.model_to_string(num_iteration=-1))
        temp_path = f.name
    return standard_lgb.Booster(model_file=temp_path)

# Train MoE model
model = lgb.train(params, train_data, num_boost_round=100)

# Get SHAP values for each component
boosters = model.get_all_boosters()

for name, booster in boosters.items():
    shap_model = to_shap_model(booster)
    explainer = shap.TreeExplainer(shap_model)
    shap_values = explainer.shap_values(X)

    # Handle multi-output models (Gate has K outputs)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]  # Use first class

    # Create beeswarm plot
    shap.summary_plot(shap_values, X, plot_type="dot", show=False)
    plt.title(f"SHAP: {name}")
    plt.savefig(f"shap_{name}.png")
    plt.close()
```

#### Component Booster APIs

| Method | Returns | Description |
|--------|---------|-------------|
| `get_gate_booster()` | `Booster` | Standalone Gate model (K-class classifier) |
| `get_expert_booster(k)` | `Booster` | Standalone Expert k model (regressor) |
| `get_all_boosters()` | `dict[str, Booster]` | All components: `{'gate': ..., 'expert_0': ..., ...}` |

#### Notes

- **Gate model**: Multi-class classifier (K outputs). SHAP returns 3D array `(N, features, K)`. Use first class or average.
- **Expert models**: Regression models. SHAP returns 2D array `(N, features)`.
- **Standard LightGBM required**: SHAP's `TreeExplainer` expects standard LightGBM Booster. The helper function above converts lightgbm_moe Booster to standard format via temporary file.

#### Benchmark SHAP Visualization

The benchmark script automatically generates SHAP beeswarm plots for the optimized MoE model:

```bash
python examples/benchmark.py --trials 100

# Skip SHAP visualization
python examples/benchmark.py --trials 100 --no-shap
```

Output files:
- `examples/shap_gate.png` - Gate feature importance
- `examples/shap_expert_0.png` - Expert 0 feature importance
- `examples/shap_expert_1.png` - Expert 1 feature importance
- `examples/moe_shap_beeswarm.png` - Combined comparison

#### SHAP Visualization Results (Synthetic Data)

**Gate Beeswarm Plot:**

![SHAP Gate](examples/shap_gate.png)

The Gate model learns to route samples based on **X1, X2, X3** - which matches the synthetic data's regime definition (`regime_score = 0.5*X1 + 0.3*X2 - 0.2*X3`).

**Component Feature Importance Comparison:**

![MoE SHAP Beeswarm](examples/moe_shap_beeswarm.png)

- **Gate**: X1, X2 are most important for regime classification
- **Expert 0**: X0 dominates (learns `5*X0 + 3*X0*X2 + 2*sin(2*X3) + 10`)
- **Expert 1**: X0, X3 are important (learns `-5*X0 - 2*X1² + 3*cos(2*X4) - 10`)

This confirms that the Gate correctly identifies regimes, and each Expert specializes in different functional relationships.

---

## Benchmark

**Setup**: 200 Optuna trials, 5-fold time-series CV, early stopping (50 rounds).

### RMSE Comparison

| Dataset | Standard | MoE | MoE-PE | Best Improvement |
|---------|----------|-----|--------|------------------|
| Synthetic | 5.0739 | **4.3049** | 4.7611 | +15.2% |
| Hamilton | 0.7205 | 0.7220 | **0.7204** | +0.0% |
| VIX | 0.0116 | **0.0116** | 0.0116 | +0.1% |

### Expert Differentiation (Regime Separation)

| Dataset | MoE K | MoE-PE K | MoE Corr (min/max) | MoE-PE Corr (min/max) | MoE Regime Acc | MoE-PE Regime Acc |
|---------|-------|----------|--------------------|-----------------------|----------------|-------------------|
| Synthetic | 2 | 3 | -0.42/-0.42 | -0.29/0.82 | 96.2% | 95.1% |
| Hamilton | 4 | 3 | 0.69/0.88 | 0.61/0.67 | 52.6% | 52.6% |
| VIX | 2 | 2 | 0.64/0.64 | 0.99/0.99 | 53.0% | 52.1% |

- **K**: Number of experts selected by Optuna
- **Expert Corr (min/max)**: Pairwise correlation between expert predictions (lower = more differentiated, negative = opposite predictions)
- **Regime Acc**: Classification accuracy of predicted regime vs true regime

**Key Findings**:
- MoE achieves **+15.2% improvement** on Synthetic data with expert correlation of **-0.42** (opposite predictions!)
- Both MoE and MoE-PE achieve **96%+ regime accuracy** on Synthetic data
- Latent regime data (Hamilton, VIX) shows limited improvement as expected (regime not determinable from X)

### Selected Hyperparameters (Synthetic Dataset)

**MoE (Shared Tree Structure):**
- num_experts: 2, max_depth: 10, num_leaves: 20, learning_rate: 0.064, alpha: 1.44, warmup: 5

**MoE-PerExpert (Per-Expert Tree Structure):**
- num_experts: 3, learning_rate: 0.068, alpha: 0.52, warmup: 5

| Expert | max_depth | num_leaves | min_data_in_leaf |
|--------|-----------|------------|------------------|
| E0 | 3 | 21 | 79 |
| E1 | 4 | 79 | 66 |
| E2 | 6 | 12 | 65 |

### Run Benchmark

```bash
# Full benchmark (Standard vs MoE vs MoE-PE, 100 trials default)
python examples/benchmark.py

# More trials for better optimization
python examples/benchmark.py --trials 200

# Quick test
python examples/benchmark.py --trials 10

# Output to markdown
python examples/benchmark.py --trials 200 --output-md BENCHMARK.md
```

### Visualization

![Benchmark Results](examples/benchmark_results.png)

- **Left**: Regime separation (% samples routed to each expert by true regime)
- **Center**: Expert prediction scatter (color = true regime)
- **Right**: RMSE comparison across methods

---

## When to Use MoE

**MoE is effective when:**
- Regime is determinable from features (X)
- Different regimes follow fundamentally different functions
- You have sufficient data for each regime

**MoE is NOT effective when:**
- Regime is latent (hidden Markov, unobserved states)
- Standard GBDT already captures the pattern
- Data has no clear regime structure

---

## Technical Deep Dive

### 1. How MoE is Achieved: Architecture

The MoE model consists of **K Expert GBDTs** and **1 Gate GBDT**:

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
          │               │               │
          │  f₀(x)        │  f₁(x)        │ logits
          ▼               ▼               ▼
    ┌─────────────────────────────────────────────┐
    │           Weighted Combination              │
    │  ŷ = Σₖ softmax(gate_logits)ₖ · fₖ(x)      │
    └─────────────────────────────────────────────┘
```

**Key implementation details** (`src/boosting/mixture_gbdt.cpp`):

| Component | Implementation | Lines |
|-----------|----------------|-------|
| Expert GBDTs | `std::vector<std::unique_ptr<GBDT>> experts_` | Same objective as mixture |
| Gate GBDT | `std::unique_ptr<GBDT> gate_` | Multiclass with K classes |
| Responsibilities | `std::vector<double> responsibilities_` | N × K soft assignments |
| Load Balancing | `std::vector<double> expert_bias_` | Prevents expert collapse |

### 2. Gate Learning Mechanism

The Gate is trained as a **K-class classification GBDT** using LightGBM's multiclass objective:

```cpp
// Gate config setup (mixture_gbdt.cpp:86-93)
gate_config_->objective = "multiclass";
gate_config_->num_class = num_experts_;
gate_config_->max_depth = config_->mixture_gate_max_depth;      // default: 3
gate_config_->num_leaves = config_->mixture_gate_num_leaves;    // default: 8
gate_config_->learning_rate = config_->mixture_gate_learning_rate;  // default: 0.1
```

**Training process** (M-Step for Gate, `MStepGate()` at line 526):

1. **Create pseudo-labels**: `z_i = argmax_k(r_ik)` (hard assignment from responsibilities)
2. **Compute gradients**: Softmax cross-entropy gradients
   ```cpp
   // For each sample i and class k:
   if (k == label) {
       grad[i,k] = p_k - 1.0;   // Gradient for correct class
   } else {
       grad[i,k] = p_k;         // Gradient for other classes
   }
   hess[i,k] = p_k * (1 - p_k); // Hessian: softmax derivative
   ```
3. **Update Gate**: `gate_->TrainOneIter(gate_grad, gate_hess)`

The Gate learns to predict **which Expert should handle each sample**, based on features X.

### 3. EM-Style Training Algorithm

Each iteration follows an **Expectation-Maximization (EM)** style update:

```
┌─────────────────────────────────────────────────────────────┐
│                    TrainOneIter()                           │
├─────────────────────────────────────────────────────────────┤
│ 1. Forward()     → Compute expert_pred[N,K], gate_proba[N,K]│
│ 2. EStep()       → Update responsibilities r[N,K]           │
│ 3. Smoothing()   → Apply EMA/Markov/Momentum (optional)     │
│ 4. MStepExperts()→ Train experts with weighted gradients    │
│ 5. MStepGate()   → Train gate with pseudo-labels            │
└─────────────────────────────────────────────────────────────┘
```

**E-Step** (line 350-383): Update responsibilities based on how well each expert fits each sample:

```cpp
// Score calculation depends on mixture_e_step_mode:

// Mode "em" (default): use gate probability + loss
s_ik = log(gate_proba[i,k] + ε) - α × loss(y_i, expert_pred[k,i])
//     ↑ Gate's belief          ↑ Expert's prediction quality

// Mode "loss_only": use only loss (simpler, more intuitive)
s_ik = -α × loss(y_i, expert_pred[k,i])
//     ↑ Simply: lower loss = higher score

// Convert scores to responsibilities via softmax:
r_ik = softmax(s_ik)  // Σₖ r_ik = 1
```

**When to use `loss_only`**: If you want the Expert with the lowest prediction error to always get the highest responsibility, regardless of what the Gate currently predicts. This avoids the "self-reinforcing loop" problem where the Gate's initial beliefs get locked in.

**M-Step for Experts** (line 481-523): Train each expert with responsibility-weighted gradients:

```cpp
// For expert k, gradient at sample i:
grad_k[i] = r_ik × ∂L(y_i, f_k(x_i)) / ∂f_k
//          ↑ Responsibility weight (soft assignment)

// High r_ik → Expert k learns more from sample i
// Low r_ik  → Expert k ignores sample i
```

**Why EM works**: Responsibilities `r_ik` act as soft cluster assignments. Each Expert specializes on samples where it has high responsibility. The Gate learns to route samples to the appropriate Expert.

### 4. Per-Expert Hyperparameters (Implemented)

Each Expert can have different tree **structural** configurations via per-expert hyperparameters:

**Implementation** (`src/boosting/mixture_gbdt.cpp`):

```cpp
// Each expert has its own config
std::vector<std::unique_ptr<Config>> expert_configs_;  // One per expert

// Per-expert structural parameters (comma-separated in config)
std::vector<int> mixture_expert_max_depths;           // e.g., "3,5,7"
std::vector<int> mixture_expert_num_leaves;           // e.g., "8,16,32"
std::vector<int> mixture_expert_min_data_in_leaf;     // e.g., "50,20,5"
std::vector<double> mixture_expert_min_gain_to_split; // e.g., "0.1,0.01,0.001"
```

**How it works**:

| Specification | Behavior |
|---------------|----------|
| Not specified | All experts use base structural hyperparameters |
| Comma-separated list | Each expert uses its corresponding value (must have exactly K values) |

**Example use cases**:

```python
# Use case 1: Coarse vs Fine experts
# Expert 0 captures broad patterns, Expert 1 captures detailed patterns
params = {
    'mixture_num_experts': 2,
    'mixture_expert_min_data_in_leaf': '50,5',    # coarse vs fine
    'mixture_expert_min_gain_to_split': '0.1,0.001',  # conservative vs aggressive
}

# Use case 2: Shallow vs Deep experts
# Different tree depths for different regime complexities
params = {
    'mixture_num_experts': 2,
    'mixture_expert_max_depths': '3,7',
    'mixture_expert_num_leaves': '8,64',
}
```

**Symmetry breaking**: Even with shared hyperparameters, experts differentiate through:
1. Per-expert random seeds (automatic, no config needed)
2. Uniform initialization with different initial predictions due to seeds

---

<a name="japanese"></a>
## Japanese (日本語)

### 概要

LightGBM-MoE は [Microsoft LightGBM](https://github.com/microsoft/LightGBM) のフォークで、**Mixture-of-Experts (MoE) / レジームスイッチング GBDT** をC++でネイティブ実装しています。

```
ŷ(x) = Σₖ gₖ(x) · fₖ(x)
```

### 動作環境

- **Python**: 3.10以上
- **OS**: Linux (x86_64, aarch64), macOS (Intel, Apple Silicon)
- **依存関係**: numpy, scipy（自動インストール）

### インストール

**推奨: GitHubから直接インストール**（ソースからビルド、CMake必須）:

```bash
pip install git+https://github.com/kyo219/LightGBM-MoE.git
```

**別方法: 手動ビルド**:

```bash
git clone https://github.com/kyo219/LightGBM-MoE.git
cd LightGBM-MoE
pip install ./python-package
```

**開発者向け（編集可能インストール）**:

```bash
git clone https://github.com/kyo219/LightGBM-MoE.git
cd LightGBM-MoE/python-package
pip install -e .
```

> **注意**: ソースからビルドするにはCMake（3.16以上）とC++コンパイラ（GCC、Clang、Apple Clang）が必要です。

### クイックスタート

```python
import lightgbm_moe as lgb

params = {
    'boosting': 'mixture',           # MoEモード有効化
    'mixture_num_experts': 2,        # エキスパート数
    'mixture_r_smoothing': 'ema',    # 平滑化手法
    'mixture_smoothing_lambda': 0.5, # 平滑化強度
    'objective': 'regression',
}

train_data = lgb.Dataset(X_train, label=y_train)
model = lgb.train(params, train_data, num_boost_round=100)

# 予測
y_pred = model.predict(X_test)                     # 重み付きミクスチャ
regime = model.predict_regime(X_test)              # レジームインデックス
regime_proba = model.predict_regime_proba(X_test)  # ゲート確率 (N, K)
expert_preds = model.predict_expert_pred(X_test)   # 各エキスパート予測 (N, K)
```

### バリデーション & Early Stopping

```python
import lightgbm_moe as lgb

params = {
    'boosting': 'mixture',
    'mixture_num_experts': 2,
    'objective': 'regression',
    'metric': 'rmse',
    'verbose': 1,
}

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

model = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[valid_data],
    valid_names=['valid'],
    callbacks=[lgb.early_stopping(stopping_rounds=10)]
)

print(f"Best iteration: {model.best_iteration}")
```

---

## API リファレンス

### MoE パラメータ

| パラメータ | 型 | デフォルト | 範囲 | 説明 |
|-----------|------|---------|-------|-------------|
| `boosting` | string | `"gbdt"` | `"gbdt"`, `"mixture"` | MoEモードを有効にするには `"mixture"` を指定 |
| `mixture_num_experts` | int | 4 | 2-10 | エキスパート数 (K)。各エキスパートは異なるデータレジームに特化する独立したGBDT。 |
| `mixture_e_step_alpha` | float | 1.0 | 0.1-5.0 | E-stepでの損失項の重み。高いほど予測精度を重視、低いほどゲート確率を重視。 |
| `mixture_e_step_mode` | string | `"em"` | `"em"`, `"loss_only"` | E-stepモード。`"em"`: ゲート確率+損失（標準EM）。`"loss_only"`: 損失のみ（シンプル、最も適合するExpertに割り当て）。 |
| `mixture_warmup_iters` | int | 10 | 0-50 | ウォームアップ回数。この期間中、責務は均等 (1/K) で、専門化前にエキスパートが学習できる。 |
| `mixture_balance_factor` | int | 10 | 2-20 | 負荷分散の強度。最小エキスパート使用率 = 1/(factor × K)。小さいほど積極的なバランシング。推奨: 5-7。 |
| `mixture_r_smoothing` | string | `"none"` | `"none"`, `"ema"`, `"markov"`, `"momentum"` | 時系列安定化のための責務平滑化手法。**推奨: `"none"`**（下記注意参照）。 |
| `mixture_smoothing_lambda` | float | 0.0 | 0.0-1.0 | 平滑化強度。`mixture_r_smoothing` が `"none"` 以外の場合のみ使用。高いほど平滑化が強い（レジーム遷移が遅い）。 |
| `mixture_gate_entropy_lambda` | float | 0.0 | 0.0-1.0 | Gateエントロピー正則化。Gateがより不確実な予測を出すよう促し、早期のExpert collapseを防止。**推奨: 0.01-0.1**。 |
| `mixture_expert_dropout_rate` | float | 0.0 | 0.0-1.0 | Expertドロップアウト率。学習中にランダムにExpertを無効化し、全Expertが有用であることを強制。**推奨: 0.1-0.3**。 |

> **重要: `mixture_r_smoothing="none"`（デフォルト）を推奨**
>
> 平滑化手法（`ema`, `markov`, `momentum`）は**Expert collapse**（全Expertが同じ予測に収束）を引き起こす可能性があります。Optuna最適化のベンチマークでは、`smoothing=none`は安定して良好なExpert分離（相関~0.02、regime精度~98%）を達成しますが、他の平滑化手法ではcollapse（相関~0.99、regime精度~50%）が頻発しました。

### Expert Collapse防止（上級者向け）

Expertがcollapseしている場合（全て同じような予測を出す）、以下の新パラメータを試してください：

| パラメータ | 使用場面 | 効果 |
|-----------|---------|------|
| `mixture_gate_entropy_lambda` | Gateが早期に全サンプルを1つのExpertに割り当てる | Gateの確信度を下げ、Expertが分化する時間を確保 |
| `mixture_expert_dropout_rate` | 1つのExpertが支配的で他が学習を停止 | 学習中にランダムに無効化し、全Expertを有用に保つ |

**例: Collapse防止**

```python
params = {
    'boosting': 'mixture',
    'mixture_num_experts': 2,
    'objective': 'regression',

    # Collapse防止
    'mixture_gate_entropy_lambda': 0.05,  # Gate予測を不確実にする
    'mixture_expert_dropout_rate': 0.2,   # 各イテレーションで20%の確率でExpertを無効化

    # 他の推奨設定
    'mixture_warmup_iters': 20,           # Expertの分化を先に許可
    'mixture_balance_factor': 5,          # より積極的な負荷分散
}
```

**動作原理:**

1. **Gateエントロピー正則化** (`mixture_gate_entropy_lambda`):
   - Gateが確信を持ちすぎる場合にペナルティを追加: `grad += λ * (p - 1/K)`
   - Gate確率をuniform分布（1/K）に近づける
   - Expertが本当に専門化すると効果が減少

2. **Expertドロップアウト** (`mixture_expert_dropout_rate`):
   - 各イテレーションでランダムにExpertを無効化（勾配がゼロに）
   - 無効化されたExpertは更新されず、他のExpertがカバーを強制
   - 少なくとも1つのExpertは常に有効
   - ニューラルネットワークのdropoutに類似

### Early Stopping

MoEはバリデーションベースのearly stoppingをサポート。Optunaでのハイパーパラメータ最適化に便利です。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|------|---------|-------------|
| `early_stopping_round` | int | 0 | N回連続でバリデーション指標が改善しない場合に学習を停止。`lgb.early_stopping()` コールバックで設定。 |
| `first_metric_only` | bool | False | 複数のmetricを指定した場合、最初のmetricのみでearly stoppingを判定。 |

**コールバックでの使用:**

```python
model = lgb.train(
    params,
    train_data,
    valid_sets=[valid_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=10),  # 10ラウンド改善なしで停止
        lgb.log_evaluation(period=10),           # 10イテレーションごとにログ
    ]
)
```

**Optunaでの使用:**

```python
def objective(trial):
    params = {
        'boosting': 'mixture',
        'mixture_num_experts': trial.suggest_int('num_experts', 2, 4),
        'objective': 'regression',
        'metric': 'rmse',
        'verbose': -1,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=20)]
    )

    return model.best_score['valid_0']['rmse']
```

### Expertごとのハイパーパラメータ（上級者向け）

各Expertに異なるツリー**構造**設定を持たせることができます。指定しない場合、全Expertは同じハイパーパラメータを共有します。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|------|---------|-------------|
| `mixture_expert_max_depths` | string | `""` | カンマ区切りの各Expertのmax_depth。指定する場合はK個の値が必要。 |
| `mixture_expert_num_leaves` | string | `""` | カンマ区切りの各Expertのnum_leaves。指定する場合はK個の値が必要。 |
| `mixture_expert_min_data_in_leaf` | string | `""` | カンマ区切りの各Expertのmin_data_in_leaf。木の粒度を制御。 |
| `mixture_expert_min_gain_to_split` | string | `""` | カンマ区切りの各Expertのmin_gain_to_split。分割の積極性を制御。 |
| `mixture_expert_extra_trees` | string | `""` | カンマ区切りの各Expertの0/1。Expertごとにextremely randomized treesを有効化。 |

#### 全Expertで同じハイパーパラメータ（デフォルト）

Expertごとのパラメータを指定しない場合、全Expertはベースのハイパーパラメータを共有します：

```python
params = {
    'boosting': 'mixture',
    'mixture_num_experts': 3,
    'max_depth': 5,           # 3つのExpert全てがmax_depth=5を使用
    'num_leaves': 31,         # 3つのExpert全てがnum_leaves=31を使用
    'min_data_in_leaf': 20,   # 3つのExpert全てがmin_data_in_leaf=20を使用
}
```

#### Expertごとに異なるハイパーパラメータ

カンマ区切りの値（Expertごとに1つ）を指定して各Expertをカスタマイズ：

```python
params = {
    'boosting': 'mixture',
    'mixture_num_experts': 3,
    # Expert 0: 粗い (高min_data), Expert 1: 中程度, Expert 2: 細かい (低min_data)
    'mixture_expert_max_depths': '3,5,7',
    'mixture_expert_num_leaves': '8,16,32',
    'mixture_expert_min_data_in_leaf': '50,20,5',      # 粗い → 細かい
    'mixture_expert_min_gain_to_split': '0.1,0.01,0.001',  # 保守的 → 積極的
}
```

これにより、異なる構造容量を持たせることができます：
- **粗いExpert** (高min_data_in_leaf): 大まかなパターンを捕捉、過学習しにくい
- **細かいExpert** (低min_data_in_leaf): 詳細なパターンを捕捉、複雑なレジームに適合

#### Expertごとのハイパラと学習の動作

**EMイテレーションの構造**: 各ブースティングイテレーションで、各Expertに1本ずつ木を追加：

```
TrainOneIter() {
  1. Forward()      → 全Expertの予測を計算
  2. EStep()        → 責務を更新
  3. MStepExperts() → 各Expertが1本ずつ木を追加（逐次実行）
  4. MStepGate()    → Gateが1本木を追加
}
```

**ハイパラで変わるもの / 変わらないもの:**

| 項目 | 深いExpert | 浅いExpert |
|------|------------|------------|
| EMイテレーション数 | 同じ | 同じ |
| 木の本数 | 同じ | 同じ |
| 1本あたりの構築時間 | 長い | 短い |
| 1本あたりの表現力 | 高い | 低い |

**重要なポイント**: `num_boost_round=100` は各Expertが100本の木を構築することを意味し、深さの設定に関係ありません。Expertごとのハイパラは**木1本あたりの表現力**を制御し、木の本数は変わりません。

```
num_boost_round = 100:
  Expert 0 (浅い): 100本の浅い木 → シンプルなパターン
  Expert 1 (深い): 100本の深い木 → 複雑なパターン
                ↓
        同じ100 EMイテレーション
```

**学習時間**: 現在の実装では各イテレーションでExpertは逐次的に学習されるため、最も深い/複雑なExpertがボトルネックになります。合計時間 ≈ 各イテレーションの全Expert木構築時間の合計。

#### 初期化と対称性の破壊

デフォルトでは **uniform初期化** を使用：全サンプルが全Expertに対して均等な責務 `1/K` で開始。

```
初期状態 (K=2):
  サンプル 0: r = [0.5, 0.5]  (両Expertに均等)
  サンプル 1: r = [0.5, 0.5]
  ...
```

**対称性の破壊** はExpertごとの乱数シードで実現：
```cpp
expert_configs_[k]->seed = config_->seed + k + 1;  // Expertごとに異なるseed
```

これにより、ラベルに基づく初期化（ターゲット情報のリークの可能性）に頼らず、学習が進むにつれてExpertが自然に分化します。

利用可能な初期化モード（`mixture_init` パラメータ）：
| モード | 説明 |
|--------|------|
| `uniform`（デフォルト） | 均等な `1/K` 責務、対称性はExpertごとのseedで破壊 |
| `random` | 各サンプルをランダムに1つのExpertに割り当て |
| `quantile` | ラベルのquantileで割り当て（y依存、注意して使用） |

#### Optuna最適化の例

**標準MoE**（全Expertで共通のハイパーパラメータ）：

```python
import optuna
import lightgbm_moe as lgb
from sklearn.metrics import mean_squared_error

def objective(trial):
    params = {
        'boosting': 'mixture',
        'objective': 'regression',
        'verbose': -1,
        # 木構造（全Expertで共通）
        'num_leaves': trial.suggest_int('num_leaves', 8, 128),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        # MoE固有
        'mixture_num_experts': trial.suggest_int('mixture_num_experts', 2, 4),
        'mixture_e_step_alpha': trial.suggest_float('mixture_e_step_alpha', 0.1, 2.0),
        'mixture_warmup_iters': trial.suggest_int('mixture_warmup_iters', 5, 30),
        'mixture_balance_factor': trial.suggest_int('mixture_balance_factor', 2, 10),
        # 平滑化（オプション、時系列向け）
        'mixture_r_smoothing': trial.suggest_categorical(
            'mixture_r_smoothing', ['none', 'ema', 'markov']
        ),
    }
    if params['mixture_r_smoothing'] != 'none':
        params['mixture_smoothing_lambda'] = trial.suggest_float(
            'mixture_smoothing_lambda', 0.1, 0.9
        )

    # 学習と評価
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=100)
    pred = model.predict(X_val)
    return mean_squared_error(y_val, pred)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

**Per-Expert MoE**（Expertごとに異なる木構造）：

```python
def objective_per_expert(trial):
    num_experts = trial.suggest_int('mixture_num_experts', 2, 4)

    # Expertごとの木構造
    max_depths = [trial.suggest_int(f'max_depth_{k}', 3, 12) for k in range(num_experts)]
    num_leaves = [trial.suggest_int(f'num_leaves_{k}', 8, 128) for k in range(num_experts)]
    min_data = [trial.suggest_int(f'min_data_{k}', 5, 100) for k in range(num_experts)]

    params = {
        'boosting': 'mixture',
        'objective': 'regression',
        'verbose': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        # MoE固有
        'mixture_num_experts': num_experts,
        'mixture_e_step_alpha': trial.suggest_float('mixture_e_step_alpha', 0.1, 2.0),
        'mixture_warmup_iters': trial.suggest_int('mixture_warmup_iters', 5, 30),
        'mixture_balance_factor': trial.suggest_int('mixture_balance_factor', 2, 10),
        # Expertごとの構造（カンマ区切り）
        'mixture_expert_max_depths': ','.join(map(str, max_depths)),
        'mixture_expert_num_leaves': ','.join(map(str, num_leaves)),
        'mixture_expert_min_data_in_leaf': ','.join(map(str, min_data)),
    }

    # 学習と評価
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=100)
    pred = model.predict(X_val)
    return mean_squared_error(y_val, pred)

# 注意: Per-expertはK×3個のハイパラが追加されるため、より多くのtrial（200+）が必要
study = optuna.create_study(direction='minimize')
study.optimize(objective_per_expert, n_trials=200)
```

**Tip**: まず標準MoEを試す。Per-expertは探索空間が大きいため収束に多くのtrialが必要。

#### 役割ベースPer-Expert（Optuna推奨）

素朴なPer-Expertアプローチには問題があります：**各Expertのパラメータが独立**のため、似たようなExpert（例：全て同じようなdepthとleaves）が生まれやすくなります。これではMoEの専門化の利点が活かせません。

**解決策**: 各Expertに異なる「役割」（性格）を割り当て、それに応じて探索空間を制限します。

```
             num_leaves
              少     多
max_depth 浅  E0     E1    ← E1: 浅いが広い
          深  E2     E3    ← E2: 深いが狭い
```

これにより多様性が保証されます：E1（浅い×多い葉）はE2（深い×少ない葉）とは異なるパターンを捕捉します。

```python
def suggest_moe_expert_params(
    trial,
    num_experts: int,
    depth_range: tuple = (2, 15),
    leaves_range: tuple = (4, 128),
    min_data_range: tuple = (5, 100),
    use_extra_trees: bool = True,
):
    """
    各Expertに異なる「役割」を割り当てつつ、具体的な値はOptunaで探索。

    - 探索空間を削減: K×3パラメータ → 4パラメータ (depth_low/high, leaves_low/high)
    - 多様性を保証: 各Expertは異なる (depth, leaves) の組み合わせを持つ
    - 全範囲探索: 全Expertが「深い」または「浅い」になりうるが、各trial内では
      相対的な差を保証 (low < high)
    - Extra trees: 浅いExpertにはextra_trees（多様性）、深いExpertには通常の木（精度）
    """

    # 全範囲から探索、ただしtrial内で low < high を保証
    # これにより: Trial A (3 vs 12), Trial B (10 vs 14), Trial C (2 vs 4) が可能
    depth_low = trial.suggest_int('depth_low', depth_range[0], depth_range[1] - 1)
    depth_high = trial.suggest_int('depth_high', depth_low + 1, depth_range[1])

    leaves_low = trial.suggest_int('leaves_low', leaves_range[0], leaves_range[1] - 1)
    leaves_high = trial.suggest_int('leaves_high', leaves_low + 1, leaves_range[1])

    # Kに応じた役割パターン
    # (depth_level, leaves_level): 0=low, 1=high, 0.5=mid
    PATTERNS = {
        2: [(0, 0), (1, 1)],                                  # 対角: simple vs complex
        3: [(0, 0), (0, 1), (1, 1)],                          # simple, 浅×広, complex
        4: [(0, 0), (0, 1), (1, 0), (1, 1)],                  # 全4象限
        5: [(0, 0), (0, 1), (1, 0), (1, 1), (0.5, 0.5)],      # 4象限 + 中央
        6: [(0, 0), (0, 1), (1, 0), (1, 1), (0, 0.5), (1, 0.5)],
    }

    if num_experts in PATTERNS:
        patterns = PATTERNS[num_experts]
    else:
        # K > 6: 4象限を繰り返し + 補間
        base = [(0, 0), (0, 1), (1, 0), (1, 1)]
        patterns = (base * ((num_experts // 4) + 1))[:num_experts]

    def interp(low, high, t):
        return round(low + t * (high - low))

    depths, leaves_list, min_datas, extra_trees = [], [], [], []
    for d_level, l_level in patterns:
        depths.append(interp(depth_low, depth_high, d_level))
        leaves_list.append(interp(leaves_low, leaves_high, l_level))
        # min_dataはdepthと逆相関（深い → 小さいmin_data）
        min_datas.append(interp(min_data_range[1], min_data_range[0], d_level))
        # 浅いExpertにはextra_trees（ランダム性）、深いExpertには精度重視
        extra_trees.append(1 if d_level < 0.5 else 0)

    result = {
        'mixture_expert_max_depths': ','.join(map(str, depths)),
        'mixture_expert_num_leaves': ','.join(map(str, leaves_list)),
        'mixture_expert_min_data_in_leaf': ','.join(map(str, min_datas)),
    }
    if use_extra_trees:
        result['mixture_expert_extra_trees'] = ','.join(map(str, extra_trees))
    return result


def objective_role_based(trial):
    num_experts = trial.suggest_int('num_experts', 2, 4)

    # 役割ベースのExpertパラメータを取得（K×3ではなく4パラメータのみ）
    expert_params = suggest_moe_expert_params(
        trial,
        num_experts=num_experts,
        depth_range=(2, 15),
        leaves_range=(4, 128),
    )

    params = {
        'boosting': 'mixture',
        'objective': 'regression',
        'verbose': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'mixture_num_experts': num_experts,
        'mixture_e_step_alpha': trial.suggest_float('mixture_e_step_alpha', 0.1, 2.0),
        'mixture_warmup_iters': trial.suggest_int('mixture_warmup_iters', 5, 30),
        **expert_params,  # 役割ベースのExpert構造
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=100)
    pred = model.predict(X_val)
    return mean_squared_error(y_val, pred)

study = optuna.create_study(direction='minimize')
study.optimize(objective_role_based, n_trials=100)
```

**出力例 (K=4):**
```
探索結果: depth_low=3, depth_high=10, leaves_low=8, leaves_high=64

E0: depth=3,  leaves=8,  min_data=100, extra_trees=1  (浅い × 少ない) → 高速、ランダム
E1: depth=3,  leaves=64, min_data=100, extra_trees=1  (浅い × 多い)   → 広い、ランダム
E2: depth=10, leaves=8,  min_data=5,   extra_trees=0  (深い × 少ない) → 狭い、精密
E3: depth=10, leaves=64, min_data=5,   extra_trees=0  (深い × 多い)   → 複雑、精密
```

**メリット:**
- 探索空間: K×3パラメータ → 4パラメータ（大幅削減）
- 多様性保証: 各Expertが異なる構造的「性格」を持つ
- 解釈性: 各Expertが何を捕捉するよう設計されているかが明確

#### モデル品質フィルタリング（崩壊モデルの除外）

MoEモデルは2つの失敗モードがあります：**Expert collapse**（全Expertが同じ予測）と**Gate confusion**（Gateがどのexpertを使うか決められない）。Optuna最適化時にこれらをフィルタリングできます：

```python
import numpy as np

def compute_model_quality(model, X_val):
    """MoEモデルの品質指標を計算（正解ラベル不要）"""
    gate_proba = model.predict_regime_proba(X_val)      # (N, K)
    expert_preds = model.predict_expert_pred(X_val)     # (N, K)
    K = gate_proba.shape[1]

    # 1. Expert相関（collapse検出）
    correlations = []
    for i in range(K):
        for j in range(i + 1, K):
            corr = np.corrcoef(expert_preds[:, i], expert_preds[:, j])[0, 1]
            correlations.append(corr)
    max_corr = max(correlations) if correlations else 0.0

    # 2. Gateエントロピー（ルーティング確信度）
    eps = 1e-10
    entropy = -np.sum(gate_proba * np.log(gate_proba + eps), axis=1)
    normalized_entropy = entropy / np.log(K)
    mean_entropy = normalized_entropy.mean()

    return {'expert_corr_max': max_corr, 'gate_entropy': mean_entropy}

def objective_with_quality_filter(trial):
    params = {
        'boosting': 'mixture',
        'mixture_num_experts': trial.suggest_int('num_experts', 2, 4),
        # ... その他のパラメータ ...
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=100)

    # 品質チェック（正解ラベル不要 - ライブデータでも使用可能）
    quality = compute_model_quality(model, X_val)

    # collapseまたはconfusionモデルを除外
    if quality['expert_corr_max'] > 0.8:  # Expertが似すぎ
        raise optuna.TrialPruned("Expert collapse detected")
    if quality['gate_entropy'] > 0.6:     # Gateが決められない
        raise optuna.TrialPruned("Gate confusion detected")

    pred = model.predict(X_val)
    return mean_squared_error(y_val, pred)
```

**品質閾値:**

| 指標 | 閾値 | 解釈 |
|------|------|------|
| `expert_corr_max` | < 0.8（厳しめ: < 0.7） | Expertは異なる予測をすべき |
| `gate_entropy` | < 0.6（厳しめ: < 0.5） | Gateは確信を持ってルーティングすべき |

**正規化エントロピーの解釈（K非依存）:**

| エントロピー | Gate確率 (K=2) | 意味 |
|-------------|----------------|------|
| 0.3 | [0.88, 0.12] | 高確信 |
| 0.5 | [0.77, 0.23] | 中程度の確信 |
| 0.6 | [0.70, 0.30] | 許容範囲 |
| 0.8 | [0.57, 0.43] | 低確信 |

### 平滑化手法

| 手法 | 数式 | 用途 |
|------|------|------|
| `none` | `r_t = r_t` (変更なし) | i.i.d.データ、レジームがXから決定可能 |
| `ema` | `r_t = λ·r_{t-1} + (1-λ)·r_t` | 持続的レジームを持つ時系列 |
| `markov` | `r_t ∝ r_t · (A·r_{t-1})` | レジーム遷移がマルコフ連鎖に従う |
| `momentum` | `r_t = λ·r_{t-1} + (1-λ)·r_t + β·Δr` | トレンドのあるレジーム変化 |

### 予測API

| メソッド | 出力形状 | 説明 |
|---------|---------|------|
| `predict(X)` | `(N,)` | 最終予測: エキスパート予測の重み付きミクスチャ |
| `predict_regime(X)` | `(N,)` int | 最も可能性の高いレジーム: `argmax_k(gate_proba)` |
| `predict_regime_proba(X)` | `(N, K)` | 各エキスパートへのゲート確率（合計1） |
| `predict_expert_pred(X)` | `(N, K)` | 各エキスパートの個別予測 |

### MoEコンポーネントのSHAP分析

LightGBM-MoEは、SHAP分析のために個別のコンポーネントモデル（GateとExpert）を抽出するAPIを提供します。これにより、各コンポーネントの特徴量重要度を個別に理解できます。

#### コンポーネントBoosterの抽出

```python
import lightgbm_moe as lgb

# MoEモデルを学習
model = lgb.train(params, train_data, num_boost_round=100)

# 個別コンポーネントをスタンドアローンBoosterとして抽出
gate_booster = model.get_gate_booster()           # Gateモデル
expert_0_booster = model.get_expert_booster(0)    # Expert 0
expert_1_booster = model.get_expert_booster(1)    # Expert 1

# または全コンポーネントを一度に取得
boosters = model.get_all_boosters()
# 戻り値: {'gate': Booster, 'expert_0': Booster, 'expert_1': Booster, ...}
```

#### SHAP分析の例

```python
import shap
import lightgbm as standard_lgb  # SHAPには標準LightGBMが必要
import tempfile

# lightgbm_moe BoosterをSHAP互換形式に変換するヘルパー関数
def to_shap_model(booster):
    """lightgbm_moe Boosterを標準lightgbm Boosterに変換（SHAP用）"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(booster.model_to_string(num_iteration=-1))
        temp_path = f.name
    return standard_lgb.Booster(model_file=temp_path)

# MoEモデルを学習
model = lgb.train(params, train_data, num_boost_round=100)

# 各コンポーネントのSHAP値を取得
boosters = model.get_all_boosters()

for name, booster in boosters.items():
    shap_model = to_shap_model(booster)
    explainer = shap.TreeExplainer(shap_model)
    shap_values = explainer.shap_values(X)

    # 多出力モデルの処理（GateはK個の出力を持つ）
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]  # 最初のクラスを使用

    # beeswarmプロット作成
    shap.summary_plot(shap_values, X, plot_type="dot", show=False)
    plt.title(f"SHAP: {name}")
    plt.savefig(f"shap_{name}.png")
    plt.close()
```

#### コンポーネントBooster API

| メソッド | 戻り値 | 説明 |
|---------|--------|------|
| `get_gate_booster()` | `Booster` | スタンドアローンGateモデル（K-クラス分類器） |
| `get_expert_booster(k)` | `Booster` | スタンドアローンExpert kモデル（回帰） |
| `get_all_boosters()` | `dict[str, Booster]` | 全コンポーネント: `{'gate': ..., 'expert_0': ..., ...}` |

#### 注意事項

- **Gateモデル**: 多クラス分類器（K個の出力）。SHAPは3D配列 `(N, features, K)` を返す。最初のクラスまたは平均を使用。
- **Expertモデル**: 回帰モデル。SHAPは2D配列 `(N, features)` を返す。
- **標準LightGBMが必要**: SHAPの `TreeExplainer` は標準LightGBM Boosterを期待。上記のヘルパー関数でlightgbm_moe Boosterを一時ファイル経由で標準形式に変換。

#### ベンチマークでのSHAP可視化

ベンチマークスクリプトは最適化されたMoEモデルのSHAP beeswarmプロットを自動生成します：

```bash
python examples/benchmark.py --trials 100

# SHAP可視化をスキップ
python examples/benchmark.py --trials 100 --no-shap
```

出力ファイル:
- `examples/shap_gate.png` - Gate特徴量重要度
- `examples/shap_expert_0.png` - Expert 0特徴量重要度
- `examples/shap_expert_1.png` - Expert 1特徴量重要度
- `examples/moe_shap_beeswarm.png` - 統合比較

#### SHAP可視化結果（Syntheticデータ）

**Gate Beeswarmプロット:**

![SHAP Gate](examples/shap_gate.png)

Gateモデルは**X1, X2, X3**に基づいてサンプルをルーティングすることを学習しています。これはSyntheticデータのレジーム定義（`regime_score = 0.5*X1 + 0.3*X2 - 0.2*X3`）と一致しています。

**コンポーネント間の特徴量重要度比較:**

![MoE SHAP Beeswarm](examples/moe_shap_beeswarm.png)

統合プロットから以下が確認できます:
- **Gate**: レジーム識別特徴量（X1, X2, X3）に焦点
- **Expert 0**: 異なる特徴量サブセットに特化
- **Expert 1**: 相補的なパターンに焦点

この結果は、MoEモデルがデータ内の異なるレジームを効果的に分離していることを示しています。

---

## ベンチマーク

**設定**: 200 Optunaトライアル、5分割時系列CV、early stopping (50 rounds)。

### RMSE比較

| データセット | Standard | MoE | MoE-PE | 改善率 |
|-------------|----------|-----|--------|--------|
| Synthetic | 5.0739 | **4.3049** | 4.7611 | +15.2% |
| Hamilton | 0.7205 | 0.7220 | **0.7204** | +0.0% |
| VIX | 0.0116 | **0.0116** | 0.0116 | +0.1% |

### Expert分化（レジーム分離）

| データセット | MoE K | MoE-PE K | MoE相関 (min/max) | MoE-PE相関 (min/max) | MoE Regime精度 | MoE-PE Regime精度 |
|-------------|-------|----------|-------------------|----------------------|----------------|-------------------|
| Synthetic | 2 | 3 | -0.42/-0.42 | -0.29/0.82 | 96.2% | 95.1% |
| Hamilton | 4 | 3 | 0.69/0.88 | 0.61/0.67 | 52.6% | 52.6% |
| VIX | 2 | 2 | 0.64/0.64 | 0.99/0.99 | 53.0% | 52.1% |

- **K**: Optunaで選択されたExpert数
- **Expert相関 (min/max)**: Expert間の予測相関（低い=分化している、負=逆の予測）
- **Regime精度**: 予測regimeと真のregimeの分類精度

**重要な発見**:
- MoEがSyntheticデータで **+15.2%の改善** 、Expert相関 **-0.42**（逆相関！）
- MoEとMoE-PE両方でSyntheticデータにおいて **96%以上のregime精度** を達成
- 潜在レジームデータ（Hamilton, VIX）では改善が限定的（レジームがXから決定できないため、想定通り）

### 選択されたハイパーパラメータ（Syntheticデータセット）

**MoE（共有木構造）:**
- num_experts: 2, max_depth: 10, num_leaves: 20, learning_rate: 0.064, alpha: 1.44, warmup: 5

**MoE-PerExpert（Expertごとの木構造）:**
- num_experts: 3, learning_rate: 0.068, alpha: 0.52, warmup: 5

| Expert | max_depth | num_leaves | min_data_in_leaf |
|--------|-----------|------------|------------------|
| E0 | 3 | 21 | 79 |
| E1 | 4 | 79 | 66 |
| E2 | 6 | 12 | 65 |

### ベンチマーク実行

```bash
# フルベンチマーク (Standard vs MoE vs MoE-PE、デフォルト100 trials)
python examples/benchmark.py

# より多くのtrials
python examples/benchmark.py --trials 200

# 軽めのテスト
python examples/benchmark.py --trials 10

# Markdown出力
python examples/benchmark.py --trials 200 --output-md BENCHMARK.md
```

### 可視化

![ベンチマーク結果](examples/benchmark_results.png)

- **左**: レジーム分離（真のレジームごとのExpertへのルーティング%）
- **中央**: Expert予測散布図（色=真のレジーム）
- **右**: 手法間のRMSE比較

---

## MoEを使うべき場面

**MoEが有効な条件:**
- レジームが特徴量(X)から決定可能
- 異なるレジームが根本的に異なる関数に従う
- 各レジームに十分なデータがある

**MoEが有効でない条件:**
- レジームが潜在的（隠れマルコフ、観測されない状態）
- 標準GBDTが既にパターンを捕捉できる
- データに明確なレジーム構造がない

---

## 技術的詳細

### 1. MoEの実現方法：アーキテクチャ

MoEモデルは **K個のExpert GBDT** と **1個のGate GBDT** で構成されます：

```
                    ┌─────────────┐
                    │   入力 X    │
                    └──────┬──────┘
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌────────────┐  ┌────────────┐  ┌────────────┐
    │  Expert 0  │  │  Expert 1  │  │    Gate    │
    │   (GBDT)   │  │   (GBDT)   │  │   (GBDT)   │
    │   回帰     │  │   回帰     │  │ 多クラス分類│
    └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
          │               │               │
          │  f₀(x)        │  f₁(x)        │ logits
          ▼               ▼               ▼
    ┌─────────────────────────────────────────────┐
    │              重み付き結合                    │
    │  ŷ = Σₖ softmax(gate_logits)ₖ · fₖ(x)      │
    └─────────────────────────────────────────────┘
```

**主要な実装詳細** (`src/boosting/mixture_gbdt.cpp`):

| コンポーネント | 実装 | 説明 |
|---------------|------|------|
| Expert群 | `std::vector<std::unique_ptr<GBDT>> experts_` | 混合モデルと同じ目的関数 |
| Gate | `std::unique_ptr<GBDT> gate_` | K-クラス分類 |
| 責務 | `std::vector<double> responsibilities_` | N × K のソフト割り当て |
| 負荷分散 | `std::vector<double> expert_bias_` | Expert崩壊を防止 |

### 2. Gateの学習メカニズム

GateはLightGBMの多クラス分類として **K-クラス分類GBDT** で学習されます：

```cpp
// Gate設定 (mixture_gbdt.cpp:86-93)
gate_config_->objective = "multiclass";
gate_config_->num_class = num_experts_;
gate_config_->max_depth = config_->mixture_gate_max_depth;      // デフォルト: 3
gate_config_->num_leaves = config_->mixture_gate_num_leaves;    // デフォルト: 8
gate_config_->learning_rate = config_->mixture_gate_learning_rate;  // デフォルト: 0.1
```

**学習プロセス** (M-Step for Gate, `MStepGate()` 526行目):

1. **疑似ラベル作成**: `z_i = argmax_k(r_ik)` (責務からのハード割り当て)
2. **勾配計算**: Softmax交差エントロピー勾配
   ```cpp
   // 各サンプルi、クラスkに対して:
   if (k == label) {
       grad[i,k] = p_k - 1.0;   // 正解クラスの勾配
   } else {
       grad[i,k] = p_k;         // 他クラスの勾配
   }
   hess[i,k] = p_k * (1 - p_k); // ヘシアン: softmax微分
   ```
3. **Gate更新**: `gate_->TrainOneIter(gate_grad, gate_hess)`

Gateは特徴量Xに基づいて **どのExpertが各サンプルを処理すべきか** を予測するよう学習します。

### 3. EMスタイルの学習アルゴリズム

各イテレーションは **期待値最大化法 (EM)** スタイルの更新に従います：

```
┌─────────────────────────────────────────────────────────────┐
│                    TrainOneIter()                           │
├─────────────────────────────────────────────────────────────┤
│ 1. Forward()     → expert_pred[N,K], gate_proba[N,K]を計算  │
│ 2. EStep()       → 責務 r[N,K] を更新                       │
│ 3. Smoothing()   → EMA/Markov/Momentum適用（オプション）    │
│ 4. MStepExperts()→ 重み付き勾配でExpertを学習               │
│ 5. MStepGate()   → 疑似ラベルでGateを学習                   │
└─────────────────────────────────────────────────────────────┘
```

**E-Step** (350-383行目): 各Expertが各サンプルにどれだけ適合するかに基づいて責務を更新：

```cpp
// スコア計算は mixture_e_step_mode に依存:

// モード "em" (デフォルト): ゲート確率 + 損失
s_ik = log(gate_proba[i,k] + ε) - α × loss(y_i, expert_pred[k,i])
//     ↑ Gateの信念               ↑ Expertの予測品質

// モード "loss_only": 損失のみ（シンプル、より直感的）
s_ik = -α × loss(y_i, expert_pred[k,i])
//     ↑ 単純に: 損失が低い = スコアが高い

// スコアをsoftmaxで責務に変換:
r_ik = softmax(s_ik)  // Σₖ r_ik = 1
```

**`loss_only`を使う場面**: Gateの現在の予測に関係なく、常に予測誤差が最も小さいExpertに高い責務を与えたい場合。Gateの初期の信念が固定される「自己強化ループ」問題を回避できます。

**M-Step for Experts** (493-535行目): 責務重み付き勾配で各Expertを学習：

```cpp
// Expert kのサンプルiでの勾配:
grad_k[i] = r_ik × ∂L(y_i, f_k(x_i)) / ∂f_k
//          ↑ 責務の重み（ソフト割り当て）

// r_ikが高い → Expert kはサンプルiからより学習
// r_ikが低い → Expert kはサンプルiを無視
```

**EMが機能する理由**: 責務 `r_ik` はソフトクラスタ割り当てとして機能します。各Expertは高い責務を持つサンプルに特化します。Gateは適切なExpertにサンプルをルーティングするよう学習します。

### 4. Expertごとのハイパーパラメータ（実装済み）

各Expertはper-expertハイパーパラメータにより異なるツリー**構造**設定を持てます：

**実装** (`src/boosting/mixture_gbdt.cpp`):

```cpp
// 各Expertは独自の設定を持つ
std::vector<std::unique_ptr<Config>> expert_configs_;  // Expertごとに1つ

// Expertごとの構造パラメータ（設定でカンマ区切り）
std::vector<int> mixture_expert_max_depths;           // 例: "3,5,7"
std::vector<int> mixture_expert_num_leaves;           // 例: "8,16,32"
std::vector<int> mixture_expert_min_data_in_leaf;     // 例: "50,20,5"
std::vector<double> mixture_expert_min_gain_to_split; // 例: "0.1,0.01,0.001"
```

**動作**:

| 指定方法 | 動作 |
|---------|------|
| 未指定 | 全Expertがベースの構造ハイパーパラメータを使用 |
| カンマ区切りリスト | 各Expertが対応する値を使用（K個の値が必要） |

**使用例**:

```python
# 使用例1: 粗いExpert vs 細かいExpert
# Expert 0は大まかなパターン、Expert 1は詳細なパターンを担当
params = {
    'mixture_num_experts': 2,
    'mixture_expert_min_data_in_leaf': '50,5',    # 粗い vs 細かい
    'mixture_expert_min_gain_to_split': '0.1,0.001',  # 保守的 vs 積極的
}

# 使用例2: 浅いExpert vs 深いExpert
# 異なるレジーム複雑度に対応
params = {
    'mixture_num_experts': 2,
    'mixture_expert_max_depths': '3,7',
    'mixture_expert_num_leaves': '8,64',
}
```

**対称性の破壊**: 共有ハイパーパラメータでも、以下の方法でExpertは差別化されます：
1. Expertごとの乱数シード（自動、設定不要）
2. シードによる異なる初期予測を持つ均一初期化

---

## License

This project is licensed under the MIT license. Based on [Microsoft LightGBM](https://github.com/microsoft/LightGBM).
