# Per-Expert Hyperparameters

By default, all experts share the same tree structural hyperparameters. The `mixture_expert_*` parameters let each expert have its own depth, leaf count, etc. — useful when different regimes have different complexity.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mixture_expert_max_depths` | string | `""` | Comma-separated `max_depth` per expert. Must have exactly K values if specified. |
| `mixture_expert_num_leaves` | string | `""` | Comma-separated `num_leaves` per expert. |
| `mixture_expert_min_data_in_leaf` | string | `""` | Comma-separated `min_data_in_leaf` per expert. |
| `mixture_expert_min_gain_to_split` | string | `""` | Comma-separated `min_gain_to_split` per expert. |
| `mixture_expert_extra_trees` | string | `""` | Comma-separated `0`/`1` per expert. Enables extremely randomized trees per expert. |

## Default (all experts share base params)

```python
params = {
    'boosting': 'mixture',
    'mixture_num_experts': 3,
    'max_depth': 5,           # All 3 experts use max_depth=5
    'num_leaves': 31,
    'min_data_in_leaf': 20,
}
```

## Per-Expert Customization

```python
params = {
    'boosting': 'mixture',
    'mixture_num_experts': 3,
    # Expert 0: coarse, Expert 1: medium, Expert 2: fine
    'mixture_expert_max_depths': '3,5,7',
    'mixture_expert_num_leaves': '8,16,32',
    'mixture_expert_min_data_in_leaf': '50,20,5',
    'mixture_expert_min_gain_to_split': '0.1,0.01,0.001',
}
```

- **Coarse expert** (high `min_data_in_leaf`): broad patterns, less overfitting
- **Fine expert** (low `min_data_in_leaf`): detailed patterns for complex regimes

## Training Behavior

Each boosting iteration adds **one tree per expert**, regardless of depth settings:

```
TrainOneIter() {
  1. Forward()      → Compute predictions from all experts
  2. EStep()        → Update responsibilities
  3. MStepExperts() → Each expert adds 1 tree (sequential)
  4. MStepGate()    → Gate adds 1 tree
}
```

`num_boost_round=100` means each expert builds 100 trees. Per-expert hyperparameters control **expressiveness per tree**, not the number of trees. Sequential training means the deepest/most complex expert is the bottleneck.

## Role-Based Recipe (for Optuna)

The naive per-expert search has K×3 independent parameters, so Optuna often produces similar experts (wasting MoE's specialization potential). The role-based approach assigns each expert a distinct (depth, leaves) "role" and only searches the corner values:

```
             num_leaves
              Low    High
max_depth Low  E0     E1    ← E1: shallow but wide
          High E2     E3    ← E2: deep but narrow
```

```python
def suggest_moe_expert_params(trial, num_experts, depth_range=(2, 15), leaves_range=(4, 128), min_data_range=(5, 100)):
    """Assign distinct roles per expert. Only 4 search params instead of K×3."""
    depth_low = trial.suggest_int('depth_low', depth_range[0], depth_range[1] - 1)
    depth_high = trial.suggest_int('depth_high', depth_low + 1, depth_range[1])
    leaves_low = trial.suggest_int('leaves_low', leaves_range[0], leaves_range[1] - 1)
    leaves_high = trial.suggest_int('leaves_high', leaves_low + 1, leaves_range[1])

    PATTERNS = {
        2: [(0, 0), (1, 1)],
        3: [(0, 0), (0, 1), (1, 1)],
        4: [(0, 0), (0, 1), (1, 0), (1, 1)],
        5: [(0, 0), (0, 1), (1, 0), (1, 1), (0.5, 0.5)],
        6: [(0, 0), (0, 1), (1, 0), (1, 1), (0, 0.5), (1, 0.5)],
    }
    patterns = PATTERNS.get(num_experts) or ([(0, 0), (0, 1), (1, 0), (1, 1)] * ((num_experts // 4) + 1))[:num_experts]

    interp = lambda lo, hi, t: round(lo + t * (hi - lo))
    depths = [interp(depth_low, depth_high, d) for d, _ in patterns]
    leaves_list = [interp(leaves_low, leaves_high, l) for _, l in patterns]
    min_datas = [interp(min_data_range[1], min_data_range[0], d) for d, _ in patterns]
    extra_trees = [1 if d < 0.5 else 0 for d, _ in patterns]

    return {
        'mixture_expert_max_depths': ','.join(map(str, depths)),
        'mixture_expert_num_leaves': ','.join(map(str, leaves_list)),
        'mixture_expert_min_data_in_leaf': ','.join(map(str, min_datas)),
        'mixture_expert_extra_trees': ','.join(map(str, extra_trees)),
    }
```

**Example output (K=4)**:

```
depth_low=3, depth_high=10, leaves_low=8, leaves_high=64

E0: depth=3,  leaves=8,  min_data=100, extra_trees=1   (shallow × few, randomized)
E1: depth=3,  leaves=64, min_data=100, extra_trees=1   (shallow × wide, randomized)
E2: depth=10, leaves=8,  min_data=5,   extra_trees=0   (deep × narrow, precise)
E3: depth=10, leaves=64, min_data=5,   extra_trees=0   (deep × wide, precise)
```

## Initialization & Symmetry Breaking

Even with shared hyperparameters, experts differentiate via per-expert random seeds:

```cpp
expert_configs_[k]->seed = config_->seed + k + 1;  // Different seed per expert
```

No label-based initialization is needed (and `quantile` init can leak target information — use with caution).
