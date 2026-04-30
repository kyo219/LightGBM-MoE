# Expert Choice Routing

An alternative routing strategy where **each expert selects its top samples** instead of each sample selecting experts (token choice). This guarantees perfect load balance.

## Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `mixture_routing_mode` | string | `"token_choice"` | `"token_choice"`, `"expert_choice"` | Routing strategy |
| `mixture_expert_capacity_factor` | float | 1.0 | 0.0-3.0 | Capacity multiplier. Each expert selects `(N/K) × factor` samples. 1.0 = balanced, >1.0 allows overlap. |
| `mixture_expert_choice_score` | string | `"combined"` | `"gate"`, `"loss"`, `"combined"` | Score function. `"gate"`: gate probability. `"loss"`: negative loss. `"combined"`: gate + α × (-loss). |
| `mixture_expert_choice_boost` | float | 10.0 | 1.0-100.0 | Multiplier for responsibility of selected samples. Higher = sharper. |
| `mixture_expert_choice_hard` | bool | `false` | `true`, `false` | Hard routing: non-selected samples get zero weight. Forces stronger specialization but reduces gradient signal. |

## Example

```python
params = {
    'boosting': 'mixture',
    'mixture_num_experts': 4,
    'objective': 'regression',
    'mixture_routing_mode': 'expert_choice',
    'mixture_expert_capacity_factor': 1.0,
    'mixture_expert_choice_score': 'combined',
    'mixture_expert_choice_boost': 10.0,
}
```

## When to use

| Scenario | Recommended |
|----------|-------------|
| Experts collapsing to similar predictions | Expert Choice |
| Load imbalance (one expert gets all samples) | Expert Choice |
| Need strict load balancing | Expert Choice |
| Standard MoE training | Token Choice (default) |

## How it works

1. **Compute Affinity**: For each sample-expert pair, compute affinity score using gate probability and/or loss.
2. **Expert Selection**: Each expert selects its top-C samples (`C = N/K × capacity_factor`).
3. **Soft Assignment**: Selected samples get high responsibility, non-selected get minimum responsibility.
4. **GBDT Compatible**: All samples contribute gradients (soft selection), maintaining tree-building requirements.

## Note on the headline study

In the [500-trial / 5-dataset study](benchmark.md), `expert_choice` produced the absolute best (min) RMSE on **3 of 5 datasets** — `fred_gdp`, `vix`, and `hmm`. `token_choice` only won on `synthetic` and `sp500`. The pattern is roughly: when the regime is latent (Hamilton/VIX/HMM) the per-expert capacity guarantee of `expert_choice` helps; when the regime is fully feature-driven (synthetic) `token_choice`'s sample-perspective routing wins. Search both with TPE rather than picking one a priori.
