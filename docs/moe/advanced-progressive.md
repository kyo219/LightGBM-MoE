# Progressive Training (EvoMoE) & Gate Temperature Annealing

Inspired by [EvoMoE (Nie et al., 2022)](https://arxiv.org/abs/2112.14397) and [Drop-Upcycling (ICLR 2025)](https://openreview.net/forum?id=nKPaFSGXmV). Instead of initializing K expert GBDTs from scratch, progressive training:

1. **Seed Phase**: Trains a single seed GBDT on all data (no gating).
2. **Spawn**: Duplicates the seed into K experts with random perturbation (Drop-Upcycling style).
3. **MoE Phase**: Runs standard EM training with the pre-trained experts.

This eliminates initialization sensitivity and enables natural expert branching from a shared foundation.

## EvoMoE Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `mixture_progressive_mode` | string | `"none"` | `"none"`, `"evomoe"` | Progressive training mode |
| `mixture_seed_iterations` | int | 50 | 0-500 | Seed GBDT training iterations before spawn |
| `mixture_spawn_perturbation` | float | 0.5 | 0.0-1.0 | Perturbation ratio. 0.0 = exact copy, 1.0 = all trees perturbed. 0.5 is the Drop-Upcycling optimum. |

## Gate Temperature Annealing

Combinable with progressive training or used independently. Lets the gate explore (high T) early and exploit (low T) late.

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `mixture_gate_temperature_init` | float | 1.0 | >0.0 | Initial softmax temperature. High (2.0-3.0) = near-uniform routing for exploration. |
| `mixture_gate_temperature_final` | float | 1.0 | >0.0 | Final temperature. Low (0.3-1.0) = sharp routing for exploitation. |

Decays exponentially: `T(t) = T_init * (T_final/T_init)^(t/T_total)`. When `init == final == 1.0` (default), no annealing occurs.

## Combined Example

```python
params = {
    'boosting': 'mixture',
    'mixture_num_experts': 3,
    'objective': 'regression',
    'num_iterations': 300,

    # Progressive training (EvoMoE)
    'mixture_progressive_mode': 'evomoe',
    'mixture_seed_iterations': 50,
    'mixture_spawn_perturbation': 0.5,

    # Gate temperature annealing
    'mixture_gate_temperature_init': 2.0,
    'mixture_gate_temperature_final': 0.5,
}
```

## Optuna Search Ranges

```python
trial.suggest_categorical('mixture_progressive_mode', ['none', 'evomoe'])
trial.suggest_int('mixture_seed_iterations', 20, 100)
trial.suggest_float('mixture_spawn_perturbation', 0.1, 0.9)
trial.suggest_float('mixture_gate_temperature_init', 1.0, 5.0)
trial.suggest_float('mixture_gate_temperature_final', 0.1, 1.0)
```
