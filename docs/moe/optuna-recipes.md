# Optuna Search Templates

The canonical Optuna setup for tuning LightGBM-MoE, derived from the [1000-trial study](benchmark.md). One template, no contradictions.

## Quick template (universal, works on any dataset)

```python
import optuna
import lightgbm_moe as lgb
from sklearn.metrics import mean_squared_error

def objective(trial):
    params = {
        'boosting': 'mixture',
        'objective': 'regression',
        'verbose': -1,

        # Universal winners from the 1000-trial study (see benchmark.md)
        'mixture_gate_type': 'gbdt',
        'mixture_routing_mode': 'token_choice',
        'extra_trees': True,

        # Tree structure
        'num_leaves': trial.suggest_int('num_leaves', 8, 128),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 0, 7),

        # MoE core (3-4 experts wins universally)
        'mixture_num_experts': trial.suggest_int('mixture_num_experts', 2, 4),
        'mixture_e_step_alpha': trial.suggest_float('mixture_e_step_alpha', 0.1, 3.0),
        'mixture_warmup_iters': trial.suggest_int('mixture_warmup_iters', 5, 50),
        'mixture_balance_factor': trial.suggest_int('mixture_balance_factor', 2, 10),

        # Diversity matters (top-3 fANOVA in all datasets, no single best value)
        'mixture_diversity_lambda': trial.suggest_float('mixture_diversity_lambda', 0.0, 0.5),

        # Dataset-dependent (let TPE find the right one)
        'mixture_e_step_mode': trial.suggest_categorical('mixture_e_step_mode', ['em', 'loss_only', 'gate_only']),
        'mixture_init': trial.suggest_categorical('mixture_init', ['gmm', 'random', 'tree_hierarchical']),
        'mixture_r_smoothing': trial.suggest_categorical('mixture_r_smoothing', ['none', 'ema', 'markov']),
        'mixture_hard_m_step': trial.suggest_categorical('mixture_hard_m_step', [True, False]),

        # Gate (search wider than expert depth)
        'mixture_gate_max_depth': trial.suggest_int('mixture_gate_max_depth', 2, 10),
        'mixture_gate_num_leaves': trial.suggest_int('mixture_gate_num_leaves', 4, 64),
        'mixture_gate_learning_rate': trial.suggest_float('mixture_gate_learning_rate', 0.01, 0.5, log=True),
        'mixture_gate_lambda_l2': trial.suggest_float('mixture_gate_lambda_l2', 1e-3, 10.0, log=True),
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    model = lgb.train(
        params, train_data, num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    pred = model.predict(X_valid)
    return mean_squared_error(y_valid, pred, squared=False)

study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=42),
)
study.optimize(objective, n_trials=300, n_jobs=6)
```

## Trial budget & parallelism

| Setting | Value | Why |
|---------|-------|-----|
| `n_trials` | 300-1000 | TPE needs ~50+ to localize good regions; 300 gives stable RMSE |
| `n_jobs` (Optuna) | `cores / num_threads` | Each parallel trial holds an OMP team — overall ≤ physical cores |
| `num_threads` (LightGBM) | 4-8 | Sweet spot for ≤10k samples; 24 threads per trial wastes time on overhead |
| `num_boost_round` | 100-500 | Pair with `early_stopping(50)` so weak trials exit early |
| TPE seed | fixed | Reproducible study comparison |

> **Why not `n_jobs=1, num_threads=24`?** On small data (≤10k samples), one trial cannot saturate 24 cores — most threads sit idle. Running 6 trials × 4 threads in parallel keeps all cores warm and finishes 3-4× faster.

## Adding Expert Choice routing to the search

If you want TPE to choose between `token_choice` and `expert_choice`:

```python
routing_mode = trial.suggest_categorical('mixture_routing_mode', ['token_choice', 'expert_choice'])
params['mixture_routing_mode'] = routing_mode

if routing_mode == 'expert_choice':
    params['mixture_expert_capacity_factor'] = trial.suggest_float('mixture_expert_capacity_factor', 0.8, 1.5)
    params['mixture_expert_choice_score'] = 'gate'  # Fixed: only 'gate' prevents collapse
    params['mixture_expert_choice_boost'] = trial.suggest_float('mixture_expert_choice_boost', 5.0, 30.0)
    params['mixture_expert_choice_hard'] = trial.suggest_categorical('mixture_expert_choice_hard', [True, False])
```

Note: in the headline 1000-trial study, `token_choice` produced the absolute best on every dataset. `expert_choice` is mainly useful for stability under load imbalance — see [advanced-routing.md](advanced-routing.md).

## Adding EvoMoE & temperature annealing

```python
params['mixture_progressive_mode'] = trial.suggest_categorical('mixture_progressive_mode', ['none', 'evomoe'])
if params['mixture_progressive_mode'] == 'evomoe':
    params['mixture_seed_iterations'] = trial.suggest_int('mixture_seed_iterations', 20, 100)
    params['mixture_spawn_perturbation'] = trial.suggest_float('mixture_spawn_perturbation', 0.1, 0.9)

params['mixture_gate_temperature_init'] = trial.suggest_float('mixture_gate_temperature_init', 1.0, 5.0)
params['mixture_gate_temperature_final'] = trial.suggest_float('mixture_gate_temperature_final', 0.1, 1.0)
```

## Adding per-expert hyperparameters (role-based)

For tree-structure differentiation across experts, use the role-based recipe so TPE only searches 4 parameters instead of K×3 independent ones — see [per-expert-hp.md](per-expert-hp.md#role-based-recipe-for-optuna).

## Pruning collapsed trials

Two ways to drop trials whose experts collapsed (saves significant Optuna time):

1. **`expert_collapse_stopper` callback** — fires during training. See [advanced-collapse.md](advanced-collapse.md#2-collapse-stopper-optuna-callback).
2. **Post-hoc quality filter** — runs after training. See [advanced-collapse.md](advanced-collapse.md#3-post-hoc-quality-filter).

## Time-series / latent regime tips

- Add time-series features (rolling means, volatility, MA crossover) to make latent regimes observable from X
- Search `mixture_r_smoothing` including `ema` and `markov` for temporal regime persistence
- Use `mixture_e_step_mode='loss_only'` or `'gate_only'` when the gate cannot determine regime from features alone — both won on Hamilton & VIX in the study
