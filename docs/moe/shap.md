# SHAP Analysis for MoE Components

LightGBM-MoE provides APIs to extract individual component models (Gate and Experts) as standalone Boosters for SHAP analysis.

## Extracting Component Boosters

```python
import lightgbm_moe as lgb

model = lgb.train(params, train_data, num_boost_round=100)

# Individual components
gate_booster = model.get_gate_booster()           # Multiclass gate
expert_0_booster = model.get_expert_booster(0)    # Expert 0
expert_1_booster = model.get_expert_booster(1)    # Expert 1

# All at once
boosters = model.get_all_boosters()
# Returns: {'gate': Booster, 'expert_0': Booster, 'expert_1': Booster, ...}
```

## SHAP Example

```python
import shap
import lightgbm as standard_lgb  # Standard LightGBM required for SHAP
import tempfile
import matplotlib.pyplot as plt

def to_shap_model(booster):
    """Convert lightgbm_moe Booster to standard lightgbm Booster for SHAP."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(booster.model_to_string(num_iteration=-1))
        temp_path = f.name
    return standard_lgb.Booster(model_file=temp_path)

model = lgb.train(params, train_data, num_boost_round=100)
boosters = model.get_all_boosters()

for name, booster in boosters.items():
    shap_model = to_shap_model(booster)
    explainer = shap.TreeExplainer(shap_model)
    shap_values = explainer.shap_values(X)

    # Gate has K outputs (multiclass) — handle 3D array
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]  # Use first class

    shap.summary_plot(shap_values, X, plot_type="dot", show=False)
    plt.title(f"SHAP: {name}")
    plt.savefig(f"shap_{name}.png")
    plt.close()
```

## Component Booster APIs

| Method | Returns | Description |
|--------|---------|-------------|
| `get_gate_booster()` | `Booster` | Standalone Gate model (K-class classifier) |
| `get_expert_booster(k)` | `Booster` | Standalone Expert k model (regressor) |
| `get_all_boosters()` | `dict[str, Booster]` | All components: `{'gate': ..., 'expert_0': ..., ...}` |

## Notes

- **Gate model**: Multi-class classifier (K outputs). SHAP returns 3D array `(N, features, K)`. Use one class or average across classes.
- **Expert models**: Regression models. SHAP returns 2D array `(N, features)`.
- **Standard LightGBM required**: SHAP's `TreeExplainer` expects standard LightGBM Booster. The helper above converts via temporary file.

## Visualization Examples

The benchmark script generates SHAP beeswarm plots for the optimized MoE model on synthetic data:

```bash
python examples/benchmark.py --trials 100
# Skip: python examples/benchmark.py --trials 100 --no-shap
```

![Gate SHAP](../../examples/shap_gate.png)

The Gate learns to route based on **X1, X2, X3** — matching the synthetic regime definition `0.5*X1 + 0.3*X2 - 0.2*X3`.

![MoE Component SHAP Comparison](../../examples/moe_shap_beeswarm.png)

- **Gate**: X1, X2 dominate routing
- **Expert 0**: X0 dominates (learns `5*X0 + 3*X0*X2 + 2*sin(2*X3) + 10`)
- **Expert 1**: X0, X3 important (learns `-5*X0 - 2*X1² + 3*cos(2*X4) - 10`)

This confirms the Gate identifies regimes correctly and each Expert specializes in different functional relationships.
