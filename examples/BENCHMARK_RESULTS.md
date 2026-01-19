## Benchmark Results

- **Optuna Trials**: 200
- **CV Splits**: 5
- **Boosting Rounds**: 100
- **Seed**: 42

### RMSE Comparison

| Dataset | Standard | MoE | MoE-PE | Best Improvement |
|---------|----------|-----|--------|------------------|
| Synthetic | 5.0739 |  **4.3049** | 4.7611 | +15.2% |
| Hamilton | 0.7205 | 0.7220 |  **0.7204** | +0.0% |
| VIX | 0.0116 | 0.0116 |  **0.0116** | +0.1% |

### Expert Differentiation (Regime Separation)

| Dataset | MoE Corr (min/max) | MoE-PE Corr (min/max) | MoE Regime Acc | MoE-PE Regime Acc |
|---------|--------------------|-----------------------|----------------|-------------------|
| Synthetic | -0.42/-0.42 | -0.29/0.82 | 96.2% | 95.1% |
| Hamilton | 0.69/0.88 | 0.61/0.67 | 52.6% | 52.6% |
| VIX | 0.64/0.64 | 0.99/0.99 | 53.0% | 52.1% |

**Notes**:
- **Expert Corr (min/max)**: Min and max pairwise correlation between expert predictions
  - min: Most differentiated pair (lower = better separation)
  - max: Most similar pair (if ~1.0, some experts may have collapsed)
- **Regime Acc**: Classification accuracy of predicted regime vs true regime (with best label permutation)

### Selected Hyperparameters

#### Synthetic

**Standard GBDT:**
- max_depth: 8
- num_leaves: 48
- min_data_in_leaf: 5
- learning_rate: 0.1232

**MoE (Shared Tree Structure):**
- num_experts: 2
- max_depth: 10
- num_leaves: 20
- min_data_in_leaf: 5
- learning_rate: 0.0642
- smoothing: none

**MoE-PerExpert (Per-Expert Tree Structure):**
- num_experts: 3

| Expert | max_depth | num_leaves | min_data_in_leaf |
|--------|-----------|------------|------------------|
| E0 | 3 | 21 | 79 |
| E1 | 4 | 79 | 66 |
| E2 | 6 | 12 | 65 |

- learning_rate: 0.0675
- smoothing: none

#### Hamilton

**Standard GBDT:**
- max_depth: 3
- num_leaves: 75
- min_data_in_leaf: 17
- learning_rate: 0.1273

**MoE (Shared Tree Structure):**
- num_experts: 4
- max_depth: 8
- num_leaves: 91
- min_data_in_leaf: 5
- learning_rate: 0.2723
- smoothing: none

**MoE-PerExpert (Per-Expert Tree Structure):**
- num_experts: 3

| Expert | max_depth | num_leaves | min_data_in_leaf |
|--------|-----------|------------|------------------|
| E0 | 6 | 86 | 12 |
| E1 | 4 | 38 | 38 |
| E2 | 4 | 46 | 87 |

- learning_rate: 0.2801
- smoothing: none

#### VIX

**Standard GBDT:**
- max_depth: 3
- num_leaves: 116
- min_data_in_leaf: 10
- learning_rate: 0.1266

**MoE (Shared Tree Structure):**
- num_experts: 2
- max_depth: 3
- num_leaves: 33
- min_data_in_leaf: 12
- learning_rate: 0.1273
- smoothing: none

**MoE-PerExpert (Per-Expert Tree Structure):**
- num_experts: 2

| Expert | max_depth | num_leaves | min_data_in_leaf |
|--------|-----------|------------|------------------|
| E0 | 3 | 64 | 16 |
| E1 | 3 | 37 | 41 |

- learning_rate: 0.1661
- smoothing: none
