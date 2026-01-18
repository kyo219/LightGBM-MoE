## Benchmark Results

- **Optuna Trials**: 200
- **CV Splits**: 5
- **Boosting Rounds**: 100
- **Seed**: 42

### RMSE Comparison

| Dataset | Standard | MoE | MoE-PE | Best Improvement |
|---------|----------|-----|--------|------------------|
| Synthetic | 5.1032 |  **4.3928** | 5.1410 | +13.9% |
| Hamilton | 0.7192 |  **0.7127** | 0.7173 | +0.9% |
| VIX | 0.0117 |  **0.0116** | 0.0116 | +0.5% |

### Expert Differentiation (Regime Separation)

| Dataset | MoE Corr | MoE-PE Corr | MoE Regime Acc | MoE-PE Regime Acc |
|---------|----------|-------------|----------------|-------------------|
| Synthetic | -0.28 | 0.98 | 96.2% | 58.6% |
| Hamilton | 0.91 | 0.95 | 50.8% | 50.2% |
| VIX | 0.94 | 0.99 | 52.0% | 52.4% |

**Notes**:
- **Expert Corr**: Correlation between expert predictions (lower = more differentiated, negative = opposite predictions)
- **Regime Acc**: Classification accuracy of predicted regime vs true regime (with best label permutation)

### Selected Hyperparameters

#### Synthetic

**Standard GBDT:**
- max_depth: 12
- num_leaves: 58
- min_data_in_leaf: 5
- learning_rate: 0.0563

**MoE (Shared Tree Structure):**
- num_experts: 2
- max_depth: 6
- num_leaves: 10
- min_data_in_leaf: 5
- learning_rate: 0.0786
- smoothing: none

**MoE-PerExpert (Per-Expert Tree Structure):**
- num_experts: 3

| Expert | max_depth | num_leaves | min_data_in_leaf |
|--------|-----------|------------|------------------|
| E0 | 7 | 100 | 20 |
| E1 | 9 | 103 | 5 |
| E2 | 5 | 96 | 94 |

- learning_rate: 0.2190
- smoothing: momentum

#### Hamilton

**Standard GBDT:**
- max_depth: 3
- num_leaves: 73
- min_data_in_leaf: 14
- learning_rate: 0.2372

**MoE (Shared Tree Structure):**
- num_experts: 2
- max_depth: 4
- num_leaves: 115
- min_data_in_leaf: 14
- learning_rate: 0.2238
- smoothing: markov

**MoE-PerExpert (Per-Expert Tree Structure):**
- num_experts: 3

| Expert | max_depth | num_leaves | min_data_in_leaf |
|--------|-----------|------------|------------------|
| E0 | 5 | 57 | 50 |
| E1 | 3 | 93 | 13 |
| E2 | 10 | 8 | 90 |

- learning_rate: 0.2556
- smoothing: ema

#### VIX

**Standard GBDT:**
- max_depth: 8
- num_leaves: 32
- min_data_in_leaf: 21
- learning_rate: 0.1994

**MoE (Shared Tree Structure):**
- num_experts: 4
- max_depth: 3
- num_leaves: 120
- min_data_in_leaf: 14
- learning_rate: 0.1082
- smoothing: markov

**MoE-PerExpert (Per-Expert Tree Structure):**
- num_experts: 3

| Expert | max_depth | num_leaves | min_data_in_leaf |
|--------|-----------|------------|------------------|
| E0 | 8 | 37 | 77 |
| E1 | 5 | 97 | 28 |
| E2 | 4 | 104 | 32 |

- learning_rate: 0.1530
- smoothing: ema
