# Technical Deep Dive

How MoE is implemented inside LightGBM-MoE: the architecture, the gate learning mechanism, the EM loop, and per-expert hyperparameters.

## 1. Architecture

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

| Component | Implementation |
|-----------|----------------|
| Expert GBDTs | `std::vector<std::unique_ptr<GBDT>> experts_` (same objective as the mixture) |
| Gate GBDT | `std::unique_ptr<GBDT> gate_` (multiclass, num_class = K) |
| Responsibilities | `std::vector<double> responsibilities_` — N × K soft assignments |
| Load Balancing | `std::vector<double> expert_bias_` — prevents expert collapse |

## 2. Gate Learning Mechanism

The Gate is trained as a **K-class classification GBDT** using LightGBM's multiclass objective:

```cpp
// Gate config setup (mixture_gbdt.cpp:86-93)
gate_config_->objective = "multiclass";
gate_config_->num_class = num_experts_;
gate_config_->max_depth = config_->mixture_gate_max_depth;      // default: 3
gate_config_->num_leaves = config_->mixture_gate_num_leaves;    // default: 8
gate_config_->learning_rate = config_->mixture_gate_learning_rate;  // default: 0.1
```

**Training process** (`MStepGate()` at line 526):

1. **Pseudo-labels**: `z_i = argmax_k(r_ik)` (hard assignment from responsibilities)
2. **Softmax cross-entropy gradients**:
   ```cpp
   // For each sample i and class k:
   if (k == label) {
       grad[i,k] = p_k - 1.0;
   } else {
       grad[i,k] = p_k;
   }
   hess[i,k] = p_k * (1 - p_k);
   ```
3. **Update**: `gate_->TrainOneIter(gate_grad, gate_hess)`

The Gate learns to predict **which Expert should handle each sample**, based on features X.

## 3. EM-Style Training Loop

Each iteration follows an **Expectation-Maximization** style update:

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

### E-Step (line 350-383)

```cpp
// Mode "em" (default): use gate probability + loss
s_ik = log(gate_proba[i,k] + ε) - α × loss(y_i, expert_pred[k,i])

// Mode "loss_only": use only loss (simpler, more intuitive)
s_ik = -α × loss(y_i, expert_pred[k,i])

// Convert scores to responsibilities via softmax:
r_ik = softmax(s_ik)  // Σₖ r_ik = 1
```

**When `loss_only` helps**: when you want the Expert with the lowest prediction error to always get the highest responsibility, regardless of what the Gate currently predicts. Avoids the "self-reinforcing loop" where the Gate's initial beliefs lock in.

### M-Step for Experts (line 481-523)

```cpp
// For expert k, gradient at sample i:
grad_k[i] = r_ik × ∂L(y_i, f_k(x_i)) / ∂f_k

// High r_ik → Expert k learns more from sample i
// Low r_ik  → Expert k ignores sample i
```

**Why EM works**: Responsibilities `r_ik` act as soft cluster assignments. Each Expert specializes on samples where it has high responsibility. The Gate learns to route samples to the appropriate Expert.

## 4. Per-Expert Hyperparameters

Each Expert can have different tree **structural** configurations:

```cpp
std::vector<std::unique_ptr<Config>> expert_configs_;  // One per expert

// Per-expert structural parameters (comma-separated in config)
std::vector<int> mixture_expert_max_depths;           // e.g., "3,5,7"
std::vector<int> mixture_expert_num_leaves;           // e.g., "8,16,32"
std::vector<int> mixture_expert_min_data_in_leaf;     // e.g., "50,20,5"
std::vector<double> mixture_expert_min_gain_to_split; // e.g., "0.1,0.01,0.001"
```

| Specification | Behavior |
|---------------|----------|
| Not specified | All experts use base structural hyperparameters |
| Comma-separated list | Each expert uses its corresponding value (must have exactly K values) |

See [per-expert-hp.md](per-expert-hp.md) for usage and the role-based Optuna recipe.

### Symmetry Breaking

Even with shared hyperparameters, experts differentiate via per-expert random seeds:

```cpp
expert_configs_[k]->seed = config_->seed + k + 1;
```

No label-based initialization is needed (and `quantile` init can leak target information — use with caution).
