# Feature Catalog & int8 Compatibility

A consolidated reference for *what knobs exist*, *which knob requires which*, and *which knobs are safe to use with `use_quantized_grad`*. See [`examples/compat_matrix.py`](../../examples/compat_matrix.py) for the regression test that produced the matrix below.

## "int8" — three independent layers

People say "int8" to mean three different things in this project. They live at different layers and are **set independently**: passing an int8 input array does **not** auto-enable `use_quantized_grad`, and vice versa.

```
Layer 1: Python input dtype                        — controlled by:  X.astype(np.int8)
   ↓ (C API boundary)
Layer 2: C++ bin storage dtype (uint8 / uint16)    — controlled by:  max_bin (auto)
   ↓ (training loop)
Layer 3: gradient/hessian quantization             — controlled by:  use_quantized_grad=True
```

| Layer | What it does | How to enable | Effect |
|---|---|---|---|
| **1. Python input int8** | numpy.int8 array reaches the C API without a Python-side float32 conversion | `X.astype(np.int8)` | 4× Python-process memory reduction (e.g. 954 MB vs 3.73 GB for 500K × 2000) |
| **2. Bin storage** | After binning, bin indices are held as `uint8_t` when `num_bin ≤ 256`, `uint16_t` up to 65535 | Always automatic (just keep `max_bin ≤ 255`) | C++-side memory minimized; this layer has *always* been int8 for moderate `max_bin` |
| **3. Quantized gradients** | Training-time `grad`/`hess` are scaled and packed to int8 (16/32-bit accumulators); histogram construction reads the smaller types | `params['use_quantized_grad'] = True` (+ `num_grad_quant_bins`, `stochastic_rounding`) | 1.05–1.30× speedup with negligible RMSE change (Standard); same on MoE after the Phase 2 fix |

### Common misconceptions

- **"I passed `np.int8` so the gradients are int8 too"** — no. Layer 1 only saves Python memory; the C++ side still computes float gradients unless Layer 3 is on.
- **"Layer 2 needs me to set anything"** — no. The bin-storage type is chosen automatically from `max_bin`. For `[0, 4]`-style Numerai features, `num_bin = 5 → uint8` storage with no flags.
- **"Layers depend on each other"** — they don't. Any combination is valid. Full speedup template:

  ```python
  X_int8 = X.astype(np.int8)             # Layer 1: Python memory ↓ 4×
  ds = lgb.Dataset(X_int8, label=y)      # Layer 2: uint8 bins automatic
  params = {
      "use_quantized_grad": True,         # Layer 3: int8 grad/hess
      "num_grad_quant_bins": 32,
      "max_bin": 255,                     # ensure layer 2 stays in uint8
      ...
  }
  ```

### Why aren't they coupled?

Input dtype is a property of the *data* (Numerai features are `[0, 4]` integers; image features are `[0, 255]`). Quantized gradients depend on the *loss landscape* (gradient magnitudes are objective-dependent). Auto-coupling would silently quantize gradients for users who only wanted the Python memory win, with surprising RMSE consequences on some objectives. The two layers stay separate explicit knobs.

## 8 Configuration Axes

| Axis | Parameter(s) | Choices | Default |
|------|--------------|---------|---------|
| **1. Model variant** | `boosting`, `mixture_progressive_mode`, per-expert vector params | `gbdt` / `mixture` / `mixture` + EvoMoE / `mixture` + per-expert HP | `gbdt` |
| **2. E-step** | `mixture_e_step_mode`, `mixture_e_step_alpha`, `mixture_e_step_loss` | `em` / `loss_only` / `gate_only` | `em` |
| **3. M-step** | `mixture_hard_m_step`, `mixture_diversity_lambda` | hard (sparse) / soft (weighted) | `true` (hard) |
| **4. Gate** | `mixture_gate_type` (+ all `mixture_gate_*` knobs) | `gbdt` / `none` / `leaf_reuse` | `gbdt` |
| **5. Routing** | `mixture_routing_mode`, `mixture_expert_*` capacity/score | `token_choice` / `expert_choice` | `token_choice` |
| **6. Smoothing** | `mixture_r_smoothing`, `mixture_smoothing_lambda` | `none` / `ema` / `markov` / `momentum` | `none` |
| **7. Initialization** | `mixture_init` | `uniform` / `random` / `quantile` / `balanced_kmeans` / `gmm` / `tree_hierarchical` | `uniform` |
| **8. Regularization** | warmup, load balance, dropout, gate entropy, gate temperature, adaptive LR | many | mostly off |

## Dependency Map

```
boosting=gbdt ─────────────────────────── (baseline; mixture_* params are no-ops)
   │
   └─ use_quantized_grad ✓  (1.05-1.30× speedup at no RMSE cost)

boosting=mixture
   │
   ├─ mixture_progressive_mode=evomoe
   │     ├ mixture_seed_iterations
   │     └ mixture_spawn_perturbation
   │
   ├─ mixture_routing_mode=expert_choice
   │     ├ mixture_expert_capacity_factor
   │     ├ mixture_expert_choice_score
   │     ├ mixture_expert_choice_boost
   │     └ mixture_expert_choice_hard
   │
   ├─ mixture_gate_type
   │     ├ "gbdt"        → all mixture_gate_* params apply
   │     ├ "none"        → E-step is forced into loss_only mode
   │     └ "leaf_reuse"  → mixture_gate_retrain_interval applies
   │
   ├─ mixture_hard_m_step=true
   │     └─ Sparse activation auto-enabled; per-expert bagging auto-disabled (#16).
   │
   ├─ mixture_r_smoothing != "none"
   │     └─ mixture_smoothing_lambda applies (assumes row order = time)
   │
   ├─ mixture_dropout_schedule != "constant"
   │     ├ mixture_dropout_rate_min
   │     └ mixture_dropout_rate_max
   │
   └─ mixture_adaptive_lr=true
         ├ mixture_adaptive_lr_window
         └ mixture_adaptive_lr_max
```

### Auto-applied settings (no user input needed)

| When | Forced setting | Where | Why |
|------|----------------|-------|-----|
| `mixture_hard_m_step=true` | per-expert `bagging_fraction=1.0`, `bagging_freq=0` | `mixture_gbdt.cpp:113-116` | Sparse activation already restricts experts to assigned samples; double-bagging produces degenerate histograms (#16) |
| `use_quantized_grad=true` (under MoE) | `quant_train_renew_leaf=true` on every expert + gate | `mixture_gbdt.cpp:125-127, 137-139` | Without renewal the quantized leaf-output path is biased by sparse-activation `hess≈1e-12` rows, causing 3-20× RMSE blow-up |
| `mixture_gate_type="none"` | E-step runs in `loss_only` regardless of `mixture_e_step_mode` | `mixture_gbdt.cpp` | No gate probabilities to weight by |

## int8 / `use_quantized_grad` Compatibility Matrix

Empirical run via `examples/compat_matrix.py` (5,000 × 100 int8, K=3, 30 rounds), all 8 axes × {float, quant} = 31 features × 2 modes = **62 trials**:

| Status | Count | Meaning |
|--------|-------|---------|
| `CRASH` | **0 / 31** | The combination produces an exception or non-finite RMSE |
| `REGRESS` | **0 / 31** | Quant RMSE is more than 30 % worse than float RMSE |
| `minor` | 6 / 31 | 5-30 % RMSE diff, all on stochastic-init or smoothing paths whose float/quant trajectories diverge by random-seed effects |
| `ok` | 25 / 31 | RMSE within 5 % |

### Per-axis compatibility (representative rows)

| Feature | float RMSE | quant RMSE | Status |
|---------|-----------|------------|--------|
| `gbdt/standard` | 1.246 | 1.246 | ok (4.4× faster) |
| `moe/default` | 1.292 | 1.315 | ok |
| `moe/hard=True` (sparse activation) | 1.292 | 1.315 | ok ← regressed pre-fix |
| `moe/hard=False` (soft) | 1.256 | 1.251 | ok |
| `moe/gate=gbdt` | 1.292 | 1.315 | ok |
| `moe/gate=none` | 2.574 | 2.925 | minor |
| `moe/gate=leaf_reuse` | 2.850 | 3.027 | minor |
| `moe/route=token_choice` | 1.292 | 1.315 | ok |
| `moe/route=expert_choice` | 1.256 | 1.293 | ok |
| `moe/evomoe` (progressive) | 1.299 | 1.294 | ok |
| `moe-pe/per_expert_hp` | 1.396 | 1.407 | ok |
| `moe/expert_dropout` | 1.517 | 1.484 | ok |
| `moe/adaptive_lr` | 1.594 | 1.584 | ok |
| `moe/dropout_curriculum` | 1.357 | 1.365 | ok |
| `moe/gate_temperature` (annealing) | 1.646 | 1.419 | ok |
| `moe/diversity_lambda=0.3` | 483.7 | 205.9 | ok ※ |
| `moe/init={uniform/quantile/gmm}` | 1.27-1.29 | 1.32-1.33 | ok |
| `moe/init={random/balanced_kmeans/tree_hierarchical}` | 1.24-1.28 | 1.32-1.49 | minor (random-seed sensitive) |
| `moe/smooth={ema/markov}` | 1.32-2.75 | 1.32-2.83 | ok |
| `moe/smooth=momentum` | 2.69 | 2.86 | minor |

※ `diversity_lambda=0.3` blows up RMSE on this tiny dataset for both modes equally — a config sensitivity, not a quantization issue.

**Bottom line: every feature in the codebase is safe to combine with `use_quantized_grad=true`** (after the Phase 2 fix on commit `c596fb93`).

## Recommended Numerai-style Configuration

```python
params = {
    'boosting': 'mixture',
    'use_quantized_grad': True,        # 1.32-1.35× faster on MoE, no RMSE penalty
    'num_grad_quant_bins': 32,         # 16 also fine; 32 is the safer default
    'mixture_num_experts': 3,
    'mixture_warmup_iters': 5,
    'mixture_hard_m_step': True,       # sparse activation
    'mixture_gate_type': 'gbdt',
    'mixture_routing_mode': 'token_choice',  # or 'expert_choice' for strict load balance
    'objective': 'regression',
}
```

For self-hosted training builds, additionally pass `-DUSE_NATIVE_ARCH=ON` to CMake. (Effect on the histogram hot path is currently ~0 % because that loop is memory-bound — see [issue #18](https://github.com/kyo219/LightGBM-MoE/issues/18) for the planned manual AVX-512 VNNI follow-up; the flag is in place to enable that work.)

## Known Limitations

1. **Input binning still rebins int8 → bins** even when the input is already discrete in `[0, max_bin)`. Tracking issue: [#17](https://github.com/kyo219/LightGBM-MoE/issues/17).
2. **Histogram construction is memory-bound** scatter-accumulate — `-march=native` alone moves training time by ~0 %. The manual AVX-512 VNNI fix is tracked in [#18](https://github.com/kyo219/LightGBM-MoE/issues/18).
3. **Distributed mode** (`tree_learner=data` / `voting`) is untested with the Phase 2 quantization fix. Single-node only for now.
