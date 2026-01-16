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

### Installation

```bash
# Linux
pip install https://github.com/kyo219/LightGBM-MoE/releases/download/v0.1.0/lightgbm_moe-0.1.0-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl

# macOS (Apple Silicon)
pip install https://github.com/kyo219/LightGBM-MoE/releases/download/v0.1.0/lightgbm_moe-0.1.0-py3-none-macosx_11_0_arm64.whl

# Windows
pip install https://github.com/kyo219/LightGBM-MoE/releases/download/v0.1.0/lightgbm_moe-0.1.0-py3-none-win_amd64.whl
```

Or build from source:
```bash
git clone https://github.com/kyo219/LightGBM-MoE.git
cd LightGBM-MoE && mkdir build && cd build
cmake .. && make -j4
cd ../python-package && pip install -e .
```

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
| `mixture_r_smoothing` | string | `"none"` | `"none"`, `"ema"`, `"markov"`, `"momentum"` | Responsibility smoothing method for time-series stability. |
| `mixture_smoothing_lambda` | float | 0.0 | 0.0-1.0 | Smoothing strength. Only used when `mixture_r_smoothing` is not `"none"`. Higher = more smoothing (slower regime transitions). |

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

---

## Benchmark

**Setup**: 100 Optuna trials, 5-fold time-series CV, full hyperparameter search for both Standard and MoE.

| Dataset | Description | Std RMSE | MoE RMSE | Improvement |
|---------|-------------|----------|----------|-------------|
| **Synthetic (X→Regime)** | Regime determinable from features | 5.22 | **4.35** | **+16.7%** |
| Hamilton GNP | Latent Markov regime | 0.74 | 0.74 | +0.0% |
| VIX Volatility | Latent volatility state | 0.012 | 0.012 | -0.7% |

**Key Finding**: MoE excels when regime is determinable from features (X).

### Visualization

![Regime Switching Prediction](examples/regime_switching_comparison.png)

- **Top**: Actual vs Predicted values with background colored by predicted regime
- **Bottom**: Gate probability over time (▼▲ = true regime markers)
- **Result**: 95.8% regime accuracy on synthetic test set

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

### 4. Future Work: Per-Expert Hyperparameters

**Current limitation**: All Experts share the same hyperparameters (`expert_config_` is single instance).

**Implementation feasibility**: **Possible** with moderate code changes:

```cpp
// Current implementation:
std::unique_ptr<Config> expert_config_;  // Shared by all experts

// Proposed change:
std::vector<std::unique_ptr<Config>> expert_configs_;  // One per expert
```

**Required changes**:
1. Add per-expert parameters to `config.h`:
   ```cpp
   std::vector<int> mixture_expert_max_depths;    // e.g., [3, 5, 7]
   std::vector<int> mixture_expert_num_leaves;    // e.g., [8, 16, 32]
   std::vector<double> mixture_expert_learning_rates;
   ```
2. Modify `Init()` to create separate configs for each expert
3. Update Python bindings to accept per-expert params

**Trade-offs**:

| Approach | Pros | Cons |
|----------|------|------|
| Shared config | Simple tuning, fewer params | All experts same capacity |
| Per-expert config | Experts can have different capacity | K × more hyperparameters |
| Expert groups | Balance of both | Moderate complexity |

**Recommended approach**: Start with expert groups (e.g., "shallow" vs "deep" experts):
```python
params = {
    'mixture_num_experts': 4,
    'mixture_expert_groups': [
        {'experts': [0, 1], 'max_depth': 3, 'num_leaves': 8},   # Shallow
        {'experts': [2, 3], 'max_depth': 6, 'num_leaves': 32},  # Deep
    ]
}
```

---

<a name="japanese"></a>
## Japanese (日本語)

### 概要

LightGBM-MoE は [Microsoft LightGBM](https://github.com/microsoft/LightGBM) のフォークで、**Mixture-of-Experts (MoE) / レジームスイッチング GBDT** をC++でネイティブ実装しています。

```
ŷ(x) = Σₖ gₖ(x) · fₖ(x)
```

### インストール

```bash
# Linux
pip install https://github.com/kyo219/LightGBM-MoE/releases/download/v0.1.0/lightgbm_moe-0.1.0-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl

# macOS (Apple Silicon)
pip install https://github.com/kyo219/LightGBM-MoE/releases/download/v0.1.0/lightgbm_moe-0.1.0-py3-none-macosx_11_0_arm64.whl

# Windows
pip install https://github.com/kyo219/LightGBM-MoE/releases/download/v0.1.0/lightgbm_moe-0.1.0-py3-none-win_amd64.whl
```

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
| `mixture_r_smoothing` | string | `"none"` | `"none"`, `"ema"`, `"markov"`, `"momentum"` | 時系列安定化のための責務平滑化手法。 |
| `mixture_smoothing_lambda` | float | 0.0 | 0.0-1.0 | 平滑化強度。`mixture_r_smoothing` が `"none"` 以外の場合のみ使用。高いほど平滑化が強い（レジーム遷移が遅い）。 |

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

---

## ベンチマーク

**設定**: 100 Optunaトライアル、5分割時系列CV、Standard/MoE両方で完全ハイパーパラメータ探索。

| データセット | 説明 | Std RMSE | MoE RMSE | 改善率 |
|-------------|------|----------|----------|--------|
| **合成 (X→Regime)** | レジームが特徴量から決定可能 | 5.22 | **4.35** | **+16.7%** |
| Hamilton GNP | 潜在マルコフレジーム | 0.74 | 0.74 | +0.0% |
| VIX Volatility | 潜在ボラティリティ状態 | 0.012 | 0.012 | -0.7% |

**重要な発見**: MoEはレジームが特徴量(X)から決定可能な場合に有効。

### 可視化

![レジームスイッチング予測](examples/regime_switching_comparison.png)

- **上段**: 実績値 vs 予測値（背景色=予測レジーム）
- **下段**: 時間経過でのゲート確率（▼▲=真のレジームマーカー）
- **結果**: 合成テストセットで95.8%のレジーム精度

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

### 4. 今後の展望：Expertごとのハイパーパラメータ

**現在の制限**: 全Expertが同じハイパーパラメータを共有（`expert_config_`が単一インスタンス）

**実装可能性**: 中程度のコード変更で **実現可能**：

```cpp
// 現在の実装:
std::unique_ptr<Config> expert_config_;  // 全Expertで共有

// 提案する変更:
std::vector<std::unique_ptr<Config>> expert_configs_;  // Expertごとに1つ
```

**必要な変更**:
1. `config.h`にExpertごとのパラメータを追加:
   ```cpp
   std::vector<int> mixture_expert_max_depths;    // 例: [3, 5, 7]
   std::vector<int> mixture_expert_num_leaves;    // 例: [8, 16, 32]
   std::vector<double> mixture_expert_learning_rates;
   ```
2. `Init()`を修正して各Expertに個別設定を作成
3. PythonバインディングをExpertごとのパラメータを受け入れるよう更新

**トレードオフ**:

| アプローチ | 長所 | 短所 |
|-----------|------|------|
| 共有設定 | チューニングが簡単、パラメータ数少 | 全Expert同じ容量 |
| Expertごと設定 | Expertごとに異なる容量可能 | K × ハイパラ数 |
| Expertグループ | 両者のバランス | 中程度の複雑さ |

**推奨アプローチ**: Expertグループから始める（例: "浅い" vs "深い" Expert）:
```python
params = {
    'mixture_num_experts': 4,
    'mixture_expert_groups': [
        {'experts': [0, 1], 'max_depth': 3, 'num_leaves': 8},   # 浅い
        {'experts': [2, 3], 'max_depth': 6, 'num_leaves': 32},  # 深い
    ]
}
```

---

## License

This project is licensed under the MIT license. Based on [Microsoft LightGBM](https://github.com/microsoft/LightGBM).
