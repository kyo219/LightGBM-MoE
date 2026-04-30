<img src="docs/logo.png" width=300 />

LightGBM-MoE
============

**LightGBM の Regime-Switching / Mixture-of-Experts 拡張。**

[English](README.md) | [日本語](README.ja.md)

---

## 概要

LightGBM-MoE は [Microsoft LightGBM](https://github.com/microsoft/LightGBM) のフォークで、**Mixture-of-Experts (MoE) / Regime-Switching GBDT** を C++ ネイティブで実装しています:

```
ŷ(x) = Σₖ gₖ(x) · fₖ(x)
```

- `fₖ(x)`: Expert k の予測 (K 個の回帰 GBDT)
- `gₖ(x)`: Gate による expert k へのルーティング確率 (softmax)
- `K`: Expert 数 (ハイパーパラメータ)

## 動作環境

- **Python**: 3.10 以降
- **OS**: Linux (x86_64, aarch64), macOS (Intel, Apple Silicon)
- **ビルド**: ソースインストール時に CMake 3.16+ と C++ コンパイラ (GCC, Clang, Apple Clang)

## インストール

```bash
# GitHub から (ソースビルド)
pip install git+https://github.com/kyo219/LightGBM-MoE.git

# 開発用 (editable install)
git clone https://github.com/kyo219/LightGBM-MoE.git
cd LightGBM-MoE/python-package
pip install -e .
```

## クイックスタート

```python
import lightgbm_moe as lgb

params = {
    'boosting': 'mixture',           # MoE モードを有効化
    'mixture_num_experts': 3,        # Expert 数
    'mixture_gate_type': 'gbdt',     # 1000-trial スタディで全データセットで勝者
    'mixture_routing_mode': 'token_choice',
    'objective': 'regression',
    'metric': 'rmse',
    'verbose': -1,
}

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

model = lgb.train(
    params, train_data,
    num_boost_round=500,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50)],
)

# 予測
y_pred = model.predict(X_test)                      # 重み付き混合
regime = model.predict_regime(X_test)               # Regime インデックス (argmax)
regime_proba = model.predict_regime_proba(X_test)   # Gate 確率 (N, K)
expert_preds = model.predict_expert_pred(X_test)    # 各 expert の予測 (N, K)
```

## MoE が効く条件

**MoE は regime が特徴量から観測可能なときのみ効く** — その逆ではない。1000-trial スタディの結果が裏付け:

- **Synthetic** (regime が X から決定可能): MoE 3.41 vs Standard 4.96 RMSE → **+31% 改善**
- **Hamilton** (latent regime + 時系列特徴量): MoE 0.6985 vs Standard 0.6990 → ほぼ互角
- **VIX**: MoE 0.0115 vs Standard 0.0115 → 互角

精度が改善しないデータセットでは、MoE は fold あたり 1.5-1.8 倍遅い。**精度が「Yes」と言うときだけ MoE を使う、デフォルトで使うものではない。**

## ベンチマーク — 1000-trial スタディ (Standard vs MoE)

5-fold time-series CV、1000 Optuna trials × (variant × dataset)。完全レポート: [`bench_results/study_1k_report.md`](bench_results/study_1k_report.md)、方法論と dataset 別推奨: [docs/moe/benchmark.md](docs/moe/benchmark.md)。

| Dataset | Shape | Standard 最良 | MoE 最良 | MoE 速度ペナルティ |
|---|---|---|---|---|
| Synthetic (feature 由来 regime) | 2000 × 5 | 4.96 | **3.41** | 1.09 × |
| Hamilton (latent regime) | 500 × 12 | 0.6990 | 0.6985 | 1.79 × |
| VIX | 1000 × 5 | 0.0115 | 0.0115 | 1.53 × |

### 全 dataset で勝った設定 (普遍ルール)

| パラメータ | 推奨 |
|---|---|
| `mixture_num_experts` | 3-4 |
| `mixture_gate_type` | `gbdt` |
| `mixture_routing_mode` | `token_choice` |
| `extra_trees` | `true` |
| `mixture_diversity_lambda` | 0.0–0.5 を探索 (fANOVA importance で常に top-3、最適値は dataset 依存) |

dataset 依存の設定 (`mixture_e_step_mode`, `mixture_init`, `mixture_r_smoothing`, `mixture_hard_m_step`, `learning_rate`) は問題ごとに探索が必要 — 詳細は benchmark ドキュメント参照。

```bash
# Headline スタディの再現 (12-core / 24-thread で約 17 分)
python examples/comparative_study.py --trials 1000 --out bench_results/study_1k.json

# 動作確認 (約 30 秒)
python examples/comparative_study.py --trials 30 --out bench_results/smoke.json
```

## ドキュメント

| トピック | ドキュメント |
|---|---|
| 全パラメータリファレンス (MoE core, Gate, Smoothing, Prediction APIs) | [docs/moe/parameters.md](docs/moe/parameters.md) |
| Optuna 探索テンプレート | [docs/moe/optuna-recipes.md](docs/moe/optuna-recipes.md) |
| 1000-trial ベンチマーク方法論と dataset 別推奨 | [docs/moe/benchmark.md](docs/moe/benchmark.md) |
| Per-expert ハイパーパラメータと role-based レシピ | [docs/moe/per-expert-hp.md](docs/moe/per-expert-hp.md) |
| Expert Choice routing | [docs/moe/advanced-routing.md](docs/moe/advanced-routing.md) |
| Progressive training (EvoMoE) と gate temperature annealing | [docs/moe/advanced-progressive.md](docs/moe/advanced-progressive.md) |
| Expert collapse 防止 と `diagnose_moe` | [docs/moe/advanced-collapse.md](docs/moe/advanced-collapse.md) |
| MoE コンポーネントの SHAP 分析 | [docs/moe/shap.md](docs/moe/shap.md) |
| `int8` と `use_quantized_grad` 互換性 (8 軸マトリクス) | [docs/moe/int8-compat.md](docs/moe/int8-compat.md) |
| アーキテクチャと EM ループ詳細 | [docs/moe/architecture.md](docs/moe/architecture.md) |

## ライセンス

MIT ライセンス。ベース: [Microsoft LightGBM](https://github.com/microsoft/LightGBM)。
