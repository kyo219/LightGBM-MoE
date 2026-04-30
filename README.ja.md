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
    'mixture_gate_type': 'gbdt',     # 500-trial / 5-dataset スタディで全データセットで勝者
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

下記の 5 dataset / 500-trial スタディでは、MoE は **5 dataset 中 4 つで精度を改善** (sp500 のみ tie)。最大の改善は **real VIX (−15.1 % RMSE)** — ここは regime 構造 (低 vol / 高 vol 期間) が一番くっきりしている dataset。ただし MoE は **CV fold あたり 1.5-4.8 倍の計算コスト** を払う。なので「精度向上が要る *かつ* wall time の余裕がある場面で使う」が今のルール — 「regime が特徴量から観測可能でないとダメ」という以前の主張ほど狭くはない。

## ベンチマーク — 500-trial スタディ (naive-lightgbm vs MoE、5 dataset)

5-fold time-series CV、500 Optuna trials × (variant × dataset)、5 dataset (synthetic 理想 → 実マクロ/金融 → controlled-latent)。完全レポート: [`bench_results/study_500_report.md`](bench_results/study_500_report.md)、方法論と dataset 別推奨: [docs/moe/benchmark.md](docs/moe/benchmark.md)。

| Dataset | Shape | naive-lightgbm 最良 | MoE 最良 | Δ RMSE | 速度 (MoE / naive、fold あたり中央値秒) |
|---|---|---|---|---|---|
| synthetic | 2000 × 5 | 4.9765 | **4.6651** | −6.3 % | 0.663 / 0.240 = **2.76 ×** |
| real_hamilton | 311 × 12 | 0.9286 | **0.9128** | −1.7 % | 0.122 / 0.055 = **2.22 ×** |
| sp500 | 3761 × 13 | 0.0100 | 0.0100 | tie | 0.136 / 0.091 = **1.49 ×** |
| real_vix | 3762 × 13 | 2.8942 | **2.4574** | **−15.1 %** | 0.386 / 0.081 = **4.77 ×** |
| hmm | 2000 × 5 | 2.1893 | **2.1096** | −3.6 % | 0.126 / 0.074 = **1.70 ×** |

### Dataset の概要

regime-switching の適用可能性スペクトラムをカバーする 5 dataset: MoE 理想合成 1 本、実時系列 3 本 (regime-switching 文献の canonical reference)、真の regime label が既知の HMM 1 本 (gate の regime recovery 評価用)。

**Synthetic** — *feature 由来 regime、MoE 理想ケース (2000 × 5)*

i.i.d. Gauss 特徴量 5 本; regime は `X` の決定論的関数なので gate が完璧にルーティングできる:

```
regime = (0.5·X1 + 0.3·X2 − 0.2·X3 > 0)
y | regime=0 :   5·X0 + 3·X0·X2 + 2·sin(2·X3) + 10  +  ε
y | regime=1 :  −5·X0 − 2·X1²   + 3·cos(2·X4) − 10  +  ε     ε ~ N(0, 0.5²)
```

2 つの regime は同じ特徴量に **符号が逆** の係数を使うので、単一 GBDT は両者を平均せざるを得ない — これが MoE が解消できる構造の典型例。Generator: `examples/benchmark.py` の `generate_synthetic_data`。

**Real Hamilton GDP** — *Hamilton 1989 の MS-AR(4) 設定を実マクロデータで (~310 × 12)*

四半期 US Real GDP (`GDPC1`) を FRED の CSV エンドポイントから取得。[Hamilton (1989, *Econometrica*)](https://www.jstor.org/stable/1912559) に従い target は四半期成長率 `100·Δlog(GDP)`、特徴量は成長率の lag 4 (Hamilton の MS-AR(4)) と派生 MA / volatility / regime-proxy。Regime (好況 / 不況) は完全に latent — 真の oracle 列はない。Generator: `generate_real_hamilton_gnp_data`。

**S&P 500 daily returns** — *volatility regime 検証の代表例 (~3760 × 13)*

`^GSPC` の日次 Close を Yahoo Finance から (デフォルト `2010-01-01` から `2024-12-31`)、log return に変換。Target: 翌日の log return (意図的に難しい予測設定)。特徴量: lag {1, 2, 3, 5, 10} の return + MA / ローリング vol / MA クロスオーバー。Regime は latent (低 vol / 高 vol 期間)。Generator: `generate_sp500_data`。

**Real VIX** — *implied volatility level 予測 (~3760 × 13)*

CBOE `^VIX` の日次 Close を Yahoo Finance から (S&P と同じ期間)。Target: 翌日の VIX レベル。特徴量: VIX の lag + MA / ローリング統計。S&P と同じ低 vol / 高 vol 構造を implied-vol レンズで見たもの。Generator: `generate_real_vix_data`。

**HMM synthetic** — *真の regime label 既知の 3 状態 Gaussian HMM (2000 × 5)*

隠れ状態が 3 状態 Markov 連鎖 (95% 対角の persistent transition) で遷移; emission は Gauss で平均がよく分離 (`{−3, 0, +3}`)、scale は `{0.4, 0.7, 1.0}`。5 本の特徴量のうち 2 本は隠れ状態の弱い線形シグナルを持ち (gate に多少の手がかりはあるが「タダ飯」ではない)、残り 3 本は純ノイズ。**真の regime label を返す** — `diagnose_moe` で RMSE だけでなく regime recovery 精度を測れる。Generator: `generate_hmm_data`。

### キャッシュ

実時系列の取得 (FRED, yfinance) は `examples/data_cache/` 以下にキャッシュ (gitignore 済)。初回はネットワーク取得、以降はオフライン動作。

### 全 dataset で勝った設定

5 dataset 全部で絶対最良 (min) RMSE を出した唯一の categorical 設定は **`mixture_gate_type='gbdt'`** のみ。それ以外は dataset 依存 — 詳細表は [docs/moe/benchmark.md](docs/moe/benchmark.md) 参照。

| パラメータ | 普遍? | 補足 |
|---|---|---|
| `mixture_gate_type` | **`gbdt`** | 全 dataset で最良 min RMSE。`leaf_reuse` と `none` は絶対最良を取れず |
| `mixture_routing_mode` | **No** | synthetic は `token_choice`、real_hamilton / real_vix / hmm は `expert_choice` が勝った。両方探索すべき |
| `mixture_num_experts` | やや 3-4 | Q4 quartile mean がほぼの dataset で最良だが差は小さい |
| `mixture_diversity_lambda` | 0.0–0.5 を探索 | MoE の fANOVA importance で常に top-5、単一最適値はないが探索は必須 |

dataset 依存の設定 (`mixture_e_step_mode`, `mixture_init`, `mixture_r_smoothing`, `mixture_hard_m_step`, `extra_trees`, `learning_rate`) は問題ごとに探索が必要 — 詳細表は [docs/moe/benchmark.md](docs/moe/benchmark.md) 参照。

```bash
# Headline スタディの再現 (12-core / 24-thread で約 25-35 分)
python examples/comparative_study.py --trials 500 --out bench_results/study_500.json

# 動作確認 (~1 分、5 dataset 全体)
python examples/comparative_study.py --trials 10 --n-jobs 2 --out bench_results/smoke.json

# Dataset を絞って実行
python examples/comparative_study.py --trials 500 \
    --datasets synthetic,hmm --out bench_results/study_two.json
```

## ドキュメント

| トピック | ドキュメント |
|---|---|
| 全パラメータリファレンス (MoE core, Gate, Smoothing, Prediction APIs) | [docs/moe/parameters.md](docs/moe/parameters.md) |
| Optuna 探索テンプレート | [docs/moe/optuna-recipes.md](docs/moe/optuna-recipes.md) |
| 500-trial / 5-dataset ベンチマーク方法論と dataset 別推奨 | [docs/moe/benchmark.md](docs/moe/benchmark.md) |
| Per-expert ハイパーパラメータと role-based レシピ | [docs/moe/per-expert-hp.md](docs/moe/per-expert-hp.md) |
| Expert Choice routing | [docs/moe/advanced-routing.md](docs/moe/advanced-routing.md) |
| Progressive training (EvoMoE) と gate temperature annealing | [docs/moe/advanced-progressive.md](docs/moe/advanced-progressive.md) |
| Expert collapse 防止 と `diagnose_moe` | [docs/moe/advanced-collapse.md](docs/moe/advanced-collapse.md) |
| MoE コンポーネントの SHAP 分析 | [docs/moe/shap.md](docs/moe/shap.md) |
| `int8` と `use_quantized_grad` 互換性 (8 軸マトリクス) | [docs/moe/int8-compat.md](docs/moe/int8-compat.md) |
| アーキテクチャと EM ループ詳細 | [docs/moe/architecture.md](docs/moe/architecture.md) |

## ライセンス

MIT ライセンス。ベース: [Microsoft LightGBM](https://github.com/microsoft/LightGBM)。
