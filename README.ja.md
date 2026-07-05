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
    'mixture_gate_type': 'gbdt',     # 勝者設定の最頻値 (11/24)。'leaf_reuse'/'none' も探索を
    'mixture_routing_mode': 'token_choice',  # 最頻値 (14/24)。'expert_choice' も探索を
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

MoE の gating が勝つのは regime 構造のあるデータ — ただし**正直なベースラインは単一の naive LightGBM ではなく、同じ総ツリー予算を持つ K-way LightGBM アンサンブル**です。seed を変えた K 個のモデルの単純平均は「タダで」分散低減を得られるので、問うべきは「学習されたルーティングがそれに勝つか」。下のスタディの答え: **regime が `X` の決定論的関数である `synthetic` で −19.7 %(vs アンサンブル)、`vix` で −5.0 %、holdout 期間に本物の regime ショック(COVID 四半期)を含む `fred_gdp` で −2.4 % 勝つ。ルーティングすべき regime 構造がない場所では引き分けか負け。** 計算コスト: MoE は naive の約 **4–10 倍**、アンサンブルの **4–5 倍**(trial あたり)。ルール: (a) regime が特徴量から観測可能だと信じられる、または評価期間に regime シフトが含まれる、かつ (b) wall time に単一 LightGBM の 5–10 倍の余裕がある場面で MoE を試す。

## ベンチマーク — holdout ファーストのスタディ (naive vs naive-ensemble vs MoE)

> **方法論ノート (v0.8.1)。** この表は v0.8.1 以前のベンチマークを置き換えるものです。旧版には監査で見つかった 4 つの欠陥がありました: (1) ヘッドラインが「500 Optuna trial の CV スコア最小値」— 選択指標そのものであり、探索空間が大きいほど有利になる winner's-curse バイアス付き推定値; (2) early stopping が採点フォールドで検証していた; (3) `vix` の特徴量が全期間平均をリークしていた(平均回帰系列に対する lookahead であり、まさに gate が悪用する regime プロキシ); (4) `sp500`/`vix` のターゲットが 1 ステップずれていた。4 つとも修正済み。旧レポートは [`bench_results/study_500_report.md`](bench_results/study_500_report.md) に記録として残置。**旧 `vix` の勝ち(−6.9 %)はリーク由来として撤回した上で、修正後プロトコルで正当に取り直し(下記 −5.0 %)**。旧 `fred_gdp` の負けは、COVID 四半期を含む holdout を実際に評価したら勝ちに反転しました。

プロトコル: 各系列の末尾 20 % は **Optuna が一度も見ない chronological holdout**。ハイパーパラメータは先頭 80 % 上の expanding-window 時系列 CV(5 fold、1 ステップ embargo、early stopping は各訓練窓の末尾で行い採点フォールドでは行わない)で選択し、CV 勝者を 1 回だけ再学習して holdout を 1 回だけ採点。**500 Optuna trials** × (variant × dataset × seed)、MoE は全探索空間(`--moe-space wide`: `gmm_features`/`kmeans_features` を含む全 init、K ∈ 2–6、gate 温度アニーリング、entropy λ、expert dropout、load-balance α、refit/regrow)、3 seeds(synthetic/hmm はデータ再抽選、TPE とモデルも再シード)、holdout RMSE mean ± std で報告。seed ごとに決定的(`--n-jobs 1`)。ビルドとデータの provenance(git commit、lib sha256、データ sha256)は出力 JSON に記録。生データ: [`bench_results/wide_v081/`](bench_results/wide_v081/)。

第 3 の variant **`naive-ensemble`** は、ハイパーパラメータを共有し `seed` だけ member ごとに変えた K-way (K ∈ {2, 3, 4}) の標準 LightGBM アンサンブル。総ツリー予算は MoE と同じ (K × `num_boost_round`)、探索空間は `naive-lightgbm` + K、trial 数も同じ。「gating は本当に仕事をしているのか、それとも K-way アンサンブルなら何でもいいのか」に答えるフェアな ablation です — [PR #33](https://github.com/kyo219/LightGBM-MoE/pull/33)。

### 結果 (holdout RMSE、3 seeds の mean ± std)

**Regime データセット** — gating 仮説が試される場所:

| Dataset | Shape | naive | ensemble | MoE | MoE vs naive | **MoE vs ensemble** *(フェアテスト)* |
|---|---|---|---|---|---|---|
| `synthetic` | 2000 × 5 | 4.840 ± 0.311 | 4.723 ± 0.327 | **3.792 ± 0.249** | −21.7 % | **−19.7 %** 🎯 |
| `vix` | 3763 × 13 | 1.768 ± 0.091 | 1.739 ± 0.051 | **1.653 ± 0.030** | −6.5 % | **−5.0 %** 🎯 |
| `fred_gdp` | 311 × 12 | 1.542 ± 0.026 | 1.530 ± 0.008 | **1.494 ± 0.010** | −3.1 % | **−2.4 %** 🎯 |
| `hmm` | 2000 × 5 | 2.216 ± 0.244 | 2.208 ± 0.242 | 2.255 ± 0.229 | +1.8 % | +2.1 % *(ensemble 勝ち)* |

**非 regime コントロールデータセット** (Grinsztajn et al. 2022 の回帰トラック) — regime 構造のない標準的な i.i.d. テーブルデータ回帰。対照群: もしここでも MoE がアンサンブルに勝つなら、「regime ルーティング」の主張は単なる容量効果と交絡していることになる。

| Dataset | Shape | naive | ensemble | MoE | MoE vs naive | **MoE vs ensemble** |
|---|---|---|---|---|---|---|
| `houses` | 10000 × 8 | 50109 ± 902 | 48816 ± 1005 | 48623 ± 683 | −3.0 % | **−0.4 %** |
| `cpu_act` | 8192 × 21 | 2.498 ± 0.30 | 2.4485 ± 0.336 | 2.431 ± 0.33 | −2.7 % | **−0.7 %** |
| `elevators` | 10000 × 18 | 0.002357 | 0.002296 | 0.002299 | −2.5 % | **+0.1 %** |
| `wine_quality` | 6497 × 11 | 0.6146 ± 0.014 | 0.6065 ± 0.013 | 0.6119 ± 0.016 | −0.4 % | **+0.9 %** |

2 つの表はセットで読んでください。regime データでは、regime が強く観測可能な場合 (`synthetic`)、volatility regime がある場合 (`vix`)、holdout が regime 断絶をまたぐ場合 (`fred_gdp`) に **容量を揃えたアンサンブル**に勝ちます。負けるのは `hmm` — regime がノイズの多いプロキシ 2 列からしか観測できないケースで、これが正直な限界です。コントロール群では MoE は *naive* に 2–3 % 勝ちますが、**アンサンブルとは全 4 データセットで ±1 % 以内の互角**: i.i.d. データでの naive に対するゲインは K-way 容量で完全に説明でき、学習されたルーティングは構造がなければ何も足しません。これが regime での勝ちに必要な帰属(attribution)の証拠です。

*(旧レポートにあった 2 つの `sp500` 翌日リターン系データセットは、本プロトコルではどの variant でも既約ノイズフロアに張り付き(全差分 < 0.4 %)、ヘッドライン表からは除外。generator と計測は repo に残っています — [`bench_results/meth2_v081/`](bench_results/meth2_v081/)。)*

### 計算コスト (trial 横断の fold あたり学習時間中央値、秒)

| Dataset | naive | ensemble | MoE | MoE / naive | MoE / ensemble |
|---|---|---|---|---|---|
| `synthetic` | 0.124 | 0.294 | 1.180 | 9.5 × | 4.0 × |
| `vix` | 0.066 | 0.124 | 0.490 | 7.4 × | 4.0 × |
| `fred_gdp` | 0.016 | 0.019 | 0.541 | 34 × *(極小データで固定費支配)* | 28 × |
| `hmm` | 0.044 | 0.078 | 0.365 | 8.3 × | 4.7 × |
| `houses` | 0.251 | 0.632 | 2.503 | 10.0 × | 4.0 × |
| `cpu_act` | 0.159 | 0.383 | 1.166 | 7.3 × | 3.0 × |
| `elevators` | 0.151 | 0.270 | 1.394 | 9.2 × | 5.2 × |
| `wine_quality` | 0.354 | 0.844 | 1.471 | 4.2 × | 1.7 × |

### Dataset の概要

regime データセットは適用可能性スペクトラムをカバー: MoE 理想合成 1 本、実時系列 2 本 (regime-switching 文献の canonical reference)、真の regime label 既知の HMM 1 本。全時系列 generator は監査済みのアライメント規約に従います: **行 t の特徴量は時刻 t までに利用可能な情報のみ、ターゲットは t + 1 の値**(`tests/python_package_test/test_benchmark_methodology.py` で不変条件として検証)。

#### `synthetic` — feature 由来 regime、MoE 理想ケース (2000 × 5)

- **Source**: 本リポジトリ内の generator (`examples/benchmark.py` の `generate_synthetic_data`)。
- **構成**: i.i.d. Gauss 特徴量 5 本; regime は `X` の決定論的関数なので gate が完璧にルーティングできる:

  ```
  regime = (0.5·X1 + 0.3·X2 − 0.2·X3 > 0)
  y | regime=0 :   5·X0 + 3·X0·X2 + 2·sin(2·X3) + 10  +  ε
  y | regime=1 :  −5·X0 − 2·X1²   + 3·cos(2·X4) − 10  +  ε     ε ~ N(0, 0.5²)
  ```

  2 つの regime は同じ特徴量に **符号が逆** の係数を使うので、単一 GBDT は両者を平均せざるを得ない — MoE が解消できる構造の典型例。

#### `vix` — CBOE Volatility Index 日次レベル (~3760 × 13)

- **Source**: Yahoo Finance、symbol [`^VIX`](https://finance.yahoo.com/quote/%5EVIX/history)、`2010-01-01` から `2024-12-31`。
- **指数定義**: [CBOE VIX](https://www.cboe.com/tradable_products/vix/)。
- **構成**: target は翌日 VIX レベル。特徴量: VIX の lag {1, 2, 3, 5, 10} + **因果的に中心化した系列**(過去のみの expanding mean で demean — v0.8.1 以前は全期間平均で demean しており「全期間平均より上か下か」が特徴量にリークしていた。現在の勝ちはリークなしで計測)上の MA / ローリング vol。Generator: `generate_vix_data`。

#### `fred_gdp` — US Real GDP、Hamilton 流 MS-AR(4) (~310 × 12)

- **Source**: FRED 系列 [`GDPC1`](https://fred.stlouisfed.org/series/GDPC1) — Real Gross Domestic Product, Chained 2017 Dollars, Quarterly, Seasonally Adjusted Annual Rate (BEA via FRED、認証不要 CSV エンドポイント)。
- **メソドロジー出典**: Hamilton, J. D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle.* **Econometrica** 57(2), 357-384. <https://www.jstor.org/stable/1912559>。
- **構成**: target は四半期成長率 `100·Δlog(GDP)`、特徴量は成長率の lag 4 (Hamilton の MS-AR(4)) と派生 MA / volatility / regime-proxy。Regime (好況 / 不況) は完全に latent。chronological holdout が COVID 四半期を含み、MoE の勝ちはまさにそこから来ています。Generator: `generate_fred_gdp_data`。

#### `hmm` — 真の regime label 既知の 3 状態 Gaussian HMM (2000 × 5)

- **Source**: 本リポジトリ内の generator (`generate_hmm_data`)。
- **メソドロジー出典**: Rabiner, L. R. (1989). *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition.* **Proceedings of the IEEE** 77(2), 257-286. <https://www.cs.ubc.ca/~murphyk/Bayes/rabiner.pdf>。
- **構成**: 隠れ状態は 3 状態 Markov 連鎖 (95 % 対角の persistent transition); emission は Gauss で平均がよく分離 `{−3, 0, +3}`、scale `{0.4, 0.7, 1.0}`。5 本の特徴量のうち 2 本は隠れ状態の弱い線形シグナルを持ち、残りは純ノイズ。**真の regime label を返す** — RMSE だけでなく regime recovery 精度を測れる。

#### コントロールデータセット — `houses`, `cpu_act`, `elevators`, `wine_quality`

- **Source**: OpenML v1 オリジナル (data_id 537 / 197 / 216 / 287)。Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). *Why do tree-based models still outperform deep learning on typical tabular data?* **NeurIPS Datasets & Benchmarks** の regression-on-numerical-features トラックから。
- **前処理**(こちらで実施、明示): 数値列のみ、NaN 行除去、seed 付きシャッフル、10k 行キャップ (tabular-benchmark の「medium-sized」慣例)。行に時間的意味はないため、chronological split はここではランダム分割と同等に振る舞います。

### キャッシュ

実データの取得 (FRED, yfinance, OpenML) は `examples/data_cache/` 以下にキャッシュ (gitignore 済)。各キャッシュファイルの sha256 はスタディの出力 JSON に記録されるので、どのランがどのバイト列を見たか追跡できます。初回はネットワーク取得、以降はオフライン動作。

### 勝った設定 — 全ノブを探索せよ

24 個の勝者 MoE 設定 (8 datasets × 3 seeds) から。ヘッドライン: **v0.8.1 以前はデフォルト値に凍結されていたノブ群が本当に仕事をしている** — 特に `vix` の勝ちは gate 温度アニーリング (T ≈ 2.4–2.9 から減衰)、expert dropout 0.1–0.2、soft M-step が主因で、init の選択ではありません。

| ノブ | 知見 |
|---|---|
| `mixture_diversity_lambda` | **23 / 24** の勝者で有効 (> 0.02) — 引き続き唯一のほぼ普遍なノブ |
| `mixture_expert_dropout_rate` | **21 / 24** で有効 (典型 0.05–0.2) — v0.8.1 以前は 0 に凍結 |
| `mixture_load_balance_alpha` | **22 / 24** で有効 — 以前は 0 に凍結 |
| `mixture_gate_entropy_lambda` | gate ありの勝者の **17 / 19** で有効 — 以前は 0 に凍結 |
| gate 温度 | **12 / 19** で `T_init > 1.5` — 旧固定 `T = 1.0` は精度を取り損ねていた |
| `mixture_refit_leaves` | **15 / 24 の勝者で ON** — v0.8 の知見(「Optuna は 5/6 データセットで refit を拒否」)を更新: v0.8.1 の stale-score 修正と ELBO トリガーのクールダウン以降、leaf refit は「bad-init 専用の安全網」ではなく普通に有効なノブ |
| `mixture_hard_m_step` | 16 / 24 で soft (False) — hard M-step デフォルトは普遍の勝者では*ない* |
| `mixture_init` / `mixture_gate_type` / `mixture_routing_mode` / K | 普遍の勝者なし (gmm 8, uniform 6, random 5, feature 系 init 4, tree 1 / gbdt 11, leaf_reuse 8, none 5 / token 14, expert 10 / K は 2–6 に分散) — **探索してください** |

```bash
# Headline スタディの再現 (holdout プロトコル、wide MoE 空間、3 seeds; seed ごとに決定的)
python examples/comparative_study.py --trials 500 --seeds 42,43,44 \
    --moe-space wide --out bench_results/study_wide.json

# 動作確認 (~2 分)
python examples/comparative_study.py --trials 10 --seeds 42 \
    --datasets synthetic,fred_gdp --out bench_results/smoke.json

# 同一プロトコルで 2 つのビルドを A/B
LGBM_MOE_PACKAGE_DIR=/path/to/other/python-package \
    python examples/comparative_study.py --trials 300 --seeds 42,43,44 \
    --out bench_results/study_other_build.json
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
