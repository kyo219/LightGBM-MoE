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
    'mixture_gate_type': 'gbdt',     # 勝者設定の最頻値 (14/30)。'leaf_reuse'/'none' も探索を
    'mixture_routing_mode': 'token_choice',  # 最頻値 (19/30)。'expert_choice' も探索を
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

MoE の gating が勝つのは regime 構造のあるデータ — ただし**正直なベースラインは単一の naive LightGBM ではなく、同じ総ツリー予算を持つ K-way LightGBM アンサンブル**です。seed を変えた K 個のモデルの単純平均は「タダで」分散低減を得られるので、問うべきは「学習されたルーティングがそれに勝つか」。下のスタディの答え: **regime が `X` の決定論的関数である `synthetic` で −20.2 %(vs アンサンブル)、holdout 期間に本物の regime ショック(COVID 四半期)を含む `fred_gdp` で −2.3 %(全 seed で勝ち)。`vix` と `hmm` は seed によって勝ち負けが分かれ、平均では +2.1 % / +2.3 %。regime 構造のないデータではアンサンブルと互角(−1.3 % 〜 +1.6 %)。** 計算コスト: MoE は fold あたり naive の約 **3–13 倍**、アンサンブルの **1–4 倍**(`houses` は 31 倍 / 8 倍)。ルール: (a) regime が特徴量から観測可能だと信じられる、または評価期間に regime シフトが含まれる、かつ (b) wall time に単一 LightGBM の 10 倍程度の余裕がある場面で MoE を試す — CV の勝者 1 つが holdout で崩れることがあるので、複数 seed で確認してください。

## ベンチマーク — holdout ファーストのスタディ (naive vs naive-ensemble vs MoE)

> **方法論ノート (v0.8.1)。** この表は v0.8.1 以前のベンチマークを置き換えるものです。旧版には監査で見つかった 4 つの欠陥がありました: (1) ヘッドラインが「500 Optuna trial の CV スコア最小値」— 選択指標そのものであり、探索空間が大きいほど有利になる winner's-curse バイアス付き推定値; (2) early stopping が採点フォールドで検証していた; (3) `vix` の特徴量が全期間平均をリークしていた(平均回帰系列に対する lookahead であり、まさに gate が悪用する regime プロキシ); (4) `sp500`/`vix` のターゲットが 1 ステップずれていた。4 つとも修正済みで、旧結果は削除しました。旧 `vix` の勝ち(−6.9 %)はリーク由来として撤回。旧 `fred_gdp` の負けは、COVID 四半期を含む holdout を実際に評価したら勝ちに反転しました。
>
> **v0.8.4 で再計測 (2026 年 10 月)。** v0.8.1 の表にはプロトコルではなくライブラリ側の欠陥がもう 1 つありました: `mixture_gate_type="leaf_reuse"` が状態を持つ sparse-bin イテレータを OpenMP スレッド間で共有しており、gate がサンプルを非決定的に誤ルーティングしていた([PR #61](https://github.com/kyo219/LightGBM-MoE/pull/61))。修正前の `leaf_reuse` の計測はすべて無効で、勝者が `leaf_reuse` に依存していた v0.8.1 の `vix` −5.0 % の勝ちも含まれます。下の表は修正後のビルドで、全 variant に `deterministic=true` を付けているので、各 (dataset, seed) セルはビット単位で再現します。

プロトコル: 各系列の末尾 20 % は **Optuna が一度も見ない chronological holdout**。ハイパーパラメータは先頭 80 % 上の expanding-window 時系列 CV(5 fold、1 ステップ embargo、early stopping は各訓練窓の末尾で行い採点フォールドでは行わない)で選択し、CV 勝者を 1 回だけ再学習して holdout を 1 回だけ採点。**500 Optuna trials** × (variant × dataset × seed)、MoE は全探索空間(`--moe-space wide`: `gmm_features`/`kmeans_features` を含む全 init、K ∈ 2–6、gate 温度アニーリング、entropy λ、expert dropout、load-balance α、refit/regrow)、3 seeds(synthetic/hmm はデータ再抽選、TPE とモデルも再シード)、holdout RMSE mean ± std で報告。各 (dataset, seed) セルは独立プロセスで実行(`--parallel 6`、trial ごとに `num_threads=4`)し、`deterministic=true` と逐次 Optuna(`--n-jobs 1`)でビット単位で再現します。マージは dirty なツリーや別ビルドの part を拒否します。ビルドとデータの provenance(git commit、lib sha256、データ sha256)は出力に記録。結果: [`bench_results/headline_v084/`](bench_results/headline_v084/)(`study_summary.json` と `study_report.md`。99 MB の trial 単位 JSON と 33 MB のセルごとの slice plot は実行時に生成され、コミットしていません)。

第 3 の variant **`naive-ensemble`** は、ハイパーパラメータを共有し `seed` だけ member ごとに変えた K-way (K ∈ {2, 3, 4}) の標準 LightGBM アンサンブル。総ツリー予算は MoE と同じ (K × `num_boost_round`)、探索空間は `naive-lightgbm` + K、trial 数も同じ。「gating は本当に仕事をしているのか、それとも K-way アンサンブルなら何でもいいのか」に答えるフェアな ablation です — [PR #33](https://github.com/kyo219/LightGBM-MoE/pull/33)。

### 結果 (holdout RMSE、3 seeds の mean ± std)

**Regime データセット** — gating 仮説が試される場所:

| Dataset | Shape | naive | ensemble | MoE | MoE vs naive | **MoE vs ensemble** |
|---|---|---|---|---|---|---|
| `synthetic` | 2000 × 5 | 4.840 ± 0.311 | 4.723 ± 0.327 | **3.770 ± 0.222** | −22.1 % | **−20.2 %** |
| `vix` | 3763 × 13 | 1.768 ± 0.091 | **1.739 ± 0.051** | 1.776 ± 0.219 | +0.4 % | **+2.1 %** |
| `fred_gdp` | 311 × 12 | 1.553 ± 0.011 | 1.540 ± 0.017 | **1.504 ± 0.003** | −3.1 % | **−2.3 %** |
| `hmm` | 2000 × 5 | 2.216 ± 0.244 | **2.208 ± 0.242** | 2.258 ± 0.308 | +1.9 % | **+2.3 %** |

**非 regime コントロールデータセット** (Grinsztajn et al. 2022 の回帰トラック) — regime 構造のない標準的な i.i.d. テーブルデータ回帰。対照群: もしここでも MoE がアンサンブルに勝つなら、「regime ルーティング」の主張は単なる容量効果と交絡していることになる。

| Dataset | Shape | naive | ensemble | MoE | MoE vs naive | **MoE vs ensemble** |
|---|---|---|---|---|---|---|
| `houses` | 10000 × 8 | 50109 ± 902 | 48816 ± 1005 | **48668 ± 533** | −2.9 % | **−0.3 %** |
| `cpu_act` | 8192 × 21 | 2.498 ± 0.297 | 2.448 ± 0.336 | **2.416 ± 0.332** | −3.3 % | **−1.3 %** |
| `elevators` | 10000 × 18 | 0.002357 ± 0.000080 | **0.002296 ± 0.000060** | 0.002332 ± 0.000099 | −1.1 % | **+1.6 %** |
| `wine_quality` | 6497 × 11 | 0.6146 ± 0.0141 | 0.6065 ± 0.0126 | **0.6034 ± 0.0120** | −1.8 % | **−0.5 %** |

**ノイズフロアのデータセット** (S&P 500 の翌日リターン) — テストではなくサニティチェック。どの variant も既約ノイズフロアに張り付き(全差分 ≤ 0.4 %)、MoE は構造のないところに構造をでっち上げません。

| Dataset | Shape | naive | ensemble | MoE | MoE vs naive | **MoE vs ensemble** |
|---|---|---|---|---|---|---|
| `sp500_basic` | 3762 × 13 | **0.01102 ± 0.00001** | 0.01103 ± 0.00002 | 0.01103 ± 0.00001 | +0.1 % | **+0.0 %** |
| `sp500` | 3711 × 28 | 0.01109 ± 0.00003 | **0.01104 ± 0.00001** | 0.01105 ± 0.00001 | −0.4 % | **+0.0 %** |

表はセットで読んでください。regime データでは、regime が `X` から強く観測可能な場合 (`synthetic`、−20 %、全 seed) と、holdout が regime 断絶をまたぐ場合 (`fred_gdp`、−2.3 %、全 seed、seed 間のばらつきもアンサンブルの 1/5) に **容量を揃えたアンサンブル**に勝ちます。`vix` と `hmm` は混戦です。`vix` では MoE は全 seed で CV に勝ち、holdout でも 3 seed 中 2 つで勝つ(−3 % 〜 −7 %)ものの、残り 1 seed の勝者が holdout で崩れて(+15 %)平均を +2.1 % に押し上げています。regime がノイズの多いプロキシ 2 列からしか観測できない `hmm` では 2 seed で互角、1 seed で負け。コントロール群では MoE は *naive* に 1–3 % 勝ちますが、**アンサンブルとは互角(−1.3 % 〜 +1.6 %)**: i.i.d. データでの naive に対するゲインは K-way 容量で説明でき、学習されたルーティングは構造がなければ何も足しません。これが regime での勝ちに必要な帰属(attribution)の証拠です。

### 計算コスト (trial 横断の fold あたり学習時間中央値、seed 平均、秒)

6 セルを `num_threads=4` で同時実行した状態での計測なので、秒数そのものは以前の表と比較できません。見るべきは比率です。

| Dataset | naive | ensemble | MoE | MoE / naive | MoE / ensemble |
|---|---|---|---|---|---|
| `synthetic` | 0.041 | 0.210 | 0.311 | 7.5 × | 1.5 × |
| `fred_gdp` | 0.013 | 0.108 | 0.098 | 7.3 × | 0.9 × |
| `sp500_basic` | 0.018 | 0.086 | 0.109 | 6.0 × | 1.3 × |
| `sp500` | 0.026 | 0.189 | 0.200 | 7.5 × | 1.1 × |
| `vix` | 0.038 | 0.146 | 0.406 | 10.7 × | 2.8 × |
| `hmm` | 0.032 | 0.184 | 0.163 | 5.0 × | 0.9 × |
| `houses` | 0.136 | 0.512 | 4.189 | 30.8 × | 8.2 × |
| `cpu_act` | 0.074 | 0.309 | 0.638 | 8.6 × | 2.1 × |
| `elevators` | 0.064 | 0.203 | 0.845 | 13.1 × | 4.2 × |
| `wine_quality` | 0.149 | 0.598 | 0.505 | 3.4 × | 0.8 × |

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
- **構成**: target は翌日 VIX レベル。特徴量: VIX の lag {1, 2, 3, 5, 10} + **因果的に中心化した系列**(過去のみの expanding mean で demean — v0.8.1 以前は全期間平均で demean しており「全期間平均より上か下か」が特徴量にリークしていた。現在の数値はリークなしで計測)上の MA / ローリング vol。Generator: `generate_vix_data`。

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

30 個の勝者 MoE 設定 (10 datasets × 3 seeds) から。**v0.8.1 以前はデフォルト値に凍結されていたノブ群**はほぼ全勝者で有効。構造的な選択 (init、gate type、routing、K) には普遍の勝者がありません。

| ノブ | 知見 |
|---|---|
| `mixture_diversity_lambda` | **28 / 30** の勝者で有効 (> 0.02) |
| `mixture_expert_dropout_rate` | **26 / 30** で有効 — v0.8.1 以前は 0 に凍結 |
| `mixture_load_balance_alpha` | **29 / 30** で有効 — 以前は 0 に凍結 |
| `mixture_gate_entropy_lambda` | gate ありの勝者の **21 / 25** で有効 — 以前は 0 に凍結 |
| gate 温度 | gate ありの勝者の **12 / 25** で `T_init > 1.5` — `T = 1.0` 固定にせず探索を |
| `mixture_refit_leaves` | **実効 ON は 11 / 30**(`leaf_reuse` では強制無効なので gbdt/none の勝者のみ数える)— デフォルトではなく普通に探索すべきノブ |
| `mixture_regrow_oldest_trees` | 実効 ON は **6 / 30** |
| `mixture_hard_m_step` | 15 / 30 で soft (False) — 五分五分なので探索を |
| `mixture_init` / `mixture_gate_type` / `mixture_routing_mode` / K | 普遍の勝者なし (random 9, gmm 6, gmm_features 5, uniform 5, tree_hierarchical 3, kmeans_features 2 / gbdt 14, leaf_reuse 11, none 5 / token 19, expert 11 / K は 2–6 に分散) — **探索してください** |

```bash
# Headline スタディの再現: 10 datasets × 3 seeds × 3 variants、500 trials、wide MoE 空間。
# 12 コアで約 5 時間。終わったセルは再利用されるので、同じコマンドの再実行で再開できる。
python examples/comparative_study.py --parallel 6 --out bench_results/headline_v084/study.json

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
| Per-expert ハイパーパラメータと role-based レシピ | [docs/moe/per-expert-hp.md](docs/moe/per-expert-hp.md) |
| Expert Choice routing | [docs/moe/advanced-routing.md](docs/moe/advanced-routing.md) |
| Progressive training (EvoMoE) と gate temperature annealing | [docs/moe/advanced-progressive.md](docs/moe/advanced-progressive.md) |
| Expert collapse 防止 と `diagnose_moe` | [docs/moe/advanced-collapse.md](docs/moe/advanced-collapse.md) |
| MoE コンポーネントの SHAP 分析 | [docs/moe/shap.md](docs/moe/shap.md) |
| `int8` と `use_quantized_grad` 互換性 (8 軸マトリクス) | [docs/moe/int8-compat.md](docs/moe/int8-compat.md) |
| アーキテクチャと EM ループ詳細 | [docs/moe/architecture.md](docs/moe/architecture.md) |

## ライセンス

MIT ライセンス。ベース: [Microsoft LightGBM](https://github.com/microsoft/LightGBM)。
