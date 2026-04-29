#!/usr/bin/env python
# coding: utf-8
"""
perf_bench.py — int8前提の高速化施策の前後比較ベンチ

固定の Numerai-like int8 データセットで Standard GBDT / MoE を訓練し、
複数のconfig (use_quantized_grad on/off, num_grad_quant_bins, etc) を
matrix で回して train/predict 時間・peak RSS・RMSE を JSON 出力する。

ビルド/コード変更の前後で同条件で実行し、save された JSON を diff することで
施策ごとの効果を測る用途。

Usage:
    # baseline 計測
    python examples/perf_bench.py --out results_baseline.json

    # 改修後の計測
    python examples/perf_bench.py --out results_phase1.json

    # 比較
    python examples/perf_bench.py --compare results_baseline.json results_phase1.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python-package"))
import lightgbm_moe as lgb


# =============================================================================
# Config
# =============================================================================
@dataclass
class BenchConfig:
    n_rows: int = 200_000
    n_cols: int = 800
    n_boost_round: int = 100
    n_experts: int = 4
    n_repeats: int = 2
    seed: int = 42
    threads: int = 8


@dataclass
class RunResult:
    label: str
    model: str  # "gbdt" or "moe"
    params: dict
    train_time_s: float
    predict_time_s: float
    peak_rss_mb: float
    rmse_train: float
    rmse_holdout: float
    n_trees: int


@dataclass
class BenchReport:
    config: dict
    cpu_info: str
    build_info: str
    runs: list = field(default_factory=list)


# =============================================================================
# Helpers
# =============================================================================
def get_peak_rss_mb() -> float:
    """ru_maxrss は Linux で KB、macOS で B。Linux 前提で KB→MB"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def reset_peak_rss():
    """新しい run の前に peak をリセットしたいが ru_maxrss はプロセス全体の最大なのでリセット不可。
    各 run の前後で差分を取る。"""
    return get_peak_rss_mb()


def get_cpu_info() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "unknown"


def get_build_info() -> str:
    """git の HEAD と .so の md5 を返す"""
    import hashlib
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    try:
        head = subprocess.check_output(["git", "-C", repo, "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        head = "?"
    so = os.path.join(repo, "lib_lightgbm.so")
    if os.path.exists(so):
        with open(so, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()[:8]
    else:
        md5 = "?"
    return f"git={head} .so md5={md5}"


# =============================================================================
# Data
# =============================================================================
def generate_data(cfg: BenchConfig):
    """Numerai-like: int8 [0,4]、ターゲットは線形+非線形"""
    rng = np.random.RandomState(cfg.seed)
    X = rng.randint(0, 5, size=(cfg.n_rows, cfg.n_cols)).astype(np.int8)

    # ターゲット: 一部の特徴量に対する線形 + 交互作用 + ノイズ
    coefs = rng.randn(cfg.n_cols).astype(np.float32) * 0.1
    Xf = X.astype(np.float32)
    y_lin = Xf @ coefs
    # 軽い交互作用 (regime っぽい構造)
    interaction = (Xf[:, 0] - 2.0) * (Xf[:, 1] - 2.0) * 0.5
    noise = rng.randn(cfg.n_rows).astype(np.float32) * 1.0
    y = (y_lin + interaction + noise).astype(np.float32)

    # 80/20 split
    split = int(cfg.n_rows * 0.8)
    return X[:split], y[:split], X[split:], y[split:]


# =============================================================================
# Run a single config
# =============================================================================
def run_one(label: str, model: str, params: dict, X_tr, y_tr, X_va, y_va, cfg: BenchConfig) -> RunResult:
    """指定パラメータで訓練+予測、メトリクスを返す"""
    base_rss = get_peak_rss_mb()

    train_times = []
    predict_times = []
    rmse_va = float("nan")
    rmse_tr = float("nan")
    n_trees = 0

    for rep in range(cfg.n_repeats):
        gc.collect()
        ds = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)

        t0 = time.perf_counter()
        bst = lgb.train(params, ds, num_boost_round=cfg.n_boost_round)
        t1 = time.perf_counter()
        train_times.append(t1 - t0)

        t0 = time.perf_counter()
        pred_va = bst.predict(X_va)
        t1 = time.perf_counter()
        predict_times.append(t1 - t0)

        if rep == 0:
            pred_tr = bst.predict(X_tr)
            rmse_tr = float(np.sqrt(np.mean((pred_tr - y_tr) ** 2)))
            rmse_va = float(np.sqrt(np.mean((pred_va - y_va) ** 2)))
            try:
                n_trees = bst.num_trees()
            except Exception:
                n_trees = 0

        del bst, ds
        gc.collect()

    peak_rss = get_peak_rss_mb()

    return RunResult(
        label=label,
        model=model,
        params=params,
        train_time_s=float(np.median(train_times)),
        predict_time_s=float(np.median(predict_times)),
        peak_rss_mb=float(peak_rss - base_rss),
        rmse_train=rmse_tr,
        rmse_holdout=rmse_va,
        n_trees=n_trees,
    )


# =============================================================================
# Build run matrix
# =============================================================================
def build_matrix(cfg: BenchConfig) -> list[tuple[str, str, dict]]:
    """(label, model, params) のリストを返す。
    model='gbdt' or 'moe' に応じて lgb.train への params を構築。"""
    common = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 31,
        "max_depth": 7,
        "learning_rate": 0.05,
        "verbose": -1,
        "num_threads": cfg.threads,
        "seed": cfg.seed,
    }

    # GBDT パターン
    matrix: list[tuple[str, str, dict]] = []

    # 1. baseline (float grad)
    matrix.append(("gbdt/float", "gbdt", {**common}))

    # 2. quantized grad (4/16/32 bins)
    for bins in (4, 16, 32):
        matrix.append((
            f"gbdt/quant{bins}",
            "gbdt",
            {**common, "use_quantized_grad": True, "num_grad_quant_bins": bins},
        ))

    # MoE パターン
    moe_common = {
        **common,
        "boosting": "mixture",
        "mixture_num_experts": cfg.n_experts,
        "mixture_warmup_iters": 5,
        "mixture_hard_m_step": True,
        "mixture_gate_type": "gbdt",
    }

    # 3. MoE float grad
    matrix.append(("moe/float", "moe", {**moe_common}))

    # 4. MoE quantized grad
    for bins in (16, 32):
        matrix.append((
            f"moe/quant{bins}",
            "moe",
            {**moe_common, "use_quantized_grad": True, "num_grad_quant_bins": bins},
        ))

    return matrix


# =============================================================================
# Main
# =============================================================================
def cmd_run(args):
    cfg = BenchConfig(
        n_rows=args.rows,
        n_cols=args.cols,
        n_boost_round=args.rounds,
        n_experts=args.experts,
        n_repeats=args.repeats,
        seed=args.seed,
        threads=args.threads,
    )

    print(f"=== perf_bench.py ===")
    print(f"  rows={cfg.n_rows:,}  cols={cfg.n_cols}  rounds={cfg.n_boost_round}  threads={cfg.threads}")
    print(f"  CPU: {get_cpu_info()}")
    print(f"  Build: {get_build_info()}")

    print(f"\nGenerating data...")
    X_tr, y_tr, X_va, y_va = generate_data(cfg)
    print(f"  X_tr={X_tr.shape} {X_tr.dtype}  X_va={X_va.shape}")

    matrix = build_matrix(cfg)
    print(f"\nRunning {len(matrix)} configs...\n")

    report = BenchReport(
        config=asdict(cfg),
        cpu_info=get_cpu_info(),
        build_info=get_build_info(),
    )

    print(f"  {'label':<22s}  {'train':>8s}  {'predict':>8s}  {'peak_MB':>8s}  {'rmse_va':>8s}  {'trees':>6s}")
    print(f"  {'-' * 22}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 6}")

    for label, model, params in matrix:
        try:
            r = run_one(label, model, params, X_tr, y_tr, X_va, y_va, cfg)
            report.runs.append(asdict(r))
            print(f"  {r.label:<22s}  {r.train_time_s:>7.2f}s  {r.predict_time_s:>7.3f}s  "
                  f"{r.peak_rss_mb:>7.0f}M  {r.rmse_holdout:>8.4f}  {r.n_trees:>6d}")
        except Exception as e:
            print(f"  {label:<22s}  FAILED: {e}")
            report.runs.append({"label": label, "model": model, "params": params, "error": str(e)})

    # Save
    with open(args.out, "w") as f:
        json.dump(asdict(report), f, indent=2)
    print(f"\nSaved → {args.out}")


def cmd_compare(args):
    """2つの結果を並べて比較表示"""
    with open(args.before) as f:
        before = json.load(f)
    with open(args.after) as f:
        after = json.load(f)

    print(f"=== Comparison ===")
    print(f"  before: {args.before}  build={before.get('build_info', '?')}")
    print(f"  after : {args.after}  build={after.get('build_info', '?')}")

    by_label_before = {r["label"]: r for r in before["runs"] if "error" not in r}
    by_label_after = {r["label"]: r for r in after["runs"] if "error" not in r}

    print(f"\n  {'label':<22s}  {'train (b→a)':>22s}  {'predict (b→a)':>22s}  {'rmse (b→a)':>22s}")
    print(f"  {'-' * 22}  {'-' * 22}  {'-' * 22}  {'-' * 22}")

    labels = sorted(set(by_label_before) | set(by_label_after))
    for label in labels:
        b = by_label_before.get(label)
        a = by_label_after.get(label)
        if b is None or a is None:
            continue
        tr_b, tr_a = b["train_time_s"], a["train_time_s"]
        pr_b, pr_a = b["predict_time_s"], a["predict_time_s"]
        rm_b, rm_a = b["rmse_holdout"], a["rmse_holdout"]
        tr_speedup = tr_b / tr_a if tr_a > 0 else float("inf")
        pr_speedup = pr_b / pr_a if pr_a > 0 else float("inf")
        rm_diff = rm_a - rm_b
        print(f"  {label:<22s}  "
              f"{tr_b:>6.2f}s → {tr_a:>6.2f}s ({tr_speedup:>4.2f}x)  "
              f"{pr_b:>6.3f}s → {pr_a:>6.3f}s ({pr_speedup:>4.2f}x)  "
              f"{rm_b:>7.4f} → {rm_a:>7.4f} ({rm_diff:+.4f})")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    pr = sub.add_parser("run", help="Run benchmark and save JSON")
    pr.add_argument("--rows", type=int, default=200_000)
    pr.add_argument("--cols", type=int, default=800)
    pr.add_argument("--rounds", type=int, default=100)
    pr.add_argument("--experts", type=int, default=4)
    pr.add_argument("--repeats", type=int, default=2)
    pr.add_argument("--seed", type=int, default=42)
    pr.add_argument("--threads", type=int, default=8)
    pr.add_argument("--out", type=str, required=True, help="Output JSON path")

    pc = sub.add_parser("compare", help="Compare two JSON results")
    pc.add_argument("before", type=str)
    pc.add_argument("after", type=str)

    # Default: 'run' で動かしやすく
    p.add_argument("--rows", type=int, default=None)
    p.add_argument("--cols", type=int, default=None)
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--experts", type=int, default=None)
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--out", type=str, default=None)

    args = p.parse_args()

    if args.cmd == "compare":
        cmd_compare(args)
    else:
        # default to 'run'; fill defaults
        if args.cmd is None:
            args.cmd = "run"
        if args.rows is None: args.rows = 200_000
        if args.cols is None: args.cols = 800
        if args.rounds is None: args.rounds = 100
        if args.experts is None: args.experts = 4
        if args.repeats is None: args.repeats = 2
        if args.seed is None: args.seed = 42
        if args.threads is None: args.threads = 8
        if args.out is None:
            print("error: --out is required for 'run'", file=sys.stderr)
            sys.exit(2)
        cmd_run(args)


if __name__ == "__main__":
    main()
