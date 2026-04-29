#!/usr/bin/env python
# coding: utf-8
"""
Benchmark: int8 vs float32 — メモリ使用量・速度比較

Numeraiスタイルの離散特徴量 (int8: 0-4) を使って、
int8モードとfloat32モードの実測比較を行う。

計測項目:
  1. Python配列のメモリ使用量
  2. Dataset構築時間
  3. 学習時間 (Standard GBDT / MoE GBDT)
  4. 予測時間 (predict / predict_regime / predict_regime_proba)

Usage:
    python examples/speed_bench_int8.py
    python examples/speed_bench_int8.py --rows 100000 --cols 500
    python examples/speed_bench_int8.py --rows 500000 --cols 2000  # Numerai-scale
"""

import argparse
import gc
import os
import sys
import time
import tracemalloc
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python-package"))
import lightgbm_moe as lgb


# =============================================================================
# Config
# =============================================================================
@dataclass
class BenchConfig:
    n_rows: int = 100_000
    n_cols: int = 500
    n_boost_round: int = 100
    n_experts: int = 4
    n_repeats: int = 3
    seed: int = 42


# =============================================================================
# Helpers
# =============================================================================
def generate_data(cfg: BenchConfig):
    """Numeraiスタイルの合成データを生成"""
    rng = np.random.RandomState(cfg.seed)
    X_int8 = rng.randint(0, 5, size=(cfg.n_rows, cfg.n_cols)).astype(np.int8)
    # ターゲット: いくつかの特徴量の線形結合 + ノイズ
    coefs = rng.randn(cfg.n_cols).astype(np.float32)
    y = (X_int8.astype(np.float32) @ coefs + rng.randn(cfg.n_rows).astype(np.float32) * 5.0)
    X_f32 = X_int8.astype(np.float32)
    return X_int8, X_f32, y


def measure_time(fn, n_repeats=3):
    """関数の実行時間を計測 (中央値)"""
    times = []
    result = None
    for _ in range(n_repeats):
        gc.collect()
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times), result


def measure_peak_memory(fn):
    """関数実行中のピークメモリ増分を計測 (bytes)"""
    gc.collect()
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()
    result = fn()
    snapshot_after = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak, result


def fmt_bytes(b):
    if b < 1024:
        return f"{b} B"
    elif b < 1024 ** 2:
        return f"{b / 1024:.1f} KB"
    elif b < 1024 ** 3:
        return f"{b / 1024 ** 2:.1f} MB"
    else:
        return f"{b / 1024 ** 3:.2f} GB"


def fmt_speedup(t_base, t_new):
    if t_new == 0:
        return "inf"
    ratio = t_base / t_new
    return f"{ratio:.2f}x"


def print_header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_row(label, val_int8, val_f32, unit="s", speedup=True):
    if unit == "s":
        s_int8 = f"{val_int8:.4f}s"
        s_f32 = f"{val_f32:.4f}s"
    elif unit == "bytes":
        s_int8 = fmt_bytes(val_int8)
        s_f32 = fmt_bytes(val_f32)
    else:
        s_int8 = str(val_int8)
        s_f32 = str(val_f32)

    if speedup and val_int8 > 0:
        ratio = val_f32 / val_int8
        sp = f"({ratio:.2f}x)"
    else:
        sp = ""

    print(f"  {label:<35s}  int8: {s_int8:>12s}  f32: {s_f32:>12s}  {sp}")


# =============================================================================
# Benchmarks
# =============================================================================
def bench_array_memory(X_int8, X_f32):
    """Python配列のメモリ比較"""
    print_header("1. Array Memory")
    print_row("numpy array size", X_int8.nbytes, X_f32.nbytes, unit="bytes")


def bench_dataset_construction(X_int8, X_f32, y, cfg: BenchConfig):
    """Dataset構築の時間とメモリ"""
    print_header("2. Dataset Construction")

    def make_ds_int8():
        ds = lgb.Dataset(X_int8, label=y, free_raw_data=False)
        ds.construct()
        return ds

    def make_ds_f32():
        ds = lgb.Dataset(X_f32, label=y, free_raw_data=False)
        ds.construct()
        return ds

    t_int8, ds_int8 = measure_time(make_ds_int8, cfg.n_repeats)
    t_f32, ds_f32 = measure_time(make_ds_f32, cfg.n_repeats)
    print_row("construction time", t_int8, t_f32)

    # メモリ計測
    mem_int8, _ = measure_peak_memory(make_ds_int8)
    mem_f32, _ = measure_peak_memory(make_ds_f32)
    print_row("peak memory (tracemalloc)", mem_int8, mem_f32, unit="bytes")

    return ds_int8, ds_f32


def bench_training(ds_int8, ds_f32, cfg: BenchConfig):
    """学習速度比較"""
    print_header("3. Training (Standard GBDT)")

    params = {
        "objective": "regression",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "verbose": -1,
        "num_threads": os.cpu_count(),
        "n_jobs": -1,
    }

    def train_int8():
        return lgb.train(params, ds_int8, num_boost_round=cfg.n_boost_round)

    def train_f32():
        return lgb.train(params, ds_f32, num_boost_round=cfg.n_boost_round)

    t_int8, bst_int8 = measure_time(train_int8, cfg.n_repeats)
    t_f32, bst_f32 = measure_time(train_f32, cfg.n_repeats)
    print_row(f"train {cfg.n_boost_round} rounds", t_int8, t_f32)

    return bst_int8, bst_f32


def bench_training_moe(ds_int8, ds_f32, cfg: BenchConfig):
    """MoE学習速度比較"""
    print_header("4. Training (MoE GBDT)")

    params = {
        "objective": "regression",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "verbose": -1,
        "num_threads": os.cpu_count(),
        "n_jobs": -1,
        "mixture_enable": True,
        "mixture_num_experts": cfg.n_experts,
        "mixture_warmup_iters": 5,
        "mixture_hard_m_step": True,
    }

    def train_int8():
        return lgb.train(params, ds_int8, num_boost_round=cfg.n_boost_round)

    def train_f32():
        return lgb.train(params, ds_f32, num_boost_round=cfg.n_boost_round)

    t_int8, moe_int8 = measure_time(train_int8, cfg.n_repeats)
    t_f32, moe_f32 = measure_time(train_f32, cfg.n_repeats)
    print_row(f"MoE train {cfg.n_boost_round} rounds (K={cfg.n_experts})", t_int8, t_f32)

    return moe_int8, moe_f32


def bench_predict(bst_int8, bst_f32, X_int8, X_f32, cfg: BenchConfig):
    """予測速度比較"""
    print_header("5. Prediction (Standard GBDT)")

    def pred_int8():
        return bst_int8.predict(X_int8)

    def pred_f32():
        return bst_f32.predict(X_f32)

    t_int8, p_int8 = measure_time(pred_int8, cfg.n_repeats)
    t_f32, p_f32 = measure_time(pred_f32, cfg.n_repeats)
    print_row(f"predict ({cfg.n_rows:,} rows)", t_int8, t_f32)

    # 結果の一致確認
    p_cross = bst_int8.predict(X_f32)
    max_diff = np.max(np.abs(p_int8 - p_cross))
    print(f"  {'result consistency check':<35s}  max|int8 - f32| = {max_diff:.2e}")


def bench_predict_moe(moe_int8, moe_f32, X_int8, X_f32, cfg: BenchConfig):
    """MoE予測速度比較"""
    print_header("6. Prediction (MoE GBDT)")

    def pred_int8():
        return moe_int8.predict(X_int8)

    def pred_f32():
        return moe_f32.predict(X_f32)

    t_int8, _ = measure_time(pred_int8, cfg.n_repeats)
    t_f32, _ = measure_time(pred_f32, cfg.n_repeats)
    print_row(f"MoE predict ({cfg.n_rows:,} rows)", t_int8, t_f32)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="int8 vs float32 speed/memory benchmark")
    parser.add_argument("--rows", type=int, default=100_000, help="Number of rows (default: 100000)")
    parser.add_argument("--cols", type=int, default=500, help="Number of columns (default: 500)")
    parser.add_argument("--rounds", type=int, default=100, help="Boosting rounds (default: 100)")
    parser.add_argument("--experts", type=int, default=4, help="Number of MoE experts (default: 4)")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per measurement (default: 3)")
    args = parser.parse_args()

    cfg = BenchConfig(
        n_rows=args.rows,
        n_cols=args.cols,
        n_boost_round=args.rounds,
        n_experts=args.experts,
        n_repeats=args.repeats,
    )

    print("=" * 60)
    print("  int8 vs float32 Benchmark")
    print("  LightGBM-MoE — Numerai-style discrete features")
    print("=" * 60)
    print(f"  rows: {cfg.n_rows:,}  cols: {cfg.n_cols:,}  rounds: {cfg.n_boost_round}")
    print(f"  experts: {cfg.n_experts}  repeats: {cfg.n_repeats}")
    print(f"  threads: {os.cpu_count()}")

    # Generate data
    print("\nGenerating data...")
    X_int8, X_f32, y = generate_data(cfg)
    print(f"  X_int8: {X_int8.shape} {X_int8.dtype}  ({fmt_bytes(X_int8.nbytes)})")
    print(f"  X_f32:  {X_f32.shape} {X_f32.dtype}  ({fmt_bytes(X_f32.nbytes)})")

    # Run benchmarks
    bench_array_memory(X_int8, X_f32)
    ds_int8, ds_f32 = bench_dataset_construction(X_int8, X_f32, y, cfg)
    bst_int8, bst_f32 = bench_training(ds_int8, ds_f32, cfg)
    moe_int8, moe_f32 = bench_training_moe(ds_int8, ds_f32, cfg)
    bench_predict(bst_int8, bst_f32, X_int8, X_f32, cfg)
    bench_predict_moe(moe_int8, moe_f32, X_int8, X_f32, cfg)

    # Summary
    print_header("Summary")
    print(f"  Array memory:    int8 = {fmt_bytes(X_int8.nbytes)},  f32 = {fmt_bytes(X_f32.nbytes)}  "
          f"({X_f32.nbytes / X_int8.nbytes:.0f}x reduction)")
    print(f"\n  int8 mode eliminates the float32 conversion overhead in Python")
    print(f"  and passes 1-byte-per-feature directly to LightGBM's C++ binning.\n")


if __name__ == "__main__":
    main()
