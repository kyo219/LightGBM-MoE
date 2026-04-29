#!/usr/bin/env python
# coding: utf-8
"""
compat_matrix.py — 機能軸 × int8(use_quantized_grad) 互換性マトリクス

各機能軸を1つずつ動かしながら quantized_grad off/on の RMSE と動作を記録。
Phase 2 で適用した MoE quant 修正の効果を全機能軸でカバー確認する用途。

短時間で全部回るように、データセットは小さめ (5K x 100, 30 rounds, K=3)。
互換性の有無を二値で見るのが目的、絶対RMSEは小さいスケール参考値。
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python-package"))
import lightgbm_moe as lgb


@dataclass
class Result:
    name: str
    quant: bool
    rmse: float
    train_s: float
    status: str  # "ok" / "regression" / "crash"
    note: str = ""


def make_data(seed=42, n=5000, p=100):
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 5, size=(n, p)).astype(np.int8)
    coefs = rng.randn(p).astype(np.float32) * 0.1
    y_lin = X.astype(np.float32) @ coefs
    interaction = (X[:, 0].astype(np.float32) - 2.0) * (X[:, 1].astype(np.float32) - 2.0) * 0.3
    y = (y_lin + interaction + rng.randn(n).astype(np.float32) * 0.5).astype(np.float32)
    return X, y


def run(name: str, params: dict, X, y, n_rounds=30) -> Result:
    """1組合せを訓練→予測→RMSE計算"""
    quant = bool(params.get("use_quantized_grad", False))
    base = {
        "objective": "regression",
        "verbose": -1,
        "num_threads": 4,
        "learning_rate": 0.1,
        "num_leaves": 15,
        "max_depth": 5,
        "seed": 42,
    }
    full = {**base, **params}
    try:
        ds = lgb.Dataset(X, label=y)
        # Suppress all stdout/stderr from C++ to keep matrix output clean.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            t0 = time.perf_counter()
            bst = lgb.train(full, ds, num_boost_round=n_rounds)
            t1 = time.perf_counter()
            pred = bst.predict(X)
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        if not np.isfinite(rmse):
            return Result(name, quant, rmse, t1 - t0, "crash", "non-finite RMSE")
        return Result(name, quant, rmse, t1 - t0, "ok")
    except Exception as e:
        return Result(name, quant, float("nan"), 0.0, "crash", f"{type(e).__name__}: {e}".splitlines()[0][:80])


def matrix(X, y) -> list[Result]:
    """全機能軸を巡回。各軸で baseline (quant=off) と quant=on を取る"""
    results: list[Result] = []
    K = 3
    common_moe = {
        "boosting": "mixture",
        "mixture_num_experts": K,
        "mixture_warmup_iters": 3,
    }

    cases: list[tuple[str, dict]] = []

    # Standard GBDT (control)
    cases.append(("gbdt/standard", {}))

    # MoE base (default config)
    cases.append(("moe/default", {**common_moe}))

    # --- E-step modes ---
    for mode in ("em", "loss_only", "gate_only"):
        cases.append((f"moe/e_step={mode}", {**common_moe, "mixture_e_step_mode": mode}))

    # --- Hard / soft M-step ---
    for hard in (True, False):
        cases.append((f"moe/hard={hard}", {**common_moe, "mixture_hard_m_step": hard}))

    # --- Gate types ---
    for gt in ("gbdt", "none", "leaf_reuse"):
        cases.append((f"moe/gate={gt}", {**common_moe, "mixture_gate_type": gt}))

    # --- Routing modes ---
    for rm in ("token_choice", "expert_choice"):
        cases.append((f"moe/route={rm}", {**common_moe, "mixture_routing_mode": rm}))

    # --- Smoothing ---
    for sm in ("none", "ema", "markov", "momentum"):
        extra = {"mixture_r_smoothing": sm}
        if sm != "none":
            extra["mixture_smoothing_lambda"] = 0.5
        cases.append((f"moe/smooth={sm}", {**common_moe, **extra}))

    # --- Init methods ---
    for init in ("uniform", "random", "quantile", "balanced_kmeans", "gmm", "tree_hierarchical"):
        cases.append((f"moe/init={init}", {**common_moe, "mixture_init": init}))

    # --- Progressive (EvoMoE) ---
    cases.append((
        "moe/evomoe",
        {**common_moe, "mixture_progressive_mode": "evomoe", "mixture_seed_iterations": 10, "mixture_spawn_perturbation": 0.5},
    ))

    # --- Per-expert structural HP (MoE-PE) ---
    cases.append((
        "moe-pe/per_expert_hp",
        {
            **common_moe,
            "mixture_expert_max_depths": [3, 5, 7],
            "mixture_expert_num_leaves": [8, 16, 32],
            "mixture_expert_min_data_in_leaf": [50, 20, 5],
        },
    ))

    # --- Regularizations / advanced ---
    cases.append(("moe/diversity_lambda", {**common_moe, "mixture_diversity_lambda": 0.3}))
    cases.append(("moe/gate_entropy", {**common_moe, "mixture_gate_entropy_lambda": 0.05}))
    cases.append(("moe/expert_dropout", {**common_moe, "mixture_expert_dropout_rate": 0.2}))
    cases.append(("moe/load_balance", {**common_moe, "mixture_load_balance_alpha": 0.5}))
    cases.append(("moe/gate_temperature", {
        **common_moe,
        "mixture_gate_temperature_init": 2.0,
        "mixture_gate_temperature_final": 0.5,
    }))
    cases.append(("moe/adaptive_lr", {**common_moe, "mixture_adaptive_lr": True}))
    cases.append(("moe/dropout_curriculum", {
        **common_moe,
        "mixture_expert_dropout_rate": 0.2,
        "mixture_dropout_schedule": "linear",
        "mixture_dropout_rate_min": 0.0,
        "mixture_dropout_rate_max": 0.3,
    }))

    # Run each case in two flavors: quant off (baseline), quant on (use_quantized_grad)
    for name, params in cases:
        for quant in (False, True):
            full = dict(params)
            if quant:
                full["use_quantized_grad"] = True
                full["num_grad_quant_bins"] = 32
            results.append(run(name, full, X, y))

    return results


def main():
    print("Generating data...")
    X, y = make_data(n=5000, p=100)
    print(f"  X={X.shape} {X.dtype}  y range=({y.min():.2f},{y.max():.2f})")

    print("\nRunning compatibility matrix...\n")
    results = matrix(X, y)

    # Group by name; show quant=off, quant=on side by side
    by_name: dict[str, dict[str, Result]] = {}
    for r in results:
        by_name.setdefault(r.name, {})["quant" if r.quant else "float"] = r

    print(f"  {'feature':<32s}  {'float RMSE':>10s}  {'quant RMSE':>10s}  {'speedup':>8s}  {'compat':>8s}")
    print(f"  {'-' * 32}  {'-' * 10}  {'-' * 10}  {'-' * 8}  {'-' * 8}")

    summary = []
    for name in sorted(by_name):
        f = by_name[name].get("float")
        q = by_name[name].get("quant")
        if f is None or q is None:
            continue

        if q.status == "crash":
            compat = "CRASH"
        elif f.status == "crash":
            compat = "f-crash"
        else:
            # Compare RMSE: degradation > 30% = regression
            if not np.isfinite(f.rmse) or not np.isfinite(q.rmse):
                compat = "?"
            else:
                ratio = q.rmse / max(f.rmse, 1e-10)
                if ratio > 1.30:
                    compat = "REGRESS"
                elif ratio > 1.05:
                    compat = "minor"
                else:
                    compat = "ok"

        speedup = (f.train_s / q.train_s) if (q.train_s > 0) else float("nan")
        f_rmse = f"{f.rmse:.4f}" if np.isfinite(f.rmse) else f.note[:10]
        q_rmse = f"{q.rmse:.4f}" if np.isfinite(q.rmse) else q.note[:10]
        sp = f"{speedup:.2f}x" if np.isfinite(speedup) else "—"
        print(f"  {name:<32s}  {f_rmse:>10s}  {q_rmse:>10s}  {sp:>8s}  {compat:>8s}")

        summary.append({
            "feature": name,
            "float_rmse": f.rmse,
            "quant_rmse": q.rmse,
            "float_train_s": f.train_s,
            "quant_train_s": q.train_s,
            "speedup": speedup,
            "compat": compat,
            "float_status": f.status,
            "quant_status": q.status,
            "float_note": f.note,
            "quant_note": q.note,
        })

    out = "bench_results/compat_matrix.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
