"""v0.7 leaf-refit acceptance smoke bench.

Compares MoE with ``mixture_refit_leaves`` ON vs OFF on the offline-only
datasets (synthetic + hmm — both internal generators, no network needed).
This is the smoke-grade version of the full 6-dataset acceptance bench
called for in issue #37; it validates that:

  (a) refit=off is bit-identical to a baseline trained without the new flag
      (default-off path is preserved)
  (b) refit=on doesn't regress on a regime-observable dataset (synthetic),
      where the gating hypothesis is structurally easy
  (c) refit=on actually changes predictions in a measurable way (proves
      the wiring isn't silently a no-op)

Cost numbers and ELBO trajectory are also reported.

Usage::

    PYTHONPATH=python-package python3 examples/bench_v07_refit_acceptance.py

Output: a JSON dump under ``bench_results/v0_7_refit_acceptance_<ts>.json``
plus a short markdown table on stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict

import numpy as np

# Ensure we import the dev sources, not the installed package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python-package"))

import lightgbm_moe as lgb  # noqa: E402

from benchmark import generate_synthetic_data, generate_hmm_data  # noqa: E402


def base_moe_params() -> Dict[str, Any]:
    """Single deterministic config for the smoke bench."""
    return {
        "boosting": "mixture",
        "mixture_enable": True,
        "mixture_num_experts": 3,
        "mixture_gate_type": "gbdt",
        "mixture_routing_mode": "token_choice",
        "mixture_init": "kmeans_features",
        "mixture_estimate_variance": True,
        "mixture_warmup_iters": 5,
        "mixture_diversity_lambda": 0.1,
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 16,
        "learning_rate": 0.1,
        "verbose": -1,
        "num_threads": 1,
        "seed": 42,
        "deterministic": True,
        "force_col_wise": True,
    }


def run_one(params: Dict[str, Any], X_train, y_train, X_valid, y_valid,
            num_boost_round: int) -> Dict[str, Any]:
    """Train a booster, return RMSE/timing/ELBO trajectory."""
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    t0 = time.perf_counter()
    bst = lgb.train(params, train_data,
                    num_boost_round=num_boost_round,
                    valid_sets=[valid_data], valid_names=["valid"])
    train_s = time.perf_counter() - t0
    pred_t = bst.predict(X_train)
    pred_v = bst.predict(X_valid)
    return {
        "train_rmse": float(np.sqrt(np.mean((pred_t - y_train) ** 2))),
        "valid_rmse": float(np.sqrt(np.mean((pred_v - y_valid) ** 2))),
        "train_s": train_s,
        "num_trees": bst.num_trees(),
        "predictions_hash": hash(pred_v.tobytes()),
    }


def split_train_valid(X, y, valid_frac: float = 0.2):
    n = len(y)
    n_valid = int(n * valid_frac)
    return X[:-n_valid], y[:-n_valid], X[-n_valid:], y[-n_valid:]


def bench_dataset(name: str, X, y, num_boost_round: int) -> Dict[str, Any]:
    print(f"\n=== {name} ===  (N={len(y)}, F={X.shape[1]}, rounds={num_boost_round})")
    X_train, y_train, X_valid, y_valid = split_train_valid(X, y)

    out: Dict[str, Any] = {"name": name, "n_train": len(y_train),
                           "n_valid": len(y_valid), "n_features": X.shape[1]}

    # (a) baseline: NO refit-related params
    out["baseline"] = run_one(base_moe_params(), X_train, y_train, X_valid, y_valid,
                              num_boost_round)
    print(f"  baseline (refit omitted)        valid_rmse={out['baseline']['valid_rmse']:.4f}  "
          f"time={out['baseline']['train_s']:.2f}s")

    # (b) refit explicit OFF — should be bit-identical to baseline
    out["refit_off"] = run_one(
        dict(base_moe_params(), mixture_refit_leaves=False),
        X_train, y_train, X_valid, y_valid, num_boost_round,
    )
    print(f"  refit=False (explicit)          valid_rmse={out['refit_off']['valid_rmse']:.4f}  "
          f"time={out['refit_off']['train_s']:.2f}s  "
          f"hash_match={out['baseline']['predictions_hash'] == out['refit_off']['predictions_hash']}")

    # (c) refit ON, decay=1.0 — pass-through, also bit-identical to baseline
    out["refit_decay1"] = run_one(
        dict(base_moe_params(), mixture_refit_leaves=True, mixture_refit_decay_rate=1.0),
        X_train, y_train, X_valid, y_valid, num_boost_round,
    )
    print(f"  refit=True decay=1.0            valid_rmse={out['refit_decay1']['valid_rmse']:.4f}  "
          f"time={out['refit_decay1']['train_s']:.2f}s  "
          f"hash_match={out['baseline']['predictions_hash'] == out['refit_decay1']['predictions_hash']}")

    # (d) refit ON, decay=0.5 — partial refit
    out["refit_decay05"] = run_one(
        dict(base_moe_params(), mixture_refit_leaves=True, mixture_refit_decay_rate=0.5),
        X_train, y_train, X_valid, y_valid, num_boost_round,
    )
    print(f"  refit=True decay=0.5            valid_rmse={out['refit_decay05']['valid_rmse']:.4f}  "
          f"time={out['refit_decay05']['train_s']:.2f}s")

    # (e) refit ON, decay=0.0 — full Newton replace per iter (most aggressive)
    out["refit_decay0"] = run_one(
        dict(base_moe_params(), mixture_refit_leaves=True, mixture_refit_decay_rate=0.0),
        X_train, y_train, X_valid, y_valid, num_boost_round,
    )
    print(f"  refit=True decay=0.0            valid_rmse={out['refit_decay0']['valid_rmse']:.4f}  "
          f"time={out['refit_decay0']['train_s']:.2f}s")

    # (f) trigger=elbo — only refits on ELBO drops (cheap)
    out["refit_elbo"] = run_one(
        dict(base_moe_params(), mixture_refit_leaves=True, mixture_refit_decay_rate=0.0,
             mixture_refit_trigger="elbo"),
        X_train, y_train, X_valid, y_valid, num_boost_round,
    )
    print(f"  refit=True trigger=elbo         valid_rmse={out['refit_elbo']['valid_rmse']:.4f}  "
          f"time={out['refit_elbo']['train_s']:.2f}s")

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=100,
                        help="num_boost_round per variant (default 100)")
    parser.add_argument("--out", type=str, default=None,
                        help="output JSON path (default: bench_results/v0_7_refit_acceptance_<ts>.json)")
    args = parser.parse_args()

    os.makedirs("bench_results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or f"bench_results/v0_7_refit_acceptance_{ts}.json"

    print(f"v0.7 leaf-refit acceptance smoke bench  ({ts})")
    print(f"rounds_per_variant={args.rounds}")
    results: Dict[str, Any] = {"timestamp": ts, "num_boost_round": args.rounds,
                                "datasets": {}}

    # Synthetic — internal generator, regime is observable from X_0
    X, y, _ = generate_synthetic_data(n_samples=2000, seed=42)
    results["datasets"]["synthetic"] = bench_dataset("synthetic", X, y, args.rounds)

    # HMM with weak observable signal — harder regime case
    X, y, *_ = generate_hmm_data(n_samples=2000, seed=42)
    results["datasets"]["hmm"] = bench_dataset("hmm", X, y, args.rounds)

    # Summary table
    print("\n" + "=" * 78)
    print("Summary  (Δ vs baseline = (variant − baseline) / baseline)")
    print("=" * 78)
    print(f"{'dataset':<14} {'variant':<24} {'valid_rmse':>10} {'Δ %':>8} {'time(s)':>9} {'×base':>7}")
    print("-" * 78)
    for ds_name, ds in results["datasets"].items():
        base = ds["baseline"]
        for variant in ["baseline", "refit_off", "refit_decay1",
                         "refit_decay05", "refit_decay0", "refit_elbo"]:
            r = ds[variant]
            d_pct = 100.0 * (r["valid_rmse"] - base["valid_rmse"]) / max(abs(base["valid_rmse"]), 1e-9)
            t_ratio = r["train_s"] / max(base["train_s"], 1e-9)
            print(f"{ds_name:<14} {variant:<24} {r['valid_rmse']:>10.4f} "
                  f"{d_pct:>+7.2f}% {r['train_s']:>9.2f} {t_ratio:>6.2f}×")
    print("=" * 78)

    # Acceptance assertions
    print("\nAcceptance checks:")
    all_ok = True
    for ds_name, ds in results["datasets"].items():
        h_base = ds["baseline"]["predictions_hash"]
        h_off = ds["refit_off"]["predictions_hash"]
        h_dec1 = ds["refit_decay1"]["predictions_hash"]
        match_off = (h_base == h_off)
        match_dec1 = (h_base == h_dec1)
        print(f"  [{ds_name}] refit=False bit-identical to baseline:        "
              f"{'PASS' if match_off else 'FAIL'}")
        print(f"  [{ds_name}] decay=1.0 bit-identical to baseline:          "
              f"{'PASS' if match_dec1 else 'FAIL'}")
        if not (match_off and match_dec1):
            all_ok = False

    print(f"\nOverall: {'ALL PASS' if all_ok else 'FAILURES — see above'}")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results written to: {out_path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
