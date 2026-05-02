"""v0.7 leaf-refit per-config ablation — eliminate TPE search divergence.

The 500-trial Optuna runs (`bench_results/study_v07_*.json`) compared MoE under
different refit triggers across the entire search space. That comparison has a
methodological problem: when refit fires for any single trial, the TPE sampler
sees a different (params → rmse) sample, the search trajectory diverges, and
the "best" found across 500 trials lands in a different basin. The synthetic
+18-32% regression in the global-search comparison was almost entirely this
TPE-divergence effect, not a real refit regression.

This bench eliminates that confounder. For each of the 6 datasets:

  1. Load the v0.6 winning MoE config from
     ``bench_results/study_500_3way_20260502_200635.json`` (the headline
     numbers in the README).
  2. Re-run that exact config under 4 refit modes, holding everything else
     constant: ``off``, ``elbo``, ``every_n=10``, ``always``.
  3. Report 5-fold time-series CV mean RMSE, std, mean per-fold training
     time, and the ratio of refit-mode time to off time.

This answers the question issue #37 actually wants answered: **given THE best
config, does the refit pass improve, hurt, or do nothing?**

Usage::

    PYTHONPATH=python-package python3 examples/bench_v07_per_config.py

Output: JSON + markdown side-by-side at
``bench_results/bench_v07_per_config_<ts>.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python-package"))

import lightgbm_moe as lgb  # noqa: E402

from benchmark import (  # noqa: E402
    generate_synthetic_data,
    generate_fred_gdp_data,
    generate_sp500_basic_data,
    generate_sp500_data,
    generate_vix_data,
    generate_hmm_data,
)


DATASET_GENERATORS = {
    "synthetic":   lambda seed: generate_synthetic_data(seed=seed),
    "fred_gdp":    lambda seed: generate_fred_gdp_data(seed=seed),
    "sp500_basic": lambda seed: generate_sp500_basic_data(seed=seed),
    "sp500":       lambda seed: generate_sp500_data(seed=seed),
    "vix":         lambda seed: generate_vix_data(seed=seed),
    "hmm":         lambda seed: generate_hmm_data(seed=seed),
}


REFIT_VARIANTS: List[Dict[str, Any]] = [
    {"name": "off",     "params": {}},
    {"name": "elbo",    "params": {"mixture_refit_leaves": True,
                                    "mixture_refit_trigger": "elbo",
                                    "mixture_refit_decay_rate": 0.0}},
    {"name": "every_n", "params": {"mixture_refit_leaves": True,
                                    "mixture_refit_trigger": "every_n",
                                    "mixture_refit_every_n": 10,
                                    "mixture_refit_decay_rate": 0.0}},
    {"name": "always",  "params": {"mixture_refit_leaves": True,
                                    "mixture_refit_trigger": "always",
                                    "mixture_refit_decay_rate": 0.0}},
]


def load_v06_best_params(json_path: str) -> Dict[str, Dict[str, Any]]:
    """Pull the moe variant's best_params for each dataset from a study JSON."""
    with open(json_path) as f:
        d = json.load(f)
    out = {}
    for ds in DATASET_GENERATORS:
        if ds not in d:
            continue
        bp = d[ds].get("moe", {}).get("best_params")
        if bp:
            out[ds] = bp
    return out


def base_static_params() -> Dict[str, Any]:
    return {
        "objective": "regression",
        "boosting": "mixture",
        "verbose": -1,
        "num_threads": 4,
        "seed": 42,
    }


def evaluate_cv(X, y, params, n_splits: int, num_boost_round: int):
    """Same CV as comparative_study.evaluate_cv_timed; collects per-fold rmse."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses, times = [], []
    for tr_idx, va_idx in tscv.split(X):
        Xt, Xv = X[tr_idx], X[va_idx]
        yt, yv = y[tr_idx], y[va_idx]
        train = lgb.Dataset(Xt, label=yt)
        valid = lgb.Dataset(Xv, label=yv, reference=train)
        try:
            t0 = time.perf_counter()
            model = lgb.train(
                params, train,
                num_boost_round=num_boost_round,
                valid_sets=[valid],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )
            t1 = time.perf_counter()
            pred = model.predict(Xv)
            rmses.append(float(np.sqrt(mean_squared_error(yv, pred))))
            times.append(t1 - t0)
        except Exception as e:  # noqa: BLE001
            rmses.append(float("inf"))
            times.append(0.0)
            print(f"    [warn] fold failed: {e}")
    return rmses, times


def bench_dataset(name: str, X, y, v06_best: Dict[str, Any],
                  n_splits: int, num_boost_round: int) -> Dict[str, Any]:
    print(f"\n=== {name} ===  N={len(y)}  F={X.shape[1]}")
    print(f"  v0.6 best config: K={v06_best.get('mixture_num_experts')}, "
          f"init={v06_best.get('mixture_init')}, "
          f"gate={v06_best.get('mixture_gate_type')}, "
          f"routing={v06_best.get('mixture_routing_mode')}, "
          f"e_step={v06_best.get('mixture_e_step_mode')}, "
          f"div={v06_best.get('mixture_diversity_lambda', 0):.3f}")

    out: Dict[str, Any] = {"name": name, "v06_best": dict(v06_best),
                           "variants": {}}
    for v in REFIT_VARIANTS:
        params = {**base_static_params(), **v06_best, **v["params"]}
        rmses, times = evaluate_cv(X, y, params, n_splits, num_boost_round)
        finite = [r for r in rmses if np.isfinite(r)]
        rmse_mean = float(np.mean(finite)) if finite else float("inf")
        rmse_std = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
        time_mean = float(np.mean(times))
        out["variants"][v["name"]] = {
            "rmse_mean": rmse_mean,
            "rmse_std": rmse_std,
            "rmse_per_fold": rmses,
            "time_mean": time_mean,
            "time_per_fold": times,
            "n_folds": len(rmses),
        }
        print(f"  refit={v['name']:<8s}  rmse={rmse_mean:.4f} ± {rmse_std:.4f}  "
              f"fold_time={time_mean:.2f}s")
    return out


def render_md(results: Dict[str, Any], n_trials_v06: int) -> str:
    lines = [
        "# v0.7 leaf-refit per-config ablation (TPE divergence eliminated)",
        "",
        f"For each dataset, the v0.6 winning MoE config from the headline {n_trials_v06}-trial study "
        "is held FIXED across the four refit-mode columns. Only `mixture_refit_*` differs.",
        "5-fold time-series CV; mean ± std across folds.",
        "",
        "## Mean validation RMSE (lower better)",
        "",
        "| Dataset | refit=off | refit=elbo | refit=every_n=10 | refit=always | Δ elbo vs off | Δ every_n vs off | Δ always vs off |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for ds_name, ds in results["datasets"].items():
        v = ds["variants"]
        off = v["off"]["rmse_mean"]

        def pct(x):
            if not (off > 0) or not np.isfinite(x):
                return "n/a"
            return f"{100.0 * (x - off) / off:+.2f}%"

        lines.append(
            f"| `{ds_name}` "
            f"| {v['off']['rmse_mean']:.4f} ± {v['off']['rmse_std']:.4f} "
            f"| {v['elbo']['rmse_mean']:.4f} ± {v['elbo']['rmse_std']:.4f} "
            f"| {v['every_n']['rmse_mean']:.4f} ± {v['every_n']['rmse_std']:.4f} "
            f"| {v['always']['rmse_mean']:.4f} ± {v['always']['rmse_std']:.4f} "
            f"| {pct(v['elbo']['rmse_mean'])} "
            f"| {pct(v['every_n']['rmse_mean'])} "
            f"| {pct(v['always']['rmse_mean'])} |"
        )
    lines.append("")

    lines.append("## Mean training time per fold (seconds)")
    lines.append("")
    lines.append("| Dataset | off | elbo | every_n=10 | always | elbo / off | every_n / off | always / off |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for ds_name, ds in results["datasets"].items():
        v = ds["variants"]
        off_t = max(v["off"]["time_mean"], 1e-9)
        lines.append(
            f"| `{ds_name}` "
            f"| {v['off']['time_mean']:.2f} "
            f"| {v['elbo']['time_mean']:.2f} "
            f"| {v['every_n']['time_mean']:.2f} "
            f"| {v['always']['time_mean']:.2f} "
            f"| {v['elbo']['time_mean']/off_t:.2f}× "
            f"| {v['every_n']['time_mean']/off_t:.2f}× "
            f"| {v['always']['time_mean']/off_t:.2f}× |"
        )
    lines.append("")

    # Summary
    n_total = len(results["datasets"])
    elbo_better = sum(1 for ds in results["datasets"].values()
                       if ds["variants"]["elbo"]["rmse_mean"] < ds["variants"]["off"]["rmse_mean"] - 1e-6)
    elbo_match = sum(1 for ds in results["datasets"].values()
                      if abs(ds["variants"]["elbo"]["rmse_mean"] - ds["variants"]["off"]["rmse_mean"]) <= 1e-6)
    every_n_better = sum(1 for ds in results["datasets"].values()
                          if ds["variants"]["every_n"]["rmse_mean"] < ds["variants"]["off"]["rmse_mean"] - 1e-6)
    always_better = sum(1 for ds in results["datasets"].values()
                         if ds["variants"]["always"]["rmse_mean"] < ds["variants"]["off"]["rmse_mean"] - 1e-6)
    lines += [
        "## Summary",
        "",
        f"- `elbo`: better than off on **{elbo_better}/{n_total}** datasets, "
        f"bit-identical (Δ < 1e-6) on **{elbo_match}/{n_total}** "
        f"(no-fire = no-op as designed)",
        f"- `every_n=10`: better than off on **{every_n_better}/{n_total}**",
        f"- `always`: better than off on **{always_better}/{n_total}**",
    ]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v06-json",
                   default="bench_results/study_500_3way_20260502_200635.json",
                   help="v0.6 study JSON to read best_params from")
    p.add_argument("--rounds", type=int, default=100,
                   help="num_boost_round per CV fold (default 100)")
    p.add_argument("--splits", type=int, default=5,
                   help="number of TimeSeriesSplit folds (default 5)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--datasets", type=str,
                   default="synthetic,fred_gdp,sp500_basic,sp500,vix,hmm")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    print(f"v0.7 leaf-refit per-config ablation")
    print(f"v0.6 best params from: {args.v06_json}")
    print(f"rounds={args.rounds}  splits={args.splits}  seed={args.seed}")

    v06_bests = load_v06_best_params(args.v06_json)
    selected = [s.strip() for s in args.datasets.split(",")]
    unknown = [s for s in selected if s not in DATASET_GENERATORS]
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {unknown}")

    results: Dict[str, Any] = {"timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                                "v06_json": args.v06_json,
                                "num_boost_round": args.rounds,
                                "n_splits": args.splits,
                                "datasets": {}}

    for name in selected:
        if name not in v06_bests:
            print(f"[skip] {name}: no v0.6 best_params found")
            continue
        gen = DATASET_GENERATORS[name]
        gen_out = gen(args.seed)
        # Each generator returns either (X, y, ...) or (X, y); take first 2.
        X = np.asarray(gen_out[0])
        y = np.asarray(gen_out[1])
        results["datasets"][name] = bench_dataset(
            name, X, y, v06_bests[name], args.splits, args.rounds)

    # Outputs
    ts = results["timestamp"]
    out_json = args.out or f"bench_results/bench_v07_per_config_{ts}.json"
    out_md = out_json.replace(".json", ".md")
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)

    n_trials = json.load(open(args.v06_json)).get("config", {}).get("n_trials", "?")
    md = render_md(results, n_trials)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(out_md, "w") as f:
        f.write(md)

    print("\n" + md)
    print(f"\n[wrote] {out_json}")
    print(f"[wrote] {out_md}")


if __name__ == "__main__":
    sys.exit(main() or 0)
