"""v0.8 acceptance bench — quantify the value of B (elbo trigger fix) and
A (partition re-grow) on top of v0.7 leaf refit, holding the v0.6 winning
configs constant.

Same protocol as ``examples/bench_v07_per_config.py`` (same 6 datasets, same
v0.6 winning configs, same 5-fold time-series CV, same num_boost_round) so
the v0.8 numbers are directly comparable to ``bench_results/v0_7_acceptance_FINAL.md``.

Variants compared (all stacked on top of the same v0.6 winning config):

  off                : ``mixture_refit_leaves=false`` — sanity, must match
                       v0.7 baseline bit-identically
  v07_elbo           : v0.7 trigger='elbo' with thresholds drop=0.05, plat=0
                       (recovers v0.7 behavior; expect 0/6 fires per the
                       v0.7 acceptance bench)
  v08_elbo           : v0.8 trigger='elbo' with default thresholds
                       (drop=0.01, plateau=0.001, window=10) — pure B
                       contribution, regrow OFF
  v08_elbo_regrow    : full v0.8 default — v0.8 elbo trigger + regrow=on,
                       mode='replace', per_fire=1
  v08_always_regrow  : control — refit trigger='always' + regrow=on,
                       mode='replace', per_fire=1 (upper-bound view of
                       what regrow can do when fired every iter)
  v08_delete_ablation: ablation — refit trigger='every_n'=20 + regrow=on,
                       mode='delete', per_fire=1 (capacity-cost isolation)

Per dataset reports 5-fold mean RMSE, std, fold time. Markdown table side-
by-side with the v0.7 baseline numbers (loaded from
``bench_results/v0_7_acceptance_FINAL.md`` if available, or from a v0.7
JSON if a path is supplied).

Usage::

    PYTHONPATH=python-package python3 examples/bench_v08_acceptance.py \
        --datasets synthetic,hmm                 # offline-only subset
    PYTHONPATH=python-package python3 examples/bench_v08_acceptance.py
        # full 6-dataset run

Output: ``bench_results/bench_v08_acceptance_<ts>.{json,md}``.
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

import lightgbm_moe as lgb

# Reuse the same dataset generators as the v0.7 bench
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
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


VARIANTS: List[Dict[str, Any]] = [
    {
        "name": "off",
        "params": {},
    },
    {
        "name": "v07_elbo",
        "params": {
            "mixture_refit_leaves": True,
            "mixture_refit_trigger": "elbo",
            "mixture_refit_decay_rate": 0.0,
            # v0.7 thresholds: drop=0.05, plateau disabled
            "mixture_elbo_drop_threshold": 0.05,
            "mixture_elbo_plateau_threshold": 0.0,
        },
    },
    {
        "name": "v08_elbo",
        "params": {
            "mixture_refit_leaves": True,
            "mixture_refit_trigger": "elbo",
            "mixture_refit_decay_rate": 0.0,
            # v0.8 defaults
            "mixture_elbo_drop_threshold": 0.01,
            "mixture_elbo_plateau_threshold": 0.001,
            "mixture_elbo_window": 10,
            "mixture_elbo_min_iter_for_plateau": 20,
        },
    },
    {
        "name": "v08_elbo_regrow",
        "params": {
            "mixture_refit_leaves": True,
            "mixture_refit_trigger": "elbo",
            "mixture_refit_decay_rate": 0.0,
            "mixture_elbo_drop_threshold": 0.01,
            "mixture_elbo_plateau_threshold": 0.001,
            "mixture_elbo_window": 10,
            "mixture_elbo_min_iter_for_plateau": 20,
            "mixture_regrow_oldest_trees": True,
            "mixture_regrow_per_fire": 1,
            "mixture_regrow_min_remaining": 5,
            "mixture_regrow_mode": "replace",
        },
    },
    {
        "name": "v08_always_regrow",
        "params": {
            "mixture_refit_leaves": True,
            "mixture_refit_trigger": "always",
            "mixture_refit_decay_rate": 0.0,
            "mixture_regrow_oldest_trees": True,
            "mixture_regrow_per_fire": 1,
            "mixture_regrow_min_remaining": 5,
            "mixture_regrow_mode": "replace",
        },
    },
    {
        "name": "v08_delete_ablation",
        "params": {
            "mixture_refit_leaves": True,
            "mixture_refit_trigger": "every_n",
            "mixture_refit_every_n": 20,
            "mixture_refit_decay_rate": 0.0,
            "mixture_regrow_oldest_trees": True,
            "mixture_regrow_per_fire": 1,
            "mixture_regrow_min_remaining": 5,
            "mixture_regrow_mode": "delete",
        },
    },
]


def load_v06_best_params(json_path: str) -> Dict[str, Dict[str, Any]]:
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
    print(f"\n=== {name} ===  N={len(y)}  F={X.shape[1]}", flush=True)
    print(f"  v0.6 best config: K={v06_best.get('mixture_num_experts')}, "
          f"init={v06_best.get('mixture_init')}, "
          f"gate={v06_best.get('mixture_gate_type')}, "
          f"div={v06_best.get('mixture_diversity_lambda', 0):.3f}", flush=True)

    out: Dict[str, Any] = {"name": name, "v06_best": dict(v06_best),
                           "variants": {}}
    for v in VARIANTS:
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
            "n_folds": len(rmses),
        }
        print(f"  {v['name']:<22s} rmse={rmse_mean:.4f} ± {rmse_std:.4f}  "
              f"fold_time={time_mean:.2f}s", flush=True)
    return out


def render_md(results: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"# v0.8 acceptance bench")
    lines.append("")
    lines.append(f"- Timestamp: `{results['timestamp']}`")
    lines.append(f"- v0.6 best params source: `{results['v06_json']}`")
    lines.append(f"- num_boost_round: {results['num_boost_round']}, "
                 f"n_splits: {results['n_splits']}")
    lines.append("")
    lines.append("## Per-config: v0.7 vs v0.8 variants on the same v0.6 winning configs")
    lines.append("")
    lines.append("RMSE table (lower is better). Δ% computed against `off` baseline within each row.")
    lines.append("")
    lines.append("| Dataset | off | v07_elbo | v08_elbo | v08_elbo_regrow | v08_always_regrow | v08_delete_ablation |")
    lines.append("|---|---|---|---|---|---|---|")
    for ds_name, ds in results["datasets"].items():
        row = [ds_name]
        off = ds["variants"].get("off", {}).get("rmse_mean", float("nan"))
        for vname in ["off", "v07_elbo", "v08_elbo", "v08_elbo_regrow",
                       "v08_always_regrow", "v08_delete_ablation"]:
            v = ds["variants"].get(vname, {})
            r = v.get("rmse_mean", float("nan"))
            s = v.get("rmse_std", float("nan"))
            if vname == "off":
                row.append(f"{r:.4f} ± {s:.4f}")
            else:
                if np.isfinite(r) and np.isfinite(off) and off != 0:
                    pct = 100.0 * (r - off) / off
                    row.append(f"{r:.4f} ({pct:+.1f}%)")
                else:
                    row.append(f"{r:.4f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Per-config: fold time (seconds, mean)")
    lines.append("")
    lines.append("| Dataset | off | v07_elbo | v08_elbo | v08_elbo_regrow | v08_always_regrow | v08_delete_ablation |")
    lines.append("|---|---|---|---|---|---|---|")
    for ds_name, ds in results["datasets"].items():
        row = [ds_name]
        for vname in ["off", "v07_elbo", "v08_elbo", "v08_elbo_regrow",
                       "v08_always_regrow", "v08_delete_ablation"]:
            v = ds["variants"].get(vname, {})
            t = v.get("time_mean", float("nan"))
            row.append(f"{t:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    n_total = len(results["datasets"])
    def count_better(name):
        return sum(
            1 for ds in results["datasets"].values()
            if ds["variants"].get(name, {}).get("rmse_mean", float("inf"))
               < ds["variants"].get("off", {}).get("rmse_mean", float("inf")) - 1e-6
        )
    def count_match(name):
        return sum(
            1 for ds in results["datasets"].values()
            if abs(ds["variants"].get(name, {}).get("rmse_mean", float("inf"))
                   - ds["variants"].get("off", {}).get("rmse_mean", float("inf"))) < 1e-6
        )
    lines.append(f"Across {n_total} datasets:")
    for vname in ["v07_elbo", "v08_elbo", "v08_elbo_regrow",
                   "v08_always_regrow", "v08_delete_ablation"]:
        b = count_better(vname)
        m = count_match(vname)
        w = n_total - b - m
        lines.append(f"- `{vname}`: {b} better, {m} match (no fire), {w} worse")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v06-json",
                   default="bench_results/study_500_3way_20260502_200635.json")
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--datasets", type=str,
                   default="synthetic,fred_gdp,sp500_basic,sp500,vix,hmm")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    print(f"v0.8 acceptance bench")
    print(f"v0.6 best params from: {args.v06_json}")
    print(f"rounds={args.rounds}  splits={args.splits}  seed={args.seed}")

    v06_bests = load_v06_best_params(args.v06_json)
    selected = [s.strip() for s in args.datasets.split(",")]
    unknown = [s for s in selected if s not in DATASET_GENERATORS]
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {unknown}")

    results: Dict[str, Any] = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "v06_json": args.v06_json,
        "num_boost_round": args.rounds,
        "n_splits": args.splits,
        "datasets": {},
    }

    for name in selected:
        if name not in v06_bests:
            print(f"[skip] {name}: no v0.6 best_params found")
            continue
        gen = DATASET_GENERATORS[name]
        gen_out = gen(args.seed)
        X = np.asarray(gen_out[0])
        y = np.asarray(gen_out[1])
        cfg = dict(v06_bests[name])
        results["datasets"][name] = bench_dataset(
            name, X, y, cfg, args.splits, args.rounds)

    ts = results["timestamp"]
    out_json = args.out or f"bench_results/bench_v08_acceptance_{ts}.json"
    out_md = out_json.replace(".json", ".md")
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)

    md = render_md(results)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(out_md, "w") as f:
        f.write(md)

    print("\n" + md)
    print(f"\n[wrote] {out_json}")
    print(f"[wrote] {out_md}")


if __name__ == "__main__":
    sys.exit(main() or 0)
