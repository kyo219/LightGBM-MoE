#!/usr/bin/env python
# coding: utf-8
"""
comparative_study.py — naive-lightgbm vs MoE 大規模比較 + ハイパラ重要度分析

`benchmark.py` の data generators (Synthetic / Hamilton / VIX) と独自の CV を再利用しつつ、
Standard GBDT と MoE (token + expert choice 横断) で Optuna を回し、

  - どっちが精度が良いか (best / median RMSE)
  - どっちが速いか (per-trial train time の中央値)
  - **MoE で「明らかに良い」設定は何か** を分析:
      A. fANOVA importance ランキング     (各パラメータが RMSE 分散にどれだけ寄与したか)
      B. categorical の値別平均 RMSE + t検定 (best vs runner-up)
      D. numeric の quartile 別平均 RMSE  (sweet spot 検出)
      E. Optuna slice plot                (各パラメータ値 vs RMSE 散布図)

mixture_init は {random, gmm, tree_hierarchical} に限定 (uniform/quantile/balanced_kmeans 除外)。

Determinism note (--n-jobs default=1):
    Earlier versions defaulted to n_jobs=6 to "speed up" Optuna. In practice,
    n_jobs>1 makes best RMSE non-reproducible across runs / builds even with
    seed fixed: the TPE sampler's recommendations depend on the order parallel
    workers report observations back, and that order is non-deterministic. On
    a regression-checking run (synthetic, 500 trials, 3 seeds × 2 builds, see
    `bench_logs/regression_check_*`) the n_jobs=6 best-RMSE std was ±0.31 —
    enough to make a single-seed n_jobs=6 result swing by ±0.6, large enough
    to falsely flag any code change as a regression. Worse, n_jobs=6 was *not*
    actually faster: with each LightGBM trial saturating all cores via OMP,
    6 concurrent trials oversubscribe the CPU and run ~18% slower per dataset
    than a single sequential pipeline (234s vs 287s on synthetic in that run).

    The default is now n_jobs=1: deterministic best RMSE across runs and
    builds, AND faster wall time. Override with --n-jobs >1 only if you
    explicitly need the (noisy) speed/coverage tradeoff and don't care about
    cross-run comparison.

Usage:
    # 小規模 sanity check
    python examples/comparative_study.py --trials 50 --out bench_results/study_smoke.json

    # 本番 500 trials × 2 variants × 6 datasets (deterministic)
    python examples/comparative_study.py --trials 500 --out bench_results/study_500.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import optuna
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python-package"))
import lightgbm_moe as lgb

sys.path.insert(0, os.path.dirname(__file__))
from benchmark import (  # noqa: E402
    BenchmarkConfig,
    generate_fred_gdp_data,
    generate_hmm_data,
    generate_sp500_basic_data,
    generate_sp500_data,
    generate_synthetic_data,
    generate_vix_data,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# matplotlib for slice plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Search space constants (per user spec)
# =============================================================================
INIT_CHOICES = ["random", "gmm", "tree_hierarchical"]
GATE_CHOICES = ["gbdt", "none", "leaf_reuse"]
ROUTING_CHOICES = ["token_choice", "expert_choice"]
E_STEP_MODES = ["em", "loss_only", "gate_only"]
SMOOTHING_CHOICES = ["none", "ema", "markov"]


# =============================================================================
# CV with per-trial timing
# =============================================================================
def evaluate_cv_timed(X, y, params, n_splits: int, num_boost_round: int):
    """5-fold time-series CV. Returns (rmse_mean, mean_train_seconds_per_fold)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses = []
    times = []
    for tr_idx, va_idx in tscv.split(X):
        Xt, Xv = X[tr_idx], X[va_idx]
        yt, yv = y[tr_idx], y[va_idx]
        train = lgb.Dataset(Xt, label=yt)
        valid = lgb.Dataset(Xv, label=yv, reference=train)
        try:
            t0 = time.perf_counter()
            model = lgb.train(
                params,
                train,
                num_boost_round=num_boost_round,
                valid_sets=[valid],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )
            t1 = time.perf_counter()
            pred = model.predict(Xv)
            rmse = float(np.sqrt(mean_squared_error(yv, pred)))
            rmses.append(rmse)
            times.append(t1 - t0)
        except Exception:
            rmses.append(float("inf"))
            times.append(0.0)
    return float(np.mean(rmses)), float(np.mean(times))


# =============================================================================
# Optuna objectives
# =============================================================================
def make_naive_lightgbm_objective(X, y, cfg: BenchmarkConfig, trial_log: list):
    def objective(trial):
        params = {
            "objective": "regression",
            "boosting": "gbdt",
            "verbose": -1,
            "num_threads": 4,
            "seed": cfg.seed,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
            "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
        }
        rmse, train_s = evaluate_cv_timed(X, y, params, cfg.n_splits, cfg.num_boost_round)
        trial_log.append({"variant": "naive-lightgbm", "rmse": rmse, "train_s": train_s, "params": dict(trial.params)})
        return rmse

    return objective


def make_moe_objective(X, y, cfg: BenchmarkConfig, trial_log: list):
    def objective(trial):
        smoothing = trial.suggest_categorical("mixture_r_smoothing", SMOOTHING_CHOICES)
        routing_mode = trial.suggest_categorical("mixture_routing_mode", ROUTING_CHOICES)
        gate_type = trial.suggest_categorical("mixture_gate_type", GATE_CHOICES)
        num_experts = trial.suggest_int("mixture_num_experts", 2, 4)

        params = {
            "objective": "regression",
            "boosting": "mixture",
            "verbose": -1,
            "num_threads": 4,
            "seed": cfg.seed,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
            "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
            "mixture_init": trial.suggest_categorical("mixture_init", INIT_CHOICES),
            "mixture_num_experts": num_experts,
            "mixture_e_step_alpha": trial.suggest_float("mixture_e_step_alpha", 0.1, 3.0),
            "mixture_e_step_mode": trial.suggest_categorical("mixture_e_step_mode", E_STEP_MODES),
            "mixture_warmup_iters": trial.suggest_int("mixture_warmup_iters", 5, 50),
            "mixture_balance_factor": trial.suggest_int("mixture_balance_factor", 2, 10),
            "mixture_r_smoothing": smoothing,
            "mixture_smoothing_lambda": (
                trial.suggest_float("mixture_smoothing_lambda", 0.0, 0.9) if smoothing != "none" else 0.0
            ),
            "mixture_diversity_lambda": trial.suggest_float("mixture_diversity_lambda", 0.0, 0.5),
            "mixture_routing_mode": routing_mode,
            "mixture_hard_m_step": trial.suggest_categorical("mixture_hard_m_step", [True, False]),
            "mixture_gate_type": gate_type,
        }

        if gate_type in ("gbdt", "leaf_reuse"):
            params["mixture_gate_max_depth"] = trial.suggest_int("mixture_gate_max_depth", 2, 10)
            params["mixture_gate_num_leaves"] = trial.suggest_int("mixture_gate_num_leaves", 4, 64)
            params["mixture_gate_learning_rate"] = trial.suggest_float("mixture_gate_learning_rate", 0.01, 0.5, log=True)
            params["mixture_gate_lambda_l2"] = trial.suggest_float("mixture_gate_lambda_l2", 1e-3, 10.0, log=True)
            params["mixture_gate_iters_per_round"] = trial.suggest_int("mixture_gate_iters_per_round", 1, 3)
        if gate_type == "leaf_reuse":
            params["mixture_gate_retrain_interval"] = trial.suggest_int("mixture_gate_retrain_interval", 5, 30)

        if routing_mode == "expert_choice":
            params["mixture_expert_capacity_factor"] = trial.suggest_float("mixture_expert_capacity_factor", 0.8, 1.5)
            params["mixture_expert_choice_score"] = "combined"
            params["mixture_expert_choice_boost"] = trial.suggest_float("mixture_expert_choice_boost", 5.0, 30.0)
            params["mixture_expert_choice_hard"] = trial.suggest_categorical("mixture_expert_choice_hard", [True, False])

        rmse, train_s = evaluate_cv_timed(X, y, params, cfg.n_splits, cfg.num_boost_round)
        trial_log.append({"variant": "moe", "rmse": rmse, "train_s": train_s, "params": dict(trial.params)})
        return rmse

    return objective


# =============================================================================
# Hyperparameter analysis
# =============================================================================
def fanova_importance(study) -> dict:
    """Optuna param importance (default: MeanDecreaseImpurity)."""
    try:
        imp = optuna.importance.get_param_importances(study)
        return {k: round(float(v), 4) for k, v in imp.items()}
    except Exception as e:
        return {"_error": str(e)[:80]}


def categorical_value_stats(trials: list[dict], param: str) -> dict:
    """For each value of `param`, compute mean RMSE / std / count.
    Then identify the best value and run a t-test against the runner-up.
    Returns {"per_value": {...}, "best": {"value":..., "vs_runner_up_p":...}}.
    """
    finite = [t for t in trials if np.isfinite(t["rmse"]) and param in t["params"]]
    if not finite:
        return {"per_value": {}, "best": None}

    by_val: dict = {}
    for t in finite:
        by_val.setdefault(t["params"][param], []).append(t["rmse"])

    per_value = {}
    for v, rmses in by_val.items():
        per_value[str(v)] = {
            "n": len(rmses),
            "mean": round(float(np.mean(rmses)), 4),
            "std": round(float(np.std(rmses)), 4),
            "min": round(float(np.min(rmses)), 4),
        }

    if len(per_value) < 2:
        return {"per_value": per_value, "best": None}

    sorted_vals = sorted(per_value.items(), key=lambda kv: kv[1]["mean"])
    best_val, runner_up = sorted_vals[0][0], sorted_vals[1][0]
    best_rmses = by_val[type(list(by_val.keys())[0])(best_val) if not isinstance(list(by_val.keys())[0], str) else best_val]
    runner_rmses = by_val[type(list(by_val.keys())[0])(runner_up) if not isinstance(list(by_val.keys())[0], str) else runner_up]

    try:
        t_stat, p_val = stats.ttest_ind(best_rmses, runner_rmses, equal_var=False)
    except Exception:
        t_stat, p_val = float("nan"), float("nan")

    return {
        "per_value": per_value,
        "best": {
            "value": str(best_val),
            "runner_up": str(runner_up),
            "delta_mean": round(per_value[str(runner_up)]["mean"] - per_value[str(best_val)]["mean"], 4),
            "p_value": round(float(p_val), 6) if np.isfinite(p_val) else None,
            "significant": (np.isfinite(p_val) and p_val < 0.01),
        },
    }


def numeric_quartile_stats(trials: list[dict], param: str) -> dict:
    """Sort trials by `param` value into Q1..Q4, return mean RMSE per quartile."""
    finite = [t for t in trials if np.isfinite(t["rmse"]) and param in t["params"]]
    if len(finite) < 8:
        return {}
    vals = np.array([float(t["params"][param]) for t in finite])
    rmses = np.array([t["rmse"] for t in finite])
    qs = np.quantile(vals, [0.25, 0.5, 0.75])

    out = {}
    for label, lo, hi in [("Q1", -np.inf, qs[0]), ("Q2", qs[0], qs[1]), ("Q3", qs[1], qs[2]), ("Q4", qs[2], np.inf)]:
        mask = (vals >= lo) & (vals < hi) if label != "Q4" else (vals >= lo)
        if not mask.any():
            continue
        out[label] = {
            "range": [round(float(lo), 4) if np.isfinite(lo) else None, round(float(hi), 4) if np.isfinite(hi) else None],
            "n": int(mask.sum()),
            "mean_rmse": round(float(rmses[mask].mean()), 4),
        }
    out["best_quartile"] = min(("Q1", "Q2", "Q3", "Q4"), key=lambda q: out.get(q, {"mean_rmse": float("inf")})["mean_rmse"])
    return out


def make_slice_plot(study, out_path: str, title: str):
    """Save an optuna slice plot as PNG."""
    try:
        from optuna.visualization.matplotlib import plot_slice
        ax = plot_slice(study)
        # plot_slice returns a single Axes or array; gather a Figure
        if hasattr(ax, "figure"):
            fig = ax.figure
        elif hasattr(ax, "__iter__"):
            fig = ax[0].figure if len(ax) > 0 else None
        else:
            fig = plt.gcf()
        if fig is not None:
            fig.suptitle(title, fontsize=14)
            fig.set_size_inches(max(12, 2 * len(study.best_params)), 8)
            fig.tight_layout()
            fig.savefig(out_path, dpi=80, bbox_inches="tight")
            plt.close(fig)
        return True
    except Exception as e:
        print(f"    [warn] slice plot for {title} failed: {e}")
        return False


# =============================================================================
# Aggregator
# =============================================================================
def aggregate_variant(name: str, trials: list[dict], study) -> dict:
    finite = [t for t in trials if np.isfinite(t["rmse"])]
    if not finite:
        return {"variant": name, "n_trials": len(trials), "n_finite": 0}

    rmses = np.array([t["rmse"] for t in finite])
    times = np.array([t["train_s"] for t in finite])

    out = {
        "variant": name,
        "n_trials": len(trials),
        "n_finite": len(finite),
        "rmse_best": float(rmses.min()),
        "rmse_p10": float(np.quantile(rmses, 0.10)),
        "rmse_median": float(np.median(rmses)),
        "train_s_median": float(np.median(times)),
        "train_s_mean": float(np.mean(times)),
        "train_s_p90": float(np.quantile(times, 0.90)),
        "best_params": dict(study.best_params),
        "importance": fanova_importance(study),
    }

    cat_params = (
        ["mixture_gate_type", "mixture_routing_mode", "mixture_e_step_mode", "mixture_init",
         "mixture_r_smoothing", "mixture_hard_m_step", "extra_trees"]
        if name == "moe" else ["extra_trees"]
    )
    out["categorical_stats"] = {p: categorical_value_stats(trials, p) for p in cat_params}

    num_params = (
        ["mixture_num_experts", "mixture_e_step_alpha", "mixture_diversity_lambda",
         "mixture_warmup_iters", "mixture_balance_factor", "learning_rate",
         "num_leaves", "max_depth", "min_data_in_leaf"]
        if name == "moe"
        else ["learning_rate", "num_leaves", "max_depth", "min_data_in_leaf",
              "lambda_l1", "lambda_l2", "feature_fraction", "bagging_fraction"]
    )
    out["numeric_quartiles"] = {p: numeric_quartile_stats(trials, p) for p in num_params}

    return out


# =============================================================================
# Markdown report
# =============================================================================
def render_markdown(results: dict, out_path: str, slice_paths: dict):
    lines = []
    lines.append("# Comparative Study Report — naive-lightgbm vs MoE\n")
    cfg = results.get("config", {})
    lines.append(f"- **Trials per (variant × dataset)**: {cfg.get('trials')}\n")
    lines.append(f"- **Datasets**: {cfg.get('datasets')}\n")
    lines.append(f"- **n_splits**: {cfg.get('splits')}, **rounds**: {cfg.get('rounds')}\n")
    lines.append("\n---\n")

    # Headline table
    lines.append("## Headline: which variant wins?\n")
    lines.append("| Dataset | Variant | best RMSE | median RMSE | median train s/fold | wall s |")
    lines.append("|---|---|---|---|---|---|")
    for ds_name, ds in results.items():
        if not isinstance(ds, dict) or "naive-lightgbm" not in ds:
            continue
        for v in ("naive-lightgbm", "moe"):
            r = ds.get(v, {})
            lines.append(
                f"| {ds_name} | {v} | {r.get('rmse_best', float('nan')):.4f} "
                f"| {r.get('rmse_median', float('nan')):.4f} "
                f"| {r.get('train_s_median', 0):.3f} | {r.get('wall_s', 0):.0f} |"
            )
    lines.append("\n")

    # Per-dataset detail
    for ds_name, ds in results.items():
        if not isinstance(ds, dict) or "naive-lightgbm" not in ds:
            continue
        lines.append(f"\n---\n\n## {ds_name}  (X={ds.get('X_shape')})\n")

        for v in ("naive-lightgbm", "moe"):
            r = ds.get(v, {})
            if not r:
                continue
            lines.append(f"\n### {v}\n")
            lines.append(f"- best RMSE: **{r.get('rmse_best', float('nan')):.4f}**, "
                         f"median: {r.get('rmse_median', float('nan')):.4f}, "
                         f"p10: {r.get('rmse_p10', float('nan')):.4f}")
            lines.append(f"- train: median {r.get('train_s_median', 0):.3f}s/fold, "
                         f"mean {r.get('train_s_mean', 0):.3f}s, p90 {r.get('train_s_p90', 0):.3f}s")
            lines.append(f"- finite trials: {r.get('n_finite', 0)} / {r.get('n_trials', 0)}")

            # Importance
            imp = r.get("importance", {})
            if imp and "_error" not in imp:
                lines.append(f"\n#### A. fANOVA importance (top 10)\n")
                lines.append("| param | importance |")
                lines.append("|---|---|")
                for k, val in list(imp.items())[:10]:
                    lines.append(f"| `{k}` | {val:.3f} |")

            # Categorical
            cat = r.get("categorical_stats", {})
            sig_rows = []
            for p, info in cat.items():
                if info.get("best") and info["best"].get("significant"):
                    b = info["best"]
                    sig_rows.append(
                        f"| `{p}` | **{b['value']}** | {info['per_value'][b['value']]['mean']:.4f} "
                        f"(n={info['per_value'][b['value']]['n']}) | {b['runner_up']} | "
                        f"Δ +{b['delta_mean']:.4f} | p={b['p_value']:.2e} |"
                    )
            if sig_rows:
                lines.append(f"\n#### B. Categorical: clearly best values (p<0.01)\n")
                lines.append("| param | best | mean RMSE | runner-up | Δ | p |")
                lines.append("|---|---|---|---|---|---|")
                lines.extend(sig_rows)

            # All categorical detail
            if cat:
                lines.append(f"\n<details><summary>All categorical breakdowns</summary>\n")
                for p, info in cat.items():
                    if not info.get("per_value"):
                        continue
                    lines.append(f"\n**`{p}`**")
                    lines.append("| value | n | mean RMSE | std | min |")
                    lines.append("|---|---|---|---|---|")
                    for val, st in sorted(info["per_value"].items(), key=lambda kv: kv[1]["mean"]):
                        lines.append(f"| {val} | {st['n']} | {st['mean']:.4f} | {st['std']:.4f} | {st['min']:.4f} |")
                lines.append("\n</details>\n")

            # Numeric quartiles
            num = r.get("numeric_quartiles", {})
            if num:
                lines.append(f"\n#### D. Numeric: quartile mean RMSE (sweet spot)\n")
                lines.append("| param | Q1 | Q2 | Q3 | Q4 | best Q (range) |")
                lines.append("|---|---|---|---|---|---|")
                for p, info in num.items():
                    if not info or "best_quartile" not in info:
                        continue
                    best_q = info["best_quartile"]
                    rng = info[best_q].get("range", [None, None])
                    cells = [f"{info[q]['mean_rmse']:.4f}" if q in info else "—" for q in ("Q1", "Q2", "Q3", "Q4")]
                    rng_str = f"[{rng[0]}, {rng[1]}]" if rng[1] is not None else f"[{rng[0]}, ∞)"
                    lines.append(f"| `{p}` | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | **{best_q}** {rng_str} |")

            # E. Slice plot reference
            sp = slice_paths.get(f"{ds_name}/{v}")
            if sp:
                lines.append(f"\n#### E. Slice plot\n")
                lines.append(f"![{ds_name}/{v}]({os.path.basename(sp)})\n")

    # Overall recommendations summary
    lines.append("\n---\n\n## Overall recommendations\n")
    moe_recs = []
    for ds_name, ds in results.items():
        if not isinstance(ds, dict) or "moe" not in ds:
            continue
        cat = ds["moe"].get("categorical_stats", {})
        for p, info in cat.items():
            if info.get("best") and info["best"].get("significant"):
                moe_recs.append((ds_name, p, info["best"]["value"], info["best"]["delta_mean"], info["best"]["p_value"]))
    if moe_recs:
        lines.append("**Categorical settings that are statistically significant winners (p<0.01):**\n")
        lines.append("| dataset | param | best value | Δ vs runner-up | p |")
        lines.append("|---|---|---|---|---|")
        for ds_name, p, val, d, pv in moe_recs:
            lines.append(f"| {ds_name} | `{p}` | **{val}** | +{d:.4f} | {pv:.2e} |")
    else:
        lines.append("(no categorical settings were universally significant — see per-dataset breakdown)\n")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved → {out_path}")


# =============================================================================
# Driver
# =============================================================================
def run_study(dataset_name: str, X, y, n_trials: int, n_jobs: int, cfg: BenchmarkConfig, slice_dir: str) -> tuple[dict, dict]:
    print(f"\n{'=' * 60}")
    print(f"  Dataset: {dataset_name}  X={X.shape}  trials={n_trials}  n_jobs={n_jobs}")
    print(f"{'=' * 60}")

    out = {"dataset": dataset_name, "X_shape": list(X.shape), "y_stats": {"mean": float(y.mean()), "std": float(y.std())}}
    slice_paths: dict = {}

    for variant, make_obj in [
        ("naive-lightgbm", lambda log: make_naive_lightgbm_objective(X, y, cfg, log)),
        ("moe", lambda log: make_moe_objective(X, y, cfg, log)),
    ]:
        print(f"\n  → {variant} ({n_trials} trials)...")
        trial_log: list = []
        sampler = optuna.samplers.TPESampler(seed=cfg.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        t0 = time.perf_counter()
        study.optimize(make_obj(trial_log), n_trials=n_trials, n_jobs=n_jobs)
        wall = time.perf_counter() - t0

        agg = aggregate_variant(variant, trial_log, study)
        agg["wall_s"] = round(wall, 1)
        out[variant] = agg
        out[f"{variant}_trials"] = trial_log

        # Slice plot
        sp_path = os.path.join(slice_dir, f"slice_{dataset_name}_{variant}.png")
        if make_slice_plot(study, sp_path, f"{dataset_name} / {variant}"):
            slice_paths[f"{dataset_name}/{variant}"] = sp_path

        print(f"    best RMSE = {agg.get('rmse_best', float('nan')):.4f}, "
              f"median train = {agg.get('train_s_median', 0):.3f}s/fold, "
              f"wall = {wall:.0f}s")

    return out, slice_paths


DATASET_GENERATORS = {
    "synthetic": lambda seed: generate_synthetic_data(seed=seed),
    "fred_gdp": lambda seed: generate_fred_gdp_data(seed=seed),
    "sp500_basic": lambda seed: generate_sp500_basic_data(seed=seed),  # ~13 features
    "sp500": lambda seed: generate_sp500_data(seed=seed),              # ~28 features (enriched)
    "vix": lambda seed: generate_vix_data(seed=seed),
    "hmm": lambda seed: generate_hmm_data(seed=seed),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=500)
    # n_jobs=1 is deliberately the default — see the module docstring for the
    # determinism / speed rationale. tl;dr Optuna n_jobs>1 makes best-RMSE
    # non-reproducible across runs even with seed fixed (TPE observation
    # ordering depends on worker scheduling), AND oversubscribes the CPU
    # against per-trial OMP threading so it's not actually faster.
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--datasets",
        type=str,
        default="synthetic,fred_gdp,sp500_basic,sp500,vix,hmm",
        help=f"Comma-separated subset of: {','.join(DATASET_GENERATORS.keys())}",
    )
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    cfg = BenchmarkConfig(n_trials=args.trials, seed=args.seed, n_splits=args.splits, num_boost_round=args.rounds)
    selected = [s.strip() for s in args.datasets.split(",")]
    unknown = [s for s in selected if s not in DATASET_GENERATORS]
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {unknown}. Known: {list(DATASET_GENERATORS)}")

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    slice_dir = out_dir

    results = {"config": {"trials": args.trials, "n_jobs": args.n_jobs, "rounds": args.rounds,
                          "splits": args.splits, "seed": args.seed, "datasets": selected}}
    all_slice_paths: dict = {}

    for ds_name in selected:
        X, y, _ = DATASET_GENERATORS[ds_name](cfg.seed)
        ds_out, sp = run_study(ds_name, X, y, args.trials, args.n_jobs, cfg, slice_dir)
        results[ds_name] = ds_out
        all_slice_paths.update(sp)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved JSON → {args.out}")

    md_path = args.out.replace(".json", "_report.md")
    render_markdown(results, md_path, all_slice_paths)


if __name__ == "__main__":
    main()
