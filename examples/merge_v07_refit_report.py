"""Merge v0.7 leaf-refit ablation runs into a single comparison report.

Reads three JSONs produced by ``comparative_study.py``:
  - baseline (mode=off): contains naive-lightgbm, naive-ensemble, moe
  - refit elbo (mode=elbo, --variants moe): contains moe-refit-elbo
  - refit every_n (mode=every_n, --variants moe): contains moe-refit-every_n

Emits a side-by-side per-dataset table comparing best RMSE, median per-fold
training time, and the wall time per (dataset × variant) combination — the
headline numbers for the v0.7 acceptance ablation in issue #37.

Usage::

    python examples/merge_v07_refit_report.py \
        --baseline bench_results/study_v07_baseline.json \
        --refit-elbo bench_results/study_v07_refit_elbo.json \
        --refit-every-n bench_results/study_v07_refit_every_n.json \
        --out bench_results/study_v07_refit_report.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def get_best_rmse(study: Dict[str, Any], variant: str) -> float:
    v = study.get(variant, {})
    return float(v.get("rmse_best", float("nan")))


def get_train_s_median(study: Dict[str, Any], variant: str) -> float:
    v = study.get(variant, {})
    return float(v.get("train_s_median", float("nan")))


def get_wall_s(study: Dict[str, Any], variant: str) -> float:
    v = study.get(variant, {})
    return float(v.get("wall_s", float("nan")))


def fmt_pct(num: float, denom: float) -> str:
    if not (denom > 0) or not (num == num):  # NaN check
        return "n/a"
    return f"{100.0 * (num - denom) / denom:+.2f}%"


def per_dataset_row(name: str, baseline_ds, elbo_ds, every_n_ds) -> Dict[str, Any]:
    out = {"dataset": name}

    # Baseline columns: from `baseline` JSON
    out["naive_rmse"] = get_best_rmse(baseline_ds, "naive-lightgbm")
    out["ensemble_rmse"] = get_best_rmse(baseline_ds, "naive-ensemble")
    out["moe_rmse"] = get_best_rmse(baseline_ds, "moe")

    # Refit ablation columns
    out["moe_elbo_rmse"] = get_best_rmse(elbo_ds, "moe-refit-elbo")
    out["moe_every_n_rmse"] = get_best_rmse(every_n_ds, "moe-refit-every_n")

    out["moe_wall"] = get_wall_s(baseline_ds, "moe")
    out["moe_elbo_wall"] = get_wall_s(elbo_ds, "moe-refit-elbo")
    out["moe_every_n_wall"] = get_wall_s(every_n_ds, "moe-refit-every_n")

    out["moe_fold_s"] = get_train_s_median(baseline_ds, "moe")
    out["moe_elbo_fold_s"] = get_train_s_median(elbo_ds, "moe-refit-elbo")
    out["moe_every_n_fold_s"] = get_train_s_median(every_n_ds, "moe-refit-every_n")

    return out


def render_markdown(rows: list[Dict[str, Any]], n_trials: int) -> str:
    lines = []
    lines.append(f"# v0.7 leaf-refit ablation — 500-trial / 6-dataset study\n")
    lines.append(f"**{n_trials} Optuna trials per (variant × dataset), 5-fold time-series CV.**")
    lines.append("Refit ablation runs the same MoE search space with "
                 "`mixture_refit_leaves=True` fixed; only the trigger differs:\n")
    lines.append("- `moe` (baseline): v0.6 behavior, refit off\n"
                 "- `moe-refit-elbo`: refit fires only when the per-10-iter "
                 "ELBO log shows >5% drop\n"
                 "- `moe-refit-every_n`: refit fires every 10 post-warmup iters\n")

    lines.append("## Best RMSE per dataset\n")
    lines.append("| Dataset | naive | ensemble | **moe (off)** | moe-refit-elbo | "
                 "moe-refit-every_n | Δ elbo vs off | Δ every_n vs off |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| `{r['dataset']}` "
            f"| {r['naive_rmse']:.4f} "
            f"| {r['ensemble_rmse']:.4f} "
            f"| **{r['moe_rmse']:.4f}** "
            f"| {r['moe_elbo_rmse']:.4f} "
            f"| {r['moe_every_n_rmse']:.4f} "
            f"| {fmt_pct(r['moe_elbo_rmse'], r['moe_rmse'])} "
            f"| {fmt_pct(r['moe_every_n_rmse'], r['moe_rmse'])} |"
        )
    lines.append("")

    lines.append("## Cost (median per-fold training, seconds)\n")
    lines.append("| Dataset | moe (off) | moe-refit-elbo | moe-refit-every_n | "
                 "elbo / off | every_n / off |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        denom = max(r["moe_fold_s"], 1e-9)
        lines.append(
            f"| `{r['dataset']}` "
            f"| {r['moe_fold_s']:.3f} "
            f"| {r['moe_elbo_fold_s']:.3f} "
            f"| {r['moe_every_n_fold_s']:.3f} "
            f"| {r['moe_elbo_fold_s']/denom:.2f}× "
            f"| {r['moe_every_n_fold_s']/denom:.2f}× |"
        )
    lines.append("")

    lines.append("## Wall-clock budget per variant (seconds, all 500 trials)\n")
    lines.append("| Dataset | moe (off) | moe-refit-elbo | moe-refit-every_n |")
    lines.append("|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| `{r['dataset']}` "
            f"| {r['moe_wall']:.0f} "
            f"| {r['moe_elbo_wall']:.0f} "
            f"| {r['moe_every_n_wall']:.0f} |"
        )

    # Acceptance summary
    lines.append("\n## Acceptance summary (issue #37)\n")
    n_elbo_better = sum(1 for r in rows if r["moe_elbo_rmse"] < r["moe_rmse"])
    n_elbo_within_1pct = sum(1 for r in rows
                              if abs(r["moe_elbo_rmse"] - r["moe_rmse"])
                                 <= 0.01 * r["moe_rmse"])
    n_every_n_better = sum(1 for r in rows
                            if r["moe_every_n_rmse"] < r["moe_rmse"])
    lines.append(f"- elbo trigger: better than off on **{n_elbo_better}/{len(rows)}** datasets, "
                 f"within ±1% on **{n_elbo_within_1pct}/{len(rows)}**\n")
    lines.append(f"- every_n trigger: better than off on **{n_every_n_better}/{len(rows)}** datasets")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True,
                   help="JSON from `--moe-refit-mode off` run (full naive/ensemble/moe)")
    p.add_argument("--refit-elbo", required=True,
                   help="JSON from `--moe-refit-mode elbo --variants moe` run")
    p.add_argument("--refit-every-n", required=True,
                   help="JSON from `--moe-refit-mode every_n --variants moe` run")
    p.add_argument("--out", required=True, help="output markdown path")
    args = p.parse_args()

    base = load_json(args.baseline)
    elbo = load_json(args.refit_elbo)
    every_n = load_json(args.refit_every_n)

    n_trials = base.get("config", {}).get("n_trials", "?")

    # Datasets are top-level keys, sibling to "config". Filter to dicts that
    # carry the variant aggregates (presence of any variant key suffices).
    def is_dataset_row(v):
        return isinstance(v, dict) and any(
            k in v for k in ("naive-lightgbm", "naive-ensemble", "moe",
                              "moe-refit-elbo", "moe-refit-every_n",
                              "moe-refit-always"))
    base_ds = {k: v for k, v in base.items() if is_dataset_row(v)}
    elbo_ds = {k: v for k, v in elbo.items() if is_dataset_row(v)}
    every_n_ds = {k: v for k, v in every_n.items() if is_dataset_row(v)}
    common = sorted(set(base_ds) & set(elbo_ds) & set(every_n_ds))

    rows = []
    for ds in common:
        rows.append(per_dataset_row(ds, base_ds[ds], elbo_ds[ds], every_n_ds[ds]))

    md = render_markdown(rows, n_trials)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(md)
    print(md)
    print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
