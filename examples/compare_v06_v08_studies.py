"""Side-by-side comparison: v0.6 study (baseline) vs v0.8 study (expanded
search space with v0.7 refit + v0.8 partition re-grow as Optuna variables).

Loads two ``comparative_study.py`` JSON outputs and emits a markdown report:

  1. **RMSE table**: per dataset, v0.6 best vs v0.8 best, Δ%.
  2. **Config diff**: per dataset, which v0.8-specific knobs the winning
     config used (refit_leaves, regrow_oldest_trees, etc.).
  3. **Search-space utilization**: across all v0.8 trials, distribution of
     refit / regrow choices — how often did Optuna pick to use them.
  4. **Summary verdict**: did v0.8 win, tie, or lose on each dataset?
     Strict win = v0.8_best < v0.6_best by more than 1 std.

Usage::

    python examples/compare_v06_v08_studies.py \
        bench_results/study_500_3way_20260502_200635.json \
        bench_results/study_v08_500_<ts>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict


V08_FEATURE_KEYS = [
    "mixture_refit_leaves",
    "mixture_refit_trigger",
    "mixture_refit_decay_rate",
    "mixture_refit_every_n",
    "mixture_elbo_drop_threshold",
    "mixture_elbo_plateau_threshold",
    "mixture_elbo_window",
    "mixture_regrow_oldest_trees",
    "mixture_regrow_per_fire",
    "mixture_regrow_mode",
]

CORE_KNOB_KEYS = [
    "mixture_init",
    "mixture_num_experts",
    "mixture_gate_type",
    "mixture_routing_mode",
    "mixture_e_step_mode",
    "mixture_diversity_lambda",
    "mixture_hard_m_step",
]


def get_moe(study: Dict[str, Any], ds: str) -> Dict[str, Any]:
    """Extract the moe variant block (handles both 'moe' and 'moe-refit-*' keys)."""
    if ds not in study:
        return {}
    for k, v in study[ds].items():
        if k == "moe" or k.startswith("moe-refit-"):
            return v
    return {}


def fmt_rmse(r, s):
    if r is None or r == "":
        return "—"
    if s is None:
        return f"{r:.4f}"
    return f"{r:.4f} ± {s:.4f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("v06_json", help="v0.6 study JSON (baseline)")
    p.add_argument("v08_json", help="v0.8 study JSON (with v0.8 params searchable)")
    p.add_argument("--out", type=str, default=None,
                   help="output markdown path (default: stdout)")
    args = p.parse_args()

    v06 = json.load(open(args.v06_json))
    v08 = json.load(open(args.v08_json))

    # Both JSONs use dataset names as top-level keys (with a 'config' key too).
    common_datasets = sorted(set(v06.keys()) & set(v08.keys()) - {"config"})

    lines = []
    lines.append("# v0.6 vs v0.8 study comparison\n")
    lines.append(f"- v0.6 source: `{args.v06_json}`")
    lines.append(f"- v0.8 source: `{args.v08_json}`")
    lines.append(f"- Datasets compared: {', '.join(common_datasets)}\n")

    # === Section 1: RMSE comparison ===
    lines.append("## 1. RMSE: v0.6 best vs v0.8 best\n")
    lines.append("| Dataset | v0.6 MoE best | v0.8 MoE best | Δ (abs) | Δ% | verdict |")
    lines.append("|---|---|---|---|---|---|")
    summary = {"v08_strict_win": 0, "v08_tie": 0, "v08_loss": 0}
    for ds in common_datasets:
        a = get_moe(v06, ds)
        b = get_moe(v08, ds)
        ra, rb = a.get("rmse_best", a.get("best_rmse")), b.get("rmse_best", b.get("best_rmse"))
        sa = a.get("rmse_std", a.get("std"))
        sb = b.get("rmse_std", b.get("std"))
        if ra is None or rb is None:
            verdict = "—"
            delta_abs = "—"
            delta_pct = "—"
        else:
            d = rb - ra
            dp = 100.0 * d / ra if ra != 0 else 0
            std = sb if sb is not None else 0
            if rb < ra - std:
                verdict = "v0.8 **strict win**"
                summary["v08_strict_win"] += 1
            elif rb > ra + std:
                verdict = "v0.8 **loss**"
                summary["v08_loss"] += 1
            else:
                verdict = "tie"
                summary["v08_tie"] += 1
            delta_abs = f"{d:+.4f}"
            delta_pct = f"{dp:+.2f}%"
        lines.append(
            f"| `{ds}` | {fmt_rmse(ra, sa)} | {fmt_rmse(rb, sb)} | {delta_abs} | {delta_pct} | {verdict} |"
        )
    n = len(common_datasets)
    lines.append("")
    lines.append(f"**Summary**: {summary['v08_strict_win']}/{n} strict win, "
                 f"{summary['v08_tie']}/{n} tie, {summary['v08_loss']}/{n} loss.")
    lines.append("")
    lines.append("(*Strict win* = v0.8 best is more than 1 std (per-fold) below v0.6 best.)\n")

    # === Section 2: v0.8-specific params in winning configs ===
    lines.append("## 2. Did the v0.8 winning config use the new knobs?\n")
    lines.append("Per dataset, value of each v0.8-specific param in the v0.8 `best_params`. "
                 "`—` = the param was not in best_params (Optuna didn't sample it for the winning trial, "
                 "e.g. because `mixture_refit_leaves=False` short-circuited the conditional sub-tree).\n")
    lines.append("| Dataset | refit_leaves | refit_trigger | regrow | regrow_per_fire | regrow_mode | init |")
    lines.append("|---|---|---|---|---|---|---|")
    for ds in common_datasets:
        b = get_moe(v08, ds).get("best_params", {})
        row = [
            f"`{ds}`",
            str(b.get("mixture_refit_leaves", "—")),
            str(b.get("mixture_refit_trigger", "—")),
            str(b.get("mixture_regrow_oldest_trees", "—")),
            str(b.get("mixture_regrow_per_fire", "—")),
            str(b.get("mixture_regrow_mode", "—")),
            str(b.get("mixture_init", "—")),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # === Section 3: search-space utilization (categorical_value_stats) ===
    lines.append("## 3. Search-space utilization across v0.8 trials\n")
    lines.append("From the per-categorical stats already computed by `comparative_study.py`. "
                 "Shows how often each value appeared in trials and the mean RMSE conditional on that value.\n")
    for ds in common_datasets:
        b = get_moe(v08, ds)
        stats = b.get("categorical_stats", {})
        if not stats:
            continue
        lines.append(f"### `{ds}`\n")
        for param in ("mixture_refit_leaves", "mixture_regrow_oldest_trees",
                      "mixture_refit_trigger", "mixture_regrow_mode", "mixture_init"):
            if param not in stats:
                continue
            per = stats[param].get("per_value", {})
            if not per:
                continue
            lines.append(f"- `{param}`: ")
            for val, s in per.items():
                lines.append(f"  - `{val}` (n={s.get('n')}): "
                             f"mean RMSE = {s.get('mean')}, std = {s.get('std')}, min = {s.get('min')}")
        lines.append("")

    # === Section 4: fanova importance ===
    lines.append("## 4. fANOVA importance of v0.8 params\n")
    lines.append("Higher value = the param explains more of the RMSE variance across trials.\n")
    for ds in common_datasets:
        b = get_moe(v08, ds)
        imp = b.get("fanova_importance", {})
        if not imp:
            continue
        v08_imp = {k: v for k, v in imp.items() if k in V08_FEATURE_KEYS}
        if not v08_imp:
            lines.append(f"- `{ds}`: no v0.8 params in top fANOVA")
            continue
        ranked = sorted(v08_imp.items(), key=lambda kv: -kv[1])
        lines.append(f"- `{ds}`: " + ", ".join(f"{k}={v:.3f}" for k, v in ranked))
    lines.append("")

    out = "\n".join(lines)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out)
        print(f"[wrote] {args.out}")
    else:
        print(out)


if __name__ == "__main__":
    sys.exit(main() or 0)
