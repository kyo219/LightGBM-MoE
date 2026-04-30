"""
Splice the latest single-dataset run output(s) into the combined study_500.json
and re-render the report.

Currently merges:
  - bench_results/study_500_sp500_basic.json -> sp500_basic key
  - bench_results/study_500_sp500_v2.json    -> sp500 key (enhanced features)

Run from repo root:
    python bench_results/_splice_sp500.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "examples"))

from comparative_study import render_markdown  # noqa: E402

BENCH = REPO / "bench_results"
COMBINED = BENCH / "study_500.json"
REPORT = BENCH / "study_500_report.md"

# (side artifact, dataset key it provides)
SIDE_RUNS = [
    (BENCH / "study_500_sp500_basic.json", "sp500_basic"),
    (BENCH / "study_500_sp500_v2.json", "sp500"),
]


def main() -> None:
    combined = json.loads(COMBINED.read_text())

    for side_path, ds_key in SIDE_RUNS:
        if not side_path.exists():
            continue
        side = json.loads(side_path.read_text())
        if ds_key not in side:
            print(f"warning: {side_path.name} missing key {ds_key}, skipping")
            continue
        combined[ds_key] = side[ds_key]
        print(f"Spliced {ds_key} from {side_path.name}")

    cfg = combined.setdefault("config", {})
    cfg["datasets"] = [k for k in combined if isinstance(combined[k], dict) and "naive-lightgbm" in combined[k]]
    cfg["sp500_features_note"] = (
        "sp500_basic uses the original ~13-column feature set; sp500 uses the enriched "
        "~28-column set (momentum / RSI / realized vol / skew / kurt / Bollinger / "
        "drawdown). Side-by-side to demonstrate that MoE's lift scales with how "
        "observable the regime is from features."
    )

    COMBINED.write_text(json.dumps(combined, indent=2, default=str))

    slice_paths = {}
    for ds in combined:
        if not isinstance(combined[ds], dict) or "naive-lightgbm" not in combined[ds]:
            continue
        for v in ("naive-lightgbm", "moe"):
            png = BENCH / f"slice_{ds}_{v}.png"
            if png.exists():
                slice_paths[f"{ds}/{v}"] = str(png)

    render_markdown(combined, str(REPORT), slice_paths)
    print(f"\nWrote {REPORT.name}")

    for side_path, _ in SIDE_RUNS:
        for stale in (side_path,
                      side_path.with_suffix(".log"),
                      side_path.with_name(side_path.stem + "_report.md")):
            if stale.exists():
                os.remove(stale)
                print(f"Removed {stale.name}")


if __name__ == "__main__":
    main()
