#!/usr/bin/env python3
"""check_features.py — Data quality and distribution analysis for JSONL feature datasets.

Usage:
    python tools/check_features.py \\
        --input logs/ml/offline_mwd.jsonl logs/ml/offline_weather_time.jsonl \\
        --target-min-per-class 300 \\
        --fail-on-zero-variance

Exit codes:
    0 — All checks pass
    1 — Check failure (zero variance or below target min)
    2 — Usage error
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smartbinocular.feature_schema import FEATURE_SET_CORE, FeatureRecord

# Features that are EXPECTED to be zero-variance in offline still-image datasets:
#   hour_of_day_sin/cos — all offline images use ts=0.0 → encode_hour returns (0.0, 1.0)
#   prev_env_class      — always 0 for still images (no prior frame state)
# These are NOT computation bugs; they will vary at runtime and during RPi field sessions.
_EXPECTED_OFFLINE_CONSTANT_FEATURES: frozenset = frozenset({
    "hour_of_day_sin",
    "hour_of_day_cos",
    "prev_env_class",
})


def load_records(paths: List[Path]) -> List[dict]:
    records = []
    for p in paths:
        if not p.exists():
            print(f"WARN: File not found: {p}", file=sys.stderr)
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distribution analysis and quality checks for feature JSONL files."
    )
    parser.add_argument("--input", "-i", nargs="+", required=True, type=Path)
    parser.add_argument(
        "--target-min-per-class", type=int, default=0,
        help="Warn/fail if any class has fewer than this many records"
    )
    parser.add_argument(
        "--fail-on-zero-variance", action="store_true",
        help="Exit 1 if any CORE feature has near-zero variance across all records"
    )
    parser.add_argument(
        "--plots", type=Path, default=None,
        help="Directory to save histogram plots (requires matplotlib)"
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Expand globs
    import glob as _glob
    paths: List[Path] = []
    for pat in args.input:
        expanded = _glob.glob(str(pat))
        paths.extend(Path(e) for e in expanded) if expanded else paths.append(pat)

    records = load_records(paths)
    if not records:
        print("ERROR: No records loaded.", file=sys.stderr)
        sys.exit(2)

    print(f"\n{'='*60}")
    print(f"Total records: {len(records)}")

    # ── Summary by source ─────────────────────────────────────────────────────
    source_counts: Dict[str, int] = defaultdict(int)
    nir_counts: Dict[str, int] = defaultdict(int)
    thermal_counts: Dict[str, int] = defaultdict(int)
    label_counts: Dict[str, int] = defaultdict(int)
    label_source_counts: Dict[str, int] = defaultdict(int)

    for r in records:
        source_counts[r.get("source", "unknown")] += 1
        nir_counts[r.get("nir_channel", "unknown")] += 1
        thermal_counts[r.get("thermal_channel", "unknown")] += 1
        lbl = r.get("label") or r.get("weak_label") or "unlabeled"
        label_counts[lbl] += 1
        ls = r.get("label_source") or "none"
        label_source_counts[ls] += 1

    print("\nRecords by source:")
    for k, v in sorted(source_counts.items()):
        print(f"  {k:35s}: {v:6d}")

    print("\nRecords by nir_channel:")
    for k, v in sorted(nir_counts.items()):
        print(f"  {k:10s}: {v:6d}")

    print("\nRecords by thermal_channel:")
    for k, v in sorted(thermal_counts.items()):
        print(f"  {k:10s}: {v:6d}")

    print("\nRecords by label_source:")
    for k, v in sorted(label_source_counts.items()):
        pct = 100.0 * v / len(records)
        print(f"  {k:20s}: {v:6d}  ({pct:.1f}%)")

    print("\nLabel distribution:")
    has_label = sum(1 for r in records if r.get("label"))
    has_weak = sum(1 for r in records if r.get("weak_label") and not r.get("label"))
    unlabeled = sum(1 for r in records if not r.get("label") and not r.get("weak_label"))
    print(f"  Ground truth (label)    : {has_label:6d}  ({100*has_label/len(records):.1f}%)")
    print(f"  Weak label only         : {has_weak:6d}  ({100*has_weak/len(records):.1f}%)")
    print(f"  Unlabeled               : {unlabeled:6d}  ({100*unlabeled/len(records):.1f}%)")
    print()
    for k, v in sorted(label_counts.items()):
        bar = "█" * min(40, v // max(1, len(records) // 40))
        print(f"  {k:20s}: {v:6d}  {bar}")

    # ── CORE feature statistics ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("CORE feature statistics:")
    feat_arrays: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        for feat in FEATURE_SET_CORE:
            val = r.get(feat)
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                feat_arrays[feat].append(float(val))

    zero_var_features: List[str] = []
    print(f"  {'feature':<28} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}  {'coverage':>8}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*8}  {'-'*8}")
    for feat in FEATURE_SET_CORE:
        vals = feat_arrays[feat]
        if not vals:
            print(f"  {feat:<28} {'N/A':>8}")
            continue
        arr = np.array(vals)
        coverage = 100.0 * len(vals) / len(records)
        print(f"  {feat:<28} {arr.mean():8.3f} {arr.std():8.3f} {arr.min():8.3f} {arr.max():8.3f}  {coverage:7.1f}%")
        if arr.std() < 1e-6:
            zero_var_features.append(feat)

    # ── Class balance check ───────────────────────────────────────────────────
    fail = False
    if args.target_min_per_class > 0:
        print(f"\n{'='*60}")
        print(f"Class balance check (target min per class: {args.target_min_per_class}):")
        for lbl, cnt in sorted(label_counts.items()):
            if lbl == "unlabeled":
                continue
            status = "OK" if cnt >= args.target_min_per_class else "LOW"
            print(f"  {lbl:20s}: {cnt:6d}  [{status}]")
            if status == "LOW":
                fail = True

    # ── Zero-variance check ───────────────────────────────────────────────────
    if zero_var_features:
        unexpected = [f for f in zero_var_features if f not in _EXPECTED_OFFLINE_CONSTANT_FEATURES]
        expected_const = [f for f in zero_var_features if f in _EXPECTED_OFFLINE_CONSTANT_FEATURES]
        print(f"\n{'='*60}")
        if expected_const:
            print(f"Zero-variance (expected offline constants — OK): {expected_const}")
            print("  These features vary at runtime (ts=0 → hour=0; prev_env_class=0 for stills).")
        if unexpected:
            print(f"ZERO VARIANCE features (UNEXPECTED — FAIL): {unexpected}")
            if args.fail_on_zero_variance:
                fail = True
        elif not unexpected and args.fail_on_zero_variance:
            pass  # Only expected constants → not a failure

    # ── Plots ─────────────────────────────────────────────────────────────────
    if args.plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            args.plots.mkdir(parents=True, exist_ok=True)
            for feat in FEATURE_SET_CORE:
                vals = feat_arrays[feat]
                if len(vals) < 2:
                    continue
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(vals, bins=50, edgecolor="black", alpha=0.7)
                ax.set_title(f"{feat} distribution (n={len(vals)})")
                ax.set_xlabel(feat)
                ax.set_ylabel("count")
                fig.tight_layout()
                fig.savefig(args.plots / f"hist_{feat}.png", dpi=90)
                plt.close(fig)
            print(f"\nPlots saved to: {args.plots}")
        except ImportError:
            print("WARN: matplotlib not installed; skipping plots.", file=sys.stderr)

    print(f"\n{'='*60}")
    print(f"Result: {'FAIL' if fail else 'PASS'}")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
