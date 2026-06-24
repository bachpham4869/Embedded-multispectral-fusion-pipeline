#!/usr/bin/env python3
"""mix_datasets.py — Merge, filter, and balance JSONL feature datasets for training.

Usage:
    # Optical-only training dataset (primary: weather + field collection)
    python tools/mix_datasets.py \\
        --input logs/ml/offline_image2weather.jsonl \\
                logs/ml/offline_weather_time.jsonl \\
                logs/ml/offline_mwd.jsonl \\
                logs/ml/field_*.jsonl \\
        --require-label-source dataset_original,manual \\
        --output data/training/optical_only.jsonl \\
        --max-per-class 3000

    # Thermal ablation dataset (KAIST supplement)
    python tools/mix_datasets.py \\
        --input data/training/optical_only.jsonl \\
                logs/ml/offline_kaist.jsonl \\
        --require-thermal \\
        --output data/training/optical_thermal_ablation.jsonl
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smartbinocular.feature_schema import ENV_CLASSES, FEATURE_SET_CORE, FeatureRecord


def load_jsonl(paths: List[Path]) -> List[dict]:
    records = []
    for p in paths:
        if not p.exists():
            print(f"WARN: {p} not found", file=sys.stderr)
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return records


def filter_records(
    records: List[dict],
    require_label_sources: Optional[List[str]] = None,
    require_thermal: bool = False,
    require_motion: bool = False,
    min_confidence: Optional[float] = None,
    nir_channel: Optional[str] = None,
    labeled_only: bool = True,
) -> List[dict]:
    """Apply all filters; return filtered list."""
    out = []
    for r in records:
        # Must have effective label
        effective = r.get("label") or r.get("weak_label")
        if labeled_only and not effective:
            continue

        # Label source filter
        if require_label_sources:
            ls = r.get("label_source")
            if ls not in require_label_sources:
                continue

        # Confidence threshold (only applies to weak_heuristic)
        if min_confidence is not None and r.get("label_source") == "weak_heuristic":
            conf = r.get("label_confidence") or 0.0
            if conf < min_confidence:
                continue

        # Thermal required
        if require_thermal and not r.get("thermal_available", False):
            continue

        # Motion required
        if require_motion and not r.get("motion_available", False):
            continue

        # nir_channel filter
        if nir_channel and r.get("nir_channel") != nir_channel:
            continue

        # CORE features must all be present (sanity gate)
        if not all(r.get(f) is not None for f in FEATURE_SET_CORE):
            continue

        out.append(r)
    return out


def balance_classes(records: List[dict], max_per_class: int, seed: int = 42) -> List[dict]:
    """Downsample over-represented classes to max_per_class records."""
    by_class: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        lbl = r.get("label") or r.get("weak_label") or "unlabeled"
        by_class[lbl].append(r)

    rng = random.Random(seed)
    out = []
    for lbl, recs in sorted(by_class.items()):
        if max_per_class > 0 and len(recs) > max_per_class:
            recs = rng.sample(recs, max_per_class)
        out.extend(recs)
    rng.shuffle(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge and filter JSONL datasets for training."
    )
    parser.add_argument("--input", "-i", nargs="+", required=True, metavar="FILE")
    parser.add_argument("--output", "-o", required=True, type=Path)
    parser.add_argument(
        "--require-label-source",
        default=None,
        help="Comma-separated list: dataset_original,manual,weak_heuristic"
    )
    parser.add_argument(
        "--labeled-only", action="store_true", default=True,
        help="Only keep records with label or weak_label (default: True)"
    )
    parser.add_argument(
        "--require-thermal", action="store_true",
        help="Only keep records with thermal_available=True"
    )
    parser.add_argument(
        "--require-motion", action="store_true",
        help="Only keep records with motion_available=True"
    )
    parser.add_argument(
        "--min-confidence", type=float, default=None,
        help="Minimum label_confidence for weak_heuristic records"
    )
    parser.add_argument(
        "--nir-channel", default=None, choices=["nir", "rgb"],
        help="Filter to specific nir_channel"
    )
    parser.add_argument(
        "--max-per-class", type=int, default=0,
        help="Downsample to this many records per class (0 = no limit)"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Expand globs
    paths: List[Path] = []
    for pat in args.input:
        expanded = _glob.glob(pat)
        paths.extend(Path(e) for e in expanded) if expanded else paths.append(Path(pat))

    print(f"Loading from {len(paths)} file(s)...")
    records = load_jsonl(paths)
    print(f"Loaded: {len(records)} records")

    require_sources = None
    if args.require_label_source:
        require_sources = [s.strip() for s in args.require_label_source.split(",")]

    filtered = filter_records(
        records,
        require_label_sources=require_sources,
        require_thermal=args.require_thermal,
        require_motion=args.require_motion,
        min_confidence=args.min_confidence,
        nir_channel=args.nir_channel,
        labeled_only=args.labeled_only,
    )
    print(f"After filtering: {len(filtered)} records")

    if args.max_per_class > 0:
        filtered = balance_classes(filtered, args.max_per_class, seed=args.seed)
        print(f"After balancing (max {args.max_per_class}/class): {len(filtered)} records")

    # Print class distribution
    by_class: Dict[str, int] = defaultdict(int)
    for r in filtered:
        lbl = r.get("label") or r.get("weak_label") or "unlabeled"
        by_class[lbl] += 1
    print("\nFinal class distribution:")
    for lbl, cnt in sorted(by_class.items()):
        print(f"  {lbl:20s}: {cnt}")

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in filtered:
            f.write(json.dumps(r) + "\n")
    print(f"\nWritten: {len(filtered)} records → {args.output}")


if __name__ == "__main__":
    main()
