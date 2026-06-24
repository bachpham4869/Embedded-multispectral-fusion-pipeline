#!/usr/bin/env python3
"""validate_schema.py — Validate a JSONL file against FeatureRecord schema.

Usage:
    python tools/validate_schema.py --input logs/ml/offline_mwd.jsonl
    python tools/validate_schema.py --input logs/ml/*.jsonl --strict

Exit codes:
    0 — PASS
    1 — FAIL (schema errors found)
    2 — Usage error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Allow running from repo root without pip install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smartbinocular.feature_schema import (
    ENV_CLASSES,
    FEATURE_SET_CORE,
    FeatureRecord,
)


# ── Checks ────────────────────────────────────────────────────────────────────

_VALID_NIR_CHANNELS = {"nir", "rgb"}
_VALID_THERMAL_CHANNELS = {"lwir", "none"}
_VALID_LABEL_SOURCES = {"dataset_original", "manual", "weak_heuristic"}
_ENV_CLASS_SET = set(ENV_CLASSES)


def _check_record(idx: int, d: dict) -> List[str]:
    """Return list of error strings for this record, empty if valid."""
    errors: List[str] = []

    # 1. CORE features must be non-None, non-NaN
    for feat in FEATURE_SET_CORE:
        val = d.get(feat)
        if val is None:
            errors.append(f"[{idx}] CORE feature '{feat}' is None")
        else:
            try:
                fval = float(val)
                import math
                if math.isnan(fval) or math.isinf(fval):
                    errors.append(f"[{idx}] CORE feature '{feat}' is NaN/Inf ({val})")
            except (TypeError, ValueError):
                errors.append(f"[{idx}] CORE feature '{feat}' is not numeric ({val!r})")

    # 2. nir_channel and thermal_channel values
    nir_ch = d.get("nir_channel")
    if nir_ch not in _VALID_NIR_CHANNELS:
        errors.append(f"[{idx}] nir_channel={nir_ch!r} not in {_VALID_NIR_CHANNELS}")

    thm_ch = d.get("thermal_channel")
    if thm_ch not in _VALID_THERMAL_CHANNELS:
        errors.append(f"[{idx}] thermal_channel={thm_ch!r} not in {_VALID_THERMAL_CHANNELS}")

    # 3. Thermal availability consistency
    if d.get("thermal_available") is True:
        for tf in ["thm_mean", "thm_std", "thm_max"]:
            if d.get(tf) is None:
                errors.append(f"[{idx}] thermal_available=True but {tf} is None")

    # 4. Temporal availability consistency
    if d.get("temporal_available") is True:
        if d.get("nir_brightness_delta_10f") is None:
            errors.append(f"[{idx}] temporal_available=True but nir_brightness_delta_10f is None")

    # 5. Offline records must NOT have runtime_only features
    source = d.get("source", "rpi")
    if source != "rpi":
        if d.get("skew_ms") is not None:
            errors.append(f"[{idx}] Offline record has skew_ms set (source={source!r})")
        if d.get("fusion_alpha") is not None:
            errors.append(f"[{idx}] Offline record has fusion_alpha set (source={source!r})")

    # 6. label_source must be valid if present
    ls = d.get("label_source")
    if ls is not None and ls not in _VALID_LABEL_SOURCES:
        errors.append(f"[{idx}] label_source={ls!r} not in {_VALID_LABEL_SOURCES}")

    # 7. If label_source = "dataset_original", label must be set
    if ls == "dataset_original" and not d.get("label"):
        errors.append(f"[{idx}] label_source='dataset_original' but label is missing")

    # 8. label value must be a valid ENV class if set
    lbl = d.get("label")
    if lbl is not None and lbl not in _ENV_CLASS_SET:
        errors.append(f"[{idx}] label={lbl!r} is not a valid ENV class")
    wlbl = d.get("weak_label")
    if wlbl is not None and wlbl not in _ENV_CLASS_SET:
        errors.append(f"[{idx}] weak_label={wlbl!r} is not a valid ENV class")

    return errors


# ── Main ──────────────────────────────────────────────────────────────────────

def validate_file(path: Path, strict: bool = False, max_errors: int = 50) -> bool:
    """Validate one JSONL file. Returns True if PASS."""
    all_errors: List[str] = []
    total = 0
    parse_errors = 0

    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                all_errors.append(f"[line {line_no}] JSON parse error: {e}")
                parse_errors += 1
                if len(all_errors) >= max_errors:
                    break
                continue
            total += 1
            record_errors = _check_record(total, d)
            all_errors.extend(record_errors)
            if len(all_errors) >= max_errors:
                all_errors.append(f"... (truncated after {max_errors} errors)")
                break

    status = "PASS" if not all_errors else "FAIL"
    label_counts: dict = {}
    source_counts: dict = {}

    # Second pass for stats if PASS
    if not all_errors:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                lbl = d.get("label") or d.get("weak_label") or "unlabeled"
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
                src = d.get("source", "unknown")
                source_counts[src] = source_counts.get(src, 0) + 1

    print(f"\n{'='*60}")
    print(f"File: {path}")
    print(f"Records: {total}  |  Status: {status}")
    if label_counts:
        print("Label distribution:")
        for lbl, cnt in sorted(label_counts.items()):
            print(f"  {lbl:20s}: {cnt}")
    if source_counts:
        print("Source distribution:")
        for src, cnt in sorted(source_counts.items()):
            print(f"  {src:30s}: {cnt}")
    if all_errors:
        print(f"\nErrors ({len(all_errors)}):")
        for e in all_errors:
            print(f"  {e}")

    return not all_errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate FeatureRecord JSONL files against schema."
    )
    parser.add_argument(
        "--input", "-i", nargs="+", required=True,
        metavar="FILE", help="JSONL file(s) to validate (glob patterns supported)"
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit 1 on any warning (not just hard errors)"
    )
    parser.add_argument(
        "--max-errors", type=int, default=50,
        help="Stop reporting after this many errors per file (default: 50)"
    )
    args = parser.parse_args()

    import glob as _glob
    paths: List[Path] = []
    for pattern in args.input:
        expanded = _glob.glob(pattern)
        if expanded:
            paths.extend(Path(p) for p in expanded)
        else:
            paths.append(Path(pattern))

    if not paths:
        print("ERROR: No input files found.", file=sys.stderr)
        sys.exit(2)

    all_pass = True
    for p in paths:
        if not p.exists():
            print(f"ERROR: File not found: {p}", file=sys.stderr)
            all_pass = False
            continue
        ok = validate_file(p, strict=args.strict, max_errors=args.max_errors)
        if not ok:
            all_pass = False

    print(f"\n{'='*60}")
    print(f"Overall: {'PASS' if all_pass else 'FAIL'} ({len(paths)} file(s))")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
