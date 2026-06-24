"""audit_night_classes.py — P2-C: count per-class samples in training JSONL files.

Reports sample distribution for the night-vision in-scope classes:
  night_clear, normal_night, nir_night, fog (at night), rain (at night)

Usage:
    python tools/audit_night_classes.py
    python tools/audit_night_classes.py --files data/training/from_logs_train.jsonl data/training/merged_logs_ml.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

# Night-vision scope: primary + secondary weather classes
_NIGHT_PRIMARY = frozenset({"night_clear", "normal_night", "nir_night"})
_NIGHT_SECONDARY = frozenset({"fog", "rain"})
_REFERENCE_ONLY = frozenset({"normal_day"})
_OUT_OF_SCOPE = frozenset({"glare", "backlight", "transition", "indoor"})
_ALL_KNOWN = _NIGHT_PRIMARY | _NIGHT_SECONDARY | _REFERENCE_ONLY | _OUT_OF_SCOPE

_DEFAULT_FILES = [
    "data/training/from_logs_train.jsonl",
    "data/training/from_logs_test.jsonl",
    "data/training/merged_logs_ml.jsonl",
]


def _load_jsonl(path: Path) -> List[Dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [warn] {path.name}:{i}: {e}", file=sys.stderr)
    return records


def audit_file(path: Path) -> Dict[str, int]:
    records = _load_jsonl(path)
    counts: Counter = Counter()
    for rec in records:
        label = rec.get("env_class") or rec.get("label") or rec.get("env_label")
        if label is not None:
            counts[str(label)] += 1
    return dict(counts)


def print_report(path: Path, counts: Dict[str, int]) -> None:
    total = max(1, sum(counts.values()))
    print(f"\n{'=' * 60}")
    print(f"File: {path}")
    print(f"Total labelled records: {sum(counts.values())}")
    print(f"{'=' * 60}")

    def _section(title: str, classes: frozenset) -> None:
        subset = {k: v for k, v in counts.items() if k in classes}
        if not subset:
            print(f"\n  [{title}] — no samples found")
            return
        print(f"\n  [{title}]")
        for cls in sorted(subset, key=lambda c: -subset[c]):
            pct = subset[cls] / total * 100
            bar = "#" * int(pct / 2)
            print(f"    {cls:<22} {subset[cls]:>6}  ({pct:5.1f}%)  {bar}")

    _section("PRIMARY night classes (in-scope)", _NIGHT_PRIMARY)
    _section("SECONDARY weather@night (in-scope)", _NIGHT_SECONDARY)
    _section("REFERENCE baseline (control only)", _REFERENCE_ONLY)
    _section("OUT-OF-SCOPE rare classes", _OUT_OF_SCOPE)

    unknown = {k: v for k, v in counts.items() if k not in _ALL_KNOWN}
    if unknown:
        print(f"\n  [UNKNOWN labels]")
        for cls, n in sorted(unknown.items(), key=lambda x: -x[1]):
            print(f"    {cls:<22} {n:>6}")

    night_total = sum(counts.get(c, 0) for c in _NIGHT_PRIMARY)
    print(f"\n  Night-primary total : {night_total} / {sum(counts.values())} "
          f"({night_total / total * 100:.1f}%)")
    if night_total > 0:
        for cls in sorted(_NIGHT_PRIMARY):
            n = counts.get(cls, 0)
            print(f"    {cls:<22} {n:>6}  ({n / max(1, night_total) * 100:.1f}% of night-primary)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit night-class sample distribution.")
    parser.add_argument(
        "--files", nargs="*", default=_DEFAULT_FILES,
        help="JSONL files to audit (default: standard training splits)",
    )
    parser.add_argument(
        "--root", default=".",
        help="Project root for resolving relative paths (default: cwd)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    any_found = False
    for f in args.files:
        path = root / f
        if not path.is_file():
            print(f"[skip] not found: {path}")
            continue
        any_found = True
        counts = audit_file(path)
        print_report(path, counts)

    if not any_found:
        print("[error] No JSONL files found. Run from project root or pass --root.")
        sys.exit(1)


if __name__ == "__main__":
    main()
