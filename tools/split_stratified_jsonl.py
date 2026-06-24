#!/usr/bin/env python3
"""split_stratified_jsonl.py — Per-class train/test split (e.g. 85/15) for FeatureRecord JSONL.

Reads either:
  - a directory of per-class files ``<label>.jsonl`` (from ``offline_pipeline --by-label-dir``), or
  - a single merged JSONL (groups rows by ``label`` / ``weak_label``).

Writes two shuffled JSONL files suitable for train vs hold-out evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smartbinocular.feature_schema import FEATURE_SET_CORE


def _effective_label(row: dict) -> str:
    return (row.get("label") or row.get("weak_label") or "unlabeled") or "unlabeled"


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _filter_core(rows: List[dict]) -> List[dict]:
    out: List[dict] = []
    for r in rows:
        if not all(r.get(f) is not None for f in FEATURE_SET_CORE):
            continue
        if _effective_label(r) == "unlabeled":
            continue
        out.append(r)
    return out


def _split_class(
    rows: List[dict], train_ratio: float, rng: random.Random
) -> Tuple[List[dict], List[dict]]:
    if not rows:
        return [], []
    xs = rows[:]
    rng.shuffle(xs)
    n = len(xs)
    n_train = int(round(n * train_ratio))
    n_train = max(0, min(n, n_train))
    if n >= 2:
        if n_train >= n:
            n_train = n - 1
        if n_train <= 0:
            n_train = 1
    else:
        n_train = 1
    return xs[:n_train], xs[n_train:]


def stratified_split(
    by_class: Dict[str, List[dict]], train_ratio: float, seed: int
) -> Tuple[List[dict], List[dict]]:
    rng = random.Random(seed)
    train_all: List[dict] = []
    test_all: List[dict] = []
    for lbl in sorted(by_class.keys()):
        tr, te = _split_class(by_class[lbl], train_ratio, rng)
        train_all.extend(tr)
        test_all.extend(te)
    rng.shuffle(train_all)
    rng.shuffle(test_all)
    return train_all, test_all


def _gather_from_dir(input_dir: Path, max_per_class: Optional[int], seed: int) -> Dict[str, List[dict]]:
    rng = random.Random(seed)
    by_class: DefaultDict[str, List[dict]] = defaultdict(list)
    for path in sorted(input_dir.glob("*.jsonl")):
        stem = path.stem
        rows = _filter_core(_load_jsonl(path))
        if max_per_class is not None and len(rows) > max_per_class:
            rows = rng.sample(rows, max_per_class)
        by_class[stem].extend(rows)
    return dict(by_class)


def _gather_from_file(path: Path, max_per_class: Optional[int], seed: int) -> Dict[str, List[dict]]:
    rng = random.Random(seed)
    by_class: DefaultDict[str, List[dict]] = defaultdict(list)
    for r in _filter_core(_load_jsonl(path)):
        by_class[_effective_label(r)].append(r)
    out: Dict[str, List[dict]] = {}
    for lbl, rows in by_class.items():
        if max_per_class is not None and len(rows) > max_per_class:
            rows = rng.sample(rows, max_per_class)
        out[lbl] = rows
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Stratified train/test split for ENV JSONL features.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--input-dir",
        type=Path,
        metavar="DIR",
        help="Directory of per-class JSONL files (e.g. logs/ml/by_label/)",
    )
    g.add_argument(
        "--input",
        type=Path,
        metavar="FILE",
        help="Single merged JSONL (group by label / weak_label)",
    )
    p.add_argument("--train-out", type=Path, required=True, help="Output train JSONL")
    p.add_argument("--test-out", type=Path, required=True, help="Output test JSONL")
    p.add_argument(
        "--train-ratio",
        type=float,
        default=0.85,
        help="Fraction of each class that goes to train (default: 0.85)",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for shuffles and splits")
    p.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        metavar="N",
        help="Optional cap per class before splitting (random subsample)",
    )
    args = p.parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        print("ERROR: --train-ratio must be between 0 and 1 (exclusive).", file=sys.stderr)
        sys.exit(1)

    if args.input_dir is not None:
        if not args.input_dir.is_dir():
            print(f"ERROR: not a directory: {args.input_dir}", file=sys.stderr)
            sys.exit(1)
        by_class = _gather_from_dir(args.input_dir, args.max_per_class, args.seed)
    else:
        if not args.input.is_file():
            print(f"ERROR: not a file: {args.input}", file=sys.stderr)
            sys.exit(1)
        by_class = _gather_from_file(args.input, args.max_per_class, args.seed)

    train_rows, test_rows = stratified_split(by_class, args.train_ratio, args.seed)

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.test_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.train_out, "w", encoding="utf-8") as tf, open(args.test_out, "w", encoding="utf-8") as vf:
        for r in train_rows:
            tf.write(json.dumps(r) + "\n")
        for r in test_rows:
            vf.write(json.dumps(r) + "\n")

    print(f"Classes: {len(by_class)}  |  train={len(train_rows)}  test={len(test_rows)}")
    print(f"Wrote {args.train_out}")
    print(f"Wrote {args.test_out}")


if __name__ == "__main__":
    main()
