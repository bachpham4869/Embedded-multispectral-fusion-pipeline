"""diff_iqa_runs.py — Compare two batch IQA CSVs and emit a Markdown delta table.

Usage:
    python tools/diff_iqa_runs.py data/eval/iqa_runs/round_A.csv data/eval/iqa_runs/round_B.csv
    python tools/diff_iqa_runs.py --baseline round_A.csv --updated round_B.csv \\
        --out docs/tables/iqa/round_diff.md
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def _load(path: str) -> dict[tuple, dict]:
    """Return {(env_class, bucket): {metric: mean}} for a batch CSV."""
    by_key: dict[tuple, list] = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get("env_class", "?"), row.get("bucket", "?"))
            by_key[key].append(row)

    result: dict[tuple, dict] = {}
    for key, rows in by_key.items():
        n = len(rows)
        result[key] = {
            "n": n,
            "mean_delta_log_rms": sum(
                float(r["after_log_rms_contrast"]) - float(r["before_log_rms_contrast"])
                for r in rows
            ) / n,
            "mean_pct_sat_after": sum(float(r["after_pct_saturated"]) for r in rows) / n,
            "mean_pct_crush_after": sum(float(r["after_pct_crushed"]) for r in rows) / n,
        }
    return result


def _fmt(v: float, decimals: int = 4) -> str:
    return f"{v:+.{decimals}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="Baseline batch CSV")
    parser.add_argument("updated", help="Updated batch CSV")
    parser.add_argument("--out", default=None, help="Write Markdown to this file")
    args = parser.parse_args()

    base = _load(args.baseline)
    upd = _load(args.updated)

    all_keys = sorted(set(base) | set(upd))

    lines = [
        f"# IQA Run Diff: {Path(args.baseline).name} → {Path(args.updated).name}\n",
        "",
        "| env_class | bucket | Δ(Δlog_rms) | Δpct_sat_after | Δpct_crush_after |",
        "|-----------|--------|------------|---------------|-----------------|",
    ]

    for key in all_keys:
        env, bkt = key
        b = base.get(key, {})
        u = upd.get(key, {})
        if not b or not u:
            lines.append(f"| {env} | {bkt} | (missing in one run) | — | — |")
            continue
        d_rms = u["mean_delta_log_rms"] - b["mean_delta_log_rms"]
        d_sat = u["mean_pct_sat_after"] - b["mean_pct_sat_after"]
        d_crush = u["mean_pct_crush_after"] - b["mean_pct_crush_after"]
        # Bold cells with meaningful change.
        flag = " ⚑" if abs(d_rms) > 0.05 or abs(d_sat) > 0.02 or abs(d_crush) > 0.02 else ""
        lines.append(
            f"| {env} | {bkt} | {_fmt(d_rms)}{flag} | {_fmt(d_sat)} | {_fmt(d_crush)} |"
        )

    output = "\n".join(lines) + "\n"
    print(output)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(output)
        print(f"Wrote diff to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
