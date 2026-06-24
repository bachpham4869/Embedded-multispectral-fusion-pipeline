"""
Compute per-stage latency statistics from session manifests.

Usage:
    python tools/measure_stage_timing.py [--session-dir PATH] [--out-dir PATH]

Inputs:
    Session JSON files matching fusion_captures/metrics/session_*.json
    (or --session-dir override). Each file must contain a list of dicts with
    at least one of the keys: "stage_timing_ms" or "fuse_stage_timing_ms".

Outputs:
    docs/tables/timing/stage_timing_summary.csv  — one row per stage; columns:
        stage, n_frames, mean_ms, p50_ms, p95_ms, p99_ms
    docs/tables/timing/stage_timing_summary.md   — markdown table of the same data

Failure mode:
    If no session files are found, the script prints a clear message and exits
    with code 1. Evidence register marks all frame-budget claims as UNVERIFIED
    until this artifact is committed.
"""
import argparse
import csv
import glob
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session-dir", default=None,
                   help="Directory to search for session_*.json (default: fusion_captures/metrics/)")
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: docs/tables/timing/)")
    return p.parse_args()


def _find_session_files(session_dir: pathlib.Path) -> list:
    pattern = str(session_dir / "session_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        alt_pattern = str(session_dir / "*.json")
        files = sorted(glob.glob(alt_pattern))
    return files


def main():
    args = _parse_args()
    repo = pathlib.Path(__file__).parent.parent

    session_dir = pathlib.Path(args.session_dir) if args.session_dir else repo / "fusion_captures/metrics"
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else repo / "docs/tables/timing"

    files = _find_session_files(session_dir)
    if not files:
        print(
            f"ERROR: No session_*.json files found in {session_dir}.\n"
            f"Run the pipeline on RPi4 first to generate timing artifacts:\n"
            f"  python -m smartbinocular\n"
            f"Until this data is committed, frame-budget claims in README.md and\n"
            f"docs/PIPELINE_EVIDENCE_REGISTER.md Part A §A.3 remain UNVERIFIED.",
            file=sys.stderr,
        )
        sys.exit(1)

    stage_samples: dict = defaultdict(list)

    for fpath in files:
        with open(fpath) as fh:
            data = json.load(fh)
        records = data if isinstance(data, list) else [data]
        for rec in records:
            for key in ("stage_timing_ms", "fuse_stage_timing_ms"):
                timing = rec.get(key)
                if not isinstance(timing, dict):
                    continue
                for stage, val in timing.items():
                    if val is None:
                        continue
                    # Support both pre-aggregated {mean_ms, std_ms, n} and raw float
                    if isinstance(val, dict):
                        mean_ms = val.get("mean_ms")
                        if mean_ms is not None:
                            stage_samples[stage].append(float(mean_ms))
                    else:
                        stage_samples[stage].append(float(val))

    if not stage_samples:
        print(
            f"ERROR: Found {len(files)} JSON file(s) but no 'stage_timing_ms' or "
            f"'fuse_stage_timing_ms' keys with non-null values.\n"
            f"Ensure the pipeline is built with stage profiling enabled (StageProfiler).",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "stage_timing_summary.csv"
    md_path = out_dir / "stage_timing_summary.md"

    rows = []
    for stage in sorted(stage_samples):
        arr = np.array(stage_samples[stage])
        rows.append({
            "stage": stage,
            "n_frames": len(arr),
            "mean_ms": round(float(np.mean(arr)), 3),
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3),
        })

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["stage", "n_frames", "mean_ms", "p50_ms", "p95_ms", "p99_ms"])
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w") as fh:
        fh.write(f"# Stage Timing Summary\n\n")
        fh.write(f"Generated from {len(files)} session file(s) in `{session_dir.relative_to(repo)}`.\n\n")
        fh.write("| stage | n_frames | mean_ms | p50_ms | p95_ms | p99_ms |\n")
        fh.write("|-------|----------|---------|--------|--------|--------|\n")
        for r in rows:
            fh.write(
                f"| {r['stage']} | {r['n_frames']} | {r['mean_ms']} "
                f"| {r['p50_ms']} | {r['p95_ms']} | {r['p99_ms']} |\n"
            )

    print(f"Written: {csv_path.relative_to(repo)}")
    print(f"Written: {md_path.relative_to(repo)}")
    print(f"Processed {len(files)} session file(s), {len(stage_samples)} stages.")


if __name__ == "__main__":
    main()
