"""plot_stage_timing.py — Stage-timing bar chart and per-frame latency histogram.

Reads session_*.json files from fusion_captures/metrics/ and produces:
  1. stage_timing_bars.png — grouped bar chart: per-stage mean latency by pipeline mode
  2. latency_hist_50ms.png — histogram of per-frame total latency with 50ms reference line
  3. stage_timing_by_mode.csv — tidy summary (mode, stage, n_frames, mean_ms, p50_ms, p95_ms)

Stage timing is reconstructed from `stage_timing_ms` in each session JSON.
Per-frame total latency = sum of all stage means for a session (not individual frame deltas;
those are not stored in the current schema). If `frames_by_mode` is present, mode is inferred
from whichever mode has the plurality of frames in the session.

Usage:
    python tools/plot_stage_timing.py
    python tools/plot_stage_timing.py \\
        --sessions fusion_captures/metrics/ \\
        --out-dir docs/thesis_eval/timing_performance/ \\
        --host RPi4

Caption note:
    "RPi4; existing sessions (24–244 s each); ≥5-min single-mode sessions pending."
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CAPTION = "RPi4; existing sessions (24–244 s); ≥5-min single-mode sessions pending."
TARGET_MS = 50.0

STAGE_ORDER = ["framecache", "nir_bucket", "thermal_proc", "jerk", "ml_infer", "blend", "hud", "display"]
MODE_COLORS = {"imx": "#2c7bb6", "thermal": "#d7191c", "fusion": "#1a9641", "mixed": "#999999"}


def _import_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("ERROR: matplotlib not installed.", file=sys.stderr)
        sys.exit(1)


def _infer_mode(session: dict) -> str:
    """Return dominant mode from frames_by_mode, or 'mixed' if even split."""
    fbm = session.get("frames_by_mode", {})
    if not fbm:
        return "mixed"
    dom = max(fbm, key=fbm.get)
    total = sum(fbm.values())
    if fbm[dom] / total >= 0.70:
        return dom
    return "mixed"


def _load_sessions(session_dir: Path) -> List[dict]:
    files = sorted(session_dir.glob("session_*.json"))
    if not files:
        print(f"ERROR: No session_*.json in {session_dir}. Run pipeline on RPi4 first.", file=sys.stderr)
        sys.exit(1)
    sessions = []
    for f in files:
        try:
            data = json.loads(f.read_text())
            if not isinstance(data, dict):
                continue
            if "stage_timing_ms" not in data:
                continue
            sessions.append(data)
        except Exception as e:
            print(f"  Skipping {f.name}: {e}", file=sys.stderr)
    print(f"Loaded {len(sessions)} sessions with stage_timing_ms from {session_dir}")
    return sessions


def _collect_by_mode(sessions: List[dict]) -> Dict[str, Dict[str, List[float]]]:
    """Map mode → stage → [mean_ms values across sessions]."""
    by_mode: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for s in sessions:
        mode = _infer_mode(s)
        timing = s.get("stage_timing_ms", {})
        for stage, stats in timing.items():
            if isinstance(stats, dict) and "mean_ms" in stats:
                by_mode[mode][stage].append(float(stats["mean_ms"]))
    return by_mode


def plot_stage_bars(by_mode: Dict, out_dir: Path, plt) -> None:
    stages = [s for s in STAGE_ORDER if any(s in by_mode[m] for m in by_mode)]
    # add any stages not in STAGE_ORDER
    all_stages_seen = set()
    for m in by_mode:
        all_stages_seen |= set(by_mode[m].keys())
    stages += sorted(all_stages_seen - set(stages))

    modes = sorted(by_mode.keys())
    x = np.arange(len(stages))
    width = 0.8 / max(len(modes), 1)

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, mode in enumerate(modes):
        means = [float(np.mean(by_mode[mode][s])) if by_mode[mode][s] else 0.0 for s in stages]
        stds = [float(np.std(by_mode[mode][s])) if len(by_mode[mode][s]) > 1 else 0.0 for s in stages]
        offset = (i - len(modes) / 2 + 0.5) * width
        color = MODE_COLORS.get(mode, "#888888")
        ax.bar(x + offset, means, width * 0.9, label=mode, color=color, alpha=0.85, yerr=stds, capsize=3)

    ax.axhline(TARGET_MS, color="red", ls="--", lw=1.5, label=f"{TARGET_MS} ms target")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean stage latency (ms)")
    ax.set_title(
        "Per-stage latency by pipeline mode — existing RPi4 sessions\n"
        f"({CAPTION})",
        fontsize=9,
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = out_dir / "figures" / "stage_timing_bars.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_latency_hist(sessions: List[dict], out_dir: Path, plt) -> None:
    """Histogram of per-session sum-of-stage-means (proxy for frame latency).

    Note: session JSON stores means, not individual frame timings. This plot
    shows distribution of per-session mean total latency, not per-frame raw values.
    """
    totals = []
    for s in sessions:
        timing = s.get("stage_timing_ms", {})
        total = sum(
            float(v["mean_ms"]) for v in timing.values()
            if isinstance(v, dict) and "mean_ms" in v
        )
        if total > 0:
            totals.append(total)

    if not totals:
        print("  No timing data for latency histogram; skipping.", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(totals, bins=20, color="#2c7bb6", edgecolor="white", alpha=0.85)
    ax.axvline(TARGET_MS, color="red", ls="--", lw=2, label=f"{TARGET_MS} ms budget")
    ax.set_xlabel("Sum-of-stage-means per session (ms)")
    ax.set_ylabel("Session count")
    pct_under = 100 * sum(1 for t in totals if t <= TARGET_MS) / len(totals)
    ax.set_title(
        f"Per-session total stage latency distribution (n={len(totals)} sessions)\n"
        f"{pct_under:.0f}% sessions under {TARGET_MS} ms; {CAPTION}",
        fontsize=9,
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = out_dir / "figures" / "latency_hist_50ms.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")
    print(f"  Note: histogram shows sum-of-stage-means, NOT individual frame latency distributions.")


def write_csv(by_mode: Dict, out_dir: Path) -> None:
    rows = []
    for mode in sorted(by_mode.keys()):
        for stage in sorted(by_mode[mode].keys()):
            vals = by_mode[mode][stage]
            if not vals:
                continue
            rows.append({
                "mode": mode,
                "stage": stage,
                "n_sessions": len(vals),
                "mean_ms": round(float(np.mean(vals)), 3),
                "std_ms": round(float(np.std(vals)), 3),
                "p50_ms": round(float(np.median(vals)), 3),
                "p95_ms": round(float(np.percentile(vals, 95)), 3),
            })
    out = out_dir / "tables" / "stage_timing_by_mode.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["mode", "stage", "n_sessions", "mean_ms", "std_ms", "p50_ms", "p95_ms"]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {out}")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sessions", type=Path, default=ROOT / "fusion_captures/metrics")
    p.add_argument("--out-dir", type=Path, default=ROOT / "docs/thesis_eval/timing_performance")
    p.add_argument("--host", default="RPi4", help="Host label for captions")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    plt = _import_mpl()

    if not args.sessions.exists():
        print(f"ERROR: session dir not found: {args.sessions}", file=sys.stderr)
        sys.exit(1)

    sessions = _load_sessions(args.sessions)
    by_mode = _collect_by_mode(sessions)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_stage_bars(by_mode, args.out_dir, plt)
    plot_latency_hist(sessions, args.out_dir, plt)
    write_csv(by_mode, args.out_dir)
    print(f"\nDone. Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
