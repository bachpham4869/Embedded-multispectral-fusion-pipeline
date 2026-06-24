"""aggregate_transitions.py — Transitions per minute from session JSON files.

Reads session_*.json and extracts ENV class and bucket distribution from
`frames_by_env` and `frames_by_bucket`. Estimates transition counts using the
observation that the number of env-class transitions is bounded by the number of
distinct env classes observed × a continuity factor.

Since the current session JSON schema stores aggregate frame counts (not per-frame
sequences), exact transition counts cannot be derived. This script uses:

    estimated_transitions = sum(nonzero bucket counts) - 1
    transitions_per_min   = estimated_transitions / (duration_wall_s / 60)

This is a lower-bound estimate (assuming minimal switching). It is labeled
ESTIMATED in the output CSV. True transitions require per-frame env/bucket logs.

Usage:
    python tools/aggregate_transitions.py
    python tools/aggregate_transitions.py \\
        --sessions fusion_captures/metrics/ \\
        --out docs/thesis_eval/timing_performance/tables/transitions_per_minute.csv

Output CSV columns:
    session_id, duration_s, mode, n_env_classes, n_buckets_used,
    est_env_transitions, est_bucket_transitions,
    est_env_transitions_per_min, est_bucket_transitions_per_min, note
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent


def _load_sessions(session_dir: Path) -> List[dict]:
    files = sorted(session_dir.glob("session_*.json"))
    if not files:
        print(f"ERROR: No session_*.json in {session_dir}", file=sys.stderr)
        sys.exit(1)
    out = []
    for f in files:
        try:
            data = json.loads(f.read_text())
            if isinstance(data, dict):
                out.append(data)
        except Exception as e:
            print(f"  Skipping {f.name}: {e}", file=sys.stderr)
    return out


def _infer_mode(session: dict) -> str:
    fbm = session.get("frames_by_mode", {})
    if not fbm:
        return "mixed"
    dom = max(fbm, key=fbm.get)
    total = sum(fbm.values())
    return dom if fbm[dom] / total >= 0.70 else "mixed"


def process_sessions(sessions: List[dict]) -> List[dict]:
    rows = []
    for s in sessions:
        sid = s.get("session_id", "unknown")
        dur = float(s.get("duration_wall_s", 0))
        if dur <= 0:
            continue
        mode = _infer_mode(s)

        # ENV transitions estimate
        fbe = s.get("frames_by_env", {})
        n_env = sum(1 for v in fbe.values() if int(v) > 0)
        est_env_trans = max(0, n_env - 1)
        est_env_tpm = round(est_env_trans / (dur / 60), 2) if dur > 0 else 0.0

        # Bucket transitions estimate
        fbb = s.get("frames_by_bucket", {})
        n_bkts = sum(1 for v in fbb.values() if int(v) > 0)
        est_bkt_trans = max(0, n_bkts - 1)
        est_bkt_tpm = round(est_bkt_trans / (dur / 60), 2) if dur > 0 else 0.0

        rows.append({
            "session_id": sid,
            "duration_s": round(dur, 1),
            "mode": mode,
            "n_env_classes": n_env,
            "n_buckets_used": n_bkts,
            "est_env_transitions": est_env_trans,
            "est_bucket_transitions": est_bkt_trans,
            "est_env_transitions_per_min": est_env_tpm,
            "est_bucket_transitions_per_min": est_bkt_tpm,
            "note": "ESTIMATED — lower bound from distinct class counts, not per-frame sequence",
        })
    return rows


def write_csv(rows: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "session_id", "duration_s", "mode", "n_env_classes", "n_buckets_used",
        "est_env_transitions", "est_bucket_transitions",
        "est_env_transitions_per_min", "est_bucket_transitions_per_min", "note",
    ]
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {out_path}  ({len(rows)} rows)")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sessions", type=Path, default=ROOT / "fusion_captures/metrics")
    p.add_argument("--out", type=Path, default=ROOT / "docs/thesis_eval/timing_performance/tables/transitions_per_minute.csv")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    sessions = _load_sessions(args.sessions)
    rows = process_sessions(sessions)
    if not rows:
        print("No usable sessions found.", file=sys.stderr)
        sys.exit(1)
    write_csv(rows, args.out)
    # Print summary
    tpms = [r["est_env_transitions_per_min"] for r in rows]
    print(f"Env transitions/min: mean={sum(tpms)/len(tpms):.2f}, "
          f"min={min(tpms):.2f}, max={max(tpms):.2f}")
    print("Note: ESTIMATED lower bounds — per-frame sequence logging would give exact counts.")


if __name__ == "__main__":
    main()
