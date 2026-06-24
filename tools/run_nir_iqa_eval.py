"""Batch NIR IQA evaluation driver — runs all buckets over a manifest.

Invokes ``batch_nir_enhancer.py`` for buckets A–F and writes a dated CSV under
``data/eval/iqa_runs/`` (override with ``--out-dir``).

Usage:
    python tools/run_nir_iqa_eval.py
    python tools/run_nir_iqa_eval.py --manifest data/eval/nir_val/manifest_v2.csv --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/eval/nir_val/manifest_v2.csv")
    parser.add_argument("--out-dir", default="data/eval/iqa_runs")
    parser.add_argument("--buckets", nargs="+", default=list("ABCDEF"),
                        choices=list("ABCDEF"))
    parser.add_argument("--rain-sequence-mode", action="store_true")
    parser.add_argument("--note", default="", help="One-line note for MANIFEST.md")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    today = date.today().isoformat()
    out_csv = out_dir / f"round_{today}.csv"

    cmd = [
        sys.executable, "tools/batch_nir_enhancer.py",
        "--manifest", args.manifest,
        "--bucket", *args.buckets,
        "--out", str(out_csv),
    ]
    if args.rain_sequence_mode:
        cmd.append("--rain-sequence-mode")
    if args.dry_run:
        cmd.append("--dry-run")

    print(f"[run_nir_iqa_eval] Writing to {out_csv}", file=sys.stderr)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print("ERROR: batch_nir_enhancer.py failed", file=sys.stderr)
        sys.exit(result.returncode)

    # Append to MANIFEST.md.
    import hashlib
    import subprocess as sp
    try:
        git_hash = sp.check_output(["git", "rev-parse", "--short", "HEAD"],
                                   text=True).strip()
    except Exception:
        git_hash = "unknown"
    try:
        manifest_sha = hashlib.sha256(
            Path(args.manifest).read_bytes()
        ).hexdigest()[:12]
    except Exception:
        manifest_sha = "unknown"

    manifest_log = out_dir / "MANIFEST.md"
    with open(manifest_log, "a") as f:
        f.write(f"| {out_csv.name} | {git_hash} | {manifest_sha} | {args.note or '—'} |\n")

    if not manifest_log.stat().st_size or "| CSV |" not in manifest_log.read_text():
        # Prepend header if file was just created or lacks it.
        content = manifest_log.read_text()
        if "| CSV |" not in content:
            header = "| CSV | git | manifest_sha | note |\n|-----|-----|-------------|------|\n"
            manifest_log.write_text(header + content)

    print(f"[run_nir_iqa_eval] Done. Logged to {manifest_log}", file=sys.stderr)


if __name__ == "__main__":
    main()
