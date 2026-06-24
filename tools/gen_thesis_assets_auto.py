#!/usr/bin/env python3
"""Auto-run thesis figure generators in a sensible order.

**Phase 1 — Static NIR panels** (single-image ``gen_thesis_charts`` targets):
  ``--clahe-sample``, ``--dcp-sample``, ``--night-sample``, ``--glare-sample``

**Phase 2 — Aggregates** (need ``docs/tables/iqa/batch_nir_iqa.csv``):
  ``--heatmap``, ``--barchart``

**Phase 3 — Thermal temporal strip**:
  ``extract_thermal_mi48_sequence`` merges consecutive MI48 PNGs (best ``FramesClass*``
  under ``video_dataset10`` / ``video_dataset5`` when clips are short) into
  ``data/thermal/scaled_mi48_sequences/auto_clip``

**Phase 4 — Thermal pipeline figure**:
  ``gen_thesis_thermal_figures`` on the sequence folder (fallback: mixed ``scaled_mi48_80x62``).

Run from repo root::

    uv run python tools/gen_thesis_assets_auto.py

Options skip phases via ``--skip-*``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
PY = sys.executable
TOOLS = PROJ / "tools"
SEQ_DIR = PROJ / "data/thermal/scaled_mi48_sequences/auto_clip"
MIXED_DIR = PROJ / "data/thermal/scaled_mi48_80x62"


def run(argv: list[str]) -> bool:
    print("\n→", " ".join(argv), flush=True)
    r = subprocess.run(argv, cwd=str(PROJ))
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-static-nir", action="store_true")
    ap.add_argument("--skip-aggregate", action="store_true")
    ap.add_argument("--skip-thermal-sequence", action="store_true")
    ap.add_argument("--skip-thermal-figure", action="store_true")
    ap.add_argument("--thermal-rows", type=int, default=3)
    ap.add_argument("--thermal-seq-length", type=int, default=56)
    ap.add_argument("--sequence-max-frames", type=int, default=120)
    args = ap.parse_args()

    ok = True

    if not args.skip_static_nir:
        ok &= run(
            [
                PY,
                str(TOOLS / "gen_thesis_charts.py"),
                "--clahe-sample",
                "--dcp-sample",
                "--night-sample",
                "--glare-sample",
            ]
        )

    if not args.skip_aggregate:
        ok &= run([PY, str(TOOLS / "gen_thesis_charts.py"), "--heatmap", "--barchart"])

    if not args.skip_thermal_sequence:
        ok &= run(
            [
                PY,
                str(TOOLS / "extract_thermal_mi48_sequence.py"),
                "--output",
                str(SEQ_DIR),
                "--max-frames",
                str(args.sequence_max_frames),
            ]
        )

    if not args.skip_thermal_figure:
        thermal_input = SEQ_DIR
        seq_glob = "seq_*.png"
        if not thermal_input.is_dir() or not any(thermal_input.glob(seq_glob)):
            print(f"\n⚠️ No {seq_glob} under {thermal_input}; fallback to mixed scaled folder.")
            thermal_input = MIXED_DIR
            seq_glob = "*thermal_image_dataset*.png"

        ok &= run(
            [
                PY,
                str(TOOLS / "gen_thesis_thermal_figures.py"),
                "--input-dir",
                str(thermal_input),
                "--glob",
                seq_glob,
                "--rows",
                str(args.thermal_rows),
                "--sequence-length",
                str(args.thermal_seq_length),
            ]
        )

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
