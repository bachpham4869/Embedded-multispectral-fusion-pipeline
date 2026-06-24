"""sweep_rain_median_n.py — Sweep rain temporal-median N on rain/night frames.

Tests RainTemporalMedian at N∈{2,3,5,7} frames: measures IQA improvement and
added per-frame latency vs single-frame (no median).

THESIS_SKIP guard: If no rain-class frames are found in the manifest, this script
exits 0 with a THESIS_SKIP note. The manifest_v2.csv has 20 rain frames, so this
script WILL run if that manifest is present. Frames feed 3-frame pseudo-sequences
(original + 2 noise variants) per the batch_nir_enhancer.py pattern.

Usage:
    python tools/sweep_rain_median_n.py
    python tools/sweep_rain_median_n.py \\
        --manifest data/eval/nir_val/manifest_v2.csv \\
        --n-list 2,3,5,7 \\
        --out docs/thesis_eval/nir_enhancement/tables/rain_median_n_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from _iqa_metrics import compute_iqa_metrics  # type: ignore[import]
from smartbinocular.nir_pipeline import RainTemporalMedian

DEFAULT_MANIFEST = ROOT / "data/eval/nir_val/manifest_v2.csv"
DEFAULT_N_LIST = [2, 3, 5, 7]
DEFAULT_OUT = ROOT / "docs/thesis_eval/nir_enhancement/tables/rain_median_n_sweep.csv"


def _load_rain_frames(manifest_path: Path) -> List[dict]:
    rows = []
    with manifest_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            p = row.get("path", "")
            cls = row.get("env_class", "")
            if cls == "rain" and p and Path(p).exists():
                rows.append(row)
    return rows


def _make_pseudo_sequence(bgr: np.ndarray, n: int) -> List[np.ndarray]:
    """Generate n frames: original + (n-1) noise/jitter variants."""
    frames = [bgr]
    rng = np.random.default_rng(42)
    for _ in range(n - 1):
        noise = rng.integers(-3, 4, bgr.shape, dtype=np.int16)
        f = np.clip(bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(f)
    return frames


def run_sweep(frames: List[dict], n_list: List[int]) -> List[dict]:
    rows = []
    for frame in frames:
        img_path = frame["path"]
        bgr = cv.imread(img_path, cv.IMREAD_COLOR)
        if bgr is None:
            continue
        bgr = cv.resize(bgr, (320, 240))

        raw_gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
        raw_iqa = compute_iqa_metrics(raw_gray)

        for n in n_list:
            processor = RainTemporalMedian(n_frames=n)
            seq = _make_pseudo_sequence(bgr, n)

            t0 = time.perf_counter()
            result = bgr
            for f in seq:
                result = processor.process(f)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            out_gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
            out_iqa = compute_iqa_metrics(out_gray)

            rows.append({
                "frame": Path(img_path).name,
                "env_class": frame.get("env_class", "rain"),
                "n_frames": n,
                "raw_log_rms": raw_iqa["log_rms_contrast"],
                "raw_pct_sat": raw_iqa["pct_saturated"],
                "out_log_rms": out_iqa["log_rms_contrast"],
                "out_pct_sat": out_iqa["pct_saturated"],
                "out_pct_crushed": out_iqa["pct_crushed"],
                "out_entropy": out_iqa["hist_entropy"],
                "delta_log_rms": round(out_iqa["log_rms_contrast"] - raw_iqa["log_rms_contrast"], 5),
                "proc_ms": round(elapsed_ms, 2),
                "surrogate_mode": "pseudo_sequence_noise_jitter",
            })

    return rows


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--n-list", default=",".join(str(n) for n in DEFAULT_N_LIST))
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    n_list = [int(n) for n in args.n_list.split(",")]

    if not args.manifest.exists():
        print(f"THESIS_SKIP: manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(0)

    rain_frames = _load_rain_frames(args.manifest)
    if not rain_frames:
        print(f"THESIS_SKIP: no rain-class frames in {args.manifest}", file=sys.stderr)
        sys.exit(0)

    print(f"Rain median sweep: {len(rain_frames)} rain frames × N∈{n_list}")
    rows = run_sweep(rain_frames, n_list)
    if not rows:
        print("No rows produced.", file=sys.stderr)
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {args.out}  ({len(rows)} rows)")

    # Summary
    for n in n_list:
        sub = [r for r in rows if r["n_frames"] == n]
        mean_lrms = sum(r["delta_log_rms"] for r in sub) / len(sub)
        mean_ms = sum(r["proc_ms"] for r in sub) / len(sub)
        print(f"  N={n}: mean Δlog_rms={mean_lrms:+.4f}  mean_ms={mean_ms:.1f}")


if __name__ == "__main__":
    main()
