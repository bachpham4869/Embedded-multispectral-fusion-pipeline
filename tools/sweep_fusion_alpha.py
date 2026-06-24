"""sweep_fusion_alpha.py — Proxy sweep of fusion_alpha on NIR frames with dummy thermal.

PROXY NOTE: This tool uses a dummy thermal channel (uniform-gray colormap) because
no aligned NIR+thermal clip pairs are available offline. The sweep measures how
alpha changes NIR-dominated IQA metrics, NOT true fusion performance. A real fusion
sweep requires synchronized on-device captures. This artifact must be labeled
PROXY in the thesis and spot-checked against on-device captures.

For each (frame, alpha) pair:
  nir_out  = bucket-A enhanced NIR frame
  thermal  = uniform mid-gray (127) resized to match — simulates absent thermal
  fused    = cv2.addWeighted(nir_out, 1-alpha, thermal_colored, alpha, 0)
  IQA      = compute_iqa_metrics(cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY))

Usage:
    python tools/sweep_fusion_alpha.py
    python tools/sweep_fusion_alpha.py \\
        --manifest data/eval/nir_val/manifest_v2.csv \\
        --alphas 0.30,0.40,0.50,0.55,0.60,0.70,0.80 \\
        --out docs/thesis_eval/fusion/tables/fusion_alpha_sweep_proxy.csv
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
from smartbinocular.nir_pipeline import HybridNIREnhancer

DEFAULT_ALPHAS = [0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80]
DEFAULT_MANIFEST = ROOT / "data/eval/nir_val/manifest_v2.csv"
DEFAULT_OUT = ROOT / "docs/thesis_eval/fusion/tables/fusion_alpha_sweep_proxy.csv"


def _load_manifest(path: Path) -> List[dict]:
    rows = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            p = row.get("path", "")
            if p and Path(p).exists():
                rows.append(row)
    return rows


def _make_dummy_thermal(h: int, w: int) -> np.ndarray:
    """Uniform mid-gray BGR image simulating absent thermal data."""
    return np.full((h, w, 3), 127, dtype=np.uint8)


def run_sweep(manifest: List[dict], alphas: List[float]) -> List[dict]:
    enhancer = HybridNIREnhancer()
    rows = []
    n = len(manifest)
    for i, frame in enumerate(manifest):
        img_path = frame["path"]
        env_cls = frame.get("env_class", "unknown")
        bgr = cv.imread(img_path, cv.IMREAD_COLOR)
        if bgr is None:
            continue

        # Process through bucket A for the "NIR" arm
        enhancer.reset()
        t0 = time.perf_counter()
        nir_out = enhancer.process(bgr)
        nir_ms = (time.perf_counter() - t0) * 1000

        h, w = nir_out.shape[:2]
        thermal = _make_dummy_thermal(h, w)

        # Raw IQA for reference
        raw_iqa = compute_iqa_metrics(cv.cvtColor(bgr, cv.COLOR_BGR2GRAY))
        nir_iqa = compute_iqa_metrics(cv.cvtColor(nir_out, cv.COLOR_BGR2GRAY))

        for alpha in alphas:
            fused = cv.addWeighted(nir_out, 1.0 - alpha, thermal, alpha, 0)
            fused_iqa = compute_iqa_metrics(cv.cvtColor(fused, cv.COLOR_BGR2GRAY))
            rows.append({
                "frame": Path(img_path).name,
                "env_class": env_cls,
                "alpha": alpha,
                "raw_log_rms": raw_iqa["log_rms_contrast"],
                "raw_pct_sat": raw_iqa["pct_saturated"],
                "nir_log_rms": nir_iqa["log_rms_contrast"],
                "nir_pct_sat": nir_iqa["pct_saturated"],
                "fused_log_rms": fused_iqa["log_rms_contrast"],
                "fused_pct_sat": fused_iqa["pct_saturated"],
                "fused_pct_crushed": fused_iqa["pct_crushed"],
                "fused_entropy": fused_iqa["hist_entropy"],
                "nir_proc_ms": round(nir_ms, 2),
                "proxy_note": "dummy_thermal_uniform_gray",
            })

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n} frames processed...")

    return rows


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--alphas", default=",".join(str(a) for a in DEFAULT_ALPHAS),
                   help="Comma-separated alpha values (default: 0.30,0.40,0.50,0.55,0.60,0.70,0.80)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.manifest.exists():
        print(f"THESIS_SKIP: manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(0)

    manifest = _load_manifest(args.manifest)
    if not manifest:
        print(f"THESIS_SKIP: no readable frames in {args.manifest}", file=sys.stderr)
        sys.exit(0)

    alphas = [float(a) for a in args.alphas.split(",")]
    print(f"Sweep: {len(manifest)} frames × {len(alphas)} alpha values")
    print(f"PROXY NOTE: thermal channel is dummy uniform-gray. Label outputs PROXY in thesis.")

    rows = run_sweep(manifest, alphas)
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


if __name__ == "__main__":
    main()
