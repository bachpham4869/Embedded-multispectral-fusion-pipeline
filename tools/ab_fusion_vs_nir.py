"""ab_fusion_vs_nir.py — A/B offline comparison: NIR-only vs dummy-thermal fusion.

PROXY NOTE: The thermal arm uses a uniform mid-gray dummy image (absent real MI48
data). This isolates the effect of the alpha-blending step on IQA proxies but does
NOT represent real fusion benefit. Real A/B requires on-device synchronized captures.
Label all outputs PROXY in thesis and spot-check vs on-device before treating as
quantitative evidence.

For each frame from the manifest:
  Arm A (NIR-only): apply env-class-appropriate bucket processor
  Arm B (fused):    arm-A output blended with dummy thermal at production alpha=0.55

Outputs:
  ab_fusion_vs_nir_proxy.csv — per-frame IQA for both arms + delta columns

Usage:
    python tools/ab_fusion_vs_nir.py
    python tools/ab_fusion_vs_nir.py \\
        --manifest data/eval/nir_val/manifest_v2.csv \\
        --alpha 0.55 \\
        --out docs/thesis_eval/fusion/tables/ab_fusion_vs_nir_proxy.csv
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
from smartbinocular.nir_pipeline import (
    HybridNIREnhancer,
    RainTemporalMedian,
    nir_anti_glare_bgr,
    nir_dehaze_lite,
    nir_nir_night_clahe,
    nir_transition_blend,
)

DEFAULT_MANIFEST = ROOT / "data/eval/nir_val/manifest_v2.csv"
DEFAULT_OUT = ROOT / "docs/thesis_eval/fusion/tables/ab_fusion_vs_nir_proxy.csv"
PRODUCTION_ALPHA = 0.55

# ENV class → bucket letter (mirrors OPTICAL_BUCKET_DISPATCH)
_ENV_TO_BUCKET = {
    "night_clear": "A",
    "normal_night": "A",
    "nir_night": "B",
    "fog": "D",
    "rain": "E",
    "glare": "C",
    "backlight": "C",
    "transition": "F",
    "normal_day": "C",
    "mixed_edge": "A",
}


def _load_manifest(path: Path) -> List[dict]:
    rows = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            p = row.get("path", "")
            if p and Path(p).exists():
                rows.append(row)
    return rows


def _run_bucket(bucket: str, bgr: np.ndarray, enhancer: HybridNIREnhancer, rain: RainTemporalMedian) -> np.ndarray:
    if bucket == "A":
        enhancer.reset()
        return enhancer.process(bgr)
    if bucket == "B":
        return nir_nir_night_clahe(bgr)
    if bucket == "C":
        return nir_anti_glare_bgr(bgr)
    if bucket == "D":
        return nir_dehaze_lite(bgr)
    if bucket == "E":
        rain.reset()
        rain.process(bgr)
        rain.process(bgr)
        return rain.process(bgr)
    if bucket == "F":
        green = bgr[:, :, 1].astype(np.float32)
        ema = float(green.mean()) / 255.0
        return nir_transition_blend(bgr, enhancer, ema)
    return bgr


def run_ab(manifest: List[dict], alpha: float) -> List[dict]:
    enhancer = HybridNIREnhancer()
    rain = RainTemporalMedian()
    rows = []
    n = len(manifest)

    for i, frame in enumerate(manifest):
        img_path = frame["path"]
        env_cls = frame.get("env_class", "unknown")
        bucket = _ENV_TO_BUCKET.get(env_cls, "A")

        bgr = cv.imread(img_path, cv.IMREAD_COLOR)
        if bgr is None:
            continue
        bgr_big = cv.resize(bgr, (THUMB_W := 320, THUMB_H := 240))

        # Arm A — NIR only
        t0 = time.perf_counter()
        nir_out = _run_bucket(bucket, bgr_big, enhancer, rain)
        nir_ms = (time.perf_counter() - t0) * 1000
        nir_iqa = compute_iqa_metrics(cv.cvtColor(nir_out, cv.COLOR_BGR2GRAY))

        # Arm B — fused with dummy thermal
        h, w = nir_out.shape[:2]
        thermal = np.full((h, w, 3), 127, dtype=np.uint8)
        fused = cv.addWeighted(nir_out, 1.0 - alpha, thermal, alpha, 0)
        fused_iqa = compute_iqa_metrics(cv.cvtColor(fused, cv.COLOR_BGR2GRAY))

        delta_log_rms = round(fused_iqa["log_rms_contrast"] - nir_iqa["log_rms_contrast"], 5)
        delta_entropy = round(fused_iqa["hist_entropy"] - nir_iqa["hist_entropy"], 5)
        delta_pct_sat = round(fused_iqa["pct_saturated"] - nir_iqa["pct_saturated"], 6)

        rows.append({
            "frame": Path(img_path).name,
            "env_class": env_cls,
            "bucket": bucket,
            "alpha": alpha,
            "nir_log_rms": nir_iqa["log_rms_contrast"],
            "nir_entropy": nir_iqa["hist_entropy"],
            "nir_pct_sat": nir_iqa["pct_saturated"],
            "fused_log_rms": fused_iqa["log_rms_contrast"],
            "fused_entropy": fused_iqa["hist_entropy"],
            "fused_pct_sat": fused_iqa["pct_saturated"],
            "delta_log_rms": delta_log_rms,
            "delta_entropy": delta_entropy,
            "delta_pct_sat": delta_pct_sat,
            "nir_proc_ms": round(nir_ms, 2),
            "proxy_note": "dummy_thermal_uniform_gray",
        })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n} frames...")

    return rows


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--alpha", type=float, default=PRODUCTION_ALPHA)
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

    print(f"A/B comparison: {len(manifest)} frames, alpha={args.alpha}")
    print("PROXY NOTE: thermal is dummy uniform-gray. Label outputs PROXY in thesis.")

    rows = run_ab(manifest, args.alpha)
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

    # Brief summary
    deltas = [r["delta_log_rms"] for r in rows]
    print(f"Delta log_rms: mean={sum(deltas)/len(deltas):.4f}, "
          f"min={min(deltas):.4f}, max={max(deltas):.4f}")
    print("(Expected: negative — dummy thermal pulls contrast down; will improve with real thermal data)")


if __name__ == "__main__":
    main()
