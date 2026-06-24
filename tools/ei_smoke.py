#!/usr/bin/env python3
"""Verifies that tflite_runtime can load and run the person_in_dark FOMO model.

Usage:
    python tools/ei_smoke.py [--tflite PATH] [--fixture PATH] [--runs N]

Exit 0 on success, non-zero on any failure.
Must pass (exit 0) before any pipeline wiring is committed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_DEFAULT_TFLITE = "models/ei/person_in_dark_fomo_int8.tflite"

_FIXTURE_CANDIDATES = [
    "tests/data/test_frame_640x480.jpg",
    "tests/data/test_frame.jpg",
]


def _load_bgr_or_synthetic(fixture_path: str | None) -> np.ndarray:
    """Return a 640×480 BGR uint8 frame from a file, or a checkerboard if no fixture."""
    if fixture_path and os.path.isfile(fixture_path):
        import cv2
        img = cv2.imread(fixture_path)
        if img is not None and img.size > 0:
            return img
    bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    tile = 80
    for r in range(0, 480, tile):
        for c in range(0, 640, tile):
            val = 200 if (r // tile + c // tile) % 2 == 0 else 40
            bgr[r : r + tile, c : c + tile] = val
    return bgr


def _center_crop_and_resize(bgr: np.ndarray, size: int = 128) -> np.ndarray:
    """Center-crop to square, then resize to size×size with INTER_AREA."""
    import cv2
    h, w = bgr.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return cv2.resize(bgr[y0 : y0 + s, x0 : x0 + s], (size, size), interpolation=cv2.INTER_AREA)


def _quantize_int8(rgb_u8: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Quantize uint8 RGB [0..255] to int8 using model input quantization params."""
    q = np.round(rgb_u8.astype(np.float32) / 255.0 / scale + zero_point)
    return np.clip(q, -128, 127).astype(np.int8)


def main() -> None:
    parser = argparse.ArgumentParser(description="FOMO person_in_dark .tflite smoke test")
    parser.add_argument("--tflite", default=None, help=f"Path to .tflite (default: {_DEFAULT_TFLITE})")
    parser.add_argument("--fixture", default=None, help="BGR image to use as input (optional)")
    parser.add_argument("--runs", type=int, default=20, help="Inference runs for latency measurement")
    args = parser.parse_args()

    tflite_path = args.tflite or _DEFAULT_TFLITE

    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
    except ImportError:
        print("[FAIL] tflite_runtime not installed. Run: pip install tflite-runtime", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(tflite_path):
        print(f"[FAIL] .tflite not found: {tflite_path}", file=sys.stderr)
        sys.exit(1)

    t_load = time.perf_counter()
    interp = Interpreter(model_path=tflite_path, num_threads=2)
    interp.allocate_tensors()
    load_ms = (time.perf_counter() - t_load) * 1000.0
    print(f"[OK]   Interpreter loaded + allocate_tensors: {load_ms:.0f} ms")

    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]

    print(f"[INFO] input  shape={inp_d['shape'].tolist()}  dtype={inp_d['dtype'].__name__}")
    print(f"[INFO] output shape={out_d['shape'].tolist()}  dtype={out_d['dtype'].__name__}")

    assert inp_d["shape"].tolist() == [1, 128, 128, 3], f"Unexpected input shape: {inp_d['shape'].tolist()}"
    assert inp_d["dtype"] == np.int8, f"Unexpected input dtype: {inp_d['dtype']}"

    in_scale, in_zp = float(inp_d["quantization"][0]), int(inp_d["quantization"][1])
    out_scale, out_zp = float(out_d["quantization"][0]), int(out_d["quantization"][1])
    print(f"[INFO] input  quantization: scale={in_scale:.6f}  zero_point={in_zp}")
    print(f"[INFO] output quantization: scale={out_scale:.6f}  zero_point={out_zp}")

    import cv2

    fixture_path = args.fixture
    if fixture_path is None:
        for c in _FIXTURE_CANDIDATES:
            if os.path.isfile(c):
                fixture_path = c
                break

    bgr = _load_bgr_or_synthetic(fixture_path)
    src = "fixture" if (fixture_path and os.path.isfile(str(fixture_path))) else "synthetic checkerboard"
    print(f"[INFO] Input source: {src}  shape={bgr.shape}")

    rgb128 = cv2.cvtColor(_center_crop_and_resize(bgr), cv2.COLOR_BGR2RGB)
    inp_tensor = _quantize_int8(rgb128, in_scale, in_zp)[np.newaxis] if in_scale > 0.0 else (rgb128.astype(np.int8) - 128)[np.newaxis]

    latencies: list[float] = []
    for _ in range(args.runs):
        interp.set_tensor(inp_d["index"], inp_tensor)
        t0 = time.perf_counter()
        interp.invoke()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    median_ms = float(np.median(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    print(f"[PERF] invoke() {args.runs} runs: median={median_ms:.1f} ms  p95={p95_ms:.1f} ms")

    raw_out = interp.get_tensor(out_d["index"])
    gh, gw, C = int(raw_out.shape[1]), int(raw_out.shape[2]), int(raw_out.shape[3])
    print(f"[INFO] output grid: {gh}×{gw}  C={C}")

    probs = (raw_out.astype(np.float32) - out_zp) * out_scale
    if C >= 2:
        shifted = probs - probs.max(axis=-1, keepdims=True)
        exp_ = np.exp(shifted)
        person_p = (exp_ / exp_.sum(axis=-1, keepdims=True))[0, :, :, 1]
        print("[INFO] postproc: C≥2 → softmax → channel 1 (person)")
    else:
        person_p = 1.0 / (1.0 + np.exp(-probs[0, :, :, 0]))
        print("[INFO] postproc: C=1 → sigmoid")

    threshold = 0.8
    hits = np.argwhere(person_p >= threshold)
    print(f"[INFO] person_p: min={person_p.min():.3f}  max={person_p.max():.3f}  mean={person_p.mean():.4f}")
    print(f"[INFO] detections above {threshold}: {len(hits)}")
    for row, col in hits:
        print(f"       cell ({row},{col})  score={person_p[row, col]:.3f}  center=({(col+0.5)/gw:.3f},{(row+0.5)/gh:.3f})")

    print("[OK]   Smoke test passed.")


if __name__ == "__main__":
    main()
