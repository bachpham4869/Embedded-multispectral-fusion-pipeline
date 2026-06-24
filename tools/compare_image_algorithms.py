#!/usr/bin/env python3
"""Compare current NIR/image processing against raw, simple, and legacy baselines."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from fusion_eval_manifest import PROCESSING_BUCKET_NAMES, markdown_table, run_manifest, write_csv, write_json, write_markdown  # type: ignore[import]
from fusion_eval_metrics import compute_image_metrics, summarize_metric_rows  # type: ignore[import]
from smartbinocular.nir_pipeline import (  # type: ignore[import]
    HybridNIREnhancer,
    RainTemporalMedian,
    nir_anti_glare_bgr,
    nir_dehaze_lite,
    nir_nir_night_clahe,
    nir_transition_blend,
)


ENV_TO_BUCKET = {
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

METRICS = [
    "entropy",
    "rms_contrast",
    "log_rms_contrast",
    "laplacian_variance",
    "tenengrad",
    "pct_dark_clipped",
    "pct_highlight_saturated",
    "noise_proxy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare current image processing against raw, CLAHE/gamma, and legacy baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/eval/nir_val/manifest_v2.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/fusion_eval/image_algorithms"))
    parser.add_argument("--summary", type=Path, default=Path("docs/tables/fusion/image_processing_algorithm_comparison.md"))
    parser.add_argument("--max-images", type=int, default=0, help="0 means all images")
    return parser.parse_args()


def _load_manifest(path: Path, max_images: int) -> list[dict[str, str]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("path") and Path(row["path"]).exists():
                rows.append(row)
            if max_images > 0 and len(rows) >= max_images:
                break
    return rows


def _run_current(bgr: np.ndarray, env_class: str, enhancer: HybridNIREnhancer, rain: RainTemporalMedian) -> tuple[str, np.ndarray]:
    bucket = ENV_TO_BUCKET.get(env_class, "A")
    if bucket == "A":
        enhancer.reset()
        return PROCESSING_BUCKET_NAMES[bucket], enhancer.process(bgr)
    if bucket == "B":
        return PROCESSING_BUCKET_NAMES[bucket], nir_nir_night_clahe(bgr)
    if bucket == "C":
        return PROCESSING_BUCKET_NAMES[bucket], nir_anti_glare_bgr(bgr)
    if bucket == "D":
        return PROCESSING_BUCKET_NAMES[bucket], nir_dehaze_lite(bgr)
    if bucket == "E":
        rain.reset()
        rain.process(bgr)
        rain.process(bgr)
        return PROCESSING_BUCKET_NAMES[bucket], rain.process(bgr)
    if bucket == "F":
        ema = float(np.mean(bgr[:, :, 1])) / 255.0
        return PROCESSING_BUCKET_NAMES[bucket], nir_transition_blend(bgr, enhancer, ema)
    return "raw_passthrough", bgr


def _clahe_baseline(bgr: np.ndarray) -> np.ndarray:
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    out = clahe.apply(gray)
    return cv.cvtColor(out, cv.COLOR_GRAY2BGR)


def _gamma_baseline(bgr: np.ndarray, gamma: float = 0.65) -> np.ndarray:
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv.LUT(bgr, lut)


def _load_legacy_enhancer():
    path = ROOT / "legacy/py/final_fusion.py"
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("legacy_final_fusion_eval", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.HybridNIREnhancer()
    except Exception:
        return None


def _legacy_output(bgr: np.ndarray, legacy) -> np.ndarray | None:
    if legacy is None:
        return None
    try:
        return legacy.process(bgr)
    except Exception:
        return None


def _summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("algorithm", "")),
            str(row.get("baseline_algorithm", "")),
            str(row.get("bucket_or_condition", "")),
            str(row.get("metric", "")),
        )
        grouped[key].append(row)
    out = []
    for (algorithm, baseline, condition, metric), values in sorted(grouped.items()):
        s = summarize_metric_rows(values)
        out.append(
            {
                "algorithm": algorithm,
                "baseline_algorithm": baseline,
                "evidence_label": "proxy",
                "bucket_or_condition": condition,
                "metric": metric,
                **{k: "" if v is None else round(float(v), 6) for k, v in s.items()},
            }
        )
    return out


def main() -> None:
    args = parse_args()
    manifest = (ROOT / args.manifest).resolve() if not args.manifest.is_absolute() else args.manifest
    out_dir = (ROOT / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    summary_path = (ROOT / args.summary).resolve() if not args.summary.is_absolute() else args.summary
    frames = _load_manifest(manifest, args.max_images)
    enhancer = HybridNIREnhancer()
    rain = RainTemporalMedian()
    legacy = _load_legacy_enhancer()

    per_algorithm: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    for frame in frames:
        path = Path(frame["path"])
        bgr = cv.imread(str(path), cv.IMREAD_COLOR)
        if bgr is None:
            continue
        env_class = frame.get("env_class", "unknown")
        t0 = time.perf_counter()
        bucket_name, current = _run_current(bgr, env_class, enhancer, rain)
        current_ms = (time.perf_counter() - t0) * 1000.0
        algorithms = {
            "raw_baseline": bgr,
            "clahe_baseline": _clahe_baseline(bgr),
            "gamma_baseline": _gamma_baseline(bgr),
        }
        legacy_out = _legacy_output(bgr, legacy)
        if legacy_out is not None:
            algorithms["legacy_pre_optimization"] = legacy_out
        current_metrics = compute_image_metrics(current)
        per_algorithm.append(
            {
                "image_path": str(path),
                "algorithm": "current_processing",
                "evidence_label": "proxy",
                "bucket_or_condition": bucket_name,
                "source_or_session": frame.get("source_dir", ""),
                "proc_ms": round(current_ms, 6),
                **{k: round(float(v), 6) for k, v in current_metrics.items() if v is not None},
            }
        )
        for baseline_name, baseline_img in algorithms.items():
            baseline_metrics = compute_image_metrics(baseline_img)
            for metric in METRICS:
                comparison_rows.append(
                    {
                        "image_path": str(path),
                        "algorithm": "current_processing",
                        "baseline_algorithm": baseline_name,
                        "evidence_label": "proxy",
                        "bucket_or_condition": bucket_name,
                        "source_or_session": frame.get("source_dir", ""),
                        "metric": metric,
                        "value": round(float(current_metrics[metric]), 6),
                        "baseline_value": round(float(baseline_metrics[metric]), 6),
                    }
                )
    summary_rows = _summary_rows(comparison_rows)
    write_csv(out_dir / "image_processing_per_algorithm.csv", per_algorithm)
    write_csv(out_dir / "image_processing_comparison_metrics.csv", comparison_rows)
    write_csv(out_dir / "image_processing_algorithm_comparison.csv", summary_rows)
    columns = [
        "algorithm",
        "baseline_algorithm",
        "evidence_label",
        "bucket_or_condition",
        "metric",
        "n",
        "mean",
        "median",
        "std",
        "p25",
        "p75",
        "p95",
        "delta_current_minus_baseline",
        "win_rate_current_vs_baseline",
    ]
    body = (
        "All rows use Tier 2/3 offline proxy evidence from still images. "
        "They compare current processing against raw, CLAHE/gamma, and legacy/pre-optimization baselines; "
        "they are not live NIR/RPi proof.\n\n"
    )
    body += markdown_table(summary_rows[:240], columns)
    write_markdown(summary_path, "Image Processing Algorithm Comparison", body)
    write_json(
        out_dir / "image_algorithm_run_manifest.json",
        run_manifest(
            "python3 tools/compare_image_algorithms.py",
            [str(manifest)],
            {"max_images": args.max_images, "legacy_available": legacy is not None},
        ),
    )
    print(f"images={len(frames)} comparison_rows={len(comparison_rows)} legacy_available={legacy is not None}")


if __name__ == "__main__":
    main()
