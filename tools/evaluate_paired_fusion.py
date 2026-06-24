#!/usr/bin/env python3
"""Evaluate strict paired NIR/thermal/fusion evidence from paired manifests."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2 as cv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fusion_eval_manifest import markdown_table, read_csv_rows, write_csv, write_json, write_markdown  # type: ignore[import]
from tools.fusion_eval_metrics import (  # type: ignore[import]
    compute_fusion_metrics,
    compute_image_metrics,
    detect_failure_cases,
    ensure_gray_u8,
    foreground_contrast,
    summarize_metric_rows,
)


STRICT_TIERS = {"frame_strict", "time_strict_100ms", "protocol_strict_1s"}
DIRECT_TASK_METRICS = {"foreground_contrast_gain", "thermal_saliency_retention", "nir_detail_retention"}
PROXY_FUSION_METRICS = {
    "mutual_information_nir_fusion",
    "mutual_information_thermal_fusion",
    "normalized_mutual_information_nir_fusion",
    "normalized_mutual_information_thermal_fusion",
    "ssim_fusion_nir_proxy",
    "ssim_fusion_thermal_proxy",
    "qabf_edge_proxy",
}
NO_REFERENCE_METRICS = {
    "brightness_mean",
    "brightness_p5",
    "brightness_p50",
    "brightness_p95",
    "rms_contrast",
    "log_rms_contrast",
    "michelson_contrast",
    "entropy",
    "laplacian_variance",
    "tenengrad",
    "edge_density",
    "noise_proxy",
    "spatial_frequency",
    "average_gradient",
    "pct_dark_clipped",
    "pct_highlight_saturated",
    "dynamic_range",
    "mask_area_ratio",
    "connected_components_count",
    "foreground_background_contrast",
}
REQUIRED_BUCKETS = [
    "night_hybrid_enhance",
    "nir_mono_clahe",
    "highlight_tone_map",
    "fog_dehaze_lite",
    "rain_temporal_median",
    "dawn_dusk_blend",
]
SUMMARY_COLUMNS = [
    "algorithm",
    "baseline_algorithm",
    "metric",
    "metric_tier",
    "pairing_tier",
    "evidence_label",
    "input_data_type",
    "thermal_modality",
    "fusion_source",
    "processing_bucket",
    "processing_bucket_source",
    "source_or_session",
    "n",
    "mean",
    "median",
    "std",
    "p25",
    "p75",
    "p95",
    "bootstrap_ci95_low",
    "bootstrap_ci95_high",
    "delta_current_minus_baseline",
    "win_rate_current_vs_baseline",
    "evidence_label_distribution",
    "caveat",
]


@dataclass(frozen=True)
class FusionCandidate:
    algorithm: str
    image: np.ndarray
    fusion_source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate strict paired image, thermal, and generated/captured fusion metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/paired_eval/strict_paired_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/paired_eval"))
    parser.add_argument("--fusion-tables-dir", type=Path, default=Path("docs/tables/fusion"))
    parser.add_argument("--paired-docs-dir", type=Path, default=Path("docs/paired"))
    parser.add_argument("--fusion-docs-dir", type=Path, default=Path("docs/fusion"))
    parser.add_argument("--figures-dir", type=Path, default=Path("docs/figures/fusion"))
    parser.add_argument("--max-pairs", type=int, default=0, help="0 means evaluate all manifest rows.")
    parser.add_argument("--grid-samples", type=int, default=6)
    parser.add_argument("--failure-grid-cases", type=int, default=4)
    parser.add_argument("--metric-max-dim", type=int, default=320, help="Resize longest side for metric computation; 0 keeps original size.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _rel(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    except Exception:
        return "unknown"


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def _split_frame_path(path_text: str) -> tuple[str, int | None]:
    if "#frame=" not in path_text:
        return path_text, None
    path, frame = path_text.split("#frame=", 1)
    try:
        return path, int(frame)
    except ValueError:
        return path, None


def _read_video_frame(path: Path, frame_index: int) -> np.ndarray | None:
    cap = cv.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


class SequentialFrameReader:
    """Read mostly-monotonic video frame requests without reopening videos."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self._caps: dict[Path, cv.VideoCapture] = {}
        self._positions: dict[Path, int] = {}

    def close(self) -> None:
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()
        self._positions.clear()

    def _video_frame(self, path: Path, frame_index: int) -> np.ndarray | None:
        cap = self._caps.get(path)
        if cap is None:
            cap = cv.VideoCapture(str(path))
            if not cap.isOpened():
                return None
            self._caps[path] = cap
            self._positions[path] = -1
        if frame_index <= self._positions.get(path, -1):
            cap.set(cv.CAP_PROP_POS_FRAMES, frame_index)
            self._positions[path] = frame_index - 1
        frame = None
        while self._positions[path] < frame_index:
            ok, frame = cap.read()
            if not ok:
                return None
            self._positions[path] += 1
        return frame.copy() if frame is not None else None

    def frame_from_manifest_path(self, path_text: str, fallback_index: int) -> np.ndarray | None:
        if not path_text:
            return None
        path_part, frame_index = _split_frame_path(path_text)
        path = Path(path_part)
        if not path.is_absolute():
            path = self.repo_root / path
        index = fallback_index if frame_index is None else frame_index
        if path.suffix.lower() in {".mp4", ".avi"}:
            return self._video_frame(path, index)
        image = cv.imread(str(path), cv.IMREAD_UNCHANGED)
        return image


def _frame_from_manifest_path(repo_root: Path, path_text: str, fallback_index: int) -> np.ndarray | None:
    if not path_text:
        return None
    path_part, frame_index = _split_frame_path(path_text)
    path = Path(path_part)
    if not path.is_absolute():
        path = repo_root / path
    index = fallback_index if frame_index is None else frame_index
    if path.suffix.lower() in {".mp4", ".avi"}:
        return _read_video_frame(path, index)
    image = cv.imread(str(path), cv.IMREAD_UNCHANGED)
    return image


def _resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if image.shape[:2] == reference.shape[:2]:
        return image
    return cv.resize(image, (reference.shape[1], reference.shape[0]), interpolation=cv.INTER_AREA)


def _resize_max_dim(image: np.ndarray, max_dim: int) -> np.ndarray:
    if max_dim <= 0:
        return image
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image
    scale = max_dim / float(longest)
    return cv.resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv.INTER_AREA)


def _to_bgr(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return cv.cvtColor(ensure_gray_u8(arr), cv.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        return cv.cvtColor(arr, cv.COLOR_BGRA2BGR)
    return arr.astype(np.uint8, copy=False)


def _thermal_mask(thermal: np.ndarray) -> np.ndarray:
    gray = ensure_gray_u8(thermal)
    threshold = max(10.0, float(np.percentile(gray, 85)))
    mask = (gray >= threshold).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    return cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)


def _thermal_colormap(thermal: np.ndarray, reference: np.ndarray) -> np.ndarray:
    gray = ensure_gray_u8(_resize_like(thermal, reference))
    return cv.applyColorMap(gray, cv.COLORMAP_JET)


def _laplacian_pyramid_fusion(nir: np.ndarray, thermal_color: np.ndarray, levels: int = 3) -> np.ndarray:
    a = _to_bgr(nir).astype(np.float32)
    b = _resize_like(_to_bgr(thermal_color), a).astype(np.float32)
    gp_a = [a]
    gp_b = [b]
    for _ in range(levels):
        if min(gp_a[-1].shape[:2]) < 16:
            break
        gp_a.append(cv.pyrDown(gp_a[-1]))
        gp_b.append(cv.pyrDown(gp_b[-1]))
    lp_a = [gp_a[-1]]
    lp_b = [gp_b[-1]]
    for i in range(len(gp_a) - 1, 0, -1):
        size = (gp_a[i - 1].shape[1], gp_a[i - 1].shape[0])
        lp_a.append(gp_a[i - 1] - cv.pyrUp(gp_a[i], dstsize=size))
        lp_b.append(gp_b[i - 1] - cv.pyrUp(gp_b[i], dstsize=size))
    blended = []
    for la, lb in zip(lp_a, lp_b):
        blended.append(0.65 * la + 0.35 * lb)
    result = blended[0]
    for level in blended[1:]:
        result = cv.pyrUp(result, dstsize=(level.shape[1], level.shape[0])) + level
    return np.clip(result, 0, 255).astype(np.uint8)


def fusion_candidates(
    nir: np.ndarray,
    thermal: np.ndarray,
    *,
    captured_fusion: np.ndarray | None = None,
) -> dict[str, FusionCandidate]:
    nir_bgr = _to_bgr(nir)
    thermal_resized = _resize_like(_to_bgr(thermal), nir_bgr)
    thermal_color = _thermal_colormap(thermal_resized, nir_bgr)
    mask = _thermal_mask(thermal_resized)
    mask_3 = (mask.astype(np.float32) / 255.0)[:, :, None]
    thermal_gray = ensure_gray_u8(thermal_resized).astype(np.float32) / 255.0
    weight = np.clip(thermal_gray, 0.15, 0.65)[:, :, None]

    alpha = cv.addWeighted(nir_bgr, 0.7, thermal_color, 0.3, 0.0)
    overlay = np.clip(nir_bgr.astype(np.float32) * (1.0 - 0.45 * mask_3) + thermal_color.astype(np.float32) * (0.45 * mask_3), 0, 255).astype(np.uint8)
    weighted = np.clip(nir_bgr.astype(np.float32) * (1.0 - weight) + thermal_color.astype(np.float32) * weight, 0, 255).astype(np.uint8)
    edges = cv.Canny(ensure_gray_u8(thermal_resized), 80, 160)
    legacy = nir_bgr.copy()
    legacy[edges > 0] = (0, 255, 255)
    generated = "paired_generated_fusion"

    candidates = {
        "nir_only_baseline": FusionCandidate("nir_only_baseline", nir_bgr, generated),
        "thermal_heatmap_only": FusionCandidate("thermal_heatmap_only", thermal_color, generated),
        "alpha_blend_baseline": FusionCandidate("alpha_blend_baseline", alpha, generated),
        "foreground_mask_overlay": FusionCandidate("foreground_mask_overlay", overlay, generated),
        "mask_weighted_blend": FusionCandidate("mask_weighted_blend", weighted, generated),
        "legacy_gradient_overlay": FusionCandidate("legacy_gradient_overlay", legacy, generated),
        "laplacian_pyramid_fusion": FusionCandidate("laplacian_pyramid_fusion", _laplacian_pyramid_fusion(nir_bgr, thermal_color), generated),
    }
    if captured_fusion is not None:
        candidates["foreground_mask_overlay"] = FusionCandidate(
            "foreground_mask_overlay",
            _resize_like(_to_bgr(captured_fusion), nir_bgr),
            "paired_captured_fusion",
        )
    return candidates


def determine_metric_tier(metric: str, pairing_tier: str, fusion_source: str, evidence_label: str) -> str:
    if evidence_label != "real_paired" or pairing_tier not in STRICT_TIERS:
        return "Tier 3"
    if metric in DIRECT_TASK_METRICS:
        return "Tier 1" if fusion_source == "paired_captured_fusion" else "Tier 3"
    if metric in PROXY_FUSION_METRICS:
        return "Tier 3"
    if metric in NO_REFERENCE_METRICS:
        return "Tier 2"
    return "Tier 3"


def failure_tier_for_type(failure_type: str, fusion_source: str, pairing_tier: str, evidence_label: str) -> str:
    if evidence_label != "real_paired" or pairing_tier not in STRICT_TIERS:
        return "Tier 3"
    if failure_type in {"fusion_target_faded", "mask_wrong_region", "alignment_drift"}:
        return "Tier 1" if fusion_source == "paired_captured_fusion" else "Tier 3"
    if failure_type in {"nir_detail_occlusion"}:
        return "Tier 3"
    return "Tier 2"


def _metric_caveat(metric_tier: str, thermal_modality: str, fusion_source: str, input_data_type: str) -> str:
    caveats = []
    if thermal_modality == "display_heatmap_like":
        caveats.append("thermal_modality=display_heatmap_like; not raw radiometric thermal")
    if fusion_source == "paired_generated_fusion":
        caveats.append("paired_generated_fusion; not runtime-captured fusion")
    if metric_tier == "Tier 2":
        caveats.append("no-reference IQA/proxy metric; not ground-truth perceptual quality")
    elif metric_tier == "Tier 3":
        caveats.append("proxy/generated/unpaired-style metric; use as preliminary diagnostic evidence")
    else:
        caveats.append("Tier 1 only for strict paired/task-specific measured evidence")
    if "display/heatmap-like" in input_data_type:
        caveats.append("thermal video is display/heatmap-like input")
    return "; ".join(dict.fromkeys(caveats))


def _common_row_context(
    *,
    pair_id: str,
    pairing_tier: str,
    thermal_modality: str,
    input_data_type: str,
    processing_bucket: str,
    processing_bucket_source: str,
    source_or_session: str,
    fusion_source: str,
) -> dict[str, str]:
    evidence = "real_paired" if pairing_tier in STRICT_TIERS else ("proxy" if pairing_tier in {"near_paired", "weak_paired"} else "unpaired")
    return {
        "pair_id": pair_id,
        "pairing_tier": pairing_tier,
        "evidence_label": evidence,
        "input_data_type": input_data_type,
        "thermal_modality": thermal_modality,
        "processing_bucket": processing_bucket,
        "processing_bucket_source": processing_bucket_source,
        "source_or_session": source_or_session,
        "fusion_source": fusion_source,
    }


def metric_rows_for_pair(
    *,
    pair_id: str,
    nir: np.ndarray,
    thermal: np.ndarray,
    candidates: dict[str, FusionCandidate],
    pairing_tier: str,
    thermal_modality: str,
    input_data_type: str,
    processing_bucket: str,
    processing_bucket_source: str,
    source_or_session: str,
) -> list[dict[str, str]]:
    mask = _thermal_mask(thermal)
    baseline_candidate = candidates["alpha_blend_baseline"]
    baseline_metrics = compute_fusion_metrics(nir, thermal, baseline_candidate.image, fg_mask=mask)
    rows: list[dict[str, str]] = []
    for name, candidate in candidates.items():
        metrics = compute_fusion_metrics(nir, thermal, candidate.image, fg_mask=mask)
        context = _common_row_context(
            pair_id=pair_id,
            pairing_tier=pairing_tier,
            thermal_modality=thermal_modality,
            input_data_type=input_data_type,
            processing_bucket=processing_bucket,
            processing_bucket_source=processing_bucket_source,
            source_or_session=source_or_session,
            fusion_source=candidate.fusion_source,
        )
        for metric, value in metrics.items():
            if value is None:
                continue
            tier = determine_metric_tier(metric, pairing_tier, candidate.fusion_source, context["evidence_label"])
            rows.append(
                {
                    **context,
                    "algorithm": name,
                    "baseline_algorithm": "alpha_blend_baseline",
                    "metric": metric,
                    "value": _fmt(value),
                    "baseline_value": _fmt(baseline_metrics.get(metric)),
                    "metric_tier": tier,
                    "caveat": _metric_caveat(tier, thermal_modality, candidate.fusion_source, input_data_type),
                }
            )
    return rows


def _image_algorithm_candidates(nir: np.ndarray) -> dict[str, np.ndarray]:
    gray = ensure_gray_u8(nir)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gamma = np.clip(((gray.astype(np.float32) / 255.0) ** 0.75) * 255.0, 0, 255).astype(np.uint8)
    tone = cv.normalize(np.minimum(gray, np.percentile(gray, 98)), None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    blur = cv.GaussianBlur(gray, (0, 0), 3.0)
    dehaze = cv.addWeighted(gray, 1.35, blur, -0.35, 0)
    blend = cv.addWeighted(clahe, 0.6, gamma, 0.4, 0)
    return {
        "raw_passthrough": gray,
        "nir_mono_clahe": clahe,
        "gamma_baseline": gamma,
        "highlight_tone_map": tone,
        "fog_dehaze_lite": dehaze,
        "night_hybrid_enhance": blend,
    }


def _long_image_metric_rows(
    *,
    pair_row: dict[str, str],
    image: np.ndarray,
) -> list[dict[str, str]]:
    candidates = _image_algorithm_candidates(image)
    baseline = compute_image_metrics(candidates["raw_passthrough"])
    rows: list[dict[str, str]] = []
    for algorithm, candidate in candidates.items():
        metrics = compute_image_metrics(candidate)
        for metric, value in metrics.items():
            tier = "Tier 2" if pair_row.get("pairing_tier") in STRICT_TIERS else "Tier 3"
            bucket_source = "algorithm_forced" if algorithm in REQUIRED_BUCKETS else "unknown"
            caveat = "offline forced bucket; not runtime bucket performance" if bucket_source == "algorithm_forced" else ""
            if pair_row.get("thermal_modality") == "display_heatmap_like":
                caveat = (caveat + "; " if caveat else "") + "thermal_modality=display_heatmap_like; not raw radiometric thermal"
            rows.append(
                {
                    "pair_id": pair_row.get("pair_id", ""),
                    "algorithm": algorithm,
                    "baseline_algorithm": "raw_passthrough",
                    "metric": metric,
                    "value": _fmt(value),
                    "baseline_value": _fmt(baseline.get(metric)),
                    "metric_tier": tier,
                    "pairing_tier": pair_row.get("pairing_tier", ""),
                    "evidence_label": pair_row.get("evidence_label", ""),
                    "input_data_type": pair_row.get("input_data_type", ""),
                    "thermal_modality": pair_row.get("thermal_modality", ""),
                    "fusion_source": "none",
                    "processing_bucket": algorithm if algorithm in REQUIRED_BUCKETS else "unknown",
                    "processing_bucket_source": bucket_source,
                    "source_or_session": pair_row.get("source_or_session", ""),
                    "caveat": caveat or "no-reference IQA metric; not ground-truth perceptual quality",
                }
            )
    return rows


def _thermal_metric_rows(pair_row: dict[str, str], thermal: np.ndarray) -> list[dict[str, str]]:
    gray = ensure_gray_u8(thermal)
    metrics = compute_image_metrics(gray)
    mask = _thermal_mask(gray)
    n_labels, labels, stats, _ = cv.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    component_areas = [int(stats[i, cv.CC_STAT_AREA]) for i in range(1, n_labels)]
    fg_contrast = foreground_contrast(gray, mask)
    metrics.update(
        {
            "dynamic_range": float(np.percentile(gray, 95) - np.percentile(gray, 5)),
            "mask_area_ratio": float(np.count_nonzero(mask)) / float(mask.size),
            "connected_components_count": float(max(0, n_labels - 1)),
            "largest_component_area": float(max(component_areas) if component_areas else 0),
            "foreground_background_contrast": fg_contrast if fg_contrast is not None else 0.0,
        }
    )
    rows = []
    for metric, value in metrics.items():
        tier = "Tier 2" if pair_row.get("pairing_tier") in STRICT_TIERS else "Tier 3"
        rows.append(
            {
                "pair_id": pair_row.get("pair_id", ""),
                "algorithm": "thermal_heatmap",
                "baseline_algorithm": "thermal_heatmap",
                "metric": metric,
                "value": _fmt(value),
                "baseline_value": _fmt(value),
                "metric_tier": tier,
                "pairing_tier": pair_row.get("pairing_tier", ""),
                "evidence_label": pair_row.get("evidence_label", ""),
                "input_data_type": pair_row.get("input_data_type", ""),
                "thermal_modality": pair_row.get("thermal_modality", ""),
                "fusion_source": "none",
                "processing_bucket": pair_row.get("processing_bucket", "unknown"),
                "processing_bucket_source": pair_row.get("processing_bucket_source", "unknown"),
                "source_or_session": pair_row.get("source_or_session", ""),
                "caveat": "thermal_modality=display_heatmap_like; not raw radiometric thermal"
                if pair_row.get("thermal_modality") == "display_heatmap_like"
                else "thermal processing metric; validate thermal units before radiometric claims",
            }
        )
    return rows


def summarize_metric_rows_by_group(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    key_fields = [
        "algorithm",
        "baseline_algorithm",
        "metric",
        "metric_tier",
        "pairing_tier",
        "input_data_type",
        "thermal_modality",
        "fusion_source",
        "processing_bucket",
        "processing_bucket_source",
        "source_or_session",
    ]
    for row in rows:
        groups[tuple(str(row.get(field, "")) for field in key_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key, values in sorted(groups.items()):
        summary = summarize_metric_rows(values, value_key="value", baseline_key="baseline_value")
        evidence_counts = Counter(str(row.get("evidence_label", "unknown")) for row in values)
        caveats = sorted({str(row.get("caveat", "")) for row in values if row.get("caveat")})
        row = {field: key[idx] for idx, field in enumerate(key_fields)}
        row.update(summary)
        row["evidence_label"] = ",".join(f"{name}:{evidence_counts[name]}" for name in sorted(evidence_counts))
        row["evidence_label_distribution"] = row["evidence_label"]
        row["caveat"] = " | ".join(caveats[:3])
        output.append(row)
    return output


def should_measure_rain_temporal(rows: list[dict[str, str]], *, min_sequence_frames: int = 5) -> bool:
    if len(rows) < min_sequence_frames:
        return False
    has_rain = any("rain" in str(row.get("env_label", "")).lower() for row in rows)
    if not has_rain:
        has_rain = any(str(row.get("processing_bucket", "")) == "rain_temporal_median" and row.get("processing_bucket_source") in {"sidecar", "metadata"} for row in rows)
    if not has_rain:
        return False
    frame_indices = sorted(int(row.get("frame_idx", "-999999")) for row in rows if str(row.get("frame_idx", "")).lstrip("-").isdigit())
    if len(frame_indices) < min_sequence_frames:
        return False
    longest = 1
    current = 1
    for prev, cur in zip(frame_indices, frame_indices[1:]):
        if cur == prev + 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest >= min_sequence_frames


def per_bucket_eval_rows(metric_rows: list[dict[str, Any]], manifest_rows: list[dict[str, str]] | None = None) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        bucket = str(row.get("processing_bucket") or "unknown")
        if bucket in REQUIRED_BUCKETS:
            by_bucket[bucket].append(row)
    for bucket in REQUIRED_BUCKETS:
        rows = by_bucket.get(bucket, [])
        if bucket == "rain_temporal_median" and not should_measure_rain_temporal(manifest_rows or []):
            output.append(
                {
                    "processing_bucket": bucket,
                    "processing_bucket_source": "unknown",
                    "status": "not measured",
                    "n": 0,
                    "metric_tier": "Tier 3",
                    "pairing_tier": "",
                    "input_data_type": "",
                    "fusion_source": "none",
                    "caveat": "rain_temporal_median requires sequence/rain evidence; not measured on this paired set.",
                }
            )
            continue
        if not rows:
            output.append(
                {
                    "processing_bucket": bucket,
                    "processing_bucket_source": "unknown",
                    "status": "missing evidence",
                    "n": 0,
                    "metric_tier": "Tier 3",
                    "pairing_tier": "",
                    "input_data_type": "",
                    "fusion_source": "none",
                    "caveat": "No paired samples or explicit metadata for this processing bucket.",
                }
            )
            continue
        source_counts = Counter(str(row.get("processing_bucket_source", "unknown")) for row in rows)
        source_summary = next(iter(source_counts)) if len(source_counts) == 1 else ",".join(f"{key}:{source_counts[key]}" for key in sorted(source_counts))
        caveat = "offline forced bucket; not runtime bucket performance" if "algorithm_forced" in source_counts else "metadata/sidecar bucket evidence"
        output.append(
            {
                "processing_bucket": bucket,
                "processing_bucket_source": source_summary,
                "status": "measured",
                "n": len({row.get("pair_id", "") for row in rows}),
                "metric_tier": ",".join(sorted({str(row.get("metric_tier", "")) for row in rows})),
                "pairing_tier": ",".join(sorted({str(row.get("pairing_tier", "")) for row in rows})),
                "input_data_type": ",".join(sorted({str(row.get("input_data_type", "")) for row in rows})),
                "fusion_source": ",".join(sorted({str(row.get("fusion_source", "")) for row in rows})),
                "caveat": caveat,
            }
        )
    return output


def _wide_failure_rows(fusion_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in fusion_rows:
        key = (row.get("pair_id", ""), row.get("algorithm", ""))
        wide = grouped.setdefault(
            key,
            {
                "pair_id": row.get("pair_id", ""),
                "algorithm": row.get("algorithm", ""),
                "baseline_algorithm": row.get("baseline_algorithm", ""),
                "evidence_label": row.get("evidence_label", ""),
                "pairing_tier": row.get("pairing_tier", ""),
                "metric_tier": row.get("metric_tier", ""),
                "fusion_source": row.get("fusion_source", ""),
                "thermal_modality": row.get("thermal_modality", ""),
                "input_data_type": row.get("input_data_type", ""),
                "source_or_session": row.get("source_or_session", ""),
                "caveat": row.get("caveat", ""),
            },
        )
        metric = row.get("metric", "")
        if metric:
            wide[metric] = row.get("value", "")
            wide[f"baseline_{metric}"] = row.get("baseline_value", "")
    return list(grouped.values())


def _failure_rows(fusion_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for wide in _wide_failure_rows(fusion_rows):
        for failure in detect_failure_cases(wide):
            evidence_tier = failure_tier_for_type(
                str(failure.get("failure_type", "")),
                str(wide.get("fusion_source", "")),
                str(wide.get("pairing_tier", "")),
                str(wide.get("evidence_label", "")),
            )
            failures.append(
                {
                    **failure,
                    "pairing_tier": wide.get("pairing_tier", ""),
                    "metric_tier": evidence_tier,
                    "evidence_tier": evidence_tier,
                    "fusion_source": wide.get("fusion_source", ""),
                    "thermal_modality": wide.get("thermal_modality", ""),
                    "input_data_type": wide.get("input_data_type", ""),
                    "caveat": wide.get("caveat", ""),
                }
            )
    return failures


def _failure_summary(failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter((row.get("failure_type", ""), row.get("algorithm", ""), row.get("pairing_tier", ""), row.get("fusion_source", ""), row.get("evidence_tier", "")) for row in failures)
    return [
        {
            "failure_type": failure_type,
            "algorithm": algorithm,
            "pairing_tier": pairing_tier,
            "fusion_source": fusion_source,
            "evidence_tier": evidence_tier,
            "n": n,
            "caveat": "Generated/proxy failures are diagnostic only, not proof of runtime fusion failure."
            if fusion_source == "paired_generated_fusion"
            else "",
        }
        for (failure_type, algorithm, pairing_tier, fusion_source, evidence_tier), n in sorted(counts.items())
    ]


def _write_grid(path: Path, samples: list[tuple[str, np.ndarray, np.ndarray, dict[str, FusionCandidate]]], *, max_rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not samples:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No paired samples available for grid", ha="center", va="center")
        ax.axis("off")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return
    selected = samples[:max_rows]
    cols = ["NIR", "Thermal display", "alpha_blend_baseline", "foreground_mask_overlay"]
    fig, axes = plt.subplots(len(selected), len(cols), figsize=(10, 2.4 * len(selected)))
    if len(selected) == 1:
        axes = np.asarray([axes])
    for row_idx, (pair_id, nir, thermal, candidates) in enumerate(selected):
        images = [
            cv.cvtColor(_to_bgr(nir), cv.COLOR_BGR2RGB),
            cv.cvtColor(_resize_like(_to_bgr(thermal), _to_bgr(nir)), cv.COLOR_BGR2RGB),
            cv.cvtColor(candidates["alpha_blend_baseline"].image, cv.COLOR_BGR2RGB),
            cv.cvtColor(candidates["foreground_mask_overlay"].image, cv.COLOR_BGR2RGB),
        ]
        for col_idx, image in enumerate(images):
            ax = axes[row_idx, col_idx]
            ax.imshow(image)
            label = cols[col_idx] if row_idx == 0 else ""
            if col_idx == 0:
                label = f"{pair_id}\nframe/time strict"
            if col_idx == 3:
                label = (label + "\n" if label else "") + candidates["foreground_mask_overlay"].fusion_source
            ax.set_title(label, fontsize=8)
            ax.axis("off")
    fig.suptitle("Strict paired diagnostic grid; generated fusion is not runtime-captured fusion", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _write_runtime_summary(
    *,
    manifest_rows: list[dict[str, str]],
    output_path: Path,
    docs_path: Path,
) -> None:
    timestamps = [_safe_float(row.get("timestamp_sec")) for row in manifest_rows]
    timestamps = [v for v in timestamps if v is not None]
    skews = [_safe_float(row.get("pairing_gap_ms")) for row in manifest_rows]
    skews = [v for v in skews if v is not None]
    intervals = np.diff(np.asarray(timestamps, dtype=np.float64)) if len(timestamps) > 1 else np.asarray([])
    fps_values = 1.0 / intervals[intervals > 0] if intervals.size else np.asarray([])
    rows = []
    if intervals.size:
        frame_ms = intervals * 1000.0
        rows.append(
            {
                "source_or_session": manifest_rows[0].get("source_or_session", "paired_data") if manifest_rows else "paired_data",
                "metric": "pair_interval_ms",
                "n": len(frame_ms),
                "mean": _fmt(float(np.mean(frame_ms))),
                "measured_p95": _fmt(float(np.percentile(frame_ms, 95))),
                "estimated_p95": "",
                "p95_source": "measured_from_timestamps_csv",
                "nir_processing_latency_ms": "not_measured",
                "thermal_processing_latency_ms": "not_measured",
                "fusion_composite_latency_ms": "not_measured",
                "caveat": "Capture cadence evidence only; stage profiler fields are not present in timestamps.csv.",
            }
        )
    if fps_values.size:
        rows.append(
            {
                "source_or_session": manifest_rows[0].get("source_or_session", "paired_data") if manifest_rows else "paired_data",
                "metric": "fps",
                "n": len(fps_values),
                "mean": _fmt(float(np.mean(fps_values))),
                "measured_p95": _fmt(float(np.percentile(fps_values, 95))),
                "estimated_p95": "",
                "p95_source": "measured_from_timestamps_csv",
                "nir_processing_latency_ms": "not_measured",
                "thermal_processing_latency_ms": "not_measured",
                "fusion_composite_latency_ms": "not_measured",
                "caveat": "FPS is measured from paired capture timestamps; processing stage latency is not measured.",
            }
        )
    if skews:
        rows.append(
            {
                "source_or_session": manifest_rows[0].get("source_or_session", "paired_data") if manifest_rows else "paired_data",
                "metric": "abs_skew_ms",
                "n": len(skews),
                "mean": _fmt(float(np.mean(skews))),
                "measured_p95": _fmt(float(np.percentile(skews, 95))),
                "estimated_p95": "",
                "p95_source": "measured_from_timestamps_csv",
                "nir_processing_latency_ms": "not_measured",
                "thermal_processing_latency_ms": "not_measured",
                "fusion_composite_latency_ms": "not_measured",
                "caveat": "Pairing skew evidence; not per-stage processing latency.",
            }
        )
    write_markdown(
        output_path,
        "Paired Runtime Timing Summary",
        markdown_table(
            rows,
            [
                "source_or_session",
                "metric",
                "n",
                "mean",
                "measured_p95",
                "estimated_p95",
                "p95_source",
                "nir_processing_latency_ms",
                "thermal_processing_latency_ms",
                "fusion_composite_latency_ms",
                "caveat",
            ],
        ),
    )
    body = "\n".join(
        [
            "## Paired Data Timing Update",
            "",
            "The paired dataset provides timestamp-derived capture cadence and NIR/thermal skew evidence.",
            "It does not provide measured NIR, thermal, or fusion processing stage profiler fields.",
            "All p95 values in the paired timing table are stored as `measured_p95` when computed directly from timestamp rows; `estimated_p95` remains blank.",
            "",
            f"- Timing table: `{output_path}`",
        ]
    )
    _append_or_replace_section(docs_path, "Paired Data Timing Update", body)


def _append_or_replace_section(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    marker = f"## {title}"
    existing = path.read_text(encoding="utf-8") if path.exists() else f"# {path.stem.replace('_', ' ').title()}\n"
    before = existing.split(marker)[0].rstrip()
    path.write_text(before + "\n\n" + body.rstrip() + "\n", encoding="utf-8")


def _write_summary_table(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    write_markdown(path, title, markdown_table(rows, SUMMARY_COLUMNS))


def _write_thesis_docs(
    *,
    repo_root: Path,
    paired_docs_dir: Path,
    fusion_docs_dir: Path,
    fusion_tables_dir: Path,
    manifest_rows: list[dict[str, str]],
    fusion_summary: list[dict[str, Any]],
    failure_summary: list[dict[str, Any]],
) -> None:
    pairing_counts = Counter(row.get("pairing_tier", "unknown") for row in manifest_rows)
    fusion_source_counts = Counter(row.get("fusion_source", "unknown") for row in fusion_summary)
    strict_count = sum(pairing_counts[tier] for tier in STRICT_TIERS)
    thesis_claim = "strict paired offline fusion evaluation" if strict_count else "proxy/no-reference fusion evaluation only"
    if strict_count and not fusion_source_counts.get("paired_captured_fusion", 0):
        thesis_claim += "; fusion candidates are generated offline, not runtime-captured fusion"
    body = "\n".join(
        [
            f"- Strict paired rows: `{strict_count}`",
            f"- Pairing distribution: `{dict(pairing_counts)}`",
            f"- Thesis claim level: `{thesis_claim}`",
            "- Thermal caveat: `thermal_modality=display_heatmap_like` means the thermal video is not raw radiometric thermal.",
            "- Generated fusion caveat: `paired_generated_fusion` is evidence for offline fusion algorithms on real paired inputs, not proof of runtime-captured fusion output.",
            f"- NIR summary: `{_rel(repo_root, fusion_tables_dir / 'strict_paired_nir_quality_summary.md')}`",
            f"- Thermal summary: `{_rel(repo_root, fusion_tables_dir / 'strict_paired_thermal_quality_summary.md')}`",
            f"- Fusion summary: `{_rel(repo_root, fusion_tables_dir / 'strict_paired_fusion_quality_summary.md')}`",
            f"- Failure summary rows: `{len(failure_summary)}`",
        ]
    )
    write_markdown(paired_docs_dir / "PAIRED_DATA_FUSION_EVALUATION_SUMMARY.md", "Paired Data Fusion Evaluation Summary", body)
    _append_or_replace_section(fusion_docs_dir / "FUSION_EVIDENCE_READINESS.md", "Paired Data Strict Evidence Update", body)
    _append_or_replace_section(fusion_docs_dir / "FUSION_EVALUATION.md", "Paired Data Strict Evidence Update", body)
    _append_or_replace_section(fusion_docs_dir / "IMAGE_PROCESSING_EVALUATION.md", "Paired Data Strict Evidence Update", body)
    _append_or_replace_section(fusion_docs_dir / "REVIEW_RESPONSE_FUSION_MATRIX.md", "Paired Data Strict Evidence Update", body)
    report_notes = repo_root / "docs/ml/REPORT_PATCH_NOTES.md"
    _append_or_replace_section(report_notes, "Fusion/Image Paired Data Evidence Update", body)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    manifest_path = _resolve(repo_root, args.manifest)
    output_dir = _resolve(repo_root, args.output_dir)
    fusion_tables_dir = _resolve(repo_root, args.fusion_tables_dir)
    paired_docs_dir = _resolve(repo_root, args.paired_docs_dir)
    fusion_docs_dir = _resolve(repo_root, args.fusion_docs_dir)
    figures_dir = _resolve(repo_root, args.figures_dir)
    manifest_rows = read_csv_rows(manifest_path)
    if args.max_pairs > 0:
        manifest_rows = manifest_rows[: args.max_pairs]
    run = {
        "timestamp_iso": _now_iso(),
        "git_commit": _git_commit(repo_root),
        "command": " ".join(sys.argv),
        "inputs": [_rel(repo_root, manifest_path)],
        "config": {
            "strict_tiers": sorted(STRICT_TIERS),
            "generated_fusion_policy": "paired_generated_fusion is not runtime-captured fusion",
            "metric_max_dim": args.metric_max_dim,
        },
    }
    if args.dry_run:
        print(json.dumps(run, indent=2, sort_keys=True))
        return 0

    nir_rows: list[dict[str, str]] = []
    thermal_rows: list[dict[str, str]] = []
    fusion_rows: list[dict[str, str]] = []
    grid_samples: list[tuple[str, np.ndarray, np.ndarray, dict[str, FusionCandidate]]] = []
    frame_reader = SequentialFrameReader(repo_root)

    try:
        for row in manifest_rows:
            frame_index = int(row.get("video_frame_index") or row.get("frame_idx") or 0)
            nir = frame_reader.frame_from_manifest_path(row.get("nir_raw_path", ""), frame_index)
            thermal = frame_reader.frame_from_manifest_path(row.get("thermal_heatmap_path") or row.get("thermal_raw_path", ""), frame_index)
            captured = frame_reader.frame_from_manifest_path(row.get("fusion_output_path", ""), frame_index) if row.get("fusion_output_path") else None
            if nir is None or thermal is None:
                continue
            nir_eval = _resize_max_dim(nir, args.metric_max_dim)
            thermal_eval = _resize_max_dim(thermal, args.metric_max_dim)
            captured_eval = _resize_max_dim(captured, args.metric_max_dim) if captured is not None else None
            candidates = fusion_candidates(nir_eval, thermal_eval, captured_fusion=captured_eval)
            nir_rows.extend(_long_image_metric_rows(pair_row=row, image=nir_eval))
            thermal_rows.extend(_thermal_metric_rows(row, thermal_eval))
            fusion_rows.extend(
                metric_rows_for_pair(
                    pair_id=row.get("pair_id", ""),
                    nir=nir_eval,
                    thermal=thermal_eval,
                    candidates=candidates,
                    pairing_tier=row.get("pairing_tier", ""),
                    thermal_modality=row.get("thermal_modality", ""),
                    input_data_type=row.get("input_data_type", ""),
                    processing_bucket=row.get("processing_bucket", "unknown"),
                    processing_bucket_source=row.get("processing_bucket_source", "unknown"),
                    source_or_session=row.get("source_or_session", ""),
                )
            )
            if len(grid_samples) < max(args.grid_samples, args.failure_grid_cases):
                grid_samples.append((row.get("pair_id", ""), nir_eval, thermal_eval, candidates))
    finally:
        frame_reader.close()

    nir_summary = summarize_metric_rows_by_group(nir_rows)
    thermal_summary = summarize_metric_rows_by_group(thermal_rows)
    fusion_summary = summarize_metric_rows_by_group(fusion_rows)
    bucket_rows = per_bucket_eval_rows(nir_rows, manifest_rows)
    failures = _failure_rows(fusion_rows)
    failure_summary = _failure_summary(failures)

    write_csv(output_dir / "nir_quality_metrics.csv", nir_rows)
    write_csv(output_dir / "thermal_quality_metrics.csv", thermal_rows)
    write_csv(output_dir / "fusion_quality_metrics.csv", fusion_rows)
    write_csv(output_dir / "strict_paired_failure_cases.csv", failures)
    write_json(output_dir / "paired_fusion_eval_run_manifest.json", run)
    _write_summary_table(fusion_tables_dir / "strict_paired_nir_quality_summary.md", "Strict Paired NIR Quality Summary", nir_summary)
    _write_summary_table(fusion_tables_dir / "strict_paired_thermal_quality_summary.md", "Strict Paired Thermal Quality Summary", thermal_summary)
    _write_summary_table(fusion_tables_dir / "strict_paired_fusion_quality_summary.md", "Strict Paired Fusion Quality Summary", fusion_summary)
    comparison_rows = [
        row
        for row in fusion_summary
        if row.get("algorithm") in {"foreground_mask_overlay", "mask_weighted_blend", "laplacian_pyramid_fusion", "alpha_blend_baseline"}
    ]
    _write_summary_table(fusion_tables_dir / "strict_paired_fusion_algorithm_comparison.md", "Strict Paired Fusion Algorithm Comparison", comparison_rows)
    write_markdown(
        fusion_tables_dir / "per_bucket_processing_eval.md",
        "Per-Bucket Processing Evaluation",
        markdown_table(
            bucket_rows,
            ["processing_bucket", "processing_bucket_source", "status", "n", "metric_tier", "pairing_tier", "input_data_type", "fusion_source", "caveat"],
        ),
    )
    missing_rows = [row for row in bucket_rows if row.get("status") != "measured"]
    write_markdown(
        fusion_tables_dir / "per_bucket_missing_evidence.md",
        "Per-Bucket Missing Evidence",
        markdown_table(missing_rows, ["processing_bucket", "status", "n", "caveat"]),
    )
    write_markdown(
        fusion_tables_dir / "strict_paired_failure_case_summary.md",
        "Strict Paired Failure Case Summary",
        markdown_table(failure_summary, ["failure_type", "algorithm", "pairing_tier", "fusion_source", "evidence_tier", "n", "caveat"]),
    )
    _write_grid(figures_dir / "strict_paired_fusion_comparison_grid.png", grid_samples, max_rows=args.grid_samples)
    _write_grid(figures_dir / "strict_paired_failure_cases_grid.png", grid_samples, max_rows=args.failure_grid_cases)
    _write_runtime_summary(
        manifest_rows=manifest_rows,
        output_path=fusion_tables_dir / "paired_runtime_timing_summary.md",
        docs_path=fusion_docs_dir / "RUNTIME_TIMING_EVIDENCE.md",
    )
    _write_thesis_docs(
        repo_root=repo_root,
        paired_docs_dir=paired_docs_dir,
        fusion_docs_dir=fusion_docs_dir,
        fusion_tables_dir=fusion_tables_dir,
        manifest_rows=manifest_rows,
        fusion_summary=fusion_summary,
        failure_summary=failure_summary,
    )
    print(
        "Evaluated paired evidence: "
        f"{len(manifest_rows)} manifest rows, {len(nir_rows)} NIR metric rows, "
        f"{len(thermal_rows)} thermal metric rows, {len(fusion_rows)} fusion metric rows."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
