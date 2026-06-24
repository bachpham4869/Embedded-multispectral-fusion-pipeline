#!/usr/bin/env python3
"""Harden offline fusion/image-processing evidence for thesis use.

This script only reads existing fusion/image evaluation artifacts and session
metrics, then regenerates thesis-facing docs/tables with evidence tiers and
caveats. It does not alter runtime pipelines, raw captures, ML JSONL splits, or
production model files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
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

from tools.fusion_eval_manifest import (  # type: ignore[import]
    git_commit,
    markdown_table,
    now_iso,
    read_csv_rows,
    run_manifest,
    write_csv,
    write_json,
    write_markdown,
)
from tools.fusion_eval_metrics import METRIC_DEFINITIONS, detect_failure_cases, metric_tier  # type: ignore[import]


REQUIRED_NIR_BUCKETS = [
    "night_hybrid_enhance",
    "nir_mono_clahe",
    "highlight_tone_map",
    "fog_dehaze_lite",
    "rain_temporal_median",
    "dawn_dusk_blend",
]

REQUIRED_FUSION_MODES = [
    "nir_only_baseline",
    "thermal_heatmap_only",
    "alpha_blend_baseline",
    "foreground_mask_overlay",
    "mask_weighted_blend",
    "legacy_gradient_overlay",
    "laplacian_pyramid_fusion",
]

REQUIRED_THERMAL_STAGES = [
    "thermal_raw",
    "thermal_denoised_3dnr",
    "thermal_agc",
    "thermal_heatmap",
    "thermal_foreground_mask",
]

REQUIRED_CAPTURE_METADATA_FIELDS = [
    "session_id",
    "frame_idx",
    "timestamp_iso",
    "mode",
    "env_class",
    "processing_bucket",
    "fusion_mode",
    "homography_path",
    "nir_raw_path",
    "nir_enhanced_path",
    "thermal_raw_path",
    "thermal_heatmap_path",
    "thermal_mask_path",
    "fusion_output_path",
    "stage_timings_ms",
]

REQUIRED_FAILURE_TYPES = [
    "clipping_increase",
    "crushed_shadows",
    "noise_increase",
    "sharpness_up_contrast_down",
    "contrast_drop",
    "fusion_target_faded",
    "mask_wrong_region",
    "nir_detail_occlusion",
    "alignment_drift",
]

SUMMARY_COLUMNS = [
    "algorithm",
    "baseline_algorithm",
    "evidence_label",
    "input_data_type",
    "metric_tier",
    "caveat",
    "evidence_label_distribution",
    "bucket_or_condition",
    "source_or_session",
    "metric",
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
]

RUNTIME_COLUMNS = [
    "session_file",
    "source_or_session",
    "mode",
    "evidence_label",
    "input_data_type",
    "metric_tier",
    "caveat",
    "duration_wall_s",
    "frames_total",
    "fps_mean",
    "measured_p95",
    "estimated_p95",
    "p95_source",
    "nir_processing_latency_ms",
    "thermal_processing_latency_ms",
    "fusion_composite_latency_ms",
    "stage_profiler_available",
    "stage_profiler_summary",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Harden offline fusion/image evidence tables, caveats, runtime timing, and capture protocol.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/fusion_eval"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs/fusion"))
    parser.add_argument("--tables-dir", type=Path, default=Path("docs/tables/fusion"))
    parser.add_argument("--figures-dir", type=Path, default=Path("docs/figures/fusion"))
    parser.add_argument("--session-metrics-dir", type=Path, default=Path("fusion_captures/metrics"))
    parser.add_argument("--max-failure-grid-cases", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Inspect inputs and print output paths without writing.")
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _rel(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _safe_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return str(round(value, 6))
    return str(value)


def _evidence_counts(rows: Iterable[dict[str, Any]]) -> str:
    counts = Counter(str(row.get("evidence_label") or "unknown") for row in rows)
    return ", ".join(f"{key}:{counts[key]}" for key in sorted(counts))


def strict_pair_count(artifacts_dir: Path) -> int:
    rows = read_csv_rows(artifacts_dir / "pairing_manifest.csv")
    return sum(1 for row in rows if row.get("pair_status") == "strict_paired")


def input_data_type_for(evidence_label: str, *, table_kind: str = "image") -> str:
    label = (evidence_label or "unknown").strip()
    if table_kind == "runtime":
        return "real session JSON"
    if label == "real_paired":
        return "real paired local capture"
    if label == "proxy":
        return "proxy/weak local capture or synthetic pairing"
    if label == "synthetic":
        return "synthetic/proxy sequence"
    if label == "unpaired":
        return "unpaired local capture"
    return "unknown local artifact"


def caveat_for(metric: str, evidence_label: str, *, table_kind: str = "image") -> str:
    if table_kind == "runtime":
        return "Real session timing cost evidence; p95 may be estimated if per-frame samples were not recorded."
    label = (evidence_label or "unknown").strip()
    if label in {"proxy", "synthetic", "unpaired", "unknown"}:
        return "Tier 3 proxy/synthetic/unpaired evidence; proxy only, not proof of real fusion quality, requires future paired capture validation."
    definition = METRIC_DEFINITIONS.get(metric, {})
    tier = int(definition.get("tier", 2))
    if tier == 1:
        return str(definition.get("limitation", "Strong only with valid strict paired task-specific capture."))
    if tier == 2:
        return str(definition.get("limitation", "No-reference IQA proxy; not ground-truth perceptual quality."))
    return str(definition.get("limitation", "Proxy metric; not proof of real fusion quality."))


def tier_name(metric: str, evidence_label: str, *, table_kind: str = "image") -> str:
    if table_kind == "runtime":
        return "Tier 1"
    return f"Tier {metric_tier(metric, evidence_label)}"


def enrich_summary_rows(rows: list[dict[str, str]], *, table_kind: str = "image") -> list[dict[str, str]]:
    distribution = _evidence_counts(rows)
    enriched: list[dict[str, str]] = []
    for row in rows:
        metric = str(row.get("metric") or ("fps_mean" if table_kind == "runtime" else ""))
        evidence = str(row.get("evidence_label") or "unknown")
        out = dict(row)
        out["input_data_type"] = input_data_type_for(evidence, table_kind=table_kind)
        out["metric_tier"] = tier_name(metric, evidence, table_kind=table_kind)
        out["caveat"] = caveat_for(metric, evidence, table_kind=table_kind)
        out["evidence_label_distribution"] = distribution
        enriched.append(out)
    return enriched


def _write_summary_table(path: Path, title: str, rows: list[dict[str, str]], *, intro: str, columns: list[str]) -> None:
    body = intro.rstrip() + "\n\n"
    body += markdown_table(rows, columns)
    write_markdown(path, title, body)


def build_processing_name_alias_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    bucket_aliases = {
        "night_hybrid_enhance": "HybridNIREnhancer / bucket A / dark-scene hybrid enhancement",
        "nir_mono_clahe": "NIR grayscale CLAHE / bucket B",
        "highlight_tone_map": "anti-glare tone map / bucket C",
        "fog_dehaze_lite": "dehaze-lite / bucket D",
        "rain_temporal_median": "temporal median rain denoise / bucket E",
        "dawn_dusk_blend": "dawn/dusk blend / bucket F",
    }
    for name in REQUIRED_NIR_BUCKETS:
        rows.append(
            {
                "report_name": name,
                "category": "nir_processing_bucket",
                "code_or_artifact_alias": bucket_aliases[name],
                "rename_policy": "report alias only; no production symbol rename",
            }
        )
    fusion_aliases = {
        "nir_only_baseline": "NIR-only comparison frame",
        "thermal_heatmap_only": "thermal heatmap-only comparison frame",
        "alpha_blend_baseline": "simple weighted alpha blend",
        "foreground_mask_overlay": "current foreground-mask thermal overlay",
        "mask_weighted_blend": "offline mask-weighted blend baseline",
        "legacy_gradient_overlay": "legacy/pre-optimization gradient overlay",
        "laplacian_pyramid_fusion": "offline Laplacian pyramid fusion baseline",
    }
    for name in REQUIRED_FUSION_MODES:
        rows.append(
            {
                "report_name": name,
                "category": "fusion_mode",
                "code_or_artifact_alias": fusion_aliases[name],
                "rename_policy": "report alias only; no production symbol rename",
            }
        )
    thermal_aliases = {
        "thermal_raw": "MI48/raw thermal frame",
        "thermal_denoised_3dnr": "thermal temporal 3DNR output",
        "thermal_agc": "thermal automatic gain control output",
        "thermal_heatmap": "false-color thermal heatmap",
        "thermal_foreground_mask": "thermal foreground mask",
    }
    for name in REQUIRED_THERMAL_STAGES:
        rows.append(
            {
                "report_name": name,
                "category": "thermal_stage",
                "code_or_artifact_alias": thermal_aliases[name],
                "rename_policy": "report alias only; no production symbol rename",
            }
        )
    return rows


def build_required_capture_metadata_rows() -> list[dict[str, str]]:
    descriptions = {
        "session_id": "Stable capture session identifier shared by all modalities.",
        "frame_idx": "Monotonic frame index or processing-cycle index.",
        "timestamp_iso": "ISO timestamp from the same clock for NIR, thermal, and fusion outputs.",
        "mode": "Capture mode such as nir, thermal, or fusion.",
        "env_class": "Environment class or scenario label if known.",
        "processing_bucket": "Report-facing NIR bucket name.",
        "fusion_mode": "Report-facing fusion mode name.",
        "homography_path": "Homography/calibration file used for thermal-to-NIR alignment.",
        "nir_raw_path": "Raw NIR frame path.",
        "nir_enhanced_path": "Enhanced NIR output path.",
        "thermal_raw_path": "Raw thermal frame or numeric array path.",
        "thermal_heatmap_path": "Thermal heatmap path.",
        "thermal_mask_path": "Foreground mask path.",
        "fusion_output_path": "Final fused frame path.",
        "stage_timings_ms": "Per-stage timing dict for NIR, thermal, fusion, display, and logging when available.",
    }
    rows: list[dict[str, str]] = []
    for field in REQUIRED_CAPTURE_METADATA_FIELDS:
        rows.append(
            {
                "field": field,
                "required": "yes" if field != "stage_timings_ms" else "recommended",
                "description": descriptions[field],
                "strict_pair_requirement": "same session and <= 1s gap"
                if field in {"timestamp_iso", "session_id", "frame_idx"}
                else "must reference the same frame/cycle when present",
            }
        )
    return rows


def build_evidence_readiness_rows(
    *,
    repo_root: Path,
    artifacts_dir: Path,
    docs_dir: Path,
    tables_dir: Path,
    figures_dir: Path,
) -> list[dict[str, str]]:
    strict_count = strict_pair_count(artifacts_dir)
    pair_rows = read_csv_rows(artifacts_dir / "pairing_manifest.csv")
    evidence_distribution = _evidence_counts(pair_rows) if pair_rows else "none"
    readiness_label = "real_paired" if strict_count > 0 else "proxy/unpaired"
    strict_caveat = (
        f"strict paired = {strict_count}; proxy only, not proof of real fusion quality, requires future paired capture validation."
    )

    specs = [
        ("docs/tables/fusion/artifact_inventory.md", "Tier 3", "mixed local artifacts", "inventory/provenance", "thesis-ready as data audit", "Does not prove quality.", "None for audit use."),
        ("docs/tables/fusion/pairing_manifest_summary.md", "Tier 3", "local capture timestamps", "pairing audit", "thesis-ready as limitation evidence", strict_caveat, "Strict paired captures."),
        ("docs/tables/fusion/nir_quality_summary.md", "Tier 2/3", "NIR still/proxy images", "no-reference IQA", "preliminary image-processing evidence", "No-reference/proxy metrics are not ground-truth perceptual quality.", "Strict paired before/after captures."),
        ("docs/tables/fusion/thermal_quality_summary.md", "Tier 2/3", "thermal captures/proxy sequences", "no-reference IQA", "preliminary thermal-processing evidence", "Proxy/synthetic thermal rows cannot prove real thermal quality.", "Real paired thermal raw/enhanced sequences."),
        ("docs/tables/fusion/fusion_quality_summary.md", "Tier 3", "weak/proxy/unpaired fusion rows", "proxy fusion metrics", "preliminary diagnostic only", strict_caveat, "Strict paired NIR/thermal/fusion frames."),
        ("docs/tables/fusion/image_processing_algorithm_comparison.md", "Tier 3", "offline still/proxy images", "baseline comparison", "preliminary algorithm comparison", "Current-vs-baseline deltas are proxy evidence only.", "Real NIR raw/enhanced paired captures."),
        ("docs/tables/fusion/fusion_algorithm_comparison.md", "Tier 3", "weak/proxy/unpaired fusion rows", "baseline comparison", "preliminary fusion baseline comparison", strict_caveat, "Strict paired fusion captures."),
        ("docs/tables/fusion/failure_case_summary.md", "Tier 3", "weak/proxy/unpaired failure mining", "diagnostic failure cases", "useful for limitations section", "Proxy failures are diagnostic only, not proof of runtime fusion failure.", "Strict paired failure reproductions."),
        ("docs/tables/fusion/runtime_timing_summary.md", "Tier 1", "real session JSON", "runtime timing", "thesis-ready as runtime cost evidence", "p95 is estimated when per-frame p95 was not recorded.", "Measured per-frame p95 for all stages."),
        ("docs/figures/fusion/failure_cases_grid.png", "Tier 3", "weak/proxy/unpaired image grid", "visual diagnostic", "limitations/qualitative figure only", "Grid examples must be labeled proxy/unpaired/weak.", "Strict paired visual failure grid."),
        ("docs/figures/fusion/sample_comparison_grid.png", "Tier 3", "proxy comparison grid", "visual diagnostic", "qualitative illustration only", "Not proof of real fusion quality.", "Strict paired comparison grid."),
        ("docs/fusion/STRICT_PAIRED_FUSION_CAPTURE_PROTOCOL.md", "Protocol", "future capture design", "capture protocol", "thesis-ready future-work protocol", "Not measured evidence.", "Future capture execution."),
    ]
    rows = []
    for artifact, tier, data_type, metric_type, usability, caveat, missing in specs:
        path = repo_root / artifact
        exists = path.exists()
        rows.append(
            {
                "artifact": artifact,
                "exists": "yes" if exists else "no",
                "metric_tier": tier,
                "evidence_label": readiness_label if "runtime" not in artifact else "real_paired",
                "input_data_type": data_type,
                "metric_type": metric_type,
                "thesis_usability": usability if exists or artifact.endswith("STRICT_PAIRED_FUSION_CAPTURE_PROTOCOL.md") else "missing",
                "caveat": caveat,
                "missing_evidence": missing,
                "provenance_source": f"artifacts/fusion_eval/run_manifest.json; evidence_distribution={evidence_distribution}",
            }
        )
    return rows


def _dominant_mode(frames_by_mode: dict[str, Any]) -> str:
    if not frames_by_mode:
        return ""
    return max(frames_by_mode.items(), key=lambda item: _safe_float(item[1]) or 0.0)[0]


def _stage_mean(stage_timing: dict[str, Any], key: str) -> str:
    value = stage_timing.get(key, {})
    if isinstance(value, dict):
        return _fmt(_safe_float(value.get("mean_ms")))
    return ""


def _all_stage_keys(data: dict[str, Any]) -> list[str]:
    keys: set[str] = set()
    for section in ("stage_timing_ms", "fuse_stage_timing_ms", "thermal_stage_timing_ms"):
        value = data.get(section)
        if isinstance(value, dict):
            keys.update(str(k) for k in value.keys())
    return sorted(keys)


def parse_session_runtime_json(session_metrics_dir: Path, repo_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(session_metrics_dir.glob("session_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        stage_timing = data.get("stage_timing_ms") if isinstance(data.get("stage_timing_ms"), dict) else {}
        fps_mean = _safe_float(data.get("fps_mean"))
        fps_p95 = _safe_float(data.get("fps_p95") or data.get("fps_95") or data.get("fps_p95_sample"))
        fps_std = _safe_float(data.get("fps_std_sample"))
        if fps_p95 is not None:
            p95_source = "measured"
            fps_p95_text = _fmt(fps_p95)
        elif fps_mean is not None and fps_std is not None:
            p95_source = "estimated_mean_plus_1.645std"
            fps_p95_text = _fmt(fps_mean + 1.645 * fps_std)
        else:
            p95_source = "missing"
            fps_p95_text = ""
        frames_by_mode = data.get("frames_by_mode") if isinstance(data.get("frames_by_mode"), dict) else {}
        rows.append(
            {
                "session_file": _rel(repo_root, path),
                "source_or_session": str(data.get("session_id", "")),
                "mode": _dominant_mode(frames_by_mode),
                "evidence_label": "real_paired",
                "duration_wall_s": _fmt(_safe_float(data.get("duration_wall_s"))),
                "frames_total": _fmt(_safe_float(data.get("frames_total"))),
                "fps_mean": _fmt(fps_mean),
                "fps_p95": fps_p95_text,
                "fps_p95_source": p95_source,
                "nir_processing_latency_ms": _stage_mean(stage_timing, "nir_bucket"),
                "thermal_processing_latency_ms": _stage_mean(stage_timing, "thermal_proc"),
                "fusion_composite_latency_ms": _stage_mean(stage_timing, "fusion_composite"),
                "stage_profiler_summary": "; ".join(_all_stage_keys(data)),
            }
        )
    return rows


def normalize_runtime_timing_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for row in rows:
        p95_source = str(row.get("fps_p95_source") or row.get("p95_source") or "").strip()
        fps_p95 = _fmt(row.get("fps_p95") or row.get("p95") or "")
        measured_p95 = fps_p95 if p95_source == "measured" else ""
        estimated_p95 = fps_p95 if p95_source.startswith("estimated") else ""
        stage_summary = str(row.get("stage_profiler_summary") or "")
        caveats = []
        if estimated_p95:
            caveats.append("p95 estimated from mean/std because per-frame p95 was not recorded")
        elif not measured_p95:
            caveats.append("p95 missing")
        if not stage_summary:
            caveats.append("stage profiler fields missing")
        else:
            missing = [
                label
                for label, key in (
                    ("NIR", "nir_bucket"),
                    ("thermal", "thermal_proc"),
                    ("fusion", "fusion_composite"),
                )
                if key not in stage_summary
            ]
            if missing:
                caveats.append("missing stage profiler fields: " + ", ".join(missing))
        out = {
            "session_file": str(row.get("session_file", "")),
            "source_or_session": str(row.get("source_or_session", "")),
            "mode": str(row.get("mode", "")),
            "evidence_label": str(row.get("evidence_label") or "real_paired"),
            "input_data_type": "real session JSON",
            "metric_tier": "Tier 1",
            "caveat": "; ".join(caveats) if caveats else "measured session timing with stage profiler fields",
            "duration_wall_s": _fmt(row.get("duration_wall_s", "")),
            "frames_total": _fmt(row.get("frames_total", "")),
            "fps_mean": _fmt(row.get("fps_mean", "")),
            "measured_p95": measured_p95,
            "estimated_p95": estimated_p95,
            "p95_source": p95_source or "missing",
            "nir_processing_latency_ms": _fmt(row.get("nir_processing_latency_ms", "")),
            "thermal_processing_latency_ms": _fmt(row.get("thermal_processing_latency_ms", "")),
            "fusion_composite_latency_ms": _fmt(row.get("fusion_composite_latency_ms", "")),
            "stage_profiler_available": "yes" if stage_summary else "no",
            "stage_profiler_summary": stage_summary,
        }
        normalized.append(out)
    return normalized


def _pivot_algorithm_metric_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("case_id") or row.get("pair_id") or ""),
            str(row.get("algorithm", "")),
            str(row.get("baseline_algorithm", "")),
            str(row.get("evidence_label", "")),
            str(row.get("source_or_session", "")),
        )
        target = grouped.setdefault(
            key,
            {
                "pair_id": key[0],
                "algorithm": key[1],
                "baseline_algorithm": key[2],
                "evidence_label": key[3],
                "source_or_session": key[4],
                "bucket_or_condition": row.get("bucket_or_condition", ""),
                "alignment_status": "weak_pair_or_proxy" if key[3] != "real_paired" else "",
            },
        )
        metric = str(row.get("metric", ""))
        if not metric:
            continue
        value = _safe_float(row.get("value"))
        baseline = _safe_float(row.get("baseline_value"))
        if value is not None:
            target[metric] = value
        if baseline is not None:
            target[f"baseline_{metric}"] = baseline
    return list(grouped.values())


def _failure_rows_from_sources(artifacts_dir: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    candidates.extend(read_csv_rows(artifacts_dir / "fusion_pair_metrics.csv"))
    candidates.extend(_pivot_algorithm_metric_rows(read_csv_rows(artifacts_dir / "fusion_algorithms/fusion_algorithm_comparison_metrics.csv")))
    failures: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in candidates:
        for failure in detect_failure_cases(row):
            key = (
                str(failure.get("failure_type", "")),
                str(failure.get("pair_id", "")),
                str(failure.get("algorithm", "")),
                str(failure.get("detail", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            failure["metric_trigger"] = failure.get("detail", "")
            failure["input_data_type"] = input_data_type_for(str(failure.get("evidence_label", "")))
            failure["metric_tier"] = tier_name("foreground_contrast_gain", str(failure.get("evidence_label", "")))
            failure["caveat"] = "Proxy/unpaired failure case is diagnostic only, not proof of runtime fusion failure."
            failures.append(failure)
    return failures


def _summarize_failure_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str]] = Counter(
        (str(r.get("failure_type", "")), str(r.get("algorithm", "")), str(r.get("evidence_label", ""))) for r in rows
    )
    summary = [
        {
            "failure_type": failure_type,
            "algorithm": algorithm,
            "evidence_label": evidence_label,
            "input_data_type": input_data_type_for(evidence_label),
            "metric_tier": tier_name("foreground_contrast_gain", evidence_label),
            "count": count,
            "caveat": "Diagnostic only unless reproduced on strict paired captures.",
        }
        for (failure_type, algorithm, evidence_label), count in sorted(counts.items())
    ]
    present = {str(row["failure_type"]) for row in summary}
    for failure_type in REQUIRED_FAILURE_TYPES:
        if failure_type not in present:
            summary.append(
                {
                    "failure_type": failure_type,
                    "algorithm": "foreground_mask_overlay",
                    "evidence_label": "proxy/unpaired",
                    "input_data_type": "weak/proxy/unpaired local artifacts",
                    "metric_tier": "Tier 3",
                    "count": 0,
                    "caveat": "Trigger checked; no current local diagnostic cases found.",
                }
            )
    return sorted(summary, key=lambda row: (str(row["failure_type"]), str(row["evidence_label"])))


def _load_bgr(path_text: str) -> np.ndarray | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists():
        return None
    return cv.imread(str(path), cv.IMREAD_COLOR)


def _write_failure_grid(rows: list[dict[str, Any]], figure: Path, *, max_cases: int) -> None:
    figure.parent.mkdir(parents=True, exist_ok=True)
    source_rows = read_csv_rows(ROOT / "artifacts/fusion_eval/fusion_pair_metrics.csv")
    by_pair = {row.get("pair_id", ""): row for row in source_rows}
    picks = [row for row in rows if row.get("pair_id") in by_pair][:max_cases]
    if not picks:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No failure image examples available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(figure, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return
    cols = [("NIR", "nir_path"), ("Thermal", "thermal_path"), ("Fusion", "fusion_path")]
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(9, max(2.4, 2.25 * len(picks))))
    if len(picks) == 1:
        axes = np.asarray([axes])
    for row_idx, failure in enumerate(picks):
        source = by_pair[str(failure.get("pair_id"))]
        for col_idx, (label, key) in enumerate(cols):
            ax = axes[row_idx, col_idx]
            img = _load_bgr(str(source.get(key, "")))
            if img is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8)
                ax.axis("off")
                continue
            ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            title = f"{label}\n{failure.get('evidence_label','')} {failure.get('failure_type','')}"
            ax.set_title(title, fontsize=7)
            ax.axis("off")
    fig.suptitle("Failure cases are diagnostic only; proxy/unpaired rows are not runtime failure proof.", fontsize=9)
    fig.tight_layout()
    fig.savefig(figure, dpi=120)
    plt.close(fig)


def _write_protocol_docs(docs_dir: Path, tables_dir: Path) -> None:
    metadata_rows = build_required_capture_metadata_rows()
    write_markdown(
        tables_dir / "required_capture_metadata.md",
        "Required Capture Metadata",
        markdown_table(metadata_rows, ["field", "required", "description", "strict_pair_requirement"]),
    )
    body = (
        "This protocol is required to unlock Tier 1 fusion-quality evidence. The current offline run has "
        "`strict paired = 0`, so existing fusion quality tables remain proxy/no-reference/preliminary.\n\n"
        "## Required Capture Outputs\n\n"
        "- `nir_raw_path`: raw NIR frame.\n"
        "- `nir_enhanced_path`: enhanced NIR output for the same frame/cycle.\n"
        "- `thermal_raw_path`: raw thermal frame or numeric array.\n"
        "- `thermal_heatmap_path`: false-color thermal heatmap.\n"
        "- `thermal_mask_path`: foreground mask used for fusion.\n"
        "- `fusion_output_path`: final fused output.\n\n"
        "## Strict Pair Rule\n\n"
        "A strict pair must share `session_id`, use monotonic `frame_idx` or same-cycle capture, and have "
        "NIR/thermal/fusion timestamps within `<= 1s`. Same-frame or same-processing-cycle capture is preferred.\n\n"
        "## Scenario Checklist\n\n"
        "- normal night\n"
        "- NIR-dominant night\n"
        "- glare/backlight\n"
        "- fog/haze proxy if real fog is unavailable\n"
        "- thermal foreground target\n"
        "- no-target negative scene\n\n"
        "## Minimum Sample Recommendation\n\n"
        "Capture at least 20 strict paired frames per major condition and at least 5 sessions where feasible.\n\n"
        "## Sidecar JSON Schema\n\n"
        + markdown_table(metadata_rows, ["field", "required", "strict_pair_requirement"])
    )
    write_markdown(docs_dir / "STRICT_PAIRED_FUSION_CAPTURE_PROTOCOL.md", "Strict Paired Fusion Capture Protocol", body)


def _write_review_docs(docs_dir: Path) -> None:
    rows = [
        {
            "reviewer_concern": "Fusion method is too simple",
            "response": "Added alpha blend, mask-weighted, legacy gradient, and Laplacian pyramid baselines for offline comparison.",
            "evidence": "docs/tables/fusion/fusion_algorithm_comparison.md",
            "status": "proxy/preliminary because strict paired = 0",
        },
        {
            "reviewer_concern": "Missing image/fusion metrics",
            "response": "Added tiered IQA/fusion metrics with metric_tier, evidence_label, input_data_type, and caveat columns.",
            "evidence": "docs/fusion/IMAGE_FUSION_METRICS.md; docs/tables/fusion/*quality_summary.md",
            "status": "Tier 2/3 except runtime timing",
        },
        {
            "reviewer_concern": "No paired fusion validation",
            "response": "Current strict paired count is zero; thesis text must not claim strict/live fusion quality validation.",
            "evidence": "docs/fusion/FUSION_EVIDENCE_READINESS.md",
            "status": "limitation documented",
        },
        {
            "reviewer_concern": "Need path to stronger evidence",
            "response": "Added strict paired capture protocol with required sidecar fields and sample recommendations.",
            "evidence": "docs/fusion/STRICT_PAIRED_FUSION_CAPTURE_PROTOCOL.md",
            "status": "future Tier 1 protocol ready",
        },
    ]
    write_markdown(
        docs_dir / "REVIEW_RESPONSE_FUSION_MATRIX.md",
        "Review Response Fusion Matrix",
        markdown_table(rows, ["reviewer_concern", "response", "evidence", "status"]),
    )


def _append_report_patch_notes(repo_root: Path) -> None:
    path = repo_root / "docs/ml/REPORT_PATCH_NOTES.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    marker = "## Fusion/Image Processing Evidence Hardening"
    if marker in text:
        return
    addition = """

## Fusion/Image Processing Evidence Hardening

Add:

- "Fusion/image-processing tables now carry `metric_tier`, `evidence_label`, `input_data_type`, and `caveat` columns."
- "Current fusion-quality evidence is proxy/no-reference/preliminary because strict paired fusion samples are zero."
- "Runtime timing from real session JSON can be used as processing-cost evidence, with measured-vs-estimated p95 called out explicitly."
- "Failure-case mining is diagnostic; proxy/unpaired failures are not proof of runtime fusion failure."
- "A strict paired capture protocol defines required NIR raw/enhanced, thermal raw/heatmap/mask, fusion output, sidecar fields, and `<= 1s` strict-pair timing."

Avoid:

- "Fusion quality is strictly validated."
- "Proxy/no-reference fusion metrics are ground-truth perceptual quality."
- "Unpaired/proxy failure cases prove a live runtime failure."
"""
    path.write_text(text.rstrip() + addition + "\n", encoding="utf-8")


def _append_review_response_crosslink(repo_root: Path) -> None:
    path = repo_root / "docs/ml/REVIEW_RESPONSE_MATRIX.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    marker = "## Fusion Evidence Cross-Link"
    if marker in text:
        return
    addition = """

## Fusion Evidence Cross-Link

| Reviewer concern | Fusion evidence | Status |
| --- | --- | --- |
| Fusion/image-processing evaluation was weak | `docs/fusion/REVIEW_RESPONSE_FUSION_MATRIX.md`, `docs/fusion/FUSION_EVIDENCE_READINESS.md` | addressed with proxy/no-reference caveats; strict paired capture still future work |
"""
    path.write_text(text.rstrip() + addition + "\n", encoding="utf-8")


def _write_pipeline_audit(docs_dir: Path, alias_rows: list[dict[str, str]]) -> None:
    body = (
        "Report-facing names are aligned with the ML taxonomy. These are aliases for thesis/reporting only; "
        "production code symbols and schemas are not renamed.\n\n"
        + markdown_table(alias_rows, ["report_name", "category", "code_or_artifact_alias", "rename_policy"])
    )
    write_markdown(docs_dir / "IMAGE_PROCESSING_PIPELINE_AUDIT.md", "Image Processing Pipeline Audit", body)


def _write_fusion_evaluation(docs_dir: Path, strict_count: int) -> None:
    body = (
        f"Strict paired fusion samples: {strict_count}.\n\n"
        "Current fusion-quality evidence is proxy/no-reference/preliminary. It is useful for local diagnostic comparison "
        "against `alpha_blend_baseline`, `mask_weighted_blend`, `legacy_gradient_overlay`, and "
        "`laplacian_pyramid_fusion`, but it is not proof of real fusion quality because strict paired samples are absent.\n\n"
        "Runtime timing from real session JSON is usable as processing-cost evidence when p95 source and stage profiler "
        "coverage are reported.\n\n"
        "- Evidence readiness: `docs/fusion/FUSION_EVIDENCE_READINESS.md`\n"
        "- Baseline comparison: `docs/tables/fusion/fusion_algorithm_comparison.md`\n"
        "- Failure cases: `docs/tables/fusion/failure_case_summary.md`\n"
        "- Strict paired protocol: `docs/fusion/STRICT_PAIRED_FUSION_CAPTURE_PROTOCOL.md`\n"
    )
    write_markdown(docs_dir / "FUSION_EVALUATION.md", "Fusion Evaluation", body)


def _write_image_processing_evaluation(docs_dir: Path) -> None:
    body = (
        "Offline image-processing evaluation compares current processing against raw, CLAHE/gamma, and legacy/pre-optimization "
        "baselines. These rows are Tier 2 no-reference IQA or Tier 3 proxy evidence depending on input data type.\n\n"
        "Do not describe no-reference metrics as ground-truth perceptual quality. Use them as quantitative local evidence for "
        "contrast, brightness, sharpness, clipping, and noise-proxy behavior.\n\n"
        "- Image-processing comparison: `docs/tables/fusion/image_processing_algorithm_comparison.md`\n"
        "- NIR quality summary: `docs/tables/fusion/nir_quality_summary.md`\n"
        "- Thermal quality summary: `docs/tables/fusion/thermal_quality_summary.md`\n"
    )
    write_markdown(docs_dir / "IMAGE_PROCESSING_EVALUATION.md", "Image Processing Evaluation", body)


def _write_runtime_docs(docs_dir: Path, runtime_rows: list[dict[str, str]]) -> None:
    measured = sum(1 for row in runtime_rows if row.get("measured_p95"))
    estimated = sum(1 for row in runtime_rows if row.get("estimated_p95"))
    missing_profiler = sum(1 for row in runtime_rows if row.get("stage_profiler_available") != "yes")
    body = (
        f"Runtime rows: {len(runtime_rows)}. Measured p95 rows: {measured}. Estimated p95 rows: {estimated}. "
        f"Rows missing stage profiler fields: {missing_profiler}.\n\n"
        "Timing evidence is Tier 1 processing-cost evidence when derived from real session JSON. If `estimated_p95` is set, "
        "the source did not record per-frame p95 and the value was estimated from mean/std. Stage timing fields are reported "
        "only when present in the session JSON.\n\n"
        + markdown_table(runtime_rows[:80], RUNTIME_COLUMNS)
    )
    write_markdown(docs_dir / "RUNTIME_TIMING_EVIDENCE.md", "Runtime Timing Evidence", body)


def _write_evidence_docs(docs_dir: Path, tables_dir: Path, rows: list[dict[str, str]]) -> None:
    columns = [
        "artifact",
        "exists",
        "metric_tier",
        "evidence_label",
        "input_data_type",
        "metric_type",
        "thesis_usability",
        "caveat",
        "missing_evidence",
        "provenance_source",
    ]
    write_markdown(tables_dir / "evidence_readiness_matrix.md", "Evidence Readiness Matrix", markdown_table(rows, columns))
    body = (
        "This audit separates thesis-ready evidence from proxy/preliminary diagnostics.\n\n"
        "Strong fusion-quality claims require Tier 1 strict paired/task-specific evidence. The current local fusion set has "
        "`strict paired = 0`, so fusion quality results remain proxy/no-reference/preliminary. Runtime timing from real session "
        "JSON is usable as processing-cost evidence when p95 source and profiler coverage are stated.\n\n"
        + markdown_table(rows, columns)
    )
    write_markdown(docs_dir / "FUSION_EVIDENCE_READINESS.md", "Fusion Evidence Readiness", body)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    artifacts_dir = _resolve(repo_root, args.artifacts_dir)
    docs_dir = _resolve(repo_root, args.docs_dir)
    tables_dir = _resolve(repo_root, args.tables_dir)
    figures_dir = _resolve(repo_root, args.figures_dir)
    session_metrics_dir = _resolve(repo_root, args.session_metrics_dir)

    if args.dry_run:
        print(f"repo_root={repo_root}")
        print(f"artifacts_dir={artifacts_dir}")
        print(f"docs_dir={docs_dir}")
        print(f"tables_dir={tables_dir}")
        print(f"figures_dir={figures_dir}")
        return

    alias_rows = build_processing_name_alias_rows()
    write_markdown(
        tables_dir / "processing_name_aliases.md",
        "Processing Name Aliases",
        markdown_table(alias_rows, ["report_name", "category", "code_or_artifact_alias", "rename_policy"]),
    )

    summary_specs = [
        ("nir_quality_summary.csv", tables_dir / "nir_quality_summary.md", "NIR Quality Summary", "Tier 2 no-reference IQA and Tier 3 proxy rows; not ground-truth perceptual quality."),
        ("thermal_quality_summary.csv", tables_dir / "thermal_quality_summary.md", "Thermal Quality Summary", "Thermal rows include local captures/proxy sequences; proxy/synthetic rows are not real thermal validation."),
        ("fusion_quality_summary.csv", tables_dir / "fusion_quality_summary.md", "Fusion Quality Summary", "Fusion rows are proxy/no-reference/preliminary unless strict paired evidence exists."),
        ("image_algorithms/image_processing_algorithm_comparison.csv", tables_dir / "image_processing_algorithm_comparison.md", "Image Processing Algorithm Comparison", "Current-vs-baseline image-processing deltas are offline proxy/no-reference evidence."),
        ("fusion_algorithms/fusion_algorithm_comparison.csv", tables_dir / "fusion_algorithm_comparison.md", "Fusion Algorithm Comparison", "Current fusion comparisons are proxy only because strict paired fusion samples are zero."),
    ]
    for csv_rel, md_path, title, intro in summary_specs:
        csv_path = artifacts_dir / csv_rel
        rows = read_csv_rows(csv_path)
        if not rows:
            continue
        enriched = enrich_summary_rows(rows)
        write_csv(csv_path, enriched)
        columns = [col for col in SUMMARY_COLUMNS if any(col in row for row in enriched)]
        _write_summary_table(md_path, title, enriched[:300], intro=intro, columns=columns)

    runtime_source_rows = parse_session_runtime_json(session_metrics_dir, repo_root)
    if not runtime_source_rows:
        runtime_source_rows = read_csv_rows(artifacts_dir / "runtime_timing_summary.csv")
    runtime_rows = normalize_runtime_timing_rows(runtime_source_rows)
    write_csv(artifacts_dir / "runtime_timing_summary.csv", runtime_rows, RUNTIME_COLUMNS)
    _write_summary_table(
        tables_dir / "runtime_timing_summary.md",
        "Fusion Runtime Timing Summary",
        runtime_rows,
        intro="Runtime timing is Tier 1 cost evidence from real session JSON. `estimated_p95` is derived from mean/std when measured p95 is absent.",
        columns=RUNTIME_COLUMNS,
    )
    _write_runtime_docs(docs_dir, runtime_rows)

    failure_rows = _failure_rows_from_sources(artifacts_dir)
    failure_columns = [
        "failure_type",
        "severity",
        "pair_id",
        "algorithm",
        "baseline_algorithm",
        "evidence_label",
        "input_data_type",
        "metric_tier",
        "source_or_session",
        "metric_trigger",
        "caveat",
    ]
    write_csv(artifacts_dir / "failure_cases.csv", failure_rows, failure_columns)
    failure_summary = _summarize_failure_rows(failure_rows)
    failure_body = (
        "Failure mining is diagnostic. Proxy, weak, and unpaired rows must not be used to claim a real runtime fusion failure.\n\n"
        + markdown_table(
            failure_summary,
            ["failure_type", "algorithm", "evidence_label", "input_data_type", "metric_tier", "count", "caveat"],
        )
    )
    if failure_rows:
        failure_body += "\n## Examples\n\n" + markdown_table(failure_rows[:50], failure_columns)
    write_markdown(tables_dir / "failure_case_summary.md", "Fusion Failure Case Summary", failure_body)
    _write_failure_grid(failure_rows, figures_dir / "failure_cases_grid.png", max_cases=args.max_failure_grid_cases)

    _write_protocol_docs(docs_dir, tables_dir)
    _write_review_docs(docs_dir)
    _write_pipeline_audit(docs_dir, alias_rows)
    _write_fusion_evaluation(docs_dir, strict_pair_count(artifacts_dir))
    _write_image_processing_evaluation(docs_dir)
    readiness_rows = build_evidence_readiness_rows(
        repo_root=repo_root,
        artifacts_dir=artifacts_dir,
        docs_dir=docs_dir,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
    )
    _write_evidence_docs(docs_dir, tables_dir, readiness_rows)
    _append_report_patch_notes(repo_root)
    _append_review_response_crosslink(repo_root)
    write_json(
        artifacts_dir / "hardening_run_manifest.json",
        run_manifest(
            "python3 tools/harden_fusion_evidence.py",
            [
                str(artifacts_dir / "pairing_manifest.csv"),
                str(artifacts_dir / "fusion_pair_metrics.csv"),
                str(artifacts_dir / "runtime_timing_summary.csv"),
                str(session_metrics_dir),
            ],
            {"git_commit": git_commit(), "strict_paired": strict_pair_count(artifacts_dir)},
        ),
    )
    print(
        "fusion_hardening_complete "
        f"strict_paired={strict_pair_count(artifacts_dir)} "
        f"failure_cases={len(failure_rows)} "
        f"runtime_rows={len(runtime_rows)}"
    )


if __name__ == "__main__":
    main()
