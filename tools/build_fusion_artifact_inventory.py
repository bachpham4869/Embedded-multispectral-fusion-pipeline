#!/usr/bin/env python3
"""Build offline fusion/image artifact inventory and pairing manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from fusion_eval_manifest import (  # type: ignore[import]
    discover_artifacts,
    format_pairing_manifest_rows,
    markdown_table,
    pair_capture_records,
    run_manifest,
    write_csv,
    write_json,
    write_markdown,
)


DEFAULT_SCAN_ROOTS = [
    "fusion_captures",
    "data/eval",
    "data/thermal_sequences",
    "docs/thesis_eval",
    "docs/tables/iqa",
    "docs/tables/timing",
    "q1_results",
    "legacy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory SmartBinocular image/fusion artifacts and create a pairing manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--scan-root", action="append", default=None, help="Relative or absolute root to scan; repeatable.")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/fusion_eval"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs/tables/fusion"))
    parser.add_argument("--fusion-doc-dir", type=Path, default=Path("docs/fusion"))
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--strict-window-sec", type=float, default=1.0)
    parser.add_argument("--qual-window-sec", type=float, default=20.0)
    return parser.parse_args()


def _resolve_roots(repo_root: Path, scan_roots: list[str] | None) -> list[Path]:
    roots = scan_roots or DEFAULT_SCAN_ROOTS
    resolved = []
    for root in roots:
        path = Path(root)
        if not path.is_absolute():
            path = repo_root / path
        if path.exists():
            resolved.append(path)
    return resolved


def _inventory_rows(records) -> list[dict[str, object]]:
    counts: Counter[tuple[str, str, str, str]] = Counter()
    sizes: Counter[tuple[str, str, str, str]] = Counter()
    for record in records:
        key = (record.kind, record.modality, record.evidence_label, record.extension)
        counts[key] += 1
        sizes[key] += int(record.size_bytes)
    return [
        {
            "kind": kind,
            "modality": modality,
            "evidence_label": evidence,
            "extension": extension,
            "count": count,
            "size_mb": round(sizes[(kind, modality, evidence, extension)] / (1024 * 1024), 3),
        }
        for (kind, modality, evidence, extension), count in sorted(counts.items())
    ]


def _pair_summary_rows(pair_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    counts: Counter[tuple[str, str]] = Counter(
        (str(row.get("pair_status", "")), str(row.get("evidence_label", ""))) for row in pair_rows
    )
    return [
        {"pair_status": status, "evidence_label": evidence, "count": count}
        for (status, evidence), count in sorted(counts.items())
    ]


def _mean_stage(stage_map: dict, preferred: list[str]) -> float | str:
    for key in preferred:
        item = stage_map.get(key)
        if isinstance(item, dict) and "mean_ms" in item:
            return item["mean_ms"]
    return ""


def _runtime_summary_rows(repo_root: Path) -> list[dict[str, object]]:
    session_files = sorted((repo_root / "fusion_captures/metrics").glob("session_*.json"))
    session_files += sorted((repo_root / "fusion_captures/metrics_rpi_optimized").glob("session_*.json"))
    rows: list[dict[str, object]] = []
    seen_sessions: set[str] = set()
    for path in session_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        session_key = str(data.get("session_id") or path.stem)
        if session_key in seen_sessions:
            continue
        seen_sessions.add(session_key)
        stage = data.get("stage_timing_ms") or {}
        fuse_stage = data.get("fuse_stage_timing_ms") or {}
        thermal_stage = data.get("thermal_stage_timing_ms") or {}
        frames_by_mode = data.get("frames_by_mode") or {}
        mode = max(frames_by_mode, key=frames_by_mode.get) if frames_by_mode else "mixed"
        fps_p95 = data.get("fps_p95", "")
        fps_p95_source = "recorded" if fps_p95 != "" else ""
        if fps_p95 == "" and data.get("fps_mean") not in ("", None) and data.get("fps_std_sample") not in ("", None):
            try:
                fps_p95 = round(float(data["fps_mean"]) + 1.645 * float(data["fps_std_sample"]), 3)
                fps_p95_source = "estimated_mean_plus_1.645std"
            except (TypeError, ValueError):
                fps_p95 = ""
        rows.append(
            {
                "session_file": str(path.relative_to(repo_root)),
                "source_or_session": session_key,
                "mode": mode,
                "evidence_label": "real_paired",
                "duration_wall_s": data.get("duration_wall_s", ""),
                "frames_total": data.get("frames_total", ""),
                "fps_mean": data.get("fps_mean", ""),
                "fps_p95": fps_p95,
                "fps_p95_source": fps_p95_source,
                "nir_processing_latency_ms": _mean_stage(stage, ["nir_bucket", "nir"]),
                "thermal_processing_latency_ms": _mean_stage(stage, ["thermal_proc", "thermal"])
                or _mean_stage(thermal_stage, ["thermal_3dnr", "thermal_agc"]),
                "fusion_composite_latency_ms": _mean_stage(stage, ["fusion_composite", "blend"])
                or _mean_stage(fuse_stage, ["fuse_blend_math", "fuse_warp_perspective"]),
                "stage_profiler_summary": "; ".join(sorted(set(stage.keys()) | set(fuse_stage.keys()) | set(thermal_stage.keys())))[:300],
            }
        )
    return rows


def _write_static_docs(repo_root: Path, docs_dir: Path, fusion_doc_dir: Path) -> None:
    metric_rows = [
        {"metric": "foreground_contrast_gain", "tier": "1", "meaning": "Task-specific target contrast gain on strict paired captures", "limitation": "Only strong if strict paired masks are valid"},
        {"metric": "fps_mean/fps_p95/stage_latency", "tier": "1", "meaning": "Real session runtime evidence", "limitation": "Session duration and thermal state still matter"},
        {"metric": "entropy/rms/tenengrad/laplacian/clipping/noise", "tier": "2", "meaning": "No-reference IQA statistics", "limitation": "Not absolute perceptual quality"},
        {"metric": "MI/NMI/SSIM/Qabf-style edge proxy", "tier": "3", "meaning": "Proxy fusion/preservation statistics", "limitation": "Not proof of real fusion quality"},
    ]
    write_markdown(
        docs_dir / "metric_definitions.md",
        "Fusion Metric Definitions",
        "Strong thesis claims are allowed only for Tier 1. Tier 2/3 must carry limitations.\n\n"
        + markdown_table(metric_rows, ["metric", "tier", "meaning", "limitation"]),
    )
    alg_rows = [
        {"taxonomy_name": "night_hybrid_enhance", "legacy_or_current": "current", "input": "NIR BGR", "output": "enhanced NIR", "latency_risk": "high"},
        {"taxonomy_name": "nir_mono_clahe", "legacy_or_current": "current", "input": "NIR BGR", "output": "CLAHE NIR", "latency_risk": "low"},
        {"taxonomy_name": "highlight_tone_map", "legacy_or_current": "current", "input": "glare/highlight NIR", "output": "compressed highlights", "latency_risk": "low"},
        {"taxonomy_name": "fog_dehaze_lite", "legacy_or_current": "current", "input": "fog NIR/RGB proxy", "output": "dehazed-lite frame", "latency_risk": "medium"},
        {"taxonomy_name": "rain_temporal_median", "legacy_or_current": "current", "input": "short sequence", "output": "temporal median", "latency_risk": "medium"},
        {"taxonomy_name": "dawn_dusk_blend", "legacy_or_current": "current", "input": "transition NIR", "output": "A/C blend", "latency_risk": "medium"},
        {"taxonomy_name": "legacy_gradient_overlay", "legacy_or_current": "legacy/pre-optimization", "input": "NIR + heat map", "output": "gradient thermal overlay", "latency_risk": "medium"},
    ]
    write_markdown(
        docs_dir / "current_vs_legacy_algorithms.md",
        "Current vs Legacy Algorithms",
        markdown_table(alg_rows, ["taxonomy_name", "legacy_or_current", "input", "output", "latency_risk"]),
    )
    fusion_survey = [
        {"mode": "nir_only_baseline", "implemented": "yes", "dependency": "OpenCV/NumPy", "rpi_suitability": "high", "evaluated": "yes"},
        {"mode": "thermal_heatmap_only", "implemented": "yes", "dependency": "OpenCV", "rpi_suitability": "high", "evaluated": "yes"},
        {"mode": "alpha_blend_baseline", "implemented": "yes", "dependency": "OpenCV", "rpi_suitability": "high", "evaluated": "yes"},
        {"mode": "foreground_mask_overlay", "implemented": "current", "dependency": "OpenCV", "rpi_suitability": "high", "evaluated": "yes"},
        {"mode": "mask_weighted_blend", "implemented": "offline baseline", "dependency": "OpenCV/NumPy", "rpi_suitability": "high", "evaluated": "yes"},
        {"mode": "legacy_gradient_overlay", "implemented": "legacy comparable", "dependency": "OpenCV", "rpi_suitability": "medium", "evaluated": "yes"},
        {"mode": "laplacian_pyramid_fusion", "implemented": "offline baseline", "dependency": "OpenCV/NumPy", "rpi_suitability": "medium", "evaluated": "yes"},
        {"mode": "DWT/guided/deep fusion", "implemented": "literature only", "dependency": "extra/heavy", "rpi_suitability": "low-medium", "evaluated": "no"},
    ]
    enhancement_survey = [
        {"algorithm": "raw passthrough", "implemented": "yes", "evaluated": "yes", "role": "baseline"},
        {"algorithm": "CLAHE", "implemented": "yes", "evaluated": "yes", "role": "simple baseline"},
        {"algorithm": "gamma correction", "implemented": "yes", "evaluated": "yes", "role": "simple baseline"},
        {"algorithm": "Retinex/DCP/dehaze-lite", "implemented": "current/offline", "evaluated": "yes", "role": "fog/low-light baseline"},
        {"algorithm": "legacy night vision", "implemented": "legacy import when safe", "evaluated": "if importable", "role": "pre-optimization comparison"},
    ]
    write_markdown(docs_dir / "fusion_algorithm_survey.md", "Fusion Algorithm Survey", markdown_table(fusion_survey, ["mode", "implemented", "dependency", "rpi_suitability", "evaluated"]))
    write_markdown(docs_dir / "enhancement_algorithm_survey.md", "Enhancement Algorithm Survey", markdown_table(enhancement_survey, ["algorithm", "implemented", "evaluated", "role"]))
    write_markdown(
        fusion_doc_dir / "IMAGE_FUSION_METRICS.md",
        "Image Fusion Metrics",
        "Metric validity tiers govern thesis wording. Tier 1 direct paired/task-specific metrics can support strong claims; "
        "Tier 2 no-reference IQA and Tier 3 proxy/synthetic/unpaired metrics must be reported with limitations.\n\n"
        + markdown_table(metric_rows, ["metric", "tier", "meaning", "limitation"]),
    )
    write_markdown(
        fusion_doc_dir / "IMAGE_PROCESSING_PIPELINE_AUDIT.md",
        "Image Processing Pipeline Audit",
        "Current taxonomy-aligned buckets and legacy comparison surfaces:\n\n"
        + markdown_table(alg_rows, ["taxonomy_name", "legacy_or_current", "input", "output", "latency_risk"]),
    )
    write_markdown(
        fusion_doc_dir / "FUSION_ALGORITHM_SURVEY.md",
        "Fusion Algorithm Survey",
        markdown_table(fusion_survey, ["mode", "implemented", "dependency", "rpi_suitability", "evaluated"]),
    )
    write_markdown(
        fusion_doc_dir / "FUSION_EVALUATION.md",
        "Fusion Evaluation",
        "This evaluation separates strict paired evidence from proxy-only stress tests. If strict paired fusion samples are too few, "
        "all proxy rows are explicitly labeled: proxy only, not proof of real fusion quality, requires future paired capture validation.\n\n"
        "- See `docs/tables/fusion/fusion_quality_summary.md` for per-metric summaries.\n"
        "- See `docs/tables/fusion/fusion_algorithm_comparison.md` for current-vs-baseline comparisons.\n"
        "- See `docs/tables/fusion/failure_case_summary.md` for failure mining.\n",
    )
    write_markdown(
        fusion_doc_dir / "IMAGE_PROCESSING_EVALUATION.md",
        "Image Processing Evaluation",
        "Offline still-image evaluation compares current processing with raw, CLAHE/gamma, and legacy/pre-optimization baselines. "
        "These are Tier 2/3 proxy results unless paired task-specific evidence is present.\n\n"
        "- See `docs/tables/fusion/image_processing_algorithm_comparison.md`.\n"
        "- See `docs/tables/fusion/nir_quality_summary.md` and `docs/tables/fusion/thermal_quality_summary.md`.\n",
    )
    reviewer_rows = [
        {"reviewer_question": "Fusion hiện tại là gì?", "response": "foreground_mask_overlay is the current mode; baselines include alpha_blend_baseline and legacy_gradient_overlay.", "evidence": "fusion_algorithm_comparison.md"},
        {"reviewer_question": "Có so với baseline không?", "response": "Yes: raw/NIR-only, thermal_heatmap_only, alpha blend, mask-weighted, legacy gradient, Laplacian pyramid where available.", "evidence": "fusion_algorithm_comparison.md"},
        {"reviewer_question": "Metric nào chứng minh cải thiện?", "response": "Only Tier 1 strict paired/task metrics support strong claims; Tier 2/3 are proxy/no-reference.", "evidence": "metric_definitions.md"},
        {"reviewer_question": "Limitation là gì?", "response": "Proxy/unpaired/synthetic rows are not proof of real fusion quality and require future paired capture validation.", "evidence": "FUSION_EVALUATION.md"},
    ]
    write_markdown(
        docs_dir / "reviewer_response_fusion_matrix.md",
        "Reviewer Response Fusion Matrix",
        markdown_table(reviewer_rows, ["reviewer_question", "response", "evidence"]),
    )


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    out_dir = repo_root / args.out_dir
    docs_dir = repo_root / args.docs_dir
    fusion_doc_dir = repo_root / args.fusion_doc_dir
    roots = _resolve_roots(repo_root, args.scan_root)
    records = discover_artifacts(roots, max_depth=args.max_depth)
    pairs = pair_capture_records(
        records,
        strict_window_sec=args.strict_window_sec,
        qualitative_window_sec=args.qual_window_sec,
    )
    pair_rows = format_pairing_manifest_rows(pairs)
    inventory_rows = _inventory_rows(records)
    pair_summary = _pair_summary_rows(pair_rows)
    runtime_rows = _runtime_summary_rows(repo_root)

    write_csv(out_dir / "pairing_manifest.csv", pair_rows)
    write_csv(out_dir / "artifact_inventory.csv", [r.__dict__ | {"path": str(r.path), "sidecar_path": str(r.sidecar_path or "")} for r in records])
    write_csv(out_dir / "runtime_timing_summary.csv", runtime_rows)
    write_markdown(
        docs_dir / "artifact_inventory.md",
        "Fusion Artifact Inventory",
        "Evidence labels: `real_paired`, `proxy`, `unpaired`, `synthetic`, `unknown`.\n\n"
        + markdown_table(inventory_rows, ["kind", "modality", "evidence_label", "extension", "count", "size_mb"]),
    )
    write_markdown(
        docs_dir / "pairing_manifest_summary.md",
        "Fusion Pairing Manifest Summary",
        "Strict paired rows are eligible for Tier 1 fusion claims. Qualitative/proxy/unpaired rows are not.\n\n"
        + markdown_table(pair_summary, ["pair_status", "evidence_label", "count"]),
    )
    write_markdown(
        docs_dir / "runtime_timing_summary.md",
        "Fusion Runtime Timing Summary",
        "Runtime timing is Tier 1 session evidence when derived from real session JSON. "
        "`fps_p95_source=estimated_mean_plus_1.645std` means the source JSON did not record per-frame p95.\n\n"
        + markdown_table(
            runtime_rows,
            [
                "session_file",
                "mode",
                "duration_wall_s",
                "frames_total",
                "fps_mean",
                "fps_p95",
                "fps_p95_source",
                "nir_processing_latency_ms",
                "thermal_processing_latency_ms",
                "fusion_composite_latency_ms",
            ],
        ),
    )
    audit_body = (
        "## Inputs\n\n"
        + "\n".join(f"- `{path.relative_to(repo_root)}`" if path.is_relative_to(repo_root) else f"- `{path}`" for path in roots)
        + "\n\n## Pairing Policy\n\n"
        + f"- Strict paired: same-session/time nearest with gap <= {args.strict_window_sec}s.\n"
        + f"- Qualitative weak: nearest gap <= {args.qual_window_sec}s or incomplete thermal/NIR evidence.\n"
        + "- Proxy/unpaired/synthetic: usable for stress testing only, not proof of real fusion quality.\n\n"
        + "## Inventory Summary\n\n"
        + markdown_table(inventory_rows, ["kind", "modality", "evidence_label", "extension", "count", "size_mb"])
        + "\n## Pairing Summary\n\n"
        + markdown_table(pair_summary, ["pair_status", "evidence_label", "count"])
    )
    write_markdown(fusion_doc_dir / "FUSION_EVAL_DATA_AUDIT.md", "Fusion Evaluation Data Audit", audit_body)
    _write_static_docs(repo_root, docs_dir, fusion_doc_dir)
    write_json(
        out_dir / "run_manifest.json",
        run_manifest(
            "python3 tools/build_fusion_artifact_inventory.py",
            [str(path) for path in roots],
            {
                "strict_window_sec": args.strict_window_sec,
                "qual_window_sec": args.qual_window_sec,
                "max_depth": args.max_depth,
            },
        ),
    )
    print(f"records={len(records)} pairs={len(pair_rows)}")


if __name__ == "__main__":
    main()
