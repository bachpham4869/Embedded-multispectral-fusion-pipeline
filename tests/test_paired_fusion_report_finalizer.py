from __future__ import annotations

import csv
from pathlib import Path

from tools.finalize_paired_fusion_evidence import (
    FINAL_SUMMARY_COLUMNS,
    build_alignment_diagnostics,
    build_evidence_readiness_rows,
    build_final_result_rows,
    build_per_bucket_report_rows,
    captured_runtime_fusion_available,
    write_final_outputs,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _minimal_inputs(tmp_path: Path) -> tuple[Path, Path]:
    repo_root = tmp_path
    paired_dir = repo_root / "artifacts" / "paired_eval"
    tables_dir = repo_root / "docs" / "tables" / "fusion"
    docs_dir = repo_root / "docs" / "fusion"
    paired_docs_dir = repo_root / "docs" / "paired"
    figures_dir = repo_root / "docs" / "figures" / "fusion"
    tables_dir.mkdir(parents=True)
    docs_dir.mkdir(parents=True)
    paired_docs_dir.mkdir(parents=True)
    figures_dir.mkdir(parents=True)

    manifest_rows = [
        {
            "pair_id": "paired_000000",
            "pairing_tier": "frame_strict",
            "pairing_gap_ms": "30",
            "timestamp_sec": "1.0",
            "nir_raw_path": "data/paired_data/imx_paired.mp4#frame=0",
            "thermal_raw_path": "data/paired_data/thermal_paired.mp4#frame=0",
            "fusion_output_path": "",
            "homography_path": "",
            "thermal_modality": "display_heatmap_like",
            "input_data_type": "paired NIR video + thermal display/heatmap-like video",
            "evidence_label": "real_paired",
            "processing_bucket": "unknown",
            "processing_bucket_source": "unknown",
            "fusion_source": "none",
            "source_or_session": "paired_data",
        },
        {
            "pair_id": "paired_000001",
            "pairing_tier": "frame_strict",
            "pairing_gap_ms": "40",
            "timestamp_sec": "1.1",
            "nir_raw_path": "data/paired_data/imx_paired.mp4#frame=1",
            "thermal_raw_path": "data/paired_data/thermal_paired.mp4#frame=1",
            "fusion_output_path": "",
            "homography_path": "",
            "thermal_modality": "display_heatmap_like",
            "input_data_type": "paired NIR video + thermal display/heatmap-like video",
            "evidence_label": "real_paired",
            "processing_bucket": "unknown",
            "processing_bucket_source": "unknown",
            "fusion_source": "none",
            "source_or_session": "paired_data",
        },
    ]
    _write_csv(paired_dir / "strict_paired_manifest.csv", manifest_rows)
    _write_csv(
        paired_dir / "paired_data_inventory.csv",
        [
            {
                "path": "data/paired_data/imx_paired.mp4",
                "kind": "video",
                "width": "640",
                "height": "480",
                "frame_count": "2",
                "fps": "9.0",
            },
            {
                "path": "data/paired_data/thermal_paired.mp4",
                "kind": "video",
                "width": "640",
                "height": "496",
                "frame_count": "2",
                "fps": "9.0",
            },
        ],
    )
    common = {
        "pairing_tier": "frame_strict",
        "evidence_label": "real_paired",
        "input_data_type": "paired NIR video + thermal display/heatmap-like video",
        "thermal_modality": "display_heatmap_like",
        "source_or_session": "paired_data",
        "metric_tier": "Tier 2",
        "processing_bucket": "unknown",
        "processing_bucket_source": "unknown",
        "caveat": "thermal_modality=display_heatmap_like; not raw radiometric thermal",
    }
    _write_csv(
        paired_dir / "fusion_quality_metrics.csv",
        [
            {
                **common,
                "pair_id": "paired_000000",
                "algorithm": "alpha_blend_baseline",
                "baseline_algorithm": "alpha_blend_baseline",
                "fusion_source": "paired_generated_fusion",
                "metric": "entropy",
                "value": "6.0",
                "baseline_value": "6.0",
            },
            {
                **common,
                "pair_id": "paired_000000",
                "algorithm": "foreground_mask_overlay",
                "baseline_algorithm": "alpha_blend_baseline",
                "fusion_source": "paired_generated_fusion",
                "metric": "entropy",
                "value": "5.0",
                "baseline_value": "6.0",
            },
            {
                **common,
                "pair_id": "paired_000000",
                "algorithm": "foreground_mask_overlay",
                "baseline_algorithm": "alpha_blend_baseline",
                "fusion_source": "paired_generated_fusion",
                "metric": "foreground_contrast_gain",
                "metric_tier": "Tier 3",
                "value": "2.0",
                "baseline_value": "-1.0",
            },
        ],
    )
    _write_csv(
        paired_dir / "nir_quality_metrics.csv",
        [
            {
                **common,
                "pair_id": "paired_000000",
                "algorithm": "nir_mono_clahe",
                "baseline_algorithm": "raw_passthrough",
                "fusion_source": "none",
                "processing_bucket": "nir_mono_clahe",
                "processing_bucket_source": "algorithm_forced",
                "metric": "entropy",
                "value": "5.5",
                "baseline_value": "4.5",
            }
        ],
    )
    _write_csv(
        paired_dir / "thermal_quality_metrics.csv",
        [
            {
                **common,
                "pair_id": "paired_000000",
                "algorithm": "thermal_heatmap",
                "baseline_algorithm": "thermal_heatmap",
                "fusion_source": "none",
                "metric": "dynamic_range",
                "value": "140.0",
                "baseline_value": "140.0",
            }
        ],
    )
    _write_csv(
        paired_dir / "strict_paired_failure_cases.csv",
        [
            {
                "failure_type": "fusion_target_faded",
                "algorithm": "alpha_blend_baseline",
                "pairing_tier": "frame_strict",
                "fusion_source": "paired_generated_fusion",
                "evidence_tier": "Tier 3",
                "thermal_modality": "display_heatmap_like",
                "input_data_type": "paired NIR video + thermal display/heatmap-like video",
                "caveat": "Generated/proxy failures are diagnostic only",
            }
        ],
    )
    return repo_root, paired_dir


def test_captured_runtime_fusion_detection_is_false_for_generated_only(tmp_path: Path):
    _, paired_dir = _minimal_inputs(tmp_path)
    assert not captured_runtime_fusion_available(
        paired_dir / "strict_paired_manifest.csv",
        paired_dir / "fusion_quality_metrics.csv",
    )


def test_final_summary_rows_include_required_guardrails_and_entropy_caveat(tmp_path: Path):
    _, paired_dir = _minimal_inputs(tmp_path)
    rows = build_final_result_rows(paired_dir)

    assert rows
    for row in rows:
        for column in FINAL_SUMMARY_COLUMNS:
            assert column in row
    fusion_rows = [row for row in rows if "foreground_mask_overlay" in row["evaluation_item"]]
    assert fusion_rows
    assert all(row["captured_runtime_fusion_available"] == "false" for row in fusion_rows)
    assert any(row["fusion_source"] == "paired_generated_fusion" for row in fusion_rows)
    entropy_rows = [row for row in rows if "entropy" in row["key_metric"]]
    assert entropy_rows
    assert any("ambiguous" in row["caveat"] for row in entropy_rows)
    assert all(row["thesis_wording"] in {"strong", "caveated", "preliminary", "not measured"} for row in rows)


def test_per_bucket_report_rows_distinguish_forced_offline_algorithm(tmp_path: Path):
    _, paired_dir = _minimal_inputs(tmp_path)
    rows = build_per_bucket_report_rows(paired_dir)

    clahe = [row for row in rows if row["processing_bucket"] == "nir_mono_clahe"][0]
    assert clahe["processing_bucket_source"] == "algorithm_forced"
    assert clahe["bucket_evidence_status"] == "forced_offline_algorithm"
    assert "not runtime bucket performance" in clahe["caveat"]
    rain = [row for row in rows if row["processing_bucket"] == "rain_temporal_median"][0]
    assert rain["bucket_evidence_status"] == "not_measured"
    assert rain["thesis_wording"] == "not measured"


def test_alignment_diagnostics_report_missing_homography_and_resolution_mismatch(tmp_path: Path):
    _, paired_dir = _minimal_inputs(tmp_path)
    rows = build_alignment_diagnostics(paired_dir)

    lookup = {row["diagnostic"]: row for row in rows}
    assert lookup["frame_skew_ms"]["measured_p95"] == "39.500000"
    assert lookup["resolution_mismatch"]["status"] == "measured"
    assert "640x480 vs 640x496" in lookup["resolution_mismatch"]["result"]
    assert lookup["homography_alignment_metadata"]["status"] == "not_measured"
    assert "Do not claim alignment quality" in lookup["homography_alignment_metadata"]["caveat"]


def test_write_final_outputs_rewrites_stale_strict_zero_claim(tmp_path: Path):
    repo_root, paired_dir = _minimal_inputs(tmp_path)
    stale = repo_root / "docs" / "fusion" / "FUSION_EVIDENCE_READINESS.md"
    stale.write_text("Strict paired fusion samples: 0.\n", encoding="utf-8")

    write_final_outputs(repo_root=repo_root, paired_eval_dir=paired_dir, max_visual_review_samples=2)

    text = stale.read_text(encoding="utf-8")
    assert "Strict paired rows: `2`" in text
    assert "Strict paired fusion samples: 0" not in text
    assert "captured_runtime_fusion_available=false" in text
