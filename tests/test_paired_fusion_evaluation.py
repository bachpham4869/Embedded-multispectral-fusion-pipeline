from __future__ import annotations

import numpy as np

from tools.evaluate_paired_fusion import (
    determine_metric_tier,
    failure_tier_for_type,
    fusion_candidates,
    metric_rows_for_pair,
    summarize_metric_rows_by_group,
)


def test_fusion_candidates_mark_generated_source_and_required_modes():
    nir = np.full((16, 16, 3), 80, dtype=np.uint8)
    thermal = np.zeros((16, 16, 3), dtype=np.uint8)
    thermal[4:12, 4:12] = 220

    candidates = fusion_candidates(nir, thermal, captured_fusion=None)

    assert "alpha_blend_baseline" in candidates
    assert "foreground_mask_overlay" in candidates
    assert "mask_weighted_blend" in candidates
    assert candidates["foreground_mask_overlay"].fusion_source == "paired_generated_fusion"


def test_captured_fusion_is_not_marked_as_generated():
    nir = np.full((8, 8, 3), 50, dtype=np.uint8)
    thermal = np.full((8, 8, 3), 100, dtype=np.uint8)
    captured = np.full((8, 8, 3), 150, dtype=np.uint8)

    candidates = fusion_candidates(nir, thermal, captured_fusion=captured)

    assert candidates["foreground_mask_overlay"].fusion_source == "paired_captured_fusion"


def test_metric_rows_include_guardrail_columns_and_no_tier1_for_generated_fusion():
    nir = np.tile(np.arange(16, dtype=np.uint8), (16, 1))
    thermal = np.flipud(nir)
    candidates = fusion_candidates(nir, thermal, captured_fusion=None)

    rows = metric_rows_for_pair(
        pair_id="paired_000001",
        nir=nir,
        thermal=thermal,
        candidates=candidates,
        pairing_tier="frame_strict",
        thermal_modality="display_heatmap_like",
        input_data_type="paired NIR video + thermal display/heatmap-like video",
        processing_bucket="unknown",
        processing_bucket_source="unknown",
        source_or_session="paired_data",
    )

    assert rows
    foreground_rows = [row for row in rows if row["algorithm"] == "foreground_mask_overlay"]
    assert foreground_rows
    assert all(row["fusion_source"] == "paired_generated_fusion" for row in foreground_rows)
    assert all(row["thermal_modality"] == "display_heatmap_like" for row in foreground_rows)
    assert all(row["metric_tier"] in {"Tier 2", "Tier 3"} for row in foreground_rows)
    assert any("not runtime-captured fusion" in row["caveat"] for row in foreground_rows)


def test_determine_metric_tier_keeps_task_metric_tier1_only_for_captured_strict_pairs():
    assert (
        determine_metric_tier("foreground_contrast_gain", "frame_strict", "paired_captured_fusion", "real_paired")
        == "Tier 1"
    )
    assert (
        determine_metric_tier("foreground_contrast_gain", "frame_strict", "paired_generated_fusion", "real_paired")
        == "Tier 3"
    )
    assert determine_metric_tier("entropy", "frame_strict", "paired_generated_fusion", "real_paired") == "Tier 2"
    assert determine_metric_tier("entropy", "weak_paired", "paired_generated_fusion", "proxy") == "Tier 3"


def test_summarize_metric_rows_by_group_includes_required_summary_columns():
    rows = [
        {
            "algorithm": "foreground_mask_overlay",
            "baseline_algorithm": "alpha_blend_baseline",
            "metric": "entropy",
            "value": "5.0",
            "baseline_value": "4.0",
            "metric_tier": "Tier 2",
            "pairing_tier": "frame_strict",
            "evidence_label": "real_paired",
            "input_data_type": "paired NIR video + thermal display/heatmap-like video",
            "thermal_modality": "display_heatmap_like",
            "fusion_source": "paired_generated_fusion",
            "caveat": "paired generated caveat",
        },
        {
            "algorithm": "foreground_mask_overlay",
            "baseline_algorithm": "alpha_blend_baseline",
            "metric": "entropy",
            "value": "7.0",
            "baseline_value": "8.0",
            "metric_tier": "Tier 2",
            "pairing_tier": "frame_strict",
            "evidence_label": "real_paired",
            "input_data_type": "paired NIR video + thermal display/heatmap-like video",
            "thermal_modality": "display_heatmap_like",
            "fusion_source": "paired_generated_fusion",
            "caveat": "paired generated caveat",
        },
    ]

    summary = summarize_metric_rows_by_group(rows)

    assert len(summary) == 1
    row = summary[0]
    assert row["n"] == 2
    assert row["delta_current_minus_baseline"] == 0.0
    assert row["win_rate_current_vs_baseline"] == 0.5
    for key in ("metric_tier", "pairing_tier", "input_data_type", "caveat", "fusion_source"):
        assert key in row


def test_failure_tier_for_type_marks_generated_task_failures_as_tier3():
    assert failure_tier_for_type("fusion_target_faded", "paired_generated_fusion", "frame_strict", "real_paired") == "Tier 3"
    assert failure_tier_for_type("noise_increase", "paired_generated_fusion", "frame_strict", "real_paired") == "Tier 2"
    assert failure_tier_for_type("alignment_drift", "paired_captured_fusion", "frame_strict", "real_paired") == "Tier 1"
