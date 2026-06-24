from __future__ import annotations

from tools.harden_fusion_evidence import (
    REQUIRED_CAPTURE_METADATA_FIELDS,
    build_required_capture_metadata_rows,
)


def test_strict_paired_capture_metadata_schema_has_required_paths_and_timing():
    required = set(REQUIRED_CAPTURE_METADATA_FIELDS)
    for field in (
        "session_id",
        "frame_idx",
        "timestamp_iso",
        "nir_raw_path",
        "nir_enhanced_path",
        "thermal_raw_path",
        "thermal_heatmap_path",
        "thermal_mask_path",
        "fusion_output_path",
        "stage_timings_ms",
    ):
        assert field in required

    rows = build_required_capture_metadata_rows()
    by_field = {row["field"]: row for row in rows}
    assert by_field["timestamp_iso"]["strict_pair_requirement"] == "same session and <= 1s gap"
    assert by_field["fusion_output_path"]["required"] == "yes"
    assert by_field["homography_path"]["required"] == "yes"
