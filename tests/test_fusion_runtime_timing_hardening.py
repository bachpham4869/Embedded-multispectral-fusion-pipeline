from __future__ import annotations

from tools.harden_fusion_evidence import normalize_runtime_timing_rows


def test_runtime_timing_splits_measured_and_estimated_p95():
    rows = normalize_runtime_timing_rows(
        [
            {
                "session_file": "session_estimated.json",
                "fps_mean": "12.0",
                "fps_p95": "14.5",
                "fps_p95_source": "estimated_mean_plus_1.645std",
                "evidence_label": "real_paired",
                "stage_profiler_summary": "nir_bucket; fusion_composite",
            },
            {
                "session_file": "session_measured.json",
                "fps_mean": "20.0",
                "fps_p95": "21.0",
                "fps_p95_source": "measured",
                "evidence_label": "real_paired",
                "stage_profiler_summary": "",
            },
        ]
    )

    estimated = rows[0]
    assert estimated["measured_p95"] == ""
    assert estimated["estimated_p95"] == "14.5"
    assert "estimated from mean/std" in estimated["caveat"]
    assert estimated["metric_tier"] == "Tier 1"
    assert estimated["input_data_type"] == "real session JSON"

    measured = rows[1]
    assert measured["measured_p95"] == "21.0"
    assert measured["estimated_p95"] == ""
    assert "stage profiler fields missing" in measured["caveat"]
