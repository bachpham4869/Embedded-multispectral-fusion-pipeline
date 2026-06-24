from __future__ import annotations

from tools.evaluate_paired_fusion import per_bucket_eval_rows, should_measure_rain_temporal


def test_per_bucket_rows_require_processing_bucket_source():
    rows = per_bucket_eval_rows(
        metric_rows=[
            {
                "processing_bucket": "nir_mono_clahe",
                "processing_bucket_source": "algorithm_forced",
                "metric": "entropy",
                "value": "5.0",
                "baseline_value": "4.0",
                "pairing_tier": "frame_strict",
                "metric_tier": "Tier 2",
                "evidence_label": "real_paired",
                "input_data_type": "paired NIR video + thermal display/heatmap-like video",
                "thermal_modality": "display_heatmap_like",
                "fusion_source": "none",
                "caveat": "offline forced bucket; not runtime bucket performance",
            }
        ]
    )

    clahe = [row for row in rows if row["processing_bucket"] == "nir_mono_clahe"][0]
    assert clahe["processing_bucket_source"] == "algorithm_forced"
    assert "not runtime bucket performance" in clahe["caveat"]
    rain = [row for row in rows if row["processing_bucket"] == "rain_temporal_median"][0]
    assert rain["status"] == "not measured"


def test_rain_temporal_requires_sequence_and_rain_evidence():
    assert not should_measure_rain_temporal([], min_sequence_frames=3)
    assert not should_measure_rain_temporal(
        [
            {"pair_id": "p1", "frame_idx": "0", "env_label": "unknown"},
            {"pair_id": "p2", "frame_idx": "1", "env_label": "unknown"},
            {"pair_id": "p3", "frame_idx": "2", "env_label": "unknown"},
        ],
        min_sequence_frames=3,
    )
    assert should_measure_rain_temporal(
        [
            {"pair_id": "p1", "frame_idx": "0", "env_label": "rain"},
            {"pair_id": "p2", "frame_idx": "1", "env_label": "rain"},
            {"pair_id": "p3", "frame_idx": "2", "env_label": "rain"},
        ],
        min_sequence_frames=3,
    )
