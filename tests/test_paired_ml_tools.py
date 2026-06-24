from __future__ import annotations

import csv
from pathlib import Path

import cv2 as cv
import numpy as np


EXPECTED_PAIRING_TIERS = {
    "frame_strict",
    "time_strict_100ms",
    "protocol_strict_1s",
    "near_paired",
    "weak_paired",
    "unpaired",
}


def _image(value: int = 80) -> np.ndarray:
    img = np.full((48, 64, 3), value, dtype=np.uint8)
    img[6:18, 8:24] = 220
    return img


def _manifest_row(**extra: str) -> dict[str, str]:
    row = {
        "pair_id": "paired_000001",
        "session_id": "paired_data",
        "frame_idx": "1",
        "video_frame_index": "0",
        "timestamp_sec": "1.25",
        "pairing_tier": "frame_strict",
        "nir_raw_path": "data/paired_data/imx_paired.mp4#frame=0",
        "thermal_heatmap_path": "data/paired_data/thermal_paired.mp4#frame=0",
        "thermal_raw_path": "",
        "imx_video_path": "data/paired_data/imx_paired.mp4",
        "thermal_video_path": "data/paired_data/thermal_paired.mp4",
        "thermal_modality": "display_heatmap_like",
        "env_label": "unknown",
        "label_source": "none",
        "label_confidence": "0",
    }
    row.update(extra)
    return row


def test_pairing_tier_vocabulary_matches_agent2_manifest() -> None:
    from tools.extract_paired_nir_features import PAIRING_TIERS

    assert set(PAIRING_TIERS) == EXPECTED_PAIRING_TIERS


def test_paired_feature_record_includes_required_stratification_fields() -> None:
    from tools.extract_paired_nir_features import feature_record_from_pair_frame

    record = feature_record_from_pair_frame(_image(), _manifest_row())

    for key in (
        "pairing_tier",
        "session_id",
        "frame_idx",
        "nir_modality",
        "thermal_modality",
        "label_source",
        "nir_channel",
        "thermal_channel",
        "evidence_type",
    ):
        assert key in record
    assert record["source_dataset"] == "paired_sensor_capture"
    assert record["metadata_status"] == "verified"
    assert record["evidence_type"] == "unlabeled"
    assert record["label"] is None
    assert "nir_mean_brightness" in record


def test_sidecar_labels_require_taxonomy_trusted_source_and_confidence() -> None:
    from tools.extract_paired_nir_features import trusted_label_from_manifest_row

    assert trusted_label_from_manifest_row(_manifest_row(env_label="normal_day", label_source="none", label_confidence="0.95"))[0] is None
    assert trusted_label_from_manifest_row(_manifest_row(env_label="not_a_class", label_source="manual", label_confidence="0.95"))[0] is None
    assert trusted_label_from_manifest_row(_manifest_row(env_label="normal_day", label_source="manual", label_confidence="0.2"))[0] is None

    label, reason = trusted_label_from_manifest_row(_manifest_row(env_label="normal_day", label_source="manual", label_confidence="0.95"))

    assert label == "normal_day"
    assert reason == "trusted_label"


def test_prediction_rows_propagate_paired_metadata_and_proxy_wording() -> None:
    from tools.predict_sensor_frames import PROXY_INFERENCE_SCOPE, summarize_prediction_rows

    rows = [
        {
            "top1_label": "normal_day",
            "top1_proba": 0.7,
            "accepted_tau1": True,
            "session_id": "paired_data",
            "pairing_tier": "frame_strict",
            "nir_modality": "unknown_optical",
            "thermal_modality": "display_heatmap_like",
            "label_source": "none",
        }
    ]

    summary = summarize_prediction_rows(rows, tau1=0.62, inference_scope=PROXY_INFERENCE_SCOPE)

    assert summary["inference_scope"] == "RGB-scaler proxy inference, not validated NIR classifier accuracy"
    assert summary["by_pairing_tier"]["frame_strict"]["row_count"] == 1
    assert summary["by_session_id"]["paired_data"]["row_count"] == 1
    assert summary["by_label_source"]["none"]["row_count"] == 1


def test_paired_manual_label_template_and_not_measured_outputs(tmp_path: Path) -> None:
    from tools.build_paired_labeling_package import write_paired_label_template, write_paired_not_measured

    labels = tmp_path / "manual_label_template_paired.csv"
    write_paired_label_template(
        labels,
        [
            {
                "pair_id": "paired_000001",
                "session_id": "paired_data",
                "frame_idx": "1",
                "nir_raw_path": "nir.mp4#frame=0",
                "thermal_raw_path": "thermal.mp4#frame=0",
                "fusion_output_path": "",
                "model_top1": "normal_day",
                "model_confidence": "0.7",
                "suggested_label": "normal_day",
            }
        ],
    )
    with labels.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))

    assert "pair_id" in header
    assert "nir_raw_path" in header
    assert "thermal_raw_path" in header
    assert "manual_label" in header

    summary = tmp_path / "paired_sensor_labeled_eval.md"
    write_paired_not_measured(summary)

    text = summary.read_text(encoding="utf-8").lower()
    assert "not measured" in text
    assert "accuracy" not in text
