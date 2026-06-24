from __future__ import annotations

import csv
from pathlib import Path

from tools.build_paired_data_manifest import (
    classify_pairing_tier,
    detect_thermal_modality,
    manifest_rows_from_timestamps,
)


def _write_timestamps(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "pair_idx",
        "pair_t_ns",
        "pair_t_sec",
        "thermal_ts_ns",
        "thermal_frame_id",
        "imx_ts_ns",
        "imx_frame_id",
        "skew_ms_imx_minus_thermal",
        "thermal_unit_for_display",
        "thermal_scale",
        "thermal_none_reads",
        "thermal_bad_frames",
        "thermal_errors",
        "imx_errors",
        "imx_video",
        "thermal_video",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_classify_pairing_tier_splits_strict_levels():
    assert classify_pairing_tier(30.0, same_csv_row=True, has_required_modalities=True) == "frame_strict"
    assert classify_pairing_tier(80.0, same_csv_row=False, has_required_modalities=True) == "time_strict_100ms"
    assert classify_pairing_tier(500.0, same_csv_row=False, has_required_modalities=True) == "protocol_strict_1s"
    assert classify_pairing_tier(2500.0, same_csv_row=False, has_required_modalities=True) == "near_paired"
    assert classify_pairing_tier(6000.0, same_csv_row=False, has_required_modalities=True) == "weak_paired"
    assert classify_pairing_tier(None, same_csv_row=False, has_required_modalities=True) == "weak_paired"
    assert classify_pairing_tier(30.0, same_csv_row=True, has_required_modalities=False) == "unpaired"


def test_detect_thermal_modality_marks_display_heatmap_like_without_numeric_arrays(tmp_path: Path):
    (tmp_path / "thermal_paired.mp4").write_bytes(b"video")
    assert detect_thermal_modality(tmp_path) == "display_heatmap_like"
    (tmp_path / "thermal_raw.npy").write_bytes(b"npy")
    assert detect_thermal_modality(tmp_path) == "raw_numeric_thermal"


def test_manifest_rows_from_timestamps_preserves_guardrail_fields(tmp_path: Path):
    (tmp_path / "imx_paired.mp4").write_bytes(b"video")
    (tmp_path / "thermal_paired.mp4").write_bytes(b"video")
    timestamps = tmp_path / "timestamps.csv"
    _write_timestamps(
        timestamps,
        [
            {
                "pair_idx": "0",
                "pair_t_ns": "1000000000",
                "pair_t_sec": "1.0",
                "thermal_ts_ns": "1000000000",
                "thermal_frame_id": "14",
                "imx_ts_ns": "1000015000",
                "imx_frame_id": "1",
                "skew_ms_imx_minus_thermal": "0.015",
                "thermal_unit_for_display": "celsius_temperature",
                "thermal_scale": "scale=31.0..35.9C(C)",
                "thermal_none_reads": "0",
                "thermal_bad_frames": "0",
                "thermal_errors": "0",
                "imx_errors": "0",
                "imx_video": "imx_paired.mp4",
                "thermal_video": "thermal_paired.mp4",
            }
        ],
    )

    rows = manifest_rows_from_timestamps(tmp_path, timestamps)

    assert len(rows) == 1
    row = rows[0]
    assert row["pair_id"] == "paired_000000"
    assert row["pairing_tier"] == "frame_strict"
    assert row["time_strict_100ms"] == "true"
    assert row["protocol_strict_1s"] == "true"
    assert row["thermal_modality"] == "display_heatmap_like"
    assert row["input_data_type"] == "paired NIR video + thermal display/heatmap-like video"
    assert row["fusion_source"] == "none"
    assert "not raw radiometric thermal" in row["caveat"]
