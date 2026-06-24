from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2 as cv
import numpy as np


def _image(path: Path, value: int, *, bright_square: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((48, 64, 3), value, dtype=np.uint8)
    if bright_square:
        img[4:20, 5:25] = 255
    assert cv.imwrite(str(path), img)


def _feature_row(label: str = "normal_day", **extra: object) -> dict[str, object]:
    row: dict[str, object] = {
        "nir_mean_brightness": 80.0,
        "nir_std": 12.0,
        "nir_entropy": 3.0,
        "nir_p95": 100.0,
        "nir_glare_score": 0.0,
        "nir_sharpness": 5.0,
        "nir_dark_fraction": 0.1,
        "nir_saturation_mean": 4.0,
        "hour_of_day_sin": 0.0,
        "hour_of_day_cos": 1.0,
        "prev_env_class": 0,
        "nir_blue_mean_ema": 80.0,
        "label": label,
        "label_source": "dataset_original",
        "label_confidence": 0.9,
        "nir_channel": "rgb",
        "thermal_channel": "none",
    }
    row.update(extra)
    return row


def test_raw_sensor_inventory_parses_ffprobe_payload() -> None:
    from tools.inventory_raw_sensor_captures import parse_ffprobe_payload

    payload = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1280,
                "height": 720,
                "r_frame_rate": "30/1",
                "avg_frame_rate": "30000/1001",
                "nb_frames": "300",
            }
        ],
        "format": {"duration": "10.0", "size": "12345", "bit_rate": "999"},
    }

    meta = parse_ffprobe_payload(payload)

    assert meta["codec"] == "h264"
    assert meta["resolution"] == "1280x720"
    assert meta["fps"] == 30000 / 1001
    assert meta["approx_frame_count"] == 300
    assert meta["modality_guess"] == "unknown optical"


def test_frame_sampling_manifest_and_modality_contact_sheet(tmp_path: Path) -> None:
    from tools.extract_sensor_video_frames import build_frame_manifest_row, sample_frame_indices, write_modality_review_sheet

    indices = sample_frame_indices(total_frames=100, fps=25.0, sample_fps=2.0, max_frames=6)
    assert indices == [0, 12, 24, 36, 48, 60]

    frame_a = tmp_path / "frames" / "a.jpg"
    frame_b = tmp_path / "frames" / "b.jpg"
    _image(frame_a, 40)
    _image(frame_b, 140)
    row = build_frame_manifest_row(
        video_path=tmp_path / "video.mp4",
        frame_path=frame_a,
        frame_idx=12,
        fps=24.0,
        modality_guess="unknown optical",
        command="extract command",
    )
    assert row["timestamp_sec"] == 0.5
    assert row["modality_guess"] == "unknown optical"
    assert len(row["file_sha256"]) == 64
    assert len(row["dhash"]) == 16

    sheet = tmp_path / "review.png"
    write_modality_review_sheet([frame_a, frame_b], sheet, title="unknown optical")
    assert sheet.is_file()
    assert cv.imread(str(sheet)) is not None


def test_sensor_feature_record_schema_marks_unlabeled_rgb_proxy_context(tmp_path: Path) -> None:
    from tools.extract_sensor_frame_features import extract_feature_record_for_frame

    frame = tmp_path / "f.jpg"
    _image(frame, 80)

    record = extract_feature_record_for_frame(
        frame_path=frame,
        video_id="vid",
        frame_idx=7,
        timestamp_sec=1.25,
        modality_guess="unknown optical",
    )

    assert record["source_dataset"] == "raw_sensor_capture"
    assert record["label"] is None
    assert record["label_source"] == "unlabeled_sensor_raw"
    assert record["nir_channel"] == "unknown"
    assert record["metadata_status"] == "verified"
    assert "nir_mean_brightness" in record


def test_domain_shift_metrics_include_psi_and_out_of_range() -> None:
    from tools.domain_shift_report import compute_feature_drift

    train = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    sensor = np.array([5, 6, 7, 8], dtype=float)

    row = compute_feature_drift("x", train, sensor)

    assert row["feature"] == "x"
    assert row["ks_statistic"] > 0
    assert row["wasserstein_distance"] > 0
    assert row["out_of_range_rate"] > 0
    assert row["psi"] >= 0


def test_prediction_summary_is_rgb_scaler_proxy_and_has_abstention() -> None:
    from tools.predict_sensor_frames import summarize_prediction_rows

    rows = [
        {"top1_label": "normal_day", "top1_proba": 0.9, "accepted_tau1": True, "video_id": "v"},
        {"top1_label": "fog", "top1_proba": 0.4, "accepted_tau1": False, "video_id": "v"},
    ]

    summary = summarize_prediction_rows(rows, tau1=0.62, inference_scope="RGB-scaler proxy inference")

    assert summary["inference_scope"] == "RGB-scaler proxy inference"
    assert summary["row_count"] == 2
    assert summary["abstention_rate"] == 0.5
    assert summary["top1_distribution"]["normal_day"] == 1


def test_manual_label_template_columns_are_fixed(tmp_path: Path) -> None:
    from tools.build_sensor_labeling_package import write_label_template

    out = tmp_path / "labels.csv"
    write_label_template(
        out,
        [
            {
                "frame_id": "f0",
                "video_id": "v",
                "frame_path": "frames/f0.jpg",
                "timestamp_sec": 0.0,
                "model_top1": "normal_day",
                "model_confidence": 0.9,
            }
        ],
    )

    with out.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == [
        "frame_id",
        "video_id",
        "frame_path",
        "timestamp_sec",
        "model_top1",
        "model_confidence",
        "suggested_label",
        "manual_label",
        "label_confidence",
        "notes",
    ]


def test_optical_v2_and_21_features_are_named_and_nonzero(tmp_path: Path) -> None:
    from tools.optical_candidate_features import OPTICAL_21_FEATURES, compute_candidate_features

    frame = tmp_path / "bright.jpg"
    _image(frame, 20, bright_square=True)
    img = cv.imread(str(frame))
    features = compute_candidate_features(img, temporal_brightness_std=3.5)

    assert len(OPTICAL_21_FEATURES) == 21
    assert "local_contrast_mean" in features
    assert features["highlight_connected_component_area"] > 0
    assert features["saturated_component_count"] >= 1
    assert features["temporal_brightness_std"] == 3.5


def test_feature_set_comparison_rejects_subset_full_baseline_comparison() -> None:
    from tools.compare_feature_sets import describe_subset_scope

    rows = [_feature_row("normal_day"), _feature_row("fog")]
    scope = describe_subset_scope(rows, full_class_count=9)

    assert scope["class_count"] == 2
    assert scope["direct_full_9class_comparison_allowed"] is False
    assert "subset" in scope["limitation"]


def test_calibrated_mlp_uses_internal_validation_split_only() -> None:
    from tools.compare_mlp_variants import internal_calibration_split

    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    fit_idx, cal_idx = internal_calibration_split(y, seed=42, validation_fraction=1 / 3)

    assert set(fit_idx).isdisjoint(set(cal_idx))
    assert len(fit_idx) + len(cal_idx) == len(y)
    assert len(cal_idx) == 3


def test_manual_label_validation_selects_newest_completed_csv(tmp_path: Path) -> None:
    from tools.validate_sensor_manual_labels import find_label_csvs, validate_label_candidates

    old = tmp_path / "manual_labels.csv"
    new = tmp_path / "manual_label_completed.csv"
    template = tmp_path / "manual_label_template.csv"
    fields = ["frame_id", "video_id", "frame_path", "timestamp_sec", "manual_label", "label_confidence", "notes"]
    for path, label in [(old, "fog"), (new, "normal_day")]:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerow(
                {
                    "frame_id": path.stem,
                    "video_id": "v",
                    "frame_path": "f.jpg",
                    "timestamp_sec": "1.0",
                    "manual_label": label,
                    "label_confidence": "0.9",
                    "notes": "",
                }
            )
    with template.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"frame_id": "template", "video_id": "v", "frame_path": "f.jpg", "timestamp_sec": "1.0"})

    candidates = find_label_csvs([tmp_path])
    old.touch()
    new.touch()
    result = validate_label_candidates(candidates)

    assert result["selected_label_path"] == str(new)
    assert result["status"] == "completed_labels_available"
    assert result["filled_labels"] == 1
    assert result["class_distribution"] == {"normal_day": 1}


def test_empty_manual_labels_write_not_measured_eval(tmp_path: Path) -> None:
    from tools.evaluate_sensor_labeled_predictions import evaluate_from_validation

    validation = tmp_path / "validation.json"
    predictions = tmp_path / "pred.csv"
    summary = tmp_path / "summary.md"
    per_class = tmp_path / "per_class.md"
    out_json = tmp_path / "eval.json"
    errors = tmp_path / "errors.csv"
    validation.write_text(json.dumps({"status": "no_completed_labels", "selected_label_path": None}), encoding="utf-8")
    with predictions.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_id", "top1_label", "top1_proba", "accepted_tau1"])
        writer.writeheader()

    payload = evaluate_from_validation(
        validation_path=validation,
        predictions_path=predictions,
        out_json=out_json,
        summary_path=summary,
        per_class_path=per_class,
        errors_path=errors,
        fig_dir=tmp_path,
    )

    assert payload["metric_status"] == "not measured"
    assert "not measured" in summary.read_text(encoding="utf-8")
    assert json.loads(out_json.read_text(encoding="utf-8"))["metric_status"] == "not measured"


def test_manual_label_validation_rejects_invalid_taxonomy(tmp_path: Path) -> None:
    from tools.validate_sensor_manual_labels import validate_label_file

    path = tmp_path / "manual_label_completed.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame_id", "video_id", "frame_path", "timestamp_sec", "manual_label", "label_confidence", "notes"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "frame_id": "f",
                "video_id": "v",
                "frame_path": "f.jpg",
                "timestamp_sec": "0.0",
                "manual_label": "snow",
                "label_confidence": "0.7",
                "notes": "",
            }
        )

    result = validate_label_file(path)

    assert result["status"] == "invalid"
    assert result["invalid_label_count"] == 1
    assert "snow" in result["invalid_labels"]


def test_sensor_prediction_timestamp_prefers_video_timestamp() -> None:
    from tools.predict_sensor_frames import predict_rows

    class DummyScaler:
        def transform(self, X):
            return X

    class DummyModel:
        classes_ = np.array([3, 4])

        def predict_proba(self, X):
            return np.asarray([[0.2, 0.8]])

    feature_row = _feature_row()
    feature_row.update({"video_id": "v", "frame_idx": 10, "timestamp_sec": 12.5, "ts": 0.0, "frame_path": "f.jpg"})
    bundle = {
        "feature_set": list(feature_row.keys())[:12],
        "scalers": {"rgb": DummyScaler()},
        "rf": DummyModel(),
        "class_int_to_label": {"3": "normal_day", "4": "fog"},
    }

    rows = predict_rows([feature_row], bundle, scaler_group="rgb", tau1=0.62)

    assert rows[0]["timestamp_sec"] == 12.5
    assert rows[0]["inference_scope"] == "RGB-scaler proxy inference"


def test_optical_21_still_excludes_temporal_feature_and_reports_actual_count() -> None:
    from tools.optical_candidate_features import FEATURE_SET_DEFINITIONS

    still = FEATURE_SET_DEFINITIONS["optical_21_candidate_still"]
    temporal = FEATURE_SET_DEFINITIONS["optical_21_candidate_temporal"]

    assert "temporal_brightness_std" not in still["features"]
    assert still["actual_feature_count"] == len(still["features"])
    assert still["actual_feature_count"] != 21
    assert "temporal_brightness_std" in temporal["features"]


def test_feature_set_scope_includes_row_counts_and_class_coverage() -> None:
    from tools.compare_feature_sets import describe_subset_scope

    rows = [_feature_row("normal_day"), _feature_row("fog"), _feature_row("rain")]
    scope = describe_subset_scope(rows, full_class_count=9)

    assert scope["row_count"] == 3
    assert scope["class_count"] == 3
    assert scope["class_coverage"] == "3/9"
    assert scope["direct_full_9class_comparison_allowed"] is False


def test_rpi4_latency_manifest_includes_required_measurements() -> None:
    from tools.rpi4_model_latency_benchmark import build_manifest

    manifest = build_manifest(
        model_path=Path("models/production/env_classifier.joblib"),
        feature_source="fixture",
        repeats=3,
        hardware_label="test",
    )

    assert manifest["metric_status"] == "requires RPi4 run for deployment claim"
    for key in [
        "feature_extraction_latency_ms",
        "model_inference_latency_ms",
        "feature_plus_predict_latency_ms",
        "model_load_time_ms",
        "model_size_bytes",
        "ram_peak_bytes",
    ]:
        assert key in manifest["measurements"]
