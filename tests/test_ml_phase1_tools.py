from __future__ import annotations

import numpy as np

from tools.analyze_dataset_distribution import summarize_records
from tools.check_dataset_leakage import (
    FEATURE_KEYS,
    check_metadata_overlap,
    feature_vector_overlap_details,
)
from tools.compare_classifiers import canonical_model_name, run_classical_benchmark
from tools.evaluate_frozen_classifier import feature_overlap_test_indices


def _feature_row(label: str, value: float, **extra: object) -> dict[str, object]:
    row: dict[str, object] = {
        key: value + idx
        for idx, key in enumerate(FEATURE_KEYS)
    }
    row.update(
        {
            "label": label,
            "source": "fixture_source",
            "label_source": "dataset_original",
            "label_confidence": 0.8,
            "nir_channel": "rgb",
            "thermal_channel": "none",
        }
    )
    row.update(extra)
    return row


def test_summarize_records_reports_counts_and_support_risks() -> None:
    rows = [
        {
            "label": "normal_day",
            "source": "offline_weather_time",
            "label_source": "dataset_original",
            "label_confidence": 0.9,
            "nir_channel": "rgb",
            "thermal_channel": "none",
        },
        {
            "label": "normal_day",
            "source": "offline_weather_time",
            "label_source": "dataset_original",
            "label_confidence": 0.8,
            "nir_channel": "rgb",
            "thermal_channel": "none",
        },
        {
            "weak_label": "glare",
            "source": "field_manual",
            "label_source": "manual",
            "label_confidence": 0.6,
            "nir_channel": "nir",
            "thermal_channel": "none",
        },
    ]

    summary = summarize_records(rows, dataset_name="fixture", min_class_count=2)

    assert summary["dataset_name"] == "fixture"
    assert summary["row_count"] == 3
    assert summary["class_counts"] == {"normal_day": 2, "glare": 1}
    assert summary["label_source_counts"] == {"dataset_original": 2, "manual": 1}
    assert summary["nir_channel_counts"] == {"rgb": 2, "nir": 1}
    assert summary["thermal_channel_counts"] == {"none": 3}
    assert summary["imbalance_ratio"] == 2.0
    assert summary["low_support_classes"] == ["glare"]
    assert summary["label_confidence"]["min"] == 0.6
    assert summary["label_confidence"]["max"] == 0.9


def test_check_metadata_overlap_reports_path_source_and_session_frame_overlap() -> None:
    train_rows = [
        {
            "label": "normal_day",
            "source": "same_source",
            "image_path": "dataset/a.jpg",
            "session_id": "s1",
            "frame_idx": 10,
        },
        {
            "label": "fog",
            "source": "train_only",
            "image_path": "dataset/train_only.jpg",
            "session_id": "s2",
            "frame_idx": 1,
        },
    ]
    test_rows = [
        {
            "label": "normal_day",
            "source": "same_source",
            "image_path": "dataset/a.jpg",
            "session_id": "s1",
            "frame_idx": 10,
        },
        {
            "label": "glare",
            "source": "test_only",
            "image_path": "dataset/test_only.jpg",
            "session_id": "s3",
            "frame_idx": 1,
        },
    ]

    report = check_metadata_overlap(train_rows, test_rows)

    assert report["source_overlap"]["count"] == 1
    assert report["path_overlap"]["count"] == 1
    assert report["filename_overlap"]["count"] == 1
    assert report["session_frame_overlap"]["count"] == 1
    assert report["metadata_limitations"] == []


def test_check_metadata_overlap_records_missing_metadata_limitations() -> None:
    report = check_metadata_overlap(
        [{"label": "normal_day", "source": "train_source"}],
        [{"label": "normal_day", "source": "test_source"}],
    )

    assert report["source_overlap"]["count"] == 0
    assert report["path_overlap"]["count"] == 0
    assert report["filename_overlap"]["count"] == 0
    assert report["session_frame_overlap"]["count"] == 0
    assert "No path-like metadata fields found; image/path overlap cannot be checked." in report[
        "metadata_limitations"
    ]
    assert "No session metadata fields found; session/frame overlap cannot be checked." in report[
        "metadata_limitations"
    ]


def test_feature_vector_overlap_details_include_indices_metadata_and_values() -> None:
    train_rows = [
        _feature_row("normal_day", 1.0, source="train_only"),
        _feature_row("fog", 10.0, source="shared_source", frame_idx=3),
    ]
    test_rows = [
        _feature_row("rain", 99.0, source="test_only"),
        _feature_row("fog", 10.0, source="shared_source", frame_idx=9),
    ]

    details = feature_vector_overlap_details(train_rows, test_rows)

    assert len(details) == 1
    detail = details[0]
    assert detail["train_index"] == 1
    assert detail["test_index"] == 1
    assert detail["train_label"] == "fog"
    assert detail["test_label"] == "fog"
    assert detail["train_source"] == "shared_source"
    assert detail["test_source"] == "shared_source"
    assert detail["match_type"] == "exact_feature_vector"
    assert "original image path/hash metadata" in detail["missing_metadata"]
    assert detail["feature_values_json"].startswith("{")


def test_canonical_model_name_accepts_phase2_aliases() -> None:
    assert canonical_model_name("logistic_regression") == "logistic_regression"
    assert canonical_model_name("logreg") == "logistic_regression"
    assert canonical_model_name("rf_200") == "random_forest_200_current_config"
    assert canonical_model_name("mlp_64_32") == "mlp_small_64_32"


def test_run_classical_benchmark_writes_manifest_and_per_model_metrics(tmp_path) -> None:
    X_train = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 1.0],
            [1.1, 1.0],
        ],
        dtype=np.float32,
    )
    y_train = np.asarray([1, 1, 2, 2], dtype=np.int32)
    X_test = np.asarray([[0.05, 0.0], [1.05, 1.0]], dtype=np.float32)
    y_test = np.asarray([1, 2], dtype=np.int32)

    result = run_classical_benchmark(
        X_train,
        y_train,
        X_test,
        y_test,
        labels=np.asarray([1, 2], dtype=np.int32),
        models=["gaussian_nb"],
        seed=7,
        bootstrap=2,
        latency_repeats=1,
        artifact_dir=tmp_path / "artifacts",
        fig_dir=tmp_path / "figures",
        hardware_label="unit-test proxy",
        persist_models=True,
        run_status="quick / preliminary / not thesis-ready",
        provenance={
            "command": "unit-test",
            "git_commit": "test",
            "branch": "test-branch",
            "feature_set_version": "optical_12_baseline",
            "split_method": "unit-test split",
            "seed": 7,
            "train_rows": 4,
            "test_rows": 2,
            "train_manifest": [],
            "test_manifest": [],
            "manifest_hash": "none",
            "versions": {"python": "test", "sklearn": "test", "numpy": "test", "scipy": "test"},
            "hardware": "unit-test proxy",
            "latency_repeats": 1,
        },
    )

    assert result["rows"][0]["model"] == "gaussian_nb"
    assert result["rows"][0]["model_path"].endswith("gaussian_nb/model.joblib")
    assert (tmp_path / "artifacts" / "run_manifest.json").is_file()
    assert (tmp_path / "artifacts" / "gaussian_nb" / "metrics.json").is_file()
    assert (tmp_path / "artifacts" / "gaussian_nb" / "classification_report.txt").is_file()


def test_feature_overlap_test_indices_identifies_only_test_rows() -> None:
    train_rows = [
        _feature_row("fog", 10.0),
        _feature_row("rain", 20.0),
    ]
    test_rows = [
        _feature_row("normal_day", 30.0),
        _feature_row("fog", 10.0),
        _feature_row("rain", 20.0),
    ]

    assert feature_overlap_test_indices(train_rows, test_rows) == {1, 2}
