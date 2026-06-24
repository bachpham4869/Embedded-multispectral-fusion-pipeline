from __future__ import annotations

import json
from pathlib import Path

import cv2 as cv
import numpy as np

from tools.compare_classifiers import parse_args
from tools.ml_metadata_utils import (
    METADATA_STATUS_INFERRED,
    METADATA_STATUS_UNRESOLVED,
    METADATA_STATUS_VERIFIED,
    classify_metadata_status,
    dhash_image,
    hamming_distance_hex,
    sha256_file,
)
from tools.check_image_level_leakage import find_near_duplicate_pairs
from tools.rebuild_training_jsonl_with_metadata import enrich_records
from tools.split_group_aware_jsonl import group_aware_split
from tools.offline_pipeline import run_pipeline


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((32, 32, 3), value, dtype=np.uint8)
    assert cv.imwrite(str(path), img)


def _feature_row(label: str, group: str, **extra: object) -> dict[str, object]:
    row: dict[str, object] = {
        "nir_mean_brightness": 1.0,
        "nir_std": 2.0,
        "nir_entropy": 3.0,
        "nir_p95": 4.0,
        "nir_glare_score": 0.0,
        "nir_sharpness": 5.0,
        "nir_dark_fraction": 0.1,
        "nir_saturation_mean": 6.0,
        "hour_of_day_sin": 0.0,
        "hour_of_day_cos": 1.0,
        "prev_env_class": 0,
        "nir_blue_mean_ema": 7.0,
        "label": label,
        "source": "fixture",
        "nir_channel": "rgb",
        "thermal_channel": "none",
        "label_source": "dataset_original",
        "label_confidence": 0.8,
        "split_group_id": group,
    }
    row.update(extra)
    return row


def test_metadata_status_classification_respects_guardrails() -> None:
    assert classify_metadata_status(has_identity=True, has_file_sha256=False, weak_hint_only=False) == (
        METADATA_STATUS_VERIFIED,
        "path_or_frame_identity",
    )
    assert classify_metadata_status(has_identity=False, has_file_sha256=True, weak_hint_only=False) == (
        METADATA_STATUS_VERIFIED,
        "file_sha256",
    )
    assert classify_metadata_status(has_identity=False, has_file_sha256=False, weak_hint_only=True) == (
        METADATA_STATUS_INFERRED,
        "weak_feature_hint",
    )
    assert classify_metadata_status(has_identity=False, has_file_sha256=False, weak_hint_only=False) == (
        METADATA_STATUS_UNRESOLVED,
        "unresolved",
    )


def test_sha256_and_dhash_helpers_are_stable(tmp_path: Path) -> None:
    image_path = tmp_path / "a.jpg"
    _write_image(image_path, 80)

    digest_a = sha256_file(image_path)
    digest_b = sha256_file(image_path)
    assert digest_a == digest_b
    assert len(digest_a) == 64

    dhash_a = dhash_image(image_path)
    dhash_b = dhash_image(image_path)
    assert dhash_a == dhash_b
    assert len(dhash_a) == 16
    assert hamming_distance_hex(dhash_a, dhash_b) == 0


def test_find_near_duplicate_pairs_uses_dhash_threshold() -> None:
    train_rows = [
        {"label": "normal_day", "dhash": "0000000000000000", "file_sha256": "train-a"},
        {"label": "fog", "dhash": "ffffffffffffffff", "file_sha256": "train-b"},
    ]
    test_rows = [
        {"label": "normal_day", "dhash": "0000000000000001", "file_sha256": "test-a"},
        {"label": "fog", "dhash": "0fffffffffffffff", "file_sha256": "test-b"},
    ]

    pairs = find_near_duplicate_pairs(train_rows, test_rows, threshold=1)

    assert len(pairs) == 1
    assert pairs[0]["train_index"] == 0
    assert pairs[0]["test_index"] == 0
    assert pairs[0]["hamming_distance"] == 1
    assert pairs[0]["method"] == "dhash"


def test_enrich_records_preserves_unresolved_rows() -> None:
    rows = [_feature_row("normal_day", "g0", source="unknown_source", frame_idx=0)]

    enriched, summary = enrich_records(rows, source_index={}, source_root=Path("data/weather"))

    assert enriched[0]["metadata_status"] == METADATA_STATUS_UNRESOLVED
    assert enriched[0]["metadata_missing"]
    assert summary["metadata_status_counts"][METADATA_STATUS_UNRESOLVED] == 1


def test_group_aware_split_has_zero_group_overlap() -> None:
    rows = [
        _feature_row("normal_day", "day-a"),
        _feature_row("normal_day", "day-b"),
        _feature_row("fog", "fog-a"),
        _feature_row("fog", "fog-b"),
    ]

    train_rows, test_rows, summary = group_aware_split(rows, train_ratio=0.5, seed=7)

    train_groups = {row["split_group_id"] for row in train_rows}
    test_groups = {row["split_group_id"] for row in test_rows}
    assert train_groups.isdisjoint(test_groups)
    assert summary["group_overlap_count"] == 0
    assert summary["split_name"] == "group-aware split"


def test_compare_classifiers_accepts_phase3_provenance_args() -> None:
    args = parse_args(
        [
            "--train",
            "train.jsonl",
            "--test",
            "test.jsonl",
            "--split-method-label",
            "Phase 3 group-aware split",
            "--metric-status",
            "preliminary / pending final thesis review",
        ]
    )

    assert args.split_method_label == "Phase 3 group-aware split"
    assert args.metric_status == "preliminary / pending final thesis review"


def test_offline_pipeline_metadata_flag_is_opt_in(tmp_path: Path) -> None:
    image_path = tmp_path / "mwd" / "dataset2" / "dataset2" / "shine1.jpg"
    _write_image(image_path, 120)
    mapping = {"mwd": {"shine": {"env": "normal_day", "confidence": 0.85}}}

    plain_out = tmp_path / "plain.jsonl"
    metadata_out = tmp_path / "metadata.jsonl"

    run_pipeline("mwd", tmp_path / "mwd", plain_out, mapping, quiet=True)
    run_pipeline("mwd", tmp_path / "mwd", metadata_out, mapping, quiet=True, emit_source_metadata=True)

    plain_row = json.loads(plain_out.read_text(encoding="utf-8").splitlines()[0])
    metadata_row = json.loads(metadata_out.read_text(encoding="utf-8").splitlines()[0])

    assert "file_sha256" not in plain_row
    assert plain_row["label"] == metadata_row["label"]
    assert metadata_row["metadata_status"] == METADATA_STATUS_VERIFIED
    assert metadata_row["file_sha256"] == sha256_file(image_path)
