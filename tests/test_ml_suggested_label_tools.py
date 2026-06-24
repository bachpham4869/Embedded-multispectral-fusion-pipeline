from __future__ import annotations

import csv
import json
from pathlib import Path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_audit_marks_rf_heuristic_as_suggestion_not_independent_teacher(monkeypatch) -> None:
    from tools.audit_labeling_support_options import audit_labeling_support_options

    monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)

    result = audit_labeling_support_options()

    assert result["independent_teacher_available"] is False
    assert result["selected_method"] == "rf_heuristic_suggested_label"
    assert result["weak_label_dataset_allowed"] is False
    rf_rows = [row for row in result["candidates"] if row["candidate_method"] == "rf_heuristic_suggested_label"]
    assert rf_rows
    assert rf_rows[0]["output_label_type"] == "suggested_label"
    assert "independent teacher" not in rf_rows[0]["selected_reason"].lower()


def test_suggested_label_row_never_uses_auto_weak_label() -> None:
    from tools.suggest_sensor_labels import build_suggestion_row

    row = build_suggestion_row(
        frame_row={
            "frame_id": "v:1",
            "video_id": "v",
            "frame_path": "frames/f.jpg",
            "timestamp_sec": "1.0",
            "modality_guess": "unknown optical",
        },
        prediction_row={
            "top1_label": "transition",
            "top1_proba": "0.74",
            "top2_label": "fog",
            "top2_proba": "0.2",
            "posterior_entropy": "1.2",
        },
        feature_row={
            "nir_mean_brightness": 95.0,
            "nir_std": 30.0,
            "nir_p95": 140.0,
            "nir_glare_score": 0.01,
            "nir_dark_fraction": 0.05,
            "nir_sharpness": 120.0,
            "nir_saturation_mean": 10.0,
        },
        high_confidence=0.8,
        low_confidence=0.45,
    )

    assert row["suggested_label"] == "transition"
    assert row["evidence_label"] == "suggested_label"
    assert row["label_source"] == "rf_heuristic_suggestion"
    assert "auto_weak_label" not in row.values()
    assert row["requires_human_review"] == "true"
    assert "transition requires manual review" in row["caveat"]


def test_labeling_package_writes_autofilled_suggestion_template(tmp_path: Path) -> None:
    from tools.build_sensor_labeling_package import write_autofilled_label_template

    out = tmp_path / "manual_label_template_autofilled.csv"
    write_autofilled_label_template(
        out,
        [
            {
                "frame_id": "v:1",
                "frame_path": "frames/f.jpg",
                "timestamp_sec": "1.0",
                "suggested_label": "fog",
                "suggested_label_confidence": "0.67",
                "production_rf_top1": "fog",
                "production_rf_confidence": "0.7",
                "heuristic_reason": "low contrast",
                "review_priority": "medium",
            }
        ],
    )

    with out.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["suggested_label"] == "fog"
    assert rows[0]["manual_label"] == ""
    assert rows[0]["accept_suggested_label"] == ""
    assert "auto_weak_label" not in rows[0]


def test_suggestion_consistency_summary_does_not_report_accuracy() -> None:
    from tools.evaluate_suggestion_consistency import summarize_suggestion_consistency, write_summary_markdown

    rows = [
        {
            "suggested_label": "fog",
            "production_rf_top1": "fog",
            "suggested_label_confidence": "0.9",
            "rf_heuristic_agreement": "true",
            "requires_human_review": "false",
            "review_priority": "low",
            "video_id": "v",
            "timestamp_sec": "0",
        },
        {
            "suggested_label": "unknown_or_out_of_scope",
            "production_rf_top1": "glare",
            "suggested_label_confidence": "0.3",
            "rf_heuristic_agreement": "false",
            "requires_human_review": "true",
            "review_priority": "high",
            "video_id": "v",
            "timestamp_sec": "2",
        },
    ]

    summary = summarize_suggestion_consistency(rows)
    md = write_summary_markdown(summary)

    assert "agreement_rate" in summary
    assert "accuracy" not in json.dumps(summary).lower()
    assert "accuracy" not in md.lower()
    assert "independent reliability" not in md.lower()


def test_sensor_eval_accepts_only_manual_or_user_confirmed_suggestion(tmp_path: Path) -> None:
    from tools.evaluate_sensor_labeled_predictions import evaluate_from_validation
    from tools.validate_sensor_manual_labels import validate_label_file

    labels = tmp_path / "manual_label_template_autofilled.csv"
    _write_csv(
        labels,
        [
            {
                "frame_id": "v:1",
                "video_id": "v",
                "frame_path": "frames/f.jpg",
                "timestamp_sec": "1.0",
                "suggested_label": "fog",
                "accept_suggested_label": "yes",
                "manual_label": "",
                "label_confidence": "0.8",
                "notes": "user accepted suggestion",
            }
        ],
    )
    predictions = tmp_path / "predictions.csv"
    _write_csv(
        predictions,
        [
            {
                "frame_id": "v:1",
                "top1_label": "fog",
                "top1_proba": "0.9",
                "accepted_tau1": "true",
                "posterior_entropy": "0.1",
            }
        ],
    )
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps({"status": "completed_labels_available", "selected_label_path": str(labels)}),
        encoding="utf-8",
    )

    report = validate_label_file(labels)
    assert report["filled_labels"] == 1
    assert report["accepted_suggested_labels"] == 1

    payload = evaluate_from_validation(
        validation_path=validation,
        predictions_path=predictions,
        out_json=tmp_path / "eval.json",
        summary_path=tmp_path / "summary.md",
        per_class_path=tmp_path / "per_class.md",
        errors_path=tmp_path / "errors.csv",
        fig_dir=tmp_path,
    )

    assert payload["joined_rows"] == 1
    assert payload["ground_truth_source_counts"]["user_confirmed_suggested_label"] == 1
