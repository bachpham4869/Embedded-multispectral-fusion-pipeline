import csv
import json
from pathlib import Path


def test_agent_label_validation_requires_provenance_and_allowed_label() -> None:
    from tools.build_agent_manual_label_subset import validate_agent_label_row

    row = {
        "item_id": "raw_sensor:test:1",
        "source": "raw_sensor",
        "agent_manual_label": "normal_day",
        "label_confidence": "0.90",
        "label_source": "agent_manual_label",
        "visual_evidence_note": "daylight outdoor frame with no adverse weather",
        "caveat": "agent/manual-reviewed; not user-confirmed gold label",
    }

    assert validate_agent_label_row(row) == []

    bad = dict(row, agent_manual_label="snow", label_source="rf_heuristic_suggestion")
    errors = validate_agent_label_row(bad)
    assert any("agent_manual_label" in error for error in errors)
    assert any("label_source" in error for error in errors)


def test_evaluation_writes_not_measured_when_confident_labels_are_insufficient(tmp_path: Path) -> None:
    from tools.evaluate_agent_labeled_sensor_subset import evaluate_agent_labeled_subset

    labels = tmp_path / "agent_manual_labels.csv"
    predictions = tmp_path / "predictions.csv"
    out_json = tmp_path / "eval.json"
    summary = tmp_path / "eval.md"
    per_class = tmp_path / "per_class.md"

    with labels.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "item_id",
                "source",
                "frame_path",
                "pair_id",
                "agent_manual_label",
                "label_confidence",
                "label_source",
                "visual_evidence_note",
                "caveat",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "item_id": "raw_sensor:test:1",
                "source": "raw_sensor",
                "frame_path": "frame.jpg",
                "pair_id": "",
                "agent_manual_label": "normal_day",
                "label_confidence": "0.6",
                "label_source": "agent_manual_label",
                "visual_evidence_note": "low confidence",
                "caveat": "agent/manual-reviewed; not user-confirmed gold label",
            }
        )

    with predictions.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_id", "frame_path", "top1_label", "top1_proba", "accepted_tau1"])
        writer.writeheader()
        writer.writerow(
            {
                "frame_id": "raw_sensor:test:1",
                "frame_path": "frame.jpg",
                "top1_label": "normal_day",
                "top1_proba": "0.9",
                "accepted_tau1": "True",
            }
        )

    result = evaluate_agent_labeled_subset(
        label_csv=labels,
        prediction_csvs=[predictions],
        out_json=out_json,
        summary_md=summary,
        per_class_md=per_class,
        min_confidence=0.8,
        min_labels=2,
    )

    assert result["status"] == "not measured"
    assert "not measured" in summary.read_text()
    assert "accuracy" not in summary.read_text().lower()
    assert json.loads(out_json.read_text())["status"] == "not measured"


def test_evaluation_uses_agent_labels_only_as_preliminary_ground_truth(tmp_path: Path) -> None:
    from tools.evaluate_agent_labeled_sensor_subset import evaluate_agent_labeled_subset

    labels = tmp_path / "agent_manual_labels.csv"
    predictions = tmp_path / "predictions.csv"
    out_json = tmp_path / "eval.json"
    summary = tmp_path / "eval.md"
    per_class = tmp_path / "per_class.md"

    with labels.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "item_id",
                "source",
                "frame_path",
                "pair_id",
                "agent_manual_label",
                "label_confidence",
                "label_source",
                "visual_evidence_note",
                "caveat",
            ],
        )
        writer.writeheader()
        for idx, label in enumerate(["normal_day", "backlight", "normal_day"]):
            writer.writerow(
                {
                    "item_id": f"raw_sensor:test:{idx}",
                    "source": "raw_sensor",
                    "frame_path": f"frame_{idx}.jpg",
                    "pair_id": "",
                    "agent_manual_label": label,
                    "label_confidence": "0.9",
                    "label_source": "agent_manual_label",
                    "visual_evidence_note": "clear visual label",
                    "caveat": "agent/manual-reviewed; not user-confirmed gold label",
                }
            )

    with predictions.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_id", "frame_path", "top1_label", "top1_proba", "accepted_tau1"])
        writer.writeheader()
        for idx, pred in enumerate(["normal_day", "normal_night", "normal_day"]):
            writer.writerow(
                {
                    "frame_id": f"raw_sensor:test:{idx}",
                    "frame_path": f"frame_{idx}.jpg",
                    "top1_label": pred,
                    "top1_proba": "0.9",
                    "accepted_tau1": "True",
                }
            )

    result = evaluate_agent_labeled_subset(
        label_csv=labels,
        prediction_csvs=[predictions],
        out_json=out_json,
        summary_md=summary,
        per_class_md=per_class,
        min_confidence=0.8,
        min_labels=2,
    )

    assert result["status"] == "preliminary"
    assert result["label_source"] == "agent_manual_label"
    assert result["metrics"]["accuracy"] == 2 / 3
    assert "agent-labeled sensor subset evaluation" in summary.read_text()
    assert "not user-confirmed gold" in summary.read_text()
