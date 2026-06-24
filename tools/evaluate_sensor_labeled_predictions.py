#!/usr/bin/env python3
"""Evaluate raw-sensor predictions when manual labels are available."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smartbinocular.feature_schema import ENV_CLASSES
from tools.ml_metadata_utils import markdown_table, write_text


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path or not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()}) if rows else ["frame_id"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _confirmed_label(row: dict[str, str]) -> tuple[str, str] | tuple[None, None]:
    manual = (row.get("manual_label") or "").strip()
    if manual:
        return manual, "manual_label"
    suggested = (row.get("suggested_label") or "").strip()
    if suggested and _truthy(row.get("accept_suggested_label")):
        return suggested, "user_confirmed_suggested_label"
    return None, None


def _not_measured_payload(reason: str) -> dict[str, Any]:
    return {
        "metric_status": "not measured",
        "reason": reason,
        "scope": "manual-labeled sensor subset",
        "accuracy_claim": "none",
    }


def _write_not_measured(summary_path: Path, per_class_path: Path, out_json: Path, errors_path: Path, reason: str) -> dict[str, Any]:
    payload = _not_measured_payload(reason)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_text(
        summary_path,
        "\n".join(
            [
                "# Raw Sensor Labeled Evaluation",
                "",
                "Status: `not measured`. No valid completed manual-label CSV was selected, so no raw sensor accuracy is claimed.",
                "",
                markdown_table(["Field", "Value"], [["metric_status", "not measured"], ["reason", reason]]),
            ]
        )
        + "\n",
    )
    write_text(per_class_path, "# Raw Sensor Labeled Per-Class Metrics\n\nStatus: `not measured`.\n")
    _write_csv(errors_path, [])
    return payload


def _confidence_ece(y_true: list[str], y_pred: list[str], confidences: list[float], bins: int = 10) -> float:
    if not y_true:
        return math.nan
    conf = np.asarray(confidences, dtype=float)
    correct = np.asarray([a == b for a, b in zip(y_true, y_pred)], dtype=float)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf <= hi if hi == 1.0 else conf < hi)
        if not np.any(mask):
            continue
        ece += float(np.mean(mask) * abs(np.mean(correct[mask]) - np.mean(conf[mask])))
    return ece


def _confidence_brier(y_true: list[str], y_pred: list[str], confidences: list[float]) -> float:
    if not y_true:
        return math.nan
    correct = np.asarray([a == b for a, b in zip(y_true, y_pred)], dtype=float)
    conf = np.asarray(confidences, dtype=float)
    return float(np.mean((conf - correct) ** 2))


def _metric_block(rows: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

    if not rows:
        return {"row_count": 0}
    y_true = [str(row["manual_label"]) for row in rows]
    y_pred = [str(row["top1_label"]) for row in rows]
    confidences = [float(row.get("top1_proba") or 0.0) for row in rows]
    present = [label for label in labels if label in set(y_true) | set(y_pred)]
    raw_cm = confusion_matrix(y_true, y_pred, labels=present)
    norm_cm = confusion_matrix(y_true, y_pred, labels=present, normalize="true")
    return {
        "row_count": len(rows),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=present, average="weighted", zero_division=0)),
        "confidence_ece": _confidence_ece(y_true, y_pred, confidences),
        "top1_brier": _confidence_brier(y_true, y_pred, confidences),
        "labels": present,
        "confusion_matrix_raw": raw_cm.tolist(),
        "confusion_matrix_normalized": norm_cm.tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=present, output_dict=True, zero_division=0),
    }


def _joined_rows(label_path: Path, predictions_path: Path) -> list[dict[str, Any]]:
    labels = []
    for row in _read_csv(label_path):
        label, source = _confirmed_label(row)
        if not label:
            continue
        copied = dict(row)
        copied["_confirmed_label"] = label
        copied["_ground_truth_source"] = source
        labels.append(copied)
    preds = {row.get("frame_id"): row for row in _read_csv(predictions_path)}
    joined = []
    for label in labels:
        pred = preds.get(label.get("frame_id"))
        if not pred:
            continue
        row = dict(pred)
        row["manual_label"] = label.get("_confirmed_label", "").strip()
        row["ground_truth_source"] = label.get("_ground_truth_source", "")
        row["manual_label_confidence"] = label.get("label_confidence", "")
        row["manual_notes"] = label.get("notes", "")
        row["accepted_tau1"] = _truthy(row.get("accepted_tau1"))
        joined.append(row)
    return joined


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    all_metrics = payload["all_frames"]
    accepted = payload["accepted_only"]
    rows = [
        ["metric_status", payload["metric_status"]],
        ["label_path", payload["label_path"]],
        ["joined_rows", payload["joined_rows"]],
        ["accuracy_all", _fmt(all_metrics.get("accuracy"))],
        ["balanced_accuracy_all", _fmt(all_metrics.get("balanced_accuracy"))],
        ["macro_f1_all", _fmt(all_metrics.get("macro_f1"))],
        ["weighted_f1_all", _fmt(all_metrics.get("weighted_f1"))],
        ["confidence_ece_all", _fmt(all_metrics.get("confidence_ece"))],
        ["top1_brier_all", _fmt(all_metrics.get("top1_brier"))],
        ["abstention_rate_tau1_0.62", _fmt(payload.get("abstention_rate_tau1_0.62"))],
        ["accuracy_accepted_only", _fmt(accepted.get("accuracy"))],
        ["macro_f1_accepted_only", _fmt(accepted.get("macro_f1"))],
    ]
    write_text(
        path,
        "\n".join(
            [
                "# Raw Sensor Labeled Evaluation",
                "",
                "Status: manual-labeled sensor subset. This is not full deployment validation and is not mixed into training.",
                "",
                markdown_table(["Metric", "Value"], rows),
            ]
        )
        + "\n",
    )


def _write_per_class(path: Path, payload: dict[str, Any]) -> None:
    report = payload["all_frames"].get("classification_report", {})
    rows = []
    for label in ENV_CLASSES:
        item = report.get(label)
        if not item:
            continue
        rows.append([label, _fmt(item.get("precision")), _fmt(item.get("recall")), _fmt(item.get("f1-score")), int(item.get("support", 0))])
    write_text(
        path,
        "\n".join(["# Raw Sensor Labeled Per-Class Metrics", "", markdown_table(["Class", "Precision", "Recall", "F1", "Support"], rows)])
        + "\n",
    )


def _write_confusion_figure(payload: dict[str, Any], fig_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    labels = payload["all_frames"].get("labels", [])
    cm = np.asarray(payload["all_frames"].get("confusion_matrix_raw", []), dtype=float)
    if cm.size == 0:
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.75), max(4, len(labels) * 0.6)))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("manual label")
    ax.set_title("Raw sensor manual-labeled subset confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(fig_dir / "raw_sensor_confusion_matrix.png", dpi=150)
    plt.close(fig)


def _fmt(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return "n/a"
    return "n/a" if math.isnan(x) else f"{x:.4f}"


def evaluate_from_validation(
    *,
    validation_path: Path,
    predictions_path: Path,
    out_json: Path,
    summary_path: Path,
    per_class_path: Path,
    errors_path: Path,
    fig_dir: Path,
) -> dict[str, Any]:
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.is_file() else {}
    label_path_value = validation.get("selected_label_path")
    if validation.get("status") != "completed_labels_available" or not label_path_value:
        return _write_not_measured(summary_path, per_class_path, out_json, errors_path, "no valid completed manual-label CSV selected")
    label_path = Path(str(label_path_value))
    joined = _joined_rows(label_path, predictions_path)
    if not joined:
        return _write_not_measured(summary_path, per_class_path, out_json, errors_path, "manual labels did not join to prediction rows")
    accepted = [row for row in joined if row.get("accepted_tau1")]
    payload = {
        "metric_status": "manual-labeled sensor subset / preliminary",
        "scope": "manual-labeled sensor subset; not full deployment validation",
        "label_path": str(label_path),
        "predictions_path": str(predictions_path),
        "joined_rows": len(joined),
        "ground_truth_source_counts": {k: sum(1 for row in joined if row.get("ground_truth_source") == k) for k in sorted({row.get("ground_truth_source") for row in joined})},
        "abstention_rate_tau1_0.62": float(1.0 - len(accepted) / len(joined)),
        "all_frames": _metric_block(joined, list(ENV_CLASSES)),
        "accepted_only": _metric_block(accepted, list(ENV_CLASSES)),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    _write_summary(summary_path, payload)
    _write_per_class(per_class_path, payload)
    errors = [
        row
        for row in joined
        if row.get("manual_label") != row.get("top1_label") and float(row.get("top1_proba") or 0.0) >= 0.75
    ]
    _write_csv(errors_path, errors)
    _write_confusion_figure(payload, fig_dir)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate raw-sensor predictions against manual labels when available.")
    parser.add_argument("--label-validation", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--per-class", type=Path, required=True)
    parser.add_argument("--errors", type=Path, required=True)
    parser.add_argument("--fig-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    evaluate_from_validation(
        validation_path=args.label_validation,
        predictions_path=args.predictions,
        out_json=args.out_json,
        summary_path=args.summary,
        per_class_path=args.per_class,
        errors_path=args.errors,
        fig_dir=args.fig_dir,
    )
    print(f"Wrote {args.summary}")
    print(f"Wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
