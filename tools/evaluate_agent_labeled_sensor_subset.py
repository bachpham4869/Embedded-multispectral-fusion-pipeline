#!/usr/bin/env python3
"""Evaluate production predictions against agent-reviewed labels.

This produces preliminary evidence only. Agent labels are not user-confirmed
gold labels and must not be used as final sensor-real accuracy claims.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

VALID_LABELS = {
    "night_clear",
    "normal_night",
    "normal_day",
    "fog",
    "rain",
    "glare",
    "backlight",
    "nir_night",
    "transition",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _prediction_index(paths: Iterable[Path]) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for path in paths:
        for row in _read_csv(path):
            keys = [
                row.get("frame_id", ""),
                row.get("frame_path", ""),
                row.get("pair_id", ""),
                f"paired_data:{row.get('pair_id', '')}" if row.get("pair_id") else "",
                f"raw_sensor:{row.get('frame_id', '')}" if row.get("frame_id") else "",
            ]
            for key in keys:
                if key:
                    index[key] = row
    return index


def _eligible_labels(rows: list[dict[str, str]], min_confidence: float) -> list[dict[str, str]]:
    eligible: list[dict[str, str]] = []
    for row in rows:
        label = (row.get("agent_manual_label") or "").strip()
        if label not in VALID_LABELS:
            continue
        if (row.get("label_source") or "").strip() != "agent_manual_label":
            continue
        try:
            confidence = float(row.get("label_confidence") or "")
        except ValueError:
            continue
        if confidence < min_confidence:
            continue
        eligible.append(row)
    return eligible


def _join_rows(labels: list[dict[str, str]], predictions: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    joined: list[dict[str, str]] = []
    for row in labels:
        keys = [row.get("item_id", ""), row.get("frame_path", ""), row.get("pair_id", "")]
        pred = next((predictions[key] for key in keys if key and key in predictions), None)
        if not pred:
            continue
        joined.append({**row, **{f"pred_{k}": v for k, v in pred.items()}})
    return joined


def _write_not_measured(out_json: Path, summary_md: Path, per_class_md: Path, reason: str) -> dict[str, object]:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    per_class_md.parent.mkdir(parents=True, exist_ok=True)
    payload = {"status": "not measured", "reason": reason, "label_source": "agent_manual_label"}
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    text = "\n".join(
        [
            "# Agent-Labeled Sensor Subset Evaluation",
            "",
            "Status: not measured.",
            "",
            f"Reason: {reason}",
            "",
            "No performance metric is reported because the confident agent-label subset is insufficient.",
        ]
    )
    summary_md.write_text(text + "\n")
    per_class_md.write_text("# Agent-Labeled Sensor Per-Class Report\n\nStatus: not measured.\n")
    return payload


def _plot_confusion(labels: list[str], preds: list[str], class_labels: list[str], fig_path: Path | None) -> None:
    if not fig_path:
        return
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(labels, preds, labels=class_labels)
    fig, ax = plt.subplots(figsize=(max(5, len(class_labels) * 0.8), max(4, len(class_labels) * 0.7)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_labels)), labels=class_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(class_labels)), labels=class_labels)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Agent label")
    ax.set_title("Agent-labeled sensor subset confusion matrix")
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def evaluate_agent_labeled_subset(
    *,
    label_csv: Path,
    prediction_csvs: list[Path],
    out_json: Path,
    summary_md: Path,
    per_class_md: Path,
    fig_path: Path | None = None,
    min_confidence: float = 0.8,
    min_labels: int = 10,
) -> dict[str, object]:
    label_rows = _read_csv(label_csv)
    eligible = _eligible_labels(label_rows, min_confidence)
    if len(eligible) < min_labels:
        return _write_not_measured(
            out_json,
            summary_md,
            per_class_md,
            f"only {len(eligible)} labels met label_confidence >= {min_confidence}; minimum required is {min_labels}",
        )

    joined = _join_rows(eligible, _prediction_index(prediction_csvs))
    if len(joined) < min_labels:
        return _write_not_measured(
            out_json,
            summary_md,
            per_class_md,
            f"only {len(joined)} confident labels could be joined to prediction rows; minimum required is {min_labels}",
        )

    y_true = [row["agent_manual_label"] for row in joined]
    y_pred = [row.get("pred_top1_label", "") for row in joined]
    labels = sorted(set(y_true) | set(y_pred))
    accepted = [str(row.get("pred_accepted_tau1", "")).lower() == "true" for row in joined]
    accepted_true = [true for true, ok in zip(y_true, accepted) if ok]
    accepted_pred = [pred for pred, ok in zip(y_pred, accepted) if ok]

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "accepted_only_accuracy": float(accuracy_score(accepted_true, accepted_pred)) if accepted_true else None,
        "accepted_rows": int(sum(accepted)),
        "row_count": len(joined),
    }
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    raw_cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    pred_counts = Counter(y_pred)
    true_counts = Counter(y_true)
    payload: dict[str, object] = {
        "status": "preliminary",
        "scope": "agent-labeled sensor subset evaluation",
        "label_source": "agent_manual_label",
        "caveat": "Labels are agent/manual-reviewed, not user-confirmed gold labels.",
        "metrics": metrics,
        "labels": labels,
        "agent_label_distribution": dict(sorted(true_counts.items())),
        "prediction_distribution": dict(sorted(pred_counts.items())),
        "classification_report": report,
        "confusion_matrix_raw": raw_cm,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _plot_confusion(y_true, y_pred, labels, fig_path)

    summary_lines = [
        "# agent-labeled sensor subset evaluation",
        "",
        "Status: preliminary. Labels are agent/manual-reviewed, not user-confirmed gold labels.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| rows | {len(joined)} |",
        f"| accepted_rows_tau1 | {metrics['accepted_rows']} |",
        f"| accuracy | {metrics['accuracy']:.4f} |",
        f"| balanced_accuracy | {metrics['balanced_accuracy']:.4f} |",
        f"| macro_f1 | {metrics['macro_f1']:.4f} |",
        f"| weighted_f1 | {metrics['weighted_f1']:.4f} |",
    ]
    if metrics["accepted_only_accuracy"] is not None:
        summary_lines.append(f"| accepted_only_accuracy | {metrics['accepted_only_accuracy']:.4f} |")
    summary_lines.extend(
        [
            "",
            "Use: preliminary sensor evidence only. User confirmation is still recommended before any final thesis claim.",
        ]
    )
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text("\n".join(summary_lines) + "\n")

    per_class_lines = ["# Agent-Labeled Sensor Per-Class Report", "", "| Class | Precision | Recall | F1 | Support |", "| --- | ---: | ---: | ---: | ---: |"]
    for label in labels:
        item = report.get(label, {})
        per_class_lines.append(
            f"| {label} | {item.get('precision', 0.0):.4f} | {item.get('recall', 0.0):.4f} | {item.get('f1-score', 0.0):.4f} | {int(item.get('support', 0))} |"
        )
    per_class_md.parent.mkdir(parents=True, exist_ok=True)
    per_class_md.write_text("\n".join(per_class_lines) + "\n")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate production predictions against agent-reviewed labels.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--per-class", type=Path, required=True)
    parser.add_argument("--fig", type=Path)
    parser.add_argument("--min-confidence", type=float, default=0.8)
    parser.add_argument("--min-labels", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = evaluate_agent_labeled_subset(
        label_csv=args.labels,
        prediction_csvs=args.predictions,
        out_json=args.out_json,
        summary_md=args.summary,
        per_class_md=args.per_class,
        fig_path=args.fig,
        min_confidence=args.min_confidence,
        min_labels=args.min_labels,
    )
    print(f"Agent-labeled evaluation status: {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
