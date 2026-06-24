#!/usr/bin/env python3
"""Build a small agent-reviewed manual-label subset for final ML evidence.

The default mode validates and enriches explicit review rows. The
``--use-contact-sheet-review`` mode is intentionally dataset-specific for the
current Phase 1 freeze: it records labels from the raw/paired contact sheets
that were visually inspected in this run. These labels are not user-confirmed
gold labels.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Iterable


ALLOWED_AGENT_LABELS = {
    "night_clear",
    "normal_night",
    "normal_day",
    "fog",
    "rain",
    "glare",
    "backlight",
    "nir_night",
    "transition",
    "unknown_or_uncertain",
}

AGENT_LABEL_SOURCE = "agent_manual_label"
AGENT_LABEL_CAVEAT = "agent/manual-reviewed; not user-confirmed gold label"

OUTPUT_COLUMNS = [
    "item_id",
    "source",
    "frame_path",
    "pair_id",
    "nir_path",
    "thermal_path",
    "timestamp",
    "model_top1",
    "model_confidence",
    "suggested_label",
    "agent_manual_label",
    "label_confidence",
    "uncertainty_reason",
    "visual_evidence_note",
    "label_source",
    "caveat",
]


RAW_CONTACT_SHEET_REVIEW = {
    "test_30fps_morning:16380": ("glare", 0.78, "direct bright indoor lighting visible; borderline normal_day/glare"),
    "test_30fps_morning:12000": ("backlight", 0.78, "dark foreground under trees against bright daylight background"),
    "test_30fps_morning:1140": ("backlight", 0.86, "dark stairwell foreground with bright doorway/window"),
    "test_30fps_morning:22500": ("normal_day", 0.82, "daylight outdoor walking scene; mild backlight but usable detail"),
    "test_30fps_morning:3060": ("backlight", 0.86, "indoor stairwell with bright doorway causing foreground underexposure"),
    "test_30fps_morning:4020": ("normal_day", 0.80, "indoor daylight scene with visible stairwell detail"),
    "test_30fps_morning:4620": ("normal_day", 0.86, "daylight outdoor tree/building scene; no fog/rain/glare evidence"),
    "test_30fps_morning:7440": ("normal_day", 0.88, "daylight outdoor field scene; no adverse-weather evidence"),
    "test_30fps_morning:7620": ("normal_day", 0.88, "daylight outdoor field scene; no night evidence"),
    "test_30fps_morning:12840": ("normal_day", 0.82, "bright indoor convenience-store lighting; no clear glare failure"),
    "test_30fps_morning:12900": ("normal_day", 0.82, "bright indoor aisle scene; no rain evidence"),
    "test_30fps_morning:13380": ("normal_day", 0.80, "indoor shelf scene with bright side window; still readable"),
}

PAIRED_CONTACT_SHEET_COUNT = 12


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in OUTPUT_COLUMNS})


def validate_agent_label_row(row: dict[str, str]) -> list[str]:
    errors: list[str] = []
    label = (row.get("agent_manual_label") or "").strip()
    source = (row.get("label_source") or "").strip()
    note = (row.get("visual_evidence_note") or "").strip()
    caveat = (row.get("caveat") or "").strip()
    try:
        confidence = float(row.get("label_confidence") or "")
    except ValueError:
        confidence = -1.0

    if label not in ALLOWED_AGENT_LABELS:
        errors.append(f"agent_manual_label must be one of {sorted(ALLOWED_AGENT_LABELS)}")
    if source != AGENT_LABEL_SOURCE:
        errors.append("label_source must be agent_manual_label")
    if not 0.0 <= confidence <= 1.0:
        errors.append("label_confidence must be numeric in [0, 1]")
    if not note:
        errors.append("visual_evidence_note is required")
    if "not user-confirmed gold" not in caveat:
        errors.append("caveat must state not user-confirmed gold label")
    return errors


def _raw_review_rows(raw_template: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in _read_csv(raw_template):
        frame_id = row.get("frame_id", "")
        review = RAW_CONTACT_SHEET_REVIEW.get(frame_id)
        if not review:
            continue
        label, confidence, note = review
        rows.append(
            {
                "item_id": f"raw_sensor:{frame_id}",
                "source": "raw_sensor",
                "frame_path": row.get("frame_path", ""),
                "pair_id": "",
                "nir_path": "",
                "thermal_path": "",
                "timestamp": row.get("timestamp_sec", ""),
                "model_top1": row.get("production_rf_top1", ""),
                "model_confidence": row.get("production_rf_confidence", ""),
                "suggested_label": row.get("suggested_label", ""),
                "agent_manual_label": label,
                "label_confidence": f"{confidence:.2f}",
                "uncertainty_reason": "contact-sheet review; raw optical modality remains unknown",
                "visual_evidence_note": note,
                "label_source": AGENT_LABEL_SOURCE,
                "caveat": AGENT_LABEL_CAVEAT,
            }
        )
    return rows


def _paired_review_rows(paired_template: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for idx, row in enumerate(_read_csv(paired_template)):
        if idx >= PAIRED_CONTACT_SHEET_COUNT:
            break
        rows.append(
            {
                "item_id": f"paired_data:{row.get('pair_id', '')}",
                "source": "paired_data",
                "frame_path": row.get("nir_raw_path", ""),
                "pair_id": row.get("pair_id", ""),
                "nir_path": row.get("nir_raw_path", ""),
                "thermal_path": row.get("thermal_raw_path", ""),
                "timestamp": row.get("timestamp_sec", ""),
                "model_top1": row.get("model_top1", ""),
                "model_confidence": row.get("model_confidence", ""),
                "suggested_label": row.get("suggested_label", ""),
                "agent_manual_label": "backlight",
                "label_confidence": "0.84",
                "uncertainty_reason": "paired contact-sheet review; modality not user-confirmed",
                "visual_evidence_note": "underexposed optical foreground with bright background; thermal side shown for review context",
                "label_source": AGENT_LABEL_SOURCE,
                "caveat": AGENT_LABEL_CAVEAT,
            }
        )
    return rows


def build_contact_sheet_review_rows(raw_template: Path, paired_template: Path) -> list[dict[str, str]]:
    rows = _raw_review_rows(raw_template) + _paired_review_rows(paired_template)
    valid_rows: list[dict[str, str]] = []
    for row in rows:
        errors = validate_agent_label_row(row)
        if errors:
            row["agent_manual_label"] = "unknown_or_uncertain"
            row["label_confidence"] = "0.00"
            row["uncertainty_reason"] = "; ".join(errors)
        valid_rows.append(row)
    return valid_rows


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    label_counts = Counter(row.get("agent_manual_label", "") for row in rows)
    source_counts = Counter(row.get("source", "") for row in rows)
    confident_rows = [row for row in rows if float(row.get("label_confidence") or 0.0) >= 0.8]
    lines = [
        "# Agent Manual Label Summary",
        "",
        "Status: preliminary agent/manual-reviewed subset. These labels are not user-confirmed gold labels.",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| total_rows | {len(rows)} |",
        f"| confident_rows_label_confidence_ge_0_8 | {len(confident_rows)} |",
        f"| label_source | `{AGENT_LABEL_SOURCE}` |",
        f"| caveat | {AGENT_LABEL_CAVEAT} |",
        "",
        "## By Source",
        "",
        "| Source | Count |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {source} | {count} |" for source, count in sorted(source_counts.items()))
    lines.extend(["", "## By Agent Label", "", "| Label | Count |", "| --- | ---: |"])
    lines.extend(f"| {label} | {count} |" for label, count in sorted(label_counts.items()))
    path.write_text("\n".join(lines) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an agent-reviewed manual-label subset for final ML evidence.")
    parser.add_argument("--raw-template", type=Path, required=True)
    parser.add_argument("--paired-template", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--use-contact-sheet-review", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.use_contact_sheet_review:
        raise SystemExit("--use-contact-sheet-review is required for this Phase 1 freeze run")
    rows = build_contact_sheet_review_rows(args.raw_template, args.paired_template)
    _write_csv(args.out_csv, rows)
    write_summary(args.summary, rows)
    print(f"Wrote {len(rows)} agent-reviewed rows to {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
