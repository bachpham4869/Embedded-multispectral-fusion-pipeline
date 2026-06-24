#!/usr/bin/env python3
"""Validate raw-sensor manual-label CSVs and select the newest completed file."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smartbinocular.feature_schema import ENV_CLASSES
from tools.ml_metadata_utils import git_branch_name, git_commit_hash, markdown_table, write_text

REQUIRED_COLUMNS = {"frame_id", "video_id", "frame_path", "timestamp_sec", "manual_label", "label_confidence"}
OPTIONAL_COLUMNS = {
    "notes",
    "model_top1",
    "model_confidence",
    "suggested_label",
    "accept_suggested_label",
    "production_rf_top1",
    "production_rf_confidence",
}
ALLOWED_LABELS = set(ENV_CLASSES)


def find_label_csvs(search_roots: list[Path]) -> list[Path]:
    candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        if root.is_file() and root.suffix.lower() == ".csv":
            candidates.append(root)
            continue
        candidates.extend(path for path in root.rglob("*.csv") if "label" in path.name.lower())
    return sorted(set(candidates), key=lambda p: (p.stat().st_mtime if p.exists() else 0.0, str(p)), reverse=True)


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _parse_confidence(value: str) -> float | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def confirmed_label(row: dict[str, str]) -> tuple[str, str] | tuple[None, None]:
    manual = (row.get("manual_label") or "").strip()
    if manual:
        return manual, "manual_label"
    suggested = (row.get("suggested_label") or "").strip()
    if suggested and _truthy(row.get("accept_suggested_label")):
        return suggested, "user_confirmed_suggested_label"
    return None, None


def validate_label_file(path: Path) -> dict[str, Any]:
    try:
        fields, rows = _read_rows(path)
    except Exception as exc:
        return {"path": str(path), "status": "invalid", "error": str(exc), "row_count": 0}

    missing = sorted(REQUIRED_COLUMNS - set(fields))
    confirmed = [(*confirmed_label(row), row) for row in rows if confirmed_label(row)[0]]
    filled_labels = [str(label) for label, _source, _row in confirmed]
    invalid_labels = sorted({label for label in filled_labels if label not in ALLOWED_LABELS})
    confidences = [_parse_confidence(row.get("label_confidence", "")) for _label, _source, row in confirmed]
    low_confidence = sum(1 for value in confidences if value is not None and value < 0.5)
    distribution = Counter(label for label in filled_labels if label in ALLOWED_LABELS)
    accepted_suggested = sum(1 for _label, source, _row in confirmed if source == "user_confirmed_suggested_label")

    if missing:
        status = "invalid"
    elif invalid_labels:
        status = "invalid"
    elif confirmed:
        status = "completed_labels_available"
    else:
        status = "template_no_completed_labels"

    return {
        "path": str(path),
        "status": status,
        "mtime": path.stat().st_mtime if path.exists() else None,
        "row_count": len(rows),
        "filled_labels": len(confirmed),
        "manual_label_count": sum(1 for _label, source, _row in confirmed if source == "manual_label"),
        "accepted_suggested_labels": accepted_suggested,
        "missing_labels": len(rows) - len(confirmed),
        "missing_columns": missing,
        "invalid_label_count": sum(1 for label in filled_labels if label not in ALLOWED_LABELS),
        "invalid_labels": invalid_labels,
        "class_distribution": dict(sorted(distribution.items())),
        "low_confidence_count": low_confidence,
        "allowed_labels": sorted(ALLOWED_LABELS),
        "optional_columns_present": sorted(set(fields) & OPTIONAL_COLUMNS),
    }


def validate_label_candidates(candidates: list[Path]) -> dict[str, Any]:
    reports = [validate_label_file(path) for path in find_label_csvs(candidates) if path.is_dir()] if candidates and all(path.is_dir() for path in candidates) else [validate_label_file(path) for path in candidates]
    completed = [report for report in reports if report.get("status") == "completed_labels_available"]
    completed.sort(key=lambda r: (float(r.get("mtime") or 0.0), str(r.get("path"))), reverse=True)
    selected = completed[0] if completed else None
    return {
        "status": "completed_labels_available" if selected else "no_completed_labels",
        "selected_label_path": selected["path"] if selected else None,
        "selected_mtime": selected.get("mtime") if selected else None,
        "row_count": selected.get("row_count", 0) if selected else 0,
        "filled_labels": selected.get("filled_labels", 0) if selected else 0,
        "manual_label_count": selected.get("manual_label_count", 0) if selected else 0,
        "accepted_suggested_labels": selected.get("accepted_suggested_labels", 0) if selected else 0,
        "missing_labels": selected.get("missing_labels", 0) if selected else 0,
        "class_distribution": selected.get("class_distribution", {}) if selected else {},
        "low_confidence_count": selected.get("low_confidence_count", 0) if selected else 0,
        "candidate_count": len(reports),
        "candidates": reports,
        "git_commit": git_commit_hash(),
        "git_branch": git_branch_name(),
        "metric_status": "manual-labeled sensor subset" if selected else "not measured",
    }


def write_status_md(path: Path, result: dict[str, Any]) -> None:
    rows = [
        ["status", result["status"]],
        ["selected_label_path", result.get("selected_label_path") or "none"],
        ["candidate_count", result["candidate_count"]],
        ["row_count", result["row_count"]],
        ["filled_labels", result["filled_labels"]],
        ["manual_label_count", result.get("manual_label_count", 0)],
        ["accepted_suggested_labels", result.get("accepted_suggested_labels", 0)],
        ["missing_labels", result["missing_labels"]],
        ["low_confidence_count", result["low_confidence_count"]],
        ["metric_status", result["metric_status"]],
    ]
    body = [
        "# Manual Label Status",
        "",
        "Runtime validation selects the newest completed CSV with valid taxonomy labels. If no completed labels exist, raw sensor accuracy is not measured.",
        "",
        markdown_table(["Field", "Value"], rows),
        "",
        "## Class Distribution",
        "",
        markdown_table(["Class", "Count"], [[k, v] for k, v in result.get("class_distribution", {}).items()]),
        "",
        "## Candidate Files",
        "",
        markdown_table(
            ["Path", "Status", "Rows", "Filled", "Missing", "Invalid labels"],
            [
                [
                    c.get("path"),
                    c.get("status"),
                    c.get("row_count", 0),
                    c.get("filled_labels", 0),
                    c.get("missing_labels", 0),
                    ",".join(c.get("invalid_labels", [])),
                ]
                for c in result.get("candidates", [])
            ],
        ),
    ]
    write_text(path, "\n".join(body) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw sensor manual-label CSVs.")
    parser.add_argument("--search", nargs="+", type=Path, required=True, help="Directories or CSV files to scan.")
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    candidates = find_label_csvs(args.search)
    result = validate_label_candidates(candidates)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_status_md(args.out_md, result)
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
