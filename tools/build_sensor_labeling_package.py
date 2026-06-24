#!/usr/bin/env python3
"""Build a small manual-labeling package from raw sensor predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.extract_sensor_video_frames import write_modality_review_sheet
from tools.ml_metadata_utils import markdown_table, write_text

LABEL_TEMPLATE_COLUMNS = [
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

SUGGESTION_TEMPLATE_COLUMNS = [
    "frame_id",
    "video_id",
    "frame_path",
    "timestamp_sec",
    "suggested_label",
    "suggested_label_confidence",
    "production_rf_top1",
    "production_rf_confidence",
    "rf_heuristic_agreement",
    "heuristic_reason",
    "review_priority",
    "priority_score",
    "priority_reasons",
    "manual_label",
    "label_confidence",
    "accept_suggested_label",
    "notes",
]


def write_label_template(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_TEMPLATE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "frame_id": row.get("frame_id", ""),
                    "video_id": row.get("video_id", ""),
                    "frame_path": row.get("frame_path", ""),
                    "timestamp_sec": row.get("timestamp_sec", ""),
                    "model_top1": row.get("model_top1", row.get("top1_label", "")),
                    "model_confidence": row.get("model_confidence", row.get("top1_proba", "")),
                    "suggested_label": "",
                    "manual_label": "",
                    "label_confidence": "",
                    "notes": "",
                }
            )


def write_autofilled_label_template(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUGGESTION_TEMPLATE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "frame_id": row.get("frame_id", ""),
                    "video_id": row.get("video_id", ""),
                    "frame_path": row.get("frame_path", ""),
                    "timestamp_sec": row.get("timestamp_sec", ""),
                    "suggested_label": row.get("suggested_label", ""),
                    "suggested_label_confidence": row.get("suggested_label_confidence", ""),
                    "production_rf_top1": row.get("production_rf_top1", row.get("model_top1", "")),
                    "production_rf_confidence": row.get("production_rf_confidence", row.get("model_confidence", "")),
                    "rf_heuristic_agreement": row.get("rf_heuristic_agreement", ""),
                    "heuristic_reason": row.get("heuristic_reason", ""),
                    "review_priority": row.get("review_priority", ""),
                    "priority_score": row.get("priority_score", ""),
                    "priority_reasons": row.get("priority_reasons", ""),
                    "manual_label": "",
                    "label_confidence": "",
                    "accept_suggested_label": "",
                    "notes": "",
                }
            )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select_label_candidates(pred_rows: list[dict[str, str]], max_frames: int) -> list[dict[str, Any]]:
    if not pred_rows or max_frames <= 0:
        return []
    rows = list(pred_rows)
    for row in rows:
        conf = float(row.get("top1_proba") or row.get("model_confidence") or 0.0)
        ent = float(row.get("posterior_entropy") or 0.0)
        priority = abs(conf - 0.62) * -1.0 + ent
        if row.get("top1_label") in {"glare", "backlight", "transition"}:
            priority += 2.0
        row["_priority"] = priority
    rows.sort(key=lambda r: float(r["_priority"]), reverse=True)
    selected = rows[:max_frames]
    return [
        {
            "frame_id": r.get("frame_id", ""),
            "video_id": r.get("video_id", ""),
            "frame_path": r.get("frame_path", ""),
            "timestamp_sec": r.get("timestamp_sec", ""),
            "model_top1": r.get("top1_label", ""),
            "model_confidence": r.get("top1_proba", ""),
        }
        for r in selected
    ]


def select_suggestion_candidates(suggestion_rows: list[dict[str, str]], max_frames: int) -> list[dict[str, Any]]:
    if not suggestion_rows or max_frames <= 0:
        return []
    rows = list(suggestion_rows)
    priority_rank = {"high": 3, "medium": 2, "low": 1}
    rows.sort(
        key=lambda r: (
            priority_rank.get(str(r.get("review_priority", "")).lower(), 0),
            float(r.get("priority_score") or 0.0),
            float(r.get("suggested_label_confidence") or 0.0) * -1.0,
        ),
        reverse=True,
    )
    return [dict(r) for r in rows[:max_frames]]


def write_protocol(path: Path, template_path: Path, out_dir: Path, count: int) -> None:
    body = [
        "# Raw Sensor Labeling Protocol",
        "",
        "Status: manual review package. No manual labels are inferred by the tool.",
        "",
        "Fill the `manual_label` column in the CSV. If you agree with a pre-filled `suggested_label`, set `accept_suggested_label=yes`; otherwise leave it blank or write the corrected `manual_label`. Suggestions are not ground truth until a human confirms them.",
        "",
        "Allowed labels: `night_clear`, `normal_night`, `normal_day`, `fog`, `rain`, `glare`, `backlight`, `nir_night`. Use `transition` only for true dawn/dusk transient scenes.",
        "",
        markdown_table(
            ["Item", "Value"],
            [
                ["candidate_frame_count", count],
                ["template", str(template_path)],
                ["review_dir", str(out_dir)],
                ["metric_status", "manual labels pending; sensor accuracy not measured"],
            ],
        ),
    ]
    write_text(path, "\n".join(body) + "\n")


def write_active_priority(path: Path, rows: list[dict[str, Any]]) -> None:
    table_rows = [
        [
            row.get("frame_id", ""),
            row.get("timestamp_sec", ""),
            row.get("suggested_label", ""),
            row.get("suggested_label_confidence", ""),
            row.get("production_rf_top1", ""),
            row.get("review_priority", ""),
            row.get("priority_reasons", ""),
        ]
        for row in rows
    ]
    write_text(
        path,
        "\n".join(
            [
                "# Active Labeling Priority",
                "",
                "Status: human-review priority list. Suggested labels are not ground truth and are not independent teacher labels.",
                "",
                markdown_table(
                    ["Frame", "Time", "Suggested", "Suggestion confidence", "RF top1", "Priority", "Reason"],
                    table_rows,
                ),
            ]
        )
        + "\n",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create raw sensor manual-label candidate package.")
    p.add_argument("--frames-manifest", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--domain-shift-json", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--contact-sheet", type=Path, default=Path("docs/figures/ml/raw_sensor_labeling_contact_sheet.png"))
    p.add_argument("--max-frames", type=int, default=120)
    p.add_argument("--include-suggestions", type=Path, help="Optional suggested-label CSV from tools/suggest_sensor_labels.py.")
    p.add_argument("--priority-md", type=Path, default=Path("docs/tables/ml/active_labeling_priority.md"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pred_rows = _read_csv(args.predictions)
    suggestion_mode = bool(args.include_suggestions)
    if suggestion_mode:
        selected = select_suggestion_candidates(_read_csv(args.include_suggestions), args.max_frames)
    else:
        selected = select_label_candidates(pred_rows, args.max_frames)
    review_dir = args.out_dir / "frames"
    review_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for row in selected:
        src = Path(str(row["frame_path"]))
        if not src.is_file():
            continue
        dst = review_dir / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        row["frame_path"] = str(dst)
        copied.append(dst)
    if suggestion_mode:
        write_autofilled_label_template(args.template, selected)
        write_active_priority(args.priority_md, selected)
    else:
        write_label_template(args.template, selected)
    if copied:
        write_modality_review_sheet(copied[:12], args.out_dir / "manual_label_contact_sheet.png", title="manual label candidates")
        write_modality_review_sheet(copied[:12], args.contact_sheet, title="manual label candidates")
    write_protocol(args.protocol, args.template, args.out_dir, len(selected))
    print(f"Wrote {args.template}")
    print(f"Wrote {args.protocol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
