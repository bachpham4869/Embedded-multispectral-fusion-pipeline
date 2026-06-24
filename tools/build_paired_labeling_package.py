#!/usr/bin/env python3
"""Build a manual-label review package for paired NIR/thermal captures."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ml_metadata_utils import markdown_table, write_text


PAIRED_LABEL_COLUMNS = [
    "pair_id",
    "session_id",
    "frame_idx",
    "timestamp_sec",
    "nir_raw_path",
    "thermal_raw_path",
    "fusion_output_path",
    "model_top1",
    "model_confidence",
    "suggested_label",
    "manual_label",
    "label_confidence",
    "notes",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_paired_label_template(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PAIRED_LABEL_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "pair_id": row.get("pair_id", ""),
                    "session_id": row.get("session_id", ""),
                    "frame_idx": row.get("frame_idx", ""),
                    "timestamp_sec": row.get("timestamp_sec", ""),
                    "nir_raw_path": row.get("nir_raw_path", ""),
                    "thermal_raw_path": row.get("thermal_raw_path", ""),
                    "fusion_output_path": row.get("fusion_output_path", ""),
                    "model_top1": row.get("model_top1", row.get("top1_label", "")),
                    "model_confidence": row.get("model_confidence", row.get("top1_proba", "")),
                    "suggested_label": row.get("suggested_label", row.get("top1_label", "")),
                    "manual_label": "",
                    "label_confidence": "",
                    "notes": "",
                }
            )


def write_paired_not_measured(path: Path) -> None:
    write_text(
        path,
        "\n".join(
            [
                "# Paired Sensor Labeled Evaluation",
                "",
                "Status: `not measured`.",
                "",
                "No trusted paired labels were found. Paired data is used only for domain shift, confidence, abstention, prediction distribution, uncertainty timeline, and manual labeling package evidence.",
            ]
        )
        + "\n",
    )


def _frame_from_video_fragment(path_text: str) -> np.ndarray | None:
    if not path_text:
        return None
    if "#frame=" in path_text:
        video_path_text, frame_text = path_text.split("#frame=", 1)
        frame_idx = int(float(frame_text))
    else:
        video_path_text, frame_idx = path_text, 0
    video_path = Path(video_path_text)
    if not video_path.is_absolute():
        video_path = REPO_ROOT / video_path
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
    finally:
        cap.release()
    return frame if ok else None


def _resize_for_sheet(img: np.ndarray, width: int = 180) -> np.ndarray:
    h, w = img.shape[:2]
    scale = width / max(1, w)
    return cv.resize(img, (width, max(1, int(h * scale))), interpolation=cv.INTER_AREA)


def write_contact_sheet(rows: list[dict[str, Any]], path: Path, *, max_pairs: int = 12) -> None:
    tiles: list[np.ndarray] = []
    for row in rows[:max_pairs]:
        nir = _frame_from_video_fragment(str(row.get("nir_raw_path", "")))
        thermal = _frame_from_video_fragment(str(row.get("thermal_raw_path", "")))
        if nir is None:
            continue
        nir = _resize_for_sheet(nir)
        thermal = _resize_for_sheet(thermal) if thermal is not None else np.zeros_like(nir)
        h = max(nir.shape[0], thermal.shape[0])
        canvas = np.zeros((h + 24, nir.shape[1] + thermal.shape[1], 3), dtype=np.uint8)
        canvas[: nir.shape[0], : nir.shape[1]] = nir
        canvas[: thermal.shape[0], nir.shape[1] :] = thermal
        cv.putText(canvas, str(row.get("pair_id", "")), (4, h + 16), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv.LINE_AA)
        tiles.append(canvas)
    if not tiles:
        return
    width = max(tile.shape[1] for tile in tiles)
    padded = []
    for tile in tiles:
        if tile.shape[1] < width:
            pad = np.zeros((tile.shape[0], width - tile.shape[1], 3), dtype=np.uint8)
            tile = np.hstack([tile, pad])
        padded.append(tile)
    sheet = np.vstack(padded)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(path), sheet)


def write_protocol(path: Path, template: Path, contact_sheet: Path) -> None:
    write_text(
        path,
        "\n".join(
            [
                "# Paired Data Labeling Protocol",
                "",
                "Use the paired NIR/thermal contact sheet and CSV template for manual review. Model outputs are suggestions only and are not ground truth.",
                "",
                "Allowed labels: `night_clear`, `normal_night`, `normal_day`, `fog`, `rain`, `glare`, `backlight`, `nir_night`; use `transition` only for true dawn/dusk transient scenes.",
                "",
                markdown_table(
                    ["Item", "Value"],
                    [
                        ["template", str(template)],
                        ["contact_sheet", str(contact_sheet)],
                        ["metric_status", "paired labeled eval not measured until trusted labels exist"],
                    ],
                ),
            ]
        )
        + "\n",
    )


def _join_rows(manifest_rows: list[dict[str, str]], prediction_rows: list[dict[str, str]], limit: int) -> list[dict[str, Any]]:
    by_pair = {row.get("pair_id"): row for row in prediction_rows if row.get("pair_id")}
    selected: list[dict[str, Any]] = []
    for row in manifest_rows:
        item: dict[str, Any] = dict(row)
        pred = by_pair.get(row.get("pair_id"), {})
        item["model_top1"] = pred.get("top1_label", "")
        item["model_confidence"] = pred.get("top1_proba", "")
        item["suggested_label"] = pred.get("top1_label", "")
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a paired NIR/thermal manual-label template and contact sheet.")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/paired_eval/paired_ml_manifest.csv"))
    parser.add_argument("--predictions", type=Path, default=Path("artifacts/paired_eval/paired_nir_predictions.csv"))
    parser.add_argument("--template", type=Path, default=Path("artifacts/paired_eval/manual_label_template_paired.csv"))
    parser.add_argument("--contact-sheet", type=Path, default=Path("docs/figures/ml/paired_manual_label_contact_sheet.png"))
    parser.add_argument("--protocol", type=Path, default=Path("docs/ml/PAIRED_DATA_LABELING_PROTOCOL.md"))
    parser.add_argument("--not-measured", type=Path, default=Path("docs/tables/ml/paired_sensor_labeled_eval.md"))
    parser.add_argument("--max-pairs", type=int, default=120)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_rows = _read_csv(args.manifest)
    prediction_rows = _read_csv(args.predictions) if args.predictions.is_file() else []
    selected = _join_rows(manifest_rows, prediction_rows, args.max_pairs)
    write_paired_label_template(args.template, selected)
    write_contact_sheet(selected, args.contact_sheet)
    write_protocol(args.protocol, args.template, args.contact_sheet)
    write_paired_not_measured(args.not_measured)
    print(f"Wrote {args.template}")
    print(f"Wrote {args.protocol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
