#!/usr/bin/env python3
"""Generate RF/heuristic suggested labels for raw-sensor manual review.

This is not an independent teacher and does not create weak ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ml_metadata_utils import git_branch_name, git_commit_hash, markdown_table, read_jsonl, sha256_file, write_text

LOW_SUPPORT_REVIEW_CLASSES = {"glare", "backlight", "transition", "nir_night"}

SUGGESTED_LABEL_FIELDS = [
    "frame_id",
    "video_id",
    "frame_path",
    "timestamp_sec",
    "suggested_label",
    "suggested_label_confidence",
    "suggestion_method",
    "suggestion_model",
    "suggestion_model_version_or_hash",
    "top_candidates",
    "production_rf_top1",
    "production_rf_confidence",
    "heuristic_label",
    "heuristic_confidence",
    "heuristic_reason",
    "rf_heuristic_agreement",
    "uncertainty_score",
    "evidence_label",
    "label_source",
    "requires_human_review",
    "review_priority",
    "priority_score",
    "priority_reasons",
    "caveat",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUGGESTED_LABEL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def _frame_id(row: dict[str, Any]) -> str:
    if row.get("frame_id"):
        return str(row["frame_id"])
    return f"{row.get('video_id', 'unknown')}:{row.get('frame_idx', '')}"


def heuristic_label_from_features(feature_row: dict[str, Any], modality_guess: str) -> tuple[str, float, str]:
    mean = _as_float(feature_row.get("nir_mean_brightness"))
    std = _as_float(feature_row.get("nir_std"))
    p95 = _as_float(feature_row.get("nir_p95"))
    glare = _as_float(feature_row.get("nir_glare_score"))
    dark = _as_float(feature_row.get("nir_dark_fraction"))
    sharpness = _as_float(feature_row.get("nir_sharpness"))
    saturation = _as_float(feature_row.get("nir_saturation_mean"))
    modality = (modality_guess or "").lower()

    if glare >= 0.08 or p95 >= 245:
        return "glare", min(0.78, 0.45 + glare * 3.0), "high glare/saturation proxy"
    if glare >= 0.035 and dark >= 0.15:
        return "backlight", 0.58, "bright highlights with dark foreground fraction"
    if dark >= 0.65 and mean < 60:
        if "nir" in modality or "ir" in modality:
            return "nir_night", 0.62, "dark low-saturation frame with NIR/IR modality hint"
        return "normal_night", 0.55, "dark frame but modality is unknown optical"
    if sharpness < 180 and std < 42 and mean > 40:
        return "fog", 0.58, "low sharpness and low local intensity spread"
    if mean >= 70 and dark < 0.20:
        return "normal_day", 0.52, "daylight-like brightness and low dark fraction"
    if saturation < 8 and mean < 85:
        return "normal_night", 0.45, "low saturation and low brightness weak night cue"
    return "unknown_or_out_of_scope", 0.25, "no heuristic rule is strong enough"


def _review_priority(
    *,
    suggested_label: str,
    rf_label: str,
    rf_conf: float,
    heuristic_label: str,
    heuristic_conf: float,
    entropy: float,
    modality_guess: str,
    low_confidence: float,
) -> tuple[str, float, list[str], bool]:
    score = 0.0
    reasons: list[str] = []
    if rf_conf < low_confidence:
        score += 3.0
        reasons.append("low RF confidence")
    elif rf_conf < 0.62:
        score += 1.5
        reasons.append("below tau1=0.62")
    if entropy >= 1.0:
        score += min(2.0, entropy)
        reasons.append("high posterior entropy")
    if heuristic_label != "unknown_or_out_of_scope" and heuristic_label != rf_label:
        score += 2.0
        reasons.append("RF/heuristic disagreement")
    if suggested_label in LOW_SUPPORT_REVIEW_CLASSES:
        score += 2.0
        reasons.append(f"{suggested_label} needs targeted review")
    if suggested_label == "transition":
        score += 1.0
        reasons.append("transition requires manual review")
    if suggested_label == "nir_night" and "unknown" in (modality_guess or "").lower():
        score += 2.0
        reasons.append("nir_night with unknown optical modality")
    if suggested_label == "unknown_or_out_of_scope":
        score += 2.0
        reasons.append("unknown suggestion")
    if heuristic_conf < 0.5:
        score += 0.5
        reasons.append("weak heuristic cue")
    if score >= 4.0:
        return "high", score, reasons, True
    if score >= 2.0:
        return "medium", score, reasons, True
    return "low", score, reasons, False


def build_suggestion_row(
    *,
    frame_row: dict[str, Any],
    prediction_row: dict[str, Any],
    feature_row: dict[str, Any],
    high_confidence: float,
    low_confidence: float,
) -> dict[str, Any]:
    rf_label = str(prediction_row.get("top1_label") or "unknown_or_out_of_scope")
    rf_conf = _as_float(prediction_row.get("top1_proba"))
    top2_label = str(prediction_row.get("top2_label") or "")
    top2_conf = _as_float(prediction_row.get("top2_proba"))
    entropy = _as_float(prediction_row.get("posterior_entropy"))
    modality_guess = str(frame_row.get("modality_guess") or feature_row.get("modality_guess") or "unknown optical")
    heuristic_label, heuristic_conf, heuristic_reason = heuristic_label_from_features(feature_row, modality_guess)

    if rf_conf < low_confidence and heuristic_conf < 0.55:
        suggested_label = "unknown_or_out_of_scope"
        suggested_conf = max(rf_conf, heuristic_conf)
    elif heuristic_conf >= high_confidence and heuristic_conf > rf_conf + 0.05:
        suggested_label = heuristic_label
        suggested_conf = heuristic_conf
    else:
        suggested_label = rf_label
        suggested_conf = rf_conf

    priority, priority_score, priority_reasons, review = _review_priority(
        suggested_label=suggested_label,
        rf_label=rf_label,
        rf_conf=rf_conf,
        heuristic_label=heuristic_label,
        heuristic_conf=heuristic_conf,
        entropy=entropy,
        modality_guess=modality_guess,
        low_confidence=low_confidence,
    )
    caveats = [
        "suggested_label is RF/heuristic review support, not ground truth",
        "RF/heuristic agreement is not independent evidence",
    ]
    if suggested_label == "transition":
        caveats.append("transition requires manual review")
    if suggested_label == "nir_night" and "unknown" in modality_guess.lower():
        caveats.append("nir_night requires modality confirmation")

    top_candidates = [
        {"label": rf_label, "confidence": rf_conf, "source": "production_rf"},
        {"label": heuristic_label, "confidence": heuristic_conf, "source": "heuristic"},
    ]
    if top2_label:
        top_candidates.append({"label": top2_label, "confidence": top2_conf, "source": "production_rf_top2"})

    return {
        "frame_id": _frame_id(frame_row),
        "video_id": frame_row.get("video_id", feature_row.get("video_id", "")),
        "frame_path": frame_row.get("frame_path", feature_row.get("frame_path", "")),
        "timestamp_sec": frame_row.get("timestamp_sec", feature_row.get("timestamp_sec", "")),
        "suggested_label": suggested_label,
        "suggested_label_confidence": f"{suggested_conf:.6f}",
        "suggestion_method": "rf_heuristic_suggested_label",
        "suggestion_model": "production_rf_plus_rules",
        "suggestion_model_version_or_hash": "read_only_production_rf_and_static_heuristics",
        "top_candidates": json.dumps(top_candidates, sort_keys=True),
        "production_rf_top1": rf_label,
        "production_rf_confidence": f"{rf_conf:.6f}",
        "heuristic_label": heuristic_label,
        "heuristic_confidence": f"{heuristic_conf:.6f}",
        "heuristic_reason": heuristic_reason,
        "rf_heuristic_agreement": str(heuristic_label == rf_label).lower(),
        "uncertainty_score": f"{max(0.0, 1.0 - rf_conf) + entropy:.6f}",
        "evidence_label": "suggested_label",
        "label_source": "rf_heuristic_suggestion",
        "requires_human_review": str(review).lower(),
        "review_priority": priority,
        "priority_score": f"{priority_score:.6f}",
        "priority_reasons": "; ".join(priority_reasons),
        "caveat": "; ".join(caveats),
    }


def _rows_by_frame_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {_frame_id(row): row for row in rows}


def _summarize(rows: list[dict[str, Any]], *, high_confidence: float, low_confidence: float) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "suggested_label_distribution": dict(sorted(Counter(row["suggested_label"] for row in rows).items())),
        "review_priority_distribution": dict(sorted(Counter(row["review_priority"] for row in rows).items())),
        "requires_human_review_count": sum(1 for row in rows if row["requires_human_review"] == "true"),
        "high_confidence_threshold": high_confidence,
        "low_confidence_threshold": low_confidence,
        "label_status": "suggested_label only; not ground truth",
        "weak_label_dataset_created": False,
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    write_text(
        path,
        "\n".join(
            [
                "# Suggested Label Summary",
                "",
                "Status: `suggested_label` review support only. No `auto_weak_label` dataset is created, and no sensor accuracy is claimed.",
                "",
                markdown_table(
                    ["Field", "Value"],
                    [
                        ["row_count", summary["row_count"]],
                        ["requires_human_review_count", summary["requires_human_review_count"]],
                        ["high_confidence_threshold", summary["high_confidence_threshold"]],
                        ["low_confidence_threshold", summary["low_confidence_threshold"]],
                        ["label_status", summary["label_status"]],
                        ["weak_label_dataset_created", summary["weak_label_dataset_created"]],
                    ],
                ),
                "",
                "## Suggested Label Distribution",
                "",
                markdown_table(["Label", "Count"], [[k, v] for k, v in summary["suggested_label_distribution"].items()]),
                "",
                "## Review Priority Distribution",
                "",
                markdown_table(["Priority", "Count"], [[k, v] for k, v in summary["review_priority_distribution"].items()]),
            ]
        )
        + "\n",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create RF/heuristic suggested labels for raw-sensor manual review.")
    parser.add_argument("--frames-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--domain-shift-json", type=Path)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--high-confidence", type=float, default=0.80)
    parser.add_argument("--low-confidence", type=float, default=0.45)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    frame_rows = _read_csv(args.frames_manifest)
    pred_rows = _rows_by_frame_id(_read_csv(args.predictions))
    feature_rows = _rows_by_frame_id(read_jsonl(args.features))
    out_rows: list[dict[str, Any]] = []
    for frame in frame_rows:
        fid = _frame_id(frame)
        pred = pred_rows.get(fid, {})
        feat = feature_rows.get(fid, {})
        if not pred:
            pred = {"top1_label": "unknown_or_out_of_scope", "top1_proba": "0.0", "posterior_entropy": "0.0"}
        out_rows.append(
            build_suggestion_row(
                frame_row=frame,
                prediction_row=pred,
                feature_row=feat,
                high_confidence=args.high_confidence,
                low_confidence=args.low_confidence,
            )
        )
    _write_csv(args.out_csv, out_rows)
    summary = _summarize(out_rows, high_confidence=args.high_confidence, low_confidence=args.low_confidence)
    write_summary(args.summary, summary)
    manifest = {
        "command": " ".join(sys.argv),
        "git_commit": git_commit_hash(),
        "git_branch": git_branch_name(),
        "frames_manifest": str(args.frames_manifest),
        "predictions": str(args.predictions),
        "features": str(args.features),
        "domain_shift_json": str(args.domain_shift_json) if args.domain_shift_json else None,
        "output_csv": str(args.out_csv),
        "output_csv_sha256": sha256_file(args.out_csv),
        "summary": summary,
        "method_status": "review-prioritization only; no independent teacher",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.summary}")
    print(f"Wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
