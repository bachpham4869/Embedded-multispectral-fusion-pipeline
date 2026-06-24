#!/usr/bin/env python3
"""Analyze RF/heuristic suggested-label consistency without treating it as accuracy."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ml_metadata_utils import markdown_table, write_text


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row}) if rows else ["frame_id"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def summarize_suggestion_consistency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    agreement = sum(1 for row in rows if _truthy(row.get("rf_heuristic_agreement")))
    review = sum(1 for row in rows if _truthy(row.get("requires_human_review")))
    by_video: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_video[str(row.get("video_id", "unknown"))][str(row.get("suggested_label", "unknown"))] += 1
    high_conf_disagreements = [
        row
        for row in rows
        if not _truthy(row.get("rf_heuristic_agreement")) and _as_float(row.get("suggested_label_confidence")) >= 0.70
    ]
    return {
        "row_count": count,
        "agreement_count": agreement,
        "agreement_rate": float(agreement / count) if count else 0.0,
        "requires_human_review_count": review,
        "requires_human_review_rate": float(review / count) if count else 0.0,
        "suggested_label_distribution": dict(sorted(Counter(str(row.get("suggested_label", "unknown")) for row in rows).items())),
        "production_rf_distribution": dict(sorted(Counter(str(row.get("production_rf_top1", "unknown")) for row in rows).items())),
        "review_priority_distribution": dict(sorted(Counter(str(row.get("review_priority", "unknown")) for row in rows).items())),
        "high_confidence_disagreement_count": len(high_conf_disagreements),
        "per_video_suggested_distribution": {k: dict(v) for k, v in sorted(by_video.items())},
        "analysis_scope": "RF/heuristic self-consistency only; not independent confidence",
    }


def write_summary_markdown(summary: dict[str, Any]) -> str:
    return (
        "\n".join(
            [
                "# Suggested Label Consistency Summary",
                "",
                "Status: agreement/consistency analysis only. RF and heuristic signals are not independent, so this table does not measure sensor performance.",
                "",
                markdown_table(
                    ["Field", "Value"],
                    [
                        ["row_count", summary["row_count"]],
                        ["agreement_count", summary["agreement_count"]],
                        ["agreement_rate", f"{summary['agreement_rate']:.4f}"],
                        ["requires_human_review_count", summary["requires_human_review_count"]],
                        ["requires_human_review_rate", f"{summary['requires_human_review_rate']:.4f}"],
                        ["high_confidence_disagreement_count", summary["high_confidence_disagreement_count"]],
                        ["analysis_scope", summary["analysis_scope"]],
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
        + "\n"
    )


def _timeline(rows: list[dict[str, Any]], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    items = sorted(rows, key=lambda r: (str(r.get("video_id", "")), _as_float(r.get("timestamp_sec"))))
    if not items:
        return
    labels = sorted({str(r.get("suggested_label", "unknown")) for r in items})
    label_to_y = {label: i for i, label in enumerate(labels)}
    x = [_as_float(r.get("timestamp_sec")) for r in items]
    y = [label_to_y[str(r.get("suggested_label", "unknown"))] for r in items]
    conf = [_as_float(r.get("suggested_label_confidence")) for r in items]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    sc = ax.scatter(x, y, c=conf, cmap="viridis", s=14)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("time (s)")
    ax.set_title("Suggested-label timeline")
    fig.colorbar(sc, ax=ax, label="suggestion confidence")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate suggested-label consistency without ground-truth metrics.")
    parser.add_argument("--suggestions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--disagreements", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--timeline", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = _read_csv(args.suggestions)
    summary = summarize_suggestion_consistency(rows)
    write_text(args.summary, write_summary_markdown(summary))
    disagreements = [
        row
        for row in rows
        if not _truthy(row.get("rf_heuristic_agreement")) or _truthy(row.get("requires_human_review"))
    ]
    disagreements.sort(key=lambda r: _as_float(r.get("priority_score")), reverse=True)
    _write_csv(args.disagreements, disagreements[:100])
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.timeline:
        _timeline(rows, args.timeline)
    print(f"Wrote {args.summary}")
    print(f"Wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
