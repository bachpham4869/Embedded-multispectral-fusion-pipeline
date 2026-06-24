#!/usr/bin/env python3
"""Read-only production-model predictions for unlabeled raw sensor frame features."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ml_metadata_utils import markdown_table, read_jsonl, write_text

PROXY_INFERENCE_SCOPE = "RGB-scaler proxy inference, not validated NIR classifier accuracy"
DEFAULT_INFERENCE_SCOPE = "RGB-scaler proxy inference"
STRATIFY_FIELDS = ("pairing_tier", "session_id", "nir_modality", "thermal_modality", "label_source")


def _entropy(proba: np.ndarray) -> float:
    p = proba[proba > 0]
    return float(-np.sum(p * np.log2(p)))


def _stratified_counts(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = str(row.get(field) or "unknown")
        buckets[value].append(row)
    out: dict[str, dict[str, Any]] = {}
    for value, items in sorted(buckets.items()):
        accepted = sum(1 for row in items if row.get("accepted_tau1"))
        out[value] = {
            "row_count": len(items),
            "accepted_count": accepted,
            "abstention_rate": float(1.0 - accepted / len(items)) if items else math.nan,
            "top1_distribution": dict(Counter(str(row.get("top1_label", "unknown")) for row in items)),
        }
    return out


def summarize_prediction_rows(rows: list[dict[str, Any]], *, tau1: float, inference_scope: str) -> dict[str, Any]:
    count = len(rows)
    accepted = sum(1 for row in rows if row.get("accepted_tau1"))
    top1 = Counter(str(row.get("top1_label", "unknown")) for row in rows)
    by_video: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_video[str(row.get("video_id", "unknown"))][str(row.get("top1_label", "unknown"))] += 1
    summary = {
        "inference_scope": inference_scope,
        "row_count": count,
        "tau1": tau1,
        "accepted_count": accepted,
        "abstention_rate": float(1.0 - accepted / count) if count else math.nan,
        "mean_confidence": float(np.mean([float(r.get("top1_proba", 0.0)) for r in rows])) if rows else math.nan,
        "mean_entropy": float(np.mean([float(r.get("posterior_entropy", 0.0)) for r in rows])) if rows else math.nan,
        "top1_distribution": dict(sorted(top1.items())),
        "per_video_top1_distribution": {k: dict(v) for k, v in sorted(by_video.items())},
    }
    for field in STRATIFY_FIELDS:
        summary[f"by_{field}"] = _stratified_counts(rows, field)
    return summary


def _feature_matrix(rows: list[dict[str, Any]], feature_set: list[str]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in feature_set] for row in rows], dtype=np.float32)


def predict_rows(
    feature_rows: list[dict[str, Any]],
    model_path: Path | dict[str, Any],
    *,
    scaler_group: str,
    tau1: float,
    inference_scope: str = DEFAULT_INFERENCE_SCOPE,
) -> list[dict[str, Any]]:
    bundle = model_path if isinstance(model_path, dict) else joblib.load(model_path)
    feature_set = list(bundle["feature_set"])
    X = _feature_matrix(feature_rows, feature_set)
    scaler = bundle["scalers"][scaler_group]
    Xs = scaler.transform(X)
    rf = bundle["rf"]
    proba = np.asarray(rf.predict_proba(Xs), dtype=float)
    class_values = list(getattr(rf, "classes_", range(proba.shape[1])))
    int_to_label = {int(k): v for k, v in (bundle.get("class_int_to_label") or {}).items()}
    out: list[dict[str, Any]] = []
    for row, probs in zip(feature_rows, proba):
        order = np.argsort(probs)[::-1]
        top1_i = int(order[0])
        top2_i = int(order[1]) if len(order) > 1 else top1_i
        top1_class = int(class_values[top1_i])
        top2_class = int(class_values[top2_i])
        top1_proba = float(probs[top1_i])
        pred_row = {
                "frame_id": row.get("frame_id") or f"{row.get('video_id')}:{row.get('frame_idx')}",
                "video_id": row.get("video_id", "unknown"),
                "pair_id": row.get("pair_id", ""),
                "session_id": row.get("session_id", row.get("video_id", "unknown")),
                "frame_idx": row.get("frame_idx", ""),
                "timestamp_sec": row.get("timestamp_sec", row.get("ts", "")),
                "frame_path": row.get("frame_path", ""),
                "nir_raw_path": row.get("nir_raw_path", ""),
                "thermal_raw_path": row.get("thermal_raw_path", ""),
                "fusion_output_path": row.get("fusion_output_path", ""),
                "pairing_tier": row.get("pairing_tier", ""),
                "nir_modality": row.get("nir_modality", ""),
                "thermal_modality": row.get("thermal_modality", ""),
                "nir_channel": row.get("nir_channel", ""),
                "thermal_channel": row.get("thermal_channel", ""),
                "label_source": row.get("label_source", ""),
                "evidence_type": row.get("evidence_type", ""),
                "top1_label": int_to_label.get(top1_class, str(top1_class)),
                "top1_proba": top1_proba,
                "top2_label": int_to_label.get(top2_class, str(top2_class)),
                "top2_proba": float(probs[top2_i]),
                "accepted_tau1": top1_proba >= tau1,
                "posterior_entropy": _entropy(probs),
                "inference_scope": inference_scope,
                "label_status": "unlabeled / preliminary",
        }
        out.append(pred_row)
    return out


def write_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "frame_id",
        "video_id",
        "pair_id",
        "session_id",
        "frame_idx",
        "timestamp_sec",
        "frame_path",
        "nir_raw_path",
        "thermal_raw_path",
        "fusion_output_path",
        "pairing_tier",
        "nir_modality",
        "thermal_modality",
        "nir_channel",
        "thermal_channel",
        "label_source",
        "evidence_type",
        "top1_label",
        "top1_proba",
        "top2_label",
        "top2_proba",
        "accepted_tau1",
        "posterior_entropy",
        "inference_scope",
        "label_status",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(path: Path, summary: dict[str, Any], *, title_prefix: str = "Raw Sensor") -> None:
    strat_lines: list[str] = []
    for field in STRATIFY_FIELDS:
        key = f"by_{field}"
        if not summary.get(key):
            continue
        strat_lines.extend(
            [
                "",
                f"## By {field}",
                "",
                markdown_table(
                    [field, "Rows", "Accepted", "Abstention", "Top-1 distribution"],
                    [
                        [
                            value,
                            item["row_count"],
                            item["accepted_count"],
                            f"{item['abstention_rate']:.4f}",
                            item["top1_distribution"],
                        ]
                        for value, item in summary[key].items()
                    ],
                ),
            ]
        )
    body = [
        f"# {title_prefix} Prediction Summary",
        "",
        f"Status: unlabeled / preliminary. Inference scope: `{summary['inference_scope']}`. No sensor accuracy is claimed.",
        "",
        markdown_table(
            ["Metric", "Value"],
            [
                ["row_count", summary["row_count"]],
                ["tau1", summary["tau1"]],
                ["accepted_count", summary["accepted_count"]],
                ["abstention_rate", f"{summary['abstention_rate']:.4f}"],
                ["mean_confidence", f"{summary['mean_confidence']:.4f}"],
                ["mean_entropy", f"{summary['mean_entropy']:.4f}"],
                ["inference_scope", summary["inference_scope"]],
            ],
        ),
        "",
        "## Top-1 Distribution",
        "",
        markdown_table(["Label", "Count"], [[k, v] for k, v in summary["top1_distribution"].items()]),
    ]
    body.extend(strat_lines)
    write_text(path, "\n".join(body) + "\n")


def write_timeline_figures(rows: list[dict[str, Any]], out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_video[str(row.get("video_id", "unknown"))].append(row)
    for video_id, items in by_video.items():
        items = sorted(items, key=lambda r: float(r.get("timestamp_sec") or 0.0))
        x = [float(r.get("timestamp_sec") or 0.0) for r in items]
        conf = [float(r.get("top1_proba") or 0.0) for r in items]
        entropy = [float(r.get("posterior_entropy") or 0.0) for r in items]
        fig, ax1 = plt.subplots(figsize=(8, 3))
        ax1.plot(x, conf, label="top1 confidence", color="#1f77b4")
        ax1.axhline(0.62, color="#888", linestyle="--", linewidth=1, label="tau1=0.62")
        ax1.set_xlabel("time (s)")
        ax1.set_ylabel("confidence")
        ax2 = ax1.twinx()
        ax2.plot(x, entropy, label="entropy", color="#d62728", alpha=0.65)
        ax2.set_ylabel("entropy")
        fig.suptitle(f"Raw sensor prediction timeline: {video_id}")
        fig.tight_layout()
        fig.savefig(out_dir / f"raw_sensor_prediction_timeline_{video_id}.png", dpi=150)
        plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict unlabeled raw sensor frame features with a frozen model.")
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--scaler-group", default="rgb")
    p.add_argument("--tau1", type=float, default=0.62)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--timeline-dir", type=Path)
    p.add_argument("--inference-scope", default=DEFAULT_INFERENCE_SCOPE)
    p.add_argument("--title-prefix", default="Raw Sensor")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_jsonl(args.features)
    pred_rows = predict_rows(rows, args.model, scaler_group=args.scaler_group, tau1=args.tau1, inference_scope=args.inference_scope)
    write_predictions_csv(args.out_csv, pred_rows)
    summary = summarize_prediction_rows(pred_rows, tau1=args.tau1, inference_scope=args.inference_scope)
    write_summary_md(args.summary, summary, title_prefix=args.title_prefix)
    if args.timeline_dir:
        write_timeline_figures(pred_rows, args.timeline_dir)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
