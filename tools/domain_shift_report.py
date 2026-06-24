#!/usr/bin/env python3
"""Domain-shift report for offline optical RGB-proxy train/test vs raw sensor frames."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ml_metadata_utils import FEATURE_KEYS, markdown_table, read_jsonl, write_text
from tools.predict_sensor_frames import PROXY_INFERENCE_SCOPE, STRATIFY_FIELDS, predict_rows, summarize_prediction_rows


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    if expected.size == 0 or actual.size == 0:
        return math.nan
    edges = np.percentile(expected, np.linspace(0, 100, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)
    exp_pct = np.clip(exp_counts / max(1, exp_counts.sum()), 1e-6, None)
    act_pct = np.clip(act_counts / max(1, act_counts.sum()), 1e-6, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def compute_feature_drift(feature: str, train_values: np.ndarray, sensor_values: np.ndarray) -> dict[str, Any]:
    from scipy.stats import ks_2samp, wasserstein_distance

    train_values = np.asarray(train_values, dtype=float)
    sensor_values = np.asarray(sensor_values, dtype=float)
    p5, p95 = np.percentile(train_values, [5, 95]) if len(train_values) else (math.nan, math.nan)
    out_of_range = np.mean((sensor_values < p5) | (sensor_values > p95)) if len(sensor_values) else math.nan
    ks = ks_2samp(train_values, sensor_values).statistic if len(train_values) and len(sensor_values) else math.nan
    wd = wasserstein_distance(train_values, sensor_values) if len(train_values) and len(sensor_values) else math.nan
    return {
        "feature": feature,
        "train_mean": float(np.mean(train_values)) if len(train_values) else math.nan,
        "sensor_mean": float(np.mean(sensor_values)) if len(sensor_values) else math.nan,
        "train_std": float(np.std(train_values)) if len(train_values) else math.nan,
        "sensor_std": float(np.std(sensor_values)) if len(sensor_values) else math.nan,
        "train_median": float(np.median(train_values)) if len(train_values) else math.nan,
        "sensor_median": float(np.median(sensor_values)) if len(sensor_values) else math.nan,
        "train_p5": float(p5),
        "train_p95": float(p95),
        "sensor_p5": float(np.percentile(sensor_values, 5)) if len(sensor_values) else math.nan,
        "sensor_p95": float(np.percentile(sensor_values, 95)) if len(sensor_values) else math.nan,
        "ks_statistic": float(ks),
        "wasserstein_distance": float(wd),
        "psi": _psi(train_values, sensor_values),
        "out_of_range_rate": float(out_of_range),
    }


def _matrix(rows: list[dict[str, Any]], features: list[str]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in features] for row in rows], dtype=float)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["feature"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(x: Any) -> str:
    try:
        val = float(x)
    except Exception:
        return str(x)
    return "n/a" if math.isnan(val) else f"{val:.4f}"


def _stratification_markdown(pred_summary: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for field in STRATIFY_FIELDS:
        key = f"by_{field}"
        if not pred_summary.get(key):
            continue
        lines.extend(
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
                            _fmt(item["abstention_rate"]),
                            item["top1_distribution"],
                        ]
                        for value, item in pred_summary[key].items()
                    ],
                ),
            ]
        )
    return lines


def write_drift_tables(
    summary_path: Path,
    feature_path: Path,
    pred_path: Path,
    drift_rows: list[dict[str, Any]],
    pred_summary: dict[str, Any],
    *,
    title_prefix: str = "Raw Sensor",
) -> None:
    high = sorted(drift_rows, key=lambda r: (r["ks_statistic"], r["out_of_range_rate"]), reverse=True)[:12]
    inference_scope = pred_summary["inference_scope"]
    write_text(
        summary_path,
        "\n".join(
            [
                f"# {title_prefix} Domain Shift Summary",
                "",
                f"Status: unlabeled / preliminary. Inference scope: `{inference_scope}`. No sensor-real accuracy is claimed.",
                "",
                markdown_table(
                    ["Metric", "Value"],
                    [
                        ["sensor_rows", pred_summary["row_count"]],
                        ["abstention_rate_tau1_0.62", _fmt(pred_summary["abstention_rate"])],
                        ["mean_confidence", _fmt(pred_summary["mean_confidence"])],
                        ["mean_entropy", _fmt(pred_summary["mean_entropy"])],
                        ["inference_scope", inference_scope],
                    ],
                ),
                "",
                "## Highest Drift Features",
                "",
                markdown_table(
                    ["Feature", "KS", "Wasserstein", "PSI", "Out-of-range"],
                    [[r["feature"], _fmt(r["ks_statistic"]), _fmt(r["wasserstein_distance"]), _fmt(r["psi"]), _fmt(r["out_of_range_rate"])] for r in high],
                ),
            ]
            + _stratification_markdown(pred_summary)
        )
        + "\n",
    )
    write_text(
        feature_path,
        "\n".join(
            [
                f"# {title_prefix} Feature Drift",
                "",
                "Compared against Phase 4 duplicate-cluster-aware train rows. Sensor rows are unlabeled unless trusted labels are explicitly documented.",
                "",
                markdown_table(
                    ["Feature", "Train mean", "Sensor mean", "KS", "Wasserstein", "PSI", "Out-of-range"],
                    [
                        [
                            r["feature"],
                            _fmt(r["train_mean"]),
                            _fmt(r["sensor_mean"]),
                            _fmt(r["ks_statistic"]),
                            _fmt(r["wasserstein_distance"]),
                            _fmt(r["psi"]),
                            _fmt(r["out_of_range_rate"]),
                        ]
                        for r in drift_rows
                    ],
                ),
            ]
        )
        + "\n",
    )
    write_text(
        pred_path,
        "\n".join(
            [
                f"# {title_prefix} Prediction Distribution",
                "",
                f"Status: unlabeled / preliminary; `{inference_scope}`.",
                "",
                markdown_table(["Label", "Count"], [[k, v] for k, v in pred_summary["top1_distribution"].items()]),
            ]
        )
        + "\n",
    )


def write_figures(train_rows: list[dict[str, Any]], sensor_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]], *, hist_path: Path, pca_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return
    conf = [float(row["top1_proba"]) for row in pred_rows]
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(conf, bins=20, color="#4c78a8", alpha=0.85)
    ax.axvline(0.62, color="#d62728", linestyle="--", label="tau1=0.62")
    ax.set_title("Raw sensor confidence histogram")
    ax.set_xlabel("top-1 probability")
    ax.set_ylabel("frames")
    ax.legend()
    fig.tight_layout()
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)

    X_train = _matrix(train_rows, list(FEATURE_KEYS))
    X_sensor = _matrix(sensor_rows, list(FEATURE_KEYS))
    X = np.vstack([X_train, X_sensor])
    Xs = StandardScaler().fit_transform(X)
    coords = PCA(n_components=2, random_state=42).fit_transform(Xs)
    n = len(X_train)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(coords[:n, 0], coords[:n, 1], s=5, alpha=0.25, label="offline train")
    ax.scatter(coords[n:, 0], coords[n:, 1], s=9, alpha=0.75, label="raw sensor")
    ax.set_title("Feature PCA: train vs raw sensor")
    ax.legend()
    fig.tight_layout()
    pca_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pca_path, dpi=150)
    plt.close(fig)


def update_domain_doc(path: Path, summary: dict[str, Any], *, title_prefix: str = "Raw sensor") -> None:
    body = [
        "# Domain Shift Evaluation",
        "",
        f"Status: measured on unlabeled {title_prefix.lower()} sampled frames. No sensor-real accuracy is claimed.",
        "",
        f"Current prediction scope is `{summary['inference_scope']}` because the production bundle uses the selected scaler. This is not live NIR/LWIR validation.",
        "",
        markdown_table(
            ["Metric", "Value"],
            [
                ["sensor_rows", summary["row_count"]],
                ["abstention_rate_tau1_0.62", _fmt(summary["abstention_rate"])],
                ["mean_confidence", _fmt(summary["mean_confidence"])],
                ["mean_entropy", _fmt(summary["mean_entropy"])],
                ["label_status", "unlabeled / preliminary"],
            ],
        ),
        "",
        "Manual labels are required before accuracy, balanced accuracy, F1, confusion matrix, or calibration-on-sensor claims can be made.",
    ]
    write_text(path, "\n".join(body) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare offline train/test features against raw sensor frame features.")
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--sensor", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--scaler-group", default="rgb")
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--summary", type=Path, default=Path("docs/tables/ml/raw_sensor_domain_shift_summary.md"))
    p.add_argument("--feature-drift", type=Path, default=Path("docs/tables/ml/raw_sensor_feature_drift.md"))
    p.add_argument("--prediction-distribution", type=Path, default=Path("docs/tables/ml/raw_sensor_prediction_distribution.md"))
    p.add_argument("--domain-doc", type=Path, default=Path("docs/ml/DOMAIN_SHIFT_EVALUATION.md"))
    p.add_argument("--confidence-hist", type=Path, default=Path("docs/figures/ml/raw_sensor_confidence_hist.png"))
    p.add_argument("--pca-fig", type=Path, default=Path("docs/figures/ml/raw_sensor_feature_pca.png"))
    p.add_argument("--title-prefix", default="Raw Sensor")
    p.add_argument("--inference-scope", default="RGB-scaler proxy inference")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_rows = read_jsonl(args.train)
    test_rows = read_jsonl(args.test)
    sensor_rows = read_jsonl(args.sensor)
    offline_rows = train_rows + test_rows
    drift_rows = [
        compute_feature_drift(
            feature,
            np.asarray([float(row[feature]) for row in train_rows], dtype=float),
            np.asarray([float(row[feature]) for row in sensor_rows], dtype=float),
        )
        for feature in FEATURE_KEYS
    ]
    pred_rows = predict_rows(sensor_rows, args.model, scaler_group=args.scaler_group, tau1=0.62, inference_scope=args.inference_scope)
    pred_summary = summarize_prediction_rows(pred_rows, tau1=0.62, inference_scope=args.inference_scope)
    write_drift_tables(args.summary, args.feature_drift, args.prediction_distribution, drift_rows, pred_summary, title_prefix=args.title_prefix)
    write_figures(train_rows, sensor_rows, pred_rows, hist_path=args.confidence_hist, pca_path=args.pca_fig)
    update_domain_doc(args.domain_doc, pred_summary, title_prefix=args.title_prefix)
    payload = {
        "status": "unlabeled / preliminary",
        "inference_scope": args.inference_scope,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "sensor_rows": len(sensor_rows),
        "feature_drift": drift_rows,
        "prediction_summary": pred_summary,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
