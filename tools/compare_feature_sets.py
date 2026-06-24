#!/usr/bin/env python3
"""Compare non-production optical feature sets on covered labeled subsets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.train_classifier import apply_scalers, fit_scalers
from smartbinocular.feature_schema import ENV_CLASS_TO_INT, ENV_INT_TO_CLASS
from tools.compare_classifiers import _brier_multiclass, _ece_multiclass, _metric_values, _per_class_rows, build_model
from tools.ml_metadata_utils import FEATURE_KEYS, effective_label, git_branch_name, git_commit_hash, markdown_table, read_jsonl, write_text
from tools.optical_candidate_features import FEATURE_SET_DEFINITIONS, OPTICAL_21_FEATURES, OPTICAL_V2_FEATURES

FEATURE_SETS = {
    "optical_12_baseline": list(FEATURE_KEYS),
    "optical_v2_candidate": [*FEATURE_KEYS, *OPTICAL_V2_FEATURES],
    "optical_21_candidate": list(OPTICAL_21_FEATURES),
    "optical_21_candidate_still": list(FEATURE_SET_DEFINITIONS["optical_21_candidate_still"]["features"]),
    "optical_21_candidate_temporal": list(FEATURE_SET_DEFINITIONS["optical_21_candidate_temporal"]["features"]),
}


def describe_subset_scope(rows: list[dict[str, Any]], *, full_class_count: int = 9) -> dict[str, Any]:
    labels = sorted({effective_label(row) for row in rows})
    return {
        "row_count": len(rows),
        "classes": labels,
        "class_count": len(labels),
        "class_coverage": f"{len(labels)}/{full_class_count}",
        "direct_full_9class_comparison_allowed": len(labels) == full_class_count,
        "limitation": "subset comparison only; do not compare directly with full 9-class baseline" if len(labels) < full_class_count else "full class coverage",
    }


def _compatible(rows: list[dict[str, Any]], feature_set: list[str]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        label = effective_label(row)
        if label not in ENV_CLASS_TO_INT:
            continue
        if all(row.get(name) is not None for name in feature_set):
            out.append(row)
    return out


def _matrix(rows: list[dict[str, Any]], feature_set: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray([[float(row[name]) for name in feature_set] for row in rows], dtype=np.float32)
    y = np.asarray([ENV_CLASS_TO_INT[effective_label(row)] for row in rows], dtype=np.int32)
    groups = np.asarray([row.get("nir_channel") or "rgb" for row in rows])
    return X, y, groups


def _predict_proba(model: Any, X: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not hasattr(model, "predict_proba"):
        return None, None
    try:
        return np.asarray(model.predict_proba(X), dtype=float), np.asarray(model.classes_)
    except Exception:
        return None, None


def _latency_ms(model: Any, X: np.ndarray, repeats: int) -> float:
    if len(X) == 0 or repeats <= 0:
        return math.nan
    sample = X[:1]
    vals = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(sample)
        vals.append((time.perf_counter() - start) * 1000.0)
    return float(np.mean(vals))


def run_one(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], *, feature_set_name: str, model_name: str, seed: int, latency_repeats: int) -> dict[str, Any]:
    feature_set = FEATURE_SETS[feature_set_name]
    definition = FEATURE_SET_DEFINITIONS.get(feature_set_name, {"actual_feature_count": len(feature_set), "status": "legacy feature set"})
    tr = _compatible(train_rows, feature_set)
    te = _compatible(test_rows, feature_set)
    scope = describe_subset_scope(tr + te)
    if len(tr) < 20 or len(te) < 10 or len(scope["classes"]) < 2:
        return {
            "feature_set": feature_set_name,
            "model": model_name,
            "status": "not_run_insufficient_compatible_labeled_rows",
            "train_rows": len(tr),
            "test_rows": len(te),
            "classes": ",".join(scope["classes"]),
            "class_coverage": scope["class_coverage"],
            "feature_count": definition.get("actual_feature_count", len(feature_set)),
            "feature_set_status": definition.get("status", ""),
            "limitation": scope["limitation"],
        }
    X_train, y_train, g_train = _matrix(tr, feature_set)
    X_test, y_test, g_test = _matrix(te, feature_set)
    scalers = fit_scalers(X_train, g_train, "nir_channel")
    X_train = apply_scalers(X_train, g_train, scalers, "nir_channel")
    X_test = apply_scalers(X_test, g_test, scalers, "nir_channel")
    labels = np.asarray(sorted(np.unique(np.concatenate([y_train, y_test]))), dtype=int)
    model = build_model(model_name, seed)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    y_pred = np.asarray(model.predict(X_test), dtype=int)
    metrics = _metric_values(y_test, y_pred, labels)
    proba, proba_classes = _predict_proba(model, X_test)
    ece = _ece_multiclass(y_test, proba, proba_classes) if proba is not None and proba_classes is not None else math.nan
    brier = _brier_multiclass(y_test, proba, proba_classes) if proba is not None and proba_classes is not None else math.nan
    payload = pickle.dumps(model)
    return {
        "feature_set": feature_set_name,
        "model": model_name,
        "status": "subset/preliminary" if not scope["direct_full_9class_comparison_allowed"] else "preliminary",
        "train_rows": len(tr),
        "test_rows": len(te),
        "classes": ",".join(scope["classes"]),
        "class_coverage": scope["class_coverage"],
        "feature_count": definition.get("actual_feature_count", len(feature_set)),
        "feature_set_status": definition.get("status", ""),
        "limitation": scope["limitation"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "ece": ece,
        "brier": brier,
        "train_time_s": train_time,
        "latency_mean_ms": _latency_ms(model, X_test, latency_repeats),
        "model_size_bytes": len(payload),
    }


def write_outputs(md_path: Path, ablation_path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
        "Feature set",
        "Model",
        "Status",
        "Train",
        "Test",
        "Classes",
        "Coverage",
        "Features",
        "Balanced acc",
        "Macro-F1",
        "ECE",
        "Brier",
        "Mean ms",
        "Limitation",
    ]
    table = [
        [
            r.get("feature_set"),
            r.get("model"),
            r.get("status"),
            r.get("train_rows"),
            r.get("test_rows"),
            r.get("classes", ""),
            r.get("class_coverage", ""),
            r.get("feature_count", ""),
            _fmt(r.get("balanced_accuracy")),
            _fmt(r.get("macro_f1")),
            _fmt(r.get("ece")),
            _fmt(r.get("brier")),
            _fmt(r.get("latency_mean_ms")),
            r.get("limitation", ""),
        ]
        for r in rows
    ]
    body = [
        "# Feature Set Comparison: 12 vs v2 vs 21",
        "",
        "Status: non-production ablation. Subset feature sets are not compared directly against full 9-class baseline unless class coverage matches.",
        "",
        "`optical_21_candidate_still` excludes temporal-only fields; its actual feature count is reported in the table and it must not be described as the full temporal 21-feature set.",
        "",
        markdown_table(headers, table),
    ]
    write_text(md_path, "\n".join(body) + "\n")
    write_text(
        ablation_path,
        "\n".join(
            [
                "# Feature 21 Ablation",
                "",
                "`optical_21_candidate` includes `temporal_brightness_std`, which is not zero-imputed for still-image rows. If rows are insufficient, this is a data coverage limitation, not a model result.",
                "`optical_21_candidate_still` is the still-compatible 21-derived candidate and reports its actual feature count separately.",
                "",
                markdown_table(headers, [row for row in table if str(row[0]).startswith("optical_21_candidate")]),
            ]
        )
        + "\n",
    )


def _fmt(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return "n/a"
    return "n/a" if math.isnan(x) else f"{x:.4f}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare optical_12/v2/21 feature sets on compatible labeled rows.")
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--feature-sets", nargs="+", default=["optical_12_baseline", "optical_v2_candidate", "optical_21_candidate"])
    p.add_argument("--models", nargs="+", default=["random_forest_100", "random_forest_200_current_config", "extra_trees", "hist_gradient_boosting", "mlp_small_32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--latency-repeats", type=int, default=100)
    p.add_argument("--out-md", type=Path, required=True)
    p.add_argument("--ablation-md", type=Path, default=Path("docs/tables/ml/feature_21_ablation.md"))
    p.add_argument("--out-csv", type=Path, default=Path("docs/tables/ml/feature_set_12_vs_v2_vs_21.csv"))
    p.add_argument("--artifact-dir", type=Path, default=Path("artifacts/ml/feature_set_12_vs_v2_vs_21"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_rows = read_jsonl(args.train)
    test_rows = read_jsonl(args.test)
    rows = [
        run_one(train_rows, test_rows, feature_set_name=fs, model_name=model, seed=args.seed, latency_repeats=args.latency_repeats)
        for fs in args.feature_sets
        for model in args.models
    ]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()})
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    (args.artifact_dir / "metrics.json").write_text(json.dumps(rows, indent=2, sort_keys=True, default=str), encoding="utf-8")
    (args.artifact_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "train": str(args.train),
                "test": str(args.test),
                "feature_sets": args.feature_sets,
                "models": args.models,
                "seed": args.seed,
                "latency_repeats": args.latency_repeats,
                "git_commit": git_commit_hash(),
                "git_branch": git_branch_name(),
                "metric_status": "subset/preliminary when class coverage is below 9/9; not live NIR/LWIR validated",
                "comparison_guardrail": "Do not compare subset feature_v2/21 results as same-condition evidence against full 9-class optical_12 benchmark.",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_outputs(args.out_md, args.ablation_md, rows)
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.ablation_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
