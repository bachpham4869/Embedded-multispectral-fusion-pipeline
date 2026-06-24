#!/usr/bin/env python3
"""Read-only frozen ENV classifier evaluation for Phase 2.

This script never fits, retrains, recalibrates, or mutates the production
model. The optional train JSONL is used only to identify optical_12 feature
overlaps and metadata limitations before filtering held-out test rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.train_classifier import apply_scalers
from smartbinocular.feature_schema import ENV_CLASS_TO_INT, ENV_INT_TO_CLASS, FeatureRecord
from tools.check_dataset_leakage import feature_vector_overlap_details


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_manifest(paths: Iterable[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        stat = path.stat()
        out.append({"path": str(path), "sha256": sha256_file(path), "bytes": stat.st_size})
    return out


def manifest_hash(manifest: list[dict[str, Any]]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True)
        return out.strip()
    except Exception:
        return "unknown"


def git_branch_name() -> str:
    try:
        out = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True)
        return out.strip() or "unknown"
    except Exception:
        return "unknown"


def package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import sklearn

        versions["sklearn"] = sklearn.__version__
    except Exception:
        versions["sklearn"] = "unavailable"
    try:
        import scipy

        versions["scipy"] = scipy.__version__
    except Exception:
        versions["scipy"] = "unavailable"
    versions["numpy"] = np.__version__
    return versions


def feature_overlap_test_indices(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> set[int]:
    return {int(detail["test_index"]) for detail in feature_vector_overlap_details(train_rows, test_rows)}


def rows_to_matrix(
    rows: list[dict[str, Any]],
    feature_set: list[str],
    *,
    exclude_indices: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    X: list[np.ndarray] = []
    y: list[int] = []
    groups: list[str] = []
    kept_indices: list[int] = []
    excluded = exclude_indices or set()
    for idx, row in enumerate(rows):
        if idx in excluded:
            continue
        label = row.get("label") or row.get("weak_label")
        if label not in ENV_CLASS_TO_INT:
            continue
        try:
            record = FeatureRecord.from_dict(row)
            vec = record.to_feature_array(feature_set)
        except (ValueError, TypeError):
            continue
        X.append(vec)
        y.append(ENV_CLASS_TO_INT[str(label)])
        groups.append(str(row.get("nir_channel", "rgb")))
        kept_indices.append(idx)
    if not X:
        raise ValueError("No compatible rows available for frozen evaluation.")
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32), np.asarray(groups), kept_indices


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> float:
    recalls = []
    for label in labels:
        mask = y_true == label
        if np.any(mask):
            recalls.append(float(np.mean(y_pred[mask] == label)))
    return float(np.mean(recalls)) if recalls else math.nan


def _normalized_confusion(cm: np.ndarray) -> list[list[float]]:
    cmf = cm.astype(float)
    row_sums = cmf.sum(axis=1, keepdims=True)
    return np.divide(cmf, row_sums, out=np.zeros_like(cmf), where=row_sums != 0).tolist()


def evaluate_bundle_on_rows(
    bundle: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    exclude_indices: set[int] | None,
    variant: str,
) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

    feature_set = list(bundle.get("feature_set") or [])
    X, y_true, groups, kept_indices = rows_to_matrix(rows, feature_set, exclude_indices=exclude_indices)
    X_scaled = apply_scalers(X, groups, bundle["scalers"], str(bundle.get("normalize_by", "nir_channel")))
    rf = bundle["rf"]
    y_pred = np.asarray(rf.predict(X_scaled), dtype=np.int32)
    proba = np.asarray(rf.predict_proba(X_scaled), dtype=float) if hasattr(rf, "predict_proba") else None
    labels = np.asarray(sorted(np.unique(np.concatenate([y_true, y_pred]))), dtype=np.int32)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = []
    for idx, label in enumerate(labels):
        per_class.append(
            {
                "class": ENV_INT_TO_CLASS.get(int(label), str(label)),
                "support": int(support[idx]),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
            }
        )
    return {
        "variant": variant,
        "metric_status": "preliminary / not thesis-ready",
        "test_rows_input": len(rows),
        "test_rows_evaluated": int(len(y_true)),
        "excluded_test_indices": sorted(exclude_indices or []),
        "kept_original_test_indices": kept_indices,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": _balanced_accuracy(y_true, y_pred, labels),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "mean_max_probability": float(np.max(proba, axis=1).mean()) if proba is not None else math.nan,
        "labels": [ENV_INT_TO_CLASS.get(int(label), str(label)) for label in labels],
        "per_class": per_class,
        "confusion_matrix_raw": cm.tolist(),
        "confusion_matrix_normalized": _normalized_confusion(cm),
    }


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out) + "\n"


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def write_summary(path: Path, original: dict[str, Any], no_overlap: dict[str, Any], provenance: dict[str, Any]) -> None:
    rows = [
        [
            original["variant"],
            original["test_rows_evaluated"],
            _fmt(original["accuracy"]),
            _fmt(original["balanced_accuracy"]),
            _fmt(original["macro_f1"]),
            _fmt(original["weighted_f1"]),
            original["metric_status"],
        ],
        [
            no_overlap["variant"],
            no_overlap["test_rows_evaluated"],
            _fmt(no_overlap["accuracy"]),
            _fmt(no_overlap["balanced_accuracy"]),
            _fmt(no_overlap["macro_f1"]),
            _fmt(no_overlap["weighted_f1"]),
            no_overlap["metric_status"],
        ],
    ]
    body = [
        "# Frozen Eval Overlap Sensitivity",
        "",
        "Status: preliminary / not thesis-ready. Production model is loaded read-only; no refit, retrain, or recalibration is performed.",
        "",
        "## Reproducibility",
        "",
        _md_table(["Field", "Value"], [[k, v] for k, v in provenance.items() if k not in {"dataset_manifest"}]),
        "## Dataset Manifest",
        "",
        _md_table(["Path", "SHA256", "Bytes"], [[m["path"], m["sha256"], m["bytes"]] for m in provenance["dataset_manifest"]]),
        "## Aggregate Metrics",
        "",
        _md_table(["Variant", "Rows", "Accuracy", "Balanced accuracy", "Macro-F1", "Weighted-F1", "Status"], rows),
        "## Interpretation",
        "",
        f"- Original held-out rows evaluated: {original['test_rows_evaluated']}.",
        f"- No-feature-overlap rows evaluated: {no_overlap['test_rows_evaluated']} after excluding {len(no_overlap['excluded_test_indices'])} test records.",
        "- If the metrics remain materially higher than sidecar CV, source overlap / split difficulty / missing image-level duplicate checks remain likely explanations.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read-only frozen classifier overlap sensitivity evaluation.")
    p.add_argument("--model", type=Path, required=True, help="Production joblib bundle to load read-only")
    p.add_argument("--train", type=Path, required=True, help="Train JSONL used only for feature-overlap detection")
    p.add_argument("--test", type=Path, required=True, help="Held-out test JSONL")
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/ml/eval"))
    p.add_argument("--summary", type=Path, default=Path("docs/tables/ml/frozen_eval_overlap_sensitivity.md"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for path in (args.model, args.train, args.test):
        if not path.is_file():
            print(f"ERROR: not found: {path}", file=sys.stderr)
            return 2
    import joblib  # type: ignore

    bundle = joblib.load(args.model)
    train_rows = read_jsonl(args.train)
    test_rows = read_jsonl(args.test)
    overlap_indices = feature_overlap_test_indices(train_rows, test_rows)
    original = evaluate_bundle_on_rows(bundle, test_rows, exclude_indices=set(), variant="original")
    no_overlap = evaluate_bundle_on_rows(
        bundle,
        test_rows,
        exclude_indices=overlap_indices,
        variant="exclude_test_feature_vector_overlaps",
    )
    model_manifest = dataset_manifest([args.model])
    dataset_files = dataset_manifest([args.train, args.test])
    provenance = {
        "command": " ".join([sys.executable, *sys.argv]),
        "git_commit": git_commit_hash(),
        "branch": git_branch_name(),
        "model_path": str(args.model),
        "model_sha256": model_manifest[0]["sha256"],
        "dataset_manifest_hash": manifest_hash(dataset_files),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "feature_set_version": f"optical_12_baseline ({len(bundle.get('feature_set') or [])} features)",
        "split_method": "existing frozen JSONL test file; sensitivity excludes feature-vector-overlap test rows",
        "random_seed": "not used",
        "python": package_versions()["python"],
        "sklearn": package_versions()["sklearn"],
        "numpy": package_versions()["numpy"],
        "scipy": package_versions()["scipy"],
        "metric_status": "preliminary / not thesis-ready",
        "dataset_manifest": dataset_files,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "frozen_eval_original.json").write_text(
        json.dumps({"provenance": provenance, "metrics": original}, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    (args.out_dir / "frozen_eval_no_feature_overlap.json").write_text(
        json.dumps({"provenance": provenance, "metrics": no_overlap}, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    write_summary(args.summary, original, no_overlap, provenance)
    print(f"Wrote {args.out_dir / 'frozen_eval_original.json'}")
    print(f"Wrote {args.out_dir / 'frozen_eval_no_feature_overlap.json'}")
    print(f"Wrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
