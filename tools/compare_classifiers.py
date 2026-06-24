#!/usr/bin/env python3
"""Compare same-feature classical classifiers for SmartBinocular ML Phase 2.

The direct benchmark scope is intentionally limited to handcrafted feature
models that consume the same optical_12_baseline feature vector. Image-input
TFLite/CNN candidates belong in a separate baseline because the input policy is
different.

This script does not write production models.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import platform
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.train_classifier import TRAINING_MODES, apply_scalers, fit_scalers, load_dataset
from smartbinocular.feature_schema import ENV_CLASSES, ENV_INT_TO_CLASS


DEFAULT_MODELS = [
    "logistic_regression",
    "linear_svm",
    "sgd_linear_svm",
    "gaussian_nb",
    "decision_tree",
    "random_forest_50",
    "random_forest_100",
    "random_forest_200_current_config",
    "random_forest_depth_8",
    "random_forest_depth_12",
    "random_forest_depth_none",
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
    "knn",
    "mlp_small_32",
    "mlp_small_64_32",
]

MODEL_ALIASES = {
    "logreg": "logistic_regression",
    "sgd_log": "sgd_linear_svm",
    "rf_50": "random_forest_50",
    "rf_100": "random_forest_100",
    "rf_200": "random_forest_200_current_config",
    "mlp_32": "mlp_small_32",
    "mlp_64_32": "mlp_small_64_32",
}


def canonical_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_manifest(paths: list[Path]) -> list[dict[str, Any]]:
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
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def git_branch_name() -> str:
    try:
        out = subprocess.check_output(
            ["git", "branch", "--show-current"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
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


def build_model(name: str, seed: int) -> Any:
    from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier

    name = canonical_model_name(name)
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    if name == "linear_svm":
        return LinearSVC(class_weight="balanced", random_state=seed, max_iter=5000)
    if name == "sgd_linear_svm":
        return SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            class_weight="balanced",
            random_state=seed,
            max_iter=2000,
            tol=1e-4,
        )
    if name == "gaussian_nb":
        return GaussianNB()
    if name == "decision_tree":
        return DecisionTreeClassifier(max_depth=20, min_samples_leaf=4, class_weight="balanced", random_state=seed)
    if name.startswith("random_forest_"):
        if name == "random_forest_200_current_config":
            n_estimators = 200
            max_depth = 20
        elif name == "random_forest_depth_8":
            n_estimators = 200
            max_depth = 8
        elif name == "random_forest_depth_12":
            n_estimators = 200
            max_depth = 12
        elif name == "random_forest_depth_none":
            n_estimators = 200
            max_depth = None
        else:
            n_estimators = int(name.rsplit("_", 1)[1])
            max_depth = 20
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=4,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=4,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        )
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=seed, max_iter=200, l2_regularization=0.01)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    if name == "mlp_small_32":
        return MLPClassifier(hidden_layer_sizes=(32,), max_iter=400, random_state=seed, early_stopping=True)
    if name == "mlp_small_64_32":
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=seed, early_stopping=True)
    raise ValueError(f"Unknown model name: {name}")


def model_config(name: str) -> dict[str, Any]:
    canonical = canonical_model_name(name)
    model = build_model(canonical, seed=42)
    params = getattr(model, "get_params", lambda deep=True: {})()
    return {"name": canonical, "estimator": type(model).__name__, "params": params}


def _balanced_accuracy_fixed(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> float:
    recalls: list[float] = []
    for label in labels:
        mask = y_true == label
        if not np.any(mask):
            continue
        recalls.append(float(np.mean(y_pred[mask] == label)))
    return float(np.mean(recalls)) if recalls else 0.0


def _metric_values(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": _balanced_accuracy_fixed(y_true, y_pred, labels),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    if n_bootstrap <= 0 or len(y_true) == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        vals.append(float(metric_func(y_true[idx], y_pred[idx], labels)))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def _ece_multiclass(y_true: np.ndarray, y_prob: np.ndarray, classes: np.ndarray, n_bins: int = 10) -> float:
    pred_col = np.argmax(y_prob, axis=1)
    conf = y_prob[np.arange(len(y_prob)), pred_col]
    pred_label = classes[pred_col]
    correct = (pred_label == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if not np.any(mask):
            continue
        ece += float(np.mean(mask)) * abs(float(np.mean(correct[mask])) - float(np.mean(conf[mask])))
    return float(ece)


def _brier_multiclass(y_true: np.ndarray, y_prob: np.ndarray, classes: np.ndarray) -> float:
    y_onehot = np.zeros_like(y_prob, dtype=float)
    class_to_col = {int(cls): idx for idx, cls in enumerate(classes)}
    for i, label in enumerate(y_true):
        col = class_to_col.get(int(label))
        if col is not None:
            y_onehot[i, col] = 1.0
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


def _model_size_and_hash(model: Any) -> tuple[int, str]:
    payload = pickle.dumps(model)
    return len(payload), hashlib.sha256(payload).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _persist_or_measure_model(model: Any, model_dir: Path, persist_models: bool) -> tuple[str, str, int, float]:
    if persist_models:
        try:
            import joblib  # type: ignore

            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "model.joblib"
            joblib.dump(model, model_path)
            size_bytes = model_path.stat().st_size
            model_sha = sha256_file(model_path)
            start = time.perf_counter()
            joblib.load(model_path)
            load_time_s = time.perf_counter() - start
            return str(model_path), model_sha, size_bytes, load_time_s
        except Exception:
            pass
    payload = pickle.dumps(model)
    start = time.perf_counter()
    pickle.loads(payload)
    load_time_s = time.perf_counter() - start
    return "in-memory benchmark model; not persisted", _sha256_bytes(payload), len(payload), load_time_s


def _latency_stats(model: Any, X_test: np.ndarray, repeats: int) -> dict[str, float | int]:
    if repeats <= 0 or len(X_test) == 0:
        return {"repeats": 0, "mean_ms": math.nan, "median_ms": math.nan, "p95_ms": math.nan}
    sample = X_test[:1]
    durations: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(sample)
        durations.append((time.perf_counter() - start) * 1000.0)
    return {
        "repeats": repeats,
        "mean_ms": float(statistics.fmean(durations)),
        "median_ms": float(statistics.median(durations)),
        "p95_ms": float(np.percentile(np.asarray(durations), 95)),
    }


def _safe_predict_proba(model: Any, X_test: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not hasattr(model, "predict_proba"):
        return None, None
    try:
        proba = model.predict_proba(X_test)
        classes = np.asarray(model.classes_)
        return np.asarray(proba, dtype=float), classes
    except Exception:
        return None, None


def _per_class_rows(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    rows: list[dict[str, Any]] = []
    for idx, label in enumerate(labels):
        label_int = int(label)

        def class_f1(a: np.ndarray, b: np.ndarray, labs: np.ndarray, cls: int = label_int) -> float:
            from sklearn.metrics import f1_score

            return float(f1_score(a == cls, b == cls, zero_division=0))

        ci = bootstrap_ci(y_true, y_pred, labels, class_f1, n_bootstrap=n_bootstrap, seed=seed + label_int)
        rows.append(
            {
                "model": model_name,
                "class": ENV_INT_TO_CLASS.get(label_int, str(label_int)),
                "support": int(support[idx]),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "f1_ci_low": ci[0],
                "f1_ci_high": ci[1],
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out) + "\n"


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(x):
        return "n/a"
    return f"{x:.{digits}f}"


def _write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    per_class_rows: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> None:
    metric_rows = [
        [
            row["model"],
            _fmt(row["accuracy"]),
            f"{_fmt(row['accuracy_ci_low'])}-{_fmt(row['accuracy_ci_high'])}",
            _fmt(row["balanced_accuracy"]),
            f"{_fmt(row['balanced_accuracy_ci_low'])}-{_fmt(row['balanced_accuracy_ci_high'])}",
            _fmt(row["macro_f1"]),
            _fmt(row["weighted_f1"]),
            _fmt(row["ece"]),
            _fmt(row["brier"]),
            row["model_size_bytes"],
            _fmt(row["train_time_s"], 3),
            _fmt(row["load_time_s"], 4),
            _fmt(row["latency_mean_ms"], 3),
            _fmt(row["latency_median_ms"], 3),
            _fmt(row["latency_p95_ms"], 3),
            row["warnings_count"],
            row["metric_status"],
        ]
        for row in rows
    ]
    model_provenance_rows = [
        [row["model"], row["model_path"], row["model_sha256"], row["model_size_bytes"]]
        for row in rows
    ]
    support_rows = [
        [
            row["model"],
            row["class"],
            row["support"],
            _fmt(row["precision"]),
            _fmt(row["recall"]),
            _fmt(row["f1"]),
            f"{_fmt(row['f1_ci_low'])}-{_fmt(row['f1_ci_high'])}",
        ]
        for row in per_class_rows
    ]
    manifest_rows = [
        [split, m["path"], m["sha256"], m["bytes"]]
        for split, manifest in (("train", provenance["train_manifest"]), ("test", provenance["test_manifest"]))
        for m in manifest
    ]
    body = [
        "# Model Comparison: optical_12_baseline",
        "",
        "Scope: same-feature classical ML baselines only. These rows are directly comparable because they use the same train/test JSONL files and the same 12 handcrafted optical features.",
        "",
        "Latency status: proxy benchmark unless hardware is Raspberry Pi 4 CPU. Do not claim final deployment latency from non-RPi runs.",
        "",
        "## Reproducibility",
        "",
        _md_table(
            ["Field", "Value"],
            [
                ["command", provenance["command"]],
                ["git_commit", provenance["git_commit"]],
                ["branch", provenance.get("branch", "unknown")],
                ["feature_set_version", provenance["feature_set_version"]],
                ["split_method", provenance["split_method"]],
                ["random_seed", provenance["seed"]],
                ["train_rows", provenance["train_rows"]],
                ["test_rows", provenance["test_rows"]],
                ["dataset_manifest_hash", provenance["manifest_hash"]],
                ["python", provenance["versions"]["python"]],
                ["sklearn", provenance["versions"]["sklearn"]],
                ["numpy", provenance["versions"]["numpy"]],
                ["scipy", provenance["versions"]["scipy"]],
                ["hardware", provenance["hardware"]],
                ["latency_repeats", provenance["latency_repeats"]],
                ["run_status", provenance["run_status"]],
            ],
        ),
        "## Dataset Files",
        "",
        _md_table(["Split", "Path", "SHA256", "Bytes"], manifest_rows),
        "## Aggregate Metrics",
        "",
        _md_table(
            [
                "Model",
                "Accuracy",
                "Acc 95% CI",
                "Balanced Acc",
                "Bal Acc 95% CI",
                "Macro-F1",
                "Weighted-F1",
                "ECE",
                "Brier",
                "Model Bytes",
                "Train s",
                "Load s",
                "Mean ms",
                "Median ms",
                "P95 ms",
                "Warnings",
                "Status",
            ],
            metric_rows,
        ),
        "## Model Provenance",
        "",
        _md_table(["Model", "Model Path", "Model SHA256", "Serialized Bytes"], model_provenance_rows),
        "## Per-Class Metrics",
        "",
        _md_table(["Model", "Class", "Support", "Precision", "Recall", "F1", "F1 95% CI"], support_rows),
        "## Confusion Matrices",
        "",
        "Raw and normalized confusion matrices are written as CSV/PNG artifacts per model when this script runs.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body), encoding="utf-8")


def _write_confusion_artifacts(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray,
    out_dir: Path,
    fig_dir: Path,
) -> None:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, out=np.zeros_like(cm_norm), where=row_sums != 0)
    class_names = [ENV_INT_TO_CLASS.get(int(label), str(label)) for label in labels]

    model_dir = out_dir / model_name
    for suffix, matrix in (("raw", cm), ("normalized", cm_norm)):
        path = model_dir / f"confusion_{suffix}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["true\\pred", *class_names])
            for cls, row in zip(class_names, matrix):
                writer.writerow([cls, *row.tolist()])

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Normalized confusion: {model_name}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"confusion_{model_name}.png", dpi=160)
    plt.close(fig)


def run_classical_benchmark(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    labels: np.ndarray,
    models: list[str],
    seed: int,
    bootstrap: int,
    latency_repeats: int,
    artifact_dir: Path,
    fig_dir: Path,
    hardware_label: str,
    persist_models: bool,
    run_status: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    from sklearn.metrics import classification_report

    rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    canonical_models = [canonical_model_name(name) for name in models]
    model_configs: dict[str, Any] = {}
    for model_name in canonical_models:
        print(f"Running {model_name}...")
        model_configs[model_name] = model_config(model_name)
        model = build_model(model_name, seed)
        captured_warnings: list[str] = []
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X_train, y_train)
            y_pred = np.asarray(model.predict(X_test), dtype=int)
        train_time = time.perf_counter() - start
        captured_warnings.extend(f"{w.category.__name__}: {w.message}" for w in caught)

        metrics = _metric_values(y_test, y_pred, labels)
        proba, proba_classes = _safe_predict_proba(model, X_test)
        ece = math.nan
        brier = math.nan
        if proba is not None and proba_classes is not None:
            ece = _ece_multiclass(y_test, proba, proba_classes)
            brier = _brier_multiclass(y_test, proba, proba_classes)

        ci_acc = bootstrap_ci(
            y_test,
            y_pred,
            labels,
            lambda a, b, labs: _metric_values(a, b, labs)["accuracy"],
            n_bootstrap=bootstrap,
            seed=seed,
        )
        ci_bal = bootstrap_ci(
            y_test,
            y_pred,
            labels,
            lambda a, b, labs: _metric_values(a, b, labs)["balanced_accuracy"],
            n_bootstrap=bootstrap,
            seed=seed + 1,
        )
        ci_macro = bootstrap_ci(
            y_test,
            y_pred,
            labels,
            lambda a, b, labs: _metric_values(a, b, labs)["macro_f1"],
            n_bootstrap=bootstrap,
            seed=seed + 2,
        )
        ci_weighted = bootstrap_ci(
            y_test,
            y_pred,
            labels,
            lambda a, b, labs: _metric_values(a, b, labs)["weighted_f1"],
            n_bootstrap=bootstrap,
            seed=seed + 3,
        )
        model_dir = artifact_dir / model_name
        model_path, model_sha, size_bytes, load_time_s = _persist_or_measure_model(model, model_dir, persist_models)
        latency = _latency_stats(model, X_test, latency_repeats)
        metric_status = run_status
        row = {
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "accuracy_ci_low": ci_acc[0],
            "accuracy_ci_high": ci_acc[1],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "balanced_accuracy_ci_low": ci_bal[0],
            "balanced_accuracy_ci_high": ci_bal[1],
            "macro_f1": metrics["macro_f1"],
            "macro_f1_ci_low": ci_macro[0],
            "macro_f1_ci_high": ci_macro[1],
            "weighted_f1": metrics["weighted_f1"],
            "weighted_f1_ci_low": ci_weighted[0],
            "weighted_f1_ci_high": ci_weighted[1],
            "ece": ece,
            "brier": brier,
            "model_path": model_path,
            "model_sha256": model_sha,
            "model_size_bytes": size_bytes,
            "model_size_mb": size_bytes / (1024 * 1024),
            "train_time_s": train_time,
            "load_time_s": load_time_s,
            "latency_mean_ms": latency["mean_ms"],
            "latency_median_ms": latency["median_ms"],
            "latency_p95_ms": latency["p95_ms"],
            "latency_repeats": latency["repeats"],
            "hardware": hardware_label,
            "warnings_count": len(captured_warnings),
            "warnings": " | ".join(captured_warnings),
            "metric_status": metric_status,
        }
        rows.append(row)
        model_per_class = _per_class_rows(model_name, y_test, y_pred, labels, n_bootstrap=bootstrap, seed=seed)
        per_class_rows.extend(model_per_class)
        _write_confusion_artifacts(model_name, y_test, y_pred, labels, artifact_dir, fig_dir)
        report_text = classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=[ENV_INT_TO_CLASS.get(int(i), str(i)) for i in labels],
            digits=4,
            zero_division=0,
        )
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
        _write_json(
            model_dir / "metrics.json",
            {
                "aggregate": row,
                "per_class": model_per_class,
                "warnings": captured_warnings,
                "model_config": model_configs[model_name],
                "status": metric_status,
            },
        )

    run_manifest = dict(provenance)
    run_manifest.update(
        {
            "models": model_configs,
            "persist_models": persist_models,
            "run_status": run_status,
            "artifact_note": "Benchmark models are research artifacts only; do not stage/commit model.joblib files without explicit confirmation.",
        }
    )
    _write_json(artifact_dir / "run_manifest.json", run_manifest)
    return {"rows": rows, "per_class_rows": per_class_rows, "run_manifest": run_manifest}


def _write_ci_markdown(path: Path, rows: list[dict[str, Any]], per_class_rows: list[dict[str, Any]]) -> None:
    aggregate = [
        [
            row["model"],
            f"{_fmt(row['accuracy_ci_low'])}-{_fmt(row['accuracy_ci_high'])}",
            f"{_fmt(row['balanced_accuracy_ci_low'])}-{_fmt(row['balanced_accuracy_ci_high'])}",
            f"{_fmt(row['macro_f1_ci_low'])}-{_fmt(row['macro_f1_ci_high'])}",
            f"{_fmt(row['weighted_f1_ci_low'])}-{_fmt(row['weighted_f1_ci_high'])}",
        ]
        for row in rows
    ]
    low_support = [
        [
            row["model"],
            row["class"],
            row["support"],
            f"{_fmt(row['f1_ci_low'])}-{_fmt(row['f1_ci_high'])}",
        ]
        for row in per_class_rows
        if row["class"] in {"glare", "backlight", "transition"}
    ]
    body = [
        "# Model Comparison CI Summary",
        "",
        "Status: preliminary / not thesis-ready until leakage and metric discrepancy are resolved.",
        "",
        "## Aggregate 95% Bootstrap CI",
        "",
        _md_table(["Model", "Accuracy", "Balanced accuracy", "Macro-F1", "Weighted-F1"], aggregate),
        "## Low-Support Class F1 95% Bootstrap CI",
        "",
        _md_table(["Model", "Class", "Support", "F1 CI"], low_support),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark same-feature classical classifiers on SmartBinocular optical_12 JSONL."
    )
    p.add_argument("--train", type=Path, required=True, help="Training JSONL")
    p.add_argument("--test", type=Path, required=True, help="Frozen held-out test JSONL")
    p.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"Model names to run. Defaults: {', '.join(DEFAULT_MODELS)}",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--bootstrap", type=int, default=200, help="Bootstrap resamples for 95%% CI")
    p.add_argument("--latency-repeats", type=int, default=200, help="Single-sample latency repeats")
    p.add_argument("--out-csv", type=Path, default=Path("docs/tables/ml/model_comparison_12features.csv"))
    p.add_argument("--out-md", type=Path, default=Path("docs/tables/ml/model_comparison_12features.md"))
    p.add_argument("--out-ci-md", type=Path, default=Path("docs/tables/ml/model_comparison_12features_ci.md"))
    p.add_argument("--per-class-csv", type=Path, default=Path("docs/tables/ml/model_comparison_12features_per_class.csv"))
    p.add_argument("--artifact-dir", type=Path, default=Path("artifacts/ml/model_comparison_12features"))
    p.add_argument("--fig-dir", type=Path, default=Path("docs/figures/ml"))
    p.add_argument("--persist-models", action="store_true", help="Persist benchmark models under --artifact-dir/<model>/model.joblib only")
    p.add_argument("--quick", action="store_true", help="Mark run as quick/preliminary; use reduced command values explicitly")
    p.add_argument(
        "--split-method-label",
        default="existing frozen JSONL train/test files; no tuning on test",
        help="Human-readable split policy recorded in benchmark provenance.",
    )
    p.add_argument(
        "--metric-status",
        default=None,
        help="Override metric status string recorded in benchmark provenance and output tables.",
    )
    p.add_argument(
        "--hardware-label",
        default=platform.platform(),
        help="Hardware/environment label for latency provenance. Use Raspberry Pi 4 CPU only when run there.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.train.is_file() or not args.test.is_file():
        print("ERROR: --train and --test must point to JSONL files.", file=sys.stderr)
        return 2

    mode_cfg = TRAINING_MODES["optical_only"]
    print(f"Loading train: {args.train}")
    X_train, y_train, g_train = load_dataset(args.train, mode_cfg)
    print(f"Loading test: {args.test}")
    X_test, y_test, g_test = load_dataset(args.test, mode_cfg)
    scalers = fit_scalers(X_train, g_train, mode_cfg["normalize_by"])
    X_train_scaled = apply_scalers(X_train, g_train, scalers, mode_cfg["normalize_by"])
    X_test_scaled = apply_scalers(X_test, g_test, scalers, mode_cfg["normalize_by"])
    labels = np.asarray(sorted(np.unique(np.concatenate([y_train, y_test]))), dtype=int)

    train_manifest = dataset_manifest([args.train])
    test_manifest = dataset_manifest([args.test])
    run_status = "preliminary / not thesis-ready"
    if args.quick or args.bootstrap <= 100 or args.latency_repeats <= 100:
        run_status = "quick / preliminary / not thesis-ready"
    if args.metric_status:
        run_status = args.metric_status
    provenance = {
        "command": " ".join([sys.executable, *sys.argv]),
        "git_commit": git_commit_hash(),
        "branch": git_branch_name(),
        "feature_set_version": f"optical_12_baseline ({len(mode_cfg['feature_set'])} features)",
        "split_method": args.split_method_label,
        "seed": args.seed,
        "train_rows": len(y_train),
        "test_rows": len(y_test),
        "train_manifest": train_manifest,
        "test_manifest": test_manifest,
        "manifest_hash": manifest_hash([*train_manifest, *test_manifest]),
        "versions": package_versions(),
        "hardware": args.hardware_label,
        "latency_repeats": args.latency_repeats,
        "run_status": run_status,
    }
    result = run_classical_benchmark(
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        labels=labels,
        models=args.models,
        seed=args.seed,
        bootstrap=args.bootstrap,
        latency_repeats=args.latency_repeats,
        artifact_dir=args.artifact_dir,
        fig_dir=args.fig_dir,
        hardware_label=args.hardware_label,
        persist_models=args.persist_models,
        run_status=run_status,
        provenance=provenance,
    )
    rows = result["rows"]
    per_class_rows = result["per_class_rows"]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_csv, rows)
    _write_csv(args.per_class_csv, per_class_rows)
    _write_markdown(args.out_md, rows, per_class_rows, provenance)
    _write_ci_markdown(args.out_ci_md, rows, per_class_rows)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.per_class_csv}")
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_ci_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
