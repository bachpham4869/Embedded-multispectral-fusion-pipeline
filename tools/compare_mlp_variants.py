#!/usr/bin/env python3
"""Compare RF baselines against MLP variants with optional internal calibration."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.train_classifier import TRAINING_MODES, apply_scalers, fit_scalers, load_dataset
from tools.compare_classifiers import _brier_multiclass, _ece_multiclass, _metric_values, _per_class_rows, build_model
from tools.ml_metadata_utils import markdown_table, write_text


MLP_CONFIGS = {
    "mlp_32": (32,),
    "mlp_64": (64,),
    "mlp_128": (128,),
    "mlp_64_32": (64, 32),
    "mlp_128_64": (128, 64),
}


def internal_calibration_split(y: np.ndarray, *, seed: int, validation_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import StratifiedShuffleSplit

    idx = np.arange(len(y))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
    fit_idx, cal_idx = next(splitter.split(idx, y))
    return fit_idx, cal_idx


def build_candidate_model(name: str, seed: int) -> Any:
    from sklearn.neural_network import MLPClassifier

    if name in {"random_forest_100", "random_forest_200_current_config", "extra_trees", "hist_gradient_boosting"}:
        return build_model(name, seed)
    if name in MLP_CONFIGS:
        return MLPClassifier(hidden_layer_sizes=MLP_CONFIGS[name], max_iter=450, random_state=seed, early_stopping=True)
    raise ValueError(f"Unknown model: {name}")


def _predict_proba(model: Any, X: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not hasattr(model, "predict_proba"):
        return None, None
    try:
        return np.asarray(model.predict_proba(X), dtype=float), np.asarray(model.classes_)
    except Exception:
        return None, None


def _latency(model: Any, X: np.ndarray, repeats: int) -> float:
    sample = X[:1]
    vals = []
    for _ in range(max(0, repeats)):
        start = time.perf_counter()
        model.predict(sample)
        vals.append((time.perf_counter() - start) * 1000.0)
    return float(np.mean(vals)) if vals else math.nan


def fit_model(name: str, X_train: np.ndarray, y_train: np.ndarray, *, seed: int, calibrated: bool) -> tuple[Any, list[str], float]:
    captured: list[str] = []
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if calibrated:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.frozen import FrozenEstimator

            fit_idx, cal_idx = internal_calibration_split(y_train, seed=seed, validation_fraction=0.2)
            base = build_candidate_model(name, seed)
            base.fit(X_train[fit_idx], y_train[fit_idx])
            model = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
            model.fit(X_train[cal_idx], y_train[cal_idx])
        else:
            model = build_candidate_model(name, seed)
            model.fit(X_train, y_train)
    captured.extend(f"{w.category.__name__}: {w.message}" for w in caught)
    return model, captured, time.perf_counter() - start


def run_variant(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    latency_repeats: int,
    calibrated: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model, captured, train_time = fit_model(name, X_train, y_train, seed=seed, calibrated=calibrated)
    y_pred = np.asarray(model.predict(X_test), dtype=int)
    metrics = _metric_values(y_test, y_pred, labels)
    proba, proba_classes = _predict_proba(model, X_test)
    ece = _ece_multiclass(y_test, proba, proba_classes) if proba is not None and proba_classes is not None else math.nan
    brier = _brier_multiclass(y_test, proba, proba_classes) if proba is not None and proba_classes is not None else math.nan
    model_name = f"calibrated_{name}" if calibrated else name
    row = {
        "model": model_name,
        "base_model": name,
        "seed": seed,
        "calibration_status": "calibrated_internal_train_validation_split" if calibrated else "uncalibrated",
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "ece": ece,
        "brier": brier,
        "abstention_rate_tau1_0.62": _abstention(proba) if proba is not None else math.nan,
        "train_time_s": train_time,
        "latency_mean_ms": _latency(model, X_test, latency_repeats),
        "model_size_bytes": len(pickle.dumps(model)),
        "warnings": " | ".join(captured),
    }
    per_class = _per_class_rows(model_name, y_test, y_pred, labels, n_bootstrap=0, seed=seed)
    for r in per_class:
        r["seed"] = seed
        r["calibration_status"] = row["calibration_status"]
    return row, per_class


def _abstention(proba: np.ndarray, tau1: float = 0.62) -> float:
    return float(np.mean(np.max(proba, axis=1) < tau1))


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)
    out = []
    metric_keys = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "ece", "brier", "abstention_rate_tau1_0.62", "latency_mean_ms", "model_size_bytes"]
    for model, items in sorted(grouped.items()):
        agg = {"model": model, "seeds": ",".join(str(i["seed"]) for i in items), "runs": len(items), "calibration_status": items[0]["calibration_status"]}
        for key in metric_keys:
            vals = np.asarray([float(i[key]) for i in items], dtype=float)
            agg[f"{key}_mean"] = float(np.nanmean(vals))
            agg[f"{key}_std"] = float(np.nanstd(vals))
        out.append(agg)
    return out


def write_outputs(md_path: Path, rows: list[dict[str, Any]], aggregate: list[dict[str, Any]], per_class: list[dict[str, Any]]) -> None:
    table = [
        [
            r["model"],
            r["calibration_status"],
            r["seeds"],
            _fmt(r["balanced_accuracy_mean"]),
            _fmt(r["macro_f1_mean"]),
            _fmt(r["ece_mean"]),
            _fmt(r["brier_mean"]),
            _fmt(r["abstention_rate_tau1_0.62_mean"]),
            _fmt(r["latency_mean_ms_mean"]),
            _fmt(r["model_size_bytes_mean"], 0),
        ]
        for r in aggregate
    ]
    low_support = [
        [r["model"], r["class"], r["seed"], _fmt(r["f1"])]
        for r in per_class
        if r["class"] in {"glare", "backlight", "transition"}
    ]
    body = [
        "# MLP Variant Comparison",
        "",
        "Status: offline optical RGB-proxy baseline; not live NIR/LWIR validated. Calibrated rows use an internal train/validation split only, not the test set.",
        "",
        markdown_table(
            ["Model", "Calibration", "Seeds", "Balanced acc mean", "Macro-F1 mean", "ECE mean", "Brier mean", "Abstain mean", "Latency ms", "Bytes"],
            table,
        ),
        "",
        "## Low-Support Class F1",
        "",
        markdown_table(["Model", "Class", "Seed", "F1"], low_support),
    ]
    write_text(md_path, "\n".join(body) + "\n")


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        x = float(value)
    except Exception:
        return str(value)
    if math.isnan(x):
        return "n/a"
    return f"{x:.{digits}f}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()}) if rows else ["model"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare RF and MLP variants on optical_12_baseline.")
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--models", nargs="+", default=["random_forest_100", "random_forest_200_current_config", "extra_trees", "hist_gradient_boosting", "mlp_32", "mlp_64", "mlp_128", "mlp_64_32", "mlp_128_64"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--include-calibrated", nargs="*", default=["mlp_32", "mlp_64"])
    p.add_argument("--latency-repeats", type=int, default=100)
    p.add_argument("--bootstrap", type=int, default=100, help="Reserved for manifest compatibility; per-class CIs not bootstrapped here.")
    p.add_argument("--hardware-label", default="proxy benchmark")
    p.add_argument("--out-md", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, default=Path("docs/tables/ml/mlp_variant_comparison.csv"))
    p.add_argument("--artifact-dir", type=Path, default=Path("artifacts/ml/mlp_variant_comparison"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mode = TRAINING_MODES["optical_only"]
    X_train, y_train, g_train = load_dataset(args.train, mode)
    X_test, y_test, g_test = load_dataset(args.test, mode)
    scalers = fit_scalers(X_train, g_train, mode["normalize_by"])
    X_train = apply_scalers(X_train, g_train, scalers, mode["normalize_by"])
    X_test = apply_scalers(X_test, g_test, scalers, mode["normalize_by"])
    labels = np.asarray(sorted(np.unique(np.concatenate([y_train, y_test]))), dtype=int)
    rows: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    calibrated_set = set(args.include_calibrated or [])
    for seed in args.seeds:
        for name in args.models:
            row, cls = run_variant(name, X_train, y_train, X_test, y_test, labels, seed=seed, latency_repeats=args.latency_repeats, calibrated=False)
            rows.append(row)
            per_class.extend(cls)
            if name in calibrated_set and name.startswith("mlp_"):
                row, cls = run_variant(name, X_train, y_train, X_test, y_test, labels, seed=seed, latency_repeats=args.latency_repeats, calibrated=True)
                rows.append(row)
                per_class.extend(cls)
    aggregate = aggregate_rows(rows)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    (args.artifact_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "train": str(args.train),
                "test": str(args.test),
                "models": args.models,
                "seeds": args.seeds,
                "calibrated_models": sorted(calibrated_set),
                "calibration_policy": "internal train/validation split only; test set is never used for calibration",
                "hardware": args.hardware_label,
                "metric_status": "offline optical RGB-proxy baseline; not live NIR/LWIR validated",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_csv(args.out_csv, rows)
    _write_csv(args.artifact_dir / "per_class.csv", per_class)
    _write_csv(args.artifact_dir / "aggregate.csv", aggregate)
    write_outputs(args.out_md, rows, aggregate, per_class)
    print(f"Wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
