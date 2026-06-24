"""threshold_sweep.py — P3-A: sweep confidence thresholds and report per-night-class OVR F1.

Loads the Phase 1 calibrated bundle, runs predict_proba on the test JSONL,
and sweeps thresholds {0.50, 0.55, 0.60, 0.62, 0.65, 0.70, 0.75}.

For each threshold τ a row is *accepted* when max(predict_proba) >= τ.
OVR binary F1 for each night class is computed on accepted rows only,
using all other multiclass labels in the model as negatives.

Usage:
    python tools/threshold_sweep.py
    python tools/threshold_sweep.py --bundle models/production/env_classifier.joblib \
                                    --dataset data/training/from_logs_test.jsonl \
                                    --out docs/tables/ml/threshold_sweep.csv

See also: ``tools/secondary_threshold_sweep.py`` for a grid over τ₂ (top-2) with fixed τ₁
and the real :func:`compose_env_from_ml_top2` compositor.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _load_apply_scalers():
    """Load apply_scalers from models/train_classifier without requiring models/ to be a package."""
    path = ROOT / "models" / "train_classifier.py"
    spec = importlib.util.spec_from_file_location("sb_train_classifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.apply_scalers

from smartbinocular.feature_schema import (
    ENV_CLASS_TO_INT,
    FEATURE_SET_OPTICAL_ONLY,
    FeatureRecord,
)

THRESHOLDS = [0.50, 0.55, 0.60, 0.62, 0.65, 0.70, 0.75]

# INT labels for the three night classes (ENV_CLASS_TO_INT encoding)
NIGHT_CLEAR_INT = 1
NORMAL_NIGHT_INT = 2
NIR_NIGHT_INT = 9
NIGHT_INTS = [NIGHT_CLEAR_INT, NORMAL_NIGHT_INT, NIR_NIGHT_INT]
NIGHT_NAMES = ["night_clear", "normal_night", "nir_night"]

DEFAULT_BUNDLE = ROOT / "models" / "production" / "env_classifier.joblib"
DEFAULT_DATASET = ROOT / "data" / "training" / "from_logs_test.jsonl"
DEFAULT_OUT = ROOT / "docs" / "tables" / "ml" / "threshold_sweep.csv"


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X float32 [N,12], y_true int32 [N], nir_channel str [N]) from labeled JSONL.

    Per-channel scaling in apply_scalers requires ``nir_channel`` (C10) — same as
    :func:`models.train_classifier.evaluate_saved_model`.
    """
    xs, ys, groups = [], [], []
    skipped = 0
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = FeatureRecord.from_dict(json.loads(line))
            label_str = rec.effective_label()
            if label_str is None:
                skipped += 1
                continue
            if not rec.is_compatible_with(FEATURE_SET_OPTICAL_ONLY):
                skipped += 1
                continue
            label_int = ENV_CLASS_TO_INT.get(label_str)
            if label_int is None:
                skipped += 1
                continue
            xs.append(rec.to_feature_array(FEATURE_SET_OPTICAL_ONLY))
            ys.append(label_int)
            groups.append(str(rec.nir_channel))
    print(f"Loaded {len(xs)} records ({skipped} skipped) from {path.name}")
    if not xs:
        raise ValueError(f"No compatible labeled rows in {path}")
    return (
        np.array(xs, dtype=np.float32),
        np.array(ys, dtype=np.int32),
        np.array(groups, dtype=object),
    )


def _ovr_f1(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int) -> float:
    """OVR binary F1: positive = pos_label, negative = all other model labels."""
    tp = int(np.sum((y_pred == pos_label) & (y_true == pos_label)))
    fp = int(np.sum((y_pred == pos_label) & (y_true != pos_label)))
    fn = int(np.sum((y_pred != pos_label) & (y_true == pos_label)))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def run_sweep(
    rf,
    scalers: dict,
    normalize_by: str,
    apply_scalers_fn,
    classes: np.ndarray,
    X: np.ndarray,
    y_true: np.ndarray,
    nir_groups: np.ndarray,
) -> list[dict]:
    """Sweep thresholds; return list of result dicts."""
    X_scaled = apply_scalers_fn(X, nir_groups, scalers, normalize_by)
    proba = rf.predict_proba(X_scaled)  # [N, n_classes]

    # classes_ alignment: proba column j corresponds to classes[j]
    assert proba.shape[1] == len(classes), "predict_proba columns / classes_ mismatch"

    conf = proba.max(axis=1)
    pred_idx = proba.argmax(axis=1)
    y_pred_all = classes[pred_idx]  # map argmax → class int label

    rows = []
    n_total = len(y_true)
    for tau in THRESHOLDS:
        mask = conf >= tau
        n_accepted = int(mask.sum())
        abstention_rate = 1.0 - n_accepted / n_total

        if n_accepted == 0:
            f1s = [0.0] * len(NIGHT_INTS)
        else:
            y_t = y_true[mask]
            y_p = y_pred_all[mask]
            f1s = [_ovr_f1(y_t, y_p, pos) for pos in NIGHT_INTS]

        macro = float(np.mean(f1s))
        rows.append(
            {
                "threshold": tau,
                "abstention_rate": round(abstention_rate, 4),
                "n_accepted": n_accepted,
                "f1_night_clear": round(f1s[0], 4),
                "f1_normal_night": round(f1s[1], 4),
                "f1_nir_night": round(f1s[2], 4),
                "macro_f1_night": round(macro, 4),
            }
        )
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written: {out_path}")


def print_table(rows: list[dict]) -> None:
    header = f"{'τ':>6}  {'abstain':>8}  {'n_acc':>6}  {'F1_nc':>7}  {'F1_nn':>7}  {'F1_nir':>7}  {'macro':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['threshold']:>6.2f}  {r['abstention_rate']:>8.4f}  {r['n_accepted']:>6d}"
            f"  {r['f1_night_clear']:>7.4f}  {r['f1_normal_night']:>7.4f}"
            f"  {r['f1_nir_night']:>7.4f}  {r['macro_f1_night']:>7.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold sweep for night-class calibration")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    apply_scalers_fn = _load_apply_scalers()

    bundle = joblib.load(args.bundle)
    rf = bundle["rf"]
    scalers = bundle["scalers"]
    normalize_by = str(bundle.get("normalize_by", "nir_channel"))
    if not scalers:
        print("[error] No scalers dict in bundle", file=sys.stderr)
        sys.exit(1)
    classes = np.array(rf.classes_)

    X, y_true, nir_groups = load_dataset(args.dataset)
    rows = run_sweep(
        rf,
        scalers,
        normalize_by,
        apply_scalers_fn,
        classes,
        X,
        y_true,
        nir_groups,
    )
    print_table(rows)
    write_csv(rows, args.out)


if __name__ == "__main__":
    main()
