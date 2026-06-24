"""secondary_threshold_sweep.py — sweep *secondary* (top-2) gate τ₂ with fixed top-1 τ₁.

Uses the same calibrated bundle and test JSONL as ``threshold_sweep.py``. For each row,
takes the top-1 and top-2 class probabilities from ``predict_proba`` and calls
:func:`smartbinocular.env_presets.compose_env_from_ml_top2` with
``primary_threshold=τ₁`` and ``secondary_threshold=τ₂`` over a grid of τ₂.

**Metrics per τ₂** (on the full test set, same rows as the primary sweep):
  * ``n_ml_active`` — ``compose`` returned non-``None`` effective env (ML drives)
  * ``n_with_hint`` — second return (secondary hint) is not ``None``
  * ``hint_rate_of_ml`` — ``n_with_hint / n_ml_active`` (if ``n_ml_active > 0``)
  * ``p2_p50``, ``p2_p90`` — percentiles of top-2 probability, restricted to rows
    with ``p1 >= τ₁`` (distribution context for τ₂)

This does *not* optimize a single scalar “accuracy of hints” (no ground-truth
second label); it characterizes the **traded volume of secondary hints** vs τ₂
so τ₂ can be chosen together with the compositor rules. Pair with
``docs/tables/ml/threshold_sweep.csv`` and OOF ECE in the model sidecar.

Usage:
    .venv/bin/python tools/secondary_threshold_sweep.py
    .venv/bin/python tools/secondary_threshold_sweep.py --primary 0.62 --out docs/tables/ml/secondary_threshold_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _load_apply_scalers():
    path = ROOT / "models" / "train_classifier.py"
    spec = importlib.util.spec_from_file_location("sb_train_classifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.apply_scalers


def _load_load_dataset():
    path = ROOT / "tools" / "threshold_sweep.py"
    spec = importlib.util.spec_from_file_location("sb_threshold_sweep", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_dataset


from smartbinocular.env_presets import compose_env_from_ml_top2
from smartbinocular.feature_schema import (
    ENV_INT_TO_CLASS,
)

DEFAULT_BUNDLE = ROOT / "models" / "production" / "env_classifier.joblib"
DEFAULT_DATASET = ROOT / "data" / "training" / "from_logs_test.jsonl"
DEFAULT_OUT = ROOT / "docs" / "tables" / "ml" / "secondary_threshold_sweep.csv"

# Finer than primary grid — secondary is typically lower
TAU2_GRID = [
    0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.32, 0.35, 0.40
]


def _top2(
    proba_row: np.ndarray, classes: np.ndarray
) -> Tuple[Optional[str], float, Optional[str], float]:
    """Return (c1, p1, c2, p2) with string class names. ``classes`` = rf.classes_."""
    n = proba_row.shape[0]
    if n < 1:
        return None, 0.0, None, 0.0
    order = np.argsort(-proba_row)
    i0 = int(order[0])
    c1i = int(classes[i0])
    p1 = float(proba_row[i0])
    c1 = ENV_INT_TO_CLASS.get(c1i)
    if n < 2:
        return c1, p1, None, 0.0
    i1 = int(order[1])
    c2i = int(classes[i1])
    p2 = float(proba_row[i1])
    c2 = ENV_INT_TO_CLASS.get(c2i)
    return c1, p1, c2, p2


def run_secondary_sweep(
    proba: np.ndarray,
    classes: np.ndarray,
    primary_tau: float,
    tau2_list: List[float],
) -> Tuple[List[dict], dict]:
    n = proba.shape[0]
    rows: List[dict] = []

    p1_ge = []  # p1 for rows where p1 >= primary_tau (for p2 percentiles)
    p2_ge = []  # p2 on same mask
    for i in range(n):
        c1, p1, c2, p2 = _top2(proba[i], classes)
        if c1 is None:
            continue
        if p1 >= primary_tau and c2 is not None:
            p1_ge.append(p1)
            p2_ge.append(p2)
    p2_arr = np.array(p2_ge, dtype=np.float64) if p2_ge else np.array([], dtype=np.float64)
    dist = {
        "n_p1_ge_tau1": int(len(p2_arr)),
        "p2_p25": float(np.percentile(p2_arr, 25)) if len(p2_arr) else 0.0,
        "p2_p50": float(np.percentile(p2_arr, 50)) if len(p2_arr) else 0.0,
        "p2_p75": float(np.percentile(p2_arr, 75)) if len(p2_arr) else 0.0,
        "p2_p90": float(np.percentile(p2_arr, 90)) if len(p2_arr) else 0.0,
    }

    for tau2 in tau2_list:
        n_active = 0
        n_hint = 0
        for i in range(n):
            c1, p1, c2, p2 = _top2(proba[i], classes)
            if c1 is None:
                continue
            eff, hint = compose_env_from_ml_top2(
                class_1=c1,
                proba_1=p1,
                class_2=c2,
                proba_2=p2,
                primary_threshold=primary_tau,
                secondary_threshold=tau2,
            )
            if eff is not None:
                n_active += 1
            if hint is not None:
                n_hint += 1
        rate = n_hint / n_active if n_active else 0.0
        rows.append(
            {
                "primary_tau": primary_tau,
                "secondary_tau": tau2,
                "n_test_rows": n,
                "n_ml_active": n_active,
                "n_with_hint": n_hint,
                "hint_rate_of_ml": round(rate, 4),
            }
        )
    return rows, dist


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep secondary (top-2) threshold with compose_env_from_ml_top2"
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--primary",
        type=float,
        default=0.62,
        help="Fixed top-1 threshold τ₁ (default 0.62, align with config)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output CSV",
    )
    args = parser.parse_args()
    apply_scalers_fn = _load_apply_scalers()
    bundle = joblib.load(args.bundle)
    rf = bundle["rf"]
    scalers = bundle["scalers"]
    normalize_by = str(bundle.get("normalize_by", "nir_channel"))
    if not scalers:
        print("[error] No scalers in bundle", file=sys.stderr)
        sys.exit(1)
    cl = np.array(rf.classes_)

    load_dataset = _load_load_dataset()
    X, _y_true, nir_groups = load_dataset(args.dataset)
    Xs = apply_scalers_fn(X, nir_groups, scalers, normalize_by)
    proba = rf.predict_proba(Xs)

    rows, dist = run_secondary_sweep(proba, cl, float(args.primary), TAU2_GRID)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Written: {args.out}")
    print(f"τ₁ = {args.primary} — p2 distribution (rows with p1 ≥ τ₁):")
    for k, v in dist.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    h = f"{'τ2':>6}  {'n_act':>6}  {'n_hint':>6}  {'hint/ml':>8}"
    print(f"\n{h}")
    print("-" * len(h))
    for r in rows:
        print(
            f"{r['secondary_tau']:>6.2f}  {r['n_ml_active']:>6d}  {r['n_with_hint']:>6d}  {r['hint_rate_of_ml']:>8.4f}"
        )


if __name__ == "__main__":
    main()
