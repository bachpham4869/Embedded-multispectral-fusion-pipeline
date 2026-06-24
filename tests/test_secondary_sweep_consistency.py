"""
Consistency test: secondary_threshold_sweep.csv τ₂=0.20 row vs ml_gate_reference sidecar.

Permitted reads: docs/tables/ml/secondary_threshold_sweep.csv,
                 models/production/env_classifier.json.
No GPU, no joblib.load, no network.

sweep_format_version: 1
Columns (version 1): primary_tau, secondary_tau, n_test_rows, n_ml_active,
    n_with_hint, hint_rate_of_ml
Parsing is column-name keyed so column-order changes do not break this test.
Regenerating the sweep and updating the sidecar in the same commit stays green;
updating only one fails fast.
"""
import csv
import json
import pathlib
import pytest

_REPO = pathlib.Path(__file__).parent.parent
_SWEEP_CSV = _REPO / "docs/tables/ml/secondary_threshold_sweep.csv"
_SIDECAR = _REPO / "models/production/env_classifier.json"
_TAU1 = 0.62
_TAU2 = 0.20
_TOL = 1e-4


def _load_sweep_row(tau1: float, tau2: float) -> dict:
    if not _SWEEP_CSV.exists():
        pytest.skip(f"secondary sweep CSV not found: {_SWEEP_CSV}")
    with _SWEEP_CSV.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if (
                abs(float(row["primary_tau"]) - tau1) < 1e-9
                and abs(float(row["secondary_tau"]) - tau2) < 1e-9
            ):
                return row
    pytest.fail(f"(τ₁={tau1}, τ₂={tau2}) row not found in {_SWEEP_CSV}")


def _load_sidecar() -> dict:
    if not _SIDECAR.exists():
        pytest.skip(f"sidecar not found: {_SIDECAR}")
    return json.loads(_SIDECAR.read_text())


def test_hint_rate_consistency():
    """CSV hint_rate_of_ml at τ₁=0.62, τ₂=0.20 must match sidecar ml_gate_reference."""
    row = _load_sweep_row(_TAU1, _TAU2)
    sidecar = _load_sidecar()
    ref = sidecar.get("ml_gate_reference", {}).get("top2_tau_0.20", {})
    csv_val = float(row["hint_rate_of_ml"])
    sidecar_val = float(ref["hint_rate_of_ml_active"])
    assert abs(csv_val - sidecar_val) < _TOL, (
        f"hint_rate_of_ml mismatch: CSV={csv_val}, sidecar={sidecar_val} (tol={_TOL})"
    )
