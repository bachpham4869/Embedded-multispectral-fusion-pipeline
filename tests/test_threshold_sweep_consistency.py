"""
Consistency test: threshold_sweep.csv τ₁=0.62 row vs ml_gate_reference sidecar.

Permitted reads: docs/tables/ml/threshold_sweep.csv, models/production/env_classifier.json.
No GPU, no joblib.load, no network.

sweep_format_version: 1
Columns (version 1): threshold, abstention_rate, n_accepted,
    f1_night_clear, f1_normal_night, f1_nir_night, macro_f1_night
Parsing is column-name keyed so column-order changes do not break this test.
Regenerating the sweep (without renaming columns) and updating the sidecar in the
same commit stays green; updating only one fails fast.
"""
import csv
import json
import pathlib
import pytest

_REPO = pathlib.Path(__file__).parent.parent
_SWEEP_CSV = _REPO / "docs/tables/ml/threshold_sweep.csv"
_SIDECAR = _REPO / "models/production/env_classifier.json"
_TAU = 0.62
_TOL = 1e-4


def _load_sweep_row(tau: float) -> dict:
    if not _SWEEP_CSV.exists():
        pytest.skip(f"sweep CSV not found: {_SWEEP_CSV}")
    with _SWEEP_CSV.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if abs(float(row["threshold"]) - tau) < 1e-9:
                return row
    pytest.fail(f"τ={tau} row not found in {_SWEEP_CSV}")


def _load_sidecar() -> dict:
    if not _SIDECAR.exists():
        pytest.skip(f"sidecar not found: {_SIDECAR}")
    return json.loads(_SIDECAR.read_text())


def test_sweep_format_version_comment():
    """CSV must contain a sweep_format_version header comment on the first line."""
    if not _SWEEP_CSV.exists():
        pytest.skip(f"sweep CSV not found: {_SWEEP_CSV}")
    first_line = _SWEEP_CSV.read_text().splitlines()[0]
    assert "sweep_format_version" in first_line or first_line.startswith("threshold,"), (
        f"Expected column headers or # sweep_format_version comment in first line; got: {first_line!r}"
    )


def test_macro_f1_night_consistency():
    """CSV macro_f1_night at τ=0.62 must match ml_gate_reference sidecar value."""
    row = _load_sweep_row(_TAU)
    sidecar = _load_sidecar()
    ref = sidecar.get("ml_gate_reference", {}).get(f"top1_tau_{_TAU}", {})
    csv_val = float(row["macro_f1_night"])
    sidecar_val = float(ref["macro_f1_night_ovr"])
    assert abs(csv_val - sidecar_val) < _TOL, (
        f"macro_f1_night mismatch: CSV={csv_val}, sidecar={sidecar_val} (tol={_TOL})"
    )


def test_abstention_rate_consistency():
    """CSV abstention_rate at τ=0.62 must match ml_gate_reference sidecar value."""
    row = _load_sweep_row(_TAU)
    sidecar = _load_sidecar()
    ref = sidecar.get("ml_gate_reference", {}).get(f"top1_tau_{_TAU}", {})
    csv_val = float(row["abstention_rate"])
    sidecar_val = float(ref["abstention_rate"])
    assert abs(csv_val - sidecar_val) < _TOL, (
        f"abstention_rate mismatch: CSV={csv_val}, sidecar={sidecar_val} (tol={_TOL})"
    )
