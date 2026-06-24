"""Consistency tests for the 3DNR alpha sweep artifact.

Validates:
- 3dnr_alpha_sweep.csv: schema, alpha range [0.40, 0.85], production default row present
- Skipped gracefully when thermal_seq data is absent (THESIS_SKIP path)
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SWEEP_CSV = ROOT / "docs/thesis_eval/thermal/tables/3dnr_alpha_sweep.csv"

PRODUCTION_DEFAULT_ALPHA = 0.65
ALPHA_MIN = 0.40
ALPHA_MAX = 0.85
REQUIRED_COLS = {
    "seq_file", "n_frames", "alpha", "is_production_default",
    "raw_residual_noise", "smoothed_residual_noise",
    "noise_reduction_pct", "psnr_vs_first_frame_db",
}
FINITE_FLOAT_COLS = (
    "raw_residual_noise", "smoothed_residual_noise",
    "noise_reduction_pct", "psnr_vs_first_frame_db",
)


@pytest.fixture(scope="module")
def sweep_rows():
    if not SWEEP_CSV.exists():
        pytest.skip(f"3DNR alpha sweep CSV not found: {SWEEP_CSV} — run after RPi4 thermal captures")
    with SWEEP_CSV.open(newline="") as fh:
        return list(csv.DictReader(fh))


def test_sweep_not_empty(sweep_rows):
    assert sweep_rows, "3DNR alpha sweep CSV is empty"


def test_sweep_schema(sweep_rows):
    cols = set(sweep_rows[0].keys())
    missing = REQUIRED_COLS - cols
    assert not missing, f"Missing columns: {missing}"


def test_alpha_range(sweep_rows):
    bad = [(r["seq_file"], float(r["alpha"]))
           for r in sweep_rows
           if not (ALPHA_MIN <= float(r["alpha"]) <= ALPHA_MAX + 1e-9)]
    assert not bad, f"Alpha values outside [{ALPHA_MIN}, {ALPHA_MAX}]: {bad[:5]}"


def test_production_default_row_present(sweep_rows):
    has_default = any(
        abs(float(r["alpha"]) - PRODUCTION_DEFAULT_ALPHA) < 1e-9
        for r in sweep_rows
    )
    assert has_default, f"Production default alpha={PRODUCTION_DEFAULT_ALPHA} not found in sweep"


def test_production_default_flagged(sweep_rows):
    for r in sweep_rows:
        if abs(float(r["alpha"]) - PRODUCTION_DEFAULT_ALPHA) < 1e-9:
            assert r["is_production_default"].lower() in ("true", "1"), (
                f"Row with alpha={r['alpha']} should have is_production_default=True"
            )


def test_float_cols_finite(sweep_rows):
    bad = []
    for i, row in enumerate(sweep_rows):
        for col in FINITE_FLOAT_COLS:
            v = float(row[col])
            if not math.isfinite(v):
                bad.append((i, col, v))
    assert not bad, f"Non-finite float values: {bad[:5]}"


def test_noise_reduction_pct_reasonable(sweep_rows):
    """noise_reduction_pct should be within [-100, 100] (percentage)."""
    bad = [(r["seq_file"], r["alpha"], r["noise_reduction_pct"])
           for r in sweep_rows
           if not (-100 <= float(r["noise_reduction_pct"]) <= 100)]
    assert not bad, f"noise_reduction_pct out of [-100, 100]: {bad[:5]}"
