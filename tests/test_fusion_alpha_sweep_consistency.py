"""Consistency tests for fusion alpha sweep artifacts.

Validates:
- fusion_alpha_sweep_proxy.csv: schema, alpha grid, IQA cols finite
- per_class_fusion_alpha.csv: best_alpha is within the swept range, proxy_note present
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SWEEP_CSV = ROOT / "docs/thesis_eval/fusion/tables/fusion_alpha_sweep_proxy.csv"
PER_CLASS_CSV = ROOT / "docs/thesis_eval/fusion/tables/per_class_fusion_alpha.csv"

EXPECTED_ALPHA_GRID = {0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80}
REQUIRED_SWEEP_COLS = {
    "frame", "env_class", "alpha",
    "raw_log_rms", "nir_log_rms", "fused_log_rms",
    "fused_pct_sat", "proxy_note",
}
REQUIRED_PER_CLASS_COLS = {"env_class", "best_alpha", "best_fused_log_rms", "proxy_note"}
IQA_FLOAT_COLS = ("raw_log_rms", "nir_log_rms", "fused_log_rms", "fused_pct_sat")


@pytest.fixture(scope="module")
def sweep_rows():
    if not SWEEP_CSV.exists():
        pytest.skip(f"fusion alpha sweep CSV not found: {SWEEP_CSV}")
    with SWEEP_CSV.open(newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def per_class_rows():
    if not PER_CLASS_CSV.exists():
        pytest.skip(f"per-class alpha CSV not found: {PER_CLASS_CSV}")
    with PER_CLASS_CSV.open(newline="") as fh:
        return list(csv.DictReader(fh))


def test_sweep_csv_schema(sweep_rows):
    assert sweep_rows, "fusion alpha sweep CSV is empty"
    cols = set(sweep_rows[0].keys())
    missing = REQUIRED_SWEEP_COLS - cols
    assert not missing, f"Missing columns in sweep CSV: {missing}"


def test_sweep_alpha_grid_complete(sweep_rows):
    found = {round(float(r["alpha"]), 2) for r in sweep_rows}
    missing = EXPECTED_ALPHA_GRID - found
    assert not missing, f"Alpha values missing from sweep: {missing}"


def test_sweep_iqa_cols_finite(sweep_rows):
    bad = []
    for i, row in enumerate(sweep_rows):
        for col in IQA_FLOAT_COLS:
            v = float(row[col])
            if not math.isfinite(v):
                bad.append((i, col, v))
    assert not bad, f"Non-finite IQA values at rows: {bad[:5]}"


def test_sweep_proxy_note_present(sweep_rows):
    missing = [r["frame"] for r in sweep_rows if not r.get("proxy_note")]
    assert not missing, f"proxy_note absent for {len(missing)} rows; first: {missing[:3]}"


def test_per_class_schema(per_class_rows):
    assert per_class_rows, "per-class alpha CSV is empty"
    cols = set(per_class_rows[0].keys())
    missing = REQUIRED_PER_CLASS_COLS - cols
    assert not missing, f"Missing columns in per-class CSV: {missing}"


def test_per_class_best_alpha_within_grid(per_class_rows):
    all_swept = EXPECTED_ALPHA_GRID
    bad = [
        (r["env_class"], r["best_alpha"])
        for r in per_class_rows
        if round(float(r["best_alpha"]), 2) not in all_swept
    ]
    assert not bad, f"best_alpha outside swept grid: {bad}"


def test_per_class_best_log_rms_finite(per_class_rows):
    bad = [r["env_class"] for r in per_class_rows
           if not math.isfinite(float(r["best_fused_log_rms"]))]
    assert not bad, f"Non-finite best_fused_log_rms for: {bad}"
