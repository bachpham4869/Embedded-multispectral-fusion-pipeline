"""Consistency tests for the rain temporal-median N sweep.

Validates:
- rain_median_n_sweep.csv: schema, N values present, IQA cols finite
- Deliberately does NOT assert strict latency monotonicity — proc_ms reflects
  RainTemporalMedian buffer fill which may not scale linearly with N.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SWEEP_CSV = ROOT / "docs/thesis_eval/nir_enhancement/tables/rain_median_n_sweep.csv"

EXPECTED_N_VALUES = {2, 3, 5, 7}
REQUIRED_COLS = {
    "frame", "env_class", "n_frames",
    "raw_log_rms", "raw_pct_sat",
    "out_log_rms", "out_pct_sat", "out_pct_crushed", "out_entropy",
    "delta_log_rms", "proc_ms", "surrogate_mode",
}
FINITE_FLOAT_COLS = ("raw_log_rms", "out_log_rms", "out_pct_sat", "delta_log_rms")


@pytest.fixture(scope="module")
def sweep_rows():
    if not SWEEP_CSV.exists():
        pytest.skip(f"rain median N sweep CSV not found: {SWEEP_CSV}")
    with SWEEP_CSV.open(newline="") as fh:
        return list(csv.DictReader(fh))


def test_sweep_not_empty(sweep_rows):
    assert sweep_rows, "Rain median N sweep CSV is empty"


def test_sweep_schema(sweep_rows):
    cols = set(sweep_rows[0].keys())
    missing = REQUIRED_COLS - cols
    assert not missing, f"Missing columns: {missing}"


def test_all_n_values_present(sweep_rows):
    found = {int(r["n_frames"]) for r in sweep_rows}
    missing = EXPECTED_N_VALUES - found
    assert not missing, f"N values missing from sweep: {missing}"


def test_env_class_all_rain(sweep_rows):
    non_rain = [r["frame"] for r in sweep_rows if r["env_class"] != "rain"]
    assert not non_rain, f"Non-rain frames in sweep: {non_rain[:5]}"


def test_iqa_cols_finite(sweep_rows):
    bad = []
    for i, row in enumerate(sweep_rows):
        for col in FINITE_FLOAT_COLS:
            v = float(row[col])
            if not math.isfinite(v):
                bad.append((i, col, v))
    assert not bad, f"Non-finite IQA values: {bad[:5]}"


def test_proc_ms_positive(sweep_rows):
    bad = [(r["frame"], r["n_frames"], r["proc_ms"])
           for r in sweep_rows if float(r["proc_ms"]) < 0]
    assert not bad, f"Negative proc_ms: {bad[:5]}"


def test_surrogate_mode_label(sweep_rows):
    unexpected = [r["surrogate_mode"] for r in sweep_rows
                  if r["surrogate_mode"] != "pseudo_sequence_noise_jitter"]
    assert not unexpected, f"Unexpected surrogate_mode values: {set(unexpected)}"
