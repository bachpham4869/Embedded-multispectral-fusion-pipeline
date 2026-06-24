"""Consistency tests for A/B fusion vs NIR-only proxy comparison artifact.

Validates:
- ab_fusion_vs_nir_proxy.csv: schema present, same frames for both arms,
  delta columns computable, proxy_note present
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
AB_CSV = ROOT / "docs/thesis_eval/fusion/tables/ab_fusion_vs_nir_proxy.csv"

REQUIRED_COLS = {
    "frame", "env_class", "bucket", "alpha",
    "nir_log_rms", "nir_entropy", "nir_pct_sat",
    "fused_log_rms", "fused_entropy", "fused_pct_sat",
    "delta_log_rms", "delta_entropy", "delta_pct_sat",
    "nir_proc_ms", "proxy_note",
}
FLOAT_COLS = (
    "nir_log_rms", "fused_log_rms", "delta_log_rms",
    "nir_entropy", "fused_entropy",
)


@pytest.fixture(scope="module")
def ab_rows():
    if not AB_CSV.exists():
        pytest.skip(f"A/B CSV not found: {AB_CSV}")
    with AB_CSV.open(newline="") as fh:
        return list(csv.DictReader(fh))


def test_ab_csv_not_empty(ab_rows):
    assert ab_rows, "A/B fusion CSV is empty"


def test_ab_csv_schema(ab_rows):
    cols = set(ab_rows[0].keys())
    missing = REQUIRED_COLS - cols
    assert not missing, f"Missing columns: {missing}"


def test_ab_float_cols_finite(ab_rows):
    bad = []
    for i, row in enumerate(ab_rows):
        for col in FLOAT_COLS:
            v = float(row[col])
            if not math.isfinite(v):
                bad.append((i, col, v))
    assert not bad, f"Non-finite float values: {bad[:5]}"


def test_ab_delta_cols_consistent(ab_rows):
    """delta_log_rms must equal fused_log_rms - nir_log_rms within tolerance."""
    tol = 1e-3
    bad = []
    for i, row in enumerate(ab_rows):
        computed = float(row["fused_log_rms"]) - float(row["nir_log_rms"])
        recorded = float(row["delta_log_rms"])
        if abs(computed - recorded) > tol:
            bad.append((i, computed, recorded))
    assert not bad, f"delta_log_rms mismatch at {len(bad)} rows; first: {bad[0]}"


def test_ab_proxy_note_present(ab_rows):
    missing = [r["frame"] for r in ab_rows if not r.get("proxy_note")]
    assert not missing, f"proxy_note absent for {len(missing)} rows"


def test_ab_buckets_valid(ab_rows):
    valid_buckets = {"A", "B", "C", "D", "E", "F"}
    bad = [(r["frame"], r["bucket"]) for r in ab_rows if r["bucket"] not in valid_buckets]
    assert not bad, f"Invalid bucket values: {bad[:5]}"


def test_ab_alpha_consistent(ab_rows):
    """All rows should use the same alpha (production default)."""
    alphas = {float(r["alpha"]) for r in ab_rows}
    assert len(alphas) == 1, f"Expected single alpha value; found: {alphas}"
