"""Tests for RPi4 session inventory and JSON schema.

Guarded by THESIS_RPI_SESSIONS_OK=1 env var — skipped by default so CI is unaffected.
When enabled, verifies:
- session_index.csv exists with ≥6 rows, each ≥300 s, host=RPi4
- Each referenced session_*.json has required keys
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SESSION_INDEX = ROOT / "docs/thesis_eval/timing_performance/tables/session_index.csv"
SESSION_JSON_DIR = ROOT / "fusion_captures/metrics"
REQUIRED_JSON_KEYS = {"stage_timing_ms", "fps_mean", "pipeline_config_sha256"}
MIN_SESSIONS = 6
MIN_DURATION_S = 300

_RPI_ENABLED = os.environ.get("THESIS_RPI_SESSIONS_OK", "0") == "1"
pytestmark = pytest.mark.skipif(
    not _RPI_ENABLED,
    reason="Set THESIS_RPI_SESSIONS_OK=1 to run RPi4 session inventory tests",
)


@pytest.fixture(scope="module")
def session_index_rows():
    if not SESSION_INDEX.exists():
        pytest.skip(f"session_index.csv not found: {SESSION_INDEX}")
    with SESSION_INDEX.open(newline="") as fh:
        return list(csv.DictReader(fh))


def test_session_index_exists():
    assert SESSION_INDEX.exists(), f"Missing: {SESSION_INDEX}"


def test_minimum_session_count(session_index_rows):
    rpi_rows = [r for r in session_index_rows if r.get("host", "").startswith("RPi4")]
    assert len(rpi_rows) >= MIN_SESSIONS, (
        f"Expected ≥{MIN_SESSIONS} RPi4 sessions; found {len(rpi_rows)}"
    )


def test_minimum_duration(session_index_rows):
    short = [
        r["session_id"]
        for r in session_index_rows
        if r.get("host", "").startswith("RPi4")
        and float(r.get("duration_s", 0)) < MIN_DURATION_S
    ]
    assert not short, (
        f"{len(short)} RPi4 sessions below {MIN_DURATION_S}s: {short[:5]}"
    )


def test_required_json_keys_present(session_index_rows):
    missing_keys = []
    for row in session_index_rows:
        if not row.get("host", "").startswith("RPi4"):
            continue
        sid = row.get("session_id", "")
        json_path = SESSION_JSON_DIR / f"session_{sid}.json"
        if not json_path.exists():
            missing_keys.append(f"{sid}: file missing")
            continue
        data = json.loads(json_path.read_text())
        absent = REQUIRED_JSON_KEYS - set(data.keys())
        if absent:
            missing_keys.append(f"{sid}: missing keys {absent}")
    assert not missing_keys, (
        f"Session JSON key errors:\n" + "\n".join(f"  {e}" for e in missing_keys[:10])
    )
