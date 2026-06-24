"""P1-G: Tests for utils — StreamSkewQualityGate transitions, CaptureIntegrityChain
HMAC chaining, and compute_homography_quality (P1-F)."""

from __future__ import annotations

import tempfile
import os

import numpy as np
import pytest

from smartbinocular.utils import (
    CaptureIntegrityChain,
    StreamSkewQualityGate,
    compute_homography_quality,
)


# ── StreamSkewQualityGate ─────────────────────────────────────────────────────

class TestStreamSkewQualityGate:
    def _gate(self, hold: int = 3) -> StreamSkewQualityGate:
        return StreamSkewQualityGate(
            ema_alpha=1.0,   # instant EMA — no lag, deterministic tests
            degraded_on_ms=20.0,
            bad_on_ms=50.0,
            degraded_back_ms=14.0,
            bad_back_ms=32.0,
            hold_frames=hold,
        )

    def test_initial_state_is_good(self):
        gate = self._gate()
        assert gate.state == "GOOD"

    def test_good_to_degraded_after_hold_frames(self):
        gate = self._gate(hold=3)
        for _ in range(2):
            state, _, changed = gate.update(25.0)
            assert state == "GOOD"
        state, _, changed = gate.update(25.0)
        assert state == "DEGRADED"
        assert changed

    def test_degraded_to_bad_after_hold_frames(self):
        gate = self._gate(hold=3)
        # Push to DEGRADED first
        for _ in range(3):
            gate.update(25.0)
        assert gate.state == "DEGRADED"
        # Now push toward BAD
        for _ in range(2):
            state, _, _ = gate.update(60.0)
            assert state == "DEGRADED"
        state, _, changed = gate.update(60.0)
        assert state == "BAD"
        assert changed

    def test_bad_to_degraded_after_hold_frames(self):
        gate = self._gate(hold=3)
        # Push to BAD
        for _ in range(3):
            gate.update(25.0)   # → DEGRADED
        for _ in range(3):
            gate.update(60.0)   # → BAD
        assert gate.state == "BAD"
        # Feed below bad_back_ms (32.0) to recover
        for _ in range(2):
            state, _, _ = gate.update(20.0)
            assert state == "BAD"
        state, _, changed = gate.update(20.0)
        assert state == "DEGRADED"
        assert changed

    def test_none_skew_does_not_change_state(self):
        gate = self._gate()
        state, ema, changed = gate.update(None)
        assert state == "GOOD"
        assert ema is None
        assert not changed

    def test_ema_is_updated(self):
        gate = self._gate()
        _, ema, _ = gate.update(30.0)
        assert ema is not None
        assert abs(ema - 30.0) < 1.0  # ema_alpha=1 → instant convergence


# ── CaptureIntegrityChain ─────────────────────────────────────────────────────

class TestCaptureIntegrityChain:
    def _chain(self) -> CaptureIntegrityChain:
        tmp = tempfile.mktemp(suffix=".key")
        return CaptureIntegrityChain(key_path=tmp)

    def test_first_sign_has_no_prev_hash(self):
        chain = self._chain()
        meta = {"frame": 0, "ts": 1.0}
        signed = chain.sign(meta)
        assert "hmac_sha256" in signed
        assert "prev_capture_hash" not in signed

    def test_second_sign_links_to_first(self):
        """The Merkle-style invariant: second capture's prev_capture_hash ==
        first capture's hmac_sha256."""
        chain = self._chain()
        m1 = chain.sign({"frame": 0, "ts": 1.0})
        first_hmac = m1["hmac_sha256"]

        m2 = chain.sign({"frame": 1, "ts": 2.0})
        assert m2.get("prev_capture_hash") == first_hmac

    def test_hmac_is_hex_string(self):
        chain = self._chain()
        signed = chain.sign({"x": 1})
        hmac_val = signed["hmac_sha256"]
        assert isinstance(hmac_val, str)
        # SHA-256 hex digest is 64 characters
        assert len(hmac_val) == 64

    def test_chain_is_deterministic_with_same_key(self):
        """Two chains using the same key file produce the same HMAC for the same payload."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".key") as f:
            key_path = f.name
        try:
            chain1 = CaptureIntegrityChain(key_path=key_path)
            chain2 = CaptureIntegrityChain(key_path=key_path)
            meta1 = {"frame": 0}
            meta2 = {"frame": 0}
            s1 = chain1.sign(meta1)
            s2 = chain2.sign(meta2)
            assert s1["hmac_sha256"] == s2["hmac_sha256"]
        finally:
            os.unlink(key_path)


# ── compute_homography_quality ────────────────────────────────────────────────

class TestComputeHomographyQuality:
    def test_identity_scale_corners_within_nir(self):
        """An 8× scale homography mapping 80×62 thermal → 640×480 NIR should
        keep all four projected corners within the NIR frame boundaries."""
        # Scale thermal (80×62) exactly to NIR (640×480)
        H = np.array([
            [8.0, 0.0, 0.0],
            [0.0, 8.0, 0.0],  # note: 62×8=496 > 480, but let's use a valid scale
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        # Use a scale that fits: thermal (80,62) → nir (640,480): sx=8, sy≈7.74
        H = np.array([
            [8.0, 0.0, 0.0],
            [0.0, 7.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        result = compute_homography_quality(H, thermal_shape=(62, 80), nir_shape=(480, 640))
        assert result["all_corners_within_nir"] is True
        assert isinstance(result["max_corner_drift_px"], float)
        assert result["max_corner_drift_px"] >= 0.0

    def test_corners_outside_nir_detected(self):
        """A homography that projects corners far outside the NIR frame should
        return all_corners_within_nir=False."""
        H = np.array([
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0,   0.0, 1.0],
        ], dtype=np.float64)
        result = compute_homography_quality(H, thermal_shape=(62, 80), nir_shape=(480, 640))
        assert result["all_corners_within_nir"] is False

    def test_drift_is_non_negative(self):
        H = np.eye(3, dtype=np.float64)
        result = compute_homography_quality(H, thermal_shape=(62, 80), nir_shape=(480, 640))
        assert result["max_corner_drift_px"] >= 0.0

    def test_return_keys_present(self):
        H = np.eye(3, dtype=np.float64)
        result = compute_homography_quality(H, thermal_shape=(62, 80), nir_shape=(480, 640))
        assert "max_corner_drift_px" in result
        assert "all_corners_within_nir" in result
