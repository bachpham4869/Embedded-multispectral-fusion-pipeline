from __future__ import annotations

import math

import numpy as np

from tools.fusion_eval_metrics import (
    METRIC_DEFINITIONS,
    average_gradient,
    bootstrap_ci,
    clipping_ratios,
    compute_image_metrics,
    entropy_u8,
    mutual_information,
    normalized_mutual_information,
    qabf_edge_proxy,
    spatial_frequency,
    summarize_metric_rows,
)


def test_metric_definitions_include_validity_tiers():
    assert METRIC_DEFINITIONS["entropy"]["tier"] == 2
    assert METRIC_DEFINITIONS["foreground_contrast_gain"]["tier"] == 1
    assert METRIC_DEFINITIONS["ssim_fusion_nir_proxy"]["tier"] == 3


def test_entropy_is_zero_for_constant_image():
    gray = np.full((8, 8), 17, dtype=np.uint8)
    assert entropy_u8(gray) == 0.0


def test_clipping_ratios_detect_dark_and_high_pixels():
    gray = np.array([[0, 3, 10], [250, 255, 20]], dtype=np.uint8)
    ratios = clipping_ratios(gray, dark_thresh=5, high_thresh=250)
    assert ratios["pct_dark_clipped"] == 2 / 6
    assert ratios["pct_highlight_saturated"] == 2 / 6


def test_compute_image_metrics_contains_expected_no_reference_metrics():
    gray = np.tile(np.arange(16, dtype=np.uint8), (16, 1))
    metrics = compute_image_metrics(gray)
    for key in (
        "brightness_mean",
        "brightness_p5",
        "brightness_p50",
        "brightness_p95",
        "rms_contrast",
        "log_rms_contrast",
        "entropy",
        "laplacian_variance",
        "tenengrad",
        "edge_density",
        "noise_proxy",
    ):
        assert key in metrics
        assert math.isfinite(metrics[key])


def test_fusion_information_metrics_are_finite():
    a = np.array([[0, 0, 255, 255], [0, 0, 255, 255]], dtype=np.uint8)
    b = a.copy()
    c = 255 - a
    assert mutual_information(a, b) > 0
    assert normalized_mutual_information(a, b) > normalized_mutual_information(a, c)
    assert spatial_frequency(a) >= 0
    assert average_gradient(a) >= 0
    assert 0 <= qabf_edge_proxy(a, b, b) <= 1


def test_summary_includes_distribution_delta_and_win_rate():
    rows = [
        {"algorithm": "current", "baseline_algorithm": "raw", "metric": "entropy", "value": 5.0, "baseline_value": 4.0},
        {"algorithm": "current", "baseline_algorithm": "raw", "metric": "entropy", "value": 7.0, "baseline_value": 8.0},
        {"algorithm": "current", "baseline_algorithm": "raw", "metric": "entropy", "value": 6.0, "baseline_value": 5.0},
    ]
    summary = summarize_metric_rows(rows, value_key="value", baseline_key="baseline_value")
    assert summary["n"] == 3
    assert summary["mean"] == 6.0
    assert summary["median"] == 6.0
    assert summary["p25"] == 5.5
    assert summary["p75"] == 6.5
    assert summary["p95"] == 6.9
    assert summary["delta_current_minus_baseline"] == pytest_approx((1.0 - 1.0 + 1.0) / 3)
    assert summary["win_rate_current_vs_baseline"] == 2 / 3


def test_bootstrap_ci_requires_enough_samples():
    assert bootstrap_ci([1.0, 2.0, 3.0], min_n=4) == (None, None)
    low, high = bootstrap_ci([1.0, 2.0, 3.0, 4.0], iterations=100, seed=7, min_n=4)
    assert low is not None
    assert high is not None
    assert low <= high


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value)
