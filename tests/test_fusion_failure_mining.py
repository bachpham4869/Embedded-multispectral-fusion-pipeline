from __future__ import annotations

from tools.fusion_eval_metrics import detect_failure_cases, mine_failure_cases


def test_detect_failure_cases_flags_metric_regressions():
    row = {
        "pair_id": "p1",
        "algorithm": "foreground_mask_overlay",
        "pct_highlight_saturated": 0.12,
        "baseline_pct_highlight_saturated": 0.04,
        "noise_proxy": 11.0,
        "baseline_noise_proxy": 5.0,
        "laplacian_variance": 40.0,
        "baseline_laplacian_variance": 20.0,
        "rms_contrast": 8.0,
        "baseline_rms_contrast": 10.0,
    }
    failures = detect_failure_cases(row)
    codes = {f["failure_type"] for f in failures}
    assert "clipping_increase" in codes
    assert "noise_increase" in codes
    assert "sharpness_up_contrast_down" in codes


def test_detect_failure_cases_flags_fusion_specific_failures():
    row = {
        "pair_id": "p2",
        "algorithm": "foreground_mask_overlay",
        "foreground_contrast_gain": -0.5,
        "mask_area_ratio": 0.82,
        "alignment_status": "failed_or_unusable",
    }
    failures = detect_failure_cases(row)
    codes = {f["failure_type"] for f in failures}
    assert "fusion_target_faded" in codes
    assert "mask_wrong_region" in codes
    assert "alignment_drift" in codes


def test_detect_failure_cases_flags_hardening_failure_modes():
    row = {
        "pair_id": "p3",
        "algorithm": "foreground_mask_overlay",
        "pct_dark_clipped": 0.18,
        "baseline_pct_dark_clipped": 0.02,
        "rms_contrast": 20.0,
        "baseline_rms_contrast": 28.0,
        "ssim_fusion_nir_proxy": 0.42,
        "baseline_ssim_fusion_nir_proxy": 0.78,
    }
    failures = detect_failure_cases(row)
    codes = {f["failure_type"] for f in failures}
    assert "crushed_shadows" in codes
    assert "contrast_drop" in codes
    assert "nir_detail_occlusion" in codes


def test_mine_failure_cases_returns_summary_and_examples():
    rows = [
        {
            "pair_id": "p1",
            "algorithm": "foreground_mask_overlay",
            "evidence_label": "real_paired",
            "pct_highlight_saturated": 0.12,
            "baseline_pct_highlight_saturated": 0.01,
        },
        {
            "pair_id": "p2",
            "algorithm": "laplacian_pyramid_fusion",
            "evidence_label": "proxy",
            "alignment_status": "failed_or_unusable",
        },
    ]
    summary, examples = mine_failure_cases(rows)
    assert summary
    assert examples
    assert any(row["failure_type"] == "clipping_increase" for row in summary)
    assert any(row["failure_type"] == "alignment_drift" for row in examples)
