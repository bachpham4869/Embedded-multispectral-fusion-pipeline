#!/usr/bin/env python3
"""Offline image/fusion metrics with evidence-tier metadata.

The functions in this module are intentionally dependency-light and do not alter
runtime pipeline behavior. They are used by thesis/evaluation scripts only.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any, Iterable

import cv2 as cv
import numpy as np


METRIC_DEFINITIONS: dict[str, dict[str, Any]] = {
    "foreground_contrast_gain": {
        "tier": 1,
        "category": "direct/paired/task-specific",
        "limitation": "Valid only for strict paired captures with a usable foreground or target mask.",
    },
    "fps_mean": {
        "tier": 1,
        "category": "direct/session timing",
        "limitation": "Real session timing, but session duration and thermal conditions must be reported.",
    },
    "entropy": {
        "tier": 2,
        "category": "no-reference IQA",
        "limitation": "No-reference tonal diversity proxy; not absolute perceptual quality.",
    },
    "rms_contrast": {
        "tier": 2,
        "category": "no-reference IQA",
        "limitation": "No-reference contrast proxy; can reward noise or artifacts.",
    },
    "log_rms_contrast": {
        "tier": 2,
        "category": "no-reference IQA",
        "limitation": "No-reference contrast proxy; use with clipping/noise checks.",
    },
    "laplacian_variance": {
        "tier": 2,
        "category": "no-reference IQA",
        "limitation": "Sharpness proxy; can increase with noise.",
    },
    "tenengrad": {
        "tier": 2,
        "category": "no-reference IQA",
        "limitation": "Gradient-energy proxy; can reward ringing or sensor noise.",
    },
    "edge_density": {
        "tier": 2,
        "category": "no-reference IQA",
        "limitation": "Edge-count proxy; sensitive to noise and thresholds.",
    },
    "spatial_frequency": {
        "tier": 2,
        "category": "no-reference fusion/IQA",
        "limitation": "Texture proxy; not a direct task metric.",
    },
    "average_gradient": {
        "tier": 2,
        "category": "no-reference fusion/IQA",
        "limitation": "Gradient proxy; should be interpreted with clipping/noise.",
    },
    "mutual_information": {
        "tier": 3,
        "category": "proxy fusion metric",
        "limitation": "Useful for relative analysis only; weak/unpaired images do not prove fusion quality.",
    },
    "normalized_mutual_information": {
        "tier": 3,
        "category": "proxy fusion metric",
        "limitation": "Proxy metric; may be inflated by overlays or deterministic transforms.",
    },
    "ssim_fusion_nir_proxy": {
        "tier": 3,
        "category": "proxy/synthetic/unpaired",
        "limitation": "SSIM to an input is not ground truth; use only as preservation proxy.",
    },
    "qabf_edge_proxy": {
        "tier": 3,
        "category": "proxy fusion metric",
        "limitation": "Dependency-free Qabf-style edge proxy, not a full-reference quality proof.",
    },
}

BOOTSTRAP_MIN_N = 4
EPS = 1e-9


def ensure_gray_u8(image: np.ndarray) -> np.ndarray:
    """Return a 2-D uint8 grayscale image."""
    if image is None:
        raise ValueError("image is None")
    arr = np.asarray(image)
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = cv.cvtColor(arr, cv.COLOR_BGRA2GRAY)
        else:
            arr = cv.cvtColor(arr, cv.COLOR_BGR2GRAY)
    elif arr.ndim != 2:
        raise ValueError(f"Expected 2-D or 3-D image, got shape {arr.shape}")
    if arr.dtype == np.uint8:
        return arr
    return np.clip(arr, 0, 255).astype(np.uint8)


def entropy_u8(gray: np.ndarray) -> float:
    gray = ensure_gray_u8(gray)
    hist = cv.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probs = hist / max(float(hist.sum()), EPS)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs))) if probs.size else 0.0


def clipping_ratios(gray: np.ndarray, *, dark_thresh: int = 5, high_thresh: int = 250) -> dict[str, float]:
    gray = ensure_gray_u8(gray)
    n = float(gray.size)
    return {
        "pct_dark_clipped": float(np.count_nonzero(gray <= dark_thresh)) / n,
        "pct_highlight_saturated": float(np.count_nonzero(gray >= high_thresh)) / n,
    }


def rms_contrast(gray: np.ndarray) -> float:
    arr = ensure_gray_u8(gray).astype(np.float32)
    return float(np.std(arr))


def michelson_contrast(gray: np.ndarray) -> float:
    arr = ensure_gray_u8(gray).astype(np.float32)
    lo = float(np.percentile(arr, 5))
    hi = float(np.percentile(arr, 95))
    return float((hi - lo) / max(hi + lo, EPS))


def laplacian_variance(gray: np.ndarray) -> float:
    gray = ensure_gray_u8(gray)
    return float(cv.Laplacian(gray, cv.CV_32F).var())


def tenengrad(gray: np.ndarray) -> float:
    gray = ensure_gray_u8(gray)
    gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def edge_density(gray: np.ndarray) -> float:
    gray = ensure_gray_u8(gray)
    edges = cv.Canny(gray, 80, 160)
    return float(np.count_nonzero(edges)) / float(edges.size)


def noise_proxy(gray: np.ndarray) -> float:
    gray = ensure_gray_u8(gray)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    residual = gray.astype(np.float32) - blurred.astype(np.float32)
    return float(np.std(residual))


def spatial_frequency(gray: np.ndarray) -> float:
    arr = ensure_gray_u8(gray).astype(np.float32)
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        return 0.0
    row_freq = np.sqrt(np.mean(np.diff(arr, axis=0) ** 2))
    col_freq = np.sqrt(np.mean(np.diff(arr, axis=1) ** 2))
    return float(np.sqrt(row_freq * row_freq + col_freq * col_freq))


def average_gradient(gray: np.ndarray) -> float:
    arr = ensure_gray_u8(gray).astype(np.float32)
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        return 0.0
    gx = np.diff(arr, axis=1)[:-1, :]
    gy = np.diff(arr, axis=0)[:, :-1]
    return float(np.mean(np.sqrt((gx * gx + gy * gy) / 2.0)))


def mutual_information(a: np.ndarray, b: np.ndarray, *, bins: int = 32) -> float:
    aa = ensure_gray_u8(a).ravel()
    bb = ensure_gray_u8(b).ravel()
    if aa.size != bb.size:
        raise ValueError("mutual_information inputs must have matching sizes")
    hist_2d, _, _ = np.histogram2d(aa, bb, bins=bins, range=[[0, 255], [0, 255]])
    pxy = hist_2d / max(float(hist_2d.sum()), EPS)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    px_py = px[:, None] * py[None, :]
    nz = pxy > 0
    return float(np.sum(pxy[nz] * np.log2(pxy[nz] / np.maximum(px_py[nz], EPS))))


def normalized_mutual_information(a: np.ndarray, b: np.ndarray, *, bins: int = 32) -> float:
    aa = ensure_gray_u8(a)
    bb = ensure_gray_u8(b)
    mi = mutual_information(aa, bb, bins=bins)
    h_a = entropy_u8(aa)
    h_b = entropy_u8(bb)
    nmi = mi / max(math.sqrt(h_a * h_b), EPS)
    corr = float(np.corrcoef(aa.ravel().astype(float), bb.ravel().astype(float))[0, 1]) if aa.size > 1 else 0.0
    if not math.isfinite(corr):
        corr = 0.0
    # For thesis use this is an intensity-retention proxy, so inverted deterministic
    # mappings should not score the same as same-polarity retention.
    return float(max(0.0, nmi * ((corr + 1.0) / 2.0)))


def ssim_proxy(a: np.ndarray, b: np.ndarray) -> float:
    aa = ensure_gray_u8(a).astype(np.float32)
    bb = ensure_gray_u8(b).astype(np.float32)
    if aa.shape != bb.shape:
        bb = cv.resize(bb, (aa.shape[1], aa.shape[0]), interpolation=cv.INTER_AREA)
    c1 = 6.5025
    c2 = 58.5225
    mu_a = cv.GaussianBlur(aa, (7, 7), 1.5)
    mu_b = cv.GaussianBlur(bb, (7, 7), 1.5)
    sigma_a = cv.GaussianBlur(aa * aa, (7, 7), 1.5) - mu_a * mu_a
    sigma_b = cv.GaussianBlur(bb * bb, (7, 7), 1.5) - mu_b * mu_b
    sigma_ab = cv.GaussianBlur(aa * bb, (7, 7), 1.5) - mu_a * mu_b
    score = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / (
        (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2) + EPS
    )
    return float(np.clip(np.mean(score), -1.0, 1.0))


def qabf_edge_proxy(source_a: np.ndarray, source_b: np.ndarray, fused: np.ndarray) -> float:
    """A simple edge-retention proxy inspired by Qabf."""
    fa = laplacian_variance(source_a)
    fb = laplacian_variance(source_b)
    ff = laplacian_variance(fused)
    denom = max(fa, fb, EPS)
    return float(np.clip(ff / denom, 0.0, 1.0))


def compute_image_metrics(image: np.ndarray) -> dict[str, float]:
    gray = ensure_gray_u8(image)
    arr = gray.astype(np.float32)
    rms = rms_contrast(gray)
    values = {
        "brightness_mean": float(np.mean(arr)),
        "brightness_p5": float(np.percentile(arr, 5)),
        "brightness_p50": float(np.percentile(arr, 50)),
        "brightness_p95": float(np.percentile(arr, 95)),
        "rms_contrast": rms,
        "log_rms_contrast": float(math.log10(rms + 1e-6)),
        "michelson_contrast": michelson_contrast(gray),
        "entropy": entropy_u8(gray),
        "laplacian_variance": laplacian_variance(gray),
        "tenengrad": tenengrad(gray),
        "edge_density": edge_density(gray),
        "noise_proxy": noise_proxy(gray),
        "spatial_frequency": spatial_frequency(gray),
        "average_gradient": average_gradient(gray),
    }
    values.update(clipping_ratios(gray))
    return values


def foreground_contrast(gray: np.ndarray, mask: np.ndarray | None) -> float | None:
    if mask is None:
        return None
    g = ensure_gray_u8(gray).astype(np.float32)
    m = ensure_gray_u8(mask) > 0
    if not np.any(m) or np.count_nonzero(~m) < 4:
        return None
    fg = g[m]
    bg = g[~m]
    return float((np.mean(fg) - np.mean(bg)) / (np.std(bg) + EPS))


def compute_fusion_metrics(
    nir: np.ndarray,
    thermal: np.ndarray,
    fused: np.ndarray,
    *,
    fg_mask: np.ndarray | None = None,
) -> dict[str, float | None]:
    nir_g = ensure_gray_u8(nir)
    th_g = ensure_gray_u8(thermal)
    fus_g = ensure_gray_u8(fused)
    if th_g.shape != fus_g.shape:
        th_g = cv.resize(th_g, (fus_g.shape[1], fus_g.shape[0]), interpolation=cv.INTER_AREA)
    if nir_g.shape != fus_g.shape:
        nir_g = cv.resize(nir_g, (fus_g.shape[1], fus_g.shape[0]), interpolation=cv.INTER_AREA)
    metrics = compute_image_metrics(fus_g)
    metrics.update(
        {
            "mutual_information_nir_fusion": mutual_information(nir_g, fus_g),
            "mutual_information_thermal_fusion": mutual_information(th_g, fus_g),
            "normalized_mutual_information_nir_fusion": normalized_mutual_information(nir_g, fus_g),
            "normalized_mutual_information_thermal_fusion": normalized_mutual_information(th_g, fus_g),
            "ssim_fusion_nir_proxy": ssim_proxy(nir_g, fus_g),
            "ssim_fusion_thermal_proxy": ssim_proxy(th_g, fus_g),
            "qabf_edge_proxy": qabf_edge_proxy(nir_g, th_g, fus_g),
            "foreground_contrast_gain": None,
        }
    )
    if fg_mask is not None:
        if fg_mask.shape != fus_g.shape:
            fg_mask = cv.resize(ensure_gray_u8(fg_mask), (fus_g.shape[1], fus_g.shape[0]), interpolation=cv.INTER_NEAREST)
        f_contrast = foreground_contrast(fus_g, fg_mask)
        n_contrast = foreground_contrast(nir_g, fg_mask)
        if f_contrast is not None and n_contrast is not None:
            metrics["foreground_contrast_gain"] = f_contrast - n_contrast
        metrics["mask_area_ratio"] = float(np.count_nonzero(fg_mask)) / float(fg_mask.size)
    return metrics


def _finite(values: Iterable[Any]) -> list[float]:
    out = []
    for value in values:
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def bootstrap_ci(
    values: Iterable[float],
    *,
    iterations: int = 1000,
    seed: int = 42,
    min_n: int = BOOTSTRAP_MIN_N,
) -> tuple[float | None, float | None]:
    vals = np.asarray(_finite(values), dtype=np.float64)
    if vals.size < min_n:
        return None, None
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(iterations):
        sample = rng.choice(vals, size=vals.size, replace=True)
        stats.append(float(np.median(sample)))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def summarize_metric_rows(
    rows: Iterable[dict[str, Any]],
    *,
    value_key: str = "value",
    baseline_key: str = "baseline_value",
    bootstrap_iterations: int = 1000,
) -> dict[str, Any]:
    row_list = list(rows)
    values = _finite(row.get(value_key) for row in row_list)
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "p25": None,
            "p75": None,
            "p95": None,
            "bootstrap_ci95_low": None,
            "bootstrap_ci95_high": None,
            "delta_current_minus_baseline": None,
            "win_rate_current_vs_baseline": None,
        }
    deltas = []
    wins = []
    for row in row_list:
        try:
            value = float(row[value_key])
            baseline = float(row[baseline_key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value) and math.isfinite(baseline):
            deltas.append(value - baseline)
            wins.append(1.0 if value > baseline else 0.0)
    low, high = bootstrap_ci(values, iterations=bootstrap_iterations)
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "bootstrap_ci95_low": low,
        "bootstrap_ci95_high": high,
        "delta_current_minus_baseline": float(np.mean(deltas)) if deltas else None,
        "win_rate_current_vs_baseline": float(np.mean(wins)) if wins else None,
    }


def metric_tier(metric_name: str, evidence_label: str = "") -> int:
    if evidence_label in {"proxy", "synthetic", "unpaired", "unknown"}:
        return 3
    return int(METRIC_DEFINITIONS.get(metric_name, {"tier": 2})["tier"])


def detect_failure_cases(row: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []

    def f(key: str, default: float | None = None) -> float | None:
        try:
            value = float(row.get(key, default))
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) else default

    def add(code: str, severity: str, detail: str) -> None:
        failures.append(
            {
                "failure_type": code,
                "severity": severity,
                "detail": detail,
                "pair_id": row.get("pair_id") or row.get("image_id") or row.get("path") or "",
                "algorithm": row.get("algorithm", ""),
                "baseline_algorithm": row.get("baseline_algorithm", ""),
                "evidence_label": row.get("evidence_label", ""),
                "source_or_session": row.get("source_or_session", ""),
            }
        )

    sat = f("pct_highlight_saturated")
    base_sat = f("baseline_pct_highlight_saturated")
    dark = f("pct_dark_clipped")
    base_dark = f("baseline_pct_dark_clipped")
    if sat is not None and base_sat is not None and sat - base_sat >= 0.02:
        add("clipping_increase", "medium", f"highlight saturation increased by {sat - base_sat:.4f}")
    if dark is not None and base_dark is not None and dark - base_dark >= 0.02:
        add("clipping_increase", "medium", f"dark clipping increased by {dark - base_dark:.4f}")
    if dark is not None and base_dark is not None and dark > max(base_dark * 1.5, base_dark + 0.05):
        add("crushed_shadows", "medium", f"dark clipping rose from {base_dark:.4f} to {dark:.4f}")

    noise = f("noise_proxy")
    base_noise = f("baseline_noise_proxy")
    if noise is not None and base_noise is not None and noise > max(base_noise * 1.2, base_noise + 0.5):
        add("noise_increase", "medium", f"noise proxy increased from {base_noise:.4f} to {noise:.4f}")

    sharp = f("laplacian_variance")
    base_sharp = f("baseline_laplacian_variance")
    contrast = f("rms_contrast")
    base_contrast = f("baseline_rms_contrast")
    if (
        sharp is not None
        and base_sharp is not None
        and contrast is not None
        and base_contrast is not None
        and sharp > base_sharp * 1.1
        and contrast < base_contrast * 0.95
    ):
        add("sharpness_up_contrast_down", "low", "sharpness rose while contrast fell")
    if contrast is not None and base_contrast is not None and contrast < base_contrast * 0.85:
        add("contrast_drop", "medium", f"contrast dropped from {base_contrast:.4f} to {contrast:.4f}")

    fg_gain = f("foreground_contrast_gain")
    if fg_gain is not None and fg_gain < 0:
        add("fusion_target_faded", "high", f"foreground contrast gain is negative ({fg_gain:.4f})")

    ssim_nir = f("ssim_fusion_nir_proxy")
    base_ssim_nir = f("baseline_ssim_fusion_nir_proxy")
    if ssim_nir is not None and base_ssim_nir is not None and ssim_nir < base_ssim_nir - 0.2:
        add("nir_detail_occlusion", "medium", f"NIR preservation proxy dropped from {base_ssim_nir:.4f} to {ssim_nir:.4f}")

    mask_area = f("mask_area_ratio")
    if mask_area is not None and (mask_area > 0.7 or 0.0 < mask_area < 0.001):
        add("mask_wrong_region", "high", f"mask area ratio is {mask_area:.4f}")

    align_status = str(row.get("alignment_status", "")).lower()
    inlier_ratio = f("alignment_inlier_ratio")
    if "fail" in align_status or "unusable" in align_status or (inlier_ratio is not None and inlier_ratio < 0.2):
        add("alignment_drift", "high", align_status or f"inlier_ratio={inlier_ratio:.4f}")

    return failures


def mine_failure_cases(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    for row in rows:
        failures.extend(detect_failure_cases(row))

    counts: Counter[tuple[str, str, str]] = Counter(
        (f["failure_type"], f.get("algorithm", ""), f.get("evidence_label", "")) for f in failures
    )
    summary = [
        {
            "failure_type": failure_type,
            "algorithm": algorithm,
            "evidence_label": evidence_label,
            "count": count,
        }
        for (failure_type, algorithm, evidence_label), count in sorted(counts.items())
    ]

    examples_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for failure in failures:
        if len(examples_by_type[failure["failure_type"]]) < 5:
            examples_by_type[failure["failure_type"]].append(failure)
    examples = [item for values in examples_by_type.values() for item in values]
    return summary, examples
