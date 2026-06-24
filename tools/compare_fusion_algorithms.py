#!/usr/bin/env python3
"""Compare current foreground-mask fusion with common offline baselines."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from fusion_eval_manifest import FUSION_MODE_NAMES, markdown_table, read_csv_rows, run_manifest, write_csv, write_json, write_markdown  # type: ignore[import]
from fusion_eval_metrics import compute_image_metrics, summarize_metric_rows  # type: ignore[import]


METRICS = [
    "entropy",
    "rms_contrast",
    "log_rms_contrast",
    "laplacian_variance",
    "tenengrad",
    "pct_dark_clipped",
    "pct_highlight_saturated",
    "noise_proxy",
    "spatial_frequency",
    "average_gradient",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare fusion modes: current foreground overlay vs alpha, mask-weighted, legacy gradient, Laplacian pyramid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pairing-manifest", type=Path, default=Path("artifacts/fusion_eval/pairing_manifest.csv"))
    parser.add_argument("--proxy-nir-manifest", type=Path, default=Path("data/eval/nir_val/manifest_v2.csv"))
    parser.add_argument("--thermal-seq-dir", type=Path, default=Path("data/eval/thermal_seq"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/fusion_eval/fusion_algorithms"))
    parser.add_argument("--summary", type=Path, default=Path("docs/tables/fusion/fusion_algorithm_comparison.md"))
    parser.add_argument("--figure", type=Path, default=Path("docs/figures/fusion/sample_comparison_grid.png"))
    parser.add_argument("--max-pairs", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.55)
    return parser.parse_args()


def _gray(bgr: np.ndarray) -> np.ndarray:
    return cv.cvtColor(bgr, cv.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr


def _thermal_bgr(thermal: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    gray = _gray(thermal)
    resized = cv.resize(np.clip(gray, 0, 255).astype(np.uint8), (shape[1], shape[0]), interpolation=cv.INTER_AREA)
    return cv.applyColorMap(resized, cv.COLORMAP_JET)


def _mask(thermal: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    gray = cv.cvtColor(_thermal_bgr(thermal, shape), cv.COLOR_BGR2GRAY)
    thr = float(np.percentile(gray, 90))
    mask = (gray >= thr).astype(np.uint8) * 255
    return cv.GaussianBlur(mask, (9, 9), 0)


def _alpha_blend(nir: np.ndarray, thermal: np.ndarray, alpha: float) -> np.ndarray:
    tb = _thermal_bgr(thermal, nir.shape[:2])
    return cv.addWeighted(nir, 1.0 - alpha, tb, alpha, 0)


def _foreground_overlay(nir: np.ndarray, thermal: np.ndarray, alpha: float) -> np.ndarray:
    tb = _thermal_bgr(thermal, nir.shape[:2]).astype(np.float32)
    m = (_mask(thermal, nir.shape[:2]).astype(np.float32) / 255.0)[:, :, None] * alpha
    out = nir.astype(np.float32) * (1.0 - m) + tb * m
    return np.clip(out, 0, 255).astype(np.uint8)


def _mask_weighted(nir: np.ndarray, thermal: np.ndarray, alpha: float) -> np.ndarray:
    tb = _thermal_bgr(thermal, nir.shape[:2]).astype(np.float32)
    heat = cv.cvtColor(tb.astype(np.uint8), cv.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    m = (0.25 + 0.75 * heat)[:, :, None] * alpha
    out = nir.astype(np.float32) * (1.0 - m) + tb * m
    return np.clip(out, 0, 255).astype(np.uint8)


def _legacy_gradient(nir: np.ndarray, thermal: np.ndarray, alpha: float) -> np.ndarray:
    tb = _thermal_bgr(thermal, nir.shape[:2]).astype(np.float32)
    heat = cv.cvtColor(tb.astype(np.uint8), cv.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    m = (heat * 0.6 * alpha)[:, :, None]
    out = nir.astype(np.float32) * (1.0 - m) + tb * m
    return np.clip(out, 0, 255).astype(np.uint8)


def _laplacian_pyramid_fusion(nir: np.ndarray, thermal: np.ndarray) -> np.ndarray:
    tb = _thermal_bgr(thermal, nir.shape[:2])
    n_gray = cv.cvtColor(nir, cv.COLOR_BGR2GRAY).astype(np.float32)
    t_gray = cv.cvtColor(tb, cv.COLOR_BGR2GRAY).astype(np.float32)
    n_edges = cv.Laplacian(n_gray, cv.CV_32F)
    t_edges = cv.Laplacian(t_gray, cv.CV_32F)
    weight = np.abs(t_edges) / (np.abs(n_edges) + np.abs(t_edges) + 1e-6)
    weight = cv.GaussianBlur(weight, (11, 11), 0)[:, :, None]
    out = nir.astype(np.float32) * (1.0 - weight) + tb.astype(np.float32) * weight
    return np.clip(out, 0, 255).astype(np.uint8)


def _load_real_cases(pairing_manifest: Path, max_pairs: int) -> list[dict[str, object]]:
    cases = []
    for row in read_csv_rows(pairing_manifest):
        nir_path = Path(row.get("nir_path", ""))
        thermal_path = Path(row.get("thermal_path", ""))
        fusion_path = Path(row.get("fusion_path", ""))
        if not (nir_path.exists() and thermal_path.exists() and fusion_path.exists()):
            continue
        nir = cv.imread(str(nir_path), cv.IMREAD_COLOR)
        thermal = cv.imread(str(thermal_path), cv.IMREAD_COLOR)
        fusion = cv.imread(str(fusion_path), cv.IMREAD_COLOR)
        if nir is None or thermal is None or fusion is None:
            continue
        cases.append(
            {
                "case_id": row.get("pair_id", f"real_{len(cases):04d}"),
                "nir": cv.resize(nir, (fusion.shape[1], fusion.shape[0]), interpolation=cv.INTER_AREA),
                "thermal": thermal,
                "current": fusion,
                "evidence_label": row.get("evidence_label", "unknown"),
                "bucket_or_condition": row.get("bucket_or_condition", ""),
                "source_or_session": row.get("source_or_session", ""),
            }
        )
        if len(cases) >= max_pairs:
            break
    return cases


def _load_proxy_cases(nir_manifest: Path, thermal_seq_dir: Path, max_pairs: int) -> list[dict[str, object]]:
    nirs = []
    with nir_manifest.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            path = Path(row.get("path", ""))
            if path.exists():
                nirs.append(row)
            if len(nirs) >= max_pairs:
                break
    thermal_frames = []
    for seq_path in sorted(thermal_seq_dir.glob("*.npy")):
        arr = np.load(seq_path)
        if arr.ndim == 3:
            thermal_frames.append(arr[arr.shape[0] // 2])
        elif arr.ndim == 2:
            thermal_frames.append(arr)
        if len(thermal_frames) >= max_pairs:
            break
    cases = []
    for idx, row in enumerate(nirs[: min(len(nirs), len(thermal_frames), max_pairs)]):
        nir = cv.imread(row["path"], cv.IMREAD_COLOR)
        if nir is None:
            continue
        thermal = np.clip(thermal_frames[idx], 0, 255).astype(np.uint8)
        cases.append(
            {
                "case_id": f"proxy_{idx:04d}",
                "nir": cv.resize(nir, (640, 480), interpolation=cv.INTER_AREA),
                "thermal": thermal,
                "current": None,
                "evidence_label": "proxy",
                "bucket_or_condition": row.get("env_class", ""),
                "source_or_session": "proxy_nir_manifest_plus_thermal_sequence",
            }
        )
    return cases


def _evaluate_case(case: dict[str, object], alpha: float) -> list[dict[str, object]]:
    nir = case["nir"]
    thermal = case["thermal"]
    assert isinstance(nir, np.ndarray)
    assert isinstance(thermal, np.ndarray)
    current = case.get("current")
    if not isinstance(current, np.ndarray):
        current = _foreground_overlay(nir, thermal, alpha)
    if current.shape[:2] != nir.shape[:2]:
        current = cv.resize(current, (nir.shape[1], nir.shape[0]), interpolation=cv.INTER_AREA)
    baselines = {
        "nir_only_baseline": nir,
        "thermal_heatmap_only": _thermal_bgr(thermal, nir.shape[:2]),
        "alpha_blend_baseline": _alpha_blend(nir, thermal, alpha),
        "mask_weighted_blend": _mask_weighted(nir, thermal, alpha),
        "legacy_gradient_overlay": _legacy_gradient(nir, thermal, alpha),
        "laplacian_pyramid_fusion": _laplacian_pyramid_fusion(nir, thermal),
    }
    current_metrics = compute_image_metrics(current)
    rows = []
    for baseline_name, baseline_img in baselines.items():
        base_metrics = compute_image_metrics(baseline_img)
        for metric in METRICS:
            rows.append(
                {
                    "case_id": case["case_id"],
                    "algorithm": "foreground_mask_overlay",
                    "baseline_algorithm": baseline_name,
                    "evidence_label": case["evidence_label"],
                    "bucket_or_condition": case["bucket_or_condition"],
                    "source_or_session": case["source_or_session"],
                    "metric": metric,
                    "value": round(float(current_metrics[metric]), 6),
                    "baseline_value": round(float(base_metrics[metric]), 6),
                }
            )
    return rows


def _summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["algorithm"]),
            str(row["baseline_algorithm"]),
            str(row["evidence_label"]),
            str(row["bucket_or_condition"]),
            str(row["metric"]),
        )
        grouped[key].append(row)
    out = []
    for (algorithm, baseline, evidence, condition, metric), values in sorted(grouped.items()):
        s = summarize_metric_rows(values)
        out.append(
            {
                "algorithm": algorithm,
                "baseline_algorithm": baseline,
                "evidence_label": evidence,
                "bucket_or_condition": condition,
                "metric": metric,
                **{k: "" if v is None else round(float(v), 6) for k, v in s.items()},
            }
        )
    return out


def _write_grid(cases: list[dict[str, object]], figure: Path, alpha: float) -> None:
    figure.parent.mkdir(parents=True, exist_ok=True)
    picks = cases[:3]
    labels = ["NIR", "Thermal", "Alpha", "Foreground", "Laplacian"]
    fig, axes = plt.subplots(max(1, len(picks)), len(labels), figsize=(12, max(3, 2.4 * max(1, len(picks)))))
    if len(picks) == 1:
        axes = np.asarray([axes])
    if not picks:
        axes[0, 0].text(0.5, 0.5, "No fusion cases", ha="center", va="center")
        for ax in axes.ravel():
            ax.axis("off")
    for row_idx, case in enumerate(picks):
        nir = case["nir"]
        thermal = case["thermal"]
        assert isinstance(nir, np.ndarray)
        assert isinstance(thermal, np.ndarray)
        imgs = [
            nir,
            _thermal_bgr(thermal, nir.shape[:2]),
            _alpha_blend(nir, thermal, alpha),
            _foreground_overlay(nir, thermal, alpha),
            _laplacian_pyramid_fusion(nir, thermal),
        ]
        for col_idx, img in enumerate(imgs):
            ax = axes[row_idx, col_idx]
            ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            ax.set_title(labels[col_idx], fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(figure, dpi=130)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    pairing_manifest = (ROOT / args.pairing_manifest).resolve() if not args.pairing_manifest.is_absolute() else args.pairing_manifest
    nir_manifest = (ROOT / args.proxy_nir_manifest).resolve() if not args.proxy_nir_manifest.is_absolute() else args.proxy_nir_manifest
    thermal_seq_dir = (ROOT / args.thermal_seq_dir).resolve() if not args.thermal_seq_dir.is_absolute() else args.thermal_seq_dir
    out_dir = (ROOT / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    summary_path = (ROOT / args.summary).resolve() if not args.summary.is_absolute() else args.summary
    figure = (ROOT / args.figure).resolve() if not args.figure.is_absolute() else args.figure

    real_cases = _load_real_cases(pairing_manifest, args.max_pairs)
    proxy_cases = _load_proxy_cases(nir_manifest, thermal_seq_dir, max(0, args.max_pairs - len(real_cases)))
    cases = real_cases + proxy_cases
    rows: list[dict[str, object]] = []
    for case in cases:
        rows.extend(_evaluate_case(case, args.alpha))
    summary_rows = _summary_rows(rows)
    write_csv(out_dir / "fusion_algorithm_comparison_metrics.csv", rows)
    write_csv(out_dir / "fusion_algorithm_comparison.csv", summary_rows)
    _write_grid(cases, figure, args.alpha)
    columns = [
        "algorithm",
        "baseline_algorithm",
        "evidence_label",
        "bucket_or_condition",
        "metric",
        "n",
        "mean",
        "median",
        "std",
        "p25",
        "p75",
        "p95",
        "delta_current_minus_baseline",
        "win_rate_current_vs_baseline",
    ]
    strict_count = sum(1 for case in real_cases if case.get("evidence_label") == "real_paired")
    body = (
        f"Strict paired fusion cases: {strict_count}. "
        "Proxy rows are proxy only, not proof of real fusion quality, and require future paired capture validation.\n\n"
        f"Fusion modes considered: {', '.join(sorted(FUSION_MODE_NAMES))}.\n\n"
    )
    body += markdown_table(summary_rows[:260], columns)
    write_markdown(summary_path, "Fusion Algorithm Comparison", body)
    write_json(
        out_dir / "fusion_algorithm_run_manifest.json",
        run_manifest(
            "python3 tools/compare_fusion_algorithms.py",
            [str(pairing_manifest), str(nir_manifest), str(thermal_seq_dir)],
            {"max_pairs": args.max_pairs, "alpha": args.alpha, "strict_paired_cases": strict_count},
        ),
    )
    print(f"real_cases={len(real_cases)} proxy_cases={len(proxy_cases)} comparison_rows={len(rows)}")


if __name__ == "__main__":
    main()
