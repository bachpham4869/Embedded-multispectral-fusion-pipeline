#!/usr/bin/env python3
"""Evaluate fusion quality metrics on paired or weakly paired local captures."""

from __future__ import annotations

import argparse
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

from fusion_eval_manifest import markdown_table, read_csv_rows, run_manifest, write_csv, write_json, write_markdown  # type: ignore[import]
from fusion_eval_metrics import compute_fusion_metrics, compute_image_metrics, mine_failure_cases, summarize_metric_rows  # type: ignore[import]


SUMMARY_METRICS = [
    "entropy",
    "rms_contrast",
    "log_rms_contrast",
    "laplacian_variance",
    "tenengrad",
    "pct_dark_clipped",
    "pct_highlight_saturated",
    "noise_proxy",
    "mutual_information_nir_fusion",
    "mutual_information_thermal_fusion",
    "normalized_mutual_information_nir_fusion",
    "normalized_mutual_information_thermal_fusion",
    "ssim_fusion_nir_proxy",
    "ssim_fusion_thermal_proxy",
    "qabf_edge_proxy",
    "foreground_contrast_gain",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate paired/weak/proxy fusion metrics with evidence labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pairing-manifest", type=Path, default=Path("artifacts/fusion_eval/pairing_manifest.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/fusion_eval"))
    parser.add_argument("--summary", type=Path, default=Path("docs/tables/fusion/fusion_quality_summary.md"))
    parser.add_argument("--failure-summary", type=Path, default=Path("docs/tables/fusion/failure_case_summary.md"))
    parser.add_argument("--figure", type=Path, default=Path("docs/figures/fusion/failure_cases_grid.png"))
    parser.add_argument("--max-pairs", type=int, default=0, help="0 means all pairs")
    return parser.parse_args()


def _load_bgr(path_text: str) -> np.ndarray | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists():
        return None
    return cv.imread(str(path), cv.IMREAD_COLOR)


def _simple_mask_from_thermal(thermal: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if thermal is None:
        return None
    gray = cv.cvtColor(thermal, cv.COLOR_BGR2GRAY) if thermal.ndim == 3 else thermal
    resized = cv.resize(gray, (shape[1], shape[0]), interpolation=cv.INTER_AREA)
    thresh = float(np.percentile(resized, 90))
    mask = (resized >= thresh).astype(np.uint8) * 255
    return cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))


def _alignment_status(nir: np.ndarray | None, fusion: np.ndarray | None, evidence_label: str) -> str:
    if nir is None or fusion is None:
        return "missing"
    if evidence_label != "real_paired":
        return "weak_pair_or_proxy"
    return "not_measured"


def _row_metrics(pair: dict[str, str]) -> dict[str, object] | None:
    fusion = _load_bgr(pair.get("fusion_path", ""))
    nir = _load_bgr(pair.get("nir_path", ""))
    thermal = _load_bgr(pair.get("thermal_path", ""))
    if fusion is None or nir is None:
        return None
    if thermal is None:
        thermal = np.zeros_like(fusion)
    if thermal.shape[:2] != fusion.shape[:2]:
        thermal = cv.resize(thermal, (fusion.shape[1], fusion.shape[0]), interpolation=cv.INTER_AREA)
    if nir.shape[:2] != fusion.shape[:2]:
        nir = cv.resize(nir, (fusion.shape[1], fusion.shape[0]), interpolation=cv.INTER_AREA)
    mask = _simple_mask_from_thermal(thermal, fusion.shape[:2])
    metrics = compute_fusion_metrics(nir, thermal, fusion, fg_mask=mask)
    baseline = compute_image_metrics(nir)
    row = {
        "pair_id": pair.get("pair_id", ""),
        "algorithm": "foreground_mask_overlay",
        "baseline_algorithm": "nir_only_baseline",
        "pair_status": pair.get("pair_status", ""),
        "evidence_label": pair.get("evidence_label", "unknown"),
        "bucket_or_condition": pair.get("bucket_or_condition", ""),
        "source_or_session": pair.get("source_or_session", ""),
        "fusion_path": pair.get("fusion_path", ""),
        "nir_path": pair.get("nir_path", ""),
        "thermal_path": pair.get("thermal_path", ""),
        "time_gap_nir_sec": pair.get("time_gap_nir_sec", ""),
        "time_gap_thermal_sec": pair.get("time_gap_thermal_sec", ""),
        "alignment_status": _alignment_status(nir, fusion, pair.get("evidence_label", "")),
    }
    for key, value in metrics.items():
        if value is not None:
            row[key] = round(float(value), 6)
    for key in ("entropy", "rms_contrast", "log_rms_contrast", "laplacian_variance", "tenengrad", "pct_dark_clipped", "pct_highlight_saturated", "noise_proxy"):
        if key in baseline:
            row[f"baseline_{key}"] = round(float(baseline[key]), 6)
    return row


def _summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        for metric in SUMMARY_METRICS:
            if metric not in row:
                continue
            value = row[metric]
            baseline = row.get(f"baseline_{metric}", "")
            key = (
                str(row.get("algorithm", "")),
                str(row.get("evidence_label", "")),
                str(row.get("bucket_or_condition", "")),
                str(row.get("source_or_session", "")),
                metric,
            )
            grouped[key].append({"value": value, "baseline_value": baseline})
    out = []
    for (algorithm, evidence, condition, source, metric), values in sorted(grouped.items()):
        s = summarize_metric_rows(values)
        out.append(
            {
                "algorithm": algorithm,
                "evidence_label": evidence,
                "bucket_or_condition": condition,
                "source_or_session": source,
                "metric": metric,
                **{k: "" if v is None else round(float(v), 6) for k, v in s.items()},
            }
        )
    return out


def _write_grid(rows: list[dict[str, object]], figure: Path) -> None:
    figure.parent.mkdir(parents=True, exist_ok=True)
    picks = rows[:4]
    if not picks:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No failure examples available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(figure, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return
    fig, axes = plt.subplots(len(picks), 3, figsize=(9, max(3, len(picks) * 2.4)))
    if len(picks) == 1:
        axes = np.asarray([axes])
    for row_idx, row in enumerate(picks):
        for col_idx, (label, key) in enumerate((("NIR", "nir_path"), ("Thermal", "thermal_path"), ("Fusion", "fusion_path"))):
            ax = axes[row_idx, col_idx]
            img = _load_bgr(str(row.get(key, "")))
            if img is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
                ax.axis("off")
                continue
            ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            ax.set_title(f"{label} {row.get('pair_id', '')}", fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(figure, dpi=130)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    manifest = (ROOT / args.pairing_manifest).resolve() if not args.pairing_manifest.is_absolute() else args.pairing_manifest
    out_dir = (ROOT / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    summary_path = (ROOT / args.summary).resolve() if not args.summary.is_absolute() else args.summary
    failure_path = (ROOT / args.failure_summary).resolve() if not args.failure_summary.is_absolute() else args.failure_summary
    figure = (ROOT / args.figure).resolve() if not args.figure.is_absolute() else args.figure

    pairs = read_csv_rows(manifest)
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]
    metric_rows = [row for pair in pairs if (row := _row_metrics(pair)) is not None]
    summary_rows = _summary_rows(metric_rows)
    failure_summary, failure_examples = mine_failure_cases(metric_rows)

    write_csv(out_dir / "fusion_pair_metrics.csv", metric_rows)
    write_csv(out_dir / "fusion_quality_summary.csv", summary_rows)
    write_csv(out_dir / "failure_case_summary.csv", failure_summary)
    columns = [
        "algorithm",
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
    body = (
        "Tier 1 claims require strict paired rows. Rows labeled proxy/unpaired are proxy only, "
        "not proof of real fusion quality, and require future paired capture validation.\n\n"
    )
    body += markdown_table(summary_rows[:240], columns)
    write_markdown(summary_path, "Fusion Quality Summary", body)
    failure_body = "Failure mining flags clipping/noise regressions, target fade, mask issues, and alignment drift.\n\n"
    failure_body += markdown_table(failure_summary, ["failure_type", "algorithm", "evidence_label", "count"])
    if failure_examples:
        failure_body += "\n## Examples\n\n" + markdown_table(
            failure_examples[:50],
            ["failure_type", "severity", "pair_id", "algorithm", "evidence_label", "detail"],
        )
    write_markdown(failure_path, "Fusion Failure Case Summary", failure_body)
    _write_grid(failure_examples or metric_rows, figure)
    write_json(
        out_dir / "fusion_quality_run_manifest.json",
        run_manifest("python3 tools/evaluate_fusion_quality.py", [str(manifest)], {"max_pairs": args.max_pairs}),
    )
    print(f"fusion_metric_rows={len(metric_rows)} failure_types={len(failure_summary)}")


if __name__ == "__main__":
    main()
