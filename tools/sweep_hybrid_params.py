"""sweep_hybrid_params.py — Bucket A (HybridNIREnhancer) parameter sweep.

Grid: detail_strength × clahe_clip_scale × proc_w/h
Runs on night-class images only (night_clear, normal_night, nir_night) from the manifest.

Pre-registered stopping rule:
    max log_rms_contrast_after s.t. pct_saturated_after < 0.05

Usage:
    python tools/sweep_hybrid_params.py \\
        --manifest data/eval/nir_val/manifest_v2.csv \\
        --out docs/tables/iqa/sweep_hybrid_a.csv \\
        --dry-run
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from itertools import product
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from _iqa_metrics import compute_iqa_metrics  # type: ignore[import]
from smartbinocular.nir_pipeline import HybridNIREnhancer

_NIGHT_CLASSES = {"night_clear", "normal_night", "nir_night"}

_DETAIL_STRENGTHS = [0.0, 0.16, 0.25, 0.35]
_CLAHE_CLIP_SCALES = [0.5, 1.0, 1.2, 1.5]
_PROC_SIZES = [(320, 240), (240, 180)]

# Stopping rule: max log_rms s.t. pct_saturated_after < SAT_LIMIT
_SAT_LIMIT = 0.05

_OUTPUT_FIELDS = [
    "env_class", "edge_case", "detail_strength", "clahe_clip_scale", "proc_w", "proc_h",
    "n", "mean_log_rms_after", "mean_pct_sat_after", "mean_pct_crush_after",
    "mean_delta_log_rms", "p90_pct_sat_after", "p90_pct_crush_after",
    "stopping_rule_pass",
]


def _load_manifest(path: str, classes: set[str]) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("env_class") in classes:
                rows.append(row)
    return rows


def _run_cell(
    manifest: list[dict],
    detail_strength: float,
    clahe_clip_scale: float,
    proc_w: int,
    proc_h: int,
) -> dict[str, list]:
    """Run one parameter cell, return per-image metric lists keyed by env_class."""
    by_class: dict[str, list] = {}

    enhancer = HybridNIREnhancer(
        proc_w=proc_w,
        proc_h=proc_h,
        update_rate=1,
        detail_strength=detail_strength,
    )
    # Set clahe clip scale if supported by the constructor signature.
    # Fall back to post-hoc attribute override if needed.
    try:
        enhancer2 = HybridNIREnhancer(
            proc_w=proc_w,
            proc_h=proc_h,
            update_rate=1,
            detail_strength=detail_strength,
            clahe_clip_scale=clahe_clip_scale,
        )
        enhancer = enhancer2
    except TypeError:
        # Constructor doesn't accept clahe_clip_scale — set attribute directly.
        if hasattr(enhancer, "clahe_clip_scale"):
            enhancer.clahe_clip_scale = clahe_clip_scale
        elif hasattr(enhancer, "_clahe_clip_scale"):
            enhancer._clahe_clip_scale = clahe_clip_scale

    for entry in manifest:
        img_path = entry["path"]
        env_class = entry.get("env_class", "unknown")
        edge_case = entry.get("edge_case", "")

        bgr = cv.imread(img_path, cv.IMREAD_COLOR)
        if bgr is None:
            continue

        enhancer.reset()
        try:
            out = enhancer.process(bgr)
        except Exception:
            continue

        gray_after = cv.cvtColor(out, cv.COLOR_BGR2GRAY)
        gray_before = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
        m_after = compute_iqa_metrics(gray_after)
        m_before = compute_iqa_metrics(gray_before)

        rec = {
            "env_class": env_class,
            "edge_case": edge_case,
            "log_rms_after": m_after["log_rms_contrast"],
            "pct_sat_after": m_after["pct_saturated"],
            "pct_crush_after": m_after["pct_crushed"],
            "delta_log_rms": m_after["log_rms_contrast"] - m_before["log_rms_contrast"],
        }
        by_class.setdefault(env_class, []).append(rec)

    return by_class


def _aggregate_cell(
    records: list[dict],
    detail_strength: float,
    clahe_clip_scale: float,
    proc_w: int,
    proc_h: int,
    env_class: str,
    edge_case: str = "",
) -> dict:
    if not records:
        return {}
    n = len(records)
    log_rms = [r["log_rms_after"] for r in records]
    pct_sat = sorted(r["pct_sat_after"] for r in records)
    pct_crush = sorted(r["pct_crush_after"] for r in records)
    delta = [r["delta_log_rms"] for r in records]

    mean_log_rms = sum(log_rms) / n
    mean_pct_sat = sum(pct_sat) / n
    mean_pct_crush = sum(r["pct_crush_after"] for r in records) / n
    mean_delta = sum(delta) / n
    p90_sat = pct_sat[int(0.9 * n)]
    p90_crush = pct_crush[int(0.9 * n)]

    return {
        "env_class": env_class,
        "edge_case": edge_case,
        "detail_strength": detail_strength,
        "clahe_clip_scale": clahe_clip_scale,
        "proc_w": proc_w,
        "proc_h": proc_h,
        "n": n,
        "mean_log_rms_after": round(mean_log_rms, 5),
        "mean_pct_sat_after": round(mean_pct_sat, 5),
        "mean_pct_crush_after": round(mean_pct_crush, 5),
        "mean_delta_log_rms": round(mean_delta, 5),
        "p90_pct_sat_after": round(p90_sat, 5),
        "p90_pct_crush_after": round(p90_crush, 5),
        "stopping_rule_pass": int(mean_pct_sat < _SAT_LIMIT),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="docs/tables/iqa/sweep_hybrid_a.csv")
    parser.add_argument(
        "--classes", nargs="+", default=sorted(_NIGHT_CLASSES),
        help="env_class values to include (default: night classes)",
    )
    parser.add_argument("--dry-run", action="store_true", help="First 10 images per class only")
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest, set(args.classes))
    if not manifest:
        print(f"No night-class images in {args.manifest}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        manifest = manifest[:10]
        print(f"[dry-run] {len(manifest)} images", file=sys.stderr)

    grid = list(product(_DETAIL_STRENGTHS, _CLAHE_CLIP_SCALES, _PROC_SIZES))
    total = len(grid)
    print(
        f"[sweep_hybrid_a] {total} cells × {len(manifest)} images = {total * len(manifest)} evals",
        file=sys.stderr,
    )

    out_rows: list[dict] = []
    for i, (ds, clip, (pw, ph)) in enumerate(grid, 1):
        t0 = time.perf_counter()
        by_class = _run_cell(manifest, ds, clip, pw, ph)
        elapsed = time.perf_counter() - t0

        for env_class, records in by_class.items():
            row = _aggregate_cell(records, ds, clip, pw, ph, env_class)
            if row:
                out_rows.append(row)

        print(
            f"  cell {i}/{total}: ds={ds} clip={clip} proc={pw}×{ph}  {elapsed:.1f}s",
            file=sys.stderr,
        )

    if not out_rows:
        print("No output.", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_OUTPUT_FIELDS)
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {len(out_rows)} rows to {out_path}", file=sys.stderr)

    # Find stopping-rule optimal cell per class.
    print("\n--- stopping rule optimal (per env_class) ---", file=sys.stderr)
    by_class_out: dict[str, list] = {}
    for r in out_rows:
        by_class_out.setdefault(r["env_class"], []).append(r)
    for cls, rows in sorted(by_class_out.items()):
        passing = [r for r in rows if r["stopping_rule_pass"]]
        if not passing:
            print(f"  {cls}: NO passing cells (all sat ≥ {_SAT_LIMIT})", file=sys.stderr)
            continue
        best = max(passing, key=lambda r: r["mean_log_rms_after"])
        print(
            f"  {cls}: ds={best['detail_strength']} clip={best['clahe_clip_scale']} "
            f"proc={best['proc_w']}×{best['proc_h']}  "
            f"log_rms={best['mean_log_rms_after']:.4f}  sat={best['mean_pct_sat_after']:.4f}",
            file=sys.stderr,
        )

    # Write summary markdown.
    summary_path = out_path.with_suffix("").with_name(out_path.stem + "_summary.md")
    with open(summary_path, "w") as f:
        f.write(f"# Bucket A Parameter Sweep Summary\n\n")
        f.write(f"**Stopping rule:** max `mean_log_rms_after` s.t. `mean_pct_sat_after < {_SAT_LIMIT}`\n\n")
        f.write(f"| env_class | detail_strength | clahe_clip_scale | proc | log_rms_after | pct_sat_after | stopping_rule_pass |\n")
        f.write(f"|-----------|----------------|-----------------|------|--------------|--------------|--------------------|\n")
        for cls, rows in sorted(by_class_out.items()):
            passing = [r for r in rows if r["stopping_rule_pass"]]
            if passing:
                best = max(passing, key=lambda r: r["mean_log_rms_after"])
                f.write(
                    f"| {cls} | {best['detail_strength']} | {best['clahe_clip_scale']} | "
                    f"{best['proc_w']}×{best['proc_h']} | {best['mean_log_rms_after']:.4f} | "
                    f"{best['mean_pct_sat_after']:.4f} | YES |\n"
                )
            else:
                f.write(f"| {cls} | — | — | — | — | — | NO (all fail sat limit) |\n")
    print(f"Wrote summary to {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
