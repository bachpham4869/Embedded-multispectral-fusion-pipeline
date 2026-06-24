"""sweep_dehaze_omega.py — Bucket D (nir_dehaze_lite) parameter sweep.

Grid: omega × downsample target
Runs on fog + night classes (mis-dispatch target).

Pre-registered stopping rule:
    max log_rms_contrast on fog class s.t. pct_crush_after < 0.05 on night classes

Usage:
    python tools/sweep_dehaze_omega.py \\
        --manifest data/eval/nir_val/manifest_v2.csv \\
        --out docs/tables/iqa/sweep_dehaze_d.csv \\
        --dry-run
"""

from __future__ import annotations

import argparse
import csv
import inspect
import sys
import time
from itertools import product
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from _iqa_metrics import compute_iqa_metrics  # type: ignore[import]
from smartbinocular.nir_pipeline import nir_dehaze_lite

_SWEEP_CLASSES = {"fog", "night_clear", "normal_night", "nir_night"}
_FOG_CLASS = "fog"
_NIGHT_CLASSES = {"night_clear", "normal_night", "nir_night"}

_OMEGAS = [0.55, 0.70, 0.85, 0.92]
_DOWNSAMPLE_SIZES = [(160, 120), (200, 150)]

_CRUSH_LIMIT_NIGHT = 0.05

_OUTPUT_FIELDS = [
    "env_class", "edge_case", "omega", "ds_w", "ds_h",
    "n", "mean_log_rms_after", "mean_pct_sat_after", "mean_pct_crush_after",
    "mean_delta_log_rms", "p90_pct_crush_after", "stopping_rule_pass",
]

# Check if nir_dehaze_lite accepts omega and ds_target_wh parameters.
_DEHAZE_SIG = inspect.signature(nir_dehaze_lite)
_ACCEPTS_OMEGA = "omega" in _DEHAZE_SIG.parameters
_ACCEPTS_DS = "ds_target_wh" in _DEHAZE_SIG.parameters


def _dehaze(bgr: np.ndarray, omega: float, ds_w: int, ds_h: int) -> np.ndarray:
    kwargs: dict = {}
    if _ACCEPTS_OMEGA:
        kwargs["omega"] = omega
    if _ACCEPTS_DS:
        kwargs["ds_target_wh"] = (ds_w, ds_h)
    return nir_dehaze_lite(bgr, **kwargs)


def _load_manifest(path: str, classes: set[str]) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("env_class") in classes:
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="docs/tables/iqa/sweep_dehaze_d.csv")
    parser.add_argument(
        "--classes", nargs="+", default=sorted(_SWEEP_CLASSES),
        help="env_class values to include",
    )
    parser.add_argument("--dry-run", action="store_true", help="First 5 images per class")
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest, set(args.classes))
    if not manifest:
        print(f"No sweep-class images in {args.manifest}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        manifest = manifest[:5]
        print(f"[dry-run] {len(manifest)} images", file=sys.stderr)

    if not _ACCEPTS_OMEGA:
        print(
            "WARNING: nir_dehaze_lite does not accept 'omega' param — sweep will use default.",
            file=sys.stderr,
        )

    grid = list(product(_OMEGAS, _DOWNSAMPLE_SIZES))
    print(
        f"[sweep_dehaze_d] {len(grid)} cells × {len(manifest)} images (accepts_omega={_ACCEPTS_OMEGA})",
        file=sys.stderr,
    )

    out_rows: list[dict] = []

    for i, (omega, (ds_w, ds_h)) in enumerate(grid, 1):
        by_class: dict[str, list] = {}
        t0 = time.perf_counter()
        for entry in manifest:
            bgr = cv.imread(entry["path"], cv.IMREAD_COLOR)
            if bgr is None:
                continue
            try:
                out = _dehaze(bgr, omega, ds_w, ds_h)
            except Exception as exc:
                print(f"  WARN: {exc}", file=sys.stderr)
                continue
            env = entry.get("env_class", "unknown")
            edge = entry.get("edge_case", "")
            g_after = cv.cvtColor(out, cv.COLOR_BGR2GRAY)
            g_before = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
            m_a = compute_iqa_metrics(g_after)
            m_b = compute_iqa_metrics(g_before)
            by_class.setdefault(env, []).append({
                "edge_case": edge,
                "log_rms_after": m_a["log_rms_contrast"],
                "pct_sat_after": m_a["pct_saturated"],
                "pct_crush_after": m_a["pct_crushed"],
                "delta_log_rms": m_a["log_rms_contrast"] - m_b["log_rms_contrast"],
            })

        elapsed = time.perf_counter() - t0
        for env_class, records in by_class.items():
            n = len(records)
            if not n:
                continue
            pct_crush_sorted = sorted(r["pct_crush_after"] for r in records)
            mean_rms = sum(r["log_rms_after"] for r in records) / n
            mean_sat = sum(r["pct_sat_after"] for r in records) / n
            mean_crush = sum(r["pct_crush_after"] for r in records) / n
            mean_delta = sum(r["delta_log_rms"] for r in records) / n
            p90_crush = pct_crush_sorted[int(0.9 * n)]
            # Stopping rule: night classes only — passes if mean crush < limit.
            passes = int(mean_crush < _CRUSH_LIMIT_NIGHT) if env_class in _NIGHT_CLASSES else 0
            out_rows.append({
                "env_class": env_class,
                "edge_case": records[0]["edge_case"],
                "omega": omega,
                "ds_w": ds_w,
                "ds_h": ds_h,
                "n": n,
                "mean_log_rms_after": round(mean_rms, 5),
                "mean_pct_sat_after": round(mean_sat, 5),
                "mean_pct_crush_after": round(mean_crush, 5),
                "mean_delta_log_rms": round(mean_delta, 5),
                "p90_pct_crush_after": round(p90_crush, 5),
                "stopping_rule_pass": passes,
            })
        print(f"  cell {i}/{len(grid)}: omega={omega} ds={ds_w}×{ds_h}  {elapsed:.1f}s", file=sys.stderr)

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

    # Stopping-rule summary.
    print(f"\n--- stopping rule optimal (fog log_rms max s.t. night crush < {_CRUSH_LIMIT_NIGHT}) ---", file=sys.stderr)
    by_class_out: dict[str, list] = {}
    for r in out_rows:
        by_class_out.setdefault(r["env_class"], []).append(r)

    # Find cells where ALL night classes pass, then pick best fog log_rms.
    passing_cells: set[tuple] = set()
    for cls in _NIGHT_CLASSES:
        for r in by_class_out.get(cls, []):
            if r["stopping_rule_pass"]:
                passing_cells.add((r["omega"], r["ds_w"], r["ds_h"]))

    fog_rows = by_class_out.get(_FOG_CLASS, [])
    if fog_rows and passing_cells:
        fog_passing = [r for r in fog_rows if (r["omega"], r["ds_w"], r["ds_h"]) in passing_cells]
        if fog_passing:
            best = max(fog_passing, key=lambda r: r["mean_log_rms_after"])
            print(
                f"  Best: omega={best['omega']} ds={best['ds_w']}×{best['ds_h']}  "
                f"fog log_rms={best['mean_log_rms_after']:.4f}  Δlog_rms={best['mean_delta_log_rms']:+.4f}",
                file=sys.stderr,
            )
        else:
            print("  No cell passes night crush limit — all have crush > limit on dark images.", file=sys.stderr)
    else:
        print("  Could not compute cross-class optimal.", file=sys.stderr)

    summary_path = out_path.with_name(out_path.stem + "_summary.md")
    with open(summary_path, "w") as f:
        f.write("# Bucket D (nir_dehaze_lite) Sweep Summary\n\n")
        f.write(f"**Stopping rule:** max fog `log_rms_after` s.t. night `pct_crush_after < {_CRUSH_LIMIT_NIGHT}`\n\n")
        f.write(f"**Note:** accepts_omega={_ACCEPTS_OMEGA}; if False, omega column is constant (default used).\n\n")
        f.write("| env_class | omega | ds | n | log_rms_after | pct_crush_after | Δlog_rms | pass |\n")
        f.write("|-----------|-------|-----|---|--------------|----------------|---------|------|\n")
        for r in out_rows:
            p = "YES" if r["stopping_rule_pass"] else "—"
            f.write(
                f"| {r['env_class']} | {r['omega']} | {r['ds_w']}×{r['ds_h']} | {r['n']} | "
                f"{r['mean_log_rms_after']:.4f} | {r['mean_pct_crush_after']:.4f} | "
                f"{r['mean_delta_log_rms']:+.4f} | {p} |\n"
            )
    print(f"Wrote summary to {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
