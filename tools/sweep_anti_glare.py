"""sweep_anti_glare.py — Bucket C (nir_anti_glare_bgr) parameter sweep.

Grid: high_pct × saturate_at × gamma
Runs on glare, backlight, normal_day, and mixed_edge classes.

Pre-registered stopping rule:
    minimize pct_sat_after s.t. mean_delta_log_rms >= -0.05

Usage:
    python tools/sweep_anti_glare.py \\
        --manifest data/eval/nir_val/manifest_v2.csv \\
        --out docs/tables/iqa/sweep_anti_glare_c.csv \\
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
from smartbinocular.nir_pipeline import nir_anti_glare_bgr

_TARGET_CLASSES = {"glare", "backlight", "normal_day", "mixed_edge"}

_HIGH_PCTS = [90.0, 93.0, 95.0, 97.0]
_SATURATE_ATS = [220.0, 233.0, 245.0]
_GAMMAS = [0.65, 0.72, 0.80]

# Stopping rule: minimize pct_sat_after s.t. delta_log_rms >= CONTRAST_FLOOR
_CONTRAST_FLOOR = -0.05

_OUTPUT_FIELDS = [
    "env_class", "edge_case", "high_pct", "saturate_at", "gamma",
    "n", "mean_log_rms_after", "mean_pct_sat_after", "mean_pct_crush_after",
    "mean_delta_log_rms", "p90_pct_sat_after", "stopping_rule_pass",
]


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
    parser.add_argument("--out", default="docs/tables/iqa/sweep_anti_glare_c.csv")
    parser.add_argument(
        "--classes", nargs="+", default=sorted(_TARGET_CLASSES),
        help="env_class values to include",
    )
    parser.add_argument("--dry-run", action="store_true", help="First 5 images per class")
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest, set(args.classes))
    if not manifest:
        print(f"No target-class images in {args.manifest}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        manifest = manifest[:5]
        print(f"[dry-run] {len(manifest)} images", file=sys.stderr)

    grid = list(product(_HIGH_PCTS, _SATURATE_ATS, _GAMMAS))
    print(
        f"[sweep_anti_glare_c] {len(grid)} cells × {len(manifest)} images",
        file=sys.stderr,
    )

    out_rows: list[dict] = []

    for i, (hp, sat_at, gamma) in enumerate(grid, 1):
        by_class: dict[str, list] = {}
        t0 = time.perf_counter()
        for entry in manifest:
            bgr = cv.imread(entry["path"], cv.IMREAD_COLOR)
            if bgr is None:
                continue
            try:
                out = nir_anti_glare_bgr(bgr, high_pct=hp, saturate_at=sat_at, gamma=gamma)
            except Exception:
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
            pct_sat_sorted = sorted(r["pct_sat_after"] for r in records)
            mean_delta = sum(r["delta_log_rms"] for r in records) / n
            mean_sat = sum(r["pct_sat_after"] for r in records) / n
            mean_crush = sum(r["pct_crush_after"] for r in records) / n
            mean_rms = sum(r["log_rms_after"] for r in records) / n
            p90_sat = pct_sat_sorted[int(0.9 * n)]
            passes = int(mean_delta >= _CONTRAST_FLOOR)
            out_rows.append({
                "env_class": env_class,
                "edge_case": records[0]["edge_case"],
                "high_pct": hp,
                "saturate_at": sat_at,
                "gamma": gamma,
                "n": n,
                "mean_log_rms_after": round(mean_rms, 5),
                "mean_pct_sat_after": round(mean_sat, 5),
                "mean_pct_crush_after": round(mean_crush, 5),
                "mean_delta_log_rms": round(mean_delta, 5),
                "p90_pct_sat_after": round(p90_sat, 5),
                "stopping_rule_pass": passes,
            })
        print(f"  cell {i}/{len(grid)}: hp={hp} sat_at={sat_at} gamma={gamma}  {elapsed:.1f}s", file=sys.stderr)

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

    # Print stopping-rule optimal per class.
    print(f"\n--- stopping rule optimal (minimize pct_sat s.t. Δlog_rms ≥ {_CONTRAST_FLOOR}) ---", file=sys.stderr)
    by_class_out: dict[str, list] = {}
    for r in out_rows:
        by_class_out.setdefault(r["env_class"], []).append(r)

    summary_rows: list[str] = []
    for cls, rows in sorted(by_class_out.items()):
        passing = [r for r in rows if r["stopping_rule_pass"]]
        if not passing:
            print(f"  {cls}: NO passing cells", file=sys.stderr)
            summary_rows.append(f"| {cls} | — | — | — | — | — | NO |")
            continue
        best = min(passing, key=lambda r: r["mean_pct_sat_after"])
        print(
            f"  {cls}: hp={best['high_pct']} sat_at={best['saturate_at']} gamma={best['gamma']}  "
            f"pct_sat={best['mean_pct_sat_after']:.4f}  Δlog_rms={best['mean_delta_log_rms']:+.4f}",
            file=sys.stderr,
        )
        summary_rows.append(
            f"| {cls} | {best['high_pct']} | {best['saturate_at']} | {best['gamma']} | "
            f"{best['mean_pct_sat_after']:.4f} | {best['mean_delta_log_rms']:+.4f} | YES |"
        )

    summary_path = out_path.with_name(out_path.stem + "_summary.md")
    with open(summary_path, "w") as f:
        f.write("# Bucket C (nir_anti_glare_bgr) Sweep Summary\n\n")
        f.write(f"**Stopping rule:** min `pct_sat_after` s.t. `Δlog_rms ≥ {_CONTRAST_FLOOR}`\n\n")
        f.write("| env_class | high_pct | saturate_at | gamma | pct_sat_after | Δlog_rms | pass |\n")
        f.write("|-----------|----------|------------|-------|--------------|---------|------|\n")
        for line in summary_rows:
            f.write(line + "\n")
    print(f"Wrote summary to {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
