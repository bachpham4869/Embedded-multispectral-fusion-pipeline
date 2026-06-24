"""sweep_transition_blend.py — Bucket F (nir_transition_blend) threshold sweep.

Grid: lo × hi
Uses single-frame EMA proxy: nir_b_ema_norm = mean(green_channel)/255 per still.

Pre-registered stopping rule:
    find (lo, hi) where ≥40% of mixed_edge images have w ∈ (0.05, 0.95)
    (genuinely blended, not degenerate to A or C alone)

NOTE: Single-frame EMA proxy saturates to w=0 or w=1 more often than live EMA
(which smooths over ~10-20 frames). This result is directional only.
Flagged in surrogate_mode column.

Usage:
    python tools/sweep_transition_blend.py \\
        --manifest data/eval/nir_val/manifest_v2.csv \\
        --out docs/tables/iqa/sweep_transition_f.csv \\
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
from smartbinocular.nir_pipeline import HybridNIREnhancer, nir_transition_blend

_LOS = [0.10, 0.15, 0.20]
_HIS = [0.40, 0.45, 0.55]

# Stopping rule: ≥40% of mixed_edge images have w ∈ (w_lo, w_hi)
_W_INNER_LO = 0.05
_W_INNER_HI = 0.95
_BLEND_RATIO_TARGET = 0.40

_OUTPUT_FIELDS = [
    "env_class", "edge_case", "lo", "hi", "n",
    "mean_blend_w", "pct_genuinely_blended", "mean_delta_log_rms",
    "mean_pct_sat_after", "mean_pct_crush_after", "surrogate_mode", "stopping_rule_pass",
]


def _load_manifest(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="docs/tables/iqa/sweep_transition_f.csv")
    parser.add_argument("--proc-w", type=int, default=320)
    parser.add_argument("--proc-h", type=int, default=240)
    parser.add_argument("--dry-run", action="store_true", help="First 5 images")
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    if not manifest:
        print(f"No images in {args.manifest}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        manifest = manifest[:5]
        print(f"[dry-run] {len(manifest)} images", file=sys.stderr)

    enhancer = HybridNIREnhancer(proc_w=args.proc_w, proc_h=args.proc_h, update_rate=1)
    grid = list(product(_LOS, _HIS))
    print(
        f"[sweep_transition_f] {len(grid)} cells × {len(manifest)} images",
        file=sys.stderr,
    )

    out_rows: list[dict] = []

    for i, (lo, hi) in enumerate(grid, 1):
        if lo >= hi:
            continue
        by_class: dict[str, list] = {}
        t0 = time.perf_counter()
        for entry in manifest:
            bgr = cv.imread(entry["path"], cv.IMREAD_COLOR)
            if bgr is None:
                continue
            env = entry.get("env_class", "unknown")
            edge = entry.get("edge_case", "")

            green = bgr[:, :, 1].astype(np.float32)
            nir_b_ema_norm = float(green.mean()) / 255.0

            try:
                out = nir_transition_blend(bgr, enhancer, nir_b_ema_norm, lo=lo, hi=hi)
            except Exception:
                continue

            w = float(np.clip((nir_b_ema_norm - lo) / max(hi - lo, 1e-6), 0.0, 1.0))
            g_after = cv.cvtColor(out, cv.COLOR_BGR2GRAY)
            g_before = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
            m_a = compute_iqa_metrics(g_after)
            m_b = compute_iqa_metrics(g_before)
            by_class.setdefault(env, []).append({
                "edge_case": edge,
                "blend_w": w,
                "genuinely_blended": int(_W_INNER_LO < w < _W_INNER_HI),
                "delta_log_rms": m_a["log_rms_contrast"] - m_b["log_rms_contrast"],
                "pct_sat_after": m_a["pct_saturated"],
                "pct_crush_after": m_a["pct_crushed"],
            })

        elapsed = time.perf_counter() - t0
        for env_class, records in by_class.items():
            n = len(records)
            if not n:
                continue
            mean_w = sum(r["blend_w"] for r in records) / n
            pct_blended = sum(r["genuinely_blended"] for r in records) / n
            mean_delta = sum(r["delta_log_rms"] for r in records) / n
            mean_sat = sum(r["pct_sat_after"] for r in records) / n
            mean_crush = sum(r["pct_crush_after"] for r in records) / n
            passes = int(pct_blended >= _BLEND_RATIO_TARGET) if env_class == "mixed_edge" else 0
            out_rows.append({
                "env_class": env_class,
                "edge_case": records[0]["edge_case"],
                "lo": lo,
                "hi": hi,
                "n": n,
                "mean_blend_w": round(mean_w, 5),
                "pct_genuinely_blended": round(pct_blended, 4),
                "mean_delta_log_rms": round(mean_delta, 5),
                "mean_pct_sat_after": round(mean_sat, 5),
                "mean_pct_crush_after": round(mean_crush, 5),
                "surrogate_mode": "single_frame_ema_proxy",
                "stopping_rule_pass": passes,
            })
        print(f"  cell {i}/{len(grid)}: lo={lo} hi={hi}  {elapsed:.1f}s", file=sys.stderr)

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

    # Stopping-rule summary for mixed_edge.
    print(f"\n--- stopping rule (mixed_edge pct_blended ≥ {_BLEND_RATIO_TARGET}) ---", file=sys.stderr)
    mixed_rows = [r for r in out_rows if r["env_class"] == "mixed_edge"]
    passing = [r for r in mixed_rows if r["stopping_rule_pass"]]
    if passing:
        best = max(passing, key=lambda r: r["pct_genuinely_blended"])
        print(
            f"  Best: lo={best['lo']} hi={best['hi']}  "
            f"pct_blended={best['pct_genuinely_blended']:.3f}  "
            f"Δlog_rms={best['mean_delta_log_rms']:+.4f}",
            file=sys.stderr,
        )
    else:
        print("  No mixed_edge cell meets the blend ratio target. EMA proxy likely saturates.", file=sys.stderr)

    # Write summary markdown.
    summary_path = out_path.with_name(out_path.stem + "_summary.md")
    with open(summary_path, "w") as f:
        f.write("# Bucket F (nir_transition_blend) Sweep Summary\n\n")
        f.write("**Surrogate:** `single_frame_ema_proxy` — nir_b_ema_norm = mean(green)/255 per still.\n")
        f.write("Results are directional only; live EMA smooths over ~10–20 frames.\n\n")
        f.write(f"**Stopping rule (mixed_edge only):** pct_genuinely_blended ≥ {_BLEND_RATIO_TARGET}\n\n")
        f.write("| env_class | lo | hi | n | mean_blend_w | pct_blended | Δlog_rms | pass |\n")
        f.write("|-----------|----|----|---|-------------|------------|---------|------|\n")
        for r in out_rows:
            p = "YES" if r["stopping_rule_pass"] else "—"
            f.write(
                f"| {r['env_class']} | {r['lo']} | {r['hi']} | {r['n']} | "
                f"{r['mean_blend_w']:.3f} | {r['pct_genuinely_blended']:.3f} | "
                f"{r['mean_delta_log_rms']:+.4f} | {p} |\n"
            )
    print(f"Wrote summary to {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
