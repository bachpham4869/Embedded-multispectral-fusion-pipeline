"""thermal_contribution_hist.py — Histogram of per-pixel thermal contribution.

PROXY NOTE: Uses dummy uniform-gray thermal (absent real MI48 data). With real
thermal, this histogram shows how much the thermal channel moves individual pixel
values above a threshold. With dummy thermal, it quantifies the suppression effect
of alpha-blending toward gray — which is the mathematical inverse of real thermal
uplift. Results must be labeled PROXY. Two Δ values are plotted for sensitivity.

Pixel rule (per Δ):
    contribution_mask = |fused[p] - nir[p]| > Δ
    contribution_pct  = 100 × mean(contribution_mask)

Output:
    thermal_contribution_hist.png — per-frame histogram at two Δ values + summary
    thermal_contribution_hist.csv — per-frame (env_class, alpha, delta_a, delta_b, pct_a, pct_b)

Usage:
    python tools/thermal_contribution_hist.py
    python tools/thermal_contribution_hist.py \\
        --manifest data/eval/nir_val/manifest_v2.csv \\
        --delta-a 5 --delta-b 10 \\
        --alpha 0.55 \\
        --out-dir docs/thesis_eval/fusion/figures/
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from smartbinocular.nir_pipeline import HybridNIREnhancer

DEFAULT_MANIFEST = ROOT / "data/eval/nir_val/manifest_v2.csv"
DEFAULT_OUT_DIR = ROOT / "docs/thesis_eval/fusion/figures"
PRODUCTION_ALPHA = 0.55


def _load_manifest(path: Path) -> List[dict]:
    rows = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            p = row.get("path", "")
            if p and Path(p).exists():
                rows.append(row)
    return rows


def _import_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("ERROR: matplotlib not installed.", file=sys.stderr)
        sys.exit(1)


def compute_contribution(
    manifest: List[dict], alpha: float, delta_a: int, delta_b: int
) -> List[dict]:
    enhancer = HybridNIREnhancer()
    rows = []
    for frame in manifest:
        img_path = frame["path"]
        env_cls = frame.get("env_class", "unknown")
        bgr = cv.imread(img_path, cv.IMREAD_COLOR)
        if bgr is None:
            continue
        bgr = cv.resize(bgr, (320, 240))

        enhancer.reset()
        nir_out = enhancer.process(bgr)

        h, w = nir_out.shape[:2]
        thermal = np.full((h, w, 3), 127, dtype=np.uint8)
        fused = cv.addWeighted(nir_out, 1.0 - alpha, thermal, alpha, 0)

        nir_g = cv.cvtColor(nir_out, cv.COLOR_BGR2GRAY).astype(np.int16)
        fus_g = cv.cvtColor(fused, cv.COLOR_BGR2GRAY).astype(np.int16)
        diff = np.abs(fus_g - nir_g)

        pct_a = float(np.mean(diff > delta_a)) * 100
        pct_b = float(np.mean(diff > delta_b)) * 100

        rows.append({
            "frame": Path(img_path).name,
            "env_class": env_cls,
            "alpha": alpha,
            f"pct_pixels_above_delta_{delta_a}": round(pct_a, 2),
            f"pct_pixels_above_delta_{delta_b}": round(pct_b, 2),
            "proxy_note": "dummy_thermal_uniform_gray",
        })

    return rows


def plot_histogram(rows: List[dict], delta_a: int, delta_b: int, alpha: float, out_dir: Path, plt) -> None:
    col_a = f"pct_pixels_above_delta_{delta_a}"
    col_b = f"pct_pixels_above_delta_{delta_b}"

    pct_a = [r[col_a] for r in rows]
    pct_b = [r[col_b] for r in rows]

    # Per-class means
    by_class: dict = {}
    for r in rows:
        cls = r["env_class"]
        by_class.setdefault(cls, {col_a: [], col_b: []})
        by_class[cls][col_a].append(r[col_a])
        by_class[cls][col_b].append(r[col_b])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(pct_a, bins=20, color="#2c7bb6", edgecolor="white", alpha=0.8, label=f"Δ={delta_a}")
    ax1.hist(pct_b, bins=20, color="#d7191c", edgecolor="white", alpha=0.6, label=f"Δ={delta_b}")
    ax1.set_xlabel("% pixels with |fused - NIR| > Δ")
    ax1.set_ylabel("Frame count")
    ax1.set_title(
        f"Thermal pixel contribution distribution (α={alpha})\n"
        f"PROXY — dummy thermal; two Δ values for sensitivity analysis",
        fontsize=9,
    )
    ax1.legend(fontsize=9)

    classes = sorted(by_class)
    means_a = [float(np.mean(by_class[c][col_a])) for c in classes]
    means_b = [float(np.mean(by_class[c][col_b])) for c in classes]
    x = np.arange(len(classes))
    w = 0.35
    ax2.bar(x - w/2, means_a, w, label=f"Δ={delta_a}", color="#2c7bb6", alpha=0.85)
    ax2.bar(x + w/2, means_b, w, label=f"Δ={delta_b}", color="#d7191c", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, rotation=35, ha="right", fontsize=8)
    ax2.set_ylabel("Mean % pixels above Δ")
    ax2.set_title("Per-env-class mean thermal contribution\n(PROXY — dummy thermal)", fontsize=9)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "thermal_contribution_hist.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--delta-a", type=int, default=5)
    p.add_argument("--delta-b", type=int, default=10)
    p.add_argument("--alpha", type=float, default=PRODUCTION_ALPHA)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    plt = _import_mpl()

    if not args.manifest.exists():
        print(f"THESIS_SKIP: manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(0)

    manifest = _load_manifest(args.manifest)
    if not manifest:
        print(f"THESIS_SKIP: no readable frames in {args.manifest}", file=sys.stderr)
        sys.exit(0)

    print(f"Thermal contribution: {len(manifest)} frames, α={args.alpha}, Δ={args.delta_a}/{args.delta_b}")
    print("PROXY NOTE: thermal is dummy uniform-gray. Results show blending suppression, not real thermal uplift.")

    rows = compute_contribution(manifest, args.alpha, args.delta_a, args.delta_b)
    if not rows:
        print("No rows produced.", file=sys.stderr)
        sys.exit(1)

    # Save CSV
    csv_out = args.out_dir / "thermal_contribution_hist.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {csv_out}")

    plot_histogram(rows, args.delta_a, args.delta_b, args.alpha, args.out_dir, plt)


if __name__ == "__main__":
    main()
