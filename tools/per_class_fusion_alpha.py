"""per_class_fusion_alpha.py — Per-env-class optimal fusion alpha from proxy sweep.

Reads the CSV from sweep_fusion_alpha.py and, for each env_class, finds the alpha
value that maximizes mean log_rms_contrast. Also produces a line-chart figure
showing IQA vs alpha stratified by env class.

PROXY NOTE: Since sweep_fusion_alpha.py uses dummy thermal, the "optimal" alpha
per class reflects minimized NIR contrast suppression, not true fusion gain. Label
PROXY in thesis and interpret with caution.

Usage:
    python tools/per_class_fusion_alpha.py
    python tools/per_class_fusion_alpha.py \\
        --sweep   docs/thesis_eval/fusion/tables/fusion_alpha_sweep_proxy.csv \\
        --out-csv docs/thesis_eval/fusion/tables/per_class_fusion_alpha.csv \\
        --out-fig docs/thesis_eval/fusion/figures/alpha_sweep_curve.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SWEEP = ROOT / "docs/thesis_eval/fusion/tables/fusion_alpha_sweep_proxy.csv"
DEFAULT_OUT_CSV = ROOT / "docs/thesis_eval/fusion/tables/per_class_fusion_alpha.csv"
DEFAULT_OUT_FIG = ROOT / "docs/thesis_eval/fusion/figures/alpha_sweep_curve.png"


def _import_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("ERROR: matplotlib not installed.", file=sys.stderr)
        sys.exit(1)


def _load_sweep(path: Path) -> List[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def compute_per_class(rows: List[dict]) -> List[dict]:
    """For each (env_class, alpha) compute mean fused_log_rms."""
    by_class_alpha: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cls = r["env_class"]
        alpha = float(r["alpha"])
        lrms = float(r["fused_log_rms"])
        by_class_alpha[cls][alpha].append(lrms)

    summary = []
    for cls in sorted(by_class_alpha):
        alphas = sorted(by_class_alpha[cls].keys())
        means = [float(np.mean(by_class_alpha[cls][a])) for a in alphas]
        best_alpha = alphas[int(np.argmax(means))]
        best_lrms = max(means)
        summary.append({
            "env_class": cls,
            "best_alpha": best_alpha,
            "best_fused_log_rms": round(best_lrms, 5),
            "proxy_note": "PROXY — dummy thermal; best alpha = max IQA proxy, not task performance",
        })
    return summary


def plot_alpha_curve(rows: List[dict], out_fig: Path, plt) -> None:
    by_class_alpha: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_class_alpha[r["env_class"]][float(r["alpha"])].append(float(r["fused_log_rms"]))

    fig, ax = plt.subplots(figsize=(9, 6))
    for cls in sorted(by_class_alpha):
        alphas = sorted(by_class_alpha[cls].keys())
        means = [float(np.mean(by_class_alpha[cls][a])) for a in alphas]
        ax.plot(alphas, means, "-o", ms=4, label=cls, lw=1.5)

    ax.axvline(0.55, color="grey", ls="--", lw=1.2, label="α=0.55 (production)")
    ax.set_xlabel("Fusion alpha α")
    ax.set_ylabel("Mean fused log_rms_contrast")
    ax.set_title(
        "Fusion alpha sweep — mean fused IQA proxy by env class\n"
        "PROXY: dummy thermal; 'best α' = best proxy, not task performance",
        fontsize=9,
    )
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_fig}")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP)
    p.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    p.add_argument("--out-fig", type=Path, default=DEFAULT_OUT_FIG)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    plt = _import_mpl()

    if not args.sweep.exists():
        print(f"THESIS_SKIP: sweep CSV not found: {args.sweep}", file=sys.stderr)
        print("Run: python tools/sweep_fusion_alpha.py first", file=sys.stderr)
        sys.exit(0)

    rows = _load_sweep(args.sweep)
    if not rows:
        print("THESIS_SKIP: sweep CSV is empty.", file=sys.stderr)
        sys.exit(0)

    summary = compute_per_class(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(summary[0].keys())
    with args.out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)
    print(f"Saved: {args.out_csv}")
    for row in summary:
        print(f"  {row['env_class']:15s}  best_alpha={row['best_alpha']}  log_rms={row['best_fused_log_rms']}")

    plot_alpha_curve(rows, args.out_fig, plt)


if __name__ == "__main__":
    main()
