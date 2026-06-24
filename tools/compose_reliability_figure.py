"""compose_reliability_figure.py — Composite per-class reliability PNGs into one figure.

Source PNGs are produced by models/train_classifier.py (plot_reliability_diagrams) from
OOF CV predictions at training time.  Assembles a grid (default 3x3 for 9 ENV classes)
or a 1x3 night-only strip.

Usage:
    python tools/compose_reliability_figure.py
    python tools/compose_reliability_figure.py --stem rf_phase1_retrain_optical12
    python tools/compose_reliability_figure.py --layout 3x3 --out docs/figures/ml/reliability/reliability_all_env_classes.png
    python tools/compose_reliability_figure.py --layout 1x3 --classes night_clear normal_night nir_night \\
        --out docs/figures/ml/reliability/reliability_night_classes.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

# Default: full ENV list order (match feature_schema.ENV_CLASSES)
_NIGHT_DEFAULT = ["night_clear", "normal_night", "nir_night"]
DEFAULT_STEM = "rf_phase1_retrain_optical12"
# Per-class reliability PNGs (from train_classifier plot_reliability); kept under docs/ so models/baseline can stay small.
DEFAULT_SRC = ROOT / "docs" / "figures" / "ml" / "reliability"
DEFAULT_OUT_NIGHT = ROOT / "docs" / "figures" / "ml" / "reliability" / "reliability_night_classes.png"
DEFAULT_OUT_ALL = ROOT / "docs" / "figures" / "ml" / "reliability" / "reliability_all_env_classes.png"


def _load_env_classes() -> List[str]:
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from smartbinocular.feature_schema import ENV_CLASSES  # noqa: WPS433

    return list(ENV_CLASSES)


def _parse_layout(s: str) -> Tuple[int, int]:
    parts = s.lower().strip().split("x")
    if len(parts) != 2:
        raise ValueError("layout must be like 3x3 or 1x3")
    return int(parts[0]), int(parts[1])


def compose(
    stem: str,
    src_dir: Path,
    out_path: Path,
    classes: List[str],
    layout: Tuple[int, int],
    dpi: int = 150,
    strict: bool = True,
) -> None:
    nrow, ncol = layout
    expected = nrow * ncol
    if len(classes) > expected:
        print(
            f"[warn] {len(classes)} classes for {nrow}x{ncol}={expected} cells; extra names ignored.",
            file=sys.stderr,
        )
        classes = classes[:expected]

    pngs = [src_dir / f"{stem}_reliability_{cls}.png" for cls in classes]
    missing = [p for p in pngs if not p.is_file()]
    if missing and strict:
        for p in missing:
            print(f"[error] PNG not found: {p}", file=sys.stderr)
        sys.exit(1)

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 5, nrow * 5))
    axes_flat = axes.flatten() if expected > 1 else [axes]

    for i in range(expected):
        ax = axes_flat[i]
        if i < len(pngs) and pngs[i].is_file():
            img = mpimg.imread(str(pngs[i]))
            ax.imshow(img)
            title = classes[i].replace("_", " ").title() if i < len(classes) else ""
            ax.set_title(title, fontsize=12, pad=6)
        else:
            if i < len(classes):
                label = classes[i]
                ax.text(
                    0.5,
                    0.5,
                    f"missing\n{label}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=11,
                )
                ax.set_title(label.replace("_", " ").title(), fontsize=12, pad=6)
            else:
                ax.axis("off")
        ax.axis("off")

    # Hide unused subplots
    for j in range(len(classes), expected):
        if j < len(axes_flat):
            axes_flat[j].axis("off")

    fig.suptitle(
        "Reliability Diagrams — ENV Classifier (OOF CV predictions)",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Written: {out_path}")


def main() -> None:
    all_env = _load_env_classes()
    parser = argparse.ArgumentParser(
        description="Composite reliability PNGs into a multi-panel figure"
    )
    parser.add_argument(
        "--stem", default=DEFAULT_STEM, help="Bundle filename stem (no extension)"
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_SRC,
        help="Directory containing source PNGs",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: night or all path from --layout)",
    )
    parser.add_argument(
        "--layout",
        default="1x3",
        help="Grid size, e.g. 1x3 (night strip) or 3x3 (nine classes)",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Class names in order; default: 9 ENV for 3x3, or 3 night for 1x3",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Exit if any required PNG is missing (default: true)",
    )
    parser.add_argument(
        "--no-strict",
        action="store_false",
        dest="strict",
        help="Draw placeholder text for missing PNGs",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    layout = _parse_layout(args.layout)
    nrow, ncol = layout
    ncells = nrow * ncol

    if args.classes is not None and len(args.classes) > 0:
        class_list = list(args.classes)
    else:
        if ncells == 3:
            class_list = list(_NIGHT_DEFAULT)
        else:
            class_list = all_env[:ncells] if ncells < len(all_env) else list(all_env)

    out = args.out
    if out is None:
        out = DEFAULT_OUT_NIGHT if args.layout == "1x3" else DEFAULT_OUT_ALL

    compose(
        args.stem, args.src, out, class_list, layout, dpi=args.dpi, strict=args.strict
    )


if __name__ == "__main__":
    main()
