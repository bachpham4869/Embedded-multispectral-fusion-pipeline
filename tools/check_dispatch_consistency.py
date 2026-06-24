"""check_dispatch_consistency.py — Compare manifest env_class labels vs rule-inferred class.

Uses the same pipeline signals as the runtime (nir_b_ema, nir_gray_std, glare_nir) to call
infer_env_tags_auto_rule, then maps tags → preset → env_class exactly as main.py does.
Outputs a confusion matrix and per-class disagreement %.

NOTE: This is NOT a fix-the-rule task. Disagreements are evidence of rule-layer fragility,
motivating the ML compositor (compose_env_from_ml_top2).

Usage:
    python tools/check_dispatch_consistency.py \\
        --manifest data/eval/nir_val/manifest_v2.csv \\
        --out docs/tables/iqa/dispatch_consistency.md \\
        --dry-run
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smartbinocular.env_presets import (
    auto_rule_preset_to_env_class,
    infer_env_tags_auto_rule,
    select_env_preset_from_tags,
)
from smartbinocular.nir_pipeline import (
    _nir_gray_std_quick,
    _nir_mean_brightness_bgr,
    nir_highlight_need_compress,
)

_STD_LOW  = 20.0  # env_auto_nir_gray_std_low  default from config.py
_STD_HIGH = 52.0  # env_auto_nir_gray_std_high default from config.py


def _load_manifest(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _infer_env_class(bgr: np.ndarray) -> str:
    """Mirror the runtime rule path: compute signals → infer_env_tags_auto_rule → env_class."""
    try:
        nir_b_ema   = _nir_mean_brightness_bgr(bgr, subsample=4)
        nir_gray_std = _nir_gray_std_quick(bgr)
        glare_nir   = nir_highlight_need_compress(bgr)
        tags = infer_env_tags_auto_rule(
            nir_b_ema=nir_b_ema,
            nir_gray_std=nir_gray_std,
            glare_nir=glare_nir,
            haze_config_on=False,
            std_low=_STD_LOW,
            std_high=_STD_HIGH,
        )
        preset = select_env_preset_from_tags(tags)
        return auto_rule_preset_to_env_class(preset)
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="docs/tables/iqa/dispatch_consistency.md")
    parser.add_argument("--dry-run", action="store_true", help="First 10 images only")
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    if not manifest:
        print(f"No rows in {args.manifest}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        manifest = manifest[:10]
        print(f"[dry-run] {len(manifest)} images", file=sys.stderr)

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total = 0
    agree = 0
    errors = 0

    for entry in manifest:
        img_path = entry["path"]
        manifest_label = entry.get("env_class", "unknown")

        bgr = cv.imread(img_path, cv.IMREAD_COLOR)
        if bgr is None:
            print(f"  SKIP: {img_path}", file=sys.stderr)
            errors += 1
            continue

        rule_label = _infer_env_class(bgr)
        confusion[manifest_label][rule_label] += 1
        total += 1
        if rule_label == manifest_label:
            agree += 1

    if total == 0:
        print("No images processed.", file=sys.stderr)
        sys.exit(1)

    agree_pct = 100.0 * agree / total
    print(f"\n[check_dispatch_consistency] Agreement: {agree}/{total} = {agree_pct:.1f}%", file=sys.stderr)

    all_manifest = sorted(set(confusion.keys()))
    all_rule = sorted({r for d in confusion.values() for r in d.keys()})

    lines = [
        "# Dispatch Consistency — Rule vs Manifest\n",
        f"**Source:** `{args.manifest}`  ",
        f"**n:** {total} images, {errors} skipped  ",
        f"**Overall agreement:** {agree}/{total} = {agree_pct:.1f}%  ",
        "",
        "Rule signals: `nir_b_ema` (`_nir_mean_brightness_bgr`), `nir_gray_std`"
        " (`_nir_gray_std_quick`), `glare_nir` (`nir_highlight_need_compress`).",
        "Same code path as runtime — `haze_config_on=False` for offline eval.",
        "",
        "**Important:** Disagreements are evidence of rule-layer fragility, not a bug to fix.",
        "The ML compositor (`compose_env_from_ml_top2`) exists precisely to cover this gap.",
        "",
        "## Confusion Matrix (rows=manifest label, cols=rule-inferred label)",
        "",
    ]

    header_cols = all_rule
    lines.append("| Manifest \\ Rule | " + " | ".join(header_cols) + " | agree% |")
    lines.append("|" + "-" * 16 + "|" + "|".join("-" * (len(c) + 2) for c in header_cols) + "|--------|")

    for ml in all_manifest:
        row_total = sum(confusion[ml].values())
        cells = [str(confusion[ml].get(rl, 0)) for rl in header_cols]
        diag = confusion[ml].get(ml, 0)
        pct = f"{100.0*diag/row_total:.0f}%" if row_total else "—"
        lines.append("| " + ml + " | " + " | ".join(cells) + f" | {pct} |")

    lines.extend(["", "## Dispatch-fragile classes (agreement < 80%)", ""])
    fragile: list[str] = []
    for ml in all_manifest:
        row_total = sum(confusion[ml].values())
        if row_total == 0:
            continue
        diag = confusion[ml].get(ml, 0)
        pct = 100.0 * diag / row_total
        if pct < 80.0:
            fragile.append(f"- **{ml}**: {diag}/{row_total} = {pct:.1f}% agreement")

    if fragile:
        lines.extend(fragile)
        lines.append("")
        lines.append(
            "> These classes are dispatch-fragile. Thesis §8: production dispatch quality"
            " depends on the ML compositor, not the rule layer alone."
        )
    else:
        lines.append("None — all classes ≥ 80% agreement.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
