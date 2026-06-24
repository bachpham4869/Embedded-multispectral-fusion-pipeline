"""build_manifest_v2.py — Expand NIR val manifest from 145 → ~280 images.

Samples from existing data/weather/ subtrees with a fixed seed, computes
per-image brightness (mean L in LAB) and pct_saturated, tags edge-case buckets,
and emits manifest_v2.csv with columns:
    path, env_class, edge_case, source_dir, mean_L, pct_sat, sha256

Nine env classes (6 original + 3 new):
    night_clear, normal_night, nir_night, fog, glare, backlight,
    rain (NEW), normal_day (NEW), mixed_edge (NEW)

Usage:
    python tools/build_manifest_v2.py \\
        --root data/weather \\
        --out data/eval/nir_val/manifest_v2.csv \\
        --seed 42

    # Dry run (only compute, print stats, write nothing):
    python tools/build_manifest_v2.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from pathlib import Path
from typing import Optional

import cv2 as cv
import numpy as np

# Ensure tools/_iqa_metrics is importable.
sys.path.insert(0, str(Path(__file__).parent))
from _iqa_metrics import compute_iqa_metrics  # type: ignore[import]

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Quota: (current, target) per env_class — matches Phase 3 v2 plan table.
_QUOTAS: dict[str, int] = {
    "night_clear":  40,
    "normal_night": 40,
    "nir_night":    40,
    "fog":          30,
    "glare":        30,
    "backlight":    25,
    "rain":         20,
    "normal_day":   20,
    "mixed_edge":   25,
}

# Existing manifest to avoid re-sampling the same images.
_EXISTING_MANIFEST = Path("data/eval/nir_val/manifest.csv")


def _all_images(directory: Path) -> list[Path]:
    return [p for p in directory.rglob("*") if p.suffix.lower() in _IMG_EXTS]


def _sha256(p: Path) -> str:
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return h


def _image_stats(p: Path) -> Optional[dict]:
    bgr = cv.imread(str(p), cv.IMREAD_COLOR)
    if bgr is None:
        return None
    lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
    mean_L = float(lab[:, :, 0].mean())
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    metrics = compute_iqa_metrics(gray)
    return {"mean_L": round(mean_L, 2), "pct_sat": round(metrics["pct_saturated"], 5)}


def _edge_case_tag(mean_L: float, pct_sat: float) -> str:
    if mean_L < 15:
        return "extreme_dark"
    if pct_sat > 0.20:
        return "extreme_bright"
    if 0.03 <= pct_sat <= 0.08:
        return "borderline_glare"
    return ""


def _load_existing_paths(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    with open(manifest_path, newline="") as f:
        return {row["path"] for row in csv.DictReader(f)}


def _sample_class(
    candidates: list[Path],
    n: int,
    rng: random.Random,
    existing: set[str],
    preselect_fn=None,
) -> list[Path]:
    """Sample up to n images, excluding already-existing paths."""
    pool = [p for p in candidates if str(p) not in existing]
    if preselect_fn:
        pool = [p for p in pool if preselect_fn(p)]
    rng.shuffle(pool)
    return pool[:n]


def build_manifest(root: Path, seed: int, dry_run: bool) -> list[dict]:
    rng = random.Random(seed)
    existing = _load_existing_paths(_EXISTING_MANIFEST)

    # Load existing manifest rows as baseline (classes that already have images).
    baseline_rows: list[dict] = []
    if _EXISTING_MANIFEST.exists():
        with open(_EXISTING_MANIFEST, newline="") as f:
            for row in csv.DictReader(f):
                p = Path(row["path"])
                stats = _image_stats(p) if p.exists() else None
                mean_L = stats["mean_L"] if stats else 0.0
                pct_sat = stats["pct_sat"] if stats else 0.0
                baseline_rows.append({
                    "path": row["path"],
                    "env_class": row["env_class"],
                    "edge_case": _edge_case_tag(mean_L, pct_sat),
                    "source_dir": str(p.parent.relative_to(root) if root in p.parents else p.parent),
                    "mean_L": mean_L,
                    "pct_sat": pct_sat,
                    "sha256": _sha256(p) if p.exists() else "",
                })

    # Count how many we already have per class.
    have: dict[str, int] = {}
    for r in baseline_rows:
        have[r["env_class"]] = have.get(r["env_class"], 0) + 1

    new_rows: list[dict] = []

    # night_clear — add hard-dark (mean_L < 30) from darkface.
    darkface = root / "darkface" / "image"
    need = _QUOTAS["night_clear"] - have.get("night_clear", 0)
    if need > 0:
        candidates = _all_images(darkface)
        sampled = _sample_class(
            candidates, need, rng, existing,
            preselect_fn=lambda p: (cv.cvtColor(cv.imread(str(p), cv.IMREAD_COLOR), cv.COLOR_BGR2LAB)[:, :, 0].mean() < 30)
            if cv.imread(str(p), cv.IMREAD_COLOR) is not None else False,
        )
        for p in sampled:
            stats = _image_stats(p)
            if stats:
                new_rows.append({
                    "path": str(p), "env_class": "night_clear",
                    "edge_case": _edge_case_tag(stats["mean_L"], stats["pct_sat"]),
                    "source_dir": "darkface/image", **stats, "sha256": _sha256(p),
                })

    # normal_night — ExDark non-car categories.
    need = _QUOTAS["normal_night"] - have.get("normal_night", 0)
    if need > 0:
        exdark_cats = ["Bicycle", "Boat", "Cup", "Dog", "Cat"]
        candidates = []
        for cat in exdark_cats:
            candidates.extend(_all_images(root / "ExDark" / cat))
        sampled = _sample_class(candidates, need, rng, existing)
        for p in sampled:
            stats = _image_stats(p)
            if stats:
                new_rows.append({
                    "path": str(p), "env_class": "normal_night",
                    "edge_case": _edge_case_tag(stats["mean_L"], stats["pct_sat"]),
                    "source_dir": "ExDark/mixed", **stats, "sha256": _sha256(p),
                })

    # nir_night — gray/, mid-brightness 60–110 to stress B vs A boundary.
    need = _QUOTAS["nir_night"] - have.get("nir_night", 0)
    if need > 0:
        candidates = _all_images(root / "gray")
        sampled = _sample_class(
            candidates, need, rng, existing,
            preselect_fn=lambda p: (
                (bgr := cv.imread(str(p), cv.IMREAD_COLOR)) is not None
                and 60 <= cv.cvtColor(bgr, cv.COLOR_BGR2LAB)[:, :, 0].mean() <= 110
            ),
        )
        for p in sampled:
            stats = _image_stats(p)
            if stats:
                new_rows.append({
                    "path": str(p), "env_class": "nir_night",
                    "edge_case": _edge_case_tag(stats["mean_L"], stats["pct_sat"]),
                    "source_dir": "gray", **stats, "sha256": _sha256(p),
                })

    # fog — weather11/fogsmog.
    need = _QUOTAS["fog"] - have.get("fog", 0)
    if need > 0:
        candidates = _all_images(root / "weather11" / "dataset" / "fogsmog")
        sampled = _sample_class(candidates, need, rng, existing)
        for p in sampled:
            stats = _image_stats(p)
            if stats:
                new_rows.append({
                    "path": str(p), "env_class": "fog",
                    "edge_case": _edge_case_tag(stats["mean_L"], stats["pct_sat"]),
                    "source_dir": "weather11/fogsmog", **stats, "sha256": _sha256(p),
                })

    # glare — glare/real/input/, preselect pct_sat > 0.05.
    need = _QUOTAS["glare"] - have.get("glare", 0)
    if need > 0:
        candidates = _all_images(root / "glare" / "real" / "input")
        sampled = _sample_class(
            candidates, need, rng, existing,
            preselect_fn=lambda p: (
                (bgr := cv.imread(str(p), cv.IMREAD_COLOR)) is not None
                and (bgr > 250).mean() > 0.03
            ),
        )
        for p in sampled:
            stats = _image_stats(p)
            if stats:
                new_rows.append({
                    "path": str(p), "env_class": "glare",
                    "edge_case": _edge_case_tag(stats["mean_L"], stats["pct_sat"]),
                    "source_dir": "glare/real/input", **stats, "sha256": _sha256(p),
                })

    # backlight.
    need = _QUOTAS["backlight"] - have.get("backlight", 0)
    if need > 0:
        candidates = _all_images(root / "backlight")
        sampled = _sample_class(candidates, need, rng, existing)
        for p in sampled:
            stats = _image_stats(p)
            if stats:
                new_rows.append({
                    "path": str(p), "env_class": "backlight",
                    "edge_case": _edge_case_tag(stats["mean_L"], stats["pct_sat"]),
                    "source_dir": "backlight", **stats, "sha256": _sha256(p),
                })

    # rain (NEW) — weather11/rain.
    need = _QUOTAS["rain"] - have.get("rain", 0)
    if need > 0:
        candidates = _all_images(root / "weather11" / "dataset" / "rain")
        sampled = _sample_class(candidates, need, rng, existing)
        for p in sampled:
            stats = _image_stats(p)
            if stats:
                new_rows.append({
                    "path": str(p), "env_class": "rain",
                    "edge_case": _edge_case_tag(stats["mean_L"], stats["pct_sat"]),
                    "source_dir": "weather11/rain", **stats, "sha256": _sha256(p),
                })

    # normal_day (NEW) — daylight images: dew, frost, rainbow.
    need = _QUOTAS["normal_day"] - have.get("normal_day", 0)
    if need > 0:
        day_dirs = ["dew", "frost", "rainbow"]
        candidates = []
        for d in day_dirs:
            candidates.extend(_all_images(root / "weather11" / "dataset" / d))
        sampled = _sample_class(candidates, need, rng, existing)
        for p in sampled:
            stats = _image_stats(p)
            if stats:
                new_rows.append({
                    "path": str(p), "env_class": "normal_day",
                    "edge_case": _edge_case_tag(stats["mean_L"], stats["pct_sat"]),
                    "source_dir": "weather11/dew+frost+rainbow", **stats, "sha256": _sha256(p),
                })

    # mixed_edge (NEW) — borderline scenes from multiple sources.
    need = _QUOTAS["mixed_edge"] - have.get("mixed_edge", 0)
    if need > 0:
        # Pull from glare (borderline_glare) + darkface (extreme_dark) + fog (borderline_fog).
        candidates_me: list[tuple[Path, str]] = []
        for p in _all_images(root / "glare" / "real" / "input"):
            bgr = cv.imread(str(p), cv.IMREAD_COLOR)
            if bgr is not None:
                pct = (bgr > 250).mean()
                if 0.02 <= pct <= 0.10:  # borderline glare
                    candidates_me.append((p, "glare/real/input"))
        for p in _all_images(root / "darkface" / "image"):
            bgr = cv.imread(str(p), cv.IMREAD_COLOR)
            if bgr is not None:
                lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
                if 15 <= lab[:, :, 0].mean() <= 35:  # dark but not extreme_dark
                    candidates_me.append((p, "darkface/image"))

        rng.shuffle(candidates_me)
        for p, src in candidates_me[:need]:
            if str(p) in existing:
                continue
            stats = _image_stats(p)
            if stats:
                tag = _edge_case_tag(stats["mean_L"], stats["pct_sat"])
                if not tag:
                    tag = "mixed_lighting"
                new_rows.append({
                    "path": str(p), "env_class": "mixed_edge",
                    "edge_case": tag,
                    "source_dir": src, **stats, "sha256": _sha256(p),
                })

    all_rows = baseline_rows + new_rows

    # Print class counts.
    counts: dict[str, int] = {}
    for r in all_rows:
        counts[r["env_class"]] = counts.get(r["env_class"], 0) + 1
    print("[build_manifest_v2] Class distribution:", file=sys.stderr)
    for cls, tgt in _QUOTAS.items():
        n = counts.get(cls, 0)
        status = "OK" if n >= tgt else f"SHORT by {tgt - n}"
        print(f"  {cls:15s}: {n:3d} / {tgt}  {status}", file=sys.stderr)
    print(f"  TOTAL: {len(all_rows)}", file=sys.stderr)

    return all_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/weather", help="Root data directory")
    parser.add_argument("--out", default="data/eval/nir_val/manifest_v2.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Print stats, do not write")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: root directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    rows = build_manifest(root, args.seed, args.dry_run)

    if args.dry_run:
        print("[dry-run] Not writing output.", file=sys.stderr)
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path", "env_class", "edge_case", "source_dir", "mean_L", "pct_sat", "sha256"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
