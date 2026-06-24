#!/usr/bin/env python3
"""Resize diverse thermal image folders under ``data/thermal`` to MI48-native 80×62.

Walks immediate subfolders of the input root (each treated as a *source* for stratified
sampling), collects common image extensions, normalizes to **8-bit BGR** (keeps false-color
/ RGB data; grayscale sources become 3-channel via ``GRAY2BGR``), resizes with
``INTER_AREA`` (good for downscaling), and writes PNGs plus a manifest CSV.

Default output stays inside ``data/thermal/...`` (gitignored) unless ``--output`` is set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _iter_image_files(root: Path) -> Iterator[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        if p.name.startswith("."):
            continue
        yield p


def _sources_under(root: Path, skip_names: frozenset[str]) -> dict[str, list[Path]]:
    """Group files by first-level subdirectory name under ``root``."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("."):
            continue
        if child.name in skip_names:
            continue
        for p in _iter_image_files(child):
            groups[child.name].append(p)
    return dict(groups)


def _round_robin_pick(
    groups: dict[str, list[Path]], target: int, rng: random.Random
) -> list[tuple[str, Path]]:
    """Pick up to ``target`` paths, cycling sources so each contributes evenly."""
    names = sorted(groups.keys())
    buckets = {n: rng.sample(groups[n], k=len(groups[n])) for n in names}
    ptr = {n: 0 for n in names}
    out: list[tuple[str, Path]] = []
    # Deterministic source order cycle
    order = names[:]
    while len(out) < target:
        made_progress = False
        for name in order:
            if len(out) >= target:
                break
            i = ptr[name]
            if i >= len(buckets[name]):
                continue
            out.append((name, buckets[name][i]))
            ptr[name] = i + 1
            made_progress = True
        if not made_progress:
            break
    return out


def _to_bgr_u8(img: np.ndarray) -> np.ndarray:
    """OpenCV ``imread`` uses BGR; keep color. Alpha dropped via ``BGRA2BGR``."""
    x = img
    if x.dtype == np.float32 or x.dtype == np.float64:
        x = np.clip(x, 0.0, 1.0)
        x = (x * 255.0).astype(np.uint8)
    elif x.dtype == np.uint16:
        xf = x.astype(np.float32)
        lo, hi = float(xf.min()), float(xf.max())
        if hi <= lo:
            x = np.zeros(x.shape, dtype=np.uint8)
        else:
            x = np.clip((xf - lo) / (hi - lo) * 255.0, 0.0, 255.0).astype(np.uint8)
    if x.ndim == 2:
        return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    if x.shape[2] == 3:
        bgr = x
    elif x.shape[2] == 4:
        bgr = cv2.cvtColor(x, cv2.COLOR_BGRA2BGR)
    else:
        raise ValueError(f"Unsupported channel count {x.shape}")
    if bgr.dtype != np.uint8:
        return np.clip(bgr, 0, 255).astype(np.uint8)
    return bgr


def _resize_mi48(bgr: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("data/thermal"),
        help="Root folder containing one subfolder per dataset source",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("data/thermal/scaled_mi48_80x62"),
        help="Output directory for PNGs + manifest",
    )
    ap.add_argument("--target", type=int, default=1000, help="Max images to export")
    ap.add_argument("--width", type=int, default=80, help="Output width (pixels)")
    ap.add_argument("--height", type=int, default=62, help="Output height (pixels)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for sampling")
    ap.add_argument(
        "--skip-dir",
        action="append",
        default=[],
        metavar="NAME",
        help="First-level subfolder under --input to ignore (repeatable)",
    )
    args = ap.parse_args()

    root: Path = args.input.expanduser().resolve()
    if not root.is_dir():
        print(f"Input not a directory: {root}", file=sys.stderr)
        return 1

    out_dir: Path = args.output.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    skip_names = frozenset(args.skip_dir)
    if out_dir.parent == root:
        skip_names = frozenset(set(skip_names) | {out_dir.name})

    groups = _sources_under(root, skip_names)
    if not groups:
        print(f"No dataset subfolders with images under {root}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    pairs = _round_robin_pick(groups, args.target, rng)
    if not pairs:
        print("No files selected.", file=sys.stderr)
        return 1

    manifest = out_dir / "manifest.csv"
    n_ok = 0
    with manifest.open("w", newline="") as f:
        mw = csv.writer(f)
        mw.writerow(
            [
                "out_png",
                "source_set",
                "src_relpath",
                "orig_w",
                "orig_h",
                "orig_dtype",
                "orig_channels",
            ]
        )
        for idx, (src_name, src_path) in enumerate(pairs):
            rel = src_path.relative_to(root)
            im = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
            if im is None:
                print(f"skip unreadable: {src_path}", file=sys.stderr)
                continue
            if im.ndim == 2:
                oh, ow = im.shape
                och = 1
            else:
                oh, ow = im.shape[:2]
                och = im.shape[2]
            try:
                bgr8 = _to_bgr_u8(im)
            except ValueError as e:
                print(f"skip {src_path}: {e}", file=sys.stderr)
                continue
            small = _resize_mi48(bgr8, args.width, args.height)
            hsh = hashlib.sha1(str(rel).encode("utf-8")).hexdigest()[:10]
            out_name = f"{idx:05d}_{src_name}_{hsh}.png"
            out_p = out_dir / out_name
            if not cv2.imwrite(str(out_p), small):
                print(f"skip write fail: {out_p}", file=sys.stderr)
                continue
            mw.writerow(
                [
                    out_name,
                    src_name,
                    str(rel),
                    ow,
                    oh,
                    im.dtype.name,
                    och,
                ]
            )
            n_ok += 1

    total_files = sum(len(v) for v in groups.values())
    print(f"Sources: {len(groups)}  files_seen≈{total_files}  written={n_ok} → {out_dir}")
    for name, paths in sorted(groups.items(), key=lambda x: -len(x[1])):
        print(f"  - {name}: {len(paths)} files")
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
