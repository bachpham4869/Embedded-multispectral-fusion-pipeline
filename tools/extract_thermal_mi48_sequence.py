#!/usr/bin/env python3
"""Extract consecutive thermal frames from video dataset folders → MI48-sized PNG sequence.

Sorts frames by numeric suffix in filename (e.g. ``..._8.jpg``, ``..._15.jpg``).
Writes ``seq_00000.png``, ``seq_00001.png``, … for use with
``tools/gen_thesis_thermal_figures.py --glob 'seq_*.png'``.

**Auto mode (no ``--clip-dir``):** leaf clip folders often contain only ~10 frames.
The script picks the ``FramesClass*`` directory (under ``video_dataset10`` or
``video_dataset5``) with the **largest total frame count**, then concatenates all
clips in lexical clip-folder order (frames sorted inside each clip).

Typical roots (``--root``, repeatable; defaults built-in)::

    data/thermal/10393655/video_dataset10
    data/thermal/10393655/video_dataset5/video_dataset5

Example::

    uv run python tools/extract_thermal_mi48_sequence.py --max-frames 120
    uv run python tools/extract_thermal_mi48_sequence.py \\
        --clip-dir data/thermal/10393655/video_dataset10/FramesClass1/SomeTake
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import cv2
import numpy as np

PROJ = Path(__file__).resolve().parent.parent

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

_FRAME_NUM_RE = re.compile(r"_(\d+)\s*\.[^.]+$", re.IGNORECASE)


def _to_bgr_u8(img: np.ndarray) -> np.ndarray:
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


def frame_sort_key(path: Path) -> tuple[int, str]:
    m = _FRAME_NUM_RE.search(path.name)
    if m:
        return int(m.group(1)), path.name
    return -1, path.name


def list_clip_dirs(roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        root = root.expanduser().resolve()
        if not root.is_dir():
            continue
        for fc in sorted(root.glob("FramesClass*")):
            if not fc.is_dir():
                continue
            for clip in sorted(fc.iterdir()):
                if clip.is_dir():
                    out.append(clip)
    return out


def count_images(clip: Path) -> int:
    return sum(1 for p in clip.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def list_sorted_frames(clip: Path) -> list[Path]:
    imgs = [p for p in clip.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(imgs, key=frame_sort_key)


def pick_best_clip(roots: list[Path], min_frames: int) -> Path | None:
    """Largest single leaf clip with at least ``min_frames`` images."""
    best: Path | None = None
    best_n = 0
    for clip in list_clip_dirs(roots):
        n = count_images(clip)
        if n >= min_frames and n > best_n:
            best = clip
            best_n = n
    return best


def iter_frames_class_dirs(roots: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for root in roots:
        root = root.expanduser().resolve()
        if not root.is_dir():
            continue
        for fc in sorted(root.glob("FramesClass*")):
            if fc.is_dir():
                rp = fc.resolve()
                if rp not in seen:
                    seen.add(rp)
                    out.append(fc)
    return out


def total_images_in_frames_class(fc_dir: Path) -> int:
    total = 0
    for clip in fc_dir.iterdir():
        if clip.is_dir():
            total += count_images(clip)
    return total


def merged_sorted_frames(fc_dir: Path) -> list[Path]:
    """All clips under a FramesClass*, clip order lexical, frames sorted per clip."""
    out: list[Path] = []
    for clip in sorted(p for p in fc_dir.iterdir() if p.is_dir()):
        out.extend(list_sorted_frames(clip))
    return out


def pick_best_frames_class(roots: list[Path]) -> tuple[Path, int] | None:
    """FramesClass directory with maximum total images across its clips."""
    best: Path | None = None
    best_total = 0
    for fc in iter_frames_class_dirs(roots):
        total = total_images_in_frames_class(fc)
        if total > best_total:
            best_total = total
            best = fc
    if best is None:
        return None
    return best, best_total


def default_roots() -> list[Path]:
    base = PROJ / "data/thermal/10393655"
    return [
        base / "video_dataset10",
        base / "video_dataset5" / "video_dataset5",
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=Path,
        action="append",
        dest="roots",
        help="Folder containing FramesClass* (repeatable). Defaults to video_dataset10 + video_dataset5.",
    )
    ap.add_argument(
        "--clip-dir",
        type=Path,
        default=None,
        help="Use this clip folder directly (bypass auto-pick)",
    )
    ap.add_argument("--output", type=Path, default=PROJ / "data/thermal/scaled_mi48_sequences/auto_clip")
    ap.add_argument("--width", type=int, default=80)
    ap.add_argument("--height", type=int, default=62)
    ap.add_argument("--max-frames", type=int, default=96, help="Cap exported consecutive frames")
    ap.add_argument(
        "--min-frames",
        type=int,
        default=48,
        help="Auto mode: require at least this many source frames after merge (or single clip)",
    )
    ap.add_argument(
        "--single-clip-only",
        action="store_true",
        help="Pick one leaf clip with ≥ min-frames instead of merging a FramesClass",
    )
    args = ap.parse_args()

    roots = list(args.roots) if args.roots else default_roots()
    out_dir = args.output.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source_label: str

    if args.clip_dir is not None:
        clip = args.clip_dir.expanduser().resolve()
        if not clip.is_dir():
            print(f"ERROR: --clip-dir not a directory: {clip}", file=sys.stderr)
            return 1
        frames = list_sorted_frames(clip)
        source_label = str(clip)
    elif args.single_clip_only:
        clip = pick_best_clip(roots, args.min_frames)
        if clip is None:
            print(
                f"ERROR: No single clip under {roots} with ≥{args.min_frames} frames",
                file=sys.stderr,
            )
            return 1
        frames = list_sorted_frames(clip)
        source_label = str(clip)
    else:
        picked = pick_best_frames_class(roots)
        if picked is None:
            print(f"ERROR: No FramesClass* under {roots}", file=sys.stderr)
            return 1
        fc_dir, total_src = picked
        if total_src < args.min_frames:
            print(
                f"ERROR: Best FramesClass {fc_dir} has only {total_src} frames "
                f"(need ≥{args.min_frames})",
                file=sys.stderr,
            )
            return 1
        frames = merged_sorted_frames(fc_dir)
        source_label = f"{fc_dir} (merged_clips total_src={total_src})"

    if len(frames) < 2:
        print(f"ERROR: Too few images after resolution ({len(frames)})", file=sys.stderr)
        return 1

    take = frames[: args.max_frames]
    manifest = out_dir / "sequence_manifest.txt"

    n_ok = 0
    with manifest.open("w", encoding="utf-8") as mf:
        mf.write(f"source\t{source_label}\n")
        mf.write(f"frames_exported\t{len(take)}\n")
        for i, src in enumerate(take):
            im = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
            if im is None:
                continue
            try:
                bgr8 = _to_bgr_u8(im)
            except ValueError as e:
                print(f"skip {src}: {e}", file=sys.stderr)
                continue
            small = _resize_mi48(bgr8, args.width, args.height)
            out_name = f"seq_{i:05d}.png"
            out_p = out_dir / out_name
            if not cv2.imwrite(str(out_p), small):
                print(f"skip write fail: {out_p}", file=sys.stderr)
                continue
            mf.write(f"{out_name}\t{src.name}\n")
            n_ok += 1

    sig = hashlib.sha1(source_label.encode()).hexdigest()[:8]
    print(f"✅ Source: {source_label}")
    print(f"✅ Wrote {n_ok} frames → {out_dir} (sig={sig})")
    print(f"   manifest: {manifest}")
    return 0 if n_ok >= 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
