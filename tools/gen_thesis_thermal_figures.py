#!/usr/bin/env python3
"""Generate thermal pipeline stage panels for thesis (Ch2 / Ch5 style).

Loads scaled MI48-like PNGs (default ``data/thermal/scaled_mi48_80x62``), converts to
grayscale 62×80, feeds a short **temporal** sequence through :class:`ThermalProcessor`
(so 3DNR + background model behave like live capture), then saves a before/after grid:

  Raw → 3DNR denoised → display path (AGC + edge, false-color LUT) → heat map (LUT)

Outputs:
  - ``docs/thesis_eval/thermal/figures/thermal_pipeline_stages.png`` (tracked path)
  - ``Thesis_report/figures/ch2_background/thermal_pipeline_stages.png`` (created via mkdir+copy)

Example::

    uv run python tools/gen_thesis_thermal_figures.py
    uv run python tools/gen_thesis_thermal_figures.py --rows 2 --sequence-length 48 --seed 7

Depends: OpenCV, matplotlib, editable ``smartbinocular`` package on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

PROJ = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJ / "data/thermal/scaled_mi48_80x62"
DOC_OUT = PROJ / "docs/thesis_eval/thermal/figures/thermal_pipeline_stages.png"
THESIS_OUT = PROJ / "Thesis_report/figures/ch2_background/thermal_pipeline_stages.png"

sys.path.insert(0, str(PROJ / "src"))

from smartbinocular.config import CONFIG  # noqa: E402
from smartbinocular.hardware import gray_to_thermal_bgr  # noqa: E402
from smartbinocular.thermal_pipeline import ThermalProcessor  # noqa: E402


def load_mi48_gray(path: Path) -> np.ndarray:
    """Return single-channel uint8 thermal frame at MI48 resolution (62×80)."""
    bgr = cv.imread(str(path), cv.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    if gray.shape != (62, 80):
        gray = cv.resize(gray, (80, 62), interpolation=cv.INTER_AREA)
    return gray


def inject_noise(
    gray: np.ndarray,
    *,
    seed: int = 42,
    noise_sigma: float = 18.0,
    fpn_sigma: float = 8.0,
    salt_pepper_prob: float = 0.002,
) -> np.ndarray:
    """Inject realistic thermal noise into the grayscale frame.

    1. Temporal/Spatial Gaussian noise: simulates detector thermal noise (NETD).
    2. Fixed-Pattern Noise (FPN): static vertical and horizontal stripes.
    3. Salt-and-Pepper noise: simulates dead/hot sensor pixels.
    """
    h, w = gray.shape
    rng = np.random.default_rng(seed)

    noisy = gray.astype(np.float32)

    # Static FPN offsets generated using a deterministic seed offset
    rng_fpn = np.random.default_rng(seed // 1000 * 1000 + 999)
    col_offsets = rng_fpn.normal(0, fpn_sigma, size=(1, w))
    row_offsets = rng_fpn.normal(0, fpn_sigma * 0.5, size=(h, 1))
    noisy = noisy + col_offsets + row_offsets

    # Temporal Gaussian noise
    gauss = rng.normal(0, noise_sigma, size=(h, w))
    noisy = noisy + gauss

    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    # Salt & Pepper
    if salt_pepper_prob > 0:
        num_salt = int(np.ceil(salt_pepper_prob * h * w * 0.5))
        coords_salt = [rng.integers(0, i, num_salt) for i in (h, w)]
        noisy[coords_salt[0], coords_salt[1]] = 255

        num_pepper = int(np.ceil(salt_pepper_prob * h * w * 0.5))
        coords_pepper = [rng.integers(0, i, num_pepper) for i in (h, w)]
        noisy[coords_pepper[0], coords_pepper[1]] = 0

    return noisy


def iter_gray_sequence(
    paths: list[Path],
    start: int,
    length: int,
    *,
    inject_noise_flag: bool = False,
    noise_sigma: float = 18.0,
    fpn_sigma: float = 8.0,
    salt_pepper_prob: float = 0.002,
    base_seed: int = 42,
) -> list[np.ndarray]:
    if not paths:
        raise ValueError("No image paths")
    seq: list[np.ndarray] = []
    for i in range(length):
        gray = load_mi48_gray(paths[(start + i) % len(paths)])
        if inject_noise_flag:
            gray = inject_noise(
                gray,
                seed=base_seed + i,
                noise_sigma=noise_sigma,
                fpn_sigma=fpn_sigma,
                salt_pepper_prob=salt_pepper_prob,
            )
        seq.append(gray)
    return seq


def build_processor() -> ThermalProcessor:
    """Match ``main.py`` thermal stack defaults, driven by ``CONFIG`` where relevant."""
    return ThermalProcessor(
        warmup=40,
        anti_glare=True,
        display_anti_glare=bool(CONFIG.get("thermal_display_glare_suppression", True)),
        thermal_high_pct=float(CONFIG.get("thermal_high_pct", 99.0)),
        thermal_saturate_at=float(CONFIG.get("thermal_saturate_at", 240.0)),
        thermal_gamma=float(CONFIG.get("thermal_gamma", 0.74)),
        thermal_floor=float(CONFIG.get("thermal_floor", 3.0)),
        thermal_fg_threshold=float(CONFIG.get("thermal_fg_threshold", 18.0)),
        thermal_fg_max_ratio=float(CONFIG.get("thermal_fg_max_ratio", 0.5)),
        thermal_edge_strength=float(CONFIG.get("thermal_edge_strength", 0.2)),
        thermal_agc_low_pct=float(CONFIG.get("thermal_agc_low_pct", 2.0)),
        thermal_agc_high_pct=float(CONFIG.get("thermal_agc_high_pct", 98.0)),
        detail_preserve_detect=False,
        thermal_3dnr_alpha=float(CONFIG.get("thermal_3dnr_alpha", 0.65)),
        bilateral_display_enable=bool(CONFIG.get("thermal_bilateral_display_enable", False)),
        bilateral_d=int(CONFIG.get("thermal_bilateral_d", 5)),
        bilateral_sigma_color=float(CONFIG.get("thermal_bilateral_sigma_color", 15.0)),
        bilateral_sigma_space=float(CONFIG.get("thermal_bilateral_sigma_space", 5.0)),
    )


def run_sequence(proc: ThermalProcessor, frames: list[np.ndarray]) -> tuple[np.ndarray, ...]:
    denoised = enhanced = heat = fg = None
    for raw in frames:
        denoised, enhanced, heat, fg = proc.process(raw, compute_enhanced=True)
    assert denoised is not None and enhanced is not None
    return denoised, enhanced, heat, fg


def upscale(img: np.ndarray, scale: int) -> np.ndarray:
    h, w = img.shape[:2]
    return cv.resize(img, (w * scale, h * scale), interpolation=cv.INTER_NEAREST)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT, help="Folder of scaled MI48 PNGs")
    ap.add_argument(
        "--glob",
        default="*thermal_image_dataset*.png",
        help="Glob filter under input-dir (quoted in shell)",
    )
    ap.add_argument("--rows", type=int, default=3, help="Number of example sequences")
    ap.add_argument(
        "--sequence-length",
        type=int,
        default=48,
        help="Frames fed per row (must exceed BG warmup ~40 for meaningful heat map)",
    )
    ap.add_argument("--seed", type=int, default=42, help="RNG for choosing sequence starts")
    ap.add_argument("--scale", type=int, default=5, help="Nearest-neighbor upscale for display")
    ap.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    ap.add_argument(
        "--inject-noise",
        action="store_true",
        help="Programmatically inject realistic thermal sensor noise (Gaussian + FPN + Salt-and-Pepper).",
    )
    ap.add_argument("--noise-sigma", type=float, default=18.0)
    ap.add_argument("--fpn-sigma", type=float, default=8.0)
    ap.add_argument("--salt-pepper-prob", type=float, default=0.002)
    args = ap.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib required", file=sys.stderr)
        return 1

    inp = args.input_dir.expanduser().resolve()
    if not inp.is_dir():
        print(f"ERROR: not a directory: {inp}", file=sys.stderr)
        return 1

    paths = sorted(inp.glob(args.glob))
    if len(paths) < 2:
        paths = sorted(p for p in inp.glob("*.png") if p.is_file())
    if len(paths) < 2:
        print(f"ERROR: need at least 2 PNGs under {inp}", file=sys.stderr)
        return 1

    # Fix sorting for non-zero-padded filenames like image_1.png, image_10.png
    try:
        if all(p.stem.startswith("image_") for p in paths):
            paths = sorted(paths, key=lambda p: int(p.stem.split("_")[1]))
    except Exception:
        pass

    rng = np.random.default_rng(args.seed)
    cmap_lv = int(CONFIG.get("thermal_colormap_levels", 256))
    if cmap_lv not in (32, 64, 128, 256):
        cmap_lv = 256

    n = len(paths)
    starts = rng.choice(np.arange(n), size=args.rows, replace=(args.rows > n)).astype(int).tolist()

    titles = ("Raw (single channel)", "After 3DNR", "Display path (AGC + edge, LUT)", "Heat map (LUT)")
    fig, axes = plt.subplots(args.rows, len(titles), figsize=(14, 3.6 * args.rows))
    if args.rows == 1:
        axes = np.asarray([axes])

    for row, start in enumerate(starts[: args.rows]):
        seq = iter_gray_sequence(
            paths,
            int(start),
            args.sequence_length,
            inject_noise_flag=args.inject_noise,
            noise_sigma=args.noise_sigma,
            fpn_sigma=args.fpn_sigma,
            salt_pepper_prob=args.salt_pepper_prob,
            base_seed=args.seed,
        )
        proc = build_processor()
        denoised, enhanced, heat, _fg = run_sequence(proc, seq)
        raw_last = seq[-1]

        disp_bgr = gray_to_thermal_bgr(enhanced, levels=cmap_lv)
        heat_u8 = heat if heat is not None else np.zeros_like(denoised)
        heat_bgr = gray_to_thermal_bgr(heat_u8, levels=cmap_lv)

        panels = (
            raw_last,
            denoised,
            cv.cvtColor(disp_bgr, cv.COLOR_BGR2RGB),
            cv.cvtColor(heat_bgr, cv.COLOR_BGR2RGB),
        )
        for col, (title, panel) in enumerate(zip(titles, panels)):
            ax = axes[row, col]
            show = panel
            if panel.ndim == 2:
                show = upscale(panel, args.scale)
                ax.imshow(show, cmap="gray", vmin=0, vmax=255, aspect="equal")
            else:
                show = upscale(panel, args.scale)
                ax.imshow(show, aspect="equal")
            if row == 0:
                ax.set_title(title, fontsize=10)
            ax.axis("off")

    fig.suptitle(
        "Thermal pipeline stages (MI48-scale inputs; nearest-neighbor upscale for visibility)",
        fontsize=13,
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    DOC_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DOC_OUT, dpi=args.dpi, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"✅ Saved: {DOC_OUT}")

    THESIS_OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOC_OUT, THESIS_OUT)
    print(f"✅ Copied to: {THESIS_OUT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
