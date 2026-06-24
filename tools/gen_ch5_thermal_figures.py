#!/usr/bin/env python3
"""Generate Ch.~5 thermal comparison figures (profiles + ENV presets).

Produces two PNG grids from the same offline MI48 scaled PNG sequence, driven by the
real :class:`ThermalProcessor` stack with ``main.py``-parity construction:

1. **Profiles** — ``raw`` / ``throughput`` / ``quality`` via :func:`apply_profile`,
   ENV preset ``default`` (thermal tuning baseline).
2. **Presets** — ``default`` / ``night`` / ``fog`` / ``glare_heavy`` via
   :func:`merge_opt_cfg_with_preset` + :meth:`ThermalProcessor.apply_env_runtime`,
   fixed ``quality`` display profile so differences isolate preset thermal overrides.

By default the script **auto-picks** a 48-frame window that maximises motion + detail (easier to read than a random clip).
Upscale uses **bilinear** interpolation unless ``--nearest-upscale`` is passed.

Rows for each figure:
  - Row 0: display path (AGC + edge + optional glare), false-colour LUT
  - Row 1: background-subtracted heat map, same LUT

Outputs (also copied under ``Thesis_report/figures/ch5_implementation/``)::

    docs/thesis_eval/ch5_thermal/figures/thermal_profile_comparison.png
    docs/thesis_eval/ch5_thermal/figures/thermal_preset_comparison.png
    docs/thesis_eval/ch5_thermal/figures/thermal_alg_3dnr.png
    docs/thesis_eval/ch5_thermal/figures/thermal_alg_background.png
    docs/thesis_eval/ch5_thermal/figures/thermal_alg_mad.png

Example::

    uv run python tools/gen_ch5_thermal_figures.py
    uv run python tools/gen_ch5_thermal_figures.py --sequence-length 48 --seed 7
    uv run python tools/gen_ch5_thermal_figures.py --no-auto-start --start 120   # fixed clip
    uv run python tools/gen_ch5_thermal_figures.py --skip-algorithm-panels   # profiles/presets only

Depends: OpenCV, matplotlib, NumPy, editable ``smartbinocular`` on PYTHONPATH.

Note: Optical buckets A–F only dispatch NIR code paths; they do **not** switch a thermal
bucket table. Thermal thresholds and 3DNR blend weight follow stabilized ENV presets.
"""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

PROJ = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJ / "data/thermal/scaled_mi48_80x62"
DOC_DIR = PROJ / "docs/thesis_eval/ch5_thermal/figures"
OUT_PROFILES = DOC_DIR / "thermal_profile_comparison.png"
OUT_PRESETS = DOC_DIR / "thermal_preset_comparison.png"
OUT_ALG_3DNR = DOC_DIR / "thermal_alg_3dnr.png"
OUT_ALG_BG = DOC_DIR / "thermal_alg_background.png"
OUT_ALG_MAD = DOC_DIR / "thermal_alg_mad.png"
THESIS_DIR = PROJ / "Thesis_report/figures/ch5_implementation"

sys.path.insert(0, str(PROJ / "src"))

from smartbinocular.config import (  # noqa: E402
    CONFIG,
    apply_profile,
    resolve_optimization_profile,
)
from smartbinocular.env_presets import ENV_PRESETS, merge_opt_cfg_with_preset  # noqa: E402
from smartbinocular.hardware import gray_to_thermal_bgr  # noqa: E402
from smartbinocular.thermal_pipeline import (  # noqa: E402
    ThermalMADAnomalyDetector,
    ThermalProcessor,
)


def load_mi48_gray(path: Path) -> np.ndarray:
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
    seq = []
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


def _motion_and_detail_score(paths: list[Path], start: int, length: int) -> float:
    """Prefer clips with frame-to-frame motion + spatial detail (easier to read in thesis)."""
    seq = iter_gray_sequence(paths, start, length)
    motion = 0.0
    for i in range(1, len(seq)):
        motion += float(np.mean(cv.absdiff(seq[i], seq[i - 1])))
    lap = cv.Laplacian(seq[-1], cv.CV_64F)
    lap_var = float(lap.var())
    return motion * 3.0 + lap_var * 0.002 + float(np.std(seq[-1]))


def pick_best_sequence_start(
    paths: list[Path],
    length: int,
    *,
    n_candidates: int,
) -> tuple[int, float]:
    """Return (start_index, score) maximizing visibility heuristic."""
    n = len(paths)
    if n <= 1:
        return 0, 0.0
    best_s = -1.0
    best_i = 0
    span = max(1, n - length)
    nc = max(1, n_candidates)
    for k in range(nc):
        start = int(round(k * span / max(1, nc - 1))) if nc > 1 else 0
        if start + length > n:
            start = max(0, n - length)
        s = _motion_and_detail_score(paths, start, length)
        if s > best_s:
            best_s = s
            best_i = start
    return best_i, best_s


def make_opt_cfg_base(cfg: dict) -> dict:
    opt = resolve_optimization_profile(
        str(cfg.get("opt_profile", "handheld_pan")),
        bool(cfg.get("opt_haze_preset", False)),
        not bool(cfg.get("opt_disable_profiles", False)),
    )
    opt = dict(opt)
    opt["thermal_fg_max_ratio"] = float(
        np.clip(float(cfg.get("thermal_fg_max_ratio", 0.8)), 0.05, 0.95)
    )
    return opt


def build_thermal_proc(
    cfg: dict,
    opt_cfg_base: dict,
    *,
    use_kalman_background: bool = False,
) -> ThermalProcessor:
    """Match ``main.py`` ThermalProcessor(...) constructor."""
    return ThermalProcessor(
        warmup=40,
        anti_glare=True,
        use_kalman_background=use_kalman_background,
        display_anti_glare=bool(cfg.get("thermal_display_glare_suppression", True)),
        thermal_high_pct=float(opt_cfg_base["thermal_high_pct"]),
        thermal_saturate_at=float(opt_cfg_base["thermal_saturate_at"]),
        thermal_gamma=float(opt_cfg_base["thermal_gamma"]),
        thermal_floor=float(opt_cfg_base["thermal_floor"]),
        thermal_fg_threshold=float(opt_cfg_base["thermal_fg_threshold"]),
        thermal_fg_max_ratio=float(opt_cfg_base["thermal_fg_max_ratio"]),
        thermal_edge_strength=float(opt_cfg_base["thermal_edge_strength"]),
        thermal_agc_low_pct=float(opt_cfg_base["thermal_agc_low_pct"]),
        thermal_agc_high_pct=float(opt_cfg_base["thermal_agc_high_pct"]),
        detail_preserve_detect=bool(cfg.get("fix_thermal_detail_preserve", False)),
        detect_raw_mix=float(cfg.get("thermal_detect_raw_mix", 0.55)),
        detail_grad_threshold=float(cfg.get("thermal_detail_grad_threshold", 9.0)),
        detail_threshold_relax=float(cfg.get("thermal_detail_threshold_relax", 3.0)),
        detect_use_anti_glare=bool(cfg.get("thermal_detect_use_anti_glare", False)),
        thermal_3dnr_alpha=float(cfg.get("thermal_3dnr_alpha", 0.65)),
        bilateral_display_enable=bool(cfg.get("thermal_bilateral_display_enable", True)),
        bilateral_d=int(cfg.get("thermal_bilateral_d", 5)),
        bilateral_sigma_color=float(cfg.get("thermal_bilateral_sigma_color", 15.0)),
        bilateral_sigma_space=float(cfg.get("thermal_bilateral_sigma_space", 5.0)),
    )


def apply_preset(proc: ThermalProcessor, cfg: dict, opt_cfg_base: dict, preset_name: str) -> None:
    merged = merge_opt_cfg_with_preset(opt_cfg_base, preset_name)
    p = ENV_PRESETS.get(preset_name) or ENV_PRESETS["default"]
    proc.apply_env_runtime(
        merged,
        p.get("thermal_extra"),
        p.get("thermal_3dnr_alpha"),
    )
    proc.update_runtime_params(merged)


def run_sequence(proc: ThermalProcessor, frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    denoised = enhanced = heat = None
    for raw in frames:
        denoised, enhanced, heat, _fg = proc.process(raw, compute_enhanced=True)
    assert denoised is not None and enhanced is not None
    heat_u8 = heat if heat is not None else np.zeros_like(denoised)
    return enhanced, heat_u8


def run_sequence_full(
    proc: ThermalProcessor,
    frames: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return last-frame pipeline outputs: raw, denoised, enhanced, heat, fg_mask."""
    denoised = enhanced = heat = fg = None
    raw_last = frames[-1]
    for raw in frames:
        raw_last = raw
        denoised, enhanced, heat, fg = proc.process(raw, compute_enhanced=True)
    assert denoised is not None and enhanced is not None
    heat_u8 = heat if heat is not None else np.zeros_like(denoised)
    fg_u8 = fg if fg is not None else np.zeros_like(denoised)
    return raw_last, denoised, enhanced, heat_u8, fg_u8


def normalize_bg_display(bg: np.ndarray) -> np.ndarray:
    """Stretch background float estimate to uint8 for visualization."""
    a = bg.astype(np.float32)
    lo, hi = float(np.percentile(a, 2.0)), float(np.percentile(a, 98.0))
    if hi <= lo + 1e-3:
        return np.zeros_like(a, dtype=np.uint8)
    return np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def mad_z_image_global(denoised: np.ndarray) -> np.ndarray:
    """Per-pixel modified z using global median/MAD over all pixels (illustration)."""
    fp = denoised.astype(np.float32).ravel()
    med = float(np.median(fp))
    mad = float(np.median(np.abs(fp - med)))
    z = 0.6745 * np.abs(denoised.astype(np.float32) - med) / (mad + 1e-6)
    return z


def overlay_fg_contours(
    base_gray: np.ndarray,
    fg_mask: np.ndarray,
    *,
    color: tuple[int, int, int] = (80, 220, 80),
    line_thickness: int = 2,
) -> np.ndarray:
    """BGR overlay of foreground contours on grayscale."""
    bgr = cv.cvtColor(base_gray, cv.COLOR_GRAY2BGR)
    if fg_mask is None or not np.any(fg_mask):
        return bgr
    m = (fg_mask > 0).astype(np.uint8) * 255
    contours, _ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(bgr, contours, -1, color, max(1, int(line_thickness)))
    return bgr


def generate_algorithm_panels(
    seq: list[np.ndarray],
    *,
    cmap_lv: int,
    scale: int,
    dpi: int,
    interp: int,
) -> None:
    """Write thermal_alg_*.png thesis panels (3DNR, background/Kalman vs EMA, MAD)."""
    import matplotlib.pyplot as plt
    from matplotlib import cm

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "figure.titlesize": 13,
        }
    )

    def up(img: np.ndarray) -> np.ndarray:
        return upscale(img, scale, interpolation=interp)

    base = thesis_base_cfg()
    cfg_q = apply_profile(copy.deepcopy(base), "quality")
    opt_q = make_opt_cfg_base(cfg_q)

    # --- 3DNR: raw vs denoised, fog vs night alpha --------------------------------
    proc_def = build_thermal_proc(cfg_q, opt_q)
    apply_preset(proc_def, cfg_q, opt_q, "default")
    raw_last, den_def, _enh_def, _heat_def, _fg_def = run_sequence_full(proc_def, list(seq))

    proc_fog = build_thermal_proc(cfg_q, opt_q)
    apply_preset(proc_fog, cfg_q, opt_q, "fog")
    _, den_fog, _, _, _ = run_sequence_full(proc_fog, list(seq))

    proc_night = build_thermal_proc(cfg_q, opt_q)
    apply_preset(proc_night, cfg_q, opt_q, "night")
    _, den_night, _, _, _ = run_sequence_full(proc_night, list(seq))

    diff_dn = cv.absdiff(raw_last, den_def)
    diff_f = diff_dn.astype(np.float32)
    p98 = float(np.percentile(diff_f, 98.0))
    if p98 < 1e-3:
        p98 = float(np.max(diff_f)) + 1e-3
    diff_vis = np.clip(diff_f / p98 * 255.0, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.5))
    panels_r0 = (
        ("Raw $I_t$ (last frame)", raw_last, "gray"),
        ("After 3DNR $T_t$ (preset default, $\\alpha\\approx0.65$)", den_def, "gray"),
        ("$|I_t - T_t|$ (98th-pct stretch)", diff_vis, "magma"),
    )
    panels_r1 = (
        ("3DNR after preset fog ($\\alpha \\approx 0.58$)", den_fog, "gray"),
        ("3DNR after preset night ($\\alpha \\approx 0.72$)", den_night, "gray"),
        ("", None, ""),
    )
    for col, (title, img, mode) in enumerate(panels_r0):
        ax = axes[0, col]
        if img is not None:
            sh = up(img)
            if mode == "gray":
                ax.imshow(sh, cmap="gray", vmin=0, vmax=255, aspect="equal")
            else:
                ax.imshow(sh, cmap=mode, vmin=0, vmax=255, aspect="equal")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    for col, (title, img, mode) in enumerate(panels_r1):
        ax = axes[1, col]
        if col < 2 and img is not None:
            sh = up(img)
            ax.imshow(sh, cmap="gray", vmin=0, vmax=255, aspect="equal")
        elif col == 2:
            ax.text(
                0.5,
                0.55,
                "Preset merges adjust\nthermal_3dnr_alpha\n(fog lower, night higher).",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.suptitle(
        r"Temporal denoising (3D-NR): raw vs smoothed frame and preset $\alpha$ sensitivity "
        "(offline MI48 replay; quality profile)",
        fontsize=13,
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_ALG_3DNR, dpi=dpi, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"✅ Saved: {OUT_ALG_3DNR}")

    # --- Background: EMA cold-frame vs Kalman estimate ----------------------------
    proc_ema = build_thermal_proc(cfg_q, opt_q, use_kalman_background=False)
    apply_preset(proc_ema, cfg_q, opt_q, "default")
    _r_ema, den_ema, _e_ema, heat_ema, fg_ema = run_sequence_full(proc_ema, list(seq))

    proc_kal = build_thermal_proc(cfg_q, opt_q, use_kalman_background=True)
    apply_preset(proc_kal, cfg_q, opt_q, "default")
    _r_kal, den_kal, _e_kal, heat_kal, fg_kal = run_sequence_full(proc_kal, list(seq))

    cf_ema = getattr(proc_ema.bg, "cold_frame", None)
    cf_kal = getattr(proc_kal.bg, "cold_frame", None)
    assert cf_ema is not None and cf_kal is not None
    bg_ema_u8 = normalize_bg_display(cf_ema)
    bg_kal_u8 = normalize_bg_display(cf_kal)

    overlay = overlay_fg_contours(den_ema, fg_ema, line_thickness=max(2, scale // 5))

    fig2, axes2 = plt.subplots(2, 3, figsize=(15.0, 9.8))
    den_show = up(den_ema)
    axes2[0, 0].imshow(den_show, cmap="gray", vmin=0, vmax=255, aspect="equal")
    axes2[0, 0].set_title("Denoised input to BG model", fontsize=11)
    axes2[0, 1].imshow(up(bg_ema_u8), cmap="gray", vmin=0, vmax=255, aspect="equal")
    axes2[0, 1].set_title("EMA cold-frame estimate\n(default production path)", fontsize=11)
    axes2[0, 2].imshow(up(bg_kal_u8), cmap="gray", vmin=0, vmax=255, aspect="equal")
    axes2[0, 2].set_title("Kalman BG estimate\n(use_kalman_background=True)", fontsize=11)

    h0 = _panel_to_rgb(heat_ema, scale=scale, cmap_lv=cmap_lv, interpolation=interp)
    h1 = _panel_to_rgb(heat_kal, scale=scale, cmap_lv=cmap_lv, interpolation=interp)
    axes2[1, 0].imshow(h0, aspect="equal")
    axes2[1, 0].set_title("Heat map (EMA background)", fontsize=11)
    axes2[1, 1].imshow(h1, aspect="equal")
    axes2[1, 1].set_title("Heat map (Kalman background)", fontsize=11)
    ov_rgb = cv.cvtColor(up(overlay), cv.COLOR_BGR2RGB)
    axes2[1, 2].imshow(ov_rgb, aspect="equal")
    axes2[1, 2].set_title("FG mask contours (EMA path)", fontsize=11)
    for ax in axes2.ravel():
        ax.axis("off")
    fig2.suptitle(
        "Background model: EMA cold-frame (deployed default) vs optional per-pixel Kalman; "
        "heat maps and foreground mask (offline replay)",
        fontsize=13,
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.savefig(OUT_ALG_BG, dpi=dpi, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig2)
    print(f"✅ Saved: {OUT_ALG_BG}")

    # --- MAD detector + z-map (fresh processor so temporal state matches sequence) ---
    proc_mad = build_thermal_proc(cfg_q, opt_q)
    apply_preset(proc_mad, cfg_q, opt_q, "default")
    mad_det = ThermalMADAnomalyDetector()
    last_score = 0.0
    last_active = False
    last_blobs: list = []
    den_m = heat_m = fg_m = None
    for raw in seq:
        den_m, _enh_m, heat_m, fg_m = proc_mad.process(raw, compute_enhanced=True)
        last_score, last_active, last_blobs = mad_det.process(den_m, fg_m, heat_map=heat_m)

    assert den_m is not None and heat_m is not None and fg_m is not None
    z_img = mad_z_image_global(den_m)
    z_hi = float(np.percentile(z_img, 99.5))
    if z_hi < 1e-6:
        z_hi = float(np.max(z_img)) + 1e-6
    z_show = np.clip(z_img / z_hi, 0, 1.0)
    z_rgb_u8 = (cm.plasma(z_show)[..., :3] * 255.0).astype(np.uint8)

    mad_vis_small = cv.cvtColor(den_m, cv.COLOR_GRAY2BGR)
    rad = max(2, 3)  # native resolution — radii scale up with linear upscale
    for b in last_blobs[:5]:
        cv.circle(
            mad_vis_small,
            (int(b["cx"]), int(b["cy"])),
            rad,
            (0, 200, 255),
            1,
        )
    mad_vis = upscale(mad_vis_small, scale, interpolation=interp)
    mad_rgb = cv.cvtColor(mad_vis, cv.COLOR_BGR2RGB)

    fig3, axes3 = plt.subplots(1, 4, figsize=(17.0, 4.8))
    axes3[0].imshow(_panel_to_rgb(heat_m, scale=scale, cmap_lv=cmap_lv, interpolation=interp), aspect="equal")
    axes3[0].set_title("Heat map (input to scoring)", fontsize=11)
    fg_bin = (fg_m > 0).astype(np.float32)
    axes3[1].imshow(up((fg_bin * 255).astype(np.uint8)), cmap="gray", vmin=0, vmax=255, aspect="equal")
    axes3[1].set_title("Foreground mask", fontsize=11)
    axes3[2].imshow(up(z_rgb_u8), aspect="equal")
    axes3[2].set_title("Global robust $|z|$ map\n(illustration)", fontsize=11)
    axes3[3].imshow(mad_rgb, aspect="equal")
    axes3[3].set_title("MAD-gated blobs (last frame)", fontsize=11)
    # Avoid cv.putText on 62×80 then upscale — long strings clip. Draw in axes space instead.
    status_lbl = "ACTIVE" if last_active else "inactive"
    axes3[3].text(
        0.02,
        0.98,
        f"MAD score={last_score:.3f}\n({status_lbl})",
        transform=axes3[3].transAxes,
        fontsize=10,
        va="top",
        ha="left",
        color="white",
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "black", "alpha": 0.55},
    )
    for ax in axes3:
        ax.axis("off")
    fig3.suptitle(
        "MAD anomaly path: heat / FG / robust-$z$ visualization and blob centres "
        f"(temporal gate $\\geq 3$ frames; fraction-over-threshold={last_score:.3f}). "
        "Score may be 0 on quiescent offline replay.",
        fontsize=12,
        y=1.06,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig3.savefig(OUT_ALG_MAD, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig3)
    print(f"✅ Saved: {OUT_ALG_MAD}")


def thesis_base_cfg() -> dict:
    """CONFIG without embedded RPi throughput merge — profiles applied explicitly."""
    c = dict(CONFIG)
    c["rpi_throughput_max"] = False
    return c


def collect_png_paths(inp: Path, glob_pat: str) -> list[Path]:
    paths = sorted(inp.glob(glob_pat))
    if len(paths) < 2:
        paths = sorted(p for p in inp.glob("*.png") if p.is_file())
    if len(paths) < 2:
        raise FileNotFoundError(f"need at least 2 PNGs under {inp}")

    # Fix sorting for non-zero-padded filenames like image_1.png, image_10.png
    try:
        if all(p.stem.startswith("image_") for p in paths):
            paths = sorted(paths, key=lambda p: int(p.stem.split("_")[1]))
    except Exception:
        pass

    return paths


def upscale(img: np.ndarray, scale: int, *, interpolation: int = cv.INTER_LINEAR) -> np.ndarray:
    h, w = img.shape[:2]
    return cv.resize(img, (w * scale, h * scale), interpolation=interpolation)


def _panel_to_rgb(
    panel: np.ndarray,
    *,
    scale: int,
    cmap_lv: int,
    interpolation: int = cv.INTER_LINEAR,
) -> np.ndarray:
    if panel.ndim == 2:
        disp_bgr = gray_to_thermal_bgr(panel, levels=cmap_lv)
        rgb = cv.cvtColor(disp_bgr, cv.COLOR_BGR2RGB)
    else:
        rgb = panel
    return upscale(rgb, scale, interpolation=interpolation)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--glob", default="*.png", help="Glob under input-dir")
    ap.add_argument("--sequence-length", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", type=int, default=-1, help="Fixed start index; default: RNG")
    ap.add_argument("--scale", type=int, default=10, help="Upscale factor (linear interp by default)")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument(
        "--nearest-upscale",
        action="store_true",
        help="Use nearest-neighbour upscale (blocky pixels) instead of linear smoothing.",
    )
    ap.add_argument(
        "--no-auto-start",
        action="store_true",
        help="Disable automatic clip selection; use --start or random --seed.",
    )
    ap.add_argument(
        "--auto-start-candidates",
        type=int,
        default=64,
        help="How many candidate starts to score when auto-selecting a clip (default: 64).",
    )
    ap.add_argument(
        "--skip-algorithm-panels",
        action="store_true",
        help="Only emit profile/preset grids (skip thermal_alg_*.png).",
    )
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

    paths = collect_png_paths(inp, args.glob)
    n = len(paths)
    seq_len = args.sequence_length
    interp = cv.INTER_NEAREST if args.nearest_upscale else cv.INTER_LINEAR

    if args.start >= 0:
        start = int(args.start)
        score = _motion_and_detail_score(paths, start, seq_len)
        print(f"[clip] fixed start={start} score={score:.2f}")
    elif args.no_auto_start:
        rng = np.random.default_rng(args.seed)
        start = int(rng.integers(0, n))
        score = _motion_and_detail_score(paths, start, seq_len)
        print(f"[clip] random start={start} (seed={args.seed}) score={score:.2f}")
    else:
        start, score = pick_best_sequence_start(paths, seq_len, n_candidates=args.auto_start_candidates)
        print(f"[clip] auto-picked start={start} score={score:.2f} (candidates={args.auto_start_candidates})")

    seq = iter_gray_sequence(
        paths,
        start,
        seq_len,
        inject_noise_flag=args.inject_noise,
        noise_sigma=args.noise_sigma,
        fpn_sigma=args.fpn_sigma,
        salt_pepper_prob=args.salt_pepper_prob,
        base_seed=args.seed,
    )

    cmap_lv = int(CONFIG.get("thermal_colormap_levels", 256))
    if cmap_lv not in (32, 64, 128, 256):
        cmap_lv = 256

    base = thesis_base_cfg()
    profile_names = ("raw", "throughput", "quality")
    preset_names = ("default", "night", "fog", "glare_heavy")

    profile_panels: list[tuple[np.ndarray, np.ndarray]] = []
    for pname in profile_names:
        cfg = apply_profile(copy.deepcopy(base), pname)
        opt_base = make_opt_cfg_base(cfg)
        proc = build_thermal_proc(cfg, opt_base)
        apply_preset(proc, cfg, opt_base, "default")
        enhanced, heat_u8 = run_sequence(proc, seq)
        profile_panels.append((enhanced, heat_u8))

    preset_panels: list[tuple[np.ndarray, np.ndarray]] = []
    cfg_q = apply_profile(copy.deepcopy(base), "quality")
    opt_q = make_opt_cfg_base(cfg_q)
    for preset in preset_names:
        proc = build_thermal_proc(cfg_q, opt_q)
        apply_preset(proc, cfg_q, opt_q, preset)
        enhanced, heat_u8 = run_sequence(proc, list(seq))  # fresh processor; same frames
        preset_panels.append((enhanced, heat_u8))

    DOC_DIR.mkdir(parents=True, exist_ok=True)

    def save_grid(
        panels: list[tuple[np.ndarray, np.ndarray]],
        col_labels: tuple[str, ...],
        path: Path,
        suptitle: str,
    ) -> None:
        ncols = len(col_labels)
        fig, axes = plt.subplots(2, ncols, figsize=(4.2 * ncols, 8.2))
        plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})
        for col in range(ncols):
            enhanced, heat_u8 = panels[col]
            top = _panel_to_rgb(enhanced, scale=args.scale, cmap_lv=cmap_lv, interpolation=interp)
            bot = _panel_to_rgb(heat_u8, scale=args.scale, cmap_lv=cmap_lv, interpolation=interp)
            axes[0, col].imshow(top, aspect="equal")
            axes[1, col].imshow(bot, aspect="equal")
            axes[0, col].set_title(col_labels[col], fontsize=11)
            axes[0, col].axis("off")
            axes[1, col].axis("off")
        axes[0, 0].set_ylabel("Display path\n(TURBO LUT)", fontsize=11)
        axes[1, 0].set_ylabel("Heat map\n(TURBO LUT)", fontsize=11)
        fig.suptitle(suptitle, fontsize=13, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
        print(f"✅ Saved: {path}")

    save_grid(
        profile_panels,
        ("raw profile", "throughput profile", "quality profile"),
        OUT_PROFILES,
        "Thermal stack vs display profile (ENV preset = default; offline MI48 replay)",
    )
    save_grid(
        preset_panels,
        ("preset: default", "preset: night", "preset: fog", "preset: glare_heavy"),
        OUT_PRESETS,
        "Thermal stack vs ENV preset (display profile = quality; offline MI48 replay)",
    )

    THESIS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT_PROFILES, THESIS_DIR / OUT_PROFILES.name)
    shutil.copy2(OUT_PRESETS, THESIS_DIR / OUT_PRESETS.name)
    if not args.skip_algorithm_panels:
        generate_algorithm_panels(
            seq,
            cmap_lv=cmap_lv,
            scale=args.scale,
            dpi=args.dpi,
            interp=interp,
        )
        shutil.copy2(OUT_ALG_3DNR, THESIS_DIR / OUT_ALG_3DNR.name)
        shutil.copy2(OUT_ALG_BG, THESIS_DIR / OUT_ALG_BG.name)
        shutil.copy2(OUT_ALG_MAD, THESIS_DIR / OUT_ALG_MAD.name)
    print(f"✅ Copied to: {THESIS_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
