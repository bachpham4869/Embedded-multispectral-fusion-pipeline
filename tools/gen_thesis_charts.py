"""gen_thesis_charts.py — Generate DATA_PENDING figures for thesis.

Produces:
1. Cross-bucket IQA heatmap (ENV class × Bucket, cell = mean metric)
2. IQA bar chart per bucket/ENV class
3. Before/after CLAHE sample pairs (Ch2)
4. Before/after DCP sample pairs (Ch2)
5. Before/after Bucket A (HybridNIR night enhancement) (Ch2)
6. Before/after Bucket C (anti-glare tone mapping) (Ch2)

Usage:
    python tools/gen_thesis_charts.py --all
    python tools/gen_thesis_charts.py --heatmap
    python tools/gen_thesis_charts.py --barchart
    python tools/gen_thesis_charts.py --clahe-sample
    python tools/gen_thesis_charts.py --dcp-sample
    python tools/gen_thesis_charts.py --night-sample
    python tools/gen_thesis_charts.py --glare-sample
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib required. pip install matplotlib", file=sys.stderr)
    sys.exit(1)

# Add project src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

PROJ = Path(__file__).parent.parent
IQA_CSV = PROJ / "docs/tables/iqa/batch_nir_iqa.csv"
MANIFEST = PROJ / "data/eval/nir_val/manifest_v2.csv"
OUT_DIR = PROJ / "Thesis_report/figures"


def load_iqa_csv() -> list[dict]:
    with open(IQA_CSV, newline="") as f:
        return list(csv.DictReader(f))


# ── 1. Cross-bucket IQA heatmap ──────────────────────────────────────────────
def gen_heatmap():
    """ENV class (rows) × Bucket (cols), cell = mean after_log_rms_contrast."""
    rows = load_iqa_csv()
    env_classes = sorted(set(r["env_class"] for r in rows))
    buckets = sorted(set(r["bucket"] for r in rows))

    matrix = np.full((len(env_classes), len(buckets)), np.nan)
    for r in rows:
        ei = env_classes.index(r["env_class"])
        bi = buckets.index(r["bucket"])
        val = float(r["after_log_rms_contrast"])
        if np.isnan(matrix[ei, bi]):
            matrix[ei, bi] = val
        else:
            # Running average
            count_key = f"{r['env_class']}_{r['bucket']}"
            matrix[ei, bi] = (matrix[ei, bi] + val) / 2

    # Recompute properly with means
    from collections import defaultdict
    sums = defaultdict(lambda: [0.0, 0])
    for r in rows:
        key = (r["env_class"], r["bucket"])
        sums[key][0] += float(r["after_log_rms_contrast"])
        sums[key][1] += 1
    matrix = np.full((len(env_classes), len(buckets)), np.nan)
    for (env, bkt), (s, c) in sums.items():
        matrix[env_classes.index(env), buckets.index(bkt)] = s / c

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels([f"Bucket {b}" for b in buckets], fontsize=11)
    ax.set_yticks(range(len(env_classes)))
    ax.set_yticklabels(env_classes, fontsize=10)
    ax.set_xlabel("Optical Processing Bucket", fontsize=12)
    ax.set_ylabel("Environment Class", fontsize=12)
    ax.set_title("Cross-Bucket IQA: Mean Log-RMS Contrast\n(higher = better contrast)", fontsize=13)

    # Annotate cells
    for i in range(len(env_classes)):
        for j in range(len(buckets)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=9, color="white" if v > np.nanmedian(matrix) else "black")

    plt.colorbar(im, ax=ax, label="log_rms_contrast (after)")
    plt.tight_layout()
    out = OUT_DIR / "ch6_evaluation/bucket_dispatch/cross_bucket_iqa_heatmap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"✅ Heatmap saved: {out}")


# ── 2. IQA bar chart per bucket ──────────────────────────────────────────────
def gen_barchart():
    """Grouped bar chart: IQA metrics per bucket, aggregated across ENV classes."""
    rows = load_iqa_csv()
    from collections import defaultdict
    
    metrics = ["after_log_rms_contrast", "after_hist_entropy", "after_pct_saturated", "after_pct_crushed"]
    labels = ["Log-RMS Contrast", "Histogram Entropy", "% Saturated", "% Crushed"]
    
    buckets = sorted(set(r["bucket"] for r in rows))
    bucket_data = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for m in metrics:
            bucket_data[r["bucket"]][m].append(float(r[m]))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.Set2(np.linspace(0, 1, len(buckets)))

    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[idx // 2, idx % 2]
        means = [np.mean(bucket_data[b][metric]) for b in buckets]
        stds = [np.std(bucket_data[b][metric]) for b in buckets]
        bars = ax.bar([f"Bucket {b}" for b in buckets], means, yerr=stds,
                      color=colors, capsize=4, edgecolor="gray", linewidth=0.5)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", labelsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("IQA Proxy Metrics per Optical Bucket (all ENV classes)", fontsize=14, y=1.01)
    plt.tight_layout()
    out = OUT_DIR / "ch6_evaluation/bucket_dispatch/iqa_barchart_per_bucket.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Bar chart saved: {out}")


# ── 3. CLAHE before/after sample ─────────────────────────────────────────────
def gen_clahe_sample():
    """Generate a before/after CLAHE comparison grid for Ch2."""
    try:
        import cv2 as cv
        from smartbinocular.nir_pipeline import nir_nir_night_clahe
    except ImportError:
        print("⚠️ OpenCV or smartbinocular not available — skipping CLAHE sample", file=sys.stderr)
        return

    manifest_rows = _load_manifest_rows()
    if not manifest_rows:
        print("⚠️ No manifest rows — skipping CLAHE sample", file=sys.stderr)
        return

    # Pick 3 dark images
    dark_rows = [r for r in manifest_rows if r["env_class"] in ("night_clear", "normal_night", "nir_night")]
    if len(dark_rows) < 3:
        dark_rows = manifest_rows[:3]
    else:
        dark_rows = dark_rows[:3]

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.subplots_adjust(hspace=0.15, wspace=0.05)
    for i, row in enumerate(dark_rows):
        bgr = cv.imread(row["path"], cv.IMREAD_COLOR)
        if bgr is None:
            continue
        enhanced = nir_nir_night_clahe(bgr)
        # Ensure same display size
        h, w = bgr.shape[:2]
        axes[i, 0].imshow(cv.cvtColor(bgr, cv.COLOR_BGR2RGB), aspect="equal")
        axes[i, 0].set_title(f"Before ({row['env_class']})", fontsize=11)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(cv.cvtColor(enhanced, cv.COLOR_BGR2RGB), aspect="equal")
        axes[i, 1].set_title("After CLAHE (Bucket B)", fontsize=11)
        axes[i, 1].axis("off")

    fig.suptitle("CLAHE Enhancement: Before vs After (NIR Night Images)", fontsize=14, y=0.98)
    out = OUT_DIR / "ch2_background/clahe_before_after.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"✅ CLAHE sample saved: {out}")


# ── 4. DCP before/after sample ───────────────────────────────────────────────
def gen_dcp_sample():
    """Generate a before/after DCP dehaze comparison grid for Ch2."""
    try:
        import cv2 as cv
        from smartbinocular.nir_pipeline import nir_dehaze_lite
    except ImportError:
        print("⚠️ OpenCV or smartbinocular not available — skipping DCP sample", file=sys.stderr)
        return

    manifest_rows = _load_manifest_rows()
    if not manifest_rows:
        print("⚠️ No manifest rows — skipping DCP sample", file=sys.stderr)
        return

    # Pick fog images
    fog_rows = [r for r in manifest_rows if r["env_class"] == "fog"]
    if len(fog_rows) < 3:
        fog_rows = manifest_rows[:3]
    else:
        fog_rows = fog_rows[:3]

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.subplots_adjust(hspace=0.15, wspace=0.05)
    for i, row in enumerate(fog_rows):
        bgr = cv.imread(row["path"], cv.IMREAD_COLOR)
        if bgr is None:
            continue
        dehazed = nir_dehaze_lite(bgr)
        axes[i, 0].imshow(cv.cvtColor(bgr, cv.COLOR_BGR2RGB), aspect="equal")
        axes[i, 0].set_title(f"Before ({row['env_class']})", fontsize=11)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(cv.cvtColor(dehazed, cv.COLOR_BGR2RGB), aspect="equal")
        axes[i, 1].set_title("After DCP Dehaze (Bucket D)", fontsize=11)
        axes[i, 1].axis("off")

    fig.suptitle("Dark Channel Prior Dehazing: Before vs After", fontsize=14, y=0.98)
    out = OUT_DIR / "ch2_background/dcp_before_after.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"✅ DCP sample saved: {out}")


def _load_manifest_rows() -> list[dict]:
    if not MANIFEST.exists():
        return []
    with open(MANIFEST, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            img_path = row.get("path") or row.get("image_path", "")
            env = row.get("env_class") or row.get("label", "unknown")
            if img_path and os.path.isfile(img_path):
                rows.append({"path": img_path, "env_class": env})
        return rows


# ── 5. Bucket A (HybridNIR night enhance) before/after ───────────────────────
def gen_night_sample():
    """Generate a before/after HybridNIREnhancer comparison grid for Ch2."""
    try:
        import cv2 as cv
        from smartbinocular.nir_pipeline import HybridNIREnhancer
    except ImportError:
        print("⚠️ OpenCV or smartbinocular not available — skipping night sample", file=sys.stderr)
        return

    manifest_rows = _load_manifest_rows()
    if not manifest_rows:
        print("⚠️ No manifest rows — skipping night sample", file=sys.stderr)
        return

    # Pick dark images for Bucket A
    dark_rows = [r for r in manifest_rows if r["env_class"] in ("night_clear", "normal_night")]
    if len(dark_rows) < 3:
        dark_rows = [r for r in manifest_rows if r["env_class"] in ("night_clear", "normal_night", "nir_night")][:3]
    else:
        dark_rows = dark_rows[:3]

    enhancer = HybridNIREnhancer(proc_w=320, proc_h=240, update_rate=1)

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.subplots_adjust(hspace=0.15, wspace=0.05)
    for i, row in enumerate(dark_rows):
        bgr = cv.imread(row["path"], cv.IMREAD_COLOR)
        if bgr is None:
            continue
        enhancer.reset()
        enhanced = enhancer.process(bgr)
        axes[i, 0].imshow(cv.cvtColor(bgr, cv.COLOR_BGR2RGB), aspect="equal")
        axes[i, 0].set_title(f"Before ({row['env_class']})", fontsize=11)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(cv.cvtColor(enhanced, cv.COLOR_BGR2RGB), aspect="equal")
        axes[i, 1].set_title("After HybridNIR (Bucket A)", fontsize=11)
        axes[i, 1].axis("off")

    fig.suptitle("Night Enhancement: HybridNIREnhancer (DCP + Adaptive CLAHE)", fontsize=14, y=0.98)
    out = OUT_DIR / "ch2_background/night_enhance_before_after.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"✅ Night enhance sample saved: {out}")


# ── 6. Bucket C (anti-glare tone-map) before/after ───────────────────────────
def gen_glare_sample():
    """Generate a before/after anti-glare comparison grid for Ch2."""
    try:
        import cv2 as cv
        from smartbinocular.nir_pipeline import nir_anti_glare_bgr
    except ImportError:
        print("⚠️ OpenCV or smartbinocular not available — skipping glare sample", file=sys.stderr)
        return

    manifest_rows = _load_manifest_rows()
    if not manifest_rows:
        print("⚠️ No manifest rows — skipping glare sample", file=sys.stderr)
        return

    # Pick glare/backlight images for Bucket C
    glare_rows = [r for r in manifest_rows if r["env_class"] in ("glare", "backlight")]
    if len(glare_rows) < 3:
        glare_rows = [r for r in manifest_rows if r["env_class"] in ("glare", "backlight", "normal_day")][:3]
    else:
        glare_rows = glare_rows[:3]

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.subplots_adjust(hspace=0.15, wspace=0.05)
    for i, row in enumerate(glare_rows):
        bgr = cv.imread(row["path"], cv.IMREAD_COLOR)
        if bgr is None:
            continue
        processed = nir_anti_glare_bgr(bgr)
        axes[i, 0].imshow(cv.cvtColor(bgr, cv.COLOR_BGR2RGB), aspect="equal")
        axes[i, 0].set_title(f"Before ({row['env_class']})", fontsize=11)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(cv.cvtColor(processed, cv.COLOR_BGR2RGB), aspect="equal")
        axes[i, 1].set_title("After Anti-Glare (Bucket C)", fontsize=11)
        axes[i, 1].axis("off")

    fig.suptitle("Anti-Glare Tone Mapping: Before vs After (Bucket C)", fontsize=14, y=0.98)
    out = OUT_DIR / "ch2_background/glare_before_after.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"✅ Glare sample saved: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Generate all charts")
    parser.add_argument("--heatmap", action="store_true")
    parser.add_argument("--barchart", action="store_true")
    parser.add_argument("--clahe-sample", action="store_true")
    parser.add_argument("--dcp-sample", action="store_true")
    parser.add_argument("--night-sample", action="store_true")
    parser.add_argument("--glare-sample", action="store_true")
    args = parser.parse_args()

    if not any([args.all, args.heatmap, args.barchart, args.clahe_sample,
                args.dcp_sample, args.night_sample, args.glare_sample]):
        args.all = True

    if args.all or args.heatmap:
        gen_heatmap()
    if args.all or args.barchart:
        gen_barchart()
    if args.all or args.clahe_sample:
        gen_clahe_sample()
    if args.all or args.dcp_sample:
        gen_dcp_sample()
    if args.all or args.night_sample:
        gen_night_sample()
    if args.all or args.glare_sample:
        gen_glare_sample()


if __name__ == "__main__":
    main()
