"""build_open_questions.py — Generate 6 open-question mini-study artifacts.

Studies:
  1. fusion_benefit_by_class: Group A/B IQA deltas by env_class → bar + CSV
  2. ml_stability: Estimate transition reduction from hysteresis simulation
  3. train_deploy_gap: Sidecar macro_f1_night vs NIR deployment proxy
  4. bucket_share: Bucket-F compositor rule coverage from session transitions
  5. dcp_resolution_tradeoff: DCP dehaze at 160×120 vs 320×240 on fog frames
  6. fps_consistency: FPS mean/std across sessions

Limitations are explicitly stated in each output.

Usage:
    python tools/build_open_questions.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from _iqa_metrics import compute_iqa_metrics  # type: ignore[import]
from smartbinocular.nir_pipeline import nir_dehaze_lite

OUT_DIR = ROOT / "docs/thesis_eval/open_questions"
AB_CSV = ROOT / "docs/thesis_eval/fusion/tables/ab_fusion_vs_nir_proxy.csv"
TRANSITIONS_CSV = ROOT / "docs/thesis_eval/timing_performance/tables/transitions_per_minute.csv"
STAGE_BY_MODE_CSV = ROOT / "docs/thesis_eval/timing_performance/tables/stage_timing_by_mode.csv"
SESSION_DIR = ROOT / "fusion_captures/metrics"
SIDECAR = ROOT / "models/production/env_classifier.json"
MANIFEST = ROOT / "data/eval/nir_val/manifest_v2.csv"


def _import_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("ERROR: matplotlib not installed.", file=sys.stderr)
        sys.exit(1)


# ── Study 1: Fusion benefit by env class ──────────────────────────────────────

def study_fusion_benefit_by_class(plt) -> None:
    if not AB_CSV.exists():
        print(f"  SKIP fusion_benefit_by_class: {AB_CSV} not found")
        return

    with AB_CSV.open(newline="") as fh:
        rows = list(csv.DictReader(fh))

    by_class: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_class[r["env_class"]].append(float(r["delta_log_rms"]))

    summary_rows = []
    for cls in sorted(by_class):
        vals = by_class[cls]
        summary_rows.append({
            "env_class": cls,
            "n_frames": len(vals),
            "mean_delta_log_rms": round(sum(vals) / len(vals), 5),
            "min_delta_log_rms": round(min(vals), 5),
            "max_delta_log_rms": round(max(vals), 5),
            "limitation": "PROXY — dummy thermal; negative delta expected; real benefit TBD",
        })

    csv_out = OUT_DIR / "tables/fusion_benefit_by_class.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"  Saved: {csv_out}")

    fig, ax = plt.subplots(figsize=(10, 5))
    classes = [r["env_class"] for r in summary_rows]
    means = [r["mean_delta_log_rms"] for r in summary_rows]
    colors = ["#d7191c" if m < 0 else "#2c7bb6" for m in means]
    ax.bar(classes, means, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("env_class")
    ax.set_ylabel("Mean Δlog_rms (fused − NIR)")
    ax.set_title(
        "Fusion benefit by environment class (PROXY — dummy thermal)\n"
        "Negative values: dummy gray thermal suppresses NIR contrast; all classes affected equally",
        fontsize=9,
    )
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig_out = OUT_DIR / "figures/fusion_benefit_by_class.png"
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fig_out}")


# ── Study 2: ML temporal stability (hysteresis simulation) ────────────────────

def study_ml_stability(plt) -> None:
    """Simulate transitions w/ and w/o hysteresis using test JSONL posteriors."""
    test_jsonl = ROOT / "data/training/from_logs_test.jsonl"
    if not test_jsonl.exists():
        print(f"  SKIP ml_stability: {test_jsonl} not found")
        return

    try:
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "src"))
        from smartbinocular.feature_schema import ENV_CLASSES
        ENV_CLASSES_SET = set(ENV_CLASSES)
    except Exception as e:
        print(f"  SKIP ml_stability: import error: {e}")
        return

    # Load test labels (sequence proxy)
    labels = []
    with test_jsonl.open() as fh:
        for line in fh:
            try:
                rec = json.loads(line.strip())
                labels.append(rec.get("label", "unknown"))
            except json.JSONDecodeError:
                pass

    if len(labels) < 50:
        print(f"  SKIP ml_stability: too few test records ({len(labels)})")
        return

    # Count raw transitions (frame-to-frame label change)
    raw_transitions = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])
    raw_per_min = raw_transitions / (len(labels) / 60)  # assume 1 label per second proxy

    # Simulate hysteresis: require 3 consecutive same-class frames before switching
    HYSTERESIS_N = 3
    stable = [labels[0]]
    pending = labels[0]
    count = 1
    for lbl in labels[1:]:
        if lbl == pending:
            count += 1
            if count >= HYSTERESIS_N:
                stable.append(pending)
            else:
                stable.append(stable[-1])
        else:
            pending = lbl
            count = 1
            stable.append(stable[-1])

    hyst_transitions = sum(1 for i in range(1, len(stable)) if stable[i] != stable[i - 1])
    hyst_per_min = hyst_transitions / (len(stable) / 60)
    reduction_pct = 100 * (1 - hyst_per_min / (raw_per_min + 1e-9))

    rows = [
        {
            "mode": "raw_argmax",
            "n_frames_simulated": len(labels),
            "transitions": raw_transitions,
            "transitions_per_min_proxy": round(raw_per_min, 2),
            "limitation": "ESTIMATED — from test JSONL label sequence; not live session data",
        },
        {
            "mode": f"hysteresis_n{HYSTERESIS_N}",
            "n_frames_simulated": len(stable),
            "transitions": hyst_transitions,
            "transitions_per_min_proxy": round(hyst_per_min, 2),
            "limitation": f"ESTIMATED — simulated {HYSTERESIS_N}-frame hysteresis on same sequence",
        },
    ]

    csv_out = OUT_DIR / "tables/ml_stability.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {csv_out}")
    print(f"    raw transitions/min={raw_per_min:.2f}  hyst={hyst_per_min:.2f}  reduction={reduction_pct:.1f}%")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Raw argmax", f"Hysteresis N={HYSTERESIS_N}"],
           [raw_per_min, hyst_per_min], color=["#d7191c", "#2c7bb6"], alpha=0.85)
    ax.set_ylabel("Estimated transitions / min (proxy)")
    ax.set_title(
        f"ML temporal stability: hysteresis N={HYSTERESIS_N}\n"
        "ESTIMATED — simulated on test JSONL label sequence; not live session replay",
        fontsize=9,
    )
    fig.tight_layout()
    fig_out = OUT_DIR / "figures/ml_stability.png"
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fig_out}")


# ── Study 3: Train/deploy gap proxy ───────────────────────────────────────────

def study_train_deploy_gap() -> None:
    if not SIDECAR.exists():
        print(f"  SKIP train_deploy_gap: {SIDECAR} not found")
        return

    sidecar = json.loads(SIDECAR.read_text())
    gate = sidecar.get("ml_gate_reference", {})
    tau_062 = gate.get("top1_tau_0.62", {})
    tau_065 = gate.get("top1_tau_0.65_alternative", {})

    rows = [
        {
            "split": "optical_test_tau_0.62",
            "macro_f1_night_ovr": tau_062.get("macro_f1_night_ovr", ""),
            "abstention_rate": tau_062.get("abstention_rate", ""),
            "host": "Mac",
            "note": "Optical test set (visible spectrum); not NIR deployment",
        },
        {
            "split": "optical_test_tau_0.65",
            "macro_f1_night_ovr": tau_065.get("macro_f1_night_ovr", ""),
            "abstention_rate": tau_065.get("abstention_rate", ""),
            "host": "Mac",
            "note": "Optical test set; alternative threshold",
        },
        {
            "split": "nir_deployment_live",
            "macro_f1_night_ovr": "RPi4_PENDING",
            "abstention_rate": "RPi4_PENDING",
            "host": "RPi4_PENDING",
            "note": "Live NIR sessions needed; domain shift from visible training UNVERIFIED",
        },
    ]

    csv_out = OUT_DIR / "tables/train_deploy_gap.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {csv_out}")
    print(f"    optical macro_f1_night@0.62 = {tau_062.get('macro_f1_night_ovr')}  RPi4 = PENDING")


# ── Study 4: Compositor bucket-F rule coverage ────────────────────────────────

def study_bucket_share(plt) -> None:
    if not TRANSITIONS_CSV.exists():
        print(f"  SKIP bucket_share: {TRANSITIONS_CSV} not found")
        return

    # Load session JSON to count bucket shares
    session_files = sorted(SESSION_DIR.glob("session_*.json")) if SESSION_DIR.exists() else []
    if not session_files:
        print(f"  SKIP bucket_share: no session JSONs in {SESSION_DIR}")
        return

    bucket_totals: dict[str, int] = defaultdict(int)
    for sf in session_files:
        try:
            data = json.loads(sf.read_text())
            for bucket, count in data.get("frames_by_bucket", {}).items():
                bucket_totals[bucket] += count
        except Exception:
            pass

    total = sum(bucket_totals.values())
    if total == 0:
        print("  SKIP bucket_share: no frames_by_bucket data in sessions")
        return

    rows = []
    for b in sorted(bucket_totals):
        pct = 100 * bucket_totals[b] / total
        rows.append({
            "bucket": b,
            "total_frames": bucket_totals[b],
            "share_pct": round(pct, 2),
            "note": "Aggregated across all available RPi4 sessions",
        })

    csv_out = OUT_DIR / "tables/bucket_share.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {csv_out}")

    fig, ax = plt.subplots(figsize=(7, 4))
    buckets = [r["bucket"] for r in rows]
    shares = [r["share_pct"] for r in rows]
    colors = ["#d7191c" if b == "F" else "#2c7bb6" for b in buckets]
    ax.bar(buckets, shares, color=colors, alpha=0.85, edgecolor="white")
    ax.set_xlabel("Bucket")
    ax.set_ylabel("Share of frames (%)")
    ax.set_title(
        "Compositor bucket share across RPi4 sessions\n"
        "Bucket F (red) = transition blend; hypothesis: <5% of frames",
        fontsize=9,
    )
    fig.tight_layout()
    fig_out = OUT_DIR / "figures/bucket_share.png"
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fig_out}")


# ── Study 5: DCP resolution tradeoff ──────────────────────────────────────────

def study_dcp_resolution_tradeoff(plt) -> None:
    if not MANIFEST.exists():
        print(f"  SKIP dcp_resolution: {MANIFEST} not found")
        return

    fog_frames = []
    with MANIFEST.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("env_class") == "fog" and Path(row["path"]).exists():
                fog_frames.append(row["path"])

    if not fog_frames:
        print("  SKIP dcp_resolution: no fog frames found")
        return

    results = []
    for img_path in fog_frames:
        bgr = cv.imread(img_path, cv.IMREAD_COLOR)
        if bgr is None:
            continue

        for label, size in [("320x240", (320, 240)), ("160x120", (160, 120))]:
            frame = cv.resize(bgr, size)
            t0 = time.perf_counter()
            out = nir_dehaze_lite(frame)
            proc_ms = (time.perf_counter() - t0) * 1000
            gray = cv.cvtColor(out, cv.COLOR_BGR2GRAY)
            iqa = compute_iqa_metrics(gray)
            results.append({
                "frame": Path(img_path).name,
                "resolution": label,
                "proc_ms": round(proc_ms, 2),
                "log_rms": iqa["log_rms_contrast"],
                "pct_sat": iqa["pct_saturated"],
                "note": "IQA proxy — not validated against subjective quality",
            })

    if not results:
        print("  SKIP dcp_resolution: no results produced")
        return

    # Summary per resolution
    by_res: dict[str, list] = defaultdict(list)
    for r in results:
        by_res[r["resolution"]].append(r)

    csv_out = OUT_DIR / "tables/dcp_resolution_tradeoff.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"  Saved: {csv_out}  ({len(results)} rows)")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, label in [
        (axes[0], "log_rms", "Mean log_rms_contrast"),
        (axes[1], "proc_ms", "Mean proc_ms"),
    ]:
        res_labels = sorted(by_res.keys())
        means = [sum(r[metric] for r in by_res[res]) / len(by_res[res]) for res in res_labels]
        ax.bar(res_labels, means, color=["#2c7bb6", "#fdae61"], alpha=0.85)
        ax.set_ylabel(label)
        ax.set_title(f"DCP dehaze: {label} by resolution\n(fog frames)")
    fig.suptitle(
        "DCP resolution tradeoff — fog frames\n"
        "IQA proxy; not validated against subjective quality",
        fontsize=9,
    )
    fig.tight_layout()
    fig_out = OUT_DIR / "figures/dcp_resolution_tradeoff.png"
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fig_out}")


# ── Study 6: FPS consistency ───────────────────────────────────────────────────

def study_fps_consistency(plt) -> None:
    session_files = sorted(SESSION_DIR.glob("session_*.json")) if SESSION_DIR.exists() else []
    if not session_files:
        print(f"  SKIP fps_consistency: no session JSONs in {SESSION_DIR}")
        return

    fps_vals = []
    rows = []
    for sf in session_files:
        try:
            data = json.loads(sf.read_text())
            fps = data.get("fps_mean", None)
            if fps is not None:
                fps_vals.append(fps)
                rows.append({"session": sf.stem, "fps_mean": fps, "host": "RPi4"})
        except Exception:
            pass

    if not fps_vals:
        print("  SKIP fps_consistency: no fps_mean in session JSONs")
        return

    mean_fps = sum(fps_vals) / len(fps_vals)
    std_fps = math.sqrt(sum((f - mean_fps) ** 2 for f in fps_vals) / len(fps_vals))
    cv_pct = 100 * std_fps / (mean_fps + 1e-9)

    summary = {
        "n_sessions": len(fps_vals),
        "fps_mean": round(mean_fps, 2),
        "fps_std": round(std_fps, 2),
        "fps_cv_pct": round(cv_pct, 1),
        "hypothesis": "fps_std/fps_mean < 10%",
        "result": "PASS" if cv_pct < 10 else "FAIL",
        "note": "Short sessions 24-244s; RPi4; ≥5-min sessions pending",
    }
    print(f"    FPS: mean={mean_fps:.1f}  std={std_fps:.1f}  CV={cv_pct:.1f}%  {summary['result']}")

    csv_out = OUT_DIR / "tables/fps_consistency.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {csv_out}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(fps_vals)), fps_vals, color="#2c7bb6", alpha=0.85)
    ax.axhline(mean_fps, color="black", ls="--", lw=1.2, label=f"mean={mean_fps:.1f}")
    ax.axhline(mean_fps + std_fps, color="grey", ls=":", lw=1.0)
    ax.axhline(mean_fps - std_fps, color="grey", ls=":", lw=1.0, label=f"±1σ (std={std_fps:.1f})")
    ax.set_xlabel("Session index")
    ax.set_ylabel("fps_mean")
    ax.set_title(
        f"FPS consistency across {len(fps_vals)} RPi4 sessions  (CV={cv_pct:.1f}% — {summary['result']})\n"
        "Note: sessions are short (24-244 s); ≥5-min sessions pending",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_out = OUT_DIR / "figures/fps_consistency.png"
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fig_out}")


def main() -> None:
    plt = _import_mpl()
    (OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    print("Study 1: fusion benefit by class")
    study_fusion_benefit_by_class(plt)

    print("Study 2: ML temporal stability")
    study_ml_stability(plt)

    print("Study 3: train/deploy gap proxy")
    study_train_deploy_gap()

    print("Study 4: bucket-F rule coverage")
    study_bucket_share(plt)

    print("Study 5: DCP resolution tradeoff")
    study_dcp_resolution_tradeoff(plt)

    print("Study 6: FPS consistency")
    study_fps_consistency(plt)

    print("Done — open_questions artifacts saved to", OUT_DIR)


if __name__ == "__main__":
    main()
